module NeuralMutationsModule

using Random: default_rng, AbstractRNG
import ONNXRunTime as ORT
using DynamicExpressions: AbstractExpressionNode, AbstractExpression, NodeSampler, has_operators, with_contents, get_contents, eval_tree_array, string_tree
using ..CoreModule: AbstractOptions, DATA_TYPE
using ..ParsingModule: nn_config, node_to_onehot, logits_to_prods, prods_to_tree, select_viable_subtree, count_nodes, OPERATOR_ARITY

# Add CUDA imports at module level
const CUDA_AVAILABLE = Ref{Bool}(false)
try
    using CUDA
    using cuDNN
    global CUDA_AVAILABLE[] = true
catch
    global CUDA_AVAILABLE[] = false
end

export neural_mutate_tree

const MODEL_REF = Ref{Union{Nothing, ORT.InferenceSession}}(nothing)
const CFG_REF = Ref{Any}(nothing)
const OP_INDEX_REF = Ref{Dict{String,Int}}(Dict{String,Int}())
const OP_TO_LOGITS_REF = Ref{Dict{Tuple{Int,Int}, Int}}(Dict{Tuple{Int,Int}, Int}())
const OPTIONS_REF = Ref{Union{Nothing, AbstractOptions}}(nothing)
const ENABLED_REF = Ref{Bool}(false)
const STATS_LOCK = ReentrantLock()

zero_sqrt(x) = x >= 0 ? sqrt(x) : zero(x)

# FIXME: Move to external package
mutable struct NeuralMutationStats
    total_attempts::Int
    successful_mutations::Int
    module_not_enabled::Int
    no_subtree_found::Int
    sample_routine_failures::Int

    total_samples::Int
    encoding_failures::Int
    decoding_failures::Int
    tree_build_failures::Int
    tree_comparison_failures::Int
    skeleton_not_novel::Int
    expr_similarity_failures::Int
    orig_tree_eval_failures::Int
    new_tree_eval_failures::Int
    returned_nonsimilar_exprs::Int
    returned_similar_exprs::Int

    # Tree sizes
    total_tree_sizes::Vector{Int}
    subtree_in_sizes::Vector{Int}
    subtree_out_sizes::Vector{Int}

    # Tree eval
    orig_subtree_string::Vector{String}
    new_subtree_string::Vector{String}

    # MSEs
    sampled_mse::Vector{Float32}

    function NeuralMutationStats(
        total_attempts=0,  # Overall methods calls
        successful_mutations=0,
        module_not_enabled=0,
        no_subtree_found=0,
        sample_routine_failures=0,  # Failed to sample new subtree
        
        total_samples=0,  # Total samples from neural network (multiple per neural_mutate() call possible)

        encoding_failures=0,  # Following are issues during sampling routine
        decoding_failures=0,
        tree_build_failures=0,
        tree_comparison_failures=0,
        skeleton_not_novel=0,
        expr_similarity_failures=0,
        orig_tree_eval_failures=0,
        new_tree_eval_failures=0,
        returned_nonsimilar_exprs=0,
        returned_similar_exprs=0,
        total_tree_sizes=Int[],  # For successfull mutations: Sizes
        subtree_in_sizes=Int[],
        subtree_out_sizes=Int[],
        sampled_mse=Float32[],
        orig_subtree_string=String[],
        new_subtree_string=String[]
    )
        new(
            total_attempts,
            successful_mutations,
            module_not_enabled,
            no_subtree_found,
            sample_routine_failures,
        
            total_samples,
            encoding_failures,
            decoding_failures,
            tree_build_failures,
            tree_comparison_failures,
            skeleton_not_novel,
            expr_similarity_failures,
            orig_tree_eval_failures,
            new_tree_eval_failures,
            returned_nonsimilar_exprs,
            returned_similar_exprs,
            total_tree_sizes,
            subtree_in_sizes,
            subtree_out_sizes,
            sampled_mse,
            orig_subtree_string,
            new_subtree_string
        )
    end
end

const STATS_REF = Ref{NeuralMutationStats}(NeuralMutationStats())

function add_to_stats!(stats::NeuralMutationStats, type::Symbol, with_lock::Bool, value)
    with_lock && lock(STATS_LOCK)
    vec = getfield(stats, type)
    push!(vec, value)
    with_lock && unlock(STATS_LOCK)
end

function increment_stats!(stats::NeuralMutationStats, type::Symbol, with_lock::Bool)
    with_lock && lock(STATS_LOCK)
    setfield!(stats, type, getfield(stats, type) + 1)
    with_lock && unlock(STATS_LOCK)
end

function reset_mutation_stats!()
    STATS_REF[] = NeuralMutationStats()
end

function get_mutation_stats()
    return STATS_REF[]
end

function setup_module(options::AbstractOptions)
    OPTIONS_REF[] = options

    if !options.neural_options.active
        @info "Neural mutation module is disabled but was called. Skipping setup. (is MutationWeights.neural_mutate_tree set to >0.0?)"
        ENABLED_REF[] = false
        return
    end
    @info "Initializing sampling model."
    load_model(options)

    # nn_config and supported operators are NN-specific and cannot be changed for now. TODO: Infer this from model 
    # (no need to specify as it only works if it agrees with model anyways)
    CFG_REF[] = nn_config(
        nbin=4,
        nuna=4, 
        nvar=1,
        seq_len=15
    )
    
    try
        ops = [options.operators.binops..., options.operators.unaops...]
        @info("ops: $ops")
        # Assert all required operators are present. FIXME: Make this dynamic, dependent on loaded sampling model
        function check_op(op_name, ops)
            any(op -> string(op) == op_name, ops) || error("$op_name operator not found in options")
        end
        check_op("+", ops)
        check_op("-", ops)
        check_op("*", ops)
        check_op("/", ops)
        check_op("sin", ops)
        check_op("cos", ops)
        check_op("exp", ops)
        check_op("zero_sqrt", ops)

        @assert length(ops) == (CFG_REF[].nbin + CFG_REF[].nuna) "Additional operators not found in options: $ops"
        # Mapping operator string to SR.jl op index (1-indexed for each arity)
        op_index = Dict{String, Int}(
            "ADD" => findfirst(op -> string(op) == "+", ops),
            "SUB" => findfirst(op -> string(op) == "-", ops),
            "MUL" => findfirst(op -> string(op) == "*", ops),
            "DIV" => findfirst(op -> string(op) == "/", ops),
            "SIN" => findfirst(op -> string(op) == "sin", ops) - (CFG_REF[].nbin),
            "COS" => findfirst(op -> string(op) == "cos", ops) - (CFG_REF[].nbin),
            "EXP" => findfirst(op -> string(op) == "exp", ops) - (CFG_REF[].nbin),
            "ZERO_SQRT" => findfirst(op -> string(op) == "zero_sqrt", ops) - (CFG_REF[].nbin),
        )
        @assert maximum(values(op_index)) == maximum([CFG_REF[].nbin,  CFG_REF[].nuna]) "Operator index out of bounds"
        
        # Conversion SR.jl op index to logit index. TODO: This should be carried by the sampler model.
        logit_index = ["ADD", "SUB", "MUL", "DIV", "SIN", "COS", "EXP", "ZERO_SQRT", "CON", "x1", "END"]

        # Create map from (SR.jl op_idx, arity) -> logits_idx
        op_to_logits = Dict{Tuple{Int,Int}, Int}()
        for (op_str, srjl_idx) in op_index
            arity = OPERATOR_ARITY[op_str]
            logits_idx = findfirst(==(op_str), logit_index)
            op_to_logits[(srjl_idx, arity)] = logits_idx
        end
        OP_TO_LOGITS_REF[] = op_to_logits

        # println("op_index: $op_index")
        # println("op_to_logits: $op_to_logits")
        OP_INDEX_REF[] = op_index
        OP_TO_LOGITS_REF[] = op_to_logits
        reset_mutation_stats!()
        ENABLED_REF[] = true
    catch e
        @error "Error setting up neural mutations. Disabling module. $e"
        ENABLED_REF[] = false
    end
end

function load_model(options::AbstractOptions)
    if options.neural_options.device == "cuda"
        if CUDA_AVAILABLE[]
            @info "CUDA available. Loading model on GPU."    
        else 
            @warn "CUDA package not available. Falling back to CPU."
            options.neural_options.device = "cpu"
        end
    end
    MODEL_REF[] = ORT.load_inference(options.neural_options.model_path, execution_provider=Symbol(options.neural_options.device))
end

"""
    sample_logits(x::AbstractArray{Float32}, eps::Float64=0.01, sample_count::Int=1)::AbstractArray{Float32}

Sample the logits of the neural network.
Currently assuming single sample as input and then sample_count samples as output.
"""
function sample_logits(x::AbstractArray{Float32}, eps::Float64=0.01, sample_count::Int=1)::AbstractArray{Float32}
    input = Dict("onnx::Flatten_0" => reshape(x, (1, size(x)...)), "sample_eps" => [eps], "onnx::Reshape_2" => [sample_count])
    raw_out = MODEL_REF[](input)
    x_out = raw_out["314"]  # [1, :, :]  FIXME: Fix this naming
    return x_out
end

"""
    get_contents_for_mutation(ex::AbstractExpression, rng::AbstractRNG)

Return the contents of an expression, which can be mutated.
You can overload this function for custom expression types that
need to be mutated in a specific way.

The second return value is an optional context object that will be
passed to the `with_contents_for_mutation` function.
"""
function get_contents_for_mutation(ex::AbstractExpression, rng::AbstractRNG)
    return get_contents(ex), nothing
end
"""
    with_contents_for_mutation(ex::AbstractExpression, context)

Replace the contents of an expression with the given context object.
You can overload this function for custom expression types that
need to be mutated in a specific way.
"""
function with_contents_for_mutation(ex::AbstractExpression, new_contents, ::Nothing)
    return with_contents(ex, new_contents)
end

function neural_mutate_tree(
    ex::AbstractExpression{T}, options::AbstractOptions, rng::AbstractRNG=default_rng()
) where {T<:DATA_TYPE}
    tree, context = get_contents_for_mutation(ex, rng)
    ex = with_contents_for_mutation(ex, neural_mutate_tree(tree, options, rng), context)
    return ex
end

function neural_mutate_tree(
    tree::AbstractExpressionNode{T},
    options::AbstractOptions,
    rng::AbstractRNG=default_rng()
) where {T}
    OPTIONS_REF[] !== options && setup_module(options)
    
    increment_stats!(STATS_REF[], :total_attempts, true)
    if !ENABLED_REF[]
        increment_stats!(STATS_REF[], :module_not_enabled, true)
        return tree
    end

    # Select a viable subtree to mutate FIXME: Could also try different tree if this one isn't successfull
    found_subtree, subtree, parent, feature = select_viable_subtree(tree, options.neural_options.subtree_min_nodes, options.neural_options.subtree_max_nodes)
    if !found_subtree
        increment_stats!(STATS_REF[], :no_subtree_found, true)
        return tree
    end

    success, new_subtree = sample_routine(subtree, feature, options)
    if !success
        increment_stats!(STATS_REF[], :sample_routine_failures, true)
        return tree
    end

    lock(STATS_LOCK) do
        add_to_stats!(STATS_REF[], :total_tree_sizes, false, count_nodes(tree))
        add_to_stats!(STATS_REF[], :subtree_in_sizes, false, count_nodes(subtree))
        add_to_stats!(STATS_REF[], :subtree_out_sizes, false, count_nodes(new_subtree))
        if options.neural_options.log_subtree_strings
            add_to_stats!(STATS_REF[], :orig_subtree_string, false, string_tree(subtree, options))
            add_to_stats!(STATS_REF[], :new_subtree_string, false, string_tree(new_subtree, options))
        end
        increment_stats!(STATS_REF[], :successful_mutations, false)
    end
    
    # Replace the old subtree with the new one
    return replace_subtree(tree, parent, subtree, new_subtree)
end

"""
    sample_routine(subtree::AbstractExpressionNode{T}, feature::Int, options::AbstractOptions)::Tuple{Bool, Union{AbstractExpressionNode{T}, Nothing}} where {T}

Sample a new subtree from the neural network. Includes encoding, sampling and decoding. 
Can possibly have multiple attempts to sample a new subtree if the first one fails.

Add: Could use attemp to be more lenient as we come closer to failing otherwise.
"""
function sample_routine(subtree::AbstractExpressionNode{T}, feature::Int, options::AbstractOptions)::Tuple{Bool, Union{AbstractExpressionNode{T}, Nothing}} where {T}
    
    # Keep track of candidates that pass all checks except similarity
    candidates = Vector{Tuple{AbstractExpressionNode{T}, Float64}}()
    current_sample_idx = Inf
    x_out_batch = nothing

    # Encode the subtree into a one-hot vector
    encode_success, x_in = node_to_onehot(subtree, CFG_REF[], OP_TO_LOGITS_REF[])
    if !encode_success
        increment_stats!(STATS_REF[], :encoding_failures, true)
        return false, subtree
    end

    for attempt in 1:(options.neural_options.max_resamples+1)
        increment_stats!(STATS_REF[], :total_samples, true)
        
        # Grab new output  TODO: Make this an object or somehow outsource current_sample_idx 
        if current_sample_idx >= options.neural_options.sample_batchsize  # Used last sample from batch: Resample.
            x_out_batch = sample_logits(x_in, options.neural_options.sampling_eps, options.neural_options.sample_batchsize)
            current_sample_idx = 1
        else  # Still have samples left in batch: Use this.
            current_sample_idx += 1
        end
        x_out = x_out_batch[current_sample_idx, :, :]

        success, prods = logits_to_prods(x_out, options.neural_options.sample_logits)
        if !success
            increment_stats!(STATS_REF[], :decoding_failures, true)
            continue
        end
        
        success, new_subtree = prods_to_tree(prods, OP_INDEX_REF[], feature, T)  # Creates subtree of same type as initial subtree
        if !success
            increment_stats!(STATS_REF[], :tree_build_failures, true)
            continue
        end 
        
        if options.neural_options.require_novel_skeleton
            novel = check_novel_skeleton(x_in, x_out, options)
            if !novel
                increment_stats!(STATS_REF[], :skeleton_not_novel, true)
                continue
            end
        end
        
        if options.neural_options.require_tree_size_similarity
            good = compare_sampled_tree(subtree, new_subtree, options)
            if !good
                increment_stats!(STATS_REF[], :tree_comparison_failures, true)
                continue
            end
        end

        if options.neural_options.require_expr_similarity
            # println("Checking expr similarity")
            is_similar, mse = check_expr_similarity(subtree, new_subtree, options)
            # println("is_similar: $is_similar, mse: $mse")
            if is_similar  # This is the best case: We found a similar expression that (if required above) is novel
                # println("Found similar expression with MSE: $mse")
                increment_stats!(STATS_REF[], :returned_similar_exprs, true)
                add_to_stats!(STATS_REF[], :sampled_mse, false, mse)
                return true, new_subtree
            else
                increment_stats!(STATS_REF[], :expr_similarity_failures, true)
                push!(candidates, (new_subtree, mse))
            end
        else
            return true, new_subtree
        end
    end

    # If we have candidates that failed only the similarity check, return the best one
    if !isempty(candidates)
        best_candidate = argmin(c -> c[2], candidates)
        increment_stats!(STATS_REF[], :returned_nonsimilar_exprs, true)
        add_to_stats!(STATS_REF[], :sampled_mse, false, best_candidate[2])
        return true, best_candidate[1]
    end
    
    return false, subtree
end

function check_expr_similarity(
    subtree::AbstractExpressionNode{T}, 
    new_subtree::AbstractExpressionNode{T}, 
    options::AbstractOptions
)::Tuple{Bool, Float32} where {T}
    # FIXME: Make this hyperparams
    # TODO: Check that Float32 is allowed
    x = Matrix{Float32}(reshape(collect(range(-10.0, 10.0, length=100)), 1, :))
    
    res = nothing
    res_new = nothing
    
    try
        (res, complete) = eval_tree_array(subtree, x, options.operators)
        good = complete && all((res .< prevfloat(typemax(Float32))) .& (res .> nextfloat(typemin(Float32)))) && !any(isnan, res) && !any(isinf, res)
        if !good
            increment_stats!(STATS_REF[], :orig_tree_eval_failures, true)
            return false, Inf
        end
    catch e
        increment_stats!(STATS_REF[], :orig_tree_eval_failures, true)
        return false, Inf
    end
    

    try
        (res_new, complete_new) = eval_tree_array(new_subtree, x, options.operators)
        good_new = complete_new && all((res_new .< prevfloat(typemax(Float32))) .& (res_new .> nextfloat(typemin(Float32)))) && !any(isnan, res_new) && !any(isinf, res_new)
        if !good_new
            increment_stats!(STATS_REF[], :new_tree_eval_failures, true)
            return false, Inf
        end
    catch e
        increment_stats!(STATS_REF[], :new_tree_eval_failures, true)
        return false, Inf
    end

    # Calculate MSE
    mse = sum((asinh.(res) - asinh.(res_new)).^2) / length(x)  # FIXME: Make this MSE metric parameter
    return mse < options.neural_options.similarity_threshold, mse
end

"""
    check_novel_skeleton(x_in::AbstractArray{Float32}, x_out::AbstractArray{Float32}, options::AbstractOptions)

Check if the skeleton of the new subtree is novel.
    FIXME: Not quite correct as logits -> tree is probabilistic and samples from token distribution. Distributions usually have low
    entropy though so this is probably good enough.
"""
function check_novel_skeleton(x_in::Matrix{Float32}, x_out::Matrix{Float32}, options::AbstractOptions)
    # Only check onehot logits, not constants vector
    x_in_onehot = x_in[:, 1:end-1]  # Remove constants column
    x_out_onehot = x_out[:, 1:end-1]
    
    for i in axes(x_in_onehot, 1)  # Iterate over rows
        in_max = argmax(x_in_onehot[i, :])
        out_max = argmax(x_out_onehot[i, :])
        if in_max != out_max
            return true
        end
    end
    return false
end

"""
Function for all comparisons between old and new subtree. FIXME: Could also do this on logits level 
    (though with sampling this might not be correct).
"""
function compare_sampled_tree(
    subtree::AbstractExpressionNode{T}, 
    new_subtree::AbstractExpressionNode{T}, 
    options::AbstractOptions
)::Bool where {T}
    return abs(count_nodes(new_subtree) - count_nodes(subtree)) <= options.neural_options.max_tree_size_diff
end

function replace_subtree(
    tree::AbstractExpressionNode{T}, 
    parent::Union{AbstractExpressionNode{T}, Nothing}, 
    subtree::AbstractExpressionNode{T}, 
    new_subtree::AbstractExpressionNode{T}
)::AbstractExpressionNode{T} where {T}
    if parent === nothing # We sampled the whole tree, no insertion needed
        return new_subtree
    end
    
    if parent.degree == 1
        parent.l = new_subtree
    elseif parent.degree == 2
        if parent.l === subtree
            parent.l = new_subtree
        else
            parent.r = new_subtree
        end
    end
    return tree
end

end