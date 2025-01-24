module NeuralMutationsModule

using Random: default_rng, AbstractRNG
import ONNXRunTime as ORT
using DynamicExpressions: AbstractExpressionNode, AbstractExpression, NodeSampler, has_operators, with_contents, get_contents
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
    # Tree sizes
    total_tree_sizes::Vector{Int}
    subtree_in_sizes::Vector{Int}
    subtree_out_sizes::Vector{Int}

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
        
        total_tree_sizes=Int[],  # For successfull mutations: Sizes
        subtree_in_sizes=Int[],
        subtree_out_sizes=Int[]
    )
        new(
            total_attempts,
            successful_mutations,
            no_subtree_found,
            module_not_enabled,
            encoding_failures,
            decoding_failures,
            tree_build_failures,
            tree_comparison_failures,
            sample_routine_failures,
            total_samples,
            total_tree_sizes,
            subtree_in_sizes,
            subtree_out_sizes
        )
    end
end

const STATS_REF = Ref{NeuralMutationStats}(NeuralMutationStats())

function add_to_stats!(stats::NeuralMutationStats, type::Symbol, with_lock::Bool, value::Int)
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

        # @assert tanh in ops "Hyperbolic tangent operator not found in options"
        # @assert cosh in ops "Hyperbolic cosine operator not found in options"
        # @assert sinh in ops "Hyperbolic sine operator not found in options"

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
    sample_logits(x::AbstractArray{Float32}, eps::Float64=0.01)::AbstractArray{Float32}

Sample the logits of the neural network.
"""
function sample_logits(x::AbstractArray{Float32}, eps::Float64=0.01)::AbstractArray{Float32}
    input = Dict("onnx::Flatten_0" => reshape(x, (1, size(x)...)), "sample_eps" => [eps])
    raw_out = MODEL_REF[](input)
    x_out = raw_out["276"][1, :, :]
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
    rng::AbstractRNG=default_rng(),
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
    
    for attempt in 1:(options.neural_options.max_resamples+1)
        increment_stats!(STATS_REF[], :total_samples, true)

        # Encode the subtree into a one-hot vector
        encode_success, x = node_to_onehot(subtree, CFG_REF[], OP_TO_LOGITS_REF[])
        if !encode_success
            increment_stats!(STATS_REF[], :encoding_failures, true)
            continue
        end
        
        # Sample new subtree
        x_out = sample_logits(x, options.neural_options.sampling_eps)
        success, prods = logits_to_prods(x_out, true)
        if !success
            increment_stats!(STATS_REF[], :decoding_failures, true)
            continue
        end

        success, new_subtree = prods_to_tree(prods, OP_INDEX_REF[], feature)
        if !success
            increment_stats!(STATS_REF[], :tree_build_failures, true)
            continue
        end 

        good = compare_sampled_tree(subtree, new_subtree, options)
        if !good
            increment_stats!(STATS_REF[], :tree_comparison_failures, true)
            continue
        end
        return true, new_subtree
    end
    return false, subtree
end

"""
Function for all comparisons between old and new subtree. 
"""
function compare_sampled_tree(subtree::AbstractExpressionNode{T1}, new_subtree::AbstractExpressionNode{T2}, options::AbstractOptions) where {T1,T2}
    is_good = true
    if options.neural_options.max_tree_size_diff != 0
        is_good &= abs(count_nodes(new_subtree) - count_nodes(subtree)) <= options.neural_options.max_tree_size_diff
    end
    return is_good
end

function replace_subtree(tree::AbstractExpressionNode{T}, parent::Union{AbstractExpressionNode{T}, Nothing}, subtree::AbstractExpressionNode{T}, new_subtree::AbstractExpressionNode{T}) where {T}
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