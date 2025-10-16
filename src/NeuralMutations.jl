module NeuralMutationsModule

using Random: default_rng, AbstractRNG
import ONNXRunTime as ORT
using Distributions: Uniform
using DynamicExpressions: AbstractExpressionNode, AbstractExpression, NodeSampler, has_operators, with_contents, get_contents, eval_tree_array, string_tree
using ..CoreModule: AbstractOptions, DATA_TYPE
using ..ParsingModule: nn_config, node_to_onehot, logits_to_prods, prods_to_tree, select_viable_subtree, count_nodes, OPERATOR_ARITY

# Add CUDA imports at module level
const CUDA_IMPORTED = Ref{Bool}(false)
try
    using CUDA
    using cuDNN
    global CUDA_IMPORTED[] = true
catch
    global CUDA_IMPORTED[] = false
end

export neural_mutate_tree

const MODEL_REF = Ref{Union{Nothing, ORT.InferenceSession}}(nothing)
const CFG_REF = Ref{Any}(nothing)
const OP_INDEX_REF = Ref{Dict{String,Int}}(Dict{String,Int}())
const OP_TO_LOGITS_REF = Ref{Dict{Tuple{Int,Int}, Int}}(Dict{Tuple{Int,Int}, Int}())
const OPTIONS_REF = Ref{Union{Nothing, AbstractOptions}}(nothing)
const ENABLED_REF = Ref{Bool}(false)
const STATS_LOCK = ReentrantLock()
const EVAL_X_UNIVARIATE_REF = Ref{Union{Nothing, Matrix}}(nothing)
const EVAL_TRANSFORM_REF = Ref{Function}(identity)

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

    # Multivariate decoding stats
    multivardec_attempts::Int
    multivardec_totree_failures::Int
    multivardec_similarity_failures::Int

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
        new_subtree_string=String[],
        multivardec_attempts=0,
        multivardec_totree_failures=0,
        multivardec_similarity_failures=0
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
            new_subtree_string,
            multivardec_attempts,
            multivardec_totree_failures,
            multivardec_similarity_failures
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

        OP_INDEX_REF[] = op_index
        OP_TO_LOGITS_REF[] = op_to_logits

        # Pre-compute evaluation data for similarity checks
        # Get the transform function by name
        transform_name = options.neural_options.eval_transform
        try
            EVAL_TRANSFORM_REF[] = getfield(Main, Symbol(transform_name))
        catch
            error("Invalid eval_transform: $transform_name. Function not found in Main module.")
        end
        # Pre-compute univariate X (always the same size)
        T = Float32  # Use Float32 for consistency with neural network operations
        EVAL_X_UNIVARIATE_REF[] = Matrix{T}(reshape(
            collect(range(T(options.neural_options.eval_min),
                          T(options.neural_options.eval_max),
                          length=options.neural_options.eval_npoints)),
            1, :
        ))

        reset_mutation_stats!()
        ENABLED_REF[] = true
    catch e
        @error "Error setting up neural mutations. Disabling module. $e"
        ENABLED_REF[] = false
    end
end

function load_model(options::AbstractOptions)
    if options.neural_options.device == "cuda"
        if CUDA_IMPORTED[] && CUDA.functional()
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
    input = Dict(
        "input_syntax" => reshape(x, (1, size(x)...)), 
        "sample_eps" => [eps], 
        "sample_count" => [sample_count]
    )
    raw_out = MODEL_REF[](input)
    x_out = raw_out["output_logits"]
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
    found_subtree, subtree, parent, feature_set = select_viable_subtree(tree, options)
    if !found_subtree
        increment_stats!(STATS_REF[], :no_subtree_found, true)
        return tree
    end

    success, new_subtree, mse = sample_routine(subtree, feature_set, options)
    if !success
        increment_stats!(STATS_REF[], :sample_routine_failures, true)
        return tree
    end

    lock(STATS_LOCK) do
        add_to_stats!(STATS_REF[], :total_tree_sizes, false, count_nodes(tree))
        add_to_stats!(STATS_REF[], :subtree_in_sizes, false, count_nodes(subtree))
        add_to_stats!(STATS_REF[], :subtree_out_sizes, false, count_nodes(new_subtree))
        if options.neural_options.require_expr_similarity
            add_to_stats!(STATS_REF[], :sampled_mse, false, mse)
        end
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

TODO: Could use attempt to be more lenient as we come closer to failing otherwise.

Note that subtree may be multivariate but node_to_onehot replaces all features with a single one. Will have to map this back to the original features.
"""
function sample_routine(subtree::AbstractExpressionNode{T}, feature_set::Set{Int}, options::AbstractOptions)::Tuple{Bool, Union{AbstractExpressionNode{T}, Nothing}, Float64} where {T}

    # Keep track of candidates that pass all checks except similarity
    candidates = Vector{Tuple{AbstractExpressionNode{T}, Float64}}()
    current_sample_idx = Inf
    x_out_batch = nothing

    # Get the original feature used in the subtree
    original_feature = length(feature_set) > 0 ? first(feature_set) : 1

    # Remap original subtree to x1 for evaluation (no-op if already x1)
    subtree_x1 = remap_features(subtree, original_feature, 1)

    # Encode the subtree into a one-hot vector (structure only, feature doesn't affect encoding)
    encode_success, x_in = node_to_onehot(subtree, CFG_REF[], OP_TO_LOGITS_REF[])
    if !encode_success
        increment_stats!(STATS_REF[], :encoding_failures, true)
        return false, subtree, Inf
    end

    for attempt in 1:(options.neural_options.max_resamples+1)
        increment_stats!(STATS_REF[], :total_samples, true)
        
        # Grab new output  TODO: Make this an object or somehow outsource current_sample_idx 
        if current_sample_idx >= options.neural_options.sample_batchsize  # Used last sample from batch: Resample.
            x_out_batch = sample_logits(x_in, options.neural_options.sampling_eps, options.neural_options.sample_batchsize)
            current_sample_idx = 1
        else  # Still have samples left in batch: Use these.
            current_sample_idx += 1
        end
        x_out = x_out_batch[current_sample_idx, :, :]

        if options.neural_options.require_novel_skeleton && !check_novel_skeleton(x_in, x_out, options)
            increment_stats!(STATS_REF[], :skeleton_not_novel, true)
            continue
        end

        success, prods = logits_to_prods(x_out, options.neural_options.sample_logits)
        if !success
            increment_stats!(STATS_REF[], :decoding_failures, true)
            continue
        end
        
        # Sampling model is implicitly univariate and returns only x1 as features after decoding via `logits_to_prods`
        feature_cnt_new = count(p -> p[2] == "'x1'", prods)

        if options.neural_options.subtree_max_features == 1  # Univariate decoding
            # Always generate with feature 1 for evaluation
            success, new_subtree_x1 = prods_to_tree(prods, OP_INDEX_REF[], fill(1, feature_cnt_new), T)

            if !success
                increment_stats!(STATS_REF[], :tree_build_failures, true)
                continue
            end

            if options.neural_options.require_tree_size_similarity
                good = verify_tree_size_similar(subtree, new_subtree_x1, options)
                if !good
                    increment_stats!(STATS_REF[], :tree_comparison_failures, true)
                    continue
                end
            end

            if options.neural_options.require_expr_similarity
                # Compare both using x1 (matches EVAL_X_UNIVARIATE_REF dimensions)
                is_similar, mse = check_expr_similarity(subtree_x1, new_subtree_x1, options, feature_cnt_new)

                if is_similar
                    increment_stats!(STATS_REF[], :returned_similar_exprs, true)
                    # Remap back to original feature before returning
                    new_subtree = remap_features(new_subtree_x1, 1, original_feature)
                    return true, new_subtree, mse
                else
                    increment_stats!(STATS_REF[], :expr_similarity_failures, true)
                    # Store x1 version in candidates
                    push!(candidates, (new_subtree_x1, mse))
                end
            else
                # No similarity requirement - remap and return immediately
                new_subtree = remap_features(new_subtree_x1, 1, original_feature)
                return true, new_subtree, Inf
            end

        else  # Multivariate decoding
            # multivariate_decoding handles feature combinations and similarity checks internally
            success, new_subtree, is_similar, mse = multivariate_decoding(subtree, prods, feature_cnt_new, feature_set, options)

            if !success
                increment_stats!(STATS_REF[], :tree_build_failures, true)
                continue
            end

            if options.neural_options.require_tree_size_similarity
                good = verify_tree_size_similar(subtree, new_subtree, options)
                if !good
                    increment_stats!(STATS_REF[], :tree_comparison_failures, true)
                    continue
                end
            end

            if options.neural_options.require_expr_similarity
                # multivariate_decoding already checked similarity
                if is_similar
                    increment_stats!(STATS_REF[], :returned_similar_exprs, true)
                    return true, new_subtree, mse
                else
                    increment_stats!(STATS_REF[], :expr_similarity_failures, true)
                    push!(candidates, (new_subtree, mse))
                end
            else
                # No similarity requirement - return immediately
                return true, new_subtree, Inf
            end
        end
    end

    # If we have candidates that failed only the similarity check, return the best one
    if !isempty(candidates)
        best_candidate = argmin(c -> c[2], candidates)
        candidate_subtree, mse = best_candidate

        if options.neural_options.subtree_max_features == 1
            # Univariate candidates stored with x1 - remap back to original feature
            new_subtree = remap_features(candidate_subtree, 1, original_feature)
        else
            # Multivariate candidates already have correct features
            new_subtree = candidate_subtree
        end

        increment_stats!(STATS_REF[], :returned_nonsimilar_exprs, true)
        return true, new_subtree, mse
    end

    return false, subtree, Inf
end

"""
Exhaustively try all feature combinations and return the best one.
"""
function multivariate_decoding(
    subtree::AbstractExpressionNode{T}, 
    prods::Vector{Tuple{String, String}}, 
    feature_cnt::Int, 
    feature_set::Set{Int}, 
    options::AbstractOptions
)::Tuple{Bool, Union{AbstractExpressionNode{T}, Nothing}, Bool, Float64} where {T}
    best_mse = Inf
    best_subtree = nothing
    best_is_similar = false

    for features in get_all_feature_combinations(feature_set, feature_cnt)
        increment_stats!(STATS_REF[], :multivardec_attempts, true)

        success, new_subtree = prods_to_tree(prods, OP_INDEX_REF[], features, T)
        if !success  # Tree build failed with this specific feature vector but will also fail with all others (skeleton is the same). Return.
            increment_stats!(STATS_REF[], :multivardec_totree_failures, true)
            return false, subtree, false, Inf
        end

        is_similar, mse = check_expr_similarity(subtree, new_subtree, options, feature_cnt)
        if mse == Inf
            continue
        end
        if mse < best_mse
            best_mse, best_subtree, best_is_similar = mse, new_subtree, is_similar
        end
    end

    if best_subtree === nothing
        increment_stats!(STATS_REF[], :multivardec_similarity_failures, true)
        return false, subtree, false, Inf
    end
    
    return true, best_subtree, best_is_similar, best_mse
end

function get_all_feature_combinations(feature_set::Set{Int}, feature_cnt::Int)
    features = collect(feature_set)
    if feature_cnt == 1
        return [[f] for f in features]
    else
        # Generate all possible combinations of length feature_cnt
        result = Vector{Int}[]
        for combo in Base.Iterators.product(fill(features, feature_cnt)...)
            push!(result, collect(combo))
        end
        return result
    end
end

function check_expr_similarity(
    subtree::AbstractExpressionNode{T},
    new_subtree::AbstractExpressionNode{T},
    options::AbstractOptions,
    feature_cnt::Int=1
)::Tuple{Bool, Float32} where {T}
    if feature_cnt == 1
        # Use pre-computed univariate X
        X = Matrix{T}(EVAL_X_UNIVARIATE_REF[])
    elseif feature_cnt > 1
        points_cnt = options.neural_options.eval_npoints * feature_cnt^2  # Is this a good scaling?
        X = Matrix{T}(rand(Uniform(T(options.neural_options.eval_min), T(options.neural_options.eval_max)), feature_cnt, points_cnt))
    end
    
    res = nothing
    res_new = nothing
    
    # Evaluate (old) subtree
    try
        (res, complete) = eval_tree_array(subtree, X, options.operators)
        if !complete || any(x -> isnan(x) || isinf(x) || x >= prevfloat(typemax(T)) || x <= nextfloat(typemin(T)), res)
            increment_stats!(STATS_REF[], :orig_tree_eval_failures, true)
            return false, Inf
        end
    catch e
        increment_stats!(STATS_REF[], :orig_tree_eval_failures, true)
        return false, Inf
    end
    
    # Evaluate new subtree
    try
        (res_new, complete_new) = eval_tree_array(new_subtree, X, options.operators)
        if !complete_new || any(x -> isnan(x) || isinf(x) || x >= prevfloat(typemax(T)) || x <= nextfloat(typemin(T)), res_new)
            increment_stats!(STATS_REF[], :new_tree_eval_failures, true)
            return false, Inf
        end
    catch e
        increment_stats!(STATS_REF[], :new_tree_eval_failures, true)
        return false, Inf
    end

    # Calculate MSE with configured transform
    transform = EVAL_TRANSFORM_REF[]
    mse_T = sum((transform.(res) .- transform.(res_new)).^2) / size(X)[2]
    mse = Float32(mse_T)
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
function verify_tree_size_similar(
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

"""
    remap_features(tree::AbstractExpressionNode{T}, from_feature::Int, to_feature::Int)::AbstractExpressionNode{T}

Creates a copy of the tree with all variable nodes using from_feature remapped to to_feature.
Returns a new tree, leaving the original unchanged.

Early returns if from_feature == to_feature to avoid unnecessary copying (optimization for x1-only cases).
"""
function remap_features(tree::AbstractExpressionNode{T}, from_feature::Int, to_feature::Int)::AbstractExpressionNode{T} where T
    # Optimization: if features are the same, no remapping needed
    if from_feature == to_feature
        return tree
    end

    # Create a copy of the current node
    new_node = typeof(tree)()
    new_node.degree = tree.degree

    if tree.degree == 0
        # Leaf node
        new_node.constant = tree.constant
        if tree.constant
            new_node.val = tree.val
        else
            # Variable node - remap if matches
            new_node.feature = (tree.feature == from_feature) ? to_feature : tree.feature
        end
    else
        # Operator node
        new_node.op = tree.op
        new_node.l = remap_features(tree.l, from_feature, to_feature)
        if tree.degree == 2
            new_node.r = remap_features(tree.r, from_feature, to_feature)
        end
    end

    return new_node
end

end