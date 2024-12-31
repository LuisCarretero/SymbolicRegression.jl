module NeuralMutationsModule

using Random: default_rng, AbstractRNG
import ONNXRunTime as ORT
using DynamicExpressions: AbstractExpressionNode, AbstractExpression, NodeSampler, has_operators, with_contents, get_contents
using ..CoreModule: AbstractOptions, DATA_TYPE
using ..ParsingModule: nn_config, node_to_onehot, _create_grammar_masks, logits_to_prods, prods_to_tree, select_viable_subtree, count_nodes

export neural_mutate_tree

const MODEL_REF = Ref{Union{Nothing, ORT.InferenceSession}}(nothing)
const CFG_REF = Ref{Any}(nothing)
const OP_INDEX_REF = Ref{Dict{String,Int}}(Dict{String,Int}())
const OPTIONS_REF = Ref{Union{Nothing, AbstractOptions}}(nothing)
const ENABLED_REF = Ref{Bool}(false)
const STATS_LOCK = ReentrantLock()

function __init__()
    @info "Initializing sampling model."
    MODEL_REF[] = ORT.load_inference("src/dev/ONNX/onnx-models/model-zwrgtnj0.onnx")
end

mutable struct NeuralMutationStats
    total_attempts::Int
    successful_mutations::Int
    no_subtree_found::Int
    module_not_enabled::Int
    encoding_failures::Int
    decoding_failures::Int
    tree_build_failures::Int
    total_tree_sizes::Vector{Int}
    subtree_in_sizes::Vector{Int}
    subtree_out_sizes::Vector{Int}

    function NeuralMutationStats(
        total_attempts=0,
        successful_mutations=0,
        no_subtree_found=0,
        module_not_enabled=0,
        encoding_failures=0,
        decoding_failures=0,
        tree_build_failures=0
    )
        new(
            total_attempts,
            successful_mutations,
            no_subtree_found,
            module_not_enabled,
            encoding_failures,
            decoding_failures,
            tree_build_failures,
            Int[],  # Initialize empty vectors
            Int[],
            Int[]
        )
    end
end

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

const STATS_REF = Ref{NeuralMutationStats}(NeuralMutationStats())
"""
    reset_mutation_stats!()

Reset all neural mutation statistics to their initial values.
"""
function reset_mutation_stats!()
    STATS_REF[] = NeuralMutationStats()
end

"""
    get_mutation_stats()

Return the current neural mutation statistics.
"""
function get_mutation_stats()
    return STATS_REF[]
end


function set_config_and_ops(options)
    CFG_REF[] = nn_config(
        nbin=4,
        nuna=6, 
        nvar=1,
        seq_len=15
    )
    
    try
        ops = [options.operators.binops..., options.operators.unaops...]
        # Assert all required operators are present
        @assert (+) in ops "Addition operator not found in options"
        @assert (-) in ops "Subtraction operator not found in options"
        @assert (*) in ops "Multiplication operator not found in options"
        @assert (/) in ops "Division operator not found in options"
        @assert sin in ops "Sine operator not found in options"
        @assert cos in ops "Cosine operator not found in options"
        @assert exp in ops "Exponential operator not found in options"
        @assert tanh in ops "Hyperbolic tangent operator not found in options"
        @assert cosh in ops "Hyperbolic cosine operator not found in options"
        @assert sinh in ops "Hyperbolic sine operator not found in options"

        op_index = Dict{String, Int}(
            "ADD" => findfirst(==(+), ops),
            "SUB" => findfirst(==(-), ops), 
            "MUL" => findfirst(==(*), ops),
            "DIV" => findfirst(==(/), ops),
            "SIN" => findfirst(==(sin), ops) - (CFG_REF[].nbin),
            "COS" => findfirst(==(cos), ops) - (CFG_REF[].nbin),
            "EXP" => findfirst(==(exp), ops) - (CFG_REF[].nbin),
            "TANH" => findfirst(==(tanh), ops) - (CFG_REF[].nbin),
            "COSH" => findfirst(==(cosh), ops) - (CFG_REF[].nbin),
            "SINH" => findfirst(==(sinh), ops) - (CFG_REF[].nbin),
        )
        OP_INDEX_REF[] = op_index
        OPTIONS_REF[] = options
        reset_mutation_stats!()
        ENABLED_REF[] = true
    catch e
        @error "Error setting config and ops: $e"
        ENABLED_REF[] = false
    end
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
    min_nodes = 3
    max_nodes = 14
    sample_eps = 0.01

    increment_stats!(STATS_REF[], :total_attempts, true)

    if OPTIONS_REF[] !== options
        set_config_and_ops(options)
    end
    if !ENABLED_REF[]
        @info "Neural mutations are disabled. Returning original tree."
        increment_stats!(STATS_REF[], :module_not_enabled, true)
        return tree
    end

    # Select a viable subtree to mutate
    found_subtree, subtree, parent, feature = select_viable_subtree(tree, min_nodes, max_nodes)
    if !found_subtree
        increment_stats!(STATS_REF[], :no_subtree_found, true)
        return tree
    end
    
    # Encode the subtree into a one-hot vector
    encode_success, x = node_to_onehot(subtree, CFG_REF[])
    if !encode_success
        increment_stats!(STATS_REF[], :encoding_failures, true)
        return tree
    end
    
    # Sample new subtree
    x_out = sample_logits(x, sample_eps)
    success, prods = logits_to_prods(x_out, true)
    if !success
        increment_stats!(STATS_REF[], :decoding_failures, true)
        return tree
    end

    success, new_subtree = prods_to_tree(prods, OP_INDEX_REF[], feature)
    if !success
        increment_stats!(STATS_REF[], :tree_build_failures, true)
        return tree
    end 

    lock(STATS_LOCK) do
        add_to_stats!(STATS_REF[], :total_tree_sizes, false, count_nodes(tree))
        add_to_stats!(STATS_REF[], :subtree_in_sizes, false, count_nodes(subtree))
        add_to_stats!(STATS_REF[], :subtree_out_sizes, false, count_nodes(new_subtree))
        increment_stats!(STATS_REF[], :successful_mutations, false)
    end
    
    # Replace the old subtree with the new one
    if parent === nothing
        return new_subtree
    else
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

end