module NeuralMutationsModule

using Random: default_rng, AbstractRNG
import ONNXRunTime as ORT
using DynamicExpressions: AbstractExpressionNode, AbstractExpression, NodeSampler, has_operators, with_contents, get_contents
using ..CoreModule: AbstractOptions, DATA_TYPE
using ..ParsingModule: nn_config, node_to_onehot, _create_grammar_masks, logits_to_prods, prods_to_tree, select_subtree

export neural_mutate_tree

const MODEL_REF = Ref{Union{Nothing, ORT.InferenceSession}}(nothing)
const CFG_REF = Ref{Any}(nothing)
const OP_INDEX_REF = Ref{Dict{String,Int}}(Dict{String,Int}())

function __init__()
    @info "Initializing sampling model."
    MODEL_REF[] = ORT.load_inference("src/dev/ONNX/onnx-models/model-zwrgtnj0.onnx")
end

function initialize_config_and_ops(options)
    if CFG_REF[] === nothing
        CFG_REF[] = nn_config(
            nbin=4,
            nuna=6, 
            nvar=1,
            seq_len=15
        )
        
        ops = [options.operators.binops..., options.operators.unaops...]
        OP_INDEX_REF[] = Dict{String, Int}(
            "ADD" => findfirst(==(+), ops),
            "SUB" => findfirst(==(-), ops), 
            "MUL" => findfirst(==(*), ops),
            "DIV" => findfirst(==(/), ops),
            "SIN" => findfirst(==(sin), ops)-CFG_REF[].nbin,
            "COS" => findfirst(==(cos), ops)-CFG_REF[].nbin,
            "EXP" => findfirst(==(exp), ops)-CFG_REF[].nbin,
            "TANH" => findfirst(==(tanh), ops)-CFG_REF[].nbin,
            "COSH" => findfirst(==(cosh), ops)-CFG_REF[].nbin,
            "SINH" => findfirst(==(sinh), ops)-CFG_REF[].nbin,
        )
    end
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
    if MODEL_REF[] === nothing || CFG_REF[] === nothing || OP_INDEX_REF[] === nothing
        initialize_config_and_ops(options)
    end

    # # Select a viable subtree to mutate
    found_subtree, subtree, parent, feature = select_subtree(tree)
    # @info "Found subtree: $found_subtree; subtree: $subtree; parent: $parent; feature: $feature"
    !found_subtree && return tree
    try
        # Encode the subtree into a one-hot vector
        encode_success, x = node_to_onehot(subtree, CFG_REF[])
        # @info "Encoded subtree: $encode_success"
        if encode_success
            # Sample new subtree
            input = Dict("onnx::Flatten_0" => reshape(x, (1, size(x)...)), "sample_eps" => [0.05])
            raw_out = MODEL_REF[](input)
            x_out = raw_out["276"][1, :, :]
            prods = logits_to_prods(x_out, true)
            new_subtree = prods_to_tree(prods, OP_INDEX_REF[], feature)
            # @info "Created new subtree: $subtree -> $new_subtree"

            # Replace the old subtree with the new one
            if parent === nothing
                # If there's no parent, this means we're replacing the root
                return new_subtree
            else
                # Replace the appropriate child in the parent node
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
        else
            return tree
        end
    catch e
        # @error "Error in neural_mutate_tree: $e"
        return tree
    end
end

end