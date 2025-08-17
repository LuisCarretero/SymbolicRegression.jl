module ParsingModule

using DynamicExpressions: AbstractExpressionNode, Node
using ..CoreModule: AbstractOptions
using Distributions: Categorical


# Define grammar rules similar to Python version
# grammar_str = """
# S -> 'ADD' S S | 'SUB' S S | 'MUL' S S | 'DIV' S S
# S -> 'SIN' S | 'COS' S | 'EXP' S | 'TANH' S | 'COSH' S | 'SINH' S 
# S -> 'CON'
# S -> 'x1'
# END -> 'END'
# """
# FIXME: This needs to be adjusted to always have the same order as syntax_cats used in model
# Currently using: logit_index = ["ADD", "SUB", "MUL", "DIV", "SIN", "COS", "EXP", "ZERO_SQRT", "CON", "x1", "END"]
GRAMMAR_STR_RAW = """
S -> 'ADD' S S | 'SUB' S S | 'MUL' S S | 'DIV' S S
S -> 'SIN' S | 'COS' S | 'EXP' S | 'ZERO_SQRT' S
S -> 'CON'
S -> 'x1'
END -> 'END'
"""

const OPERATOR_ARITY = Dict{String, Int}(
    # Elementary functions
    "ADD" => 2,
    "SUB" => 2,
    "MUL" => 2,
    "DIV" => 2,

    # Trigonometric Functions
    "SIN" => 1,
    "COS" => 1,
    "EXP" => 1,

    # Hyperbolic Functions
    "SINH" => 1,
    "COSH" => 1,
    "TANH" => 1,

    # Custom operator
    "ZERO_SQRT" => 1,

    # FIXME: Handle this differently!
    "x1" => 0
)

# Split each production rule into separate lines
function _split_grammar_rules(grammar_str::String)::String
    grammar_lines = String[]
    for line in split(grammar_str, "\n")
        line = strip(line)
        if !isempty(line)
            lhs, rhs = split(line, "->")
            lhs = strip(lhs)
            # Split on | and create new lines
            for rule in split(rhs, "|")
                push!(grammar_lines, "$lhs -> $(strip(rule))")
            end
        end
    end
    return join(grammar_lines, "\n")
end

function _create_grammar_masks(grammar_str::String)::Tuple{Matrix{Bool}, Vector{Int}, Vector{String}}
    # Collect all LHS symbols and unique set
    all_lhs = String[]
    unique_lhs = String[]
    
    # Parse grammar string to get productions
    for line in split(grammar_str, "\n")
        if !isempty(strip(line))
            lhs = strip(split(line, "->")[1])
            push!(all_lhs, lhs)
            if !(lhs in unique_lhs)
                push!(unique_lhs, lhs)
            end
        end
    end

    # Create masks matrix - each row corresponds to a unique LHS symbol
    # and has 1s for productions with that LHS
    masks = falses(length(unique_lhs), length(all_lhs))
    for (i, symbol) in enumerate(unique_lhs)
        for (j, lhs) in enumerate(all_lhs)
            masks[i,j] = (symbol == lhs)
        end
    end

    allowed_prod_idx = findall(vec(any(masks, dims=1)))

    return masks, allowed_prod_idx, unique_lhs
end

# FIXME: Call in some kind of init function
grammar_str = _split_grammar_rules(GRAMMAR_STR_RAW)
grammar_masks, allowed_prod_idx, unique_lhs = _create_grammar_masks(grammar_str)

mutable struct nn_config
    nbin::Int
    nuna::Int
    nvar::Int

    seq_len::Int
    nb_onehot_cats::Int

    function nn_config(;nbin::Int, nuna::Int, nvar::Int, seq_len::Int)
        nb_onehot_cats = nbin + nuna + nvar + 2  # +1 for constants, +1 for end token
        return new(nbin, nuna, nvar, seq_len, nb_onehot_cats)
    end
end


function node_to_onehot(node::Node{T}, cfg::nn_config, op_to_logits::Dict{Tuple{Int,Int}, Int})::Tuple{Bool, Matrix{Float32}} where T <: Number
    prefix = _tree_to_prefix(node)
    idx = [_node_to_token_idx(node, op_to_logits) for node in prefix]
    success, onehot, consts = _onehot_encode(idx, cfg)
    !success && return (false, nothing)

    x = zeros(Float32, cfg.seq_len, cfg.nb_onehot_cats+1)
    x[:, 1:cfg.nb_onehot_cats] = onehot
    x[:, cfg.nb_onehot_cats+1] = consts
    return (true, x)
end

function _onehot_encode(idx::Vector{Tuple{Int, Float64}}, cfg::nn_config)::Tuple{Bool, Union{BitMatrix, Nothing}, Union{Vector{Float64}, Nothing}}
    onehot = falses(cfg.seq_len, cfg.nb_onehot_cats)
    consts = zeros(Float64, cfg.seq_len)

    length(idx) > cfg.seq_len && return (false, nothing, nothing)

    for (node_i, (token_idx, token_val)) in enumerate(idx)  # Each node has corresponding index and value
        onehot[node_i, token_idx] = true
        consts[node_i] = token_val
    end
    # Pad with empty token (last category)
    onehot[length(idx)+1:end, end] .= true 
    return (true, onehot, consts)
end

function _tree_to_prefix(tree::Node{T})::Vector{Node{T}} where T <: Number
    result = [tree]
    if tree.degree >= 1
        append!(result, _tree_to_prefix(tree.l))
    end
    if tree.degree == 2
        append!(result, _tree_to_prefix(tree.r))
    end
    return result
end

"""
Encoding node (with op index) into token index used by NN.

Needs mapping from (srjl_op_idx, op_deg) -> token_idx.
"""
function _node_to_token_idx(node::Node{T}, op_to_logits::Dict{Tuple{Int,Int}, Int})::Tuple{Int, Float64} where T <: Number
    idx_const = length(op_to_logits) + 1 # [..., "last_op", "CON", "x1", "END"] 
    idx_var = idx_const + 1  # Always assuming we have a single 
    if node.degree == 2
        return (op_to_logits[(node.op, 2)], 0)
    elseif node.degree == 1
        return (op_to_logits[(node.op, 1)], 0)
    elseif node.degree == 0
        if node.constant
            return (idx_const, node.val)  # Only 1 const token
        else
            # FIXME: For multiple variables, do + node.feature instead. 
            return (idx_var, 0)
        end
    end
end

"""
Convert logits to production rules.
First flag is ``success``, second flag is ``prods``.

Uses GRAMMAR to get mapping from token_idx -> op_string.
"""
function logits_to_prods(  # FIXME: Think of better way than to just use global vars for grammar_masks, etc.
    logits::Matrix{Float32},
    sample::Bool=false, 
    max_length::Int=15
)::Tuple{Bool, Union{Vector{Tuple{String, String}}, Nothing}}
    # Initialize empty stack with start symbol 'S'
    stack = ["S"]
    
    # Split logits into productions and constants
    logits_prods = clamp.(logits[:, 1:end-1], -84.0f0, 84.0f0) # Clamp to avoid overflow in exp(x) and sum(exp(x)) for Float32 ( log(floatmax(Float32)/Ncats) )
    constants = logits[:, end]
    
    prods = []
    t = 1
    
    while !isempty(stack)
        alpha = pop!(stack)  # Current LHS toke
        
        # Get mask for current symbol
        symbol_idx = findfirst(==(alpha), unique_lhs)
        mask = grammar_masks[symbol_idx, :]
        
        # Calculate probabilities
        probs = mask .* exp.(logits_prods[t, :])
        tot = sum(probs)
        if tot == 0 || !isfinite(tot)
            return (false, nothing)  # No valid productions or numerical issues
        end
        probs = probs ./ tot
        if !isapprox(sum(probs), one(Float32))  # Same test as in Categorical() to not throw errors
            return (false, nothing)  # Check for NaN/Inf and verify distribution sums to 1
        end
        
        # Select production rule
        if sample
            i = rand(Categorical(probs))
        else
            _, i = findmax(probs)
        end
        
        # Get selected rule
        rule = split(grammar_str, "\n")[i]
        lhs, rhs = split(strip(rule), "->")
        lhs = strip(lhs)
        rhs = strip(rhs)

        if lhs == "END"
            break
        end
        
        # If rule produces CONST, replace with actual constant
        if rhs == "'CON'"
            rhs = string(constants[t])
        end
        
        # Add production to list
        push!(prods, (lhs, rhs))
        
        # Add RHS nonterminals to stack in reverse order
        rhs_symbols = split(rhs)
        for symbol in reverse(rhs_symbols)
            clean_symbol = replace(symbol, "'" => "")
            if clean_symbol in unique_lhs
                push!(stack, clean_symbol)
            end
        end
        
        t += 1
        if t > max_length
            break
        end
    end
    
    return true, prods
end

function is_tree_valid(tree::AbstractExpressionNode{T})::Bool where {T}
    try
        # Basic structure checks
        tree === nothing && return false
        
        # Check if we can access basic properties without crashing
        degree = tree.degree
        (degree < 0 || degree > 2) && return false
        
        # Check children based on degree
        if degree >= 1
            tree.l === nothing && return false
            !is_tree_valid(tree.l) && return false
        end
        if degree == 2
            tree.r === nothing && return false
            !is_tree_valid(tree.r) && return false
        end
        
        # Check if we can access other properties
        if degree == 0
            # For leaf nodes, check if we can access the value/feature
            try
                _ = tree.constant
                _ = tree.feature
            catch
                return false
            end
        else
            # For operator nodes, check if we can access the operator
            try
                _ = tree.op
            catch
                return false
            end
        end
        
        return true
    catch
        return false
    end
end

function prods_to_tree(
    prods::Vector{Tuple{String, String}}, 
    OP_INDEX::Dict{String, Int}, 
    features::Vector{Int}, 
    tree_type::Type{T}
)::Tuple{Bool, Node{T}} where {T <: Number}
    prefix_list = _prods_to_prefix(prods, OP_INDEX, features, tree_type)
    success, tree = _prefix_to_tree!(prefix_list)
    !is_tree_valid(tree) && return false, Node{T}()
    return success, tree
end

"""
Taken productions in the form (lhs, rhs) and convert prefix list of nodes with op index. lhs, rhs are strings (from GRAMMAR).

Needs mapping from (op_deg, op_idx) -> token_idx.
"""
function _prods_to_prefix(prods::Vector{Tuple{String, String}}, OP_INDEX::Dict{String, Int}, features::Vector{Int}, tree_type::Type{T})::Vector{Node{T}} where {T <: Number}
    prefix_list = []
    i = 1
    for prod in prods
        op_match = match(r"'([^']+)'", prod[2])  # Alternatively, use prod to infer arity?
        if op_match !== nothing
            op = op_match.captures[1]
            arity = OPERATOR_ARITY[op]
            if arity == 0
                push!(prefix_list, Node{T}(; feature=features[i]))
                i += 1
            elseif arity == 1
                push!(prefix_list, _make_childless_op(arity, OP_INDEX[op], T))
            elseif arity == 2
                push!(prefix_list, _make_childless_op(arity, OP_INDEX[op], T))
            end
        else  # Constant
            push!(prefix_list, Node{T}(; val=parse(T, prod[2])))
        end
    end
    return prefix_list
end


"""
Copied from datagen/ExpressionGenerator.jl. Consolidate.
"""
function _make_childless_op(degree::Int, op_index::Int, ::Type{T})::Node{T} where {T<:Number}
    op_node = Node{T}()
    op_node.degree = degree
    op_node.op = op_index
    return op_node
end

function _build_subtree(prefix_list::Vector{Node{T}})::Tuple{Bool, Node{T}} where {T<:Number}
    isempty(prefix_list) && return false, Node{T}()  # If prefix_list is empty initially or children are created but prefix_list stack is empty.

    node = popfirst!(prefix_list)
    success = true
    if node.degree == 0
        # Leaf node, no children to add
    elseif node.degree == 1
        success, node.l = _build_subtree(prefix_list)
    elseif node.degree == 2
        success1, node.l = _build_subtree(prefix_list)
        success2, node.r = _build_subtree(prefix_list)
        success = success1 && success2
    else
        error("Prefix_to_tree: Invalid node degree: $(node.degree)")
    end
    return success, node
end

"""
Copied from datagen/ExpressionGenerator.jl. Consolidate.

First flag is ``success``, second flag is ``node``.
"""
function _prefix_to_tree!(prefix_list::Vector{Node{T}})::Tuple{Bool, Node{T}} where {T<:Number}
    return _build_subtree(prefix_list)
end




"""
    select_viable_subtree(tree::Node{T}, options::AbstractOptions) where T<:Number

Analyzes a tree to find valid subtrees for mutation. A valid subtree must:
- Have between min_nodes and max_nodes total nodes
- Contain only a single feature variable

Returns a tuple containing:
- success: Boolean indicating if a valid subtree was found
- node: The selected subtree node
- parent: The parent node of the selected subtree (or nothing if root)
- feature: The feature number used in the subtree

The function works by:
1. Computing descendant counts and feature sets for each node
2. Finding all valid subtrees that meet the criteria
3. Randomly selecting one valid subtree

Returns:
- success: Boolean indicating if a valid subtree was found
- node: The selected subtree node
- parent: The parent node of the selected subtree (or nothing if root)
- feature: The feature number used in the subtree
"""
function select_viable_subtree(
    tree::Node{T}, 
    options::AbstractOptions
)::Tuple{Bool, Union{Node{T}, Nothing}, Union{Node{T}, Nothing}, Union{Set{Int}, Nothing}} where T<:Number
    # Store descendant count and feature set for each node
    desc_counts = Dict{Node{T}, Int}() 
    feature_sets = Dict{Node{T}, Set{Int}}()
    
    # Helper function to compute descendants recursively
    function count_descendants(node::Node{T})::Tuple{Int,Set{Int}}
        if haskey(desc_counts, node)
            return desc_counts[node], feature_sets[node]
        end
        
        features = Set{Int}()
        if node.degree == 0
            if !node.constant
                push!(features, node.feature)
            end
            desc_counts[node] = 1
            feature_sets[node] = features
            return 1, features
        end
        
        left_count, left_features = count_descendants(node.l)
        right_count = 0
        right_features = Set{Int}()
        if node.degree == 2
            right_count, right_features = count_descendants(node.r)
        end
        
        total = 1 + left_count + right_count
        union!(features, left_features, right_features)
        
        desc_counts[node] = total
        feature_sets[node] = features
        return total, features
    end
    
    # Compute descendants for whole tree
    count_descendants(tree)
    
    # Find valid subtrees
    valid_subtrees = []
    function check_node(node::Node{T}, parent::Union{Node{T},Nothing})
        count = desc_counts[node]
        features = feature_sets[node]
        
        if options.neural_options.subtree_min_nodes <= count <= options.neural_options.subtree_max_nodes &&
            length(features) <= options.neural_options.subtree_max_features
            push!(valid_subtrees, (node, parent, features))
        end
        
        if node.degree >= 1
            check_node(node.l, node)
            if node.degree == 2
                check_node(node.r, node) 
            end
        end
    end
    
    check_node(tree, nothing)
    
    isempty(valid_subtrees) && return (false, tree, nothing, Set{Int}())
    
    # Randomly select one valid subtree
    selected = rand(valid_subtrees)
    return (true, selected[1], selected[2], selected[3])
end

function count_nodes(tree::Node)::Int
    if tree.degree == 0
        return 1
    elseif tree.degree == 1
        return 1 + count_nodes(tree.l)
    elseif tree.degree == 2
        return 1 + count_nodes(tree.l) + count_nodes(tree.r)
    else
        error("Invalid node degree: $(tree.degree)")
    end
end

end