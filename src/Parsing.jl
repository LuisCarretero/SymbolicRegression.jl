module ParsingModule

import SymbolicRegression: Node
using Distributions: Categorical

# Define grammar rules similar to Python version
grammar_str = """
S -> 'ADD' S S | 'SUB' S S | 'MUL' S S | 'DIV' S S
S -> 'SIN' S | 'COS' S | 'EXP' S | 'TANH' S | 'COSH' S | 'SINH' S 
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

    # FIXME: Handle this differently!
    "x1" => 0
)

# Split each production rule into separate lines
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
grammar_str = join(grammar_lines, "\n")

function _create_grammar_masks(grammar_str::String)
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
masks, allowed_prod_idx, unique_lhs = _create_grammar_masks(grammar_str)

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


function node_to_onehot(node::Node{T}, cfg::nn_config)::Tuple{Bool, Matrix{Float32}} where T <: Number
    prefix = _tree_to_prefix(node)
    idx = [_node_to_token_idx(node, cfg) for node in prefix]
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

function _node_to_token_idx(node::Node{T}, cfg::nn_config)::Tuple{Int, Float64} where T <: Number
    offset_unaop = cfg.nbin
    offset_const = offset_unaop + cfg.nuna
    offset_var = offset_const + cfg.nvar
    if node.degree == 2
        return (node.op, 0)
    elseif node.degree == 1
        return (offset_unaop + node.op, 0)
    elseif node.degree == 0
        if node.constant
            return (offset_const + 1, node.val)  # Only 1 const token
        else
            # For multiple variables, do + node.feature instead. 
            return (offset_var + 1, 0)
        end
    end
end

"""
Convert logits to production rules.
First flag is ``success``, second flag is ``prods``.
"""
function logits_to_prods(logits::Matrix{Float32}, sample::Bool=false, max_length::Int=15)::Tuple{Bool, Union{Vector{Tuple{String, String}}, Nothing}}
    # Initialize empty stack with start symbol 'S'
    stack = ["S"]
    
    # Split logits into productions and constants
    logits_prods = logits[:, 1:end-1] 
    constants = logits[:, end]
    
    prods = []
    t = 1
    
    while !isempty(stack)
        alpha = pop!(stack)  # Current LHS toke
        
        # Get mask for current symbol
        symbol_idx = findfirst(==(alpha), unique_lhs)
        mask = masks[symbol_idx, :]
        
        # Calculate probabilities
        probs = mask .* exp.(logits_prods[t, :])
        tot = sum(probs)
        tot == 0 && return (false, nothing)  # No valid productions found
        probs = probs ./ tot
        any(isnan.(probs)) && return (false, nothing)
        
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

function prods_to_tree(prods::Vector{Tuple{String, String}}, OP_INDEX::Dict{String, Int}, feature::Int)
    # global prods stack
    # Create node for each production
    # Depending on arity, create 0, 1 or 2 children
    # 
    prefix_list = _prods_to_prefix(prods, OP_INDEX, feature)
    tree = _prefix_to_tree!(prefix_list)

    return tree
end

function _prods_to_prefix(prods::Vector{Tuple{String, String}}, OP_INDEX::Dict{String, Int}, feature::Int)::Vector{Node{Float64}}
    prefix_list = []
    for prod in prods
        op_match = match(r"'([^']+)'", prod[2])  # Alternatively, use prod to infer arity?
        if op_match !== nothing
            op = op_match.captures[1]
            arity = OPERATOR_ARITY[op]
            if arity == 0
                push!(prefix_list, Node{Float64}(; feature=feature))  # FIXME: Only univariate for now
            elseif arity == 1
                push!(prefix_list, _make_childless_op(arity, OP_INDEX[op], Float64))
            elseif arity == 2
                push!(prefix_list, _make_childless_op(arity, OP_INDEX[op], Float64))
            end
        else  # Constant
            push!(prefix_list, Node{Float64}(; val=parse(Float64, prod[2])))
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

function build_subtree(prefix_list::Vector{Node{T}})::Tuple{Bool, Union{Node{T}}} where {T<:Number}
    isempty(prefix_list) && return false, Node{T}()

    node = popfirst!(prefix_list)
    success = true
    if node.degree == 0
        # Leaf node, no children to add
    elseif node.degree == 1
        success, node.l = build_subtree(prefix_list)
    elseif node.degree == 2
        success1, node.l = build_subtree(prefix_list)
        success2, node.r = build_subtree(prefix_list)
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
function _prefix_to_tree!(prefix_list::Vector{Node{T}})::Tuple{Bool, Union{Node{T}, Nothing}} where {T<:Number}
    return build_subtree(prefix_list)
end




"""
    select_viable_subtree(tree::Node{T}, min_nodes::Int, max_nodes::Int) where T<:Number

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
"""
function select_viable_subtree(tree::Node{T}, min_nodes::Int, max_nodes::Int) where T<:Number
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
        
        if min_nodes <= count <= max_nodes && length(features) == 1
            push!(valid_subtrees, (node, parent, first(features)))
        end
        
        if node.degree >= 1
            check_node(node.l, node)
            if node.degree == 2
                check_node(node.r, node) 
            end
        end
    end
    
    check_node(tree, nothing)
    
    isempty(valid_subtrees) && return (false, tree, nothing, 0)
    
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