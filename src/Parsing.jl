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

"""
    GrammarRule

Per-rule precomputed metadata used by `logits_to_prods` / `_prods_to_prefix`.
Built once from `grammar_str` at module load — eliminates the per-call
`split`/`replace`/`match` work that previously ran on every iteration of the
decode loop (which is called ~13× per `logits_to_prods` × thousands of calls
per neural mutation run).

Fields:
- `lhs`        : LHS symbol, e.g. "S" or "END"
- `lhs_idx`    : index of `lhs` in `unique_lhs` (1-based)
- `rhs`        : raw RHS string, kept for backward-compatible `(lhs, rhs)`
                  output — for CON rules this slot is overwritten per-call
                  with the decoded constant.
- `op_name`    : operator name inside quotes (e.g. "ADD", "SIN", "x1"),
                  or "" for the END rule.
- `arity`      : 0/1/2 for op rules, -1 for CON, -2 for END (used as a fast
                  switch in `_prods_to_prefix_indexed`).
- `is_end`     : true for the "END -> 'END'" rule.
- `is_con`     : true for the "S -> 'CON'" rule.
- `is_var`     : true for the variable rule (e.g. "S -> 'x1'").
- `n_nonterm`  : number of nonterminals in RHS to push onto the parse stack.
- `nonterm_lhs_idx`: vector of `unique_lhs` indices to push (in *forward* order
                  — the consumer pushes in reverse so we don't reverse here).
"""
struct GrammarRule
    lhs::String
    lhs_idx::Int
    rhs::String
    op_name::String
    arity::Int
    is_end::Bool
    is_con::Bool
    is_var::Bool
    n_nonterm::Int
    nonterm_lhs_idx::Vector{Int}
end

const _OP_QUOTE_RE = r"'([^']+)'"

function _build_grammar_rules(g_str::String, ulhs::Vector{String})::Vector{GrammarRule}
    out = GrammarRule[]
    lhs_to_idx = Dict{String,Int}(s => i for (i, s) in enumerate(ulhs))
    for line in split(g_str, "\n")
        sline = strip(line)
        isempty(sline) && continue
        lhs_raw, rhs_raw = split(sline, "->")
        lhs = String(strip(lhs_raw))
        rhs = String(strip(rhs_raw))
        m = match(_OP_QUOTE_RE, rhs)
        op_name = m === nothing ? "" : String(m.captures[1])
        is_end = (lhs == "END")
        is_con = (op_name == "CON")
        # Variable rule has no arity entry in OPERATOR_ARITY for "x1" with a
        # non-zero arity, but we tag it explicitly via OPERATOR_ARITY[op]==0
        # (the convention used throughout this file).
        arity = if is_end
            -2
        elseif is_con
            -1
        elseif haskey(OPERATOR_ARITY, op_name)
            OPERATOR_ARITY[op_name]
        else
            error("Unknown op in grammar rule: $rhs")
        end
        is_var = (arity == 0 && !is_con && !is_end)
        # Walk RHS tokens after the quoted op, picking up nonterminals to push.
        # E.g. for "'ADD' S S" we get ["S","S"]; for "'CON'" we get [].
        nonterm = Int[]
        for tok in split(rhs)
            clean = replace(tok, "'" => "")
            if haskey(lhs_to_idx, clean)
                push!(nonterm, lhs_to_idx[clean])
            end
        end
        push!(out, GrammarRule(lhs, lhs_to_idx[lhs], rhs, op_name, arity,
                               is_end, is_con, is_var, length(nonterm), nonterm))
    end
    return out
end

const GRAMMAR_RULES = _build_grammar_rules(grammar_str, unique_lhs)
const _LHS_TO_IDX = Dict{String,Int}(s => i for (i, s) in enumerate(unique_lhs))

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

Uses the precomputed `GRAMMAR_RULES` table to look up parsed rule metadata
(LHS, RHS, nonterminals to push) — eliminating the per-call grammar string
splits that previously dominated this function.
"""
function logits_to_prods(
    logits::AbstractMatrix{Float32},
    sample::Bool=false,
    max_length::Int=15
)::Tuple{Bool, Union{Vector{Tuple{String, String}}, Nothing}}
    # Stack holds `unique_lhs` indices, not strings — the only LHS symbols that
    # ever land here are members of `unique_lhs` (we filter them at table-build
    # time), so this is exact and avoids string compares.
    n_lhs = length(unique_lhs)
    n_rules = length(GRAMMAR_RULES)
    n_cats = size(grammar_masks, 2)
    @assert n_rules == n_cats

    # `S` is index 1, but be explicit:
    stack = Int[_LHS_TO_IDX["S"]]
    sizehint!(stack, max_length)

    prods = Tuple{String,String}[]
    sizehint!(prods, max_length)

    # Reused per-step probability buffer.
    probs = Vector{Float32}(undef, n_rules)

    t = 1
    while !isempty(stack)
        alpha_idx = pop!(stack)

        # Compute mask .* exp(clamp(logits[t, :], -84, 84)) into `probs`.
        @inbounds for j in 1:n_rules
            if grammar_masks[alpha_idx, j]
                lj = logits[t, j]
                if lj < -84.0f0
                    lj = -84.0f0
                elseif lj > 84.0f0
                    lj = 84.0f0
                end
                probs[j] = exp(lj)
            else
                probs[j] = 0.0f0
            end
        end

        tot = zero(Float32)
        @inbounds for j in 1:n_rules
            tot += probs[j]
        end
        if tot == 0 || !isfinite(tot)
            return (false, nothing)
        end
        inv_tot = one(Float32) / tot
        @inbounds for j in 1:n_rules
            probs[j] *= inv_tot
        end

        # Optional renormalization sanity check (matches the Categorical()
        # tolerance in the previous implementation).
        s = zero(Float32)
        @inbounds for j in 1:n_rules
            s += probs[j]
        end
        if !isapprox(s, one(Float32))
            return (false, nothing)
        end

        # Select production rule (1-based index into GRAMMAR_RULES).
        i = if sample
            rand(Categorical(probs))
        else
            argmax_idx = 1
            best = probs[1]
            @inbounds for j in 2:n_rules
                if probs[j] > best
                    best = probs[j]
                    argmax_idx = j
                end
            end
            argmax_idx
        end

        rule = @inbounds GRAMMAR_RULES[i]
        rule.is_end && break

        # Build the (lhs, rhs) tuple. For CON rules, materialize the literal.
        rhs_str = rule.is_con ? string(@inbounds logits[t, end]) : rule.rhs
        push!(prods, (rule.lhs, rhs_str))

        # Push nonterminals in reverse order (matches the prior `for ... in
        # reverse(rhs_symbols)` loop).
        @inbounds for k in length(rule.nonterm_lhs_idx):-1:1
            push!(stack, rule.nonterm_lhs_idx[k])
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
# Extract `op_name` from an RHS like `"'ADD' S S"` or `"'CON'"`. Returns `nothing`
# when the RHS has no leading single-quoted token (i.e. the constant slot, where
# the rhs string is the literal numeric value, e.g. "0.123"). Faster than
# `match(r"'([^']+)'", s)` because it skips the Regex compile/dispatch overhead
# on the hot decode path.
@inline function _rhs_op_name(rhs::AbstractString)::Union{String, Nothing}
    n = ncodeunits(rhs)
    n >= 2 || return nothing
    @inbounds first(rhs) == '\'' || return nothing
    j = findnext('\'', rhs, 2)
    j === nothing && return nothing
    return String(SubString(rhs, 2, prevind(rhs, j)))
end

function _prods_to_prefix(prods::Vector{Tuple{String, String}}, OP_INDEX::Dict{String, Int}, features::Vector{Int}, tree_type::Type{T})::Vector{Node{T}} where {T <: Number}
    prefix_list = Node{T}[]
    sizehint!(prefix_list, length(prods))
    i = 1
    for prod in prods
        op = _rhs_op_name(prod[2])
        if op !== nothing
            arity = OPERATOR_ARITY[op]
            if arity == 0
                push!(prefix_list, Node{T}(; feature=features[i]))
                i += 1
            else
                push!(prefix_list, _make_childless_op(arity, OP_INDEX[op], T))
            end
        else  # Constant — rhs is the literal numeric string
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