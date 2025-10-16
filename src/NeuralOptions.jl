module NeuralOptionsModule

Base.@kwdef mutable struct NeuralOptions
    active::Bool = false
    sampling_eps::Float64 = 0.01
    subtree_min_nodes::Int = 5
    subtree_max_nodes::Int = 14
    model_path::String = ""
    device::String = "cpu"  # TODO: Make this symbol? But needs to work with Python interface.

    max_resamples::Int = 10  # Set (1+max_resamples) to a multiple of sample_batchsize
    max_tree_size_diff::Int = 1
    require_tree_size_similarity::Bool = true
    require_novel_skeleton::Bool = true
    require_expr_similarity::Bool = true
    similarity_threshold::Float64 = 0.2
    sample_batchsize::Int = 10  #
    sample_logits::Bool = true  # If false, use argmax to sample.
    subtree_max_features::Int = 1

    # Expression similarity evaluation settings
    eval_min::Float64 = -10.0
    eval_max::Float64 = 10.0
    eval_npoints::Int = 40
    eval_transform::String = "asinh"  # "asinh" or "identity"

    # Loggings
    verbose::Bool = false
    log_subtree_strings::Bool = false
end

function validate_neural_options(options::NeuralOptions)
    if !options.active
        return
    end
    if options.model_path == ""
        throw(ArgumentError("Model path is required"))
    end
end

end