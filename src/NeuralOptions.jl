module NeuralOptionsModule

Base.@kwdef mutable struct NeuralOptions
    active::Bool = false
    sampling_eps::Float64 = 0.01
    subtree_min_nodes::Int = 5
    subtree_max_nodes::Int = 14
    model_path::String = ""
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