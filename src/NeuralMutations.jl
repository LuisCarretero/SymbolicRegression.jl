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

export neural_mutate_tree,
    NeuralStageTimes,
    enable_stage_timing!,
    reset_stage_times!,
    get_stage_times

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

# Per-stage wall-clock accumulators for profiling. Toggled via env var so
# regular runs are unaffected. See `enable_stage_timing!`.
mutable struct NeuralStageTimes
    select_subtree_ns::Int
    encode_ns::Int
    onnx_inference_ns::Int
    novelty_check_ns::Int
    decode_logits_ns::Int
    build_tree_ns::Int
    size_check_ns::Int
    similarity_eval_ns::Int
    remap_features_ns::Int
    replace_subtree_ns::Int
    other_ns::Int

    select_subtree_calls::Int
    encode_calls::Int
    onnx_inference_calls::Int
    novelty_check_calls::Int
    decode_logits_calls::Int
    build_tree_calls::Int
    size_check_calls::Int
    similarity_eval_calls::Int
    remap_features_calls::Int
    replace_subtree_calls::Int
    neural_mutate_calls::Int

    NeuralStageTimes() = new(
        0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,
    )
end

const STAGE_TIMES_REF = Ref{NeuralStageTimes}(NeuralStageTimes())
const STAGE_TIMING_ENABLED = Ref{Bool}(false)
const STAGE_TIMING_LOCK = ReentrantLock()

enable_stage_timing!(b::Bool=true) = (STAGE_TIMING_ENABLED[] = b)
reset_stage_times!() = (STAGE_TIMES_REF[] = NeuralStageTimes())
get_stage_times() = STAGE_TIMES_REF[]

@inline function _add_stage_ns!(field::Symbol, ns::Int)
    STAGE_TIMING_ENABLED[] || return nothing
    lock(STAGE_TIMING_LOCK) do
        st = STAGE_TIMES_REF[]
        setfield!(st, field, getfield(st, field) + ns)
    end
    return nothing
end

@inline function _bump_stage_call!(field::Symbol)
    STAGE_TIMING_ENABLED[] || return nothing
    lock(STAGE_TIMING_LOCK) do
        st = STAGE_TIMES_REF[]
        setfield!(st, field, getfield(st, field) + 1)
    end
    return nothing
end

# Macro: time a block, attribute to ns_field, increment count_field.
# Always returns the block's value.
macro time_stage(ns_field, count_field, expr)
    quote
        if STAGE_TIMING_ENABLED[]
            local _t0 = time_ns()
            local _v = $(esc(expr))
            _add_stage_ns!($(QuoteNode(ns_field)), Int(time_ns() - _t0))
            _bump_stage_call!($(QuoteNode(count_field)))
            _v
        else
            $(esc(expr))
        end
    end
end

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
    device_str = options.neural_options.device
    if device_str == "cuda" || device_str == "tensorrt"
        if CUDA_IMPORTED[] && CUDA.functional()
            @info "CUDA available. Loading model on GPU ($(device_str))."
        else
            @warn "CUDA package not available. Falling back to CPU."
            options.neural_options.device = "cpu"
            device_str = "cpu"
        end
    end
    # `intra_op_num_threads`/`inter_op_num_threads` left at ORT defaults (0 = library
    # picks). T2.1 attempt (job 82414) showed pinning to 1 hurts per-call latency
    # ~+8 % even though it removes the 41-line affinity warning storm — apparently
    # ORT uses the intra-op pool for output staging / CPU-fallback ops introduced
    # by the Memcpy nodes. Re-evaluate once T1.1 removes those CPU-fallback nodes.
    #
    # TensorRT EP: opt in via `device="tensorrt"`. The first session-load
    # JIT-compiles a TRT engine (slow); we cache it in a per-model
    # directory next to the .onnx file so subsequent loads are instant.
    # Pick up an optional SR_TRT_FP16=1 env var to flip TRT to fp16
    # internally — no re-export needed (TRT casts the fp32 weights at
    # engine-build time).
    trt_cache_path = if device_str == "tensorrt"
        cache_dir = joinpath(dirname(options.neural_options.model_path), "trt_cache")
        mkpath(cache_dir)
        cache_dir
    else
        nothing
    end
    trt_fp16 = device_str == "tensorrt" && get(ENV, "SR_TRT_FP16", "0") == "1"
    MODEL_REF[] = load_inference_perf(
        options.neural_options.model_path;
        execution_provider=Symbol(device_str),
        intra_op_num_threads=0,
        inter_op_num_threads=0,
        graph_optimization_level=99,  # ORT_ENABLE_ALL
        trt_fp16=trt_fp16,
        trt_engine_cache_path=trt_cache_path,
    )
    _init_model_input_contract!(MODEL_REF[])
    if !_MODEL_HAS_SAMPLE_COUNT[]
        @info "ONNX model uses baked sample_count contract (T1.1): sample_count not passed per call; sample_eps is Float32."
    end
    if device_str == "tensorrt"
        @info "TensorRT EP enabled (fp16=$(trt_fp16); engine_cache=$(trt_cache_path)). First call will build the engine."
    end
    _build_fast_inference_state!(MODEL_REF[])
end

# ORT graph optimization levels (from onnxruntime_c_api.h GraphOptimizationLevel)
const ORT_DISABLE_ALL = Cint(0)
const ORT_ENABLE_BASIC = Cint(1)
const ORT_ENABLE_EXTENDED = Cint(2)
const ORT_ENABLE_ALL = Cint(99)

"""
    _ort_check_and_release(api, status)

Wrapper around ORT status pointer: throw if non-NULL, otherwise release.
Mirrors `ONNXRunTime.CAPI.check_and_release` (not exported).
"""
@inline function _ort_check_and_release(api::ORT.CAPI.OrtApi, status::Ptr{Cvoid})
    if status != C_NULL
        msg = unsafe_string(@ccall $(api.GetErrorMessage)(status::Ptr{Cvoid})::Cstring)
        @ccall $(api.ReleaseStatus)(status::Ptr{Cvoid})::Cvoid
        throw(ORT.CAPI.OrtException(msg))
    end
    return nothing
end

@inline function _set_intra_op_num_threads!(api::ORT.CAPI.OrtApi, opts::ORT.CAPI.OrtSessionOptions, n::Integer)
    status = @ccall $(api.SetIntraOpNumThreads)(opts.ptr::Ptr{Cvoid}, Cint(n)::Cint)::Ptr{Cvoid}
    _ort_check_and_release(api, status)
end

@inline function _set_inter_op_num_threads!(api::ORT.CAPI.OrtApi, opts::ORT.CAPI.OrtSessionOptions, n::Integer)
    status = @ccall $(api.SetInterOpNumThreads)(opts.ptr::Ptr{Cvoid}, Cint(n)::Cint)::Ptr{Cvoid}
    _ort_check_and_release(api, status)
end

@inline function _set_graph_optimization_level!(api::ORT.CAPI.OrtApi, opts::ORT.CAPI.OrtSessionOptions, level::Integer)
    status = @ccall $(api.SetSessionGraphOptimizationLevel)(opts.ptr::Ptr{Cvoid}, Cint(level)::Cint)::Ptr{Cvoid}
    _ort_check_and_release(api, status)
end

# OrtTensorRTProviderOptions (V1) — mirrors onnxruntime_c_api.h. Field order
# and types match the C struct exactly; passing the Julia struct by-Ref to
# the ccall hands ORT a pointer to compatible memory. Pointer fields
# (calibration table name, cache path, decryption lib path) are C_NULL when
# unused.
struct _OrtTensorRTProviderOptionsV1
    device_id::Cint
    has_user_compute_stream::Cint
    user_compute_stream::Ptr{Cvoid}
    trt_max_partition_iterations::Cint
    trt_min_subgraph_size::Cint
    trt_max_workspace_size::Csize_t
    trt_fp16_enable::Cint
    trt_int8_enable::Cint
    trt_int8_calibration_table_name::Ptr{UInt8}
    trt_int8_use_native_calibration_table::Cint
    trt_dla_enable::Cint
    trt_dla_core::Cint
    trt_dump_subgraphs::Cint
    trt_engine_cache_enable::Cint
    trt_engine_cache_path::Ptr{UInt8}
    trt_engine_decryption_enable::Cint
    trt_engine_decryption_lib_path::Ptr{UInt8}
    trt_force_sequential_engine_build::Cint
end

# `gchandle` keeps Julia-owned strings alive as long as the caller wants
# the options to be valid (ORT copies the path internally during
# SessionOptionsAppendExecutionProvider_TensorRT, so in practice the
# gchandle only needs to outlive that single ccall).
function _make_trt_options(;
    device_id::Int=0,
    fp16::Bool=false,
    max_workspace_size::Int=1 << 30,           # 1 GiB
    engine_cache_enable::Bool=true,
    engine_cache_path::Union{Nothing, String}=nothing,
)::Tuple{_OrtTensorRTProviderOptionsV1, Vector{Any}}
    gchandle = Any[]
    cache_ptr = if engine_cache_path === nothing
        Ptr{UInt8}(C_NULL)
    else
        push!(gchandle, engine_cache_path)
        Ptr{UInt8}(Base.unsafe_convert(Cstring, engine_cache_path))
    end
    opts = _OrtTensorRTProviderOptionsV1(
        Cint(device_id),
        Cint(0), C_NULL,
        Cint(1000),                                    # max_partition_iterations (ORT default)
        Cint(1),                                       # min_subgraph_size (ORT default)
        Csize_t(max_workspace_size),
        Cint(fp16 ? 1 : 0), Cint(0),                   # fp16, int8
        Ptr{UInt8}(C_NULL), Cint(0),                   # int8 calibration
        Cint(0), Cint(0), Cint(0),                     # DLA, dump_subgraphs
        Cint(engine_cache_enable ? 1 : 0), cache_ptr,
        Cint(0), Ptr{UInt8}(C_NULL),                   # decryption
        Cint(0),                                       # force_sequential_engine_build
    )
    return opts, gchandle
end

@inline function _append_tensorrt_provider!(
    api::ORT.CAPI.OrtApi,
    opts::ORT.CAPI.OrtSessionOptions,
    trt_opts::_OrtTensorRTProviderOptionsV1,
)
    trt_ref = Ref(trt_opts)
    status = GC.@preserve trt_ref begin
        @ccall $(api.SessionOptionsAppendExecutionProvider_TensorRT)(
            opts.ptr::Ptr{Cvoid},
            trt_ref::Ptr{Cvoid},
        )::Ptr{Cvoid}
    end
    _ort_check_and_release(api, status)
end

"""
    load_inference_perf(path; ...)

Drop-in replacement for `ORT.load_inference` that sets a few performance-relevant
session options before `CreateSession`:

- `intra_op_num_threads`: cap intra-op thread pool (0 = ORT default, the value we
  ship with — pinning to 1 was tried in job 82414 and was inconclusive within
  noise; revisit once T1.1 removes the CPU-fallback Memcpy nodes that the pool
  serves).
- `inter_op_num_threads`: cap inter-op thread pool (0 = ORT default).
- `graph_optimization_level`: 0/1/2/99 (default 99 = `ORT_ENABLE_ALL`).

ONNXRunTime.jl's high-level `load_inference` does not expose these (T2.1 / T2.3
in `src/perf_experiments/IDEAS.md`), so we duplicate the body here, calling the
already-loaded function pointers in `OrtApi` directly via ccall.
"""
function load_inference_perf(
    path::AbstractString;
    execution_provider::Symbol=:cpu,
    envname::AbstractString="defaultenv",
    intra_op_num_threads::Int=0,
    inter_op_num_threads::Int=0,
    graph_optimization_level::Int=Int(ORT_ENABLE_ALL),
    trt_fp16::Bool=false,
    trt_engine_cache_path::Union{Nothing, String}=nothing,
)::ORT.InferenceSession
    # The :tensorrt EP lives in the same `libonnxruntime.so` as :cuda
    # (the gpu artifact); :tensorrt is just a different ORT-side EP-append
    # call. Use :cuda for the artifact-resolution path so libpath finds
    # the gpu lib + libonnxruntime_providers_tensorrt.so (which itself
    # dlopens libnvinfer.so.10 from LD_LIBRARY_PATH — pip-installed
    # tensorrt-cu12 wheel ships it under
    # .pixi/envs/cam/lib/python3.12/site-packages/tensorrt_libs/).
    artifact_provider = execution_provider === :tensorrt ? :cuda : execution_provider
    api = ORT.CAPI.GetApi(; execution_provider=artifact_provider)
    env = ORT.CAPI.CreateEnv(api; name=envname, logging_level=ORT.CAPI.ORT_LOGGING_LEVEL_WARNING)
    session_options = ORT.CAPI.CreateSessionOptions(api)

    # 0 means "leave at ORT default". Only call the setters with positive values.
    intra_op_num_threads > 0 && _set_intra_op_num_threads!(api, session_options, intra_op_num_threads)
    inter_op_num_threads > 0 && _set_inter_op_num_threads!(api, session_options, inter_op_num_threads)
    _set_graph_optimization_level!(api, session_options, graph_optimization_level)

    if execution_provider === :cuda
        cuda_options = ORT.CAPI.OrtCUDAProviderOptions()
        ORT.CAPI.SessionOptionsAppendExecutionProvider_CUDA(api, session_options, cuda_options)
    elseif execution_provider === :tensorrt
        # TRT EP first; CUDA EP after as fallback for any subgraph TRT
        # can't take. Without the CUDA fallback, unsupported ops hit the
        # CPU EP and introduce host↔device copies similar to T1.1's old
        # Memcpy nodes. Engine cache: TRT JIT-compiles a TensorRT engine
        # on first session load (slow — seconds to minutes); persisting
        # it lets subsequent sessions reuse the engine instantly.
        trt_opts, _gchandle = _make_trt_options(
            fp16=trt_fp16,
            engine_cache_path=trt_engine_cache_path,
        )
        _append_tensorrt_provider!(api, session_options, trt_opts)
        cuda_options = ORT.CAPI.OrtCUDAProviderOptions()
        ORT.CAPI.SessionOptionsAppendExecutionProvider_CUDA(api, session_options, cuda_options)
    elseif execution_provider !== :cpu
        error("Unsupported execution_provider $execution_provider")
    end

    session = ORT.CAPI.CreateSession(api, env, path, session_options)
    meminfo = ORT.CAPI.CreateCpuMemoryInfo(api)
    allocator = ORT.CAPI.CreateAllocator(api, session, meminfo)

    # Mirror ONNXRunTime.input_names/output_names (private helpers).
    n_in = ORT.CAPI.SessionGetInputCount(api, session)
    n_out = ORT.CAPI.SessionGetOutputCount(api, session)
    in_names = String[ORT.CAPI.SessionGetInputName(api, session, Csize_t(i), allocator) for i in 0:n_in-1]
    out_names = String[ORT.CAPI.SessionGetOutputName(api, session, Csize_t(i), allocator) for i in 0:n_out-1]
    @assert allunique(in_names)
    @assert allunique(out_names)

    # Record the InferenceSession as :cuda so any high-level
    # ONNXRunTime.jl path (we don't currently use it, but the session
    # struct's `execution_provider` is checked by
    # `(::InferenceSession)(inputs)`) accepts it. Our fast-path bypasses
    # that check anyway.
    sess_provider = execution_provider === :tensorrt ? :cuda : execution_provider
    return ORT.InferenceSession(api, sess_provider, session, meminfo, allocator, in_names, out_names)
end

"""
    sample_logits(x::AbstractArray{Float32}, eps::Float64=0.01, sample_count::Int=1)::AbstractArray{Float32}

Sample the logits of the neural network.
Currently assuming single sample as input and then sample_count samples as output.

Two ONNX graph contracts are supported:

  * legacy 3-input: `input_syntax`, `sample_eps` (Float64 [1]), `sample_count` (Int64 [1])
  * baked 2-input: `input_syntax`, `sample_eps` (Float32 [1]) — sample_count is
    constant-folded into the graph (T1.1, removes CPU-fallback Memcpy nodes).
    The caller's `sample_count` is then implicit; mismatch is an error.

The variant is detected once at session-load time from `MODEL_REF[].input_names`
(see `_init_model_input_contract!`), so the per-call cost is just a single Bool
check — no string lookup.
"""
const _MODEL_HAS_SAMPLE_COUNT = Ref{Bool}(true)

# Fast-path inference state. Built once at session-load to avoid the per-call
# `Dict` / `prepare_inputs` / `make_output` work in ONNXRunTime.jl's high-level
# wrapper:
#
#   * `input_syntax_buf` and `sample_eps_buf[/_count_buf]` are Julia-owned
#     buffers; we mutate their contents in place per call. The pre-built
#     `OrtValue`s in `input_ortvalues` were created from these vectors via
#     `CreateTensorWithDataAsOrtValue`, which uses the buffer pointer
#     directly — so each `Run` reads whatever we wrote last.
#   * `output_ortvalue_ref` holds the most recent output `OrtValue` alive
#     across calls. That lets us return a zero-copy `unsafe_GetTensorMutableData`
#     view instead of the high-level path's `copy(parent(...))` (which was a
#     fresh ~92 KB host allocation + memcpy on every call). The previous
#     output is dropped — and ORT-released — when the *next* call replaces it.
#     Safe because `sample_routine` consumes all `@view` slices into batch i
#     before triggering the next inference call.
#
# All paths bypass `(::InferenceSession)(inputs)` and call `api.Run` via ccall.
mutable struct _FastInferenceState
    api::ORT.CAPI.OrtApi
    session::ORT.CAPI.OrtSession
    input_syntax_buf::Vector{Float32}        # length 180 = 1*15*12 (row-major)
    sample_eps_f32_buf::Vector{Float32}      # length 1, baked contract
    sample_eps_f64_buf::Vector{Float64}      # length 1, legacy contract
    sample_count_buf::Vector{Int64}          # length 1, legacy contract only
    input_ortvalues::Vector{ORT.OrtValue}    # 2 (baked) or 3 (legacy)
    input_name_cstrings::Vector{Cstring}
    output_name_cstrings::Vector{Cstring}
    input_name_storage::Vector{String}       # keeps Cstrings alive
    output_name_storage::Vector{String}
    output_ortvalue_ref::Ref{Union{Nothing, ORT.OrtValue}}
end

const _FAST_STATE = Ref{Union{Nothing, _FastInferenceState}}(nothing)

function _init_model_input_contract!(sess::ORT.InferenceSession)
    _MODEL_HAS_SAMPLE_COUNT[] = "sample_count" in sess.input_names
    # `sample_eps` element type is implied by the contract (legacy=Float64,
    # baked=Float32); we don't introspect ONNX tensor element types because
    # the high-level ONNXRunTime.jl API doesn't expose them and the input
    # name alone is enough.
    return nothing
end

function _build_fast_inference_state!(sess::ORT.InferenceSession)
    api = sess.api
    meminfo = sess.meminfo

    # Persistent input buffers. `CreateTensorWithDataAsOrtValue` uses
    # `pointer(data)` as the backing storage, so as long as we don't reallocate
    # these vectors, ORT will see whatever we write to them on each `Run`.
    input_syntax_buf = zeros(Float32, 1 * 15 * 12)            # row-major (1,15,12)
    sample_eps_f32_buf = zeros(Float32, 1)
    sample_eps_f64_buf = zeros(Float64, 1)
    sample_count_buf = zeros(Int64, 1)

    # Build OrtValues once. Shape vectors here are in ORT (C-row-major) order
    # to match `vec(reversedims(arr))` from the high-level path: a Julia
    # array of size (1,15,12) becomes shape [1,15,12] for ORT, but its
    # contents must be laid out in row-major order — which the caller writes
    # into `input_syntax_buf` directly (see `sample_logits`).
    syntax_val = ORT.CAPI.CreateTensorWithDataAsOrtValue(api, meminfo, input_syntax_buf, (1, 15, 12))

    eps_val, count_val = if _MODEL_HAS_SAMPLE_COUNT[]
        ev = ORT.CAPI.CreateTensorWithDataAsOrtValue(api, meminfo, sample_eps_f64_buf, (1,))
        cv = ORT.CAPI.CreateTensorWithDataAsOrtValue(api, meminfo, sample_count_buf, (1,))
        ev, cv
    else
        ev = ORT.CAPI.CreateTensorWithDataAsOrtValue(api, meminfo, sample_eps_f32_buf, (1,))
        ev, nothing
    end

    input_ortvalues = count_val === nothing ? ORT.OrtValue[syntax_val, eps_val] :
                                              ORT.OrtValue[syntax_val, eps_val, count_val]

    input_name_storage = copy(sess.input_names)
    output_name_storage = copy(sess.output_names)
    input_name_cstrings = Cstring[Base.unsafe_convert(Cstring, s) for s in input_name_storage]
    output_name_cstrings = Cstring[Base.unsafe_convert(Cstring, s) for s in output_name_storage]

    _FAST_STATE[] = _FastInferenceState(
        api, sess.session,
        input_syntax_buf, sample_eps_f32_buf, sample_eps_f64_buf, sample_count_buf,
        input_ortvalues,
        input_name_cstrings, output_name_cstrings,
        input_name_storage, output_name_storage,
        Ref{Union{Nothing, ORT.OrtValue}}(nothing),
    )
    return nothing
end

# Direct ccall to api.Run that asks ORT to allocate the output OrtValue (we
# pass a NULL output pointer). Returns the new output OrtValue with a
# finalizer that releases it — which is what lets us drop the previous output
# by simply replacing the Ref.
function _run_with_persistent_inputs!(state::_FastInferenceState)::ORT.OrtValue
    api = state.api
    n_in = Csize_t(length(state.input_ortvalues))
    n_out = Csize_t(length(state.output_name_cstrings))
    inputs_ptrs = Ptr{Cvoid}[(v::ORT.OrtValue).ptr for v in state.input_ortvalues]
    outputs_ptrs = Ptr{Cvoid}[C_NULL for _ in 1:n_out]

    GC.@preserve state inputs_ptrs outputs_ptrs begin
        status = @ccall $(api.Run)(
            state.session.ptr::Ptr{Cvoid},
            C_NULL::Ptr{Cvoid},
            state.input_name_cstrings::Ptr{Cstring},
            inputs_ptrs::Ptr{Ptr{Cvoid}},
            n_in::Csize_t,
            state.output_name_cstrings::Ptr{Cstring},
            n_out::Csize_t,
            outputs_ptrs::Ptr{Ptr{Cvoid}},
        )::Ptr{Cvoid}
        _ort_check_and_release(api, status)
    end

    # We expect exactly one output ("output_logits"); construct an OrtValue
    # wrapper and attach a finalizer mirroring `Run` in capi.jl.
    out = ORT.OrtValue(outputs_ptrs[1], Any[])
    finalizer(out) do v
        ORT.CAPI.release(api, v)
    end
    return out
end

function sample_logits(x::AbstractArray{Float32}, eps::Float64=0.01, sample_count::Int=1)::AbstractArray{Float32}
    state = _FAST_STATE[]
    @assert state !== nothing "Fast inference state not initialised — call load_model first."

    # Write input_syntax into the persistent buffer in ORT row-major layout.
    # `vec(reversedims(reshape(x, (1, ...))))` is what the high-level path
    # used to produce; we replicate it in place. `permutedims!` writes into
    # the destination buffer without an intermediate alloc.
    x3d = reshape(x, (1, size(x)...))
    permutedims!(reshape(state.input_syntax_buf, (12, 15, 1)), x3d, (3, 2, 1))

    if _MODEL_HAS_SAMPLE_COUNT[]
        state.sample_eps_f64_buf[1] = eps
        state.sample_count_buf[1] = sample_count
    else
        state.sample_eps_f32_buf[1] = Float32(eps)
        # sample_count is baked into the graph; the caller's value is implicit.
    end

    # Drop the previous batch's OrtValue *before* triggering the next Run so
    # ORT can reuse its internal buffer pool. Safe: by contract, all `@view`
    # slices into the previous batch were consumed before this call (see
    # `sample_routine`).
    state.output_ortvalue_ref[] = nothing

    out = _run_with_persistent_inputs!(state)
    state.output_ortvalue_ref[] = out

    # Zero-copy view into the ORT-owned buffer. Same `PermutedDimsArray{...}`
    # wrapper shape as the high-level `make_output` path returned, so the
    # decode/novelty hot path doesn't change.
    return ORT.CAPI.unsafe_GetTensorMutableData(state.api, out)
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
    _bump_stage_call!(:neural_mutate_calls)
    if !ENABLED_REF[]
        increment_stats!(STATS_REF[], :module_not_enabled, true)
        return tree
    end

    # Select a viable subtree to mutate FIXME: Could also try different tree if this one isn't successfull
    found_subtree, subtree, parent, feature_set = @time_stage(
        select_subtree_ns, select_subtree_calls,
        select_viable_subtree(tree, options),
    )
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
    return @time_stage(
        replace_subtree_ns, replace_subtree_calls,
        replace_subtree(tree, parent, subtree, new_subtree),
    )
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
    subtree_x1 = @time_stage(
        remap_features_ns, remap_features_calls,
        remap_features(subtree, original_feature, 1),
    )

    # Encode the subtree into a one-hot vector (structure only, feature doesn't affect encoding)
    encode_success, x_in = @time_stage(
        encode_ns, encode_calls,
        node_to_onehot(subtree, CFG_REF[], OP_TO_LOGITS_REF[]),
    )
    if !encode_success
        increment_stats!(STATS_REF[], :encoding_failures, true)
        return false, subtree, Inf
    end

    # Evaluate the *original* subtree ONCE, outside the resampling loop, when
    # similarity evaluation is needed. The original tree doesn't change across
    # resamples, so per-attempt re-evaluation is pure waste. (Univariate path
    # only — multivariate uses a different X per `multivariate_decoding`
    # iteration and stays on the legacy `check_expr_similarity`.)
    orig_eval_X, orig_eval_res = nothing, nothing
    if options.neural_options.subtree_max_features == 1 && options.neural_options.require_expr_similarity
        ok_orig, X_orig, res_orig = @time_stage(
            similarity_eval_ns, similarity_eval_calls,
            _eval_orig_subtree(subtree_x1, options, 1),
        )
        if !ok_orig
            return false, subtree, Inf
        end
        orig_eval_X = X_orig
        orig_eval_res = res_orig
    end

    for attempt in 1:(options.neural_options.max_resamples+1)
        increment_stats!(STATS_REF[], :total_samples, true)

        # Grab new output  TODO: Make this an object or somehow outsource current_sample_idx
        if current_sample_idx >= options.neural_options.sample_batchsize  # Used last sample from batch: Resample.
            x_out_batch = @time_stage(
                onnx_inference_ns, onnx_inference_calls,
                sample_logits(x_in, options.neural_options.sampling_eps, options.neural_options.sample_batchsize),
            )
            current_sample_idx = 1
        else  # Still have samples left in batch: Use these.
            current_sample_idx += 1
        end
        # `view` (not a copy) — `logits_to_prods` and `check_novel_skeleton`
        # only read from this slice, never mutate it.
        x_out = @view x_out_batch[current_sample_idx, :, :]

        if options.neural_options.require_novel_skeleton
            novel = @time_stage(
                novelty_check_ns, novelty_check_calls,
                check_novel_skeleton(x_in, x_out, options),
            )
            if !novel
                increment_stats!(STATS_REF[], :skeleton_not_novel, true)
                continue
            end
        end

        success, prods = @time_stage(
            decode_logits_ns, decode_logits_calls,
            logits_to_prods(x_out, options.neural_options.sample_logits),
        )
        if !success
            increment_stats!(STATS_REF[], :decoding_failures, true)
            continue
        end

        # Sampling model is implicitly univariate and returns only x1 as features after decoding via `logits_to_prods`
        feature_cnt_new = count(p -> p[2] == "'x1'", prods)

        if options.neural_options.subtree_max_features == 1  # Univariate decoding
            # Always generate with feature 1 for evaluation
            success, new_subtree_x1 = @time_stage(
                build_tree_ns, build_tree_calls,
                prods_to_tree(prods, OP_INDEX_REF[], fill(1, feature_cnt_new), T),
            )

            if !success
                increment_stats!(STATS_REF[], :tree_build_failures, true)
                continue
            end

            if options.neural_options.require_tree_size_similarity
                good = @time_stage(
                    size_check_ns, size_check_calls,
                    verify_tree_size_similar(subtree, new_subtree_x1, options),
                )
                if !good
                    increment_stats!(STATS_REF[], :tree_comparison_failures, true)
                    continue
                end
            end

            if options.neural_options.require_expr_similarity
                # Compare both using x1 (matches EVAL_X_UNIVARIATE_REF dimensions).
                # Reuse the once-per-routine cached evaluation of the original
                # subtree (see `_eval_orig_subtree` above) instead of redoing
                # `eval_tree_array(subtree_x1, X, ...)` on every attempt.
                is_similar, mse = @time_stage(
                    similarity_eval_ns, similarity_eval_calls,
                    _check_expr_similarity_with_orig(new_subtree_x1, orig_eval_X, orig_eval_res, options),
                )

                if is_similar
                    increment_stats!(STATS_REF[], :returned_similar_exprs, true)
                    # Remap back to original feature before returning
                    new_subtree = @time_stage(
                        remap_features_ns, remap_features_calls,
                        remap_features(new_subtree_x1, 1, original_feature),
                    )
                    return true, new_subtree, mse
                else
                    increment_stats!(STATS_REF[], :expr_similarity_failures, true)
                    # Store x1 version in candidates
                    push!(candidates, (new_subtree_x1, mse))
                end
            else
                # No similarity requirement - remap and return immediately
                new_subtree = @time_stage(
                    remap_features_ns, remap_features_calls,
                    remap_features(new_subtree_x1, 1, original_feature),
                )
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
            new_subtree = @time_stage(
                remap_features_ns, remap_features_calls,
                remap_features(candidate_subtree, 1, original_feature),
            )
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

"""
    _eval_orig_subtree(subtree, options, feature_cnt) -> (success, X, res_orig)

Evaluate the *original* subtree once per `sample_routine` call (not per
resampling attempt). The original subtree never changes inside the loop, so
re-evaluating it on every `check_expr_similarity` call is pure waste —
`similarity_eval` was the second-largest stage cost (~10 % of neural wall) in
job 82421 (post-decoder rewrite).

Returns `(true, X, res_orig)` on success or `(false, X, res_orig)` if the orig
eval fails or produces non-finite values (in which case the rest of the
sample routine should bail out — there's no point trying alternatives if the
original tree itself is broken).
"""
function _eval_orig_subtree(
    subtree::AbstractExpressionNode{T},
    options::AbstractOptions,
    feature_cnt::Int=1
)::Tuple{Bool, Matrix{T}, Vector{T}} where {T}
    X = if feature_cnt == 1
        Matrix{T}(EVAL_X_UNIVARIATE_REF[])
    else
        points_cnt = options.neural_options.eval_npoints * feature_cnt^2
        Matrix{T}(rand(Uniform(T(options.neural_options.eval_min),
                               T(options.neural_options.eval_max)),
                       feature_cnt, points_cnt))
    end
    try
        (res, complete) = eval_tree_array(subtree, X, options.operators)
        if !complete || any(x -> isnan(x) || isinf(x) || x >= prevfloat(typemax(T)) || x <= nextfloat(typemin(T)), res)
            increment_stats!(STATS_REF[], :orig_tree_eval_failures, true)
            return false, X, T[]
        end
        return true, X, res
    catch e
        increment_stats!(STATS_REF[], :orig_tree_eval_failures, true)
        return false, X, T[]
    end
end

function check_expr_similarity(
    subtree::AbstractExpressionNode{T},
    new_subtree::AbstractExpressionNode{T},
    options::AbstractOptions,
    feature_cnt::Int=1
)::Tuple{Bool, Float32} where {T}
    # Backwards-compatible path: evaluate orig + new ourselves. Used only by
    # `multivariate_decoding`, which builds X with random points each call so
    # the orig eval really *does* depend on the iteration. Hot path
    # (`sample_routine` univariate) calls the cached overload below instead.
    success, X, res = _eval_orig_subtree(subtree, options, feature_cnt)
    success || return false, Inf
    return _check_expr_similarity_with_orig(new_subtree, X, res, options)
end

"""
    _check_expr_similarity_with_orig(new_subtree, X, res_orig, options) -> (is_similar, mse)

Cached-orig variant of `check_expr_similarity`: caller provides the already-
evaluated `res_orig` for the original subtree on `X`, so we only evaluate the
candidate `new_subtree` here. Used by the resampling loop in `sample_routine`
where the original subtree is fixed across attempts.
"""
function _check_expr_similarity_with_orig(
    new_subtree::AbstractExpressionNode{T},
    X::Matrix{T},
    res::Vector{T},
    options::AbstractOptions,
)::Tuple{Bool, Float32} where {T}
    res_new = nothing
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

    transform = EVAL_TRANSFORM_REF[]
    mse_T = sum((transform.(res) .- transform.(res_new)).^2) / size(X, 2)
    mse = Float32(mse_T)
    return mse < options.neural_options.similarity_threshold, mse
end

"""
    check_novel_skeleton(x_in::AbstractArray{Float32}, x_out::AbstractArray{Float32}, options::AbstractOptions)

Check if the skeleton of the new subtree is novel.
    FIXME: Not quite correct as logits -> tree is probabilistic and samples from token distribution. Distributions usually have low
    entropy though so this is probably good enough.
"""
function check_novel_skeleton(x_in::AbstractMatrix{Float32}, x_out::AbstractMatrix{Float32}, options::AbstractOptions)
    # Compare argmax-per-row over the one-hot columns (skip the trailing
    # constants column). No materialization of intermediate slices: walk both
    # rows in lockstep with `@views` so we work directly on the underlying
    # buffer (`x_out` is typically a `view` into the per-batch output).
    n_rows = size(x_in, 1)
    n_cols_onehot = size(x_in, 2) - 1
    @inbounds for i in 1:n_rows
        in_max = argmax(@view x_in[i, 1:n_cols_onehot])
        out_max = argmax(@view x_out[i, 1:n_cols_onehot])
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