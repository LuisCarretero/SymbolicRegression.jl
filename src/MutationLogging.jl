module MutationLoggingModule

using Dates

export Logger, init_logger, get_logger, log_event!, close_global_logger!

DTYPE = Float32

mutable struct Logger
    dirpath::String
    prefix::String
    fpath::String
    buffer::Vector{NamedTuple{
        (:timestamp, :mutation_type, :num_evals, :attempts, :loss_before, :loss_after, :score_before, :score_after, :successful_mutation, :mutation_accepted, :result_reason, :TED), 
        Tuple{Float64, String, DTYPE, Int, DTYPE, DTYPE, DTYPE, DTYPE, Bool, Bool, String, DTYPE}
    }}
    buffer_size::Int
    buffer_count::Int
    lock::ReentrantLock
    
    function Logger(dirpath::String, prefix::String; buffer_size::Int=10_000)
        # Create directory if it doesn't exist
        mkpath(dirpath)

        # Create filepath with timestamp
        timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
        fpath = joinpath(dirpath, "$(prefix)_$(timestamp).csv")

        # Initialize empty buffer
        buffer = Vector{NamedTuple{
            (:timestamp, :mutation_type, :num_evals, :attempts, :loss_before, :loss_after, :score_before, :score_after, :successful_mutation, :mutation_accepted, :result_reason, :TED), 
            Tuple{Float64, String, DTYPE, Int, DTYPE, DTYPE, DTYPE, DTYPE, Bool, Bool, String, DTYPE}
        }}(undef, buffer_size)
        
        logger = new(dirpath, prefix, fpath, buffer, buffer_size, 0, ReentrantLock())
        
        # Write headers
        open(fpath, "w") do io
            println(io, "timestamp,mutation_type,num_evals,attempts,loss_before,loss_after,score_before,score_after,successful_mutation,mutation_accepted,result_reason,TED")
        end
        
        return logger
    end
end

# Global reference to the logger
const GLOBAL_LOGGER = Ref{Union{Nothing, Logger}}(nothing)

# Initialize global logger
function init_logger(dirpath::String, prefix::String; buffer_size::Int=10_000)
    if !logger_initialized()
        GLOBAL_LOGGER[] = Logger(dirpath, prefix; buffer_size=buffer_size)
    else
        error("Logger already initialized. Call close_global_logger! first!")
    end
end

# Get the global logger
function get_logger()
    if GLOBAL_LOGGER[] === nothing
        error("Logger not initialized. Call init_logger first!")
    end
    GLOBAL_LOGGER[]
end

function logger_initialized()
    return GLOBAL_LOGGER[] !== nothing
end

function log_event!(
    logger::Logger, 
    timestamp::Float64, 
    mutation_type::String, 
    num_evals::DTYPE, 
    attempts::Int, 
    loss_before::DTYPE, 
    loss_after::DTYPE, 
    score_before::DTYPE, 
    score_after::DTYPE, 
    successful_mutation::Bool, 
    mutation_accepted::Bool, 
    result_reason::String, 
    TED::DTYPE)

    event = (
        timestamp=timestamp, 
        mutation_type=mutation_type, 
        num_evals=num_evals, 
        attempts=attempts, 
        loss_before=loss_before, 
        loss_after=loss_after, 
        score_before=score_before, 
        score_after=score_after, 
        successful_mutation=successful_mutation, 
        mutation_accepted=mutation_accepted, 
        result_reason=result_reason, TED=TED
    )
    
    lock(logger.lock) do
        # Add to buffer
        logger.buffer_count += 1
        
        # If buffer would overflow, write to file first
        if logger.buffer_count > logger.buffer_size
            open(logger.fpath, "a") do io
                for i in 1:logger.buffer_size
                    e = logger.buffer[i]
                    println(io, "$(e.timestamp),$(e.mutation_type),$(e.num_evals),$(e.attempts),
                    $(e.loss_before),$(e.loss_after),$(e.score_before),$(e.score_after),
                    $(e.successful_mutation),$(e.mutation_accepted),$(e.result_reason),$(e.TED)")
                end
            end
            logger.buffer_count = 1
        end
        
        # Store the event
        logger.buffer[logger.buffer_count] = event
        
        # If buffer is full, write to file
        if logger.buffer_count >= logger.buffer_size
            open(logger.fpath, "a") do io
                for i in 1:logger.buffer_count
                    e = logger.buffer[i]
                    println(io, "$(e.timestamp),$(e.mutation_type),$(e.num_evals),$(e.attempts),
                    $(e.loss_before),$(e.loss_after),$(e.score_before),$(e.score_after),
                    $(e.successful_mutation),$(e.mutation_accepted),$(e.result_reason),$(e.TED)")
                end
            end
            logger.buffer_count = 0
        end
    end
end

function log_event!(
    mutation_type::String, 
    num_evals::DTYPE, 
    attempts::Int, 
    loss_before::DTYPE, 
    loss_after::DTYPE, 
    score_before::DTYPE, 
    score_after::DTYPE, 
    successful_mutation::Bool, 
    mutation_accepted::Bool, 
    result_reason::String, 
    TED::DTYPE
)
    log_event!(
        get_logger(), 
        time(), 
        mutation_type, 
        num_evals, 
        attempts, 
        loss_before, 
        loss_after, 
        score_before, 
        score_after, 
        successful_mutation, 
        mutation_accepted, 
        result_reason, 
        TED
    )
end

function log_event!(  # FIXME: Is this casting function really needed?
    mutation_type::String, 
    num_evals::Number, 
    attempts::Number, 
    loss_before::Number, 
    loss_after::Number, 
    score_before::Number, 
    score_after::Number, 
    successful_mutation::Bool, 
    mutation_accepted::Bool, 
    result_reason::String, 
    TED::DTYPE
)
    log_event!(
        get_logger(), 
        time(), 
        mutation_type, 
        DTYPE(num_evals), 
        Int(attempts), 
        DTYPE(loss_before), 
        DTYPE(loss_after), 
        DTYPE(score_before), 
        DTYPE(score_after), 
        successful_mutation, 
        mutation_accepted, 
        result_reason, 
        TED
    )
end

function close_global_logger!()
    if GLOBAL_LOGGER[] !== nothing
        # Write any remaining buffer contents
        logger = GLOBAL_LOGGER[]
        if logger.buffer_count > 0
            lock(logger.lock) do
                open(logger.fpath, "a") do io
                    # Only iterate up to buffer_count, not exceeding buffer size
                    for i in 1:logger.buffer_count
                        e = logger.buffer[i]
                        println(io, "$(e.timestamp),$(e.mutation_type),$(e.num_evals),$(e.attempts),
                        $(e.loss_before),$(e.loss_after),$(e.score_before),$(e.score_after),
                        $(e.successful_mutation),$(e.mutation_accepted),$(e.result_reason),$(e.TED)")
                    end
                end
                # Reset buffer count after writing
                logger.buffer_count = 0
            end
        end
        GLOBAL_LOGGER[] = nothing
    end
end

end
