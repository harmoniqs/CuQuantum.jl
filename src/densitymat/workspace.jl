# WorkStream: high-level handle bundling library context, workspace, stream, and communicator
#
# Mirrors Python's WorkStream dataclass. Owns the library handle, workspace descriptor,
# CUDA stream reference, and optional distributed communicator configuration.

export WorkStream

"""
    WorkStream(; stream=nothing, memory_limit=nothing, device_id=nothing)

A workspace context bundling a CuDensityMat library handle, workspace descriptor,
and CUDA stream. All CuDensityMat operations require a `WorkStream`.

# Keyword Arguments
- `stream::Union{Nothing, CUDA.CuStream}`: CUDA stream. Uses the default stream if `nothing`.
- `memory_limit::Union{Nothing, Int}`: Maximum workspace memory in bytes. Default is unlimited.
- `device_id::Union{Nothing, Int}`: CUDA device ordinal. Uses current device if `nothing`.

# Examples
```julia
ws = WorkStream()                          # default stream, current device
ws = WorkStream(stream=CUDA.CuStream())    # explicit stream
close(ws)                                  # release resources
```
"""
mutable struct WorkStream
    handle::cudensitymatHandle_t
    workspace::cudensitymatWorkspaceDescriptor_t
    stream::CUDA.CuStream
    device_id::Int
    memory_limit::Union{Nothing, Int}

    # Workspace buffer (managed CuVector for scratch space)
    workspace_buffer::Union{Nothing, CUDA.CuVector{UInt8}}
    workspace_size::Int

    # Communicator state
    comm_provider::cudensitymatDistributedProvider_t
    comm_set::Bool
    # Hold a reference to any comm data to prevent GC
    _comm_ref::Union{Nothing, Vector{Int}}

    function WorkStream(;
            stream::Union{Nothing, CUDA.CuStream} = nothing,
            memory_limit::Union{Nothing, Int} = nothing,
            device_id::Union{Nothing, Int} = nothing,
        )
        dev = if device_id !== nothing
            device_id
        else
            Int(CUDA.device())
        end

        s = stream !== nothing ? stream : CUDA.default_stream()

        # Create library handle
        h = create()

        # Create workspace descriptor
        ws_ref = Ref{cudensitymatWorkspaceDescriptor_t}()
        cudensitymatCreateWorkspace(h, ws_ref)
        ws = ws_ref[]

        obj = new(
            h,
            ws,
            s,
            dev,
            memory_limit,
            nothing,
            0,
            CUDENSITYMAT_DISTRIBUTED_PROVIDER_NONE,
            false,
            nothing,
        )

        # Register finalizer
        finalizer(_destroy!, obj)

        return obj
    end
end

function _destroy!(ws::WorkStream)
    if ws.workspace != C_NULL
        # Free workspace buffer first
        if ws.workspace_buffer !== nothing
            try
                CUDA.unsafe_free!(ws.workspace_buffer)
            catch
            end
            ws.workspace_buffer = nothing
        end
        try
            cudensitymatDestroyWorkspace(ws.workspace)
        catch
        end
        ws.workspace = C_NULL
    end
    if ws.handle != C_NULL
        try
            destroy(ws.handle)
        catch
        end
        ws.handle = C_NULL
    end
    return nothing
end

"""
    close(ws::WorkStream)

Explicitly release all resources held by the WorkStream.
"""
function Base.close(ws::WorkStream)
    _destroy!(ws)
    return nothing
end

"""
    isopen(ws::WorkStream)

Check if the WorkStream is still valid (not yet closed/finalized).
"""
Base.isopen(ws::WorkStream) = ws.handle != C_NULL && ws.workspace != C_NULL

function _check_valid(ws::WorkStream)
    return isopen(ws) || error("WorkStream has been closed or finalized")
end

# --- Communicator management ---

"""
    set_communicator!(ws::WorkStream, provider::Symbol; comm_ptr=nothing, comm_size=nothing)

Configure distributed execution on the WorkStream.

# Arguments
- `provider`: `:none`, `:mpi`, or `:nccl`
- `comm_ptr`: Pointer to the communicator (MPI_Comm* or ncclComm_t*)
- `comm_size`: Size of the communicator object in bytes

# MPI Example
```julia
using MPI
MPI.Init()
comm = MPI.COMM_WORLD
set_communicator!(ws, :mpi;
    comm_ptr=MPI.API.MPI_Comm_c2f(comm),
    comm_size=sizeof(MPI.MPI_Comm))
```
"""
function set_communicator!(
        ws::WorkStream,
        provider::Symbol;
        comm_ptr::Union{Nothing, Ptr{Cvoid}, Integer} = nothing,
        comm_size::Union{Nothing, Integer} = nothing,
    )
    _check_valid(ws)

    if ws.comm_set
        error("Communicator has already been set. Resetting is not supported.")
    end

    prov = if provider == :none
        CUDENSITYMAT_DISTRIBUTED_PROVIDER_NONE
    elseif provider == :mpi
        CUDENSITYMAT_DISTRIBUTED_PROVIDER_MPI
    elseif provider == :nccl
        CUDENSITYMAT_DISTRIBUTED_PROVIDER_NCCL
    else
        error("Unknown provider: $provider. Supported: :none, :mpi, :nccl")
    end

    if provider == :none
        cudensitymatResetDistributedConfiguration(ws.handle, prov, C_NULL, 0)
    else
        comm_ptr === nothing && error("$provider provider requires comm_ptr")
        comm_size === nothing && error("$provider provider requires comm_size")

        # Convert integer to pointer if needed
        ptr = comm_ptr isa Ptr{Cvoid} ? comm_ptr : Ptr{Cvoid}(UInt(comm_ptr))

        # Store the pointer in a stable Int array to prevent GC and ensure type stability
        if provider == :nccl
            holder = Int[Int(comm_ptr)]
            ws._comm_ref = holder
            ptr = Ptr{Cvoid}(pointer(holder))
            comm_size = sizeof(Int)
        else
            holder = Int[Int(UInt(comm_ptr))]
            ws._comm_ref = holder
        end

        cudensitymatResetDistributedConfiguration(ws.handle, prov, ptr, Csize_t(comm_size))
    end

    ws.comm_provider = prov
    ws.comm_set = true
    return nothing
end

"""
    get_num_ranks(ws::WorkStream) -> Int

Return the number of distributed processes.
"""
function get_num_ranks(ws::WorkStream)
    _check_valid(ws)
    n = Ref{Int32}()
    cudensitymatGetNumRanks(ws.handle, n)
    return Int(n[])
end

"""
    get_proc_rank(ws::WorkStream) -> Int

Return the rank of the current process.
"""
function get_proc_rank(ws::WorkStream)
    _check_valid(ws)
    r = Ref{Int32}()
    cudensitymatGetProcRank(ws.handle, r)
    return Int(r[])
end

# --- Workspace memory management ---

"""
    workspace_query_size(ws::WorkStream; memspace=:device, kind=:scratch) -> Int

Query the required workspace buffer size in bytes after a `prepare` call.
"""
function workspace_query_size(
        ws::WorkStream;
        memspace::Symbol = :device,
        kind::Symbol = :scratch,
    )
    _check_valid(ws)
    ms = memspace == :device ? CUDENSITYMAT_MEMSPACE_DEVICE : CUDENSITYMAT_MEMSPACE_HOST
    wk = CUDENSITYMAT_WORKSPACE_SCRATCH  # only scratch is supported
    size_ref = Ref{Csize_t}()
    cudensitymatWorkspaceGetMemorySize(ws.handle, ws.workspace, ms, wk, size_ref)
    return Int(size_ref[])
end

"""
    workspace_allocate!(ws::WorkStream, size::Int; memspace=:device, kind=:scratch)

Allocate and attach a workspace buffer of the given size.
"""
function workspace_allocate!(
        ws::WorkStream,
        size::Int;
        memspace::Symbol = :device,
        kind::Symbol = :scratch,
    )
    _check_valid(ws)
    size <= 0 && return nothing

    ms = memspace == :device ? CUDENSITYMAT_MEMSPACE_DEVICE : CUDENSITYMAT_MEMSPACE_HOST
    wk = CUDENSITYMAT_WORKSPACE_SCRATCH

    # Free existing buffer if too small
    if ws.workspace_buffer !== nothing && length(ws.workspace_buffer) < size
        CUDA.unsafe_free!(ws.workspace_buffer)
        ws.workspace_buffer = nothing
    end

    # Allocate if needed
    if ws.workspace_buffer === nothing
        ws.workspace_buffer = CUDA.CuVector{UInt8}(undef, size)
    end
    ws.workspace_size = size

    # Attach to workspace descriptor
    buf_ptr = pointer(ws.workspace_buffer)
    cudensitymatWorkspaceSetMemory(
        ws.handle,
        ws.workspace,
        ms,
        wk,
        reinterpret(CuPtr{Cvoid}, buf_ptr),
        Csize_t(size),
    )
    return nothing
end

"""
    release_workspace!(ws::WorkStream)

Release the workspace buffer and create a fresh workspace descriptor.
"""
function release_workspace!(ws::WorkStream)
    _check_valid(ws)

    # Free buffer
    if ws.workspace_buffer !== nothing
        CUDA.unsafe_free!(ws.workspace_buffer)
        ws.workspace_buffer = nothing
    end
    ws.workspace_size = 0

    # Destroy old workspace and create new one
    cudensitymatDestroyWorkspace(ws.workspace)
    ws_ref = Ref{cudensitymatWorkspaceDescriptor_t}()
    cudensitymatCreateWorkspace(ws.handle, ws_ref)
    ws.workspace = ws_ref[]

    return nothing
end

"""
    set_random_seed!(ws::WorkStream, seed::Integer)

Reset the random seed used by the library's internal random number generator.
"""
function set_random_seed!(ws::WorkStream, seed::Integer)
    _check_valid(ws)
    cudensitymatResetRandomSeed(ws.handle, Int32(seed))
    return nothing
end
