# Quantum state types: DensePureState, DenseMixedState
#
# High-level Julia wrappers around cudensitymat state API.
# Mirrors Python's DensePureState/DenseMixedState classes.

export DensePureState, DenseMixedState
export allocate_storage!, attach_storage!, state_view
export initialize_zero!, inplace_scale!, inplace_accumulate!, clone
export num_components, component_storage_size, storage_size, local_info

# --- Julia type → cudaDataType mapping ---

function julia_to_cuda_dtype(::Type{T}) where {T}
    return if T === Float32
        CUDA.R_32F
    elseif T === Float64
        CUDA.R_64F
    elseif T === ComplexF32
        CUDA.C_32F
    elseif T === ComplexF64
        CUDA.C_64F
    else
        error(
            "Unsupported data type: $T. Supported: Float32, Float64, ComplexF32, ComplexF64",
        )
    end
end

function real_eltype(::Type{T}) where {T}
    return if T === ComplexF32 || T === Float32
        Float32
    elseif T === ComplexF64 || T === Float64
        Float64
    else
        error("Unsupported type: $T")
    end
end

# --- Abstract State ---

abstract type AbstractState{T} end

"""
    DensePureState{T}(ws, hilbert_space_dims; batch_size=1)

Pure state in dense (state-vector) representation.

# Arguments
- `ws::WorkStream`: Execution context
- `hilbert_space_dims::Vector{Int}` or `Tuple`: Local Hilbert space dimensions
- `batch_size::Int=1`: Batch dimension

# Example
```julia
ws = WorkStream()
psi = DensePureState{ComplexF64}(ws, (2, 2, 2))
allocate_storage!(psi)
```
"""
mutable struct DensePureState{T} <: AbstractState{T}
    ws::WorkStream
    hilbert_space_dims::Vector{Int64}
    batch_size::Int64
    handle::cudensitymatState_t
    storage::Union{Nothing, CUDA.CuVector{T}}
    _owns_storage::Bool

    function DensePureState{T}(
            ws::WorkStream,
            hilbert_space_dims;
            batch_size::Integer = 1,
        ) where {T}
        _check_valid(ws)
        dims = Int64[d for d in hilbert_space_dims]
        state_ref = Ref{cudensitymatState_t}()
        cudensitymatCreateState(
            ws.handle,
            CUDENSITYMAT_STATE_PURITY_PURE,
            Int32(length(dims)),
            dims,
            Int64(batch_size),
            julia_to_cuda_dtype(T),
            state_ref,
        )
        obj = new{T}(ws, dims, Int64(batch_size), state_ref[], nothing, false)
        finalizer(_destroy!, obj)
        return obj
    end
end

function _destroy!(x::DensePureState)
    if x.handle != C_NULL
        try
            cudensitymatDestroyState(x.handle)
        catch
        end
        x.handle = C_NULL
    end
    if x._owns_storage && x.storage !== nothing
        try
            CUDA.unsafe_free!(x.storage)
        catch
        end
        x.storage = nothing
    end
end

Base.close(x::DensePureState) = _destroy!(x)

"""
    DenseMixedState{T}(ws, hilbert_space_dims; batch_size=1)

Mixed state in dense (density-matrix) representation.

# Arguments
- `ws::WorkStream`: Execution context
- `hilbert_space_dims::Vector{Int}` or `Tuple`: Local Hilbert space dimensions
- `batch_size::Int=1`: Batch dimension

# Example
```julia
ws = WorkStream()
rho = DenseMixedState{ComplexF64}(ws, (2, 2))
allocate_storage!(rho)
```
"""
mutable struct DenseMixedState{T} <: AbstractState{T}
    ws::WorkStream
    hilbert_space_dims::Vector{Int64}
    batch_size::Int64
    handle::cudensitymatState_t
    storage::Union{Nothing, CUDA.CuVector{T}}
    _owns_storage::Bool

    function DenseMixedState{T}(
            ws::WorkStream,
            hilbert_space_dims;
            batch_size::Integer = 1,
        ) where {T}
        _check_valid(ws)
        dims = Int64[d for d in hilbert_space_dims]
        state_ref = Ref{cudensitymatState_t}()
        cudensitymatCreateState(
            ws.handle,
            CUDENSITYMAT_STATE_PURITY_MIXED,
            Int32(length(dims)),
            dims,
            Int64(batch_size),
            julia_to_cuda_dtype(T),
            state_ref,
        )
        obj = new{T}(ws, dims, Int64(batch_size), state_ref[], nothing, false)
        finalizer(_destroy!, obj)
        return obj
    end
end

function _destroy!(x::DenseMixedState)
    if x.handle != C_NULL
        try
            cudensitymatDestroyState(x.handle)
        catch
        end
        x.handle = C_NULL
    end
    if x._owns_storage && x.storage !== nothing
        try
            CUDA.unsafe_free!(x.storage)
        catch
        end
        x.storage = nothing
    end
end

Base.close(x::DenseMixedState) = _destroy!(x)

const DenseState{T} = Union{DensePureState{T}, DenseMixedState{T}}

ispure(::DensePureState) = true
ispure(::DenseMixedState) = false

Base.isopen(s::AbstractState) = s.handle != C_NULL

function _check_state_valid(s::AbstractState)
    isopen(s) || error("State has been destroyed")
    return _check_valid(s.ws)
end

# --- Component info queries ---

"""
    num_components(state) -> Int

Number of storage components (1 for single-GPU, >1 for distributed).
"""
function num_components(state::AbstractState)
    _check_state_valid(state)
    n = Ref{Int32}()
    cudensitymatStateGetNumComponents(state.ws.handle, state.handle, n)
    return Int(n[])
end

"""
    component_storage_size(state) -> Vector{Int}

Storage size in bytes for each component.
"""
function component_storage_size(state::AbstractState)
    _check_state_valid(state)
    nc = num_components(state)
    sizes = Vector{Csize_t}(undef, nc)
    cudensitymatStateGetComponentStorageSize(
        state.ws.handle,
        state.handle,
        Int32(nc),
        sizes,
    )
    return Int[s for s in sizes]
end

"""
    storage_size(state) -> Int

Storage buffer size in number of elements of the state's data type.
"""
function storage_size(state::DenseState{T}) where {T}
    sizes = component_storage_size(state)
    return sizes[1] ÷ sizeof(T)
end

"""
    local_info(state) -> (shape::Tuple, offsets::Tuple)

Local storage buffer dimensions and mode offsets.
The last dimension is always the batch dimension.
"""
function local_info(state::DenseState{T}) where {T}
    _check_state_valid(state)

    # Get number of modes for component 0
    global_id = Ref{Int32}()
    num_modes = Ref{Int32}()
    batch_loc = Ref{Int32}()
    cudensitymatStateGetComponentNumModes(
        state.ws.handle,
        state.handle,
        Int32(0),
        global_id,
        num_modes,
        batch_loc,
    )
    nm = Int(num_modes[])

    # Get extents and offsets
    extents = Vector{Int64}(undef, nm)
    offsets = Vector{Int64}(undef, nm)
    gid = Ref{Int32}()
    nmod = Ref{Int32}()
    cudensitymatStateGetComponentInfo(
        state.ws.handle,
        state.handle,
        Int32(0),
        gid,
        nmod,
        extents,
        offsets,
    )

    shape = Tuple(extents)
    offs = Tuple(offsets)

    # For batch_size=1, the API may not include a batch dimension,
    # so we add it for consistency with Python
    if state.batch_size == 1 &&
            length(shape) == length(state.hilbert_space_dims) * (ispure(state) ? 1 : 2)
        shape = (shape..., 1)
        offs = (offs..., 0)
    end

    return shape, offs
end

# --- Storage management ---

"""
    attach_storage!(state, data::CuVector{T})

Attach a GPU buffer to the state. The buffer must be F-contiguous and
match the required storage size.
"""
function attach_storage!(state::DenseState{T}, data::CUDA.CuVector{T}) where {T}
    _check_state_valid(state)
    expected_size = storage_size(state)
    length(data) >= expected_size ||
        error("Buffer size $(length(data)) < required $(expected_size)")
    buf_ptrs = [pointer(data)]
    buf_sizes = Csize_t[length(data) * sizeof(T)]
    cudensitymatStateAttachComponentStorage(
        state.ws.handle,
        state.handle,
        Int32(1),
        buf_ptrs,
        buf_sizes,
    )
    state.storage = data
    state._owns_storage = false
    return nothing
end

"""
    allocate_storage!(state)

Allocate an appropriately sized buffer and attach it to the state.
"""
function allocate_storage!(state::DenseState{T}) where {T}
    _check_state_valid(state)
    sz = storage_size(state)
    buf = CUDA.zeros(T, sz)
    attach_storage!(state, buf)
    state._owns_storage = true
    return nothing
end

"""
    state_view(state) -> CuArray

Return a multidimensional view of the state's storage buffer.
"""
function state_view(state::DenseState{T}) where {T}
    _check_state_valid(state)
    state.storage === nothing && error("No storage attached")
    shape, _ = local_info(state)
    n = prod(shape)
    return reshape(view(state.storage, 1:n), shape)
end

"""
    initialize_zero!(state)

Set all elements of the state to zero using the C API.
"""
function initialize_zero!(state::AbstractState)
    _check_state_valid(state)
    cudensitymatStateInitializeZero(state.ws.handle, state.handle, CUDA.stream().handle)
    return nothing
end

# --- State computations ---

"""
    inplace_scale!(state, factors)

Scale the state by scalar factor(s).

`factors` can be a scalar, a Vector, or a CuVector of length `batch_size`.
"""
function inplace_scale!(state::DenseState{T}, factors) where {T}
    _check_state_valid(state)
    factors_gpu = _prepare_factors(state, factors)
    cudensitymatStateComputeScaling(
        state.ws.handle,
        state.handle,
        pointer(factors_gpu),
        CUDA.stream().handle,
    )
    CUDA.synchronize()
    return nothing
end

"""
    norm(state) -> Vector{real(T)}

Compute the squared Frobenius norm(s). Returns a CPU vector of length `batch_size`.
"""
function LinearAlgebra.norm(state::DenseState{T}) where {T}
    _check_state_valid(state)
    RT = real_eltype(T)
    result = CUDA.zeros(RT, state.batch_size)
    cudensitymatStateComputeNorm(
        state.ws.handle,
        state.handle,
        pointer(result),
        CUDA.stream().handle,
    )
    CUDA.synchronize()
    return Array(result)
end

"""
    trace(state) -> Vector{T}

Compute the trace(s). Returns a CPU vector of length `batch_size`.
"""
function LinearAlgebra.tr(state::DenseState{T}) where {T}
    _check_state_valid(state)
    result = CUDA.zeros(T, state.batch_size)
    cudensitymatStateComputeTrace(
        state.ws.handle,
        state.handle,
        pointer(result),
        CUDA.stream().handle,
    )
    CUDA.synchronize()
    return Array(result)
end

"""
    inplace_accumulate!(dest, src, factors=1)

Accumulate: `dest += factors * src`. Both states must be compatible.
"""
function inplace_accumulate!(
        dest::DenseState{T},
        src::DenseState{T},
        factors = one(T),
    ) where {T}
    _check_state_compatibility(dest, src)
    factors_gpu = _prepare_factors(dest, factors)
    cudensitymatStateComputeAccumulation(
        dest.ws.handle,
        src.handle,
        dest.handle,
        pointer(factors_gpu),
        CUDA.stream().handle,
    )
    CUDA.synchronize()
    return nothing
end

"""
    inner_product(left, right) -> Vector{T}

Compute inner product(s) ⟨left|right⟩. Returns a CPU vector of length `batch_size`.
"""
function LinearAlgebra.dot(left::DenseState{T}, right::DenseState{T}) where {T}
    _check_state_compatibility(left, right)
    result = CUDA.zeros(T, left.batch_size)
    cudensitymatStateComputeInnerProduct(
        left.ws.handle,
        left.handle,
        right.handle,
        pointer(result),
        CUDA.stream().handle,
    )
    CUDA.synchronize()
    return Array(result)
end

# Backward-compatible aliases — prefer the LinearAlgebra extensions above.
const norm = LinearAlgebra.norm
const trace = LinearAlgebra.tr
const inner_product = LinearAlgebra.dot

# --- Clone ---

"""
    clone(state, buf::CuVector{T}) -> State

Clone a state with a new storage buffer.
"""
function clone(state::DensePureState{T}, buf::CUDA.CuVector{T}) where {T}
    new_state =
        DensePureState{T}(state.ws, state.hilbert_space_dims; batch_size = state.batch_size)
    attach_storage!(new_state, buf)
    return new_state
end

function clone(state::DenseMixedState{T}, buf::CUDA.CuVector{T}) where {T}
    new_state = DenseMixedState{T}(
        state.ws,
        state.hilbert_space_dims;
        batch_size = state.batch_size,
    )
    attach_storage!(new_state, buf)
    return new_state
end

# --- Internal helpers ---

function _prepare_factors(state::DenseState{T}, factors) where {T}
    if factors isa Number
        return CUDA.fill(T(factors), state.batch_size)
    elseif factors isa AbstractVector
        length(factors) == state.batch_size ||
            error("factors length $(length(factors)) != batch_size $(state.batch_size)")
        if factors isa CUDA.CuVector{T}
            return factors
        else
            return CUDA.CuVector{T}(T.(factors))
        end
    else
        error("factors must be a Number or AbstractVector, got $(typeof(factors))")
    end
end

function _check_state_compatibility(a::AbstractState{T}, b::AbstractState{T}) where {T}
    typeof(a) == typeof(b) || error("State types must match: $(typeof(a)) vs $(typeof(b))")
    a.hilbert_space_dims == b.hilbert_space_dims || error("Hilbert space dims must match")
    a.batch_size == b.batch_size || error("Batch sizes must match")
    return a.ws === b.ws || error("WorkStreams must be the same object")
end
