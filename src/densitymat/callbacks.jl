# Callbacks for time-dependent operators
#
# The cuDensityMat C API uses callback function pointers for time-dependent
# scalar coefficients and tensor elements. The WrappedCallback structs contain:
#   - callback: void* — stores a key to look up the Julia function
#   - device:   CPU or GPU
#   - wrapper:  C function pointer — our @cfunction trampoline
#
# When wrapper != NULL, the library calls wrapper(callback, ...) instead of
# callback(...) directly. Our trampoline receives the key via the callback arg,
# looks up the actual Julia function, and invokes it.
#
# GC safety: all registered callbacks are held in a global IdDict to prevent
# garbage collection while the C library holds raw pointers.
#
# Two kinds of trampolines exist, selected by the `device` keyword of
# `wrap_scalar_callback` / `wrap_tensor_callback`:
#   - CPU (CUDENSITYMAT_CALLBACK_DEVICE_CPU): buffers are host memory, wrapped as
#     `Array`s; the user function computes on the host.
#   - GPU (CUDENSITYMAT_CALLBACK_DEVICE_GPU): buffers are device memory, wrapped as
#     `CuDeviceArray`s, and the library's CUDA stream is passed as a `CuStream`.
#     The user function enqueues CUDA.jl kernels on that stream
#     (`@cuda stream=stream kernel(storage, ...)`) and returns without
#     synchronizing. The stream is not owned by CUDA.jl unless it is the task's
#     stream, so it can't be used with `CuArray`s (see `unsafe_wrap(CuStream, ...)`).

export wrap_scalar_callback, wrap_tensor_callback

# =============================================================================
# Global callback registry (prevents GC of Julia functions)
# =============================================================================

# Maps UInt(id) → (julia_function, refs...) to prevent GC
const _callback_registry = Dict{UInt,Any}()
const _callback_lock = ReentrantLock()
# Monotonic counter for unique callback IDs
const _callback_id_counter = Ref{UInt}(0)

function _register_callback(f)
    lock(_callback_lock) do
        _callback_id_counter[] += 1
        id = _callback_id_counter[]
        _callback_registry[id] = f
        return id
    end
end

function _unregister_callback(id::UInt)
    lock(_callback_lock) do
        delete!(_callback_registry, id)
    end
end

function _get_callback(id::UInt)
    lock(_callback_lock) do
        return _callback_registry[id]
    end
end

# =============================================================================
# CUDA data type → Julia element type mapping
# =============================================================================

function _cuda_dtype_to_julia(dt::cudaDataType_t)
    if dt == R_32F
        return Float32
    elseif dt == R_64F
        return Float64
    elseif dt == C_32F
        return ComplexF32
    elseif dt == C_64F
        return ComplexF64
    else
        error("Unsupported CUDA data type: $dt")
    end
end

# =============================================================================
# Scalar callback trampoline (CPU)
#
# C signature of the wrapper:
#   int32_t wrapper(cudensitymatScalarCallback_t callback,
#                   double time, int64_t batchSize, int32_t numParams,
#                   const double* params, cudaDataType_t dataType,
#                   void* scalarStorage, cudaStream_t stream)
#
# Our Julia user function signature:
#   f(time::Float64, params::Matrix{Float64}, storage::Vector{T}) -> nothing
#   where params is (numParams, batchSize) and storage is (batchSize,)
# =============================================================================

function _cpu_scalar_callback_wrapper(
    callback_ptr::Ptr{Cvoid},  # our callback ID disguised as a function pointer
    time::Cdouble,
    batch_size::Int64,
    num_params::Int32,
    params_ptr::Ptr{Cdouble},
    data_type::cudaDataType_t,
    storage_ptr::Ptr{Cvoid},
    stream::CUstream,
)::Int32
    try
        id = UInt(callback_ptr)
        f = _get_callback(id)
        T = _cuda_dtype_to_julia(data_type)

        # Wrap raw pointers as Julia arrays (zero-copy)
        # params is F-order: params[numParams, batchSize]
        params = if num_params > 0 && params_ptr != C_NULL
            unsafe_wrap(Array, params_ptr, (Int(num_params), Int(batch_size)))
        else
            Matrix{Float64}(undef, 0, Int(batch_size))
        end

        storage = unsafe_wrap(Array, Ptr{T}(storage_ptr), (Int(batch_size),))

        f(time, params, storage)
        return Int32(0)
    catch e
        @error "Scalar callback error" exception=(e, catch_backtrace())
        return Int32(-1)
    end
end

# Generate the @cfunction pointer (must be at top level, not inside a function)
const _cpu_scalar_wrapper_ptr = Ref{Ptr{Cvoid}}(C_NULL)

function _init_scalar_wrapper!()
    _cpu_scalar_wrapper_ptr[] = @cfunction(
        _cpu_scalar_callback_wrapper,
        Int32,
        (
            Ptr{Cvoid},
            Cdouble,
            Int64,
            Int32,
            Ptr{Cdouble},
            cudaDataType_t,
            Ptr{Cvoid},
            CUstream,
        )
    )
end

# =============================================================================
# Tensor callback trampoline (CPU)
#
# C signature of the wrapper:
#   int32_t wrapper(cudensitymatTensorCallback_t callback,
#                   cudensitymatElementaryOperatorSparsity_t sparsity,
#                   int32_t numModes, const int64_t* modeExtents,
#                   const int32_t* diagonalOffsets,
#                   double time, int64_t batchSize, int32_t numParams,
#                   const double* params, cudaDataType_t dataType,
#                   void* tensorStorage, cudaStream_t stream)
#
# Our Julia user function signature:
#   f(time::Float64, params::Matrix{Float64}, storage::Array{T}) -> nothing
# =============================================================================

function _cpu_tensor_callback_wrapper(
    callback_ptr::Ptr{Cvoid},
    sparsity::cudensitymatElementaryOperatorSparsity_t,
    num_modes::Int32,
    mode_extents_ptr::Ptr{Int64},
    diagonal_offsets_ptr::Ptr{Int32},
    time::Cdouble,
    batch_size::Int64,
    num_params::Int32,
    params_ptr::Ptr{Cdouble},
    data_type::cudaDataType_t,
    storage_ptr::Ptr{Cvoid},
    stream::CUstream,
)::Int32
    try
        id = UInt(callback_ptr)
        f = _get_callback(id)
        T = _cuda_dtype_to_julia(data_type)

        params = if num_params > 0 && params_ptr != C_NULL
            unsafe_wrap(Array, params_ptr, (Int(num_params), Int(batch_size)))
        else
            Matrix{Float64}(undef, 0, Int(batch_size))
        end

        # Compute tensor shape from mode extents
        mode_extents = unsafe_wrap(Array, mode_extents_ptr, (Int(num_modes),))

        # The library passes the full tensor shape via modeExtents.
        # For dense single-mode d: modeExtents = [d, d] (ket, bra indices already included).
        # We just append the batch dimension.
        shape = (mode_extents..., Int(batch_size))

        storage = unsafe_wrap(Array, Ptr{T}(storage_ptr), shape)

        f(time, params, storage)
        return Int32(0)
    catch e
        @error "Tensor callback error" exception=(e, catch_backtrace())
        return Int32(-1)
    end
end

const _cpu_tensor_wrapper_ptr = Ref{Ptr{Cvoid}}(C_NULL)

function _init_tensor_wrapper!()
    _cpu_tensor_wrapper_ptr[] = @cfunction(
        _cpu_tensor_callback_wrapper,
        Int32,
        (
            Ptr{Cvoid},
            cudensitymatElementaryOperatorSparsity_t,
            Int32,
            Ptr{Int64},
            Ptr{Int32},
            Cdouble,
            Int64,
            Int32,
            Ptr{Cdouble},
            cudaDataType_t,
            Ptr{Cvoid},
            CUstream,
        )
    )
end

# =============================================================================
# Scalar gradient callback trampoline (CPU)
#
# C signature of the wrapper:
#   int32_t wrapper(cudensitymatScalarGradientCallback_t callback,
#                   double time, int64_t batchSize, int32_t numParams,
#                   const double* params, cudaDataType_t dataType,
#                   void* scalarGrad, double* paramsGrad, cudaStream_t stream)
#
# User function: f(time, params, scalar_grad, params_grad) -> nothing
# =============================================================================

function _cpu_scalar_gradient_callback_wrapper(
    callback_ptr::Ptr{Cvoid},
    time::Cdouble,
    batch_size::Int64,
    num_params::Int32,
    params_ptr::Ptr{Cdouble},
    data_type::cudaDataType_t,
    scalar_grad_ptr::Ptr{Cvoid},
    params_grad_ptr::Ptr{Cdouble},
    stream::CUstream,
)::Int32
    try
        id = UInt(callback_ptr)
        f = _get_callback(id)
        T = _cuda_dtype_to_julia(data_type)

        params = if num_params > 0 && params_ptr != C_NULL
            unsafe_wrap(Array, params_ptr, (Int(num_params), Int(batch_size)))
        else
            Matrix{Float64}(undef, 0, Int(batch_size))
        end

        scalar_grad = unsafe_wrap(Array, Ptr{T}(scalar_grad_ptr), (Int(batch_size),))
        params_grad =
            unsafe_wrap(Array, params_grad_ptr, (Int(num_params), Int(batch_size)))

        f(time, params, scalar_grad, params_grad)
        return Int32(0)
    catch e
        @error "Scalar gradient callback error" exception=(e, catch_backtrace())
        return Int32(-1)
    end
end

const _cpu_scalar_gradient_wrapper_ptr = Ref{Ptr{Cvoid}}(C_NULL)

function _init_scalar_gradient_wrapper!()
    _cpu_scalar_gradient_wrapper_ptr[] = @cfunction(
        _cpu_scalar_gradient_callback_wrapper,
        Int32,
        (
            Ptr{Cvoid},
            Cdouble,
            Int64,
            Int32,
            Ptr{Cdouble},
            cudaDataType_t,
            Ptr{Cvoid},
            Ptr{Cdouble},
            CUstream,
        )
    )
end

# =============================================================================
# Tensor gradient callback trampoline (CPU)
# =============================================================================

function _cpu_tensor_gradient_callback_wrapper(
    callback_ptr::Ptr{Cvoid},
    sparsity::cudensitymatElementaryOperatorSparsity_t,
    num_modes::Int32,
    mode_extents_ptr::Ptr{Int64},
    diagonal_offsets_ptr::Ptr{Int32},
    time::Cdouble,
    batch_size::Int64,
    num_params::Int32,
    params_ptr::Ptr{Cdouble},
    data_type::cudaDataType_t,
    tensor_grad_ptr::Ptr{Cvoid},
    params_grad_ptr::Ptr{Cdouble},
    stream::CUstream,
)::Int32
    try
        id = UInt(callback_ptr)
        f = _get_callback(id)
        T = _cuda_dtype_to_julia(data_type)

        params = if num_params > 0 && params_ptr != C_NULL
            unsafe_wrap(Array, params_ptr, (Int(num_params), Int(batch_size)))
        else
            Matrix{Float64}(undef, 0, Int(batch_size))
        end

        mode_extents = unsafe_wrap(Array, mode_extents_ptr, (Int(num_modes),))

        shape = (mode_extents..., Int(batch_size))

        tensor_grad = unsafe_wrap(Array, Ptr{T}(tensor_grad_ptr), shape)
        params_grad =
            unsafe_wrap(Array, params_grad_ptr, (Int(num_params), Int(batch_size)))

        f(time, params, tensor_grad, params_grad)
        return Int32(0)
    catch e
        @error "Tensor gradient callback error" exception=(e, catch_backtrace())
        return Int32(-1)
    end
end

const _cpu_tensor_gradient_wrapper_ptr = Ref{Ptr{Cvoid}}(C_NULL)

function _init_tensor_gradient_wrapper!()
    _cpu_tensor_gradient_wrapper_ptr[] = @cfunction(
        _cpu_tensor_gradient_callback_wrapper,
        Int32,
        (
            Ptr{Cvoid},
            cudensitymatElementaryOperatorSparsity_t,
            Int32,
            Ptr{Int64},
            Ptr{Int32},
            Cdouble,
            Int64,
            Int32,
            Ptr{Cdouble},
            cudaDataType_t,
            Ptr{Cvoid},
            Ptr{Cdouble},
            CUstream,
        )
    )
end

# =============================================================================
# GPU trampolines (CUDENSITYMAT_CALLBACK_DEVICE_GPU)
#
# Same C signatures as the CPU trampolines. The buffers are device pointers and
# are exposed as `CuDeviceArray`s (no CUDA.jl memory management involved), and
# the stream is exposed as a `CuStream` for launching kernels on.
#
# User function signatures (all return nothing; work is enqueued on `stream`):
#   scalar:          f(time, params::CuDeviceMatrix{Float64}, storage::CuDeviceVector{T}, stream)
#   tensor:          f(time, params, storage::CuDeviceArray{T}, stream)
#   scalar gradient: g(time, params, scalar_grad::CuDeviceVector{T}, params_grad::CuDeviceMatrix{Float64}, stream)
#   tensor gradient: g(time, params, tensor_grad::CuDeviceArray{T}, params_grad, stream)
# =============================================================================

const _AS_Global = CUDACore.AS.Global

function _device_array(::Type{T}, ptr::Ptr, dims::Dims{N}) where {T,N}
    return CuDeviceArray{T,N,_AS_Global}(reinterpret(Core.LLVMPtr{T,_AS_Global}, ptr), dims)
end

# The library passes the stream it was given (normally the task's stream, which
# CUDA.jl owns), or a stream CUDA.jl doesn't know about, or the NULL stream.
function _callback_stream(handle::CUstream)
    handle == C_NULL && return CUDACore.default_stream()
    task_stream = CUDACore.stream()
    task_stream.handle == handle && return task_stream
    return unsafe_wrap(CuStream, handle)
end

_device_params(params_ptr, num_params, batch_size) =
    _device_array(Float64, params_ptr, (Int(num_params), Int(batch_size)))

function _gpu_scalar_callback_wrapper(
    callback_ptr::Ptr{Cvoid},
    time::Cdouble,
    batch_size::Int64,
    num_params::Int32,
    params_ptr::Ptr{Cdouble},
    data_type::cudaDataType_t,
    storage_ptr::Ptr{Cvoid},
    stream::CUstream,
)::Int32
    try
        f = _get_callback(UInt(callback_ptr))
        T = _cuda_dtype_to_julia(data_type)
        params = _device_params(params_ptr, num_params, batch_size)
        storage = _device_array(T, storage_ptr, (Int(batch_size),))
        f(time, params, storage, _callback_stream(stream))
        return Int32(0)
    catch e
        @error "GPU scalar callback error" exception=(e, catch_backtrace())
        return Int32(-1)
    end
end

function _gpu_tensor_callback_wrapper(
    callback_ptr::Ptr{Cvoid},
    sparsity::cudensitymatElementaryOperatorSparsity_t,
    num_modes::Int32,
    mode_extents_ptr::Ptr{Int64},
    diagonal_offsets_ptr::Ptr{Int32},
    time::Cdouble,
    batch_size::Int64,
    num_params::Int32,
    params_ptr::Ptr{Cdouble},
    data_type::cudaDataType_t,
    storage_ptr::Ptr{Cvoid},
    stream::CUstream,
)::Int32
    try
        f = _get_callback(UInt(callback_ptr))
        T = _cuda_dtype_to_julia(data_type)
        params = _device_params(params_ptr, num_params, batch_size)
        # modeExtents is a host array (see the CPU trampoline for the layout)
        mode_extents = unsafe_wrap(Array, mode_extents_ptr, (Int(num_modes),))
        shape = (Int.(mode_extents)..., Int(batch_size))
        storage = _device_array(T, storage_ptr, shape)
        f(time, params, storage, _callback_stream(stream))
        return Int32(0)
    catch e
        @error "GPU tensor callback error" exception=(e, catch_backtrace())
        return Int32(-1)
    end
end

function _gpu_scalar_gradient_callback_wrapper(
    callback_ptr::Ptr{Cvoid},
    time::Cdouble,
    batch_size::Int64,
    num_params::Int32,
    params_ptr::Ptr{Cdouble},
    data_type::cudaDataType_t,
    scalar_grad_ptr::Ptr{Cvoid},
    params_grad_ptr::Ptr{Cdouble},
    stream::CUstream,
)::Int32
    try
        f = _get_callback(UInt(callback_ptr))
        T = _cuda_dtype_to_julia(data_type)
        params = _device_params(params_ptr, num_params, batch_size)
        scalar_grad = _device_array(T, scalar_grad_ptr, (Int(batch_size),))
        params_grad = _device_params(params_grad_ptr, num_params, batch_size)
        f(time, params, scalar_grad, params_grad, _callback_stream(stream))
        return Int32(0)
    catch e
        @error "GPU scalar gradient callback error" exception=(e, catch_backtrace())
        return Int32(-1)
    end
end

function _gpu_tensor_gradient_callback_wrapper(
    callback_ptr::Ptr{Cvoid},
    sparsity::cudensitymatElementaryOperatorSparsity_t,
    num_modes::Int32,
    mode_extents_ptr::Ptr{Int64},
    diagonal_offsets_ptr::Ptr{Int32},
    time::Cdouble,
    batch_size::Int64,
    num_params::Int32,
    params_ptr::Ptr{Cdouble},
    data_type::cudaDataType_t,
    tensor_grad_ptr::Ptr{Cvoid},
    params_grad_ptr::Ptr{Cdouble},
    stream::CUstream,
)::Int32
    try
        f = _get_callback(UInt(callback_ptr))
        T = _cuda_dtype_to_julia(data_type)
        params = _device_params(params_ptr, num_params, batch_size)
        mode_extents = unsafe_wrap(Array, mode_extents_ptr, (Int(num_modes),))
        shape = (Int.(mode_extents)..., Int(batch_size))
        tensor_grad = _device_array(T, tensor_grad_ptr, shape)
        params_grad = _device_params(params_grad_ptr, num_params, batch_size)
        f(time, params, tensor_grad, params_grad, _callback_stream(stream))
        return Int32(0)
    catch e
        @error "GPU tensor gradient callback error" exception=(e, catch_backtrace())
        return Int32(-1)
    end
end

const _gpu_scalar_wrapper_ptr = Ref{Ptr{Cvoid}}(C_NULL)
const _gpu_tensor_wrapper_ptr = Ref{Ptr{Cvoid}}(C_NULL)
const _gpu_scalar_gradient_wrapper_ptr = Ref{Ptr{Cvoid}}(C_NULL)
const _gpu_tensor_gradient_wrapper_ptr = Ref{Ptr{Cvoid}}(C_NULL)

function _init_gpu_wrappers!()
    _gpu_scalar_wrapper_ptr[] = @cfunction(
        _gpu_scalar_callback_wrapper,
        Int32,
        (
            Ptr{Cvoid},
            Cdouble,
            Int64,
            Int32,
            Ptr{Cdouble},
            cudaDataType_t,
            Ptr{Cvoid},
            CUstream,
        )
    )
    _gpu_tensor_wrapper_ptr[] = @cfunction(
        _gpu_tensor_callback_wrapper,
        Int32,
        (
            Ptr{Cvoid},
            cudensitymatElementaryOperatorSparsity_t,
            Int32,
            Ptr{Int64},
            Ptr{Int32},
            Cdouble,
            Int64,
            Int32,
            Ptr{Cdouble},
            cudaDataType_t,
            Ptr{Cvoid},
            CUstream,
        )
    )
    _gpu_scalar_gradient_wrapper_ptr[] = @cfunction(
        _gpu_scalar_gradient_callback_wrapper,
        Int32,
        (
            Ptr{Cvoid},
            Cdouble,
            Int64,
            Int32,
            Ptr{Cdouble},
            cudaDataType_t,
            Ptr{Cvoid},
            Ptr{Cdouble},
            CUstream,
        )
    )
    _gpu_tensor_gradient_wrapper_ptr[] = @cfunction(
        _gpu_tensor_gradient_callback_wrapper,
        Int32,
        (
            Ptr{Cvoid},
            cudensitymatElementaryOperatorSparsity_t,
            Int32,
            Ptr{Int64},
            Ptr{Int32},
            Cdouble,
            Int64,
            Int32,
            Ptr{Cdouble},
            cudaDataType_t,
            Ptr{Cvoid},
            Ptr{Cdouble},
            CUstream,
        )
    )
end

# =============================================================================
# Initialize all wrapper pointers (called from __init__)
# =============================================================================

function _init_callback_wrappers!()
    _init_scalar_wrapper!()
    _init_tensor_wrapper!()
    _init_scalar_gradient_wrapper!()
    _init_tensor_gradient_wrapper!()
    _init_gpu_wrappers!()
end

# cuDensityMat device enum for a `device` keyword value
function _callback_device(device::Symbol)
    device === :cpu && return CUDENSITYMAT_CALLBACK_DEVICE_CPU
    device === :gpu && return CUDENSITYMAT_CALLBACK_DEVICE_GPU
    throw(ArgumentError("callback device must be :cpu or :gpu, got :$device"))
end

# =============================================================================
# Public API: wrap user functions into C callback structs
# =============================================================================

"""
    wrap_scalar_callback(f; gradient=nothing, device=:cpu) -> (cb, gcb, refs)

Wrap a Julia function as a scalar callback for time-dependent coefficients.

# Arguments
- `f`: the coefficient function. For `device = :cpu`,
  `f(time::Float64, params::Matrix{Float64}, storage::Vector{T})` computes on the host;
  for `device = :gpu`,
  `f(time::Float64, params::CuDeviceMatrix{Float64}, storage::CuDeviceVector{T}, stream::CuStream)`
  enqueues CUDA.jl kernels on `stream` (e.g. `@cuda stream=stream kernel(storage, params, time)`)
  and returns without synchronizing. `T` matches the operator's data type, `params` is
  `(numParams, batchSize)` and `storage` is `(batchSize,)`. Write the coefficient values
  into `storage`.
- `gradient`: Optional gradient function, `g(time, params, scalar_grad, params_grad)` on
  the CPU, or `g(time, params, scalar_grad, params_grad, stream)` on the GPU.
  `scalar_grad[b]` is the (unconjugated) adjoint of the coefficient `f`, input only;
  the function accumulates `params_grad[i, b] += real(scalar_grad[b] * ∂f[b]/∂params[i, b])`
  — `Re`, with no factor 2, in the gradient convention of
  [`compute_operator_action_backward!`](@ref).
- `device`: `:cpu` (`CUDENSITYMAT_CALLBACK_DEVICE_CPU`) or `:gpu`
  (`CUDENSITYMAT_CALLBACK_DEVICE_GPU`); applies to both `f` and `gradient`.

# Returns
- `cb::cudensitymatWrappedScalarCallback_t` — the callback struct
- `gcb::cudensitymatWrappedScalarGradientCallback_t` — the gradient callback struct
- `refs` — opaque reference holder (keep alive while callback is in use)

# Example
```julia
# Time-dependent coefficient: f(t) = exp(iΩt) where Ω = params[1,:]
function my_coeff(time, params, storage)
    for i in axes(storage, 1)
        ω = params[1, i]
        storage[i] = complex(cos(ω * time), sin(ω * time))
    end
end
cb, gcb, refs = wrap_scalar_callback(my_coeff)

# The same on the GPU
function cis_kernel!(storage, params, t)
    i = threadIdx().x
    i <= length(storage) && (storage[i] = cis(params[1, i] * t))
    return
end
gpu_coeff(t, params, storage, stream) =
    @cuda threads=length(storage) stream=stream cis_kernel!(storage, params, t)
cb, gcb, refs = wrap_scalar_callback(gpu_coeff; device=:gpu)
```
"""
function wrap_scalar_callback(
    f::Function;
    gradient::Union{Nothing,Function} = nothing,
    device::Symbol = :cpu,
)
    dev = _callback_device(device)
    if _cpu_scalar_wrapper_ptr[] == C_NULL
        _init_callback_wrappers!()
    end
    gpu = dev == CUDENSITYMAT_CALLBACK_DEVICE_GPU

    id = _register_callback(f)
    cb = cudensitymatWrappedScalarCallback_t(
        Ptr{Cvoid}(id),                      # callback = our ID
        dev,
        gpu ? _gpu_scalar_wrapper_ptr[] : _cpu_scalar_wrapper_ptr[],
    )

    grad_id = gradient === nothing ? nothing : _register_callback(gradient)
    gcb = if grad_id === nothing
        NULL_SCALAR_GRADIENT_CALLBACK
    else
        cudensitymatWrappedScalarGradientCallback_t(
            Ptr{Cvoid}(grad_id),
            dev,
            gpu ? _gpu_scalar_gradient_wrapper_ptr[] : _cpu_scalar_gradient_wrapper_ptr[],
            CUDENSITYMAT_DIFFERENTIATION_DIR_BACKWARD,
        )
    end

    # Return refs to prevent accidental unregistration
    refs = (id = id, grad_id = grad_id, f = f, gradient = gradient)
    return cb, gcb, refs
end

"""
    wrap_tensor_callback(f; gradient=nothing, device=:cpu) -> (cb, gcb, refs)

Wrap a Julia function as a tensor callback for time-dependent operator elements.

# Arguments
- `f`: `f(time::Float64, params::Matrix{Float64}, storage::Array{T})` for `device = :cpu`,
  or `f(time, params::CuDeviceMatrix{Float64}, storage::CuDeviceArray{T}, stream::CuStream)`
  for `device = :gpu` (enqueue kernels on `stream`, don't synchronize). `storage` has
  shape `(d1, d2, ..., d1, d2, ..., batchSize)` for dense operators.
  Write the tensor elements into `storage`.
- `gradient`: Optional gradient function, `g(time, params, tensor_grad, params_grad)`
  (CPU) or `g(time, params, tensor_grad, params_grad, stream)` (GPU). `tensor_grad` has
  the shape of `storage` and holds the (unconjugated) adjoint of the tensor; the function
  accumulates `params_grad[i, b] += real(sum(tensor_grad[.., b] .* ∂storage[.., b]/∂params[i, b]))`
  — `Re`, no factor 2, see [`compute_operator_action_backward!`](@ref).
- `device`: `:cpu` or `:gpu`, see [`wrap_scalar_callback`](@ref).

# Returns
- `cb::cudensitymatWrappedTensorCallback_t`
- `gcb::cudensitymatWrappedTensorGradientCallback_t`
- `refs` — opaque reference holder
"""
function wrap_tensor_callback(
    f::Function;
    gradient::Union{Nothing,Function} = nothing,
    device::Symbol = :cpu,
)
    dev = _callback_device(device)
    if _cpu_tensor_wrapper_ptr[] == C_NULL
        _init_callback_wrappers!()
    end
    gpu = dev == CUDENSITYMAT_CALLBACK_DEVICE_GPU

    id = _register_callback(f)
    cb = cudensitymatWrappedTensorCallback_t(
        Ptr{Cvoid}(id),
        dev,
        gpu ? _gpu_tensor_wrapper_ptr[] : _cpu_tensor_wrapper_ptr[],
    )

    grad_id = gradient === nothing ? nothing : _register_callback(gradient)
    gcb = if grad_id === nothing
        NULL_TENSOR_GRADIENT_CALLBACK
    else
        cudensitymatWrappedTensorGradientCallback_t(
            Ptr{Cvoid}(grad_id),
            dev,
            gpu ? _gpu_tensor_gradient_wrapper_ptr[] : _cpu_tensor_gradient_wrapper_ptr[],
            CUDENSITYMAT_DIFFERENTIATION_DIR_BACKWARD,
        )
    end

    refs = (id = id, grad_id = grad_id, f = f, gradient = gradient)
    return cb, gcb, refs
end

"""
    unregister_callback!(refs)

Release a callback from the global registry, allowing GC.
Call this after destroying the operator that uses the callback.
"""
function unregister_callback!(refs)
    if refs.id !== nothing
        _unregister_callback(refs.id)
    end
    if refs.grad_id !== nothing
        _unregister_callback(refs.grad_id)
    end
end
