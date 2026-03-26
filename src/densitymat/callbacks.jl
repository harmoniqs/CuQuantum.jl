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

export wrap_scalar_callback, wrap_tensor_callback

# =============================================================================
# Global callback registry (prevents GC of Julia functions)
# =============================================================================

# Maps UInt(id) → (julia_function, refs...) to prevent GC
const _callback_registry = Dict{UInt, Any}()
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
    if dt == CUDA.R_32F
        return Float32
    elseif dt == CUDA.R_64F
        return Float64
    elseif dt == CUDA.C_32F
        return ComplexF32
    elseif dt == CUDA.C_64F
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
    stream::CUstream
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
        (Ptr{Cvoid}, Cdouble, Int64, Int32, Ptr{Cdouble}, cudaDataType_t, Ptr{Cvoid}, CUstream)
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
    stream::CUstream
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
        (Ptr{Cvoid}, cudensitymatElementaryOperatorSparsity_t, Int32, Ptr{Int64},
         Ptr{Int32}, Cdouble, Int64, Int32, Ptr{Cdouble}, cudaDataType_t,
         Ptr{Cvoid}, CUstream)
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
    stream::CUstream
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
        params_grad = unsafe_wrap(Array, params_grad_ptr, (Int(num_params), Int(batch_size)))

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
        (Ptr{Cvoid}, Cdouble, Int64, Int32, Ptr{Cdouble}, cudaDataType_t,
         Ptr{Cvoid}, Ptr{Cdouble}, CUstream)
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
    stream::CUstream
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
        params_grad = unsafe_wrap(Array, params_grad_ptr, (Int(num_params), Int(batch_size)))

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
        (Ptr{Cvoid}, cudensitymatElementaryOperatorSparsity_t, Int32, Ptr{Int64},
         Ptr{Int32}, Cdouble, Int64, Int32, Ptr{Cdouble}, cudaDataType_t,
         Ptr{Cvoid}, Ptr{Cdouble}, CUstream)
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
end

# =============================================================================
# Public API: wrap user functions into C callback structs
# =============================================================================

"""
    wrap_scalar_callback(f; gradient=nothing) -> (cb, gcb, refs)

Wrap a Julia function as a scalar callback for time-dependent coefficients.

# Arguments
- `f`: Function with signature `f(time::Float64, params::Matrix{Float64}, storage::Vector{T})`
  where `T` matches the operator's data type. `params` is `(numParams, batchSize)`,
  `storage` is `(batchSize,)`. Write the coefficient values into `storage`.
- `gradient`: Optional gradient function with signature
  `g(time::Float64, params::Matrix{Float64}, scalar_grad::Vector{T}, params_grad::Matrix{Float64})`

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
```
"""
function wrap_scalar_callback(f::Function; gradient::Union{Nothing, Function}=nothing)
    # Ensure wrappers are initialized
    if _cpu_scalar_wrapper_ptr[] == C_NULL
        _init_callback_wrappers!()
    end

    id = _register_callback(f)

    cb = cudensitymatWrappedScalarCallback_t(
        Ptr{Cvoid}(id),                      # callback = our ID
        CUDENSITYMAT_CALLBACK_DEVICE_CPU,     # device
        _cpu_scalar_wrapper_ptr[]             # wrapper = our trampoline
    )

    gcb = if gradient !== nothing
        grad_id = _register_callback(gradient)
        cudensitymatWrappedScalarGradientCallback_t(
            Ptr{Cvoid}(grad_id),
            CUDENSITYMAT_CALLBACK_DEVICE_CPU,
            _cpu_scalar_gradient_wrapper_ptr[],
            CUDENSITYMAT_DIFFERENTIATION_DIR_BACKWARD
        )
    else
        NULL_SCALAR_GRADIENT_CALLBACK
    end

    # Return refs to prevent accidental unregistration
    refs = (id=id, grad_id=(gradient !== nothing ? grad_id : nothing), f=f, gradient=gradient)
    return cb, gcb, refs
end

"""
    wrap_tensor_callback(f; gradient=nothing) -> (cb, gcb, refs)

Wrap a Julia function as a tensor callback for time-dependent operator elements.

# Arguments
- `f`: Function with signature `f(time::Float64, params::Matrix{Float64}, storage::Array{T})`
  where `storage` shape is `(d1, d2, ..., d1, d2, ..., batchSize)` for dense operators.
  Write the tensor elements into `storage`.
- `gradient`: Optional gradient function.

# Returns
- `cb::cudensitymatWrappedTensorCallback_t`
- `gcb::cudensitymatWrappedTensorGradientCallback_t`
- `refs` — opaque reference holder
"""
function wrap_tensor_callback(f::Function; gradient::Union{Nothing, Function}=nothing)
    if _cpu_tensor_wrapper_ptr[] == C_NULL
        _init_callback_wrappers!()
    end

    id = _register_callback(f)

    cb = cudensitymatWrappedTensorCallback_t(
        Ptr{Cvoid}(id),
        CUDENSITYMAT_CALLBACK_DEVICE_CPU,
        _cpu_tensor_wrapper_ptr[]
    )

    gcb = if gradient !== nothing
        grad_id = _register_callback(gradient)
        cudensitymatWrappedTensorGradientCallback_t(
            Ptr{Cvoid}(grad_id),
            CUDENSITYMAT_CALLBACK_DEVICE_CPU,
            _cpu_tensor_gradient_wrapper_ptr[],
            CUDENSITYMAT_DIFFERENTIATION_DIR_BACKWARD
        )
    else
        NULL_TENSOR_GRADIENT_CALLBACK
    end

    refs = (id=id, grad_id=(gradient !== nothing ? grad_id : nothing), f=f, gradient=gradient)
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
