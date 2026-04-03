# Error handling for cuDensityMat

export CUDENSITYMATError

"""
    CUDENSITYMATError(code)

Exception thrown when a cuDensityMat API call returns a non-success status code.
The `code` field holds the raw `cudensitymatStatus_t` value.
"""
struct CUDENSITYMATError <: Exception
    code::cudensitymatStatus_t
end

Base.convert(::Type{cudensitymatStatus_t}, err::CUDENSITYMATError) = err.code

Base.showerror(io::IO, err::CUDENSITYMATError) = print(
    io,
    "CUDENSITYMATError: ",
    description(err),
    " (code $(reinterpret(Int32, err.code)), $(err.code))",
)

function description(err::CUDENSITYMATError)
    return if err.code == CUDENSITYMAT_STATUS_SUCCESS
        "the operation completed successfully"
    elseif err.code == CUDENSITYMAT_STATUS_NOT_INITIALIZED
        "the library was not initialized"
    elseif err.code == CUDENSITYMAT_STATUS_ALLOC_FAILED
        "resource allocation failed"
    elseif err.code == CUDENSITYMAT_STATUS_INVALID_VALUE
        "an invalid value was used as an argument"
    elseif err.code == CUDENSITYMAT_STATUS_ARCH_MISMATCH
        "the device architecture is not supported"
    elseif err.code == CUDENSITYMAT_STATUS_EXECUTION_FAILED
        "an error occurred during GPU execution"
    elseif err.code == CUDENSITYMAT_STATUS_INTERNAL_ERROR
        "an internal error occurred"
    elseif err.code == CUDENSITYMAT_STATUS_NOT_SUPPORTED
        "the requested operation is not supported"
    elseif err.code == CUDENSITYMAT_STATUS_CALLBACK_ERROR
        "an error occurred in a user callback function"
    elseif err.code == CUDENSITYMAT_STATUS_CUBLAS_ERROR
        "a cuBLAS error occurred"
    elseif err.code == CUDENSITYMAT_STATUS_CUDA_ERROR
        "a CUDA error occurred"
    elseif err.code == CUDENSITYMAT_STATUS_INSUFFICIENT_WORKSPACE
        "insufficient workspace buffer"
    elseif err.code == CUDENSITYMAT_STATUS_INSUFFICIENT_DRIVER
        "the CUDA driver version is insufficient"
    elseif err.code == CUDENSITYMAT_STATUS_IO_ERROR
        "a file I/O error occurred"
    elseif err.code == CUDENSITYMAT_STATUS_CUTENSOR_VERSION_MISMATCH
        "the cuTENSOR library version is incompatible"
    elseif err.code == CUDENSITYMAT_STATUS_NO_DEVICE_ALLOCATOR
        "no device memory pool has been set"
    elseif err.code == CUDENSITYMAT_STATUS_CUTENSOR_ERROR
        "a cuTENSOR error occurred"
    elseif err.code == CUDENSITYMAT_STATUS_CUSOLVER_ERROR
        "a cuSOLVER error occurred"
    elseif err.code == CUDENSITYMAT_STATUS_DEVICE_ALLOCATOR_ERROR
        "device memory pool operation failed"
    elseif err.code == CUDENSITYMAT_STATUS_DISTRIBUTED_FAILURE
        "distributed communication failure"
    elseif err.code == CUDENSITYMAT_STATUS_INTERRUPTED
        "operation interrupted by user"
    elseif err.code == CUDENSITYMAT_STATUS_CUTENSORNET_ERROR
        "a cuTensorNet error occurred"
    else
        "unknown error"
    end
end

## API call error handling

function throw_api_error(res)
    throw(CUDENSITYMATError(res))
end

@inline function check(f)
    retry_if(res) = res in (
        CUDENSITYMAT_STATUS_NOT_INITIALIZED,
        CUDENSITYMAT_STATUS_ALLOC_FAILED,
        CUDENSITYMAT_STATUS_INTERNAL_ERROR,
    )
    res = retry_reclaim(f, retry_if)

    return if res != CUDENSITYMAT_STATUS_SUCCESS
        throw_api_error(res)
    end
end
