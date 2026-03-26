# cuDensityMat type definitions: enums, opaque handles, callback structs

using CEnum: @cenum

# --- Status codes ---

@cenum cudensitymatStatus_t::Int32 begin
    CUDENSITYMAT_STATUS_SUCCESS                  = 0
    CUDENSITYMAT_STATUS_NOT_INITIALIZED          = 1
    CUDENSITYMAT_STATUS_ALLOC_FAILED             = 3
    CUDENSITYMAT_STATUS_INVALID_VALUE            = 7
    CUDENSITYMAT_STATUS_ARCH_MISMATCH            = 8
    CUDENSITYMAT_STATUS_EXECUTION_FAILED         = 13
    CUDENSITYMAT_STATUS_INTERNAL_ERROR           = 14
    CUDENSITYMAT_STATUS_NOT_SUPPORTED            = 15
    CUDENSITYMAT_STATUS_CALLBACK_ERROR           = 16
    CUDENSITYMAT_STATUS_CUBLAS_ERROR             = 17
    CUDENSITYMAT_STATUS_CUDA_ERROR               = 18
    CUDENSITYMAT_STATUS_INSUFFICIENT_WORKSPACE   = 19
    CUDENSITYMAT_STATUS_INSUFFICIENT_DRIVER      = 20
    CUDENSITYMAT_STATUS_IO_ERROR                 = 21
    CUDENSITYMAT_STATUS_CUTENSOR_VERSION_MISMATCH = 22
    CUDENSITYMAT_STATUS_NO_DEVICE_ALLOCATOR      = 23
    CUDENSITYMAT_STATUS_CUTENSOR_ERROR           = 24
    CUDENSITYMAT_STATUS_CUSOLVER_ERROR           = 25
    CUDENSITYMAT_STATUS_DEVICE_ALLOCATOR_ERROR   = 26
    CUDENSITYMAT_STATUS_DISTRIBUTED_FAILURE      = 27
    CUDENSITYMAT_STATUS_INTERRUPTED              = 28
    CUDENSITYMAT_STATUS_CUTENSORNET_ERROR        = 29
end

# --- Compute type ---

@cenum cudensitymatComputeType_t::UInt32 begin
    CUDENSITYMAT_COMPUTE_32F = (UInt32(1) << UInt32(2))   # 4
    CUDENSITYMAT_COMPUTE_64F = (UInt32(1) << UInt32(4))   # 16
end

# --- Distributed provider ---

@cenum cudensitymatDistributedProvider_t::Int32 begin
    CUDENSITYMAT_DISTRIBUTED_PROVIDER_NONE = 0
    CUDENSITYMAT_DISTRIBUTED_PROVIDER_MPI  = 1
    CUDENSITYMAT_DISTRIBUTED_PROVIDER_NCCL = 2
end

# --- Callback device ---

@cenum cudensitymatCallbackDevice_t::Int32 begin
    CUDENSITYMAT_CALLBACK_DEVICE_CPU = 0
    CUDENSITYMAT_CALLBACK_DEVICE_GPU = 1
end

# --- Differentiation direction ---

@cenum cudensitymatDifferentiationDir_t::Int32 begin
    CUDENSITYMAT_DIFFERENTIATION_DIR_BACKWARD = 1
end

# --- Memory space ---

@cenum cudensitymatMemspace_t::Int32 begin
    CUDENSITYMAT_MEMSPACE_DEVICE = 0
    CUDENSITYMAT_MEMSPACE_HOST   = 1
end

# --- Workspace kind ---

@cenum cudensitymatWorkspaceKind_t::Int32 begin
    CUDENSITYMAT_WORKSPACE_SCRATCH = 0
end

# --- State purity ---

@cenum cudensitymatStatePurity_t::Int32 begin
    CUDENSITYMAT_STATE_PURITY_PURE  = 0
    CUDENSITYMAT_STATE_PURITY_MIXED = 1
end

# --- Elementary operator sparsity ---

@cenum cudensitymatElementaryOperatorSparsity_t::Int32 begin
    CUDENSITYMAT_OPERATOR_SPARSITY_NONE           = 0
    CUDENSITYMAT_OPERATOR_SPARSITY_MULTIDIAGONAL  = 1
end

# --- Operator spectrum kind ---

@cenum cudensitymatOperatorSpectrumKind_t::Int32 begin
    CUDENSITYMAT_OPERATOR_SPECTRUM_LARGEST       = 0
    CUDENSITYMAT_OPERATOR_SPECTRUM_SMALLEST      = 1
    CUDENSITYMAT_OPERATOR_SPECTRUM_LARGEST_REAL   = 2
    CUDENSITYMAT_OPERATOR_SPECTRUM_SMALLEST_REAL  = 3
end

# --- Operator spectrum config ---

@cenum cudensitymatOperatorSpectrumConfig_t::Int32 begin
    CUDENSITYMAT_OPERATOR_SPECTRUM_CONFIG_MAX_EXPANSION  = 0
    CUDENSITYMAT_OPERATOR_SPECTRUM_CONFIG_MAX_RESTARTS   = 1
    CUDENSITYMAT_OPERATOR_SPECTRUM_CONFIG_MIN_BLOCK_SIZE = 2
end

# --- Opaque handle types ---
# All opaque handles are void* in C, represented as Ptr{Cvoid} in Julia.
# We define type aliases for clarity and type safety.

const cudensitymatHandle_t              = Ptr{Cvoid}
const cudensitymatWorkspaceDescriptor_t = Ptr{Cvoid}
const cudensitymatState_t               = Ptr{Cvoid}
const cudensitymatElementaryOperator_t  = Ptr{Cvoid}
const cudensitymatMatrixOperator_t      = Ptr{Cvoid}
const cudensitymatOperatorTerm_t        = Ptr{Cvoid}
const cudensitymatOperator_t            = Ptr{Cvoid}
const cudensitymatOperatorAction_t      = Ptr{Cvoid}
const cudensitymatExpectation_t         = Ptr{Cvoid}
const cudensitymatOperatorSpectrum_t    = Ptr{Cvoid}

# --- Distributed communicator struct ---

struct cudensitymatDistributedCommunicator_t
    commPtr::Ptr{Cvoid}
    commSize::Csize_t
end

# --- Callback function pointer types ---
# These are the C function pointer types for callbacks.
# In Julia, @cfunction produces a Ptr{Cvoid} which is compatible.

# cudensitymatScalarCallback_t:
#   int32_t (*)(double time, int64_t batchSize, int32_t numParams,
#               const double* params, cudaDataType_t dataType,
#               void* scalarStorage, cudaStream_t stream)
const cudensitymatScalarCallback_t = Ptr{Cvoid}

# cudensitymatTensorCallback_t:
#   int32_t (*)(cudensitymatElementaryOperatorSparsity_t sparsity,
#               int32_t numModes, const int64_t modeExtents[],
#               const int32_t diagonalOffsets[], double time,
#               int64_t batchSize, int32_t numParams, const double* params,
#               cudaDataType_t dataType, void* tensorStorage, cudaStream_t stream)
const cudensitymatTensorCallback_t = Ptr{Cvoid}

# cudensitymatScalarGradientCallback_t:
#   int32_t (*)(double time, int64_t batchSize, int32_t numParams,
#               const double* params, cudaDataType_t dataType,
#               void* scalarGrad, double* paramsGrad, cudaStream_t stream)
const cudensitymatScalarGradientCallback_t = Ptr{Cvoid}

# cudensitymatTensorGradientCallback_t:
#   int32_t (*)(cudensitymatElementaryOperatorSparsity_t sparsity,
#               int32_t numModes, const int64_t modeExtents[],
#               const int32_t diagonalOffsets[], double time,
#               int64_t batchSize, int32_t numParams, const double* params,
#               cudaDataType_t dataType, void* tensorGrad, double* paramsGrad,
#               cudaStream_t stream)
const cudensitymatTensorGradientCallback_t = Ptr{Cvoid}

# --- Callback wrapper structs ---
# These are passed by value to the C API.
# Layout must match C struct exactly.

struct cudensitymatWrappedScalarCallback_t
    callback::cudensitymatScalarCallback_t              # function pointer
    device::cudensitymatCallbackDevice_t                # CPU or GPU
    wrapper::Ptr{Cvoid}                                 # NULL from C/Julia (Python only)
end

struct cudensitymatWrappedTensorCallback_t
    callback::cudensitymatTensorCallback_t              # function pointer
    device::cudensitymatCallbackDevice_t                # CPU or GPU
    wrapper::Ptr{Cvoid}                                 # NULL from C/Julia (Python only)
end

struct cudensitymatWrappedScalarGradientCallback_t
    callback::cudensitymatScalarGradientCallback_t      # function pointer
    device::cudensitymatCallbackDevice_t                # CPU or GPU
    wrapper::Ptr{Cvoid}                                 # NULL from C/Julia (Python only)
    direction::cudensitymatDifferentiationDir_t          # differentiation direction
end

struct cudensitymatWrappedTensorGradientCallback_t
    callback::cudensitymatTensorGradientCallback_t      # function pointer
    device::cudensitymatCallbackDevice_t                # CPU or GPU
    wrapper::Ptr{Cvoid}                                 # NULL from C/Julia (Python only)
    direction::cudensitymatDifferentiationDir_t          # differentiation direction
end

# --- CUDA type aliases for ccall ---

const cudaStream_t = CUstream
const cudaDataType_t = cudaDataType
