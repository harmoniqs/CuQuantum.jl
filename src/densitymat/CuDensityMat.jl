"""
    CuDensityMat

Julia bindings for NVIDIA's cuDensityMat library — high-performance density matrix
simulation for analog quantum dynamics (Lindblad master equation, time-dependent
Hamiltonians, backward differentiation/gradients).

Part of the CuQuantum.jl package.
"""
module CuDensityMat

using CUDA
using CUDA: CUstream, cudaDataType, libraryPropertyType
using CUDA: CuPtr, PtrOrCuPtr
using CUDA: unsafe_free!, retry_reclaim, initialize_context
using CUDA: @checked, @gcsafe_ccall
using CUDA: HandleCache

using CEnum: @cenum

# JLL or local toolkit
if CUDA.local_toolkit
    using CUDA_Runtime_Discovery
else
    import cuQuantum_jll
end

# Library handle — set in __init__
libcudensitymat = nothing

# Type definitions (enums, opaque handles, callback structs)
include("types.jl")

# Error handling
include("error.jl")

# Low-level ccall wrappers (58 functions)
include("libcudensitymat.jl")

# Convenience wrappers
include("wrappers.jl")

# WorkStream (high-level handle + workspace + stream + communicator)
include("workspace.jl")

# Quantum states (DensePureState, DenseMixedState)
include("state.jl")

# Operators (ElementaryOperator, OperatorTerm, Operator, OperatorAction)
include("operators.jl")

# Callbacks (time-dependent scalar/tensor coefficients)
include("callbacks.jl")

# Expectation values
include("expectation.jl")

# Eigenspectrum solver
include("spectrum.jl")

# --- Handle management ---

# Per-context handle cache following CUDA.jl pattern
const _handle_cache =
    HandleCache{CUDA.CuContext,cudensitymatHandle_t}((_) -> create(), (_, h) -> destroy(h))

"""
    handle() -> cudensitymatHandle_t

Get a cuDensityMat library handle for the current CUDA context.
Handles are cached per-context and reused.
"""
function handle()
    ctx = CUDA.context()
    return pop!(_handle_cache, ctx)
end

"""
    release_handle(h::cudensitymatHandle_t)

Return a handle to the cache for reuse.
"""
function release_handle(h::cudensitymatHandle_t)
    ctx = CUDA.context()
    push!(_handle_cache, ctx, h)
end

function __init__()
    precompiling = ccall(:jl_generating_output, Cint, ()) != 0

    CUDA.functional() || return

    # Find the library
    global libcudensitymat
    if CUDA.local_toolkit
        local dirs = CUDA_Runtime_Discovery.find_toolkit()
        local path =
            CUDA_Runtime_Discovery.get_library(dirs, "cudensitymat"; optional = true)
        if path === nothing
            precompiling || @error "cuDensityMat is not available on your system"
            return
        end
        libcudensitymat = path
    else
        if !cuQuantum_jll.is_available()
            precompiling ||
                @error "cuQuantum is not available for your platform ($(Base.BinaryPlatforms.triplet(cuQuantum_jll.host_platform)))"
            return
        end
        libcudensitymat = cuQuantum_jll.libcudensitymat
    end

    # Initialize callback @cfunction wrappers
    _init_callback_wrappers!()
end

end # module CuDensityMat
