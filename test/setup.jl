# Shared test utilities for cuQuantum.jl test suite

using Test
using CUDA
using LinearAlgebra
using CuQuantum
using CuQuantum.CuDensityMat

# cuDensityMat itself requires compute capability >= 7.5 (Turing or newer); older
# devices may still run plain CUDA.jl kernels.
const CUDENSITYMAT_MIN_CAPABILITY = v"7.5"

# A CUDA device usable by CUDA.jl (enough for the GPU-callback trampoline tests).
const HAS_CUDA = try
    CUDA.functional()
catch
    false
end

# A CUDA device that cuDensityMat supports (required for every library call that
# takes a handle).
const HAS_GPU = HAS_CUDA && CUDA.capability(CUDA.device()) >= CUDENSITYMAT_MIN_CAPABILITY

const GPU_SKIP_REASON = if HAS_GPU
    ""
elseif HAS_CUDA
    "cuDensityMat needs compute capability >= $CUDENSITYMAT_MIN_CAPABILITY, " *
    "found $(CUDA.name(CUDA.device())) (CC $(CUDA.capability(CUDA.device())))"
else
    "no functional CUDA device"
end

# Names of the GPU testsets that were skipped, reported at the end of the run so
# they can be scheduled on a capable runner.
const SKIPPED_GPU_TESTS = String[]

# Testsets that need a cuDensityMat-capable GPU.
macro gpu_test(name, body)
    esc(quote
        @testset $name begin
            if HAS_GPU
                $body
            else
                push!(SKIPPED_GPU_TESTS, $name)
                @test_skip $name
            end
        end
    end)
end

# Testsets that only need CUDA.jl to launch kernels (no cuDensityMat handle).
macro cuda_test(name, body)
    esc(quote
        @testset $name begin
            if HAS_CUDA
                $body
            else
                push!(SKIPPED_GPU_TESTS, $name)
                @test_skip $name
            end
        end
    end)
end

# Common test parameters
const TEST_DTYPES = HAS_GPU ? [ComplexF32, ComplexF64] : []
const TEST_DIMS = [[2], [2, 2], [2, 3], [3, 3]]
const TEST_BATCH_SIZES = [0, 1, 4]
