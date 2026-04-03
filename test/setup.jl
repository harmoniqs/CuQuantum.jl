# Shared test utilities for cuQuantum.jl test suite

using Test
using CUDA
using LinearAlgebra
using CuQuantum
using CuQuantum.CuDensityMat

# Check if a GPU is available for tests that require one
const HAS_GPU = try
    CUDA.functional()
catch
    false
end

# Skip macro for GPU-requiring tests
macro gpu_test(name, body)
    return esc(
        quote
            @testset $name begin
                if HAS_GPU
                    $body
                else
                    @test_skip "No GPU available"
                end
            end
        end
    )
end

# Common test parameters
const TEST_DTYPES = HAS_GPU ? [ComplexF32, ComplexF64] : []
const TEST_DIMS = [[2], [2, 2], [2, 3], [3, 3]]
const TEST_BATCH_SIZES = [1, 4]

# Sync the CUDA stream and pull a CuArray to CPU.
# Use instead of bare Array(x) in GPU tests so CUDA errors surface immediately.
sync_and_pull(x) = (CUDA.synchronize(); Array(x))
