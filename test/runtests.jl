using Test

include("setup.jl")

HAS_GPU || @warn "Skipping cuDensityMat GPU tests: $GPU_SKIP_REASON"

@testset "CuQuantum.jl" begin

    @testset "Module loading" begin
        @test isdefined(CuQuantum, :CuDensityMat)
        @test isdefined(CuDensityMat, :WorkStream)
        @test isdefined(CuDensityMat, :DensePureState)
        @test isdefined(CuDensityMat, :DenseMixedState)
    end

    @testset "CuDensityMat" begin

        # Binding / ABI / version checks (no GPU needed)
        include("densitymat/test_bindings.jl")

        # Phase 1: WorkStream tests
        include("densitymat/test_workstream.jl")
        include("densitymat/test_workstream_mpi.jl")

        # Phase 2: State tests
        include("densitymat/test_state.jl")
        include("densitymat/test_state_mpi.jl")

        # Phase 3: Elementary Operator tests
        include("densitymat/test_elementary_operator.jl")

        # Phase 4: Composite Operator tests
        include("densitymat/test_operators.jl")

        # Phase 5: Callback tests
        include("densitymat/test_callbacks.jl")
        include("densitymat/test_gpu_callbacks.jl")

        # Phase 7: Gradient / backward differentiation tests
        include("densitymat/test_gradients.jl")

        # Phase 8: Expectation value tests
        include("densitymat/test_expectation.jl")

        # Phase 9: Eigenspectrum tests
        include("densitymat/test_spectrum.jl")

        # Phase 10: Integration test — full Lindblad simulation
        include("densitymat/test_integration.jl")

    end
end

if !isempty(SKIPPED_GPU_TESTS)
    @info "GPU tests skipped ($GPU_SKIP_REASON); run them on a CC >= $CUDENSITYMAT_MIN_CAPABILITY device:\n" *
          join("  - " .* SKIPPED_GPU_TESTS, "\n")
end
