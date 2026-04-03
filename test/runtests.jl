using Test

include("setup.jl")
include("aqua.jl")

@testset "CuQuantum.jl" begin

    @testset "Module loading" begin
        @test isdefined(CuQuantum, :CuDensityMat)
        @test isdefined(CuDensityMat, :WorkStream)
        @test isdefined(CuDensityMat, :DensePureState)
        @test isdefined(CuDensityMat, :DenseMixedState)
    end

    @testset "CuDensityMat" begin

        @testset "Version" begin
            if HAS_GPU
                v = CuDensityMat.version()
                @test v isa VersionNumber
                @test v >= v"0.0.0"
            else
                @test_skip "No GPU available for version query"
            end
        end

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

        # Phase 6: Batch operator construction and compute tests
        include("densitymat/test_batch_operators.jl")

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
