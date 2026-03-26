# Phase 9: Eigenspectrum tests
#
# The eigenspectrum API may not be supported on all SDK versions / GPU architectures.
# Tests are guarded to skip gracefully if NOT_SUPPORTED is returned.

# Helper: check if eigenspectrum is supported on this setup
function _spectrum_supported()
    HAS_GPU || return false
    try
        ws = WorkStream()
        dims = [2]
        data = CUDA.CuVector{ComplexF64}([1.0+0im, 0.0, 0.0, -1.0+0im])
        elem = CuDensityMat.create_elementary_operator(ws, [2], data)
        term = CuDensityMat.create_operator_term(ws, dims)
        CuDensityMat.append_elementary_product!(term, [elem], Int32[0], Int32[0])
        op = CuDensityMat.create_operator(ws, dims)
        CuDensityMat.append_term!(op, term; duality=0)
        spec = CuDensityMat.create_operator_spectrum(ws, op)
        CuDensityMat.destroy_operator_spectrum(spec)
        close(ws)
        return true
    catch e
        if e isa CuDensityMat.CUDENSITYMATError
            return false
        end
        rethrow(e)
    end
end

const SPECTRUM_SUPPORTED = _spectrum_supported()

@testset "Eigenspectrum (local)" begin

    if !SPECTRUM_SUPPORTED
        @test_broken false  # eigenspectrum not supported on this SDK version / GPU
    end

    @gpu_test "spectrum creation and destruction" begin
        SPECTRUM_SUPPORTED || return
        ws = WorkStream()
        dims = [2]
        T = ComplexF64

        data = CUDA.CuVector{T}([1.0+0im, 0.0, 0.0, -1.0+0im])
        elem = CuDensityMat.create_elementary_operator(ws, [2], data)
        term = CuDensityMat.create_operator_term(ws, dims)
        CuDensityMat.append_elementary_product!(term, [elem], Int32[0], Int32[0])
        operator = CuDensityMat.create_operator(ws, dims)
        CuDensityMat.append_term!(operator, term; duality=0)

        spec = CuDensityMat.create_operator_spectrum(ws, operator;
            is_hermitian=true,
            spectrum_kind=CuDensityMat.CUDENSITYMAT_OPERATOR_SPECTRUM_LARGEST)
        @test isopen(spec)
        CuDensityMat.destroy_operator_spectrum(spec)
        @test !isopen(spec)
        close(ws)
    end

    @gpu_test "spectrum configuration" begin
        SPECTRUM_SUPPORTED || return
        ws = WorkStream()
        dims = [2]
        T = ComplexF64

        data = CUDA.CuVector{T}([1.0+0im, 0.0, 0.0, -1.0+0im])
        elem = CuDensityMat.create_elementary_operator(ws, [2], data)
        term = CuDensityMat.create_operator_term(ws, dims)
        CuDensityMat.append_elementary_product!(term, [elem], Int32[0], Int32[0])
        operator = CuDensityMat.create_operator(ws, dims)
        CuDensityMat.append_term!(operator, term; duality=0)

        spec = CuDensityMat.create_operator_spectrum(ws, operator)
        # Configure max restarts
        CuDensityMat.configure_spectrum!(ws, spec,
            CuDensityMat.CUDENSITYMAT_OPERATOR_SPECTRUM_CONFIG_MAX_RESTARTS, 30)
        # Configure max expansion
        CuDensityMat.configure_spectrum!(ws, spec,
            CuDensityMat.CUDENSITYMAT_OPERATOR_SPECTRUM_CONFIG_MAX_EXPANSION, 10)
        @test true  # no error means success

        CuDensityMat.destroy_operator_spectrum(spec)
        close(ws)
    end

    @gpu_test "eigenvalues of sigma_z (largest)" begin
        SPECTRUM_SUPPORTED || return
        ws = WorkStream()
        dims = [2]
        T = ComplexF64

        # sigma_z = diag(1, -1), eigenvalues are +1 and -1
        data = CUDA.CuVector{T}([1.0+0im, 0.0, 0.0, -1.0+0im])
        elem = CuDensityMat.create_elementary_operator(ws, [2], data)
        term = CuDensityMat.create_operator_term(ws, dims)
        CuDensityMat.append_elementary_product!(term, [elem], Int32[0], Int32[0])
        operator = CuDensityMat.create_operator(ws, dims)
        CuDensityMat.append_term!(operator, term; duality=0)

        spec = CuDensityMat.create_operator_spectrum(ws, operator;
            is_hermitian=true,
            spectrum_kind=CuDensityMat.CUDENSITYMAT_OPERATOR_SPECTRUM_LARGEST)

        # Template state for Hilbert space structure
        template = DensePureState{T}(ws, (2,); batch_size=1)
        CuDensityMat.allocate_storage!(template)

        # Prepare for 1 eigenstate
        CuDensityMat.prepare_spectrum!(ws, spec, 1, template)

        # Eigenstate to receive result
        eigenstate = DensePureState{T}(ws, (2,); batch_size=1)
        CuDensityMat.allocate_storage!(eigenstate)

        # Eigenvalue storage (real for Hermitian)
        eigenvalues = CUDA.zeros(Float64, 1)

        # Compute
        tolerances = CuDensityMat.compute_spectrum!(ws, spec, 1, [eigenstate], eigenvalues;
            time=0.0, batch_size=1)

        evals = Array(eigenvalues)
        # Largest eigenvalue of sigma_z should be +1
        @test abs(evals[1] - 1.0) < 1e-6
        @test tolerances[1] < 1e-6  # should converge well

        CuDensityMat.destroy_operator_spectrum(spec)
        close(ws)
    end

    @gpu_test "eigenvalues of sigma_z (smallest)" begin
        SPECTRUM_SUPPORTED || return
        ws = WorkStream()
        dims = [2]
        T = ComplexF64

        data = CUDA.CuVector{T}([1.0+0im, 0.0, 0.0, -1.0+0im])
        elem = CuDensityMat.create_elementary_operator(ws, [2], data)
        term = CuDensityMat.create_operator_term(ws, dims)
        CuDensityMat.append_elementary_product!(term, [elem], Int32[0], Int32[0])
        operator = CuDensityMat.create_operator(ws, dims)
        CuDensityMat.append_term!(operator, term; duality=0)

        spec = CuDensityMat.create_operator_spectrum(ws, operator;
            is_hermitian=true,
            spectrum_kind=CuDensityMat.CUDENSITYMAT_OPERATOR_SPECTRUM_SMALLEST)

        template = DensePureState{T}(ws, (2,); batch_size=1)
        CuDensityMat.allocate_storage!(template)
        CuDensityMat.prepare_spectrum!(ws, spec, 1, template)

        eigenstate = DensePureState{T}(ws, (2,); batch_size=1)
        CuDensityMat.allocate_storage!(eigenstate)
        eigenvalues = CUDA.zeros(Float64, 1)

        tolerances = CuDensityMat.compute_spectrum!(ws, spec, 1, [eigenstate], eigenvalues;
            time=0.0, batch_size=1)

        evals = Array(eigenvalues)
        # Smallest eigenvalue of sigma_z should be -1
        @test abs(evals[1] - (-1.0)) < 1e-6

        CuDensityMat.destroy_operator_spectrum(spec)
        close(ws)
    end

end
