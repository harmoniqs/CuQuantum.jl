# Phase 9: Eigenspectrum tests
#
# cuDensityMat 0.7's extreme-eigenspectrum solver supports Hermitian operators with the
# LARGEST_REAL / SMALLEST_REAL kinds on non-batched double-precision dense pure states,
# and needs a Hilbert space that is large compared to the Krylov subspace, so the
# eigenvalue tests use a 16-level mode.

# Single-mode operator diag(-7.5, -6.5, …, 7.5) on a d = 16 Hilbert space.
const SPECTRUM_DIM = 16
const SPECTRUM_DIAG = collect(range(-7.5, 7.5; length = SPECTRUM_DIM))

function _spectrum_operator(ws, diag::AbstractVector)
    d = length(diag)
    T = ComplexF64
    m = zeros(T, d, d)
    for i = 1:d
        m[i, i] = diag[i]
    end
    elem = CuDensityMat.create_elementary_operator(ws, [d], CUDA.CuVector{T}(vec(m)))
    term = CuDensityMat.create_operator_term(ws, [d])
    CuDensityMat.append_elementary_product!(term, [elem], Int32[0], Int32[0])
    op = CuDensityMat.create_operator(ws, [d])
    CuDensityMat.append_term!(op, term; duality = 0)
    return op
end

function _extreme_eigenvalue(kind)
    T = ComplexF64
    ws = WorkStream()
    op = _spectrum_operator(ws, SPECTRUM_DIAG)
    spec = CuDensityMat.create_operator_spectrum(ws, op; spectrum_kind = kind)
    template = DensePureState{T}(ws, (SPECTRUM_DIM,); batch_size = 1)
    CuDensityMat.allocate_storage!(template)
    CuDensityMat.prepare_spectrum!(ws, spec, 1, template)
    # The initial guess must not be zero (it seeds the first Krylov block).
    eigenstate = DensePureState{T}(ws, (SPECTRUM_DIM,); batch_size = 1)
    CuDensityMat.allocate_storage!(eigenstate)
    guess = T[1 / i for i = 1:SPECTRUM_DIM]
    copyto!(eigenstate.storage, CUDA.CuVector{T}(guess ./ LinearAlgebra.norm(guess)))
    eigenvalues = CUDA.zeros(T, 1)
    residuals = CuDensityMat.compute_spectrum!(ws, spec, 1, [eigenstate], eigenvalues)
    λ = Array(eigenvalues)[1]
    ψ = Array(eigenstate.storage)
    CuDensityMat.destroy_operator_spectrum(spec)
    close(ws)
    return λ, ψ, residuals
end

@testset "Eigenspectrum (local)" begin

    @gpu_test "spectrum creation and destruction" begin
        ws = WorkStream()
        op = _spectrum_operator(ws, [1.0, -1.0])
        spec = CuDensityMat.create_operator_spectrum(
            ws,
            op;
            is_hermitian = true,
            spectrum_kind = CuDensityMat.CUDENSITYMAT_OPERATOR_SPECTRUM_LARGEST_REAL,
        )
        @test isopen(spec)
        CuDensityMat.destroy_operator_spectrum(spec)
        @test !isopen(spec)
        close(ws)
    end

    @gpu_test "spectrum configuration" begin
        ws = WorkStream()
        op = _spectrum_operator(ws, [1.0, -1.0])
        spec = CuDensityMat.create_operator_spectrum(ws, op)
        CuDensityMat.configure_spectrum!(
            ws,
            spec,
            CuDensityMat.CUDENSITYMAT_OPERATOR_SPECTRUM_CONFIG_MAX_RESTARTS,
            30,
        )
        CuDensityMat.configure_spectrum!(
            ws,
            spec,
            CuDensityMat.CUDENSITYMAT_OPERATOR_SPECTRUM_CONFIG_MAX_EXPANSION,
            10,
        )
        @test isopen(spec)
        CuDensityMat.destroy_operator_spectrum(spec)
        close(ws)
    end

    @gpu_test "unsupported spectrum configurations raise ArgumentError" begin
        ws = WorkStream()
        op = _spectrum_operator(ws, [1.0, -1.0])
        for kind in (
            CuDensityMat.CUDENSITYMAT_OPERATOR_SPECTRUM_LARGEST,
            CuDensityMat.CUDENSITYMAT_OPERATOR_SPECTRUM_SMALLEST,
        )
            @test_throws ArgumentError CuDensityMat.create_operator_spectrum(
                ws,
                op;
                spectrum_kind = kind,
            )
        end
        @test_throws ArgumentError CuDensityMat.create_operator_spectrum(
            ws,
            op;
            is_hermitian = false,
        )
        close(ws)
    end

    @gpu_test "largest real eigenvalue" begin
        λ, ψ, residuals = _extreme_eigenvalue(
            CuDensityMat.CUDENSITYMAT_OPERATOR_SPECTRUM_LARGEST_REAL,
        )
        @test real(λ) ≈ maximum(SPECTRUM_DIAG) atol = 1e-6
        @test abs(imag(λ)) < 1e-8
        @test abs(ψ[end]) ≈ 1 atol = 1e-6
        @test residuals[1] < 1e-6
    end

    @gpu_test "smallest real eigenvalue" begin
        λ, ψ, residuals = _extreme_eigenvalue(
            CuDensityMat.CUDENSITYMAT_OPERATOR_SPECTRUM_SMALLEST_REAL,
        )
        @test real(λ) ≈ minimum(SPECTRUM_DIAG) atol = 1e-6
        @test abs(imag(λ)) < 1e-8
        @test abs(ψ[1]) ≈ 1 atol = 1e-6
        @test residuals[1] < 1e-6
    end

    @gpu_test "eigenvalue buffer must match the state element type" begin
        T = ComplexF64
        ws = WorkStream()
        op = _spectrum_operator(ws, SPECTRUM_DIAG)
        spec = CuDensityMat.create_operator_spectrum(ws, op)
        template = DensePureState{T}(ws, (SPECTRUM_DIM,); batch_size = 1)
        CuDensityMat.allocate_storage!(template)
        CuDensityMat.prepare_spectrum!(ws, spec, 1, template)
        @test_throws ArgumentError CuDensityMat.compute_spectrum!(
            ws,
            spec,
            1,
            [template],
            CUDA.zeros(Float64, 1),
        )
        CuDensityMat.destroy_operator_spectrum(spec)
        close(ws)
    end

end
