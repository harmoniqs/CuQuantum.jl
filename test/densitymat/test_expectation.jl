# Phase 8: Expectation value tests

@testset "Expectation Values (local)" begin

    @gpu_test "expectation value creation and destruction" begin
        ws = WorkStream()
        dims = [2]
        T = ComplexF64

        # Build sigma_z operator
        data = CUDA.CuVector{T}([1.0+0im, 0.0, 0.0, -1.0+0im])
        elem = CuDensityMat.create_elementary_operator(ws, [2], data)
        term = CuDensityMat.create_operator_term(ws, dims)
        CuDensityMat.append_elementary_product!(term, [elem], Int32[0], Int32[0])
        operator = CuDensityMat.create_operator(ws, dims)
        CuDensityMat.append_term!(operator, term; duality=0)

        exp = CuDensityMat.create_expectation(ws, operator)
        @test isopen(exp)
        CuDensityMat.destroy_expectation(exp)
        @test !isopen(exp)
        close(ws)
    end

    @gpu_test "expectation value of sigma_z on |0><0|" begin
        ws = WorkStream()
        dims = [2]
        T = ComplexF64

        # sigma_z = diag(1, -1)
        data = CUDA.CuVector{T}([1.0+0im, 0.0, 0.0, -1.0+0im])
        elem = CuDensityMat.create_elementary_operator(ws, [2], data)
        term = CuDensityMat.create_operator_term(ws, dims)
        CuDensityMat.append_elementary_product!(term, [elem], Int32[0], Int32[0])
        operator = CuDensityMat.create_operator(ws, dims)
        CuDensityMat.append_term!(operator, term; duality=0)

        # State: |0><0| = [[1,0],[0,0]]
        rho = DenseMixedState{T}(ws, (2,); batch_size=1)
        CuDensityMat.allocate_storage!(rho)
        copyto!(rho.storage, CUDA.CuVector{T}([1.0, 0.0, 0.0, 0.0]))

        # Create expectation, prepare, compute
        exp = CuDensityMat.create_expectation(ws, operator)
        CuDensityMat.prepare_expectation!(ws, exp, rho)
        result = CUDA.zeros(T, 1)
        CuDensityMat.compute_expectation!(ws, exp, rho, result;
            time=0.0, batch_size=1)

        # Tr(sigma_z * |0><0|) = Tr(diag(1,-1) * diag(1,0)) = 1
        val = Array(result)
        @test abs(real(val[1]) - 1.0) < 1e-10
        @test abs(imag(val[1])) < 1e-10

        CuDensityMat.destroy_expectation(exp)
        close(ws)
    end

    @gpu_test "expectation value of sigma_z on |1><1|" begin
        ws = WorkStream()
        dims = [2]
        T = ComplexF64

        # sigma_z
        data = CUDA.CuVector{T}([1.0+0im, 0.0, 0.0, -1.0+0im])
        elem = CuDensityMat.create_elementary_operator(ws, [2], data)
        term = CuDensityMat.create_operator_term(ws, dims)
        CuDensityMat.append_elementary_product!(term, [elem], Int32[0], Int32[0])
        operator = CuDensityMat.create_operator(ws, dims)
        CuDensityMat.append_term!(operator, term; duality=0)

        # State: |1><1| = [[0,0],[0,1]]
        rho = DenseMixedState{T}(ws, (2,); batch_size=1)
        CuDensityMat.allocate_storage!(rho)
        copyto!(rho.storage, CUDA.CuVector{T}([0.0, 0.0, 0.0, 1.0]))

        exp = CuDensityMat.create_expectation(ws, operator)
        CuDensityMat.prepare_expectation!(ws, exp, rho)
        result = CUDA.zeros(T, 1)
        CuDensityMat.compute_expectation!(ws, exp, rho, result;
            time=0.0, batch_size=1)

        # Tr(sigma_z * |1><1|) = Tr(diag(1,-1) * diag(0,1)) = -1
        val = Array(result)
        @test abs(real(val[1]) - (-1.0)) < 1e-10
        @test abs(imag(val[1])) < 1e-10

        CuDensityMat.destroy_expectation(exp)
        close(ws)
    end

    @gpu_test "expectation value of identity on mixed state" begin
        ws = WorkStream()
        dims = [2]
        T = ComplexF64

        # identity operator = diag(1, 1)
        data = CUDA.CuVector{T}([1.0+0im, 0.0, 0.0, 1.0+0im])
        elem = CuDensityMat.create_elementary_operator(ws, [2], data)
        term = CuDensityMat.create_operator_term(ws, dims)
        CuDensityMat.append_elementary_product!(term, [elem], Int32[0], Int32[0])
        operator = CuDensityMat.create_operator(ws, dims)
        CuDensityMat.append_term!(operator, term; duality=0)

        # State: I/2 = [[0.5,0],[0,0.5]]
        rho = DenseMixedState{T}(ws, (2,); batch_size=1)
        CuDensityMat.allocate_storage!(rho)
        copyto!(rho.storage, CUDA.CuVector{T}([0.5, 0.0, 0.0, 0.5]))

        exp = CuDensityMat.create_expectation(ws, operator)
        CuDensityMat.prepare_expectation!(ws, exp, rho)
        result = CUDA.zeros(T, 1)
        CuDensityMat.compute_expectation!(ws, exp, rho, result;
            time=0.0, batch_size=1)

        # Tr(I * I/2) = Tr(I/2) = 1
        val = Array(result)
        @test abs(real(val[1]) - 1.0) < 1e-10

        CuDensityMat.destroy_expectation(exp)
        close(ws)
    end

end
