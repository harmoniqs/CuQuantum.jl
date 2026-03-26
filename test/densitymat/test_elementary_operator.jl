# Phase 3: Elementary Operator tests (local, single-GPU)
#
# Tests dense and multidiagonal elementary operators:
# creation, lifecycle, batched creation, data attachment.
# Algebraic operations (*, +, -, dag, @) are Python-side sugar —
# at the C API level we test handle creation and correct wiring.

@testset "ElementaryOperator (local)" begin

    # ---- Dense elementary operators ----

    @gpu_test "dense elementary operator creation" begin
        ws = WorkStream()
        for T in [ComplexF64, ComplexF32, Float64, Float32]
            # Single-mode operator: d×d matrix stored flat
            d = 3
            data = CUDA.zeros(T, d * d)
            op = CuDensityMat.create_elementary_operator(ws, [d], data)
            @test isopen(op)
            @test op.handle != C_NULL
            CuDensityMat.destroy_elementary_operator(op)
            @test !isopen(op)
        end
        close(ws)
    end

    @gpu_test "dense elementary operator multi-mode" begin
        ws = WorkStream()
        # Two-mode operator: (d1*d2) × (d1*d2) but stored as d1×d2×d1×d2
        d1, d2 = 2, 3
        data = CUDA.zeros(ComplexF64, d1 * d2 * d1 * d2)
        op = CuDensityMat.create_elementary_operator(ws, [d1, d2], data)
        @test isopen(op)
        CuDensityMat.destroy_elementary_operator(op)
        close(ws)
    end

    @gpu_test "dense elementary operator batched" begin
        ws = WorkStream()
        d = 2
        batch_size = 3
        # Batched: d×d×batch_size
        data = CUDA.rand(ComplexF64, d * d * batch_size)
        op = CuDensityMat.create_elementary_operator_batch(ws, [d], data, batch_size)
        @test isopen(op)
        CuDensityMat.destroy_elementary_operator(op)
        close(ws)
    end

    @gpu_test "dense elementary operator finalizer" begin
        ws = WorkStream()
        d = 2
        data = CUDA.zeros(ComplexF64, d * d)
        handle_val = let
            op = CuDensityMat.create_elementary_operator(ws, [d], data)
            h = op.handle
            @test h != C_NULL
            h
        end
        GC.gc(true)
        @test true  # no crash
        close(ws)
    end

    # ---- Multidiagonal elementary operators ----

    @gpu_test "multidiagonal elementary operator creation" begin
        ws = WorkStream()
        d = 4
        num_diags = 3
        offsets = Int32[-1, 0, 1]  # sub-diagonal, main diagonal, super-diagonal
        # Data shape: d × num_diags (stored flat)
        data = CUDA.rand(ComplexF64, d * num_diags)
        op = CuDensityMat.create_elementary_operator(
            ws,
            [d],
            data;
            sparsity = :multidiagonal,
            diagonal_offsets = offsets,
        )
        @test isopen(op)
        CuDensityMat.destroy_elementary_operator(op)
        close(ws)
    end

    @gpu_test "multidiagonal elementary operator batched" begin
        ws = WorkStream()
        d = 4
        num_diags = 2
        batch_size = 2
        offsets = Int32[0, 1]
        data = CUDA.rand(ComplexF64, d * num_diags * batch_size)
        op = CuDensityMat.create_elementary_operator_batch(
            ws,
            [d],
            data,
            batch_size;
            sparsity = :multidiagonal,
            diagonal_offsets = offsets,
        )
        @test isopen(op)
        CuDensityMat.destroy_elementary_operator(op)
        close(ws)
    end

    # ---- MatrixOperator (full Hilbert space) ----

    @gpu_test "matrix operator creation" begin
        ws = WorkStream()
        dims = [2, 3]
        d = prod(dims)
        data = CUDA.rand(ComplexF64, d * d)
        op = CuDensityMat.create_matrix_operator(ws, dims, data)
        @test isopen(op)
        CuDensityMat.destroy_matrix_operator(op)
        close(ws)
    end

    @gpu_test "matrix operator batched" begin
        ws = WorkStream()
        dims = [2, 2]
        d = prod(dims)
        batch_size = 3
        data = CUDA.rand(ComplexF64, d * d * batch_size)
        op = CuDensityMat.create_matrix_operator_batch(ws, dims, data, batch_size)
        @test isopen(op)
        CuDensityMat.destroy_matrix_operator(op)
        close(ws)
    end

    @gpu_test "matrix operator finalizer" begin
        ws = WorkStream()
        dims = [2]
        data = CUDA.zeros(ComplexF64, 4)
        let
            op = CuDensityMat.create_matrix_operator(ws, dims, data)
            @test op.handle != C_NULL
        end
        GC.gc(true)
        @test true
        close(ws)
    end

    # ---- Multiple dtypes ----

    @gpu_test "elementary operator dtype coverage" begin
        ws = WorkStream()
        d = 2
        for T in [Float32, Float64, ComplexF32, ComplexF64]
            data = CUDA.zeros(T, d * d)
            op = CuDensityMat.create_elementary_operator(ws, [d], data)
            @test isopen(op)
            CuDensityMat.destroy_elementary_operator(op)
        end
        close(ws)
    end

end
