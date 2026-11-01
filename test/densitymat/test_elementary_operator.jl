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
        @test isopen(ws)  # no crash, WorkStream still usable after the finalizer
        close(ws)
    end

    # Batch layout: operator data is (d, d, batch) in F-order and batch member b acts
    # on state b (σz on |+⟩, σx on |0⟩ — swapping the members changes the result).
    @gpu_test "batched elementary operator acts per batch member" begin
        T = ComplexF64
        ws = WorkStream()
        dims = [2]
        σz = T[1 0; 0 -1]
        σx = T[0 1; 1 0]
        elem = CuDensityMat.create_elementary_operator_batch(
            ws,
            [2],
            CUDA.CuVector{T}([vec(σz); vec(σx)]),
            2,
        )
        term = CuDensityMat.create_operator_term(ws, dims)
        CuDensityMat.append_elementary_product!(term, [elem], Int32[0], Int32[0])
        op = CuDensityMat.create_operator(ws, dims)
        CuDensityMat.append_term!(op, term; duality = 0)
        ψin = DensePureState{T}(ws, (2,); batch_size = 2)
        ψout = DensePureState{T}(ws, (2,); batch_size = 2)
        CuDensityMat.allocate_storage!(ψin)
        CuDensityMat.allocate_storage!(ψout)
        ψ = (T[1, 1] / sqrt(2), T[1, 0])
        copyto!(ψin.storage, CUDA.CuVector{T}([ψ[1]; ψ[2]]))
        CuDensityMat.prepare_operator_action!(ws, op, ψin, ψout)
        CuDensityMat.initialize_zero!(ψout)
        CuDensityMat.compute_operator_action!(ws, op, ψin, ψout)
        out = reshape(Array(ψout.storage), 2, 2)
        @test out[:, 1] ≈ σz * ψ[1] rtol = 1e-12
        @test out[:, 2] ≈ σx * ψ[2] rtol = 1e-12
        @test !isapprox(out[:, 1], σx * ψ[1]; rtol = 0.1) # negative control
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
        @test isopen(ws)
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

    # ---- Buffer re-attachment (cuDensityMat >= 0.7) ----

    @testset "attach_buffer! is defined for elementary and matrix operators" begin
        @test hasmethod(
            CuDensityMat.attach_buffer!,
            Tuple{CuDensityMat.ElementaryOperator,CUDA.CuArray},
        )
        @test hasmethod(
            CuDensityMat.attach_buffer!,
            Tuple{CuDensityMat.MatrixOperator,CUDA.CuArray},
        )
    end

    @gpu_test "attach_buffer! swaps elementary operator data" begin
        T = ComplexF64
        ws = WorkStream()
        dims = [2]
        elem = CuDensityMat.create_elementary_operator(
            ws,
            [2],
            CUDA.CuVector{T}([1, 0, 0, -1]),
        )
        sx = CUDA.CuVector{T}([0, 1, 1, 0])
        CuDensityMat.attach_buffer!(elem, sx)
        @test elem._data_ref === sx
        term = CuDensityMat.create_operator_term(ws, dims)
        CuDensityMat.append_elementary_product!(term, [elem], Int32[0], Int32[0])
        op = CuDensityMat.create_operator(ws, dims)
        CuDensityMat.append_term!(op, term; duality = 0)
        ψin = DensePureState{T}(ws, (2,); batch_size = 1)
        ψout = DensePureState{T}(ws, (2,); batch_size = 1)
        CuDensityMat.allocate_storage!(ψin)
        CuDensityMat.allocate_storage!(ψout)
        copyto!(ψin.storage, CUDA.CuVector{T}([1, 0]))
        CuDensityMat.prepare_operator_action!(ws, op, ψin, ψout)
        CuDensityMat.initialize_zero!(ψout)
        CuDensityMat.compute_operator_action!(ws, op, ψin, ψout)
        @test Array(ψout.storage) ≈ T[0, 1]
        close(ws)
    end

    @gpu_test "attach_buffer! after prepare is used by the next compute" begin
        T = ComplexF64
        ws = WorkStream()
        dims = [2]
        elem = CuDensityMat.create_elementary_operator(ws, [2], CUDA.CuVector{T}([1, 0, 0, -1]))
        term = CuDensityMat.create_operator_term(ws, dims)
        CuDensityMat.append_elementary_product!(term, [elem], Int32[0], Int32[0])
        op = CuDensityMat.create_operator(ws, dims)
        CuDensityMat.append_term!(op, term; duality = 0)
        ψin = DensePureState{T}(ws, (2,); batch_size = 1)
        ψout = DensePureState{T}(ws, (2,); batch_size = 1)
        CuDensityMat.allocate_storage!(ψin)
        CuDensityMat.allocate_storage!(ψout)
        copyto!(ψin.storage, CUDA.CuVector{T}([1, 0]))
        CuDensityMat.prepare_operator_action!(ws, op, ψin, ψout)
        CuDensityMat.initialize_zero!(ψout)
        CuDensityMat.compute_operator_action!(ws, op, ψin, ψout)
        @test Array(ψout.storage) ≈ T[1, 0]
        CuDensityMat.attach_buffer!(elem, CUDA.CuVector{T}([0, 1, 1, 0]))
        CuDensityMat.initialize_zero!(ψout)
        CuDensityMat.compute_operator_action!(ws, op, ψin, ψout)
        @test Array(ψout.storage) ≈ T[0, 1]
        close(ws)
    end

    @gpu_test "attach_buffer! on a matrix operator" begin
        T = ComplexF64
        ws = WorkStream()
        op = CuDensityMat.create_matrix_operator(ws, [2], CUDA.zeros(T, 4))
        buf = CUDA.CuVector{T}([0, 1, 1, 0])
        CuDensityMat.attach_buffer!(op, buf)
        @test op._data_ref === buf
        @test_throws ArgumentError CuDensityMat.attach_buffer!(op, CUDA.zeros(T, 3))
        close(ws)
    end

end
