# Phase 4: Composite Operator tests (local, single-GPU)
#
# Tests OperatorTerm, Operator, OperatorAction construction,
# and the full prepare/compute pipeline for operator action on states.

@testset "Composite Operators (local)" begin

    # ---- OperatorTerm ----

    @gpu_test "operator term creation" begin
        ws = WorkStream()
        dims = [2, 3, 2]
        term = CuDensityMat.create_operator_term(ws, dims)
        @test isopen(term)
        @test term.hilbert_space_dims == Int64[2, 3, 2]
        CuDensityMat.destroy_operator_term(term)
        @test !isopen(term)
        close(ws)
    end

    @gpu_test "operator term with elementary product" begin
        ws = WorkStream()
        dims = [2, 3]

        # Create two elementary operators for modes 0 and 1
        data_a = CUDA.rand(ComplexF64, 2 * 2)  # 2×2 for mode 0
        data_b = CUDA.rand(ComplexF64, 3 * 3)  # 3×3 for mode 1
        op_a = CuDensityMat.create_elementary_operator(ws, [2], data_a)
        op_b = CuDensityMat.create_elementary_operator(ws, [3], data_b)

        # Create term and append product (all ket-side: duality=0)
        term = CuDensityMat.create_operator_term(ws, dims)
        CuDensityMat.append_elementary_product!(
            term,
            [op_a, op_b],
            Int32[0, 1],
            Int32[0, 0],
        )
        @test length(term._elem_op_refs) == 2

        CuDensityMat.destroy_operator_term(term)
        CuDensityMat.destroy_elementary_operator(op_a)
        CuDensityMat.destroy_elementary_operator(op_b)
        close(ws)
    end

    @gpu_test "operator term with ket and bra products" begin
        ws = WorkStream()
        dims = [2, 2]
        data = CUDA.rand(ComplexF64, 2 * 2)
        op = CuDensityMat.create_elementary_operator(ws, [2], data)

        term = CuDensityMat.create_operator_term(ws, dims)
        # Append ket-side (duality=0) on mode 0
        CuDensityMat.append_elementary_product!(term, [op], Int32[0], Int32[0])
        # Append bra-side (duality=1) on mode 1
        CuDensityMat.append_elementary_product!(term, [op], Int32[1], Int32[1])
        @test length(term._elem_op_refs) == 2  # two appends

        CuDensityMat.destroy_operator_term(term)
        CuDensityMat.destroy_elementary_operator(op)
        close(ws)
    end

    @gpu_test "operator term with matrix product" begin
        ws = WorkStream()
        dims = [2, 3]
        d = prod(dims)  # 6
        data = CUDA.rand(ComplexF64, d * d)
        mat_op = CuDensityMat.create_matrix_operator(ws, dims, data)

        term = CuDensityMat.create_operator_term(ws, dims)
        # Single matrix operator, no conjugation, ket-side
        CuDensityMat.append_matrix_product!(term, [mat_op], Int32[0], Int32[0])
        @test length(term._matrix_op_refs) == 1

        CuDensityMat.destroy_operator_term(term)
        CuDensityMat.destroy_matrix_operator(mat_op)
        close(ws)
    end

    # ---- Operator ----

    @gpu_test "operator creation" begin
        ws = WorkStream()
        dims = [2, 3]
        op = CuDensityMat.create_operator(ws, dims)
        @test isopen(op)
        @test op.hilbert_space_dims == Int64[2, 3]
        CuDensityMat.destroy_operator(op)
        @test !isopen(op)
        close(ws)
    end

    @gpu_test "operator with terms" begin
        ws = WorkStream()
        dims = [2, 3]

        # Build elementary ops
        data_a = CUDA.rand(ComplexF64, 4)  # 2×2
        data_b = CUDA.rand(ComplexF64, 9)  # 3×3
        op_a = CuDensityMat.create_elementary_operator(ws, [2], data_a)
        op_b = CuDensityMat.create_elementary_operator(ws, [3], data_b)

        # Build term: A⊗B (both ket-side)
        term1 = CuDensityMat.create_operator_term(ws, dims)
        CuDensityMat.append_elementary_product!(
            term1,
            [op_a, op_b],
            Int32[0, 1],
            Int32[0, 0],
        )

        # Build term: A on mode 0 (bra-side)
        term2 = CuDensityMat.create_operator_term(ws, dims)
        CuDensityMat.append_elementary_product!(term2, [op_a], Int32[0], Int32[1])

        # Assemble operator
        operator = CuDensityMat.create_operator(ws, dims)
        CuDensityMat.append_term!(operator, term1; duality = 0, coefficient = 1.0)
        CuDensityMat.append_term!(operator, term2; duality = 1, coefficient = 0.5 + 0.1im)
        @test length(operator._term_refs) == 2

        CuDensityMat.destroy_operator(operator)
        close(ws)
    end

    @gpu_test "operator with no coefficient (defaults)" begin
        ws = WorkStream()
        dims = [2]
        data = CUDA.rand(ComplexF64, 4)
        elem = CuDensityMat.create_elementary_operator(ws, [2], data)
        term = CuDensityMat.create_operator_term(ws, dims)
        CuDensityMat.append_elementary_product!(term, [elem], Int32[0], Int32[0])

        operator = CuDensityMat.create_operator(ws, dims)
        CuDensityMat.append_term!(operator, term)  # default coefficient=1.0+0im
        @test length(operator._term_refs) == 1

        CuDensityMat.destroy_operator(operator)
        close(ws)
    end

    # ---- Operator Action (full pipeline) ----

    @gpu_test "single-operator action pipeline" begin
        ws = WorkStream()
        dims = [2]
        T = ComplexF64

        # Build a simple operator: sigma_z = diag(1, -1)
        sigma_z_data = CUDA.CuVector{T}([1.0 + 0im, 0.0, 0.0, -1.0 + 0im])
        elem = CuDensityMat.create_elementary_operator(ws, [2], sigma_z_data)

        term = CuDensityMat.create_operator_term(ws, dims)
        CuDensityMat.append_elementary_product!(term, [elem], Int32[0], Int32[0])

        operator = CuDensityMat.create_operator(ws, dims)
        CuDensityMat.append_term!(operator, term; duality = 0, coefficient = 1.0)

        # Create input and output states
        psi_in = DenseMixedState{T}(ws, (2,); batch_size = 1)
        psi_out = DenseMixedState{T}(ws, (2,); batch_size = 1)
        CuDensityMat.allocate_storage!(psi_in)
        CuDensityMat.allocate_storage!(psi_out)

        # Set input to identity/2: rho = [[0.5, 0], [0, 0.5]]
        copyto!(psi_in.storage, CUDA.CuVector{T}([0.5, 0.0, 0.0, 0.5]))

        # Prepare action
        CuDensityMat.prepare_operator_action!(ws, operator, psi_in, psi_out)

        # Zero output state, then compute
        CuDensityMat.initialize_zero!(psi_out)
        CuDensityMat.compute_operator_action!(
            ws,
            operator,
            psi_in,
            psi_out;
            time = 0.0,
            batch_size = 1,
        )

        # Verify output is non-zero (sigma_z * rho should be non-trivial)
        result = Array(psi_out.storage)
        @test any(x -> abs(x) > 0, result)

        close(ws)
    end

    @testset "compute action (batch_size=$bs)" for bs in TEST_BATCH_SIZES
        @gpu_test "sigma_z action batch_size=$bs" begin
            ws = WorkStream()
            dims = [2]
            T = ComplexF64

            # Build a simple operator: sigma_z = diag(1, -1)
            sigma_z_data = CUDA.CuVector{T}([1.0 + 0im, 0.0, 0.0, -1.0 + 0im])
            elem = CuDensityMat.create_elementary_operator(ws, [2], sigma_z_data)

            term = CuDensityMat.create_operator_term(ws, dims)
            CuDensityMat.append_elementary_product!(term, [elem], Int32[0], Int32[0])

            operator = CuDensityMat.create_operator(ws, dims)
            CuDensityMat.append_term!(operator, term; duality = 0, coefficient = 1.0)

            # Create batched input and output states
            psi_in = DenseMixedState{T}(ws, (2,); batch_size = bs)
            psi_out = DenseMixedState{T}(ws, (2,); batch_size = bs)
            CuDensityMat.allocate_storage!(psi_in)
            CuDensityMat.allocate_storage!(psi_out)

            # Set each batch entry to identity/2: rho = [[0.5, 0], [0, 0.5]]
            rho_single = T[0.5, 0.0, 0.0, 0.5]
            copyto!(psi_in.storage, CUDA.CuVector{T}(repeat(rho_single, bs)))

            # Prepare and compute action
            CuDensityMat.prepare_operator_action!(ws, operator, psi_in, psi_out)
            CuDensityMat.initialize_zero!(psi_out)
            CuDensityMat.compute_operator_action!(
                ws,
                operator,
                psi_in,
                psi_out;
                time = 0.0,
                batch_size = bs,
            )

            # Verify output is non-zero (sigma_z * rho should be non-trivial)
            result = sync_and_pull(psi_out.storage)
            @test any(x -> abs(x) > 0, result)

            close(ws)
        end
    end

    @gpu_test "multi-operator action pipeline" begin
        ws = WorkStream()
        dims = [2]
        T = ComplexF64

        # Two operators
        sigma_z_data = CUDA.CuVector{T}([1.0 + 0im, 0.0, 0.0, -1.0 + 0im])
        sigma_x_data = CUDA.CuVector{T}([0.0 + 0im, 1.0, 1.0, 0.0 + 0im])

        elem_z = CuDensityMat.create_elementary_operator(ws, [2], sigma_z_data)
        elem_x = CuDensityMat.create_elementary_operator(ws, [2], sigma_x_data)

        term_z = CuDensityMat.create_operator_term(ws, dims)
        CuDensityMat.append_elementary_product!(term_z, [elem_z], Int32[0], Int32[0])
        term_x = CuDensityMat.create_operator_term(ws, dims)
        CuDensityMat.append_elementary_product!(term_x, [elem_x], Int32[0], Int32[0])

        op1 = CuDensityMat.create_operator(ws, dims)
        CuDensityMat.append_term!(op1, term_z; duality = 0)
        op2 = CuDensityMat.create_operator(ws, dims)
        CuDensityMat.append_term!(op2, term_x; duality = 0)

        # Create action with two operators
        action = CuDensityMat.create_operator_action(ws, [op1, op2])
        @test isopen(action)

        # Input states (one per operator)
        psi_in1 = DenseMixedState{T}(ws, (2,); batch_size = 1)
        psi_in2 = DenseMixedState{T}(ws, (2,); batch_size = 1)
        psi_out = DenseMixedState{T}(ws, (2,); batch_size = 1)
        CuDensityMat.allocate_storage!(psi_in1)
        CuDensityMat.allocate_storage!(psi_in2)
        CuDensityMat.allocate_storage!(psi_out)
        copyto!(psi_in1.storage, CUDA.CuVector{T}([0.5, 0.0, 0.0, 0.5]))
        copyto!(psi_in2.storage, CUDA.CuVector{T}([0.5, 0.0, 0.0, 0.5]))

        # Prepare action
        CuDensityMat.prepare_action!(ws, action, [psi_in1, psi_in2], psi_out)

        # Zero output state, then compute
        CuDensityMat.initialize_zero!(psi_out)
        CuDensityMat.compute_action!(
            ws,
            action,
            [psi_in1, psi_in2],
            psi_out;
            time = 0.0,
            batch_size = 1,
        )

        result = Array(psi_out.storage)
        @test any(x -> abs(x) > 0, result)

        CuDensityMat.destroy_operator_action(action)
        close(ws)
    end

    @gpu_test "operator action finalizer cleanup" begin
        ws = WorkStream()
        dims = [2]
        data = CUDA.zeros(ComplexF64, 4)
        elem = CuDensityMat.create_elementary_operator(ws, [2], data)
        term = CuDensityMat.create_operator_term(ws, dims)
        CuDensityMat.append_elementary_product!(term, [elem], Int32[0], Int32[0])
        operator = CuDensityMat.create_operator(ws, dims)
        CuDensityMat.append_term!(operator, term)

        let
            action = CuDensityMat.create_operator_action(ws, [operator])
            @test action.handle != C_NULL
        end
        GC.gc(true)
        @test true
        close(ws)
    end

end
