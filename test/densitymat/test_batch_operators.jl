@testset "Batch operator API" begin

    @testset "create_elementary_operator_batch" begin
        @gpu_test "batch elementary operator construction and lifecycle" begin
            ws = WorkStream()
            hilbert_dims = [2]
            batch_size = 4
            # 4 identical sigma_z operators stacked as (2, 2, 4) CuArray
            sigma_z = ComplexF64[1.0 0.0; 0.0 -1.0]
            data = CUDA.CuArray(cat([sigma_z for _ in 1:batch_size]...; dims=3))
            op = create_elementary_operator_batch(ws, hilbert_dims, data, batch_size)
            @test op isa ElementaryOperator
            @test isopen(op)
            close(op)
            @test !isopen(op)
            close(ws)
        end
    end

    @testset "append_elementary_product_batch!" begin
        @gpu_test "batch term append with static coefficients" begin
            ws = WorkStream()
            hilbert_dims = [2]
            batch_size = 2
            sigma_z = ComplexF64[1.0 0.0; 0.0 -1.0]
            data = CUDA.CuArray(cat([sigma_z for _ in 1:batch_size]...; dims=3))
            op = create_elementary_operator_batch(ws, hilbert_dims, data, batch_size)
            term = create_operator_term(ws, Int64.(hilbert_dims))
            static_coeffs = CUDA.CuArray(ComplexF64[1.0, 1.0])
            append_elementary_product_batch!(
                term, [op], Int32[0], Int32[0], batch_size, static_coeffs
            )
            @test isopen(term)
            close(term); close(op); close(ws)
        end
    end

    @testset "append_term_batch!" begin
        @gpu_test "batch term append to operator" begin
            ws = WorkStream()
            hilbert_dims = [2]
            batch_size = 2
            sigma_z = ComplexF64[1.0 0.0; 0.0 -1.0]
            data = CUDA.CuArray(cat([sigma_z for _ in 1:batch_size]...; dims=3))
            elem_op = create_elementary_operator_batch(ws, hilbert_dims, data, batch_size)
            term = create_operator_term(ws, Int64.(hilbert_dims))
            static_coeffs = CUDA.CuArray(ComplexF64[1.0, 1.0])
            append_elementary_product_batch!(
                term, [elem_op], Int32[0], Int32[0], batch_size, static_coeffs
            )
            op = create_operator(ws, Int64.(hilbert_dims))
            batch_term_coeffs = CUDA.CuArray(ComplexF64[1.0, 1.0])
            append_term_batch!(op, term, batch_size, batch_term_coeffs)
            @test isopen(op)
            close(op); close(term); close(elem_op); close(ws)
        end
    end

end
