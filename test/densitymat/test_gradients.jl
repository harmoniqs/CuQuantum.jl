# Phase 7: Backward differentiation tests
#
# Tests the gradient/backward differentiation pipeline:
# prepare_backward, compute_backward, parameter gradients.
# Single-GPU only (no MPI backward diff support yet from NVIDIA).

@testset "Gradients (local)" begin

    @gpu_test "backward diff prepare + compute" begin
        ws = WorkStream()
        dims = [2]
        T = ComplexF64

        # Build operator with time-dependent coefficient
        # H = f(t) * sigma_z, where f(t) = exp(iΩt), Ω = params[1]
        sigma_z_data = CUDA.CuVector{T}([1.0+0im, 0.0, 0.0, -1.0+0im])
        elem = CuDensityMat.create_elementary_operator(ws, [2], sigma_z_data)

        term = CuDensityMat.create_operator_term(ws, dims)
        CuDensityMat.append_elementary_product!(term, [elem], Int32[0], Int32[0])

        function td_coeff(time, params, storage)
            for b in eachindex(storage)
                ω = params[1, b]
                storage[b] = complex(cos(ω * time), sin(ω * time))
            end
        end
        function td_grad(time, params, scalar_grad, params_grad)
            for b in axes(scalar_grad, 1)
                ω = params[1, b]
                # d/dΩ [exp(iΩt)] = i*t*exp(iΩt) → real/imag derivatives
                scalar_grad[b] = complex(-time * sin(ω * time), time * cos(ω * time))
                params_grad[1, b] = 1.0  # placeholder
            end
        end
        cb, gcb, cb_refs = CuDensityMat.wrap_scalar_callback(td_coeff; gradient = td_grad)

        operator = CuDensityMat.create_operator(ws, dims)
        CuDensityMat.append_term!(
            operator,
            term;
            duality = 0,
            coefficient = 1.0+0im,
            coefficient_callback = cb,
            coefficient_gradient_callback = gcb,
        )

        # Forward pass states
        psi_in = DenseMixedState{T}(ws, (2,); batch_size = 1)
        psi_out = DenseMixedState{T}(ws, (2,); batch_size = 1)
        CuDensityMat.allocate_storage!(psi_in)
        CuDensityMat.allocate_storage!(psi_out)
        copyto!(psi_in.storage, CUDA.CuVector{T}([1.0, 0.0, 0.0, 0.0]))

        # Forward prepare + compute
        CuDensityMat.prepare_operator_action!(ws, operator, psi_in, psi_out)
        CuDensityMat.initialize_zero!(psi_out)
        params = CUDA.CuVector{Float64}([2.0])
        CuDensityMat.compute_operator_action!(
            ws,
            operator,
            psi_in,
            psi_out;
            time = 0.3,
            batch_size = 1,
            num_params = 1,
            params = params,
        )

        # Backward pass states
        psi_in_adj = DenseMixedState{T}(ws, (2,); batch_size = 1)
        CuDensityMat.allocate_storage!(psi_in_adj)
        params_grad = CUDA.zeros(Float64, 1)

        # Use output state as adjoint of output (simplified, like C++ sample)
        # Backward prepare
        CuDensityMat.prepare_operator_action_backward!(ws, operator, psi_in, psi_out)

        # Zero adjoint state and param grads, then compute backward
        CuDensityMat.initialize_zero!(psi_in_adj)
        params_grad .= 0.0
        CuDensityMat.compute_operator_action_backward!(
            ws,
            operator,
            psi_in,
            psi_out,
            psi_in_adj,
            params_grad;
            time = 0.3,
            batch_size = 1,
            num_params = 1,
            params = params,
        )

        # Check outputs are non-zero
        adj_result = Array(psi_in_adj.storage)
        @test any(x -> abs(x) > 0, adj_result)

        grad_result = Array(params_grad)
        @test any(x -> abs(x) > 0, grad_result)

        CuDensityMat.unregister_callback!(cb_refs)
        close(ws)
    end

    # Note: backward diff without callbacks/params is not supported by the C API
    # (requires numParams >= 1 with gradient callbacks). This is confirmed by
    # NVIDIA's C++ samples which always use at least 1 parameter.

end
