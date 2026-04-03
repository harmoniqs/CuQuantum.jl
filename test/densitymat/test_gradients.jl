# Phase 7: Backward differentiation tests
#
# Tests the gradient/backward differentiation pipeline:
# prepare_backward, compute_backward, parameter gradients.
# Single-GPU only (no MPI backward diff support yet from NVIDIA).
#
# The backward pass computes the VJP (vector-Jacobian product):
#   state_in_adj  +=  A(t,p)^H * state_out_adj
#   params_grad[n] += 2 * Re(adjoint_Q * dQ/dp[n])  (via user gradient callback)

@testset "Gradients (local)" begin

    # =========================================================================
    # Test 1: Smoke test — backward produces non-zero outputs
    # =========================================================================

    @gpu_test "backward diff prepare + compute" begin
        ws = WorkStream()
        dims = [2]
        T = ComplexF64

        # Build operator with time-dependent coefficient
        # H = f(t) * sigma_z, where f(t) = exp(iΩt), Ω = params[1]
        sigma_z_data = CUDA.CuVector{T}([1.0 + 0im, 0.0, 0.0, -1.0 + 0im])
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
            coefficient = 1.0 + 0im,
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

    # =========================================================================
    # Test 2: Finite-difference verification of params_grad
    #
    # Operator: A(θ) = θ * σ_z  (ket-side only, duality=0)
    # Forward: ρ_out = θ * σ_z * ρ_in
    # Cost: c(θ) = Re(Tr(ρ_out_adj† * ρ_out)) = Re(⟨ρ_out_adj, ρ_out⟩)
    # ∂c/∂θ via finite differences vs backward
    # =========================================================================

    @gpu_test "params_grad matches finite differences (scalar coeff)" begin
        ws = WorkStream()
        dims = [2]
        T = ComplexF64

        # Elementary operator: σ_z
        sigma_z_data = CUDA.CuVector{T}([1.0 + 0im, 0.0, 0.0, -1.0 + 0im])
        elem = CuDensityMat.create_elementary_operator(ws, [2], sigma_z_data)

        term = CuDensityMat.create_operator_term(ws, dims)
        CuDensityMat.append_elementary_product!(term, [elem], Int32[0], Int32[0])

        # Coefficient callback: f(t, params) = params[1]  (just θ)
        function coeff_fn(time, params, storage)
            for b in eachindex(storage)
                storage[b] = complex(params[1, b], 0.0)
            end
        end
        # Gradient callback: df/dθ = 1, so params_grad += 2*Re(scalar_grad * 1)
        function grad_fn(time, params, scalar_grad, params_grad)
            for b in axes(scalar_grad, 1)
                params_grad[1, b] += real(scalar_grad[b])
            end
        end

        cb, gcb, cb_refs = CuDensityMat.wrap_scalar_callback(coeff_fn; gradient = grad_fn)

        operator = CuDensityMat.create_operator(ws, dims)
        CuDensityMat.append_term!(
            operator,
            term;
            duality = 0,
            coefficient = 1.0 + 0im,
            coefficient_callback = cb,
            coefficient_gradient_callback = gcb,
        )

        # States
        rho_in = DenseMixedState{T}(ws, (2,); batch_size = 1)
        rho_out = DenseMixedState{T}(ws, (2,); batch_size = 1)
        CuDensityMat.allocate_storage!(rho_in)
        CuDensityMat.allocate_storage!(rho_out)

        # Input: ρ = [0.7, 0.3i; -0.3i, 0.3]  (valid density matrix)
        rho_in_data = T[0.7, -0.3im, 0.3im, 0.3]
        copyto!(rho_in.storage, CUDA.CuVector{T}(rho_in_data))

        # Adjoint of output: pick a fixed "seed" for the VJP
        rho_out_adj_data = T[1.0, 0.2 + 0.1im, 0.2 - 0.1im, 0.5]

        θ₀ = 1.5

        # --- Forward at θ₀ ---
        CuDensityMat.prepare_operator_action!(ws, operator, rho_in, rho_out)
        CuDensityMat.initialize_zero!(rho_out)
        params_vec = CUDA.CuVector{Float64}([θ₀])
        CuDensityMat.compute_operator_action!(
            ws,
            operator,
            rho_in,
            rho_out;
            time = 0.0,
            batch_size = 1,
            num_params = 1,
            params = params_vec,
        )
        rho_out_result = Array(rho_out.storage)

        # Cost = Re(dot(rho_out_adj, rho_out)) = Re(Σ conj(adj_i) * out_i)
        cost_at_θ₀ = real(dot(rho_out_adj_data, rho_out_result))

        # --- Backward pass ---
        rho_out_adj = DenseMixedState{T}(ws, (2,); batch_size = 1)
        CuDensityMat.allocate_storage!(rho_out_adj)
        copyto!(rho_out_adj.storage, CUDA.CuVector{T}(rho_out_adj_data))

        rho_in_adj = DenseMixedState{T}(ws, (2,); batch_size = 1)
        CuDensityMat.allocate_storage!(rho_in_adj)
        params_grad = CUDA.zeros(Float64, 1)

        CuDensityMat.prepare_operator_action_backward!(ws, operator, rho_in, rho_out_adj)
        CuDensityMat.initialize_zero!(rho_in_adj)
        params_grad .= 0.0
        CuDensityMat.compute_operator_action_backward!(
            ws,
            operator,
            rho_in,
            rho_out_adj,
            rho_in_adj,
            params_grad;
            time = 0.0,
            batch_size = 1,
            num_params = 1,
            params = params_vec,
        )

        backward_grad = Array(params_grad)[1]

        # --- Finite difference ---
        ε = 1.0e-6
        function cost_at(θ)
            p = CUDA.CuVector{Float64}([θ])
            CuDensityMat.initialize_zero!(rho_out)
            CuDensityMat.compute_operator_action!(
                ws,
                operator,
                rho_in,
                rho_out;
                time = 0.0,
                batch_size = 1,
                num_params = 1,
                params = p,
            )
            out = Array(rho_out.storage)
            return real(dot(rho_out_adj_data, out))
        end
        fd_grad = (cost_at(θ₀ + ε) - cost_at(θ₀ - ε)) / (2ε)

        @test backward_grad ≈ fd_grad rtol = 1.0e-4

        CuDensityMat.unregister_callback!(cb_refs)
        close(ws)
    end

    # =========================================================================
    # Test 3: Finite-difference verification of state_in_adj
    #
    # Verify that state_in_adj = A(θ)^H * state_out_adj by comparing
    # each element against finite differences on the cost.
    # =========================================================================

    @gpu_test "state_in_adj matches finite differences" begin
        ws = WorkStream()
        dims = [2]
        T = ComplexF64

        # Operator: A = θ * σ_z (same setup as above)
        sigma_z_data = CUDA.CuVector{T}([1.0 + 0im, 0.0, 0.0, -1.0 + 0im])
        elem = CuDensityMat.create_elementary_operator(ws, [2], sigma_z_data)
        term = CuDensityMat.create_operator_term(ws, dims)
        CuDensityMat.append_elementary_product!(term, [elem], Int32[0], Int32[0])

        function coeff_fn(time, params, storage)
            for b in eachindex(storage)
                storage[b] = complex(params[1, b], 0.0)
            end
        end
        function grad_fn(time, params, scalar_grad, params_grad)
            for b in axes(scalar_grad, 1)
                params_grad[1, b] += real(scalar_grad[b])
            end
        end
        cb, gcb, cb_refs = CuDensityMat.wrap_scalar_callback(coeff_fn; gradient = grad_fn)

        operator = CuDensityMat.create_operator(ws, dims)
        CuDensityMat.append_term!(
            operator,
            term;
            duality = 0,
            coefficient = 1.0 + 0im,
            coefficient_callback = cb,
            coefficient_gradient_callback = gcb,
        )

        D = 2
        θ₀ = 1.5
        rho_in_data = T[0.7, -0.3im, 0.3im, 0.3]
        rho_out_adj_data = T[1.0, 0.2 + 0.1im, 0.2 - 0.1im, 0.5]

        # States
        rho_in = DenseMixedState{T}(ws, (2,); batch_size = 1)
        rho_out = DenseMixedState{T}(ws, (2,); batch_size = 1)
        CuDensityMat.allocate_storage!(rho_in)
        CuDensityMat.allocate_storage!(rho_out)
        copyto!(rho_in.storage, CUDA.CuVector{T}(rho_in_data))

        rho_out_adj = DenseMixedState{T}(ws, (2,); batch_size = 1)
        CuDensityMat.allocate_storage!(rho_out_adj)
        copyto!(rho_out_adj.storage, CUDA.CuVector{T}(rho_out_adj_data))

        rho_in_adj = DenseMixedState{T}(ws, (2,); batch_size = 1)
        CuDensityMat.allocate_storage!(rho_in_adj)
        params_grad = CUDA.zeros(Float64, 1)
        params_vec = CUDA.CuVector{Float64}([θ₀])

        # Forward + backward
        CuDensityMat.prepare_operator_action!(ws, operator, rho_in, rho_out)
        CuDensityMat.initialize_zero!(rho_out)
        CuDensityMat.compute_operator_action!(
            ws,
            operator,
            rho_in,
            rho_out;
            time = 0.0,
            batch_size = 1,
            num_params = 1,
            params = params_vec,
        )

        CuDensityMat.prepare_operator_action_backward!(ws, operator, rho_in, rho_out_adj)
        CuDensityMat.initialize_zero!(rho_in_adj)
        params_grad .= 0.0
        CuDensityMat.compute_operator_action_backward!(
            ws,
            operator,
            rho_in,
            rho_out_adj,
            rho_in_adj,
            params_grad;
            time = 0.0,
            batch_size = 1,
            num_params = 1,
            params = params_vec,
        )

        backward_adj = Array(rho_in_adj.storage)

        # Finite-difference: perturb each real and imaginary component of rho_in
        ε = 1.0e-7
        fd_adj = zeros(T, D * D)

        for idx in 1:(D * D)
            for (δ_re, δ_im) in [(ε, 0.0), (0.0, ε)]
                perturbed = copy(rho_in_data)
                perturbed[idx] += complex(δ_re, δ_im)
                copyto!(rho_in.storage, CUDA.CuVector{T}(perturbed))
                CuDensityMat.initialize_zero!(rho_out)
                CuDensityMat.compute_operator_action!(
                    ws,
                    operator,
                    rho_in,
                    rho_out;
                    time = 0.0,
                    batch_size = 1,
                    num_params = 1,
                    params = params_vec,
                )
                out_plus = Array(rho_out.storage)
                cost_plus = real(dot(rho_out_adj_data, out_plus))

                perturbed = copy(rho_in_data)
                perturbed[idx] -= complex(δ_re, δ_im)
                copyto!(rho_in.storage, CUDA.CuVector{T}(perturbed))
                CuDensityMat.initialize_zero!(rho_out)
                CuDensityMat.compute_operator_action!(
                    ws,
                    operator,
                    rho_in,
                    rho_out;
                    time = 0.0,
                    batch_size = 1,
                    num_params = 1,
                    params = params_vec,
                )
                out_minus = Array(rho_out.storage)
                cost_minus = real(dot(rho_out_adj_data, out_minus))

                dc = (cost_plus - cost_minus) / (2ε)
                if δ_re > 0
                    fd_adj[idx] += dc      # real part of adjoint
                else
                    fd_adj[idx] += dc * im  # imaginary part of adjoint
                end
            end
        end

        # Restore original state
        copyto!(rho_in.storage, CUDA.CuVector{T}(rho_in_data))

        # The backward state_in_adj should be conjugated relative to the
        # Wirtinger derivative: state_in_adj = conj(∂c/∂rho_in*)
        # For a linear operator, state_in_adj = A^H * rho_out_adj
        for idx in 1:(D * D)
            @test real(backward_adj[idx]) ≈ real(fd_adj[idx]) atol = 1.0e-4
            @test imag(backward_adj[idx]) ≈ imag(fd_adj[idx]) atol = 1.0e-4
        end

        CuDensityMat.unregister_callback!(cb_refs)
        close(ws)
    end

    # =========================================================================
    # Test 4: Multi-parameter gradient — two coefficients
    #
    # Operator: A(θ₁,θ₂) = θ₁ * σ_z (duality=0) + θ₂ * σ_x (duality=0)
    # Two separate terms with separate scalar callbacks, each reading
    # one parameter from the shared params vector.
    # Verify both ∂c/∂θ₁ and ∂c/∂θ₂ via finite differences.
    # =========================================================================

    @gpu_test "multi-parameter gradient matches finite differences" begin
        ws = WorkStream()
        dims = [2]
        T = ComplexF64

        # σ_z and σ_x elementary operators
        sigma_z_data = CUDA.CuVector{T}([1.0 + 0im, 0.0, 0.0, -1.0 + 0im])
        sigma_x_data = CUDA.CuVector{T}([0.0 + 0im, 1.0, 1.0, 0.0 + 0im])
        elem_z = CuDensityMat.create_elementary_operator(ws, [2], sigma_z_data)
        elem_x = CuDensityMat.create_elementary_operator(ws, [2], sigma_x_data)

        term_z = CuDensityMat.create_operator_term(ws, dims)
        CuDensityMat.append_elementary_product!(term_z, [elem_z], Int32[0], Int32[0])

        term_x = CuDensityMat.create_operator_term(ws, dims)
        CuDensityMat.append_elementary_product!(term_x, [elem_x], Int32[0], Int32[0])

        # Callback for term_z: reads params[1] = θ₁
        function coeff_z(time, params, storage)
            for b in eachindex(storage)
                storage[b] = complex(params[1, b], 0.0)
            end
        end
        function grad_z(time, params, scalar_grad, params_grad)
            for b in axes(scalar_grad, 1)
                # df/dθ₁ = 1, df/dθ₂ = 0
                params_grad[1, b] += real(scalar_grad[b])
            end
        end

        # Callback for term_x: reads params[2] = θ₂
        function coeff_x(time, params, storage)
            for b in eachindex(storage)
                storage[b] = complex(params[2, b], 0.0)
            end
        end
        function grad_x(time, params, scalar_grad, params_grad)
            for b in axes(scalar_grad, 1)
                # df/dθ₁ = 0, df/dθ₂ = 1
                params_grad[2, b] += real(scalar_grad[b])
            end
        end

        cb_z, gcb_z, refs_z = CuDensityMat.wrap_scalar_callback(coeff_z; gradient = grad_z)
        cb_x, gcb_x, refs_x = CuDensityMat.wrap_scalar_callback(coeff_x; gradient = grad_x)

        operator = CuDensityMat.create_operator(ws, dims)
        CuDensityMat.append_term!(
            operator,
            term_z;
            duality = 0,
            coefficient = 1.0 + 0im,
            coefficient_callback = cb_z,
            coefficient_gradient_callback = gcb_z,
        )
        CuDensityMat.append_term!(
            operator,
            term_x;
            duality = 0,
            coefficient = 1.0 + 0im,
            coefficient_callback = cb_x,
            coefficient_gradient_callback = gcb_x,
        )

        # States
        rho_in = DenseMixedState{T}(ws, (2,); batch_size = 1)
        rho_out = DenseMixedState{T}(ws, (2,); batch_size = 1)
        CuDensityMat.allocate_storage!(rho_in)
        CuDensityMat.allocate_storage!(rho_out)

        rho_in_data = T[0.6, 0.1 - 0.2im, 0.1 + 0.2im, 0.4]
        rho_out_adj_data = T[0.8, 0.3im, -0.3im, 0.2]
        copyto!(rho_in.storage, CUDA.CuVector{T}(rho_in_data))

        rho_out_adj = DenseMixedState{T}(ws, (2,); batch_size = 1)
        CuDensityMat.allocate_storage!(rho_out_adj)
        copyto!(rho_out_adj.storage, CUDA.CuVector{T}(rho_out_adj_data))

        rho_in_adj = DenseMixedState{T}(ws, (2,); batch_size = 1)
        CuDensityMat.allocate_storage!(rho_in_adj)

        θ = [2.0, 0.5]
        num_params = 2
        params_vec = CUDA.CuVector{Float64}(θ)
        params_grad = CUDA.zeros(Float64, num_params)

        # Forward
        CuDensityMat.prepare_operator_action!(ws, operator, rho_in, rho_out)
        CuDensityMat.initialize_zero!(rho_out)
        CuDensityMat.compute_operator_action!(
            ws,
            operator,
            rho_in,
            rho_out;
            time = 0.0,
            batch_size = 1,
            num_params = num_params,
            params = params_vec,
        )

        # Backward
        CuDensityMat.prepare_operator_action_backward!(ws, operator, rho_in, rho_out_adj)
        CuDensityMat.initialize_zero!(rho_in_adj)
        params_grad .= 0.0
        CuDensityMat.compute_operator_action_backward!(
            ws,
            operator,
            rho_in,
            rho_out_adj,
            rho_in_adj,
            params_grad;
            time = 0.0,
            batch_size = 1,
            num_params = num_params,
            params = params_vec,
        )

        backward_grads = Array(params_grad)

        # Finite difference for each parameter
        ε = 1.0e-6
        function cost_at_params(θ_vals)
            p = CUDA.CuVector{Float64}(θ_vals)
            CuDensityMat.initialize_zero!(rho_out)
            CuDensityMat.compute_operator_action!(
                ws,
                operator,
                rho_in,
                rho_out;
                time = 0.0,
                batch_size = 1,
                num_params = num_params,
                params = p,
            )
            out = Array(rho_out.storage)
            return real(dot(rho_out_adj_data, out))
        end

        for n in 1:num_params
            θ_plus = copy(θ)
            θ_plus[n] += ε
            θ_minus = copy(θ)
            θ_minus[n] -= ε
            fd = (cost_at_params(θ_plus) - cost_at_params(θ_minus)) / (2ε)
            @test backward_grads[n] ≈ fd rtol = 1.0e-4
        end

        CuDensityMat.unregister_callback!(refs_z)
        CuDensityMat.unregister_callback!(refs_x)
        close(ws)
    end

    # =========================================================================
    # Test 5: Time-dependent coefficient gradient
    #
    # Operator: A(Ω,t) = exp(-Ωt) * σ_z
    # Parameter: Ω (decay rate)
    # df/dΩ = -t * exp(-Ωt)
    # Verify ∂c/∂Ω at t=0.5 via finite differences.
    # =========================================================================

    @gpu_test "time-dependent coefficient gradient matches finite diff" begin
        ws = WorkStream()
        dims = [2]
        T = ComplexF64
        t_eval = 0.5

        sigma_z_data = CUDA.CuVector{T}([1.0 + 0im, 0.0, 0.0, -1.0 + 0im])
        elem = CuDensityMat.create_elementary_operator(ws, [2], sigma_z_data)
        term = CuDensityMat.create_operator_term(ws, dims)
        CuDensityMat.append_elementary_product!(term, [elem], Int32[0], Int32[0])

        # f(t, Ω) = exp(-Ω*t)
        function coeff_fn(time, params, storage)
            for b in eachindex(storage)
                Ω = params[1, b]
                storage[b] = complex(exp(-Ω * time), 0.0)
            end
        end
        # df/dΩ = -t * exp(-Ω*t)
        function grad_fn(time, params, scalar_grad, params_grad)
            for b in axes(scalar_grad, 1)
                Ω = params[1, b]
                dfdΩ = complex(-time * exp(-Ω * time), 0.0)
                params_grad[1, b] += real(scalar_grad[b] * dfdΩ)
            end
        end

        cb, gcb, cb_refs = CuDensityMat.wrap_scalar_callback(coeff_fn; gradient = grad_fn)

        operator = CuDensityMat.create_operator(ws, dims)
        CuDensityMat.append_term!(
            operator,
            term;
            duality = 0,
            coefficient = 1.0 + 0im,
            coefficient_callback = cb,
            coefficient_gradient_callback = gcb,
        )

        rho_in = DenseMixedState{T}(ws, (2,); batch_size = 1)
        rho_out = DenseMixedState{T}(ws, (2,); batch_size = 1)
        CuDensityMat.allocate_storage!(rho_in)
        CuDensityMat.allocate_storage!(rho_out)

        rho_in_data = T[0.8, 0.1 + 0.05im, 0.1 - 0.05im, 0.2]
        rho_out_adj_data = T[1.0, 0.0, 0.0, 1.0]
        copyto!(rho_in.storage, CUDA.CuVector{T}(rho_in_data))

        rho_out_adj = DenseMixedState{T}(ws, (2,); batch_size = 1)
        CuDensityMat.allocate_storage!(rho_out_adj)
        copyto!(rho_out_adj.storage, CUDA.CuVector{T}(rho_out_adj_data))

        rho_in_adj = DenseMixedState{T}(ws, (2,); batch_size = 1)
        CuDensityMat.allocate_storage!(rho_in_adj)

        Ω₀ = 1.0
        params_vec = CUDA.CuVector{Float64}([Ω₀])
        params_grad = CUDA.zeros(Float64, 1)

        # Forward
        CuDensityMat.prepare_operator_action!(ws, operator, rho_in, rho_out)
        CuDensityMat.initialize_zero!(rho_out)
        CuDensityMat.compute_operator_action!(
            ws,
            operator,
            rho_in,
            rho_out;
            time = t_eval,
            batch_size = 1,
            num_params = 1,
            params = params_vec,
        )

        # Backward
        CuDensityMat.prepare_operator_action_backward!(ws, operator, rho_in, rho_out_adj)
        CuDensityMat.initialize_zero!(rho_in_adj)
        params_grad .= 0.0
        CuDensityMat.compute_operator_action_backward!(
            ws,
            operator,
            rho_in,
            rho_out_adj,
            rho_in_adj,
            params_grad;
            time = t_eval,
            batch_size = 1,
            num_params = 1,
            params = params_vec,
        )

        backward_grad = Array(params_grad)[1]

        # Finite difference
        ε = 1.0e-6
        function cost_at(Ω)
            p = CUDA.CuVector{Float64}([Ω])
            CuDensityMat.initialize_zero!(rho_out)
            CuDensityMat.compute_operator_action!(
                ws,
                operator,
                rho_in,
                rho_out;
                time = t_eval,
                batch_size = 1,
                num_params = 1,
                params = p,
            )
            out = Array(rho_out.storage)
            return real(dot(rho_out_adj_data, out))
        end
        fd_grad = (cost_at(Ω₀ + ε) - cost_at(Ω₀ - ε)) / (2ε)

        @test backward_grad ≈ fd_grad rtol = 1.0e-4

        CuDensityMat.unregister_callback!(cb_refs)
        close(ws)
    end

    # Note: backward diff without callbacks/params is not supported by the C API
    # (requires numParams >= 1 with gradient callbacks). This is confirmed by
    # NVIDIA's C++ samples which always use at least 1 parameter.

end
