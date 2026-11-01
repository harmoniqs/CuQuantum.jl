# Phase 7: Backward differentiation tests
#
# Tests the gradient/backward differentiation pipeline:
# prepare_backward, compute_backward, parameter gradients.
# Single-GPU only (no MPI backward diff support yet from NVIDIA).
#
# Gradient convention (pinned below; see `compute_operator_action_backward!`):
#
# Seeding `state_out_adj = a` defines the real cost  c = Re⟨a, ρ_out⟩ = Re Tr(a† ρ_out).
# The library then returns
#   state_in_adj  += A^H a            (= ∂c/∂Re ρ_in + i ∂c/∂Im ρ_in)
# and passes each gradient callback the adjoint of its quantity Q,
#   scalar_grad    = ⟨a, ∂ρ_out/∂Q⟩  (no conjugation; includes static coefficients),
#   tensor_grad[k] = ∂/∂U[k] of ⟨a, ρ_out⟩ (holomorphic derivative),
# so a callback must accumulate  params_grad[n] += Re(scalar_grad * ∂Q/∂θ_n),
# with NO factor 2. The header's "2·Re" is the same number written for a Wirtinger
# seed a = ∂c/∂ρ_out* (half of ours): 2·Re(⟨a/2, ·⟩ ∂Q/∂θ) = Re(⟨a, ·⟩ ∂Q/∂θ).
#
# Every FD test here has an analytic dense-CPU oracle too, and a negative control:
# the fixtures are chosen so that 2·Re, conj(scalar_grad) and dropping the static
# coefficient each miss the oracle by far more than the tolerance (checked on the
# CPU in "gradient convention oracle (CPU)").

const _σz = ComplexF64[1 0; 0 -1]
const _σx = ComplexF64[0 1; 1 0]

# Fixture for the convention tests: a generic complex operator B, a Hermitian input
# state, a non-Hermitian adjoint seed, a complex static coefficient and the angle θ
# for Q(θ) = cis(θ), whose Jacobian i·cis(θ) is complex (a real Jacobian cannot
# tell scalar_grad from conj(scalar_grad)).
const CONV_B = ComplexF64[1.0 0.5-0.25im; 0.3im -0.7]
const CONV_ρ = ComplexF64[0.6 0.2-0.3im; 0.2+0.3im 0.4]
const CONV_a = ComplexF64[0.3+0.5im 0.1+0.4im; -0.2+0.8im 0.3]
const CONV_c0 = 0.6 + 0.3im
const CONV_θ = 0.4

# Cost of the dense CPU model, c(θ) = Re⟨a, c0·Q(θ)·B ρ⟩ (ket-side action).
_conv_cost(θ) = real(dot(CONV_a, CONV_c0 * cis(θ) * CONV_B * CONV_ρ))
# Holomorphic adjoint of the callback coefficient, ⟨a, c0·B ρ⟩.
const CONV_w = CONV_c0 * dot(CONV_a, CONV_B * CONV_ρ)
# dc/dθ under the Re convention.
const CONV_grad = real(CONV_w * im * cis(CONV_θ))

# Tensor-callback fixture: U(θ) = exp(-iθσx) = cosθ·I - i sinθ·σx.
_conv_U(θ) = cos(θ) * I - im * sin(θ) * _σx
_conv_dU(θ) = -sin(θ) * I - im * cos(θ) * _σx
_conv_tensor_cost(θ) = real(dot(CONV_a, _conv_U(θ) * CONV_ρ))
# Holomorphic adjoint of U: tensor_grad[i,k] = Σ_j conj(a[i,j]) ρ[k,j].
const CONV_tensor_adj = conj(CONV_a) * transpose(CONV_ρ)
const CONV_tensor_grad = real(sum(CONV_tensor_adj .* _conv_dU(CONV_θ)))

_central_diff(f, x; ε = 1e-5) = (f(x + ε) - f(x - ε)) / (2ε)

# One forward + backward pass of the single-mode operator `c0 · Q · term` on 2×2
# mixed states; returns (ρ_out, params_grad, state_in_adj) as host arrays.
function _backward_pass(ws, operator, ρ, a, θ; time = 0.0)
    T = ComplexF64
    rho_in = DenseMixedState{T}(ws, (2,); batch_size = 1)
    rho_out = DenseMixedState{T}(ws, (2,); batch_size = 1)
    rho_out_adj = DenseMixedState{T}(ws, (2,); batch_size = 1)
    rho_in_adj = DenseMixedState{T}(ws, (2,); batch_size = 1)
    foreach(CuDensityMat.allocate_storage!, (rho_in, rho_out, rho_out_adj, rho_in_adj))
    copyto!(rho_in.storage, CUDA.CuVector{T}(vec(ρ)))
    copyto!(rho_out_adj.storage, CUDA.CuVector{T}(vec(a)))
    params = CUDA.CuVector{Float64}([θ])
    CuDensityMat.prepare_operator_action!(ws, operator, rho_in, rho_out)
    CuDensityMat.initialize_zero!(rho_out)
    CuDensityMat.compute_operator_action!(
        ws,
        operator,
        rho_in,
        rho_out;
        time,
        num_params = 1,
        params,
    )
    CuDensityMat.prepare_operator_action_backward!(ws, operator, rho_in, rho_out_adj)
    CuDensityMat.initialize_zero!(rho_in_adj)
    params_grad = CUDA.zeros(Float64, 1)
    CuDensityMat.compute_operator_action_backward!(
        ws,
        operator,
        rho_in,
        rho_out_adj,
        rho_in_adj,
        params_grad;
        time,
        num_params = 1,
        params,
    )
    cost(θ′) = begin
        CuDensityMat.initialize_zero!(rho_out)
        CuDensityMat.compute_operator_action!(
            ws,
            operator,
            rho_in,
            rho_out;
            time,
            num_params = 1,
            params = CUDA.CuVector{Float64}([θ′]),
        )
        real(dot(vec(a), Array(rho_out.storage)))
    end
    out = reshape(Array(rho_out.storage), 2, 2)
    fd = _central_diff(cost, θ)
    return (out, Array(params_grad)[1], reshape(Array(rho_in_adj.storage), 2, 2), fd)
end

@testset "Gradients (local)" begin

    # =========================================================================
    # Convention oracle (CPU): the fixtures discriminate the candidate conventions
    # =========================================================================

    @testset "gradient convention oracle (CPU)" begin
        # The analytic Re-convention gradient matches finite differences of the
        # dense model ...
        @test CONV_grad ≈ _central_diff(_conv_cost, CONV_θ) rtol = 1e-9
        @test CONV_tensor_grad ≈ _central_diff(_conv_tensor_cost, CONV_θ) rtol = 1e-9
        # ... and is far from zero, so a relative tolerance means something.
        @test abs(CONV_grad) > 0.1
        @test abs(CONV_tensor_grad) > 0.1
        # Negative controls: each wrong convention misses by ≫ the GPU tolerances.
        wrong_scalar = (
            two_re = 2 * CONV_grad,
            conjugated = real(conj(CONV_w) * im * cis(CONV_θ)),
            no_static_coefficient = real(CONV_w / CONV_c0 * im * cis(CONV_θ)),
        )
        for (name, g) in pairs(wrong_scalar)
            @test abs(g - CONV_grad) > 0.5 * abs(CONV_grad)
        end
        @test abs(imag(CONV_w)) > 0.1 # scalar_grad ≠ conj(scalar_grad)
        conj_tensor = real(sum(conj(CONV_tensor_adj) .* _conv_dU(CONV_θ)))
        @test abs(conj_tensor - CONV_tensor_grad) > 0.5 * abs(CONV_tensor_grad)
        @test abs(2 * CONV_tensor_grad - CONV_tensor_grad) > 0.5 * abs(CONV_tensor_grad)
        # state_in_adj = A^H a is the full real gradient of c w.r.t. ρ_in.
        A = CONV_c0 * cis(CONV_θ) * CONV_B
        cost_in(ρ) = real(dot(CONV_a, A * ρ))
        fd_adj = map(CartesianIndices(CONV_ρ)) do k
            e = zeros(ComplexF64, 2, 2)
            e[k] = 1
            _central_diff(t -> cost_in(CONV_ρ + t * e), 0.0) +
            im * _central_diff(t -> cost_in(CONV_ρ + im * t * e), 0.0)
        end
        @test fd_adj ≈ A' * CONV_a rtol = 1e-9
    end

    # =========================================================================
    # Test 1: the scalar adjoint handed to the gradient callback (pins conjugation)
    # =========================================================================

    @gpu_test "scalar_grad is the unconjugated adjoint ⟨a, ∂ρ_out/∂Q⟩" begin
        ws = WorkStream()
        dims = [2]
        T = ComplexF64
        elem =
            CuDensityMat.create_elementary_operator(ws, [2], CUDA.CuVector{T}(vec(CONV_B)))
        term = CuDensityMat.create_operator_term(ws, dims)
        CuDensityMat.append_elementary_product!(term, [elem], Int32[0], Int32[0])
        captured = ComplexF64[]
        cb, gcb, cb_refs = CuDensityMat.wrap_scalar_callback(
            (t, p, s) -> (s .= cis.(p[1, :]); nothing);
            gradient = function (t, p, sg, pg)
                append!(captured, sg)
                for b in axes(sg, 1)
                    pg[1, b] += real(sg[b] * im * cis(p[1, b]))
                end
            end,
        )
        operator = CuDensityMat.create_operator(ws, dims)
        CuDensityMat.append_term!(
            operator,
            term;
            duality = 0,
            coefficient = CONV_c0,
            coefficient_callback = cb,
            coefficient_gradient_callback = gcb,
        )
        out, g, adj, fd = _backward_pass(ws, operator, CONV_ρ, CONV_a, CONV_θ)

        # forward pass against the dense model
        A = CONV_c0 * cis(CONV_θ) * CONV_B
        @test out ≈ A * CONV_ρ rtol = 1e-12
        # the adjoint passed to the callback: includes c0, not conjugated
        @test length(captured) == 1
        @test captured[1] ≈ CONV_w rtol = 1e-12
        @test !isapprox(captured[1], conj(CONV_w); rtol = 0.1)        # negative control
        @test !isapprox(captured[1], CONV_w / CONV_c0; rtol = 0.1)    # negative control
        # params_grad: analytic oracle, FD oracle, and the factor-2 negative control
        @test g ≈ CONV_grad rtol = 1e-12
        @test g ≈ fd rtol = 1e-8
        @test !isapprox(2g, fd; rtol = 0.1)
        # state_in_adj = A^H a
        @test adj ≈ A' * CONV_a rtol = 1e-12

        CuDensityMat.unregister_callback!(cb_refs)
        close(ws)
    end

    # =========================================================================
    # Test 1b: tensor gradient callback (complex Jacobian dU/dθ)
    # =========================================================================

    @gpu_test "tensor gradient callback: Re(Σ tensor_grad .* dU/dθ) matches FD" begin
        ws = WorkStream()
        dims = [2]
        T = ComplexF64
        captured = Array{ComplexF64}[]
        tcb, tgcb, refs = CuDensityMat.wrap_tensor_callback(
            function (t, p, s)
                for b in axes(s, 3)
                    s[:, :, b] .= _conv_U(p[1, b])
                end
            end;
            gradient = function (t, p, tg, pg)
                push!(captured, copy(tg))
                for b in axes(tg, 3)
                    pg[1, b] += real(sum(tg[:, :, b] .* _conv_dU(p[1, b])))
                end
            end,
        )
        elem = CuDensityMat.create_elementary_operator(
            ws,
            [2],
            CUDA.zeros(T, 4);
            tensor_callback = tcb,
            tensor_gradient_callback = tgcb,
        )
        term = CuDensityMat.create_operator_term(ws, dims)
        CuDensityMat.append_elementary_product!(term, [elem], Int32[0], Int32[0])
        operator = CuDensityMat.create_operator(ws, dims)
        CuDensityMat.append_term!(operator, term; duality = 0)
        out, g, adj, fd = _backward_pass(ws, operator, CONV_ρ, CONV_a, CONV_θ)

        U = _conv_U(CONV_θ)
        @test out ≈ U * CONV_ρ rtol = 1e-12
        @test length(captured) == 1
        @test captured[1][:, :, 1] ≈ CONV_tensor_adj rtol = 1e-12
        @test g ≈ CONV_tensor_grad rtol = 1e-12
        @test g ≈ fd rtol = 1e-8
        @test !isapprox(2g, fd; rtol = 0.1)                                    # negative control
        @test !isapprox(captured[1][:, :, 1], conj(CONV_tensor_adj); rtol = 0.1) # negative control
        @test adj ≈ U' * CONV_a rtol = 1e-12

        CuDensityMat.unregister_callback!(refs)
        close(ws)
    end

    # =========================================================================
    # Test 1c: batched backward pass — params_grad[n, b] per batch member
    # =========================================================================

    @gpu_test "batched params_grad: one gradient per batch member" begin
        ws = WorkStream()
        dims = [2]
        T = ComplexF64
        nb = 2
        elem =
            CuDensityMat.create_elementary_operator(ws, [2], CUDA.CuVector{T}(vec(CONV_B)))
        term = CuDensityMat.create_operator_term(ws, dims)
        CuDensityMat.append_elementary_product!(term, [elem], Int32[0], Int32[0])
        cb, gcb, refs = CuDensityMat.wrap_scalar_callback(
            (t, p, s) -> (s .= cis.(p[1, :]); nothing);
            gradient = (t, p, sg, pg) ->
                (pg[1, :] .+= real.(sg .* im .* cis.(p[1, :])); nothing),
        )
        operator = CuDensityMat.create_operator(ws, dims)
        CuDensityMat.append_term!(
            operator,
            term;
            coefficient = CONV_c0,
            coefficient_callback = cb,
            coefficient_gradient_callback = gcb,
        )
        θs = [CONV_θ, -1.3]
        ρs = (CONV_ρ, ComplexF64[0.25 0.1im; -0.1im 0.75])
        as = (CONV_a, ComplexF64[0.2 -0.6im; 0.5+0.1im 0.8im])
        rho_in = DenseMixedState{T}(ws, (2,); batch_size = nb)
        rho_out = DenseMixedState{T}(ws, (2,); batch_size = nb)
        rho_out_adj = DenseMixedState{T}(ws, (2,); batch_size = nb)
        rho_in_adj = DenseMixedState{T}(ws, (2,); batch_size = nb)
        foreach(CuDensityMat.allocate_storage!, (rho_in, rho_out, rho_out_adj, rho_in_adj))
        copyto!(rho_in.storage, CUDA.CuVector{T}(reduce(vcat, vec.(ρs))))
        copyto!(rho_out_adj.storage, CUDA.CuVector{T}(reduce(vcat, vec.(as))))
        params = CUDA.CuVector{Float64}(θs)            # (numParams = 1, batchSize = 2)
        CuDensityMat.prepare_operator_action!(ws, operator, rho_in, rho_out)
        CuDensityMat.initialize_zero!(rho_out)
        CuDensityMat.compute_operator_action!(
            ws,
            operator,
            rho_in,
            rho_out;
            num_params = 1,
            params,
        )
        CuDensityMat.prepare_operator_action_backward!(ws, operator, rho_in, rho_out_adj)
        CuDensityMat.initialize_zero!(rho_in_adj)
        params_grad = CUDA.zeros(Float64, nb)
        CuDensityMat.compute_operator_action_backward!(
            ws,
            operator,
            rho_in,
            rho_out_adj,
            rho_in_adj,
            params_grad;
            num_params = 1,
            params,
        )
        out = reshape(Array(rho_out.storage), 2, 2, nb)
        g = Array(params_grad)
        for b = 1:nb
            A = CONV_c0 * cis(θs[b]) * CONV_B
            @test out[:, :, b] ≈ A * ρs[b] rtol = 1e-12
            analytic = real(CONV_c0 * dot(as[b], CONV_B * ρs[b]) * im * cis(θs[b]))
            @test abs(analytic) > 0.05
            @test g[b] ≈ analytic rtol = 1e-12
        end
        # negative control: the batch members are not swapped or summed
        @test !isapprox(g, reverse(g); rtol = 0.1)
        CuDensityMat.unregister_callback!(refs)
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
        # Gradient callback: df/dθ = 1, so params_grad += Re(scalar_grad * 1)
        # (no factor 2 — see the convention note at the top of this file)
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
        rho_out_adj_data = T[1.0, 0.2+0.1im, 0.2-0.1im, 0.5]

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
        ε = 1e-6
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

        # c(θ) = θ·Re⟨a, σz ρ⟩ is linear in θ, so central differences are exact up to
        # roundoff (~1e-16/ε): 1e-8 is ~100× the noise, 1e-4 was ~10⁶×.
        analytic = real(dot(rho_out_adj_data, vec(_σz * reshape(rho_in_data, 2, 2))))
        @test abs(analytic) > 0.1
        @test backward_grad ≈ analytic rtol = 1e-12
        @test backward_grad ≈ fd_grad rtol = 1e-8
        @test !isapprox(2 * backward_grad, fd_grad; rtol = 0.1) # negative control

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
        rho_out_adj_data = T[1.0, 0.2+0.1im, 0.2-0.1im, 0.5]

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
        ε = 1e-7
        fd_adj = zeros(T, D * D)

        for idx = 1:(D*D)
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
        # Exact oracle: A^H a with A = θ σz (ket side).
        exact_adj = vec((θ₀ * _σz)' * reshape(rho_out_adj_data, 2, 2))
        @test backward_adj ≈ exact_adj rtol = 1e-12
        # FD oracle: cost is linear in ρ_in, so the error is roundoff ~eps/ε ≈ 2e-9.
        for idx = 1:(D*D)
            @test real(backward_adj[idx]) ≈ real(fd_adj[idx]) atol = 1e-7
            @test imag(backward_adj[idx]) ≈ imag(fd_adj[idx]) atol = 1e-7
        end
        # negative controls: the conjugate (or a factor 2) is far from the FD gradient
        @test !isapprox(conj.(backward_adj), fd_adj; atol = 0.1)
        @test !isapprox(2 .* backward_adj, fd_adj; atol = 0.1)

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

        rho_in_data = T[0.6, 0.1-0.2im, 0.1+0.2im, 0.4]
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
        ε = 1e-6
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

        for n = 1:num_params
            θ_plus = copy(θ)
            θ_plus[n] += ε
            θ_minus = copy(θ)
            θ_minus[n] -= ε
            fd = (cost_at_params(θ_plus) - cost_at_params(θ_minus)) / (2ε)
            @test backward_grads[n] ≈ fd rtol = 1e-8
            @test !isapprox(2 * backward_grads[n], fd; rtol = 0.1) # negative control
        end
        ρm = reshape(rho_in_data, 2, 2)
        am = reshape(rho_out_adj_data, 2, 2)
        analytic = [real(dot(am, _σz * ρm)), real(dot(am, _σx * ρm))]
        @test all(abs.(analytic) .> 0.05)
        @test backward_grads ≈ analytic rtol = 1e-12

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

        rho_in_data = T[0.8, 0.1+0.05im, 0.1-0.05im, 0.2]
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
        ε = 1e-6
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

        # c(Ω) = e^{-Ωt}·Tr(σz ρ): central-difference truncation ε²|c‴|/6 ~ 1e-14.
        analytic = -t_eval * exp(-Ω₀ * t_eval) * real(tr(_σz * reshape(rho_in_data, 2, 2)))
        @test abs(analytic) > 0.1
        @test backward_grad ≈ analytic rtol = 1e-12
        @test backward_grad ≈ fd_grad rtol = 1e-8
        @test !isapprox(2 * backward_grad, fd_grad; rtol = 0.1) # negative control

        CuDensityMat.unregister_callback!(cb_refs)
        close(ws)
    end

    # cuDensityMat 0.7 refuses to prepare backward differentiation before the forward
    # action ("Super-operator action on a mixed dense quantum state required by
    # backward differentiation has not been prepared!");
    # prepare_operator_action_backward! prepares it on demand.
    @gpu_test "backward diff without a preceding forward prepare" begin
        T = ComplexF64
        dims = [2]
        function grad(prepare_forward::Bool)
            ws = WorkStream()
            sz = CUDA.CuVector{T}([1.0 + 0im, 0.0, 0.0, -1.0 + 0im])
            elem = CuDensityMat.create_elementary_operator(ws, [2], sz)
            term = CuDensityMat.create_operator_term(ws, dims)
            CuDensityMat.append_elementary_product!(term, [elem], Int32[0], Int32[0])
            cb, gcb, refs = CuDensityMat.wrap_scalar_callback(
                (t, p, s) -> (s .= complex.(p[1, :]); nothing);
                gradient = (t, p, sg, pg) -> (pg[1, :] .+= real.(sg); nothing),
            )
            op = CuDensityMat.create_operator(ws, dims)
            CuDensityMat.append_term!(
                op,
                term;
                coefficient_callback = cb,
                coefficient_gradient_callback = gcb,
            )
            ρin = DenseMixedState{T}(ws, (2,); batch_size = 1)
            ρout_adj = DenseMixedState{T}(ws, (2,); batch_size = 1)
            ρin_adj = DenseMixedState{T}(ws, (2,); batch_size = 1)
            foreach(CuDensityMat.allocate_storage!, (ρin, ρout_adj, ρin_adj))
            copyto!(ρin.storage, CUDA.CuVector{T}([0.7, -0.3im, 0.3im, 0.3]))
            copyto!(
                ρout_adj.storage,
                CUDA.CuVector{T}([1.0, 0.2 + 0.1im, 0.2 - 0.1im, 0.5]),
            )
            prepare_forward && CuDensityMat.prepare_operator_action!(ws, op, ρin, ρout_adj)
            CuDensityMat.prepare_operator_action_backward!(ws, op, ρin, ρout_adj)
            CuDensityMat.initialize_zero!(ρin_adj)
            params_grad = CUDA.zeros(Float64, 1)
            CuDensityMat.compute_operator_action_backward!(
                ws,
                op,
                ρin,
                ρout_adj,
                ρin_adj,
                params_grad;
                num_params = 1,
                params = CUDA.CuVector{Float64}([1.5]),
            )
            out = (Array(params_grad), Array(ρin_adj.storage))
            CuDensityMat.unregister_callback!(refs)
            close(ws)
            return out
        end
        g_auto, adj_auto = grad(false)
        g_ref, adj_ref = grad(true)
        @test g_auto ≈ g_ref
        @test adj_auto ≈ adj_ref
        # both runs against the analytic oracle, not just each other
        a = ComplexF64[1.0 0.2-0.1im; 0.2+0.1im 0.5]
        ρ = ComplexF64[0.7 0.3im; -0.3im 0.3]
        @test g_ref[1] ≈ real(dot(a, _σz * ρ)) rtol = 1e-12
        @test adj_ref ≈ vec((1.5 * _σz)' * a) rtol = 1e-12
    end

    # Note: backward diff without callbacks/params is not supported by the C API
    # (requires numParams >= 1 with gradient callbacks). This is confirmed by
    # NVIDIA's C++ samples which always use at least 1 parameter.

end
