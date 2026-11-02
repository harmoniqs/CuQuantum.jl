# Lindbladian builder + batched fixed-step integrators.
#
# Host tests run everywhere: the plan lowering (`lindblad_plan`, executed by
# `CpuLindbladian`) is checked against the textbook dense superoperator
# (`lindblad_superoperator`, built directly from the spec), and the integrators /
# stiffness helpers are exercised through the host backend. GPU tests check the
# cuDensityMat Operator built by `LindbladOperator` against the same references.
#
# Tolerance policy: Lindbladian actions are compared as max|Δ| / max|ref| ≤ 1e-12 —
# FP64 sums of ≲10² terms of size ≤ max|ref| give ~1e-15–1e-14; 1e-12 leaves room for
# GPU contraction order and is ≥ 1e10 below every signal we compare (the negative
# controls below differ by ≥ 1e-2).

include(joinpath(@__DIR__, "lindblad_models.jl"))

@testset "Lindblad builder + integrators" begin

    @testset "multidiagonal storage" begin
        d = 6
        M = zeros(ComplexF64, d, d)
        for (k, v) in ((-2, 0.3 + 0.1im), (0, -1.2), (1, 0.7im), (3, 2.0 - 0.5im))
            for i = 1:(d-abs(k))
                k >= 0 ? (M[i, i+k] = v * i) : (M[i-k, i] = v * i)
            end
        end
        offs, data = multidiagonal_pack(M)
        @test offs == Int32[-2, 0, 1, 3]
        @test length(data) == d * 4
        D = reshape(data, d, 4)
        @test D[:, 1] == [M[3, 1], M[4, 2], M[5, 3], M[6, 4], 0, 0]   # below: from the top-left, zero-padded
        @test D[:, 4] == [M[1, 4], M[2, 5], M[3, 6], 0, 0, 0]          # above: from the top-left, zero-padded
        @test multidiagonal_unpack(offs, data, d) == M
        a = annihilation(d)
        @test multidiagonal_pack(a)[1] == Int32[1]
        @test multidiagonal_pack(a')[1] == Int32[-1]
        @test multidiagonal_pack(a * a - 4.0I)[1] == Int32[0, 2]
        @test multidiagonal_pack(zeros(ComplexF64, d, d))[1] == Int32[0]
        @test multidiagonal_unpack(multidiagonal_pack(a' * a' * a * a)..., d) ==
              a' * a' * a * a
    end

    @testset "plan (host executor) == dense superoperator" begin
        for (name, spec, P, t) in test_models()
            n = prod(spec.dims)
            B = size(P, 2)
            x = random_states(n, B; seed = 7)
            refs = [
                lindblad_superoperator(spec, t, P[:, b]) * x[((b-1)*n^2+1):(b*n^2)] for
                b = 1:B
            ]
            for merge in (true, false)
                L = CpuLindbladian(spec; batch_size = B, num_params = size(P, 1), merge)
                out = similar(x)
                L(out, x, t, P)
                @test maxrel(out, reduce(vcat, refs)) < 1e-12
            end
            # negative controls: the comparison resolves sign, conjugation and member mixups
            L = CpuLindbladian(spec; batch_size = B, num_params = size(P, 1))
            out = similar(x)
            L(out, x, t, P)
            bad = lindblad_superoperator(conj_spec(spec), t, P[:, 1]) * x[1:(n^2)]
            @test maxrel(out[1:(n^2)], bad) > 1e-2
            if B > 1
                @test maxrel(
                    out[1:(n^2)],
                    lindblad_superoperator(spec, t, P[:, 2]) * x[1:(n^2)],
                ) > 1e-2
            end
        end
        # merging cuts the product count (one ket + one bra operator per mode for the
        # static single-mode part) without changing the action (checked above)
        _, spec, _, _ = test_models()[2]
        @test sum(length(g.products) for g in lindblad_plan(spec)) <
              sum(length(g.products) for g in lindblad_plan(spec; merge = false))
        @test all(
            length(f.matrix) > 0 for g in lindblad_plan(spec) for p in g.products for
            f in p.factors
        )
    end

    @testset "trace preservation of the action" begin
        for (name, spec, P, t) in test_models()
            n = prod(spec.dims)
            B = size(P, 2)
            L = CpuLindbladian(spec; batch_size = B, num_params = size(P, 1))
            x = random_states(n, B; seed = 3)
            out = similar(x)
            L(out, x, t, P)
            @test maximum(abs.(batch_traces(out, n, B))) < 1e-12 * maximum(abs, out)
        end
    end

    @testset "RK4: order 4 and exp(Lt)" begin
        spec, P = two_mode_model(4; g = 0.35, shift = 1.0, rate1 = 0.05, dephase = 0.02, K = 0.1)
        n = 16
        S = lindblad_superoperator(spec, 0.0, P[:, 1])
        L = CpuLindbladian(spec; batch_size = 1, num_params = 1)
        x0 = random_states(n, 1; seed = 11, hermitian = true, unit_trace = true)
        T = 1.0
        ref = exp(S * T) * x0
        err(N) = maxrel(lindblad_rk4!(L, copy(x0), 0.0, T; nsteps = N, params = P), ref)
        # (N = 25 is pre-asymptotic, ratio 18.8; from N = 50 the ratios are 17.4, 16.7, 16.3)
        e = [err(N) for N in (50, 100, 200, 400)]
        r = e[1:(end-1)] ./ e[2:end]
        @test all(15 .< r .< 18)          # classic RK4: error ∝ h⁴ (ratio 16 per halving)
        @test e[end] > 1e-12              # still ≥ 10³ above round-off (order test is meaningful)
        @test err(4000) < 1e-10           # converged agreement with exp(Lt)
        @test !(err(25) < 1e-10)          # negative control: a coarse run is visibly off

        # time-dependent coefficient (g(t) = g0 cos(ω t) via the batched callback path)
        spec_t, Pt = two_mode_model(4; g = (t, p) -> p[1] * cos(3t), shift = 1.0, rate1 = 0.05)
        Lt = CpuLindbladian(spec_t; batch_size = 1, num_params = 1)
        reft = lindblad_rk4!(Lt, copy(x0), 0.0, T; nsteps = 6400, params = Pt)
        et = [
            maxrel(lindblad_rk4!(Lt, copy(x0), 0.0, T; nsteps = N, params = Pt), reft)
            for N in (50, 100, 200)
        ]
        @test all(15 .< et[1:(end-1)] ./ et[2:end] .< 18)
    end

    @testset "RK4: trace drift over many steps" begin
        spec, P = two_mode_model(4; g = 0.35, shift = 1.0, rate1 = 0.05, dephase = 0.02, K = 0.1)
        L = CpuLindbladian(spec; batch_size = 2, num_params = 1)
        P2 = hcat(P, 2P)
        x = random_states(16, 2; seed = 5, hermitian = true, unit_trace = true)
        drift = Ref(0.0)
        lindblad_rk4!(
            L,
            x,
            0.0,
            2.0;
            nsteps = 2000,
            params = P2,
            callback = (y, t, s) ->
                (drift[] = max(drift[], maximum(abs.(batch_traces(y, 16, 2) .- 1)))),
        )
        @test drift[] < 1e-12
        # negative control: drop the sandwich terms (not trace preserving) — drifts at O(1)
        bad = drop_sandwich(L)
        y = random_states(16, 2; seed = 5, hermitian = true, unit_trace = true)
        lindblad_rk4!(bad, y, 0.0, 2.0; nsteps = 2000, params = P2)
        @test maximum(abs.(batch_traces(y, 16, 2) .- 1)) > 1e-2
    end

    @testset "stiffness: λ_max estimate vs dense spectrum" begin
        for shift in (0.0, 4.0, 6.0), c in (6, 8, 10, 12)
            spec = single_mode_loss(c, shift)
            dense = maximum(abs, eigvals(lindblad_superoperator(spec)))
            est = lindblad_lambda_max(CpuLindbladian(spec))
            @test abs(est.abs[1] / dense - 1) < 0.10
        end
        # For the two-body loss rate·D[a²], the spectral radius at shift = 0 is exactly
        # rate·(c-1)(c-2) (the |c-1⟩⟨c-1| population decay); 2·rate·(c-1)(c-2) is the 1-norm
        # (Gershgorin) bound of the superoperator, not its spectral radius.
        for c in (6, 8, 12), r2 in (1.0, 2.5)
            S = lindblad_superoperator(single_mode_loss(c, 0.0; rate2 = r2))
            @test maximum(abs, eigvals(S)) ≈ r2 * (c - 1) * (c - 2) rtol = 1e-9
            @test opnorm(S, 1) ≈ 2r2 * (c - 1) * (c - 2) rtol = 1e-12
        end
    end

    @testset "RK4 stability bounds" begin
        @test rk4_stability_radius(π) ≈ RK4_REAL_AXIS_BOUND rtol = 1e-12
        @test abs(1 + sum((-RK4_REAL_AXIS_BOUND)^k / factorial(k) for k = 1:4)) ≈ 1 rtol =
            1e-12
        @test rk4_stability_radius(π / 2) ≈ 2sqrt(2) rtol = 1e-9
        @test rk4_stability_radius(0.3) == 0.0
        radii = [rk4_stability_radius(θ) for θ in range(π / 2 + 1e-9, π; length = 4001)]
        @test minimum(radii) ≈ RK4_HALF_DISK_BOUND rtol = 1e-6
        @test rk4_stable_step(-100.0) ≈ RK4_REAL_AXIS_BOUND / 100 rtol = 1e-12
    end

    @testset "RK4: h = 1.1 h_stab diverges, 0.9 h_stab does not" begin
        for shift in (0.0, 4.0)
            c = 10
            spec = single_mode_loss(c, shift)
            L = CpuLindbladian(spec)
            hstab = rk4_stable_step(lindblad_lambda_max(L).lambda)
            x0 = random_states(c, 1; seed = 2, hermitian = true, unit_trace = true)
            grow(f) = (
                x = lindblad_rk4!(L, copy(x0), 0.0, 300 * f * hstab; nsteps = 300);
                LinearAlgebra.norm(x) / LinearAlgebra.norm(x0)
            )
            @test grow(1.1) > 1e6
            @test grow(0.9) < 2
        end
    end

    @testset "batched members are independent" begin
        spec, _ = two_mode_model(
            4;
            g = (t, p) -> p[1] * cos(p[2] * t),
            shift = (1.0),
            rate1 = 0.05,
            dephase = 0.02,
            rate = (t, p) -> p[3],
        )
        P = [0.35 -0.2 0.6; 1.0 3.0 0.5; 0.1 0.4 0.0]
        B = 3
        n = 16
        x0 = random_states(n, B; seed = 13, hermitian = true, unit_trace = true)
        LB = CpuLindbladian(spec; batch_size = B, num_params = 3)
        xb = lindblad_rk4!(LB, copy(x0), 0.0, 0.8; nsteps = 200, params = P)
        L1 = CpuLindbladian(spec; batch_size = 1, num_params = 3)
        for b = 1:B
            idx = ((b-1)*n^2+1):(b*n^2)
            xs = lindblad_rk4!(L1, x0[idx], 0.0, 0.8; nsteps = 200, params = P[:, b:b])
            @test maxrel(xb[idx], xs) < 1e-14
        end
        xp = lindblad_rk4!(LB, copy(x0), 0.0, 0.8; nsteps = 200, params = P[:, [2, 1, 3]])
        @test maxrel(xp[1:(n^2)], xb[1:(n^2)]) > 1e-3   # negative control: params do reach the member
    end

    # =========================================================================
    # GPU
    # =========================================================================

    @gpu_test "GPU builder == dense superoperator (multidiagonal, batched params)" begin
        ws = WorkStream()
        # model 3 (batch 1 + coefficient callbacks) aborted the process inside libcudensitymat
        # 0.7.0 prepare on the A10 (exatn "Connections are invalid") before the builder used
        # non-batched callback terms at batch size 1; it is checked in a child process below
        # so a regression shows as a failure, not a dead suite.
        for (name, spec, P, t) in test_models()[1:2], sparsity in (:auto, :dense)
            n = prod(spec.dims)
            B = size(P, 2)
            L = LindbladOperator(
                ws,
                spec;
                batch_size = B,
                num_params = size(P, 1),
                sparsity,
            )
            # Regression (A10, 0.7.0): host argument arrays (diagonal offsets, extents, product
            # mode/duality lists) must stay rooted after construction — collected ones were read
            # back as garbage in prepare (exatn "Connections are invalid" abort / heap corruption).
            GC.gc(true)
            junk = [rand(64) for _ = 1:10_000]
            GC.gc(true)
            x = random_states(n, B; seed = 7)
            ρ = DenseMixedState{ComplexF64}(ws, spec.dims; batch_size = B)
            out = DenseMixedState{ComplexF64}(ws, spec.dims; batch_size = B)
            CuDensityMat.allocate_storage!(ρ)
            CuDensityMat.allocate_storage!(out)
            copyto!(ρ.storage, x)
            c = LindbladRK4Cache(ws, L, ρ)
            lindblad_rhs!(ws, c, out, ρ, t; params_fn = (Q, τ) -> (Q .= P))
            y = Array(out.storage)
            ref = reduce(
                vcat,
                [
                    lindblad_superoperator(spec, t, P[:, b]) * x[((b-1)*n^2+1):(b*n^2)]
                    for b = 1:B
                ],
            )
            @test maxrel(y, ref) < 1e-12
            B > 1 && @test maxrel(
                y[1:(n^2)],
                lindblad_superoperator(spec, t, P[:, 2]) * x[1:(n^2)],
            ) > 1e-2
            @test maxrel(
                y[1:(n^2)],
                lindblad_superoperator(conj_spec(spec), t, P[:, 1]) * x[1:(n^2)],
            ) > 1e-2
        end
        close(ws)
    end

    @gpu_test "GPU builder, one-mode model with callbacks (child process)" begin
        code = """
        using CUDA, CuQuantum, CuQuantum.CuDensityMat, LinearAlgebra
        include($(repr(joinpath(@__DIR__, "lindblad_models.jl"))))
        (name, spec, P, t) = test_models()[3]
        ws = WorkStream(); n = prod(spec.dims); B = size(P, 2)
        for sparsity in (:auto, :dense)
            L = LindbladOperator(ws, spec; batch_size = B, num_params = size(P, 1), sparsity)
            ρ = DenseMixedState{ComplexF64}(ws, spec.dims; batch_size = B); o = DenseMixedState{ComplexF64}(ws, spec.dims; batch_size = B)
            CuDensityMat.allocate_storage!(ρ); CuDensityMat.allocate_storage!(o)
            x = random_states(n, B; seed = 7); copyto!(ρ.storage, x)
            lindblad_rhs!(ws, LindbladRK4Cache(ws, L, ρ), o, ρ, t; params_fn = (Q, τ) -> (Q .= P))
            ref = reduce(vcat, [lindblad_superoperator(spec, t, P[:, b]) * x[(b-1)*n^2+1:b*n^2] for b = 1:B])
            println("RELERR ", maxrel(Array(o.storage), ref))
        end
        """
        out = IOBuffer()
        p = run(
            pipeline(
                ignorestatus(
                    `$(Base.julia_cmd()) --project=$(Base.active_project()) -e $code`,
                );
                stdout = out,
                stderr = out,
            ),
        )
        txt = String(take!(out))
        errs = [parse(Float64, m.captures[1]) for m in eachmatch(r"RELERR (\S+)", txt)]
        p.exitcode == 0 && p.termsignal == 0 ||
            @info "child process failed" p.exitcode p.termsignal last(txt, 2000)
        @test p.exitcode == 0 && p.termsignal == 0
        @test length(errs) == 2 && all(<(1e-12), errs)
    end

    @gpu_test "GPU RK4 == host RK4, exp(Lt), order 4, trace" begin
        ws = WorkStream()
        spec, P = two_mode_model(4; g = 0.35, shift = 1.0, rate1 = 0.05, dephase = 0.02, K = 0.1)
        n = 16
        S = lindblad_superoperator(spec, 0.0, P[:, 1])
        x0 = random_states(n, 1; seed = 11, hermitian = true, unit_trace = true)
        L = LindbladOperator(ws, spec; batch_size = 1, num_params = 1)
        ρ = DenseMixedState{ComplexF64}(ws, spec.dims; batch_size = 1)
        CuDensityMat.allocate_storage!(ρ)
        cache = LindbladRK4Cache(ws, L, ρ)
        gpu(N; T = 1.0) = (
            copyto!(ρ.storage, x0);
            lindblad_rk4!(ws, L, ρ, 0.0, T; nsteps = N, params = P, cache);
            Array(ρ.storage)
        )
        host = lindblad_rk4!(
            CpuLindbladian(spec; num_params = 1),
            copy(x0),
            0.0,
            1.0;
            nsteps = 100,
            params = P,
        )
        @test maxrel(gpu(100), host) < 1e-12
        ref = exp(S) * x0
        e = [maxrel(gpu(N), ref) for N in (50, 100, 200, 400)]
        @test all(15 .< e[1:(end-1)] ./ e[2:end] .< 18)
        @test maxrel(gpu(4000), ref) < 1e-10
        drift = Ref(0.0)
        copyto!(ρ.storage, x0)
        lindblad_rk4!(
            ws,
            L,
            ρ,
            0.0,
            2.0;
            nsteps = 2000,
            params = P,
            cache,
            callback = (r, t, s) ->
                (drift[] = max(drift[], abs(CuDensityMat.trace(r)[1] - 1))),
        )
        @test drift[] < 1e-12
        close(ws)
    end

    @gpu_test "GPU stiffness estimate + divergence at 1.1 h_stab" begin
        ws = WorkStream()
        for shift in (0.0, 4.0)
            c = 10
            spec = single_mode_loss(c, shift)
            dense = maximum(abs, eigvals(lindblad_superoperator(spec)))
            L = LindbladOperator(ws, spec)
            est = lindblad_lambda_max(ws, L)
            @test abs(est.abs[1] / dense - 1) < 0.10
            hstab = rk4_stable_step(est.lambda)
            x0 = random_states(c, 1; seed = 2, hermitian = true, unit_trace = true)
            ρ = DenseMixedState{ComplexF64}(ws, spec.dims)
            CuDensityMat.allocate_storage!(ρ)
            cache = LindbladRK4Cache(ws, L, ρ)
            grow(f) = (
                copyto!(ρ.storage, x0);
                lindblad_rk4!(ws, L, ρ, 0.0, 300 * f * hstab; nsteps = 300, cache);
                LinearAlgebra.norm(Array(ρ.storage)) / LinearAlgebra.norm(x0)
            )
            @test grow(1.1) > 1e6
            @test grow(0.9) < 2
        end
        close(ws)
    end

    @gpu_test "GPU batched members independent, per-member params; no device allocation" begin
        ws = WorkStream()
        spec, _ = two_mode_model(
            4;
            g = (t, p) -> p[1] * cos(p[2] * t),
            shift = 1.0,
            rate1 = 0.05,
            dephase = 0.02,
            rate = (t, p) -> p[3],
        )
        P = [0.35 -0.2 0.6; 1.0 3.0 0.5; 0.1 0.4 0.0]
        B = 3
        n = 16
        x0 = random_states(n, B; seed = 13, hermitian = true, unit_trace = true)
        LB = LindbladOperator(ws, spec; batch_size = B, num_params = 3)
        ρ = DenseMixedState{ComplexF64}(ws, spec.dims; batch_size = B)
        CuDensityMat.allocate_storage!(ρ)
        copyto!(ρ.storage, x0)
        cache = LindbladRK4Cache(ws, LB, ρ)
        lindblad_rk4!(ws, LB, ρ, 0.0, 0.8; nsteps = 200, params = P, cache)
        xb = Array(ρ.storage)
        host = lindblad_rk4!(
            CpuLindbladian(spec; batch_size = B, num_params = 3),
            copy(x0),
            0.0,
            0.8;
            nsteps = 200,
            params = P,
        )
        @test maxrel(xb, host) < 1e-12
        L1 = LindbladOperator(ws, spec; batch_size = 1, num_params = 3)
        ρ1 = DenseMixedState{ComplexF64}(ws, spec.dims; batch_size = 1)
        CuDensityMat.allocate_storage!(ρ1)
        for b = 1:B
            idx = ((b-1)*n^2+1):(b*n^2)
            copyto!(ρ1.storage, x0[idx])
            lindblad_rk4!(ws, L1, ρ1, 0.0, 0.8; nsteps = 200, params = P[:, b:b])
            @test maxrel(xb[idx], Array(ρ1.storage)) < 1e-12
        end
        copyto!(ρ.storage, x0)
        lindblad_rk4!(ws, LB, ρ, 0.0, 0.8; nsteps = 200, params = P[:, [2, 1, 3]], cache)
        @test maxrel(Array(ρ.storage)[1:(n^2)], xb[1:(n^2)]) > 1e-3
        # steady state: a warm cached integration allocates no device memory
        copyto!(ρ.storage, x0)
        lindblad_rk4!(ws, LB, ρ, 0.0, 0.01; nsteps = 2, params = P, cache)
        dev = CuDensityMat.CUDACore.@allocated lindblad_rk4!(
            ws,
            LB,
            ρ,
            0.0,
            0.1;
            nsteps = 20,
            params = P,
            cache,
        )
        @test dev == 0
        host_bytes =
            @allocated lindblad_rk4!(ws, LB, ρ, 0.0, 0.1; nsteps = 20, params = P, cache)
        @info "host bytes per RK4 step (batched callbacks on)" host_bytes / 20
        close(ws)
    end
end
