# =============================================================================
# Benchmark: Batched Lindblad evolution
# =============================================================================
#
# Two use cases for batched density matrix evolution:
#
#   1. Trajectory parallelism: same system, different time points along a
#      control pulse. Each batch member evaluates L[ρ] at a different t,
#      useful for parallelizing RK4 stages or shooting methods.
#
#   2. Parameter sweep: same system structure, different control parameters
#      per batch member (grid search over coupling strengths, drive amplitudes).
#      Each batch member has its own θ vector.
#
# Compares:
#   A. cuDensityMat native batching (batch_size > 1, single kernel launch)
#   B. cuDensityMat sequential (batch_size=1, loop over members)
#   C. cuSPARSE sequential (build Liouvillian at each param, SpMV)
#
# System: M coupled cavities, Fock truncation d=3
#   H(t,θ) = Σ_m χ n_m(n_m-1) + θ_coupling(t) Σ_{n≠m} a_n†a_m
#   L[ρ]   = -i[H,ρ] + γ Σ_m (a_m ρ a_m† - ½{n_m, ρ})
#
# The coupling strength θ_coupling varies across batch members.
#
# Usage:
#   julia --project=. benchmark/batched/bench_batched.jl

using CUDA
using CuQuantum
using CuQuantum.CuDensityMat
using LinearAlgebra
using Statistics
using Printf

# Try to load QuantumToolbox for cuSPARSE comparison
const HAS_QT = try
    @eval using QuantumToolbox
    @eval using CUDA.CUSPARSE
    true
catch
    false
end

# =============================================================================
# Helper: generate kappa values for a batch
# =============================================================================

function kappa_values_for_batch(batch_size::Int)
    if batch_size == 1
        return [2pi * 0.1]
    else
        return collect(range(2pi * 0.05, 2pi * 0.5; length = batch_size))
    end
end

# =============================================================================
# System construction: cuDensityMat with batched callback
# =============================================================================

function build_batched_system(M::Int, d::Int, batch_size::Int)
    T = ComplexF64
    ws = WorkStream()
    dims = fill(d, M)

    chi = 2pi * 0.2
    gamma = 0.01

    # Single-cavity matrices
    a_matrix = zeros(Float64, d, d)
    for n = 1:(d-1)
        a_matrix[n, n+1] = sqrt(n)
    end
    a_dag = transpose(a_matrix)
    n_mat = diagm(0 => T.(0:(d-1)))
    kerr_mat = diagm(0 => T.([n * (n - 1) for n = 0:(d-1)]))

    # Elementary operators
    elem_kerr =
        CuDensityMat.create_elementary_operator(ws, [d], CUDA.CuVector{T}(vec(kerr_mat)))
    elem_n = CuDensityMat.create_elementary_operator(ws, [d], CUDA.CuVector{T}(vec(n_mat)))
    elem_a = CuDensityMat.create_elementary_operator(
        ws,
        [d],
        CUDA.CuVector{T}(vec(T.(a_matrix))),
    )
    elem_a_dag =
        CuDensityMat.create_elementary_operator(ws, [d], CUDA.CuVector{T}(vec(T.(a_dag))))

    # Dissipation: fused a⊗a† for sandwich
    aa_dag_fused = zeros(T, d, d, d, d)
    for j1 = 1:d, j0 = 1:d, i1 = 1:d, i0 = 1:d
        aa_dag_fused[i0, i1, j0, j1] = a_matrix[i0, j0] * a_dag[i1, j1]
    end
    elem_aa_dag = CuDensityMat.create_elementary_operator(
        ws,
        [d, d],
        CUDA.CuVector{T}(vec(aa_dag_fused)),
    )

    # === Kerr term (static) ===
    kerr_term = CuDensityMat.create_operator_term(ws, dims)
    for m = 0:(M-1)
        CuDensityMat.append_elementary_product!(
            kerr_term,
            [elem_kerr],
            Int32[m],
            Int32[0];
            coefficient = ComplexF64(chi),
        )
    end

    # === Coupling term (per-batch coefficient via callback) ===
    coupling_term = CuDensityMat.create_operator_term(ws, dims)
    for n = 0:(M-1), m = 0:(M-1)
        n == m && continue
        CuDensityMat.append_elementary_product!(
            coupling_term,
            [elem_a_dag, elem_a],
            Int32[n, m],
            Int32[0, 0];
            coefficient = ComplexF64(1.0),
        )
    end

    # Callback: coupling = params[1, b] (different per batch member)
    function coupling_coeff(time, params, storage)
        for b in eachindex(storage)
            storage[b] = complex(params[1, b], 0.0)
        end
    end
    cb, gcb, cb_refs = CuDensityMat.wrap_scalar_callback(coupling_coeff)

    # === Dissipation terms (static) ===
    diss_sandwich = CuDensityMat.create_operator_term(ws, dims)
    for m = 0:(M-1)
        CuDensityMat.append_elementary_product!(
            diss_sandwich,
            [elem_aa_dag],
            Int32[m, m],
            Int32[0, 1];
            coefficient = ComplexF64(1.0),
        )
    end

    diss_number = CuDensityMat.create_operator_term(ws, dims)
    for m = 0:(M-1)
        CuDensityMat.append_elementary_product!(
            diss_number,
            [elem_n],
            Int32[m],
            Int32[0];
            coefficient = ComplexF64(1.0),
        )
    end

    # === Assemble Liouvillian ===
    liouvillian = CuDensityMat.create_operator(ws, dims)
    CuDensityMat.append_term!(
        liouvillian,
        kerr_term;
        duality = 0,
        coefficient = ComplexF64(0, -1),
    )
    CuDensityMat.append_term!(
        liouvillian,
        kerr_term;
        duality = 1,
        coefficient = ComplexF64(0, +1),
    )
    CuDensityMat.append_term!(
        liouvillian,
        coupling_term;
        duality = 0,
        coefficient = ComplexF64(0, -1),
        coefficient_callback = cb,
        coefficient_gradient_callback = gcb,
    )
    CuDensityMat.append_term!(
        liouvillian,
        coupling_term;
        duality = 1,
        coefficient = ComplexF64(0, +1),
        coefficient_callback = cb,
        coefficient_gradient_callback = gcb,
    )
    CuDensityMat.append_term!(
        liouvillian,
        diss_sandwich;
        duality = 0,
        coefficient = ComplexF64(gamma),
    )
    CuDensityMat.append_term!(
        liouvillian,
        diss_number;
        duality = 0,
        coefficient = ComplexF64(-gamma / 2),
    )
    CuDensityMat.append_term!(
        liouvillian,
        diss_number;
        duality = 1,
        coefficient = ComplexF64(-gamma / 2),
    )

    # === Batched states ===
    D = d^M
    rho_in = DenseMixedState{T}(ws, Tuple(dims); batch_size = batch_size)
    rho_out = DenseMixedState{T}(ws, Tuple(dims); batch_size = batch_size)
    CuDensityMat.allocate_storage!(rho_in)
    CuDensityMat.allocate_storage!(rho_out)

    # Initialize all batch members to |1,0,...,0><1,0,...,0|
    rho_init = zeros(T, D * D)
    rho_init[2+D*(2-1)] = 1.0
    rho_init_batched = repeat(rho_init, batch_size)
    copyto!(rho_in.storage, CUDA.CuVector{T}(rho_init_batched))

    # Prepare action
    CuDensityMat.prepare_operator_action!(ws, liouvillian, rho_in, rho_out)

    return ws, liouvillian, rho_in, rho_out, cb_refs, D
end

# =============================================================================
# Benchmark A: cuDensityMat native batching
# =============================================================================

function bench_cudensitymat_batched(
    M::Int,
    d::Int,
    batch_size::Int;
    n_warmup::Int = 3,
    n_trials::Int = 20,
)
    ws, liouvillian, rho_in, rho_out, cb_refs, D = build_batched_system(M, d, batch_size)
    kappas = kappa_values_for_batch(batch_size)
    params = CUDA.CuVector{Float64}(kappas)

    for _ = 1:n_warmup
        CuDensityMat.initialize_zero!(rho_out)
        CuDensityMat.compute_operator_action!(
            ws,
            liouvillian,
            rho_in,
            rho_out;
            time = 0.0,
            batch_size = batch_size,
            num_params = 1,
            params = params,
        )
    end
    CUDA.synchronize()

    times = Float64[]
    for _ = 1:n_trials
        CuDensityMat.initialize_zero!(rho_out)
        CUDA.synchronize()
        t0 = time_ns()
        CuDensityMat.compute_operator_action!(
            ws,
            liouvillian,
            rho_in,
            rho_out;
            time = 0.0,
            batch_size = batch_size,
            num_params = 1,
            params = params,
        )
        CUDA.synchronize()
        t1 = time_ns()
        push!(times, (t1 - t0) / 1e6)
    end

    CuDensityMat.unregister_callback!(cb_refs)
    close(ws)
    return median(times), minimum(times), maximum(times)
end

# =============================================================================
# Benchmark B: cuDensityMat sequential (batch_size=1, loop)
# =============================================================================

function bench_cudensitymat_sequential(
    M::Int,
    d::Int,
    batch_size::Int;
    n_warmup::Int = 3,
    n_trials::Int = 10,
)
    ws, liouvillian, rho_in, rho_out, cb_refs, D = build_batched_system(M, d, 1)
    kappas = kappa_values_for_batch(batch_size)

    params = CUDA.CuVector{Float64}([kappas[1]])
    for _ = 1:n_warmup
        CuDensityMat.initialize_zero!(rho_out)
        CuDensityMat.compute_operator_action!(
            ws,
            liouvillian,
            rho_in,
            rho_out;
            time = 0.0,
            batch_size = 1,
            num_params = 1,
            params = params,
        )
    end
    CUDA.synchronize()

    times = Float64[]
    for _ = 1:n_trials
        CUDA.synchronize()
        t0 = time_ns()
        for b = 1:batch_size
            params_b = CUDA.CuVector{Float64}([kappas[b]])
            CuDensityMat.initialize_zero!(rho_out)
            CuDensityMat.compute_operator_action!(
                ws,
                liouvillian,
                rho_in,
                rho_out;
                time = 0.0,
                batch_size = 1,
                num_params = 1,
                params = params_b,
            )
        end
        CUDA.synchronize()
        t1 = time_ns()
        push!(times, (t1 - t0) / 1e6)
    end

    CuDensityMat.unregister_callback!(cb_refs)
    close(ws)
    return median(times), minimum(times), maximum(times)
end

# =============================================================================
# Benchmark C: cuSPARSE sequential (build L at each param, SpMV)
# =============================================================================

function bench_cusparse_sequential(
    M::Int,
    d::Int,
    batch_size::Int;
    n_warmup::Int = 3,
    n_trials::Int = 10,
)
    HAS_QT || return NaN, NaN, NaN

    T = ComplexF64
    D = d^M
    chi = 2pi * 0.2
    gamma = 0.01
    kappas = kappa_values_for_batch(batch_size)

    a_single = QuantumToolbox.destroy(d)
    n_single = QuantumToolbox.num(d)
    kerr_single = n_single * (n_single - QuantumToolbox.qeye(d))

    function embed(op, mode)
        ops = [i == mode ? op : QuantumToolbox.qeye(d) for i = 1:M]
        return QuantumToolbox.tensor(ops...)
    end

    a_ops = [embed(a_single, m) for m = 1:M]

    function build_L(kappa)
        H = sum(chi * embed(kerr_single, m) for m = 1:M)
        for n = 1:M, m = 1:M
            n == m && continue
            H += kappa * (a_ops[n]' * a_ops[m])
        end
        c_ops = [sqrt(gamma) * a_ops[m] for m = 1:M]
        L = QuantumToolbox.liouvillian(H, c_ops)
        return CUSPARSE.CuSparseMatrixCSC(L.data)
    end

    # Pre-build all Liouvillians
    L_gpus = [build_L(kappas[b]) for b = 1:batch_size]

    # Initial state
    fock_list = [m == 1 ? QuantumToolbox.fock(d, 1) : QuantumToolbox.fock(d, 0) for m = 1:M]
    psi0 = QuantumToolbox.tensor(fock_list...)
    rho0 = QuantumToolbox.ket2dm(psi0)
    rho_vec = CUDA.CuVector{T}(QuantumToolbox.operator_to_vector(rho0).data)

    for _ = 1:n_warmup
        out = L_gpus[1] * rho_vec
    end
    CUDA.synchronize()

    times = Float64[]
    for _ = 1:n_trials
        CUDA.synchronize()
        t0 = time_ns()
        for b = 1:batch_size
            out = L_gpus[b] * rho_vec
        end
        CUDA.synchronize()
        t1 = time_ns()
        push!(times, (t1 - t0) / 1e6)
    end

    return median(times), minimum(times), maximum(times)
end

# =============================================================================
# Main
# =============================================================================

function main()
    d = 3
    M_values = [2, 4, 6]
    batch_sizes = [1, 4, 16, 64, 256]

    gpu_name = CUDA.functional() ? CUDA.name(CUDA.device()) : "none"
    free = CUDA.functional() ? CUDA.free_memory() : 0

    println("=" ^ 78)
    println("Benchmark: Batched Lindblad evolution")
    println("  System: M coupled cavities, d=$d Fock truncation")
    println("  GPU: $gpu_name ($(round(free/1e9, digits=1)) GB free)")
    println("  Batch: different coupling strength κ per member (parameter sweep)")
    println("  QuantumToolbox.jl: $(HAS_QT ? "available" : "not available")")
    println("=" ^ 78)
    println()

    for M in M_values
        D = d^M
        println("─── M=$M (D=$D, ρ: $(D)×$(D) = $(D^2) elements) ───")
        println()

        @printf(
            "  %6s │ %12s │ %12s │ %12s │ %8s\n",
            "Batch",
            "cuDM batched",
            "cuDM seq",
            "cuSPARSE seq",
            "Speedup"
        )
        @printf(
            "  %6s │ %12s │ %12s │ %12s │ %8s\n",
            "",
            "(ms)",
            "(ms)",
            "(ms)",
            "batch/seq"
        )
        println("  " * "─" ^ 65)

        for B in batch_sizes
            # Memory check
            rho_bytes = B * D^2 * 16 * 2
            if rho_bytes > free * 0.5
                @printf("  %6d │ %12s │ %12s │ %12s │ %8s\n", B, "OOM", "—", "—", "—")
                continue
            end

            # A: cuDensityMat batched
            t_batched = try
                med, _, _ = bench_cudensitymat_batched(M, d, B)
                med
            catch e
                NaN
            end

            # B: cuDensityMat sequential
            t_seq = try
                med, _, _ = bench_cudensitymat_sequential(M, d, B)
                med
            catch e
                NaN
            end

            # C: cuSPARSE sequential
            t_sparse = if M <= 6 && B <= 64
                try
                    med, _, _ = bench_cusparse_sequential(M, d, B)
                    med
                catch e
                    NaN
                end
            else
                NaN
            end

            speedup = isnan(t_seq) || isnan(t_batched) ? NaN : t_seq / t_batched

            @printf(
                "  %6d │ %10.2f ms │ %10.2f ms │ %10.2f ms │ %7.1f×\n",
                B,
                isnan(t_batched) ? NaN : t_batched,
                isnan(t_seq) ? NaN : t_seq,
                isnan(t_sparse) ? NaN : t_sparse,
                isnan(speedup) ? NaN : speedup,
            )
        end
        println()
    end
end

main()
