# =============================================================================
# Benchmark: QuantumToolbox.jl — GPU mesolve (CUDA sparse)
# =============================================================================
#
# Same problem as bench_quantumtoolbox_cpu.jl but with operators converted
# to GPU via cu(). Uses CuSparseMatrixCSC for sparse ops on GPU.
#
# Usage:
#   julia --project=benchmark/comparison benchmark/comparison/bench_quantumtoolbox_gpu.jl
#
# Requires: QuantumToolbox.jl, CUDA.jl, OrdinaryDiffEq

using QuantumToolbox
using OrdinaryDiffEq
using CUDA
using LinearAlgebra
using Statistics
using Printf

CUDA.allowscalar(false)

# =============================================================================
# Problem construction (GPU)
# =============================================================================

"""
Build the M-cavity Lindblad problem with GPU-resident operators.
"""
function build_system_gpu(M::Int, d::Int)
    chi = 2pi * 0.2
    kappa0 = 2pi * 0.1
    gamma = 0.01

    a_single = destroy(d)
    n_single = num(d)
    kerr_single = n_single * (n_single - qeye(d))

    function embed(op, mode)
        ops = [i == mode ? op : qeye(d) for i = 1:M]
        return tensor(ops...)
    end

    # Build on CPU first, then convert to GPU
    a_cpu = [embed(a_single, m) for m = 1:M]
    n_cpu = [embed(n_single, m) for m = 1:M]

    H_cpu = sum(chi * embed(kerr_single, m) for m = 1:M)
    for n = 1:M, m = 1:M
        n == m && continue
        H_cpu += kappa0 * (a_cpu[n]' * a_cpu[m])
    end

    c_ops_cpu = [sqrt(gamma) * a_cpu[m] for m = 1:M]

    fock_list = [m == 1 ? fock(d, 1) : fock(d, 0) for m = 1:M]
    psi0_cpu = tensor(fock_list...)
    rho0_cpu = ket2dm(psi0_cpu)

    # Convert to GPU
    H_gpu = cu(H_cpu)
    c_ops_gpu = [cu(c) for c in c_ops_cpu]
    rho0_gpu = cu(rho0_cpu)
    e_ops_gpu = [cu(a_cpu[m]' * a_cpu[m]) for m = 1:M]

    return H_gpu, c_ops_gpu, rho0_gpu, e_ops_gpu
end

# =============================================================================
# Benchmark: full mesolve on GPU
# =============================================================================

"""
Run GPU-accelerated mesolve.
Returns (wall_time_seconds, final_populations).
"""
function bench_mesolve_gpu(M::Int, d::Int; n_steps::Int = 100, dt::Float64 = 0.01)
    D = d^M
    t_final = n_steps * dt
    tlist = range(0, t_final, length = n_steps + 1)

    H, c_ops, rho0, e_ops = build_system_gpu(M, d)

    options = Dict(:abstol => 1e-8, :reltol => 1e-6, :save_everystep => false)

    # Warmup (JIT + GPU kernel compilation)
    _ = mesolve(
        H,
        rho0,
        range(0, 0.02, length = 3),
        c_ops;
        e_ops = e_ops,
        alg = DP5(),
        progress_bar = Val(false),
        options...,
    )
    CUDA.synchronize()

    # Timed run
    GC.gc()
    CUDA.synchronize()
    t_start = time_ns()
    sol = mesolve(
        H,
        rho0,
        tlist,
        c_ops;
        e_ops = e_ops,
        alg = DP5(),
        progress_bar = Val(false),
        options...,
    )
    CUDA.synchronize()
    t_end = time_ns()
    wall_time = (t_end - t_start) / 1e9

    final_pops = [real(sol.expect[m, end]) for m = 1:M]
    return wall_time, final_pops
end

# =============================================================================
# Benchmark: single Liouvillian action on GPU
# =============================================================================

function bench_single_action_gpu(M::Int, d::Int; n_warmup::Int = 5, n_trials::Int = 50)
    D = d^M

    # Build CPU operators, get Liouvillian, convert to GPU
    chi = 2pi * 0.2
    kappa0 = 2pi * 0.1
    gamma = 0.01
    a_single = destroy(d)
    n_single = num(d)
    kerr_single = n_single * (n_single - qeye(d))

    function embed(op, mode)
        ops = [i == mode ? op : qeye(d) for i = 1:M]
        return tensor(ops...)
    end

    a_ops = [embed(a_single, m) for m = 1:M]
    H = sum(chi * embed(kerr_single, m) for m = 1:M)
    for n = 1:M, m = 1:M
        n == m && continue
        H += kappa0 * (a_ops[n]' * a_ops[m])
    end
    c_ops = [sqrt(gamma) * a_ops[m] for m = 1:M]

    # Build Liouvillian on CPU, convert to GPU sparse
    L_cpu = liouvillian(H, c_ops)
    L_gpu = cu(L_cpu)

    fock_list = [m == 1 ? fock(d, 1) : fock(d, 0) for m = 1:M]
    psi0 = tensor(fock_list...)
    rho0 = ket2dm(psi0)
    rho_vec = cu(operator_to_vector(rho0))

    L_data = L_gpu.data
    rho_data = rho_vec.data

    # Warmup
    for _ = 1:n_warmup
        out = L_data * rho_data
    end
    CUDA.synchronize()

    # Timed runs
    times = Float64[]
    for _ = 1:n_trials
        CUDA.synchronize()
        t0 = time_ns()
        out = L_data * rho_data
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
    if !CUDA.functional()
        println("ERROR: CUDA not functional. GPU benchmarks require a CUDA-capable GPU.")
        return
    end

    d = 3
    M_values = [2, 4, 6]

    # Check GPU memory for larger sizes
    free = CUDA.free_memory()
    for M_try in [8]
        D_try = d^M_try
        # Rough estimate: Liouvillian sparse on GPU + state vectors
        # CSR nnz ~ M * d^2 * D^2, each entry = 24 bytes (val + indices)
        liouv_nnz_est = M_try * d^2 * D_try^2
        mem_est = liouv_nnz_est * 24 + D_try^2 * 16 * 4  # sparse + states
        if mem_est < free * 0.8
            push!(M_values, M_try)
        else
            @printf(
                "Skipping M=%d (D=%d): estimated GPU memory ~%.1f GB, free=%.1f GB\n",
                M_try,
                D_try,
                mem_est / 1e9,
                free / 1e9
            )
        end
    end

    n_steps = 100
    dt = 0.01

    gpu_name = CUDA.name(CUDA.device())
    println("=" ^ 78)
    println("Benchmark: QuantumToolbox.jl GPU mesolve (CUDA sparse)")
    println("  System: M coupled cavities, d=$d Fock truncation")
    println("  GPU: $gpu_name ($(round(free/1e9, digits=1)) GB free)")
    println("  Time integration: DP5 adaptive (atol=1e-8, rtol=1e-6)")
    println("  Simulation: $(n_steps) steps, dt=$(dt)")
    println("=" ^ 78)
    println()

    # --- Single action benchmark ---
    println("--- Single Liouvillian action L*vec(rho) on GPU ---")
    for M in M_values
        D = d^M
        @printf("  M=%d (D=%d): ", M, D)
        try
            med, mn, mx = bench_single_action_gpu(M, d)
            @printf("%.3f ms (min=%.3f, max=%.3f)\n", med, mn, mx)
        catch e
            println("FAILED: $e")
        end
    end
    println()

    # --- Full mesolve benchmark ---
    println("--- Full mesolve simulation ($n_steps steps) ---")
    results = []
    for M in M_values
        D = d^M
        @printf("  M=%d (D=%d): ", M, D)
        try
            wall_time, pops = bench_mesolve_gpu(M, d; n_steps = n_steps, dt = dt)
            @printf("%.4f s\n", wall_time)
            @printf("    Final <n>: %s\n", join([@sprintf("%.4f", p) for p in pops], ", "))
            push!(results, (M = M, D = D, wall_time_s = wall_time, pops = pops))
        catch e
            println("FAILED: $e")
            push!(results, (M = M, D = D, wall_time_s = NaN, pops = Float64[]))
        end
    end
    println()

    # --- Save CSV ---
    csv_file = joinpath(@__DIR__, "results_quantumtoolbox_gpu.csv")
    open(csv_file, "w") do io
        println(io, "framework,backend,M,D,rho_elements,wall_time_s,n_steps,dt")
        for r in results
            println(
                io,
                "QuantumToolbox.jl,GPU-$(gpu_name),$(r.M),$(r.D),$(r.D^2),$(r.wall_time_s),$(n_steps),$(dt)",
            )
        end
    end
    println("Results saved to $csv_file")
end

main()
