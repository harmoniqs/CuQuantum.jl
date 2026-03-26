# =============================================================================
# Benchmark: QuantumToolbox.jl — CPU mesolve
# =============================================================================
#
# Solves the same dual-rail cavity Lindblad problem as the cuDensityMat
# benchmark using QuantumToolbox.jl's mesolve on CPU.
#
# System: M coupled cavities, Fock truncation d=3
#   H = Σ_m χ n_m(n_m-1) + Σ_{n≠m} κ a_n†a_m
#   L[ρ] = -i[H,ρ] + γ Σ_m (a_m ρ a_m† - ½{n_m, ρ})
#
# Usage:
#   julia --project=benchmark/comparison benchmark/comparison/bench_quantumtoolbox_cpu.jl
#
# Requires: QuantumToolbox.jl, OrdinaryDiffEq

using QuantumToolbox
using OrdinaryDiffEq
using LinearAlgebra
using Statistics
using Printf

# =============================================================================
# Problem construction
# =============================================================================

"""
Build the M-cavity Lindblad problem in QuantumToolbox.jl.
Returns (H, c_ops, rho0, e_ops) ready for mesolve.
"""
function build_system(M::Int, d::Int)
    # Physical parameters (must match cuDensityMat benchmark)
    chi = 2pi * 0.2    # Kerr nonlinearity
    kappa0 = 2pi * 0.1 # coupling strength
    gamma = 0.01       # photon loss rate

    # --- Single-cavity operators ---
    a_single = destroy(d)
    n_single = num(d)
    kerr_single = n_single * (n_single - qeye(d))

    # --- Embed into full tensor product space ---
    function embed(op, mode)
        ops = [i == mode ? op : qeye(d) for i = 1:M]
        return tensor(ops...)
    end

    a_ops = [embed(a_single, m) for m = 1:M]
    n_ops = [embed(n_single, m) for m = 1:M]

    # --- Hamiltonian ---
    H = sum(chi * embed(kerr_single, m) for m = 1:M)
    for n = 1:M, m = 1:M
        n == m && continue
        H += kappa0 * (a_ops[n]' * a_ops[m])
    end

    # --- Collapse operators ---
    c_ops = [sqrt(gamma) * a_ops[m] for m = 1:M]

    # --- Initial state: |1,0,...,0⟩⟨1,0,...,0| ---
    fock_list = [m == 1 ? fock(d, 1) : fock(d, 0) for m = 1:M]
    psi0 = tensor(fock_list...)
    rho0 = ket2dm(psi0)

    # --- Expectation operators: diagonal populations ---
    e_ops = [a_ops[m]' * a_ops[m] for m = 1:M]

    return H, c_ops, rho0, e_ops
end

# =============================================================================
# Benchmark: full mesolve simulation
# =============================================================================

"""
Run mesolve for M cavities with d Fock levels.
Returns (wall_time_seconds, final_populations).
"""
function bench_mesolve(M::Int, d::Int; n_steps::Int = 100, dt::Float64 = 0.01)
    D = d^M
    t_final = n_steps * dt
    tlist = range(0, t_final, length = n_steps + 1)

    H, c_ops, rho0, e_ops = build_system(M, d)

    # Use DP5 (Dormand-Prince 5th order) with matched tolerances
    # disable progress bar and state storage for speed
    options = Dict(:abstol => 1e-8, :reltol => 1e-6, :save_everystep => false)

    # Warmup (JIT compilation)
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

    # Timed run
    GC.gc()
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
    t_end = time_ns()
    wall_time = (t_end - t_start) / 1e9

    # Extract final populations
    final_pops = [real(sol.expect[m, end]) for m = 1:M]

    return wall_time, final_pops
end

# =============================================================================
# Benchmark: single L[rho] action (for comparison with cuDensityMat)
# =============================================================================

"""
Time a single Liouvillian action L * vec(rho) using QuantumToolbox's internal
superoperator representation.
"""
function bench_single_action(M::Int, d::Int; n_warmup::Int = 5, n_trials::Int = 50)
    D = d^M

    H, c_ops, rho0, _ = build_system(M, d)

    # Build Liouvillian superoperator explicitly
    L = liouvillian(H, c_ops)

    # Vectorized initial state
    rho_vec = operator_to_vector(rho0)
    out_vec = similar(rho_vec)

    # Get the underlying sparse matrix for direct mul!
    L_data = L.data
    rho_data = rho_vec.data

    # Warmup
    for _ = 1:n_warmup
        out = L_data * rho_data
    end

    # Timed runs
    times = Float64[]
    for _ = 1:n_trials
        t0 = time_ns()
        out = L_data * rho_data
        t1 = time_ns()
        push!(times, (t1 - t0) / 1e6)  # ms
    end

    return median(times), minimum(times), maximum(times)
end

# =============================================================================
# Main
# =============================================================================

function main()
    d = 3
    M_values = [2, 4, 6]

    # Check if larger sizes are feasible (CPU memory: D^2 * D^2 * 16 bytes for Liouvillian)
    for M_try in [8]
        D_try = d^M_try
        liouv_nnz_est = M_try * d^2 * D_try^2  # rough non-zero count
        mem_est = liouv_nnz_est * 24  # CSC: value + row_idx + col_ptr
        if mem_est < Sys.total_memory() * 0.5
            push!(M_values, M_try)
        else
            @printf(
                "Skipping M=%d (D=%d): estimated Liouvillian memory ~%.1f GB\n",
                M_try,
                D_try,
                mem_est / 1e9
            )
        end
    end

    n_steps = 100
    dt = 0.01

    println("=" ^ 78)
    println("Benchmark: QuantumToolbox.jl CPU mesolve")
    println("  System: M coupled cavities, d=$d Fock truncation")
    println("  Time integration: DP5 adaptive (atol=1e-8, rtol=1e-6)")
    println("  Simulation: $(n_steps) steps, dt=$(dt), t_final=$(n_steps*dt)")
    println(
        "  CPU: $(Sys.CPU_THREADS) threads, $(round(Sys.total_memory()/1e9, digits=1)) GB RAM",
    )
    println("=" ^ 78)
    println()

    # --- Single action benchmark ---
    println("--- Single Liouvillian action L*vec(rho) ---")
    for M in M_values
        D = d^M
        @printf("  M=%d (D=%d, Liouv=%dx%d): ", M, D, D^2, D^2)
        try
            med, mn, mx = bench_single_action(M, d)
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
            wall_time, pops = bench_mesolve(M, d; n_steps = n_steps, dt = dt)
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
    csv_file = joinpath(@__DIR__, "results_quantumtoolbox_cpu.csv")
    open(csv_file, "w") do io
        println(io, "framework,backend,M,D,rho_elements,wall_time_s,n_steps,dt")
        for r in results
            println(
                io,
                "QuantumToolbox.jl,CPU,$(r.M),$(r.D),$(r.D^2),$(r.wall_time_s),$(n_steps),$(dt)",
            )
        end
    end
    println("Results saved to $csv_file")
end

main()
