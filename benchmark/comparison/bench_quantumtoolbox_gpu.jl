# =============================================================================
# Benchmark: QuantumToolbox.jl — GPU (cuSPARSE Liouvillian + manual RK4)
# =============================================================================
#
# Same problem as all other benchmarks. Builds the Liouvillian superoperator
# using QuantumToolbox.jl on CPU, transfers to GPU as CuSparseMatrixCSC,
# then runs manual RK4 via cuSPARSE SpMV.
#
# This avoids the OrdinaryDiffEq.jl GPU kernel compilation bug (PTX .NaN
# modifier) that prevents mesolve from running on GPU.
#
# Usage:
#   julia --project=. benchmark/comparison/bench_quantumtoolbox_gpu.jl
#
# Requires: QuantumToolbox.jl, CUDA.jl

using QuantumToolbox
using CUDA
using CUDA.CUSPARSE
using LinearAlgebra
using Statistics
using Printf

CUDA.allowscalar(false)

# =============================================================================
# Build Liouvillian on GPU
# =============================================================================

"""
Build the M-cavity Liouvillian as a GPU sparse matrix.
Returns (L_gpu, rho0_gpu_vec, D) where L_gpu is CuSparseMatrixCSC{ComplexF64}
and rho0_gpu_vec is CuVector{ComplexF64} (vectorized density matrix).
"""
function build_liouvillian_gpu(M::Int, d::Int)
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

    # Hamiltonian
    H = sum(chi * embed(kerr_single, m) for m = 1:M)
    for n = 1:M, m = 1:M
        n == m && continue
        H += kappa0 * (a_ops[n]' * a_ops[m])
    end

    # Collapse operators
    c_ops = [sqrt(gamma) * a_ops[m] for m = 1:M]

    # Build Liouvillian superoperator on CPU (sparse)
    L_cpu = liouvillian(H, c_ops)

    # Transfer to GPU
    L_gpu = CUSPARSE.CuSparseMatrixCSC(L_cpu.data)

    # Initial state: |1,0,...,0><1,0,...,0|
    fock_list = [m == 1 ? fock(d, 1) : fock(d, 0) for m = 1:M]
    psi0 = tensor(fock_list...)
    rho0 = ket2dm(psi0)
    rho0_vec = CUDA.CuVector{ComplexF64}(operator_to_vector(rho0).data)

    D = d^M
    return L_gpu, rho0_vec, D
end

# =============================================================================
# Benchmark: single L*vec(rho) action
# =============================================================================

function bench_single_action_gpu(M::Int, d::Int; n_warmup::Int = 5, n_trials::Int = 50)
    L_gpu, rho_vec, D = build_liouvillian_gpu(M, d)

    # Warmup
    for _ = 1:n_warmup
        out = L_gpu * rho_vec
    end
    CUDA.synchronize()

    # Timed runs
    times = Float64[]
    for _ = 1:n_trials
        CUDA.synchronize()
        t0 = time_ns()
        out = L_gpu * rho_vec
        CUDA.synchronize()
        t1 = time_ns()
        push!(times, (t1 - t0) / 1e6)
    end

    return median(times), minimum(times), maximum(times)
end

# =============================================================================
# Benchmark: full RK4 simulation via cuSPARSE SpMV
# =============================================================================

"""
Run manual RK4 with cuSPARSE Liouvillian.
Same integrator as the CuQuantum.jl benchmark for fair comparison.
Returns (wall_time_seconds, trace).
"""
function bench_rk4_gpu(M::Int, d::Int; n_steps::Int = 100, dt::Float64 = 0.01)
    L_gpu, rho0_vec, D = build_liouvillian_gpu(M, d)

    # Allocate scratch vectors for RK4
    rho = copy(rho0_vec)
    k1 = similar(rho)
    k2 = similar(rho)
    k3 = similar(rho)
    k4 = similar(rho)
    rho_tmp = similar(rho)

    # Warmup: one L*rho
    mul!(k1, L_gpu, rho)
    CUDA.synchronize()

    # Reset state
    copyto!(rho, rho0_vec)
    CUDA.synchronize()

    # Timed RK4 loop
    GC.gc()
    CUDA.synchronize()
    t_start = time_ns()

    for step = 1:n_steps
        # k1 = L * rho
        mul!(k1, L_gpu, rho)

        # rho_tmp = rho + dt/2 * k1
        copyto!(rho_tmp, rho)
        axpy!(dt / 2, k1, rho_tmp)
        # k2 = L * rho_tmp
        mul!(k2, L_gpu, rho_tmp)

        # rho_tmp = rho + dt/2 * k2
        copyto!(rho_tmp, rho)
        axpy!(dt / 2, k2, rho_tmp)
        # k3 = L * rho_tmp
        mul!(k3, L_gpu, rho_tmp)

        # rho_tmp = rho + dt * k3
        copyto!(rho_tmp, rho)
        axpy!(dt, k3, rho_tmp)
        # k4 = L * rho_tmp
        mul!(k4, L_gpu, rho_tmp)

        # rho += dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        axpy!(dt / 6, k1, rho)
        axpy!(dt / 3, k2, rho)
        axpy!(dt / 3, k3, rho)
        axpy!(dt / 6, k4, rho)
    end

    CUDA.synchronize()
    t_end = time_ns()
    wall_time = (t_end - t_start) / 1e9

    # Compute trace: sum of diagonal elements of the D×D density matrix
    rho_cpu = Array(rho)
    trace = real(sum(rho_cpu[(i-1)*D+i] for i = 1:D))

    return wall_time, trace
end

# =============================================================================
# Main
# =============================================================================

function main()
    if !CUDA.functional()
        println("ERROR: CUDA not functional.")
        return
    end

    d = 3
    M_values = [2, 4, 6]

    # Check GPU memory for M=8
    # Liouvillian is D^2 × D^2 sparse. For M=8, D=6561, D^2=43M.
    # CSR with ~M*d^2*D^2 nnz ≈ 8*9*43M ≈ 3.1B entries → ~75 GB. Won't fit.
    free = CUDA.free_memory()
    for M_try in [8]
        D_try = d^M_try
        liouv_nnz_est = M_try * d^2 * D_try^2
        mem_est = liouv_nnz_est * 24 + D_try^2 * 16 * 4
        if mem_est < free * 0.8
            push!(M_values, M_try)
        else
            @printf(
                "Skipping M=%d (D=%d): Liouvillian ~%.1f GB, free=%.1f GB\n",
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
    println("Benchmark: QuantumToolbox.jl GPU (cuSPARSE Liouvillian + manual RK4)")
    println("  System: M coupled cavities, d=$d Fock truncation")
    println("  GPU: $gpu_name ($(round(free/1e9, digits=1)) GB free)")
    println("  Time integration: RK4 fixed-step (dt=$dt)")
    println("  Simulation: $(n_steps) steps, t_final=$(n_steps*dt)")
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

    # --- Full RK4 benchmark ---
    println("--- Full RK4 simulation ($n_steps steps) ---")
    results = []
    for M in M_values
        D = d^M
        @printf("  M=%d (D=%d): ", M, D)
        try
            wall_time, trace = bench_rk4_gpu(M, d; n_steps = n_steps, dt = dt)
            @printf("%.4f s  (Tr(rho)=%.6f)\n", wall_time, trace)
            push!(results, (M = M, D = D, wall_time_s = wall_time, trace = trace))
        catch e
            println("FAILED: $e")
            push!(results, (M = M, D = D, wall_time_s = NaN, trace = 0.0))
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
                "QuantumToolbox.jl-RK4,GPU-$(gpu_name),$(r.M),$(r.D),$(r.D^2),$(r.wall_time_s),$(n_steps),$(dt)",
            )
        end
    end
    println("Results saved to $csv_file")
end

main()
