# =============================================================================
# Benchmark: CuQuantum.jl (cuDensityMat GPU)
# =============================================================================
#
# Same problem as all other comparison benchmarks but using our CuQuantum.jl
# Julia wrapper. Both single-action and full RK4 simulation.
#
# System: M coupled cavities, Fock truncation d=3
#   H = Σ_m χ n_m(n_m-1) + Σ_{n≠m} κ a_n†a_m
#   L[ρ] = -i[H,ρ] + γ Σ_m (a_m ρ a_m† - ½{n_m, ρ})
#
# Usage:
#   julia --project=. benchmark/comparison/bench_cuquantum_jl.jl
#
# Output: results_cuquantum_jl.csv

using CUDA
using CuQuantum
using CuQuantum.CuDensityMat
using LinearAlgebra
using Statistics
using Printf

# =============================================================================
# System construction (reused from run_benchmarks.jl)
# =============================================================================

function build_system(M::Int, d::Int)
    T = ComplexF64
    ws = WorkStream()
    dims = fill(d, M)

    # Single-cavity matrices
    a_matrix = zeros(Float64, d, d)
    for n = 1:(d-1)
        a_matrix[n, n+1] = sqrt(n)
    end
    a_dag = transpose(a_matrix)
    n_mat = diagm(0 => T.(0:(d-1)))
    kerr_mat = diagm(0 => T.([n * (n - 1) for n = 0:(d-1)]))

    n_gpu = CUDA.CuVector{T}(vec(n_mat))
    kerr_gpu = CUDA.CuVector{T}(vec(kerr_mat))

    elem_n = CuDensityMat.create_elementary_operator(ws, [d], n_gpu)
    elem_kerr = CuDensityMat.create_elementary_operator(ws, [d], kerr_gpu)

    # Fused aa_dag for dissipation sandwich
    aa_dag_fused = zeros(T, d, d, d, d)
    for j1 = 1:d, j0 = 1:d, i1 = 1:d, i0 = 1:d
        aa_dag_fused[i0, i1, j0, j1] = a_matrix[i0, j0] * a_dag[i1, j1]
    end
    elem_aa_dag = CuDensityMat.create_elementary_operator(
        ws, [d, d], CUDA.CuVector{T}(vec(aa_dag_fused))
    )

    # Coupling operators
    coupling_elems = Dict{Tuple{Int,Int},Any}()
    for n = 0:(M-1), m = 0:(M-1)
        n == m && continue
        c = zeros(T, d, d, d, d)
        for j1 = 1:d, j0 = 1:d, i1 = 1:d, i0 = 1:d
            c[i0, i1, j0, j1] = a_dag[i0, j0] * a_matrix[i1, j1]
        end
        coupling_elems[(n, m)] =
            CuDensityMat.create_elementary_operator(ws, [d, d], CUDA.CuVector{T}(vec(c)))
    end

    # Parameters
    chi = 2pi * 0.2
    kappa0 = 2pi * 0.1
    gamma = 0.01

    # Kerr term
    kerr_term = CuDensityMat.create_operator_term(ws, dims)
    for m = 0:(M-1)
        CuDensityMat.append_elementary_product!(
            kerr_term, [elem_kerr], Int32[m], Int32[0];
            coefficient = ComplexF64(chi),
        )
    end

    # Coupling term
    coupling_term = CuDensityMat.create_operator_term(ws, dims)
    for n = 0:(M-1), m = 0:(M-1)
        n == m && continue
        CuDensityMat.append_elementary_product!(
            coupling_term, [coupling_elems[(n, m)]], Int32[n, m], Int32[0, 0];
            coefficient = ComplexF64(kappa0),
        )
    end

    # Dissipation sandwich
    diss_sandwich = CuDensityMat.create_operator_term(ws, dims)
    for m = 0:(M-1)
        CuDensityMat.append_elementary_product!(
            diss_sandwich, [elem_aa_dag], Int32[m, m], Int32[0, 1];
            coefficient = ComplexF64(1.0),
        )
    end

    # Dissipation anticommutator
    diss_number = CuDensityMat.create_operator_term(ws, dims)
    for m = 0:(M-1)
        CuDensityMat.append_elementary_product!(
            diss_number, [elem_n], Int32[m], Int32[0];
            coefficient = ComplexF64(1.0),
        )
    end

    # Assemble Liouvillian
    liouvillian = CuDensityMat.create_operator(ws, dims)
    for term in [kerr_term, coupling_term]
        CuDensityMat.append_term!(
            liouvillian, term; duality = 0, coefficient = ComplexF64(0, -1),
        )
        CuDensityMat.append_term!(
            liouvillian, term; duality = 1, coefficient = ComplexF64(0, +1),
        )
    end
    CuDensityMat.append_term!(
        liouvillian, diss_sandwich; duality = 0, coefficient = ComplexF64(gamma),
    )
    CuDensityMat.append_term!(
        liouvillian, diss_number; duality = 0, coefficient = ComplexF64(-gamma / 2),
    )
    CuDensityMat.append_term!(
        liouvillian, diss_number; duality = 1, coefficient = ComplexF64(-gamma / 2),
    )

    # States
    D = d^M
    rho = DenseMixedState{T}(ws, Tuple(dims); batch_size = 1)
    CuDensityMat.allocate_storage!(rho)
    rho_init = zeros(T, D * D)
    rho_init[2+D*(2-1)] = 1.0  # |1,0,...,0><1,0,...,0|
    copyto!(rho.storage, CUDA.CuVector{T}(rho_init))

    # RK4 scratch states
    k1 = DenseMixedState{T}(ws, Tuple(dims); batch_size = 1)
    k2 = DenseMixedState{T}(ws, Tuple(dims); batch_size = 1)
    k3 = DenseMixedState{T}(ws, Tuple(dims); batch_size = 1)
    k4 = DenseMixedState{T}(ws, Tuple(dims); batch_size = 1)
    rho_tmp = DenseMixedState{T}(ws, Tuple(dims); batch_size = 1)
    rho_dot = DenseMixedState{T}(ws, Tuple(dims); batch_size = 1)
    for s in [k1, k2, k3, k4, rho_tmp, rho_dot]
        CuDensityMat.allocate_storage!(s)
    end

    # Prepare action
    CuDensityMat.prepare_operator_action!(ws, liouvillian, rho, rho_dot)

    return ws, liouvillian, rho, rho_dot, k1, k2, k3, k4, rho_tmp
end

# =============================================================================
# Benchmark: single L[rho] action
# =============================================================================

function bench_single_action(
    M::Int, d::Int;
    n_warmup = M >= 9 ? 1 : 3,
    n_trials = M >= 9 ? 5 : (M >= 8 ? 10 : 50),
)
    ws, liouvillian, rho, rho_dot, _ = build_system(M, d)

    for _ = 1:n_warmup
        CuDensityMat.initialize_zero!(rho_dot)
        CuDensityMat.compute_operator_action!(
            ws, liouvillian, rho, rho_dot; time = 0.1, batch_size = 1,
        )
    end

    times = Float64[]
    for _ = 1:n_trials
        CuDensityMat.initialize_zero!(rho_dot)
        CUDA.synchronize()
        t0 = time_ns()
        CuDensityMat.compute_operator_action!(
            ws, liouvillian, rho, rho_dot; time = 0.1, batch_size = 1,
        )
        CUDA.synchronize()
        t1 = time_ns()
        push!(times, (t1 - t0) / 1e6)
    end

    close(ws)
    return median(times), minimum(times), maximum(times)
end

# =============================================================================
# Benchmark: full RK4 simulation
# =============================================================================

function bench_rk4_simulation(M::Int, d::Int; n_steps::Int = 100, dt::Float64 = 0.01)
    D = d^M
    ws, liouvillian, rho, _, k1, k2, k3, k4, rho_tmp = build_system(M, d)

    function eval_L!(k_out, rho_in, t)
        CuDensityMat.initialize_zero!(k_out)
        CuDensityMat.compute_operator_action!(
            ws, liouvillian, rho_in, k_out; time = t, batch_size = 1,
        )
    end

    copy_state!(dst, src) = copyto!(dst.storage, src.storage)

    # Warmup
    eval_L!(k1, rho, 0.0)
    CUDA.synchronize()

    # Reset state
    rho_init = zeros(ComplexF64, D * D)
    rho_init[2+D*(2-1)] = 1.0
    copyto!(rho.storage, CUDA.CuVector{ComplexF64}(rho_init))
    CUDA.synchronize()

    # Timed RK4 loop
    GC.gc()
    CUDA.synchronize()
    t_start = time_ns()

    t = 0.0
    for step = 1:n_steps
        eval_L!(k1, rho, t)

        copy_state!(rho_tmp, rho)
        CuDensityMat.inplace_accumulate!(rho_tmp, k1, ComplexF64(dt / 2))
        eval_L!(k2, rho_tmp, t + dt / 2)

        copy_state!(rho_tmp, rho)
        CuDensityMat.inplace_accumulate!(rho_tmp, k2, ComplexF64(dt / 2))
        eval_L!(k3, rho_tmp, t + dt / 2)

        copy_state!(rho_tmp, rho)
        CuDensityMat.inplace_accumulate!(rho_tmp, k3, ComplexF64(dt))
        eval_L!(k4, rho_tmp, t + dt)

        CuDensityMat.inplace_accumulate!(rho, k1, ComplexF64(dt / 6))
        CuDensityMat.inplace_accumulate!(rho, k2, ComplexF64(dt / 3))
        CuDensityMat.inplace_accumulate!(rho, k3, ComplexF64(dt / 3))
        CuDensityMat.inplace_accumulate!(rho, k4, ComplexF64(dt / 6))

        t += dt
    end

    CUDA.synchronize()
    t_end = time_ns()
    wall_time = (t_end - t_start) / 1e9

    # Compute trace
    rho_final = reshape(Array(rho.storage), D, D)
    trace = real(tr(rho_final))

    close(ws)
    return wall_time, trace
end

# =============================================================================
# Main
# =============================================================================

function main()
    d = 3
    M_values = [2, 4, 6]

    if CUDA.functional()
        free = CUDA.free_memory()
        for M_try in Int[]  # M=8,9 require long prepare_operator_action! (~15 min)
            D_try = d^M_try
            rho_bytes = D_try^2 * 16
            needed = rho_bytes * 8  # states + workspace
            if free > needed
                push!(M_values, M_try)
            else
                @printf(
                    "Skipping M=%d (D=%d): need ~%.1f GB, free=%.1f GB\n",
                    M_try, D_try, needed / 1e9, free / 1e9
                )
            end
        end
    end

    n_steps = 100
    dt = 0.01
    gpu_name = CUDA.functional() ? CUDA.name(CUDA.device()) : "none"
    free = CUDA.functional() ? CUDA.free_memory() : 0

    println("=" ^ 78)
    println("Benchmark: CuQuantum.jl (cuDensityMat GPU)")
    println("  System: M coupled cavities, d=$d Fock truncation")
    println("  GPU: $gpu_name ($(round(free/1e9, digits=1)) GB free)")
    println("  Time integration: RK4 fixed-step (dt=$dt)")
    println("  Simulation: $(n_steps) steps, t_final=$(n_steps*dt)")
    println("=" ^ 78)
    println()

    # --- Single action benchmark ---
    println("--- Single Liouvillian action L[rho] ---")
    for M in M_values
        D = d^M
        @printf("  M=%d (D=%d): ", M, D)
        try
            med, mn, mx = bench_single_action(M, d)
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
            wall_time, trace = bench_rk4_simulation(M, d; n_steps = n_steps, dt = dt)
            @printf("%.4f s  (Tr(rho)=%.6f)\n", wall_time, trace)
            push!(results, (M = M, D = D, wall_time_s = wall_time, trace = trace))
        catch e
            println("FAILED: $e")
            push!(results, (M = M, D = D, wall_time_s = NaN, trace = 0.0))
        end
    end
    println()

    # --- Save CSV ---
    csv_file = joinpath(@__DIR__, "results_cuquantum_jl.csv")
    open(csv_file, "w") do io
        println(io, "framework,backend,M,D,rho_elements,wall_time_s,n_steps,dt")
        for r in results
            println(
                io,
                "CuQuantum.jl,GPU-$(gpu_name),$(r.M),$(r.D),$(r.D^2),$(r.wall_time_s),$(n_steps),$(dt)",
            )
        end
    end
    println("Results saved to $csv_file")
end

main()
