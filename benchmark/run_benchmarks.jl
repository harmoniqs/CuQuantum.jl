# =============================================================================
# Benchmark: Liouvillian action L[ρ] — cuDensityMat vs QuantumToolbox.jl
# =============================================================================
#
# Times a single L[ρ] evaluation for the dual-rail cavity system across:
#   - cuDensityMat (GPU, direct library binding)
#   - QT.jl GPU cuSPARSE (sparse Liouvillian on-device via QuantumObject)
#   - QT.jl CPU sparse
#   - CPU dense (small sizes only)
#
# System: M coupled cavities, Fock truncation d=3
#   H = Σ_m χ n_m(n_m-1) + Σ_{n≠m} κ a_n†a_m
#   L[ρ] = -i[H,ρ] + γ Σ_m (a_m ρ a_m† - ½{n_m, ρ})
#
# Usage:
#   julia --project=. benchmark/run_benchmarks.jl
#
# Outputs (written to benchmark/results/):
#   benchmark_results.csv — raw per-backend median/min/max
#   bench.json            — customSmallerIsBetter format for github-action-benchmark

using CUDA
using CuQuantum
using CuQuantum.CuDensityMat
using LinearAlgebra
using QuantumToolbox
using Statistics

# =============================================================================
# 1. cuDensityMat benchmark: build system and time L[ρ] evaluation
# =============================================================================

"""
Build the dual-rail Lindblad system using CuDensityMat for M cavities with
Fock truncation d. Returns (ws, liouvillian, rho, rho_dot, callback_refs)
ready for timing.
"""
function build_cudensitymat_system(M::Int, d::Int)
    T = ComplexF64
    ws = WorkStream()
    dims = fill(d, M)

    # --- Operator matrices (single cavity, d×d) ---
    a_matrix = zeros(Float64, d, d)
    for n = 1:(d-1)
        a_matrix[n, n+1] = sqrt(n)  # a|n⟩ = √n|n-1⟩
    end
    a_dag = transpose(a_matrix)

    n_mat =
        T[a_dag[i, k]*a_matrix[k, j] for i = 1:d, j = 1:d, k = 1:d] |> x -> dropdims(sum(x, dims = 3), dims = 3)
    # Simpler: n = diag(0, 1, ..., d-1)
    n_mat = diagm(0 => T.(0:(d-1)))
    kerr_mat = diagm(0 => T.([n*(n-1) for n = 0:(d-1)]))

    # Upload to GPU
    n_gpu = CUDA.CuVector{T}(vec(n_mat))
    kerr_gpu = CUDA.CuVector{T}(vec(kerr_mat))

    # --- Elementary operators ---
    elem_n = CuDensityMat.create_elementary_operator(ws, [d], n_gpu)
    elem_kerr = CuDensityMat.create_elementary_operator(ws, [d], kerr_gpu)

    # Fused aa_dag for dissipation sandwich (d×d×d×d)
    aa_dag_fused = zeros(T, d, d, d, d)
    for j1 = 1:d, j0 = 1:d, i1 = 1:d, i0 = 1:d
        aa_dag_fused[i0, i1, j0, j1] = a_matrix[i0, j0] * a_dag[i1, j1]
    end
    aa_dag_gpu = CUDA.CuVector{T}(vec(aa_dag_fused))
    elem_aa_dag = CuDensityMat.create_elementary_operator(ws, [d, d], aa_dag_gpu)

    # Fused coupling operators a_n†a_m for all pairs
    coupling_elems = Dict{Tuple{Int,Int},Any}()
    for n = 0:(M-1), m = 0:(M-1)
        n == m && continue
        # a_n† a_m fused into a 2-mode operator
        c = zeros(T, d, d, d, d)
        for j1 = 1:d, j0 = 1:d, i1 = 1:d, i0 = 1:d
            c[i0, i1, j0, j1] = a_dag[i0, j0] * a_matrix[i1, j1]
        end
        c_gpu = CUDA.CuVector{T}(vec(c))
        coupling_elems[(n, m)] = CuDensityMat.create_elementary_operator(ws, [d, d], c_gpu)
    end

    # --- Parameters ---
    χ = 2π * 0.2
    κ₀ = 2π * 0.1
    γ = 0.01

    # --- Operator terms ---

    # Kerr: Σ_m χ n_m(n_m-1)
    kerr_term = CuDensityMat.create_operator_term(ws, dims)
    for m = 0:(M-1)
        CuDensityMat.append_elementary_product!(
            kerr_term,
            [elem_kerr],
            Int32[m],
            Int32[0];
            coefficient = ComplexF64(χ),
        )
    end

    # Coupling: Σ_{n≠m} κ a_n†a_m  (static κ for benchmark simplicity)
    coupling_term = CuDensityMat.create_operator_term(ws, dims)
    for n = 0:(M-1), m = 0:(M-1)
        n == m && continue
        CuDensityMat.append_elementary_product!(
            coupling_term,
            [coupling_elems[(n, m)]],
            Int32[n, m],
            Int32[0, 0];
            coefficient = ComplexF64(κ₀),
        )
    end

    # Dissipation sandwich: Σ_m a_m ρ a_m†
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

    # Dissipation anticommutator: Σ_m n_m
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

    # --- Assemble Liouvillian ---
    liouvillian = CuDensityMat.create_operator(ws, dims)

    # Hamiltonian: -i[H, ρ]
    for term in [kerr_term, coupling_term]
        CuDensityMat.append_term!(
            liouvillian,
            term;
            duality = 0,
            coefficient = ComplexF64(0, -1),
        )
        CuDensityMat.append_term!(
            liouvillian,
            term;
            duality = 1,
            coefficient = ComplexF64(0, +1),
        )
    end

    # Lindblad dissipation
    CuDensityMat.append_term!(
        liouvillian,
        diss_sandwich;
        duality = 0,
        coefficient = ComplexF64(γ),
    )
    CuDensityMat.append_term!(
        liouvillian,
        diss_number;
        duality = 0,
        coefficient = ComplexF64(-γ/2),
    )
    CuDensityMat.append_term!(
        liouvillian,
        diss_number;
        duality = 1,
        coefficient = ComplexF64(-γ/2),
    )

    # --- States ---
    rho = DenseMixedState{T}(ws, Tuple(dims); batch_size = 1)
    rho_dot = DenseMixedState{T}(ws, Tuple(dims); batch_size = 1)
    CuDensityMat.allocate_storage!(rho)
    CuDensityMat.allocate_storage!(rho_dot)

    # Initialize to |1,0,...,0⟩⟨1,0,...,0|
    d_total = d^M
    rho_init = zeros(T, d_total * d_total)
    idx = 2  # |1,0,...,0⟩ = index 2 in column-major flattening
    rho_init[idx+d_total*(idx-1)] = 1.0
    copyto!(rho.storage, CUDA.CuVector{T}(rho_init))

    # Prepare action
    CuDensityMat.prepare_operator_action!(ws, liouvillian, rho, rho_dot)

    return ws, liouvillian, rho, rho_dot
end

"""
Time a single L[ρ] evaluation using CuDensityMat.
"""
function benchmark_cudensitymat(
    M::Int,
    d::Int;
    n_warmup = M >= 9 ? 1 : 3,
    n_trials = M >= 9 ? 5 : (M >= 8 ? 10 : 20),
)
    ws, liouvillian, rho, rho_dot = build_cudensitymat_system(M, d)

    # Warmup
    for _ = 1:n_warmup
        CuDensityMat.initialize_zero!(rho_dot)
        CuDensityMat.compute_operator_action!(
            ws,
            liouvillian,
            rho,
            rho_dot;
            time = 0.1,
            batch_size = 1,
        )
    end

    # Timed runs
    times = Float64[]
    for _ = 1:n_trials
        CuDensityMat.initialize_zero!(rho_dot)
        CUDA.synchronize()
        t0 = time_ns()
        CuDensityMat.compute_operator_action!(
            ws,
            liouvillian,
            rho,
            rho_dot;
            time = 0.1,
            batch_size = 1,
        )
        CUDA.synchronize()
        t1 = time_ns()
        push!(times, (t1 - t0) / 1e6)  # ms
    end

    close(ws)
    return median(times), minimum(times), maximum(times)
end

# =============================================================================
# 2. CPU dense baseline: explicit superoperator matrix-vector product
# =============================================================================

"""
Build the full Liouvillian superoperator matrix on CPU and time a single
matrix-vector product L * vec(ρ).
"""
function build_cpu_system(M::Int, d::Int)
    T = ComplexF64
    D = d^M  # Hilbert space dimension

    # Single-cavity operators
    a_single = zeros(T, d, d)
    for n = 1:(d-1)
        a_single[n, n+1] = sqrt(n)
    end

    # Embed single-cavity operator into full Hilbert space
    function embed(op_single, mode, M, d)
        D = d^M
        result = zeros(T, D, D)
        # Build using Kronecker product: I ⊗ ... ⊗ op ⊗ ... ⊗ I
        mats = [i == mode ? op_single : Matrix{T}(I, d, d) for i = 1:M]
        kron_result = mats[1]
        for i = 2:M
            kron_result = kron(kron_result, mats[i])
        end
        return kron_result
    end

    # Build Hamiltonian
    χ = 2π * 0.2
    κ₀ = 2π * 0.1
    γ = 0.01

    n_single = a_single' * a_single
    kerr_single = n_single * (n_single - I(d))

    H = zeros(T, D, D)
    a_ops = [embed(a_single, m, M, d) for m = 1:M]
    n_ops = [embed(n_single, m, M, d) for m = 1:M]

    # Kerr
    for m = 1:M
        H += χ * embed(kerr_single, m, M, d)
    end

    # Coupling
    for n = 1:M, m = 1:M
        n == m && continue
        H += κ₀ * (a_ops[n]' * a_ops[m])
    end

    # Build Liouvillian superoperator L (D²×D² matrix)
    # L[ρ] = -i(H⊗I - I⊗Hᵀ)vec(ρ) + γ Σ_m (a_m⊗conj(a_m) - ½ n_m⊗I - ½ I⊗n_mᵀ) vec(ρ)
    eye = Matrix{T}(I, D, D)

    L_super = -im * (kron(eye, H) - kron(transpose(H), eye))
    for m = 1:M
        am = a_ops[m]
        nm = n_ops[m]
        L_super +=
            γ * (kron(conj(am), am) - 0.5*kron(eye, nm) - 0.5*kron(transpose(nm), eye))
    end

    # Initial state vec(ρ)
    rho_vec = zeros(T, D*D)
    rho_vec[2+D*(2-1)] = 1.0  # |1,0,...⟩⟨1,0,...|

    return L_super, rho_vec
end

function benchmark_cpu_dense(M::Int, d::Int; n_warmup = 3, n_trials = 20)
    D = d^M
    # Superoperator is D²×D² — memory = D⁴ × 16 bytes
    # D=81 (M=4,d=3): 81⁴×16 = 688 MB — OK
    # D=729 (M=6,d=3): 729⁴×16 = 4.5 TB — impossible
    if D > 100
        return NaN, NaN, NaN  # Skip — superoperator too large
    end

    L_super, rho_vec = build_cpu_system(M, d)
    rho_dot = similar(rho_vec)

    # Warmup
    for _ = 1:n_warmup
        mul!(rho_dot, L_super, rho_vec)
    end

    # Timed runs
    times = Float64[]
    for _ = 1:n_trials
        t0 = time_ns()
        mul!(rho_dot, L_super, rho_vec)
        t1 = time_ns()
        push!(times, (t1 - t0) / 1e6)
    end

    return median(times), minimum(times), maximum(times)
end

# =============================================================================
# 3. QuantumToolbox.jl path: sparse Liouvillian via QT.jl, CPU + GPU cuSPARSE
# =============================================================================
#
# Routed through QuantumToolbox.jl's `liouvillian` + `QuantumObject` so the
# tracked series includes QT.jl dispatch/allocation overhead — regressions in
# the user-facing QT.jl path show up here.

"""
Build the Lindblad system as QuantumToolbox.jl `QuantumObject`s. Returns the
Liouvillian super-operator and the vectorized initial density matrix.
"""
function build_qt_system(M::Int, d::Int)
    a = destroy(d)
    Id = qeye(d)

    a_ops = [tensor([i == m ? a : Id for i = 1:M]...) for m = 1:M]
    n_ops = [adjoint(ao) * ao for ao in a_ops]

    χ = 2π * 0.2
    κ₀ = 2π * 0.1
    γ = 0.01

    Id_full = tensor(fill(Id, M)...)
    H = sum(χ * nm * (nm - Id_full) for nm in n_ops)
    for n = 1:M, m = 1:M
        n == m && continue
        H += κ₀ * (adjoint(a_ops[n]) * a_ops[m])
    end

    c_ops = [sqrt(γ) * ao for ao in a_ops]
    L = liouvillian(H, c_ops)

    # Initial state: |1,0,...,0⟩⟨1,0,...,0|
    ψ = tensor(fock(d, 1), [fock(d, 0) for _ = 2:M]...)
    ρ_vec = mat2vec(ket2dm(ψ))

    return L, ρ_vec
end

function benchmark_cpu_sparse(M::Int, d::Int; n_warmup = 3, n_trials = 20)
    D = d^M
    # Sparse superoperator is D²×D² with O(M·d²·D²) non-zeros.
    # D=729 (M=6): ~50M nnz ≈ 1.2 GB — feasible. D=6561 (M=8): ~5B nnz — too large.
    if D > 1000
        return NaN, NaN, NaN
    end

    L, ρ_vec = build_qt_system(M, d)

    # Warmup (QObject `*` returns a new QObject — captures allocation overhead)
    for _ = 1:n_warmup
        L * ρ_vec
    end

    times = Float64[]
    for _ = 1:n_trials
        t0 = time_ns()
        L * ρ_vec
        t1 = time_ns()
        push!(times, (t1 - t0) / 1e6)
    end

    return median(times), minimum(times), maximum(times)
end

function benchmark_cusparse_gpu(M::Int, d::Int; n_warmup = 5, n_trials = 50)
    D = d^M
    if D > 1000
        return NaN, NaN, NaN
    end

    L_cpu, ρ_vec_cpu = build_qt_system(M, d)
    L_gpu = cu(L_cpu)          # QObject wrapping CuSparseMatrixCSC
    ρ_vec_gpu = cu(ρ_vec_cpu)  # QObject wrapping CuVector

    # Warmup
    for _ = 1:n_warmup
        L_gpu * ρ_vec_gpu
    end
    CUDA.synchronize()

    times = Float64[]
    for _ = 1:n_trials
        CUDA.synchronize()
        t0 = time_ns()
        L_gpu * ρ_vec_gpu
        CUDA.synchronize()
        t1 = time_ns()
        push!(times, (t1 - t0) / 1e6)
    end

    return median(times), minimum(times), maximum(times)
end

# =============================================================================
# 4. Run all benchmarks
# =============================================================================

function main()
    d = 3  # Fock truncation

    # System sizes to benchmark
    M_values = [2, 4, 6]

    # Dynamically add larger sizes based on available GPU memory
    if CUDA.functional()
        free = CUDA.free_memory()
        println("GPU: $(CUDA.name(CUDA.device())), $(round(free / 1e9, digits=1)) GB free")
        for M_try in [8, 9]
            D_try = d^M_try
            rho_bytes = D_try^2 * 16
            needed = rho_bytes * 4  # rho + rho_dot + workspace
            if free > needed
                push!(M_values, M_try)
            else
                println(
                    "Skipping M=$M_try (D=$(D_try)): need ~$(round(needed / 1e9, digits=1)) GB, have $(round(free / 1e9, digits=1)) GB free",
                )
            end
        end
    end

    gpu_name = CUDA.functional() ? CUDA.name(CUDA.device()) : "none"
    cpu_cores = Sys.CPU_THREADS
    println("=" ^ 78)
    println("Benchmark: Liouvillian Action L[ρ] — Dual-Rail Cavities (d=$d)")
    println(
        "GPU: $gpu_name | CPU: $(cpu_cores) threads | $(round(Sys.total_memory()/1e9, digits=1)) GB RAM",
    )
    println("=" ^ 78)
    println()

    results = []

    for M in M_values
        D = d^M
        println(
            "--- M=$M cavities, Hilbert dim D=$D, ρ size $(D)×$(D) = $(D^2) elements ---",
        )

        # cuDensityMat (GPU)
        print("  cuDensityMat (GPU): ")
        gpu_med, gpu_min, gpu_max = try
            benchmark_cudensitymat(M, d)
        catch e
            println("FAILED: $e")
            (NaN, NaN, NaN)
        end
        if !isnan(gpu_med)
            println(
                "$(round(gpu_med, digits=3)) ms (min=$(round(gpu_min, digits=3)), max=$(round(gpu_max, digits=3)))",
            )
        end

        # CPU dense
        print("  CPU dense SpMV:     ")
        cpu_dense_med, cpu_dense_min, cpu_dense_max = try
            benchmark_cpu_dense(M, d)
        catch e
            println("FAILED: $e")
            (NaN, NaN, NaN)
        end
        if !isnan(cpu_dense_med)
            println(
                "$(round(cpu_dense_med, digits=3)) ms (min=$(round(cpu_dense_min, digits=3)), max=$(round(cpu_dense_max, digits=3)))",
            )
        elseif cpu_dense_med === NaN
            println("SKIPPED (D=$D too large for dense)")
        end

        # CPU sparse
        print("  CPU sparse SpMV:    ")
        cpu_sparse_med, cpu_sparse_min, cpu_sparse_max = try
            benchmark_cpu_sparse(M, d)
        catch e
            println("FAILED: $e")
            (NaN, NaN, NaN)
        end
        if !isnan(cpu_sparse_med)
            println(
                "$(round(cpu_sparse_med, digits=3)) ms (min=$(round(cpu_sparse_min, digits=3)), max=$(round(cpu_sparse_max, digits=3)))",
            )
        elseif cpu_sparse_med === NaN
            println("SKIPPED (D=$D too large)")
        end

        # GPU cuSPARSE SpMV
        print("  GPU cuSPARSE SpMV:  ")
        gpu_cusparse_med, gpu_cusparse_min, gpu_cusparse_max = try
            benchmark_cusparse_gpu(M, d)
        catch e
            println("FAILED: $e")
            (NaN, NaN, NaN)
        end
        if !isnan(gpu_cusparse_med)
            println(
                "$(round(gpu_cusparse_med, digits=3)) ms (min=$(round(gpu_cusparse_min, digits=3)), max=$(round(gpu_cusparse_max, digits=3)))",
            )
        elseif gpu_cusparse_med === NaN
            println("SKIPPED (D=$D too large)")
        end

        # Speedup
        if !isnan(gpu_med) && !isnan(cpu_sparse_med)
            println(
                "  Speedup (sparse/GPU cuDensityMat): $(round(cpu_sparse_med / gpu_med, digits=1))x",
            )
        end
        if !isnan(gpu_med) && !isnan(cpu_dense_med)
            println(
                "  Speedup (dense/GPU cuDensityMat):  $(round(cpu_dense_med / gpu_med, digits=1))x",
            )
        end
        if !isnan(gpu_cusparse_med) && !isnan(gpu_med)
            println(
                "  Speedup (cuSPARSE/cuDensityMat):   $(round(gpu_cusparse_med / gpu_med, digits=1))x",
            )
        end

        push!(
            results,
            (
                M = M,
                D = D,
                gpu_ms = gpu_med,
                cpu_dense_ms = cpu_dense_med,
                cpu_sparse_ms = cpu_sparse_med,
                gpu_cusparse_ms = gpu_cusparse_med,
            ),
        )
        println()
    end

    # Save CSV
    results_dir = joinpath(@__DIR__, "results")
    mkpath(results_dir)
    csv_file = joinpath(results_dir, "benchmark_results.csv")
    open(csv_file, "w") do io
        println(io, "M,D,rho_elements,gpu_ms,cpu_dense_ms,cpu_sparse_ms,gpu_cusparse_ms")
        for r in results
            println(
                io,
                "$(r.M),$(r.D),$(r.D^2),$(r.gpu_ms),$(r.cpu_dense_ms),$(r.cpu_sparse_ms),$(r.gpu_cusparse_ms)",
            )
        end
    end
    println("Results saved to $csv_file")

    # Emit customSmallerIsBetter JSON for github-action-benchmark
    json_file = joinpath(results_dir, "bench.json")
    open(json_file, "w") do io
        print(io, "[")
        first = true
        for r in results
            for (label, value) in (
                ("cuDensityMat GPU L[ρ] M=$(r.M) D=$(r.D)", r.gpu_ms),
                ("QT.jl GPU cuSPARSE L[ρ] M=$(r.M) D=$(r.D)", r.gpu_cusparse_ms),
                ("CPU dense SpMV L[ρ] M=$(r.M) D=$(r.D)", r.cpu_dense_ms),
                ("QT.jl CPU sparse L[ρ] M=$(r.M) D=$(r.D)", r.cpu_sparse_ms),
            )
                isnan(value) && continue
                first || print(io, ",")
                first = false
                print(
                    io,
                    "{\"name\":\"",
                    label,
                    "\",\"unit\":\"ms\",\"value\":",
                    value,
                    "}",
                )
            end
        end
        print(io, "]")
    end
    println("JSON results saved to $json_file")
end

main()
