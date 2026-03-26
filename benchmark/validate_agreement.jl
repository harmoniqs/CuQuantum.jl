# =============================================================================
# Validation: cuDensityMat (GPU) vs CPU sparse Liouvillian
# =============================================================================
#
# Runs both methods on the same M=2 dual-rail system for 50 RK4 steps,
# then compares the density matrices element-by-element.
#
# Usage: julia --project=. benchmark/validate_agreement.jl

using CUDA
using CuQuantum
using CuQuantum.CuDensityMat
using LinearAlgebra
using SparseArrays

const T = ComplexF64
const M = 2
const d = 3
const D = d^M  # 9

# Physical parameters (same as integration test)
const χ = 2π * 0.2
const κ₀ = 2π * 0.1
const γ = 0.01
const Δ = [2π * 1.0, 2π * 0.8]
const ω_δ = [2π * 0.5, 2π * 0.4]
const ε = 0.3
const ω_κ = 2π * 0.2

const dt = 0.02
const n_steps = 50  # 1.0 μs total

# =============================================================================
# CPU sparse reference
# =============================================================================

function build_cpu_liouvillian(t::Float64)
    # Single-cavity operators
    a = spzeros(T, d, d)
    for n = 1:(d-1)
        a[n, n+1] = sqrt(n)
    end
    n_op = a' * a
    kerr_op = n_op * (n_op - sparse(T(1)*I, d, d))
    eye_d = sparse(T(1)*I, d, d)
    eye_D = sparse(T(1)*I, D, D)

    function embed(op, mode)
        mats = [i == mode ? op : eye_d for i = 1:M]
        result = mats[1]
        for i = 2:M
            result = kron(result, mats[i])
        end
        return result
    end

    # Hamiltonian at time t
    H = spzeros(T, D, D)
    a_ops = [embed(a, m) for m = 1:M]
    n_ops = [embed(n_op, m) for m = 1:M]

    # Kerr
    for m = 1:M
        H += χ * embed(kerr_op, m)
    end

    # Time-dependent detuning
    for m = 1:M
        δ_m = Δ[m] * sin(ω_δ[m] * t)
        H += δ_m * n_ops[m]
    end

    # Time-dependent coupling
    κ_t = κ₀ * (1.0 + ε * cos(ω_κ * t))
    for n = 1:M, m = 1:M
        n == m && continue
        H += κ_t * (a_ops[n]' * a_ops[m])
    end

    # Liouvillian superoperator
    L = -im * (kron(eye_D, H) - kron(transpose(H), eye_D))
    for m = 1:M
        am = a_ops[m]
        nm = n_ops[m]
        L += γ * (kron(conj(am), am) - 0.5*kron(eye_D, nm) - 0.5*kron(transpose(nm), eye_D))
    end
    return L
end

function run_cpu_rk4()
    rho = zeros(T, D*D)
    rho[2+D*(2-1)] = 1.0  # |1,0⟩⟨1,0|

    for step = 1:n_steps
        t = (step - 1) * dt
        L1 = build_cpu_liouvillian(t)
        L2 = build_cpu_liouvillian(t + dt/2)
        L4 = build_cpu_liouvillian(t + dt)

        k1 = L1 * rho
        k2 = L2 * (rho + (dt/2) * k1)
        k3 = L2 * (rho + (dt/2) * k2)
        k4 = L4 * (rho + dt * k3)
        rho .+= (dt/6) .* (k1 .+ 2k2 .+ 2k3 .+ k4)
    end
    return reshape(rho, D, D)
end

# =============================================================================
# cuDensityMat GPU
# =============================================================================

function run_gpu_rk4()
    ws = WorkStream()
    dims = [d, d]

    # Operator matrices
    a_matrix = zeros(Float64, d, d)
    for n = 1:(d-1)
        a_matrix[n, n+1] = sqrt(n)
    end
    a_dag = transpose(a_matrix)

    n_mat = diagm(0 => T.(0:(d-1)))
    kerr_mat = diagm(0 => T.([n*(n-1) for n = 0:(d-1)]))

    n_gpu = CUDA.CuVector{T}(vec(n_mat))
    kerr_gpu = CUDA.CuVector{T}(vec(kerr_mat))

    elem_n = CuDensityMat.create_elementary_operator(ws, [d], n_gpu)
    elem_kerr = CuDensityMat.create_elementary_operator(ws, [d], kerr_gpu)

    # Fused aa_dag
    aa_dag_fused = zeros(T, d, d, d, d)
    for j1 = 1:d, j0 = 1:d, i1 = 1:d, i0 = 1:d
        aa_dag_fused[i0, i1, j0, j1] = a_matrix[i0, j0] * a_dag[i1, j1]
    end
    elem_aa_dag = CuDensityMat.create_elementary_operator(
        ws,
        [d, d],
        CUDA.CuVector{T}(vec(aa_dag_fused)),
    )

    # Fused coupling operators
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

    # Kerr term
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

    # Detuning callbacks
    function det1_cb_fn(time, params, storage)
        for b in eachindex(storage)
            storage[b] = ComplexF64(Δ[1] * sin(ω_δ[1] * time))
        end
    end
    function det2_cb_fn(time, params, storage)
        for b in eachindex(storage)
            storage[b] = ComplexF64(Δ[2] * sin(ω_δ[2] * time))
        end
    end
    det1_cb, det1_gcb, det1_refs = CuDensityMat.wrap_scalar_callback(det1_cb_fn)
    det2_cb, det2_gcb, det2_refs = CuDensityMat.wrap_scalar_callback(det2_cb_fn)

    det_term_1 = CuDensityMat.create_operator_term(ws, dims)
    CuDensityMat.append_elementary_product!(
        det_term_1,
        [elem_n],
        Int32[0],
        Int32[0];
        coefficient = ComplexF64(1.0),
        coefficient_callback = det1_cb,
        coefficient_gradient_callback = det1_gcb,
    )

    det_term_2 = CuDensityMat.create_operator_term(ws, dims)
    CuDensityMat.append_elementary_product!(
        det_term_2,
        [elem_n],
        Int32[1],
        Int32[0];
        coefficient = ComplexF64(1.0),
        coefficient_callback = det2_cb,
        coefficient_gradient_callback = det2_gcb,
    )

    # Coupling callback
    function coup_cb_fn(time, params, storage)
        κ_t = κ₀ * (1.0 + ε * cos(ω_κ * time))
        for b in eachindex(storage)
            storage[b] = ComplexF64(κ_t)
        end
    end
    coup_cb, coup_gcb, coup_refs = CuDensityMat.wrap_scalar_callback(coup_cb_fn)

    coupling_term = CuDensityMat.create_operator_term(ws, dims)
    for n = 0:(M-1), m = 0:(M-1)
        n == m && continue
        CuDensityMat.append_elementary_product!(
            coupling_term,
            [coupling_elems[(n, m)]],
            Int32[n, m],
            Int32[0, 0];
            coefficient = ComplexF64(1.0),
            coefficient_callback = coup_cb,
            coefficient_gradient_callback = coup_gcb,
        )
    end

    # Dissipation
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

    # Assemble Liouvillian
    liouvillian = CuDensityMat.create_operator(ws, dims)
    for term in [kerr_term, det_term_1, det_term_2, coupling_term]
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

    # States
    rho = DenseMixedState{T}(ws, Tuple(dims); batch_size = 1)
    CuDensityMat.allocate_storage!(rho)
    rho_init = zeros(T, D*D)
    rho_init[2+D*(2-1)] = 1.0
    copyto!(rho.storage, CUDA.CuVector{T}(rho_init))

    k1 = DenseMixedState{T}(ws, Tuple(dims); batch_size = 1)
    k2 = DenseMixedState{T}(ws, Tuple(dims); batch_size = 1)
    k3 = DenseMixedState{T}(ws, Tuple(dims); batch_size = 1)
    k4 = DenseMixedState{T}(ws, Tuple(dims); batch_size = 1)
    rho_tmp = DenseMixedState{T}(ws, Tuple(dims); batch_size = 1)
    for s in [k1, k2, k3, k4, rho_tmp]
        CuDensityMat.allocate_storage!(s)
    end

    CuDensityMat.prepare_operator_action!(ws, liouvillian, rho, k1)

    function eval_L!(k_out, rho_in, t)
        CuDensityMat.initialize_zero!(k_out)
        CuDensityMat.compute_operator_action!(
            ws,
            liouvillian,
            rho_in,
            k_out;
            time = t,
            batch_size = 1,
        )
    end

    function copy_state!(dst, src)
        copyto!(dst.storage, src.storage)
    end

    # RK4 loop
    for step = 1:n_steps
        t = (step - 1) * dt

        eval_L!(k1, rho, t)

        copy_state!(rho_tmp, rho)
        CuDensityMat.inplace_accumulate!(rho_tmp, k1, ComplexF64(dt/2))
        eval_L!(k2, rho_tmp, t + dt/2)

        copy_state!(rho_tmp, rho)
        CuDensityMat.inplace_accumulate!(rho_tmp, k2, ComplexF64(dt/2))
        eval_L!(k3, rho_tmp, t + dt/2)

        copy_state!(rho_tmp, rho)
        CuDensityMat.inplace_accumulate!(rho_tmp, k3, ComplexF64(dt))
        eval_L!(k4, rho_tmp, t + dt)

        CuDensityMat.inplace_accumulate!(rho, k1, ComplexF64(dt/6))
        CuDensityMat.inplace_accumulate!(rho, k2, ComplexF64(dt/3))
        CuDensityMat.inplace_accumulate!(rho, k3, ComplexF64(dt/3))
        CuDensityMat.inplace_accumulate!(rho, k4, ComplexF64(dt/6))
    end

    rho_final = reshape(Array(rho.storage), D, D)

    CuDensityMat.unregister_callback!(det1_refs)
    CuDensityMat.unregister_callback!(det2_refs)
    CuDensityMat.unregister_callback!(coup_refs)
    close(ws)

    return rho_final
end

# =============================================================================
# Compare
# =============================================================================

function main()
    println("Running CPU sparse reference (M=$M, d=$d, $n_steps RK4 steps)...")
    rho_cpu = run_cpu_rk4()

    println("Running cuDensityMat GPU (M=$M, d=$d, $n_steps RK4 steps)...")
    rho_gpu = run_gpu_rk4()

    # Compare
    diff = rho_gpu - rho_cpu
    max_abs_diff = maximum(abs.(diff))
    fro_norm_diff = norm(diff)
    fro_norm_cpu = norm(rho_cpu)
    rel_err = fro_norm_diff / fro_norm_cpu

    println()
    println("=" ^ 60)
    println("Validation: cuDensityMat vs CPU sparse (M=$M, d=$d)")
    println("=" ^ 60)
    println("  Max |element-wise diff|: $max_abs_diff")
    println("  Frobenius norm of diff:  $fro_norm_diff")
    println("  Relative error:          $rel_err")
    println()

    # Diagonal populations comparison
    println("  Diagonal populations:")
    println("  State      | CPU           | GPU           | Diff")
    println("  " * "-"^56)
    labels =
        ["|0,0⟩", "|1,0⟩", "|2,0⟩", "|0,1⟩", "|1,1⟩", "|2,1⟩", "|0,2⟩", "|1,2⟩", "|2,2⟩"]
    for i = 1:D
        cpu_p = real(rho_cpu[i, i])
        gpu_p = real(rho_gpu[i, i])
        label = i <= length(labels) ? labels[i] : "[$i]"
        d_val = abs(cpu_p - gpu_p)
        if abs(cpu_p) > 1e-10 || abs(gpu_p) > 1e-10
            println(
                "  $(rpad(label, 10)) | $(lpad(round(cpu_p, digits=8), 13)) | $(lpad(round(gpu_p, digits=8), 13)) | $(round(d_val, sigdigits=3))",
            )
        end
    end

    # Trace comparison
    tr_cpu = real(tr(rho_cpu))
    tr_gpu = real(tr(rho_gpu))
    println()
    println("  Tr(ρ_cpu) = $tr_cpu")
    println("  Tr(ρ_gpu) = $tr_gpu")
    println()

    if max_abs_diff < 1e-6
        println("  ✓ PASS: Agreement within 1e-6")
    elseif max_abs_diff < 1e-3
        println("  ~ CLOSE: Agreement within 1e-3 (acceptable for RK4 at dt=$dt)")
    else
        println("  ✗ FAIL: Disagreement exceeds 1e-3")
    end
end

main()
