# Compare a single L[ρ] evaluation at t=0 between CPU and GPU
using CUDA, CuQuantum, CuQuantum.CuDensityMat, LinearAlgebra, SparseArrays

const T = ComplexF64
const M = 2
const d = 3
const D = d^M

χ = 2π * 0.2
κ₀ = 2π * 0.1
γ = 0.01

# ---- CPU ----
a_s = spzeros(T, d, d)
for n in 1:(d-1); a_s[n, n+1] = sqrt(n); end
n_s = a_s' * a_s
kerr_s = n_s * (n_s - sparse(T(1)*I, d, d))
eye_d = sparse(T(1)*I, d, d)
eye_D = sparse(T(1)*I, D, D)

function embed(op, mode)
    mats = [i == mode ? op : eye_d for i in 1:M]
    r = mats[1]
    for i in 2:M; r = kron(r, mats[i]); end
    return r
end

H = spzeros(T, D, D)
a_ops = [embed(a_s, m) for m in 1:M]
n_ops = [embed(n_s, m) for m in 1:M]

for m in 1:M
    global H += χ * embed(kerr_s, m)
end
for n in 1:M, m in 1:M
    n == m && continue
    global H += κ₀ * (a_ops[n]' * a_ops[m])
end

L = -im * (kron(eye_D, H) - kron(transpose(H), eye_D))
for m in 1:M
    am = a_ops[m]; nm = n_ops[m]
    global L += γ * (kron(conj(am), am) - 0.5*kron(eye_D, nm) - 0.5*kron(transpose(nm), eye_D))
end

rho_vec = zeros(T, D*D)
rho_vec[2 + D*(2-1)] = 1.0
rhodot_cpu = L * rho_vec

# ---- GPU ----
ws = WorkStream()
dims = [d, d]

a_matrix = zeros(Float64, d, d)
for n in 1:(d-1); a_matrix[n, n+1] = sqrt(n); end
a_dag = transpose(a_matrix)

n_mat = diagm(0 => T.(0:d-1))
kerr_mat = diagm(0 => T.([n*(n-1) for n in 0:d-1]))

elem_n = CuDensityMat.create_elementary_operator(ws, [d], CUDA.CuVector{T}(vec(n_mat)))
elem_kerr = CuDensityMat.create_elementary_operator(ws, [d], CUDA.CuVector{T}(vec(kerr_mat)))

aa_dag_f = zeros(T, d, d, d, d)
for j1 in 1:d, j0 in 1:d, i1 in 1:d, i0 in 1:d
    aa_dag_f[i0, i1, j0, j1] = a_matrix[i0, j0] * a_dag[i1, j1]
end
elem_aa_dag = CuDensityMat.create_elementary_operator(ws, [d, d], CUDA.CuVector{T}(vec(aa_dag_f)))

coupling_elems = Dict{Tuple{Int,Int}, Any}()
for n in 0:M-1, m in 0:M-1
    n == m && continue
    c = zeros(T, d, d, d, d)
    for j1 in 1:d, j0 in 1:d, i1 in 1:d, i0 in 1:d
        c[i0, i1, j0, j1] = a_dag[i0, j0] * a_matrix[i1, j1]
    end
    coupling_elems[(n, m)] = CuDensityMat.create_elementary_operator(ws, [d, d], CUDA.CuVector{T}(vec(c)))
end

# Static Liouvillian (no callbacks, t=0 means detuning=0, coupling=κ₀)
kerr_term = CuDensityMat.create_operator_term(ws, dims)
for m in 0:M-1
    CuDensityMat.append_elementary_product!(kerr_term, [elem_kerr],
        Int32[m], Int32[0]; coefficient=ComplexF64(χ))
end

coupling_term = CuDensityMat.create_operator_term(ws, dims)
for n in 0:M-1, m in 0:M-1
    n == m && continue
    CuDensityMat.append_elementary_product!(coupling_term, [coupling_elems[(n, m)]],
        Int32[n, m], Int32[0, 0]; coefficient=ComplexF64(κ₀))
end

diss_sandwich = CuDensityMat.create_operator_term(ws, dims)
for m in 0:M-1
    CuDensityMat.append_elementary_product!(diss_sandwich, [elem_aa_dag],
        Int32[m, m], Int32[0, 1]; coefficient=ComplexF64(1.0))
end

diss_number = CuDensityMat.create_operator_term(ws, dims)
for m in 0:M-1
    CuDensityMat.append_elementary_product!(diss_number, [elem_n],
        Int32[m], Int32[0]; coefficient=ComplexF64(1.0))
end

liouvillian = CuDensityMat.create_operator(ws, dims)
for term in [kerr_term, coupling_term]
    CuDensityMat.append_term!(liouvillian, term; duality=0, coefficient=ComplexF64(0, -1))
    CuDensityMat.append_term!(liouvillian, term; duality=1, coefficient=ComplexF64(0, +1))
end
CuDensityMat.append_term!(liouvillian, diss_sandwich; duality=0, coefficient=ComplexF64(γ))
CuDensityMat.append_term!(liouvillian, diss_number; duality=0, coefficient=ComplexF64(-γ/2))
CuDensityMat.append_term!(liouvillian, diss_number; duality=1, coefficient=ComplexF64(-γ/2))

rho = DenseMixedState{T}(ws, Tuple(dims); batch_size=1)
rho_dot = DenseMixedState{T}(ws, Tuple(dims); batch_size=1)
CuDensityMat.allocate_storage!(rho)
CuDensityMat.allocate_storage!(rho_dot)
rho_init_gpu = CUDA.CuVector{T}(rho_vec)
copyto!(rho.storage, rho_init_gpu)

CuDensityMat.prepare_operator_action!(ws, liouvillian, rho, rho_dot)
CuDensityMat.initialize_zero!(rho_dot)
CuDensityMat.compute_operator_action!(ws, liouvillian, rho, rho_dot;
    time=0.0, batch_size=1)

rhodot_gpu = Array(rho_dot.storage)

# ---- Compare ----
diff = rhodot_gpu - rhodot_cpu
println("\n=== Single-step L[ρ] comparison at t=0 ===")
println("CPU rhodot norm:  ", norm(rhodot_cpu))
println("GPU rhodot norm:  ", norm(rhodot_gpu))
println("Diff norm:        ", norm(diff))
println("Max |diff|:       ", maximum(abs.(diff)))
println("Relative error:   ", norm(diff) / norm(rhodot_cpu))
println()

# Show element-by-element for the 9x9 matrix
rhodot_cpu_mat = reshape(rhodot_cpu, D, D)
rhodot_gpu_mat = reshape(rhodot_gpu, D, D)
diff_mat = reshape(diff, D, D)

println("Diagonal of rhodot (CPU vs GPU):")
for i in 1:D
    c = round(real(rhodot_cpu_mat[i,i]), sigdigits=8)
    g = round(real(rhodot_gpu_mat[i,i]), sigdigits=8)
    d_val = abs(rhodot_cpu_mat[i,i] - rhodot_gpu_mat[i,i])
    println("  [$i,$i] CPU=$c  GPU=$g  |diff|=$(round(d_val, sigdigits=3))")
end

println("\nLargest off-diagonal differences:")
diffs_offdiag = [(abs(diff_mat[i,j]), i, j) for i in 1:D, j in 1:D if i != j]
sort!(diffs_offdiag, rev=true)
for (val, i, j) in diffs_offdiag[1:5]
    println("  [$i,$j] |diff|=$(round(val, sigdigits=6))  CPU=$(round(rhodot_cpu_mat[i,j], sigdigits=6))  GPU=$(round(rhodot_gpu_mat[i,j], sigdigits=6))")
end

close(ws)
