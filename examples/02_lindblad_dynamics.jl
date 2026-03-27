# Lindblad Dynamics: Decaying Qubit
#
# Simulate amplitude damping of a qubit initially in |1⟩ via the Lindblad
# master equation:
#
#   ρ̇ = -i[H, ρ] + γ(a ρ a† - ½{a†a, ρ})
#
# with H = (ω/2)σ_z (free precession) and γ = 0.1 (decay rate).
#
# The qubit starts in |1⟩⟨1| and decays toward |0⟩⟨0|. The excited-state
# population follows P₁(t) = e^{-γt} analytically.

using Printf
using CUDA
using CuQuantum
using CuQuantum.CuDensityMat

# --- Parameters ---
ω = 2π * 1.0   # qubit frequency (MHz)
γ = 0.1         # decay rate (MHz)
dt = 0.01       # time step (μs)
n_steps = 200   # total steps → t_final = 2.0 μs
d = 2           # qubit dimension

# --- Setup ---
ws = WorkStream()
dims = [d]

# --- Operator matrices (2×2, column-major) ---
# σ_z = diag(1, -1)
σ_z = CUDA.CuVector{ComplexF64}([1.0+0im, 0.0, 0.0, -1.0+0im])

# Lowering operator: a = |0⟩⟨1| = [[0,1],[0,0]]
a_mat = CUDA.CuVector{ComplexF64}([0.0+0im, 0.0, 1.0+0im, 0.0])

# Number operator: n = a†a = |1⟩⟨1| = diag(0, 1)
n_mat = CUDA.CuVector{ComplexF64}([0.0+0im, 0.0, 0.0, 1.0+0im])

# Fused sandwich operator: a ρ a† uses a 4-index tensor (d,d,d,d)
# F[i0, i1, j0, j1] = a[i0,j0] * a†[i1,j1]
# with mode_action_duality = [0,1] to get: first pair acts on ket, second on bra
a_matrix = [0.0 1.0; 0.0 0.0]
a_dag = transpose(a_matrix)
aa_dag = zeros(ComplexF64, d, d, d, d)
for j1 = 1:d, j0 = 1:d, i1 = 1:d, i0 = 1:d
    aa_dag[i0, i1, j0, j1] = a_matrix[i0, j0] * a_dag[i1, j1]
end
aa_dag_gpu = CUDA.CuVector{ComplexF64}(vec(aa_dag))

# --- Create elementary operators ---
elem_σz = CuDensityMat.create_elementary_operator(ws, [d], σ_z)
elem_n = CuDensityMat.create_elementary_operator(ws, [d], n_mat)
elem_aa_dag = CuDensityMat.create_elementary_operator(ws, [d, d], aa_dag_gpu)

# --- Build Hamiltonian: H = (ω/2) σ_z ---
hamiltonian_term = CuDensityMat.create_operator_term(ws, dims)
CuDensityMat.append_elementary_product!(
    hamiltonian_term,
    [elem_σz],
    Int32[0],
    Int32[0];
    coefficient = ComplexF64(ω / 2),
)

# --- Build Lindblad dissipator ---
# Sandwich: γ (a ρ a†)
sandwich_term = CuDensityMat.create_operator_term(ws, dims)
CuDensityMat.append_elementary_product!(
    sandwich_term,
    [elem_aa_dag],
    Int32[0, 0],
    Int32[0, 1];
    coefficient = ComplexF64(1.0),
)

# Anticommutator: -½ γ {a†a, ρ} = -½ γ (n ρ + ρ n)
number_term = CuDensityMat.create_operator_term(ws, dims)
CuDensityMat.append_elementary_product!(
    number_term,
    [elem_n],
    Int32[0],
    Int32[0];
    coefficient = ComplexF64(1.0),
)

# --- Assemble Liouvillian L[ρ] = -i[H,ρ] + D[ρ] ---
liouvillian = CuDensityMat.create_operator(ws, dims)

# -i[H, ρ] = (-i)(Hρ) + (i)(ρH)
CuDensityMat.append_term!(
    liouvillian,
    hamiltonian_term;
    duality = 0,
    coefficient = ComplexF64(0, -1),
)
CuDensityMat.append_term!(
    liouvillian,
    hamiltonian_term;
    duality = 1,
    coefficient = ComplexF64(0, +1),
)

# D[ρ] = γ(a ρ a†) - (γ/2)(nρ) - (γ/2)(ρn)
CuDensityMat.append_term!(
    liouvillian,
    sandwich_term;
    duality = 0,
    coefficient = ComplexF64(γ),
)
CuDensityMat.append_term!(
    liouvillian,
    number_term;
    duality = 0,
    coefficient = ComplexF64(-γ / 2),
)
CuDensityMat.append_term!(
    liouvillian,
    number_term;
    duality = 1,
    coefficient = ComplexF64(-γ / 2),
)

# --- Initial state: ρ = |1⟩⟨1| ---
ρ = DenseMixedState{ComplexF64}(ws, (d,); batch_size = 1)
CuDensityMat.allocate_storage!(ρ)
copyto!(ρ.storage, CUDA.CuVector{ComplexF64}([0.0, 0.0, 0.0, 1.0]))

# Scratch state for L[ρ]
ρ_dot = DenseMixedState{ComplexF64}(ws, (d,); batch_size = 1)
CuDensityMat.allocate_storage!(ρ_dot)

# --- Prepare the action (one-time) ---
CuDensityMat.prepare_operator_action!(ws, liouvillian, ρ, ρ_dot)

# --- Forward Euler time-stepping ---
println("  t (μs)    P(|1⟩)    P(|1⟩) exact    error")
println("  " * "-"^50)

for step = 1:n_steps
    t = (step - 1) * dt

    # Compute ρ̇ = L[ρ]
    CuDensityMat.initialize_zero!(ρ_dot)
    CuDensityMat.compute_operator_action!(
        ws,
        liouvillian,
        ρ,
        ρ_dot;
        time = t,
        batch_size = 1,
    )

    # Euler step: ρ += dt * ρ̇
    CuDensityMat.inplace_accumulate!(ρ, ρ_dot, ComplexF64(dt))

    # Print every 20 steps
    if step % 20 == 0
        ρ_cpu = Array(ρ.storage)
        p1 = real(ρ_cpu[4])  # ⟨1|ρ|1⟩
        p1_exact = exp(-γ * step * dt)
        err = abs(p1 - p1_exact)
        @printf("  %.2f      %.4f    %.4f          %.2e\n", step * dt, p1, p1_exact, err)
    end
end

# --- Final state ---
ρ_final = Array(ρ.storage)
println("\nFinal ρ:")
println("  ⟨0|ρ|0⟩ = $(real(ρ_final[1]))")
println("  ⟨1|ρ|1⟩ = $(real(ρ_final[4]))")
println("  Tr(ρ)   = $(real(ρ_final[1] + ρ_final[4]))")

close(ws)
