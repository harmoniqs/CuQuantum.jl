# Qubit Expectation Value
#
# Compute Tr(σ_z ρ) for a single qubit in state |0⟩⟨0|.
# This is the simplest possible CuDensityMat workflow:
#   1. Create a WorkStream
#   2. Define an operator (σ_z)
#   3. Create a density matrix state
#   4. Compute the expectation value
#
# Expected result: Tr(σ_z |0⟩⟨0|) = 1.0

using CUDA
using CuQuantum
using CuQuantum.CuDensityMat

# --- Setup ---
ws = WorkStream()
dims = [2]  # single qubit (Hilbert space dimension 2)

# --- Build σ_z operator ---
# σ_z = diag(1, -1) stored column-major as a 2×2 matrix
σ_z_data = CUDA.CuVector{ComplexF64}([1.0+0im, 0.0, 0.0, -1.0+0im])
elem = CuDensityMat.create_elementary_operator(ws, [2], σ_z_data)

# Wrap into operator hierarchy: ElementaryOperator → OperatorTerm → Operator
term = CuDensityMat.create_operator_term(ws, dims)
CuDensityMat.append_elementary_product!(term, [elem], Int32[0], Int32[0])

operator = CuDensityMat.create_operator(ws, dims)
CuDensityMat.append_term!(operator, term; duality = 0)

# --- Create state ρ = |0⟩⟨0| ---
ρ = DenseMixedState{ComplexF64}(ws, (2,); batch_size = 1)
CuDensityMat.allocate_storage!(ρ)
# |0⟩⟨0| = [[1,0],[0,0]] stored column-major
copyto!(ρ.storage, CUDA.CuVector{ComplexF64}([1.0, 0.0, 0.0, 0.0]))

# --- Compute expectation value ---
exp = CuDensityMat.create_expectation(ws, operator)
CuDensityMat.prepare_expectation!(ws, exp, ρ)

result = CUDA.zeros(ComplexF64, 1)
CuDensityMat.compute_expectation!(ws, exp, ρ, result; time = 0.0, batch_size = 1)

val = Array(result)[1]
println("Tr(σ_z |0⟩⟨0|) = $(real(val))")  # should print 1.0

# --- Cleanup ---
CuDensityMat.destroy_expectation(exp)
close(ws)
