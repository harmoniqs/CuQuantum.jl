# Operators

## Operator Hierarchy

CuQuantum.jl follows the cuDensityMat operator hierarchy:

### Elementary Operators

An `ElementaryOperator` is a tensor acting on one or more modes of the Hilbert space. It can be:
- **Dense** — full ``d \times d`` matrix (e.g., Pauli matrices, annihilation operators)
- **Multidiagonal** — sparse diagonal structure (e.g., number operators)

Elementary operators are uploaded to GPU memory once and reused across many terms.

```julia
# Single-mode annihilation operator (d=3 Fock space)
a_data = CUDA.CuVector{ComplexF64}([0, 0, 0, 1, 0, 0, 0, √2, 0])
elem_a = create_elementary_operator(ws, [3], a_data)

# Two-mode fused operator for a⊗a† (used in Lindblad sandwich terms)
aa_dag = zeros(ComplexF64, 3, 3, 3, 3)
# ... fill in tensor elements ...
elem_fused = create_elementary_operator(ws, [3, 3], CUDA.CuVector{ComplexF64}(vec(aa_dag)))
```

### Operator Terms

An `OperatorTerm` is a sum of tensor products of elementary operators. Each product specifies:
- Which elementary operators participate
- Which modes they act on (`modes_acted_on`)
- Whether each mode acts on the ket (left) or bra (right) side of ``\rho`` (`mode_action_duality`)
- A scalar coefficient (static or time-dependent via callback)

```julia
term = create_operator_term(ws, [3, 3])  # 2-mode system

# Append: σ_z on mode 0, ket-side, with coefficient χ
append_elementary_product!(term, [elem_σz], Int32[0], Int32[0];
    coefficient=ComplexF64(χ))
```

The `mode_action_duality` array controls how each index of the elementary operator interacts with the density matrix:
- `0` — acts on the ket (left) side: ``O \rho``
- `1` — acts on the bra (right) side: ``\rho O``

For Lindblad sandwich terms ``a \rho a^\dagger``, use a **fused** 2-mode elementary operator with `mode_action_duality = [0, 1]` and `modes_acted_on = [m, m]` (same physical mode for both).

### Operators

An `Operator` is a sum of `OperatorTerm`s, each appended with a duality flag and coefficient:

```julia
operator = create_operator(ws, dims)

# -i H ρ  (Hamiltonian acts from the left)
append_term!(operator, hamiltonian_term; duality=0, coefficient=ComplexF64(0, -1))

# +i ρ H  (Hamiltonian acts from the right)
append_term!(operator, hamiltonian_term; duality=1, coefficient=ComplexF64(0, +1))
```

### Operator Action

Once assembled, the operator action ``L[\rho]`` is computed in two phases:

1. **`prepare_operator_action!`** — one-time contraction planning and workspace allocation
2. **`compute_operator_action!`** — evaluate ``L[\rho]`` at a given time (can be called repeatedly)

```julia
prepare_operator_action!(ws, operator, ρ_in, ρ_out)

# Time-stepping loop
for t in time_steps
    initialize_zero!(ρ_out)
    compute_operator_action!(ws, operator, ρ_in, ρ_out; time=t, batch_size=1)
    # ... RK4 or other integration ...
end
```

## Matrix Operators

For operators that act on the **full** Hilbert space (not decomposed into tensor products), use `MatrixOperator`:

```julia
# Full d_total × d_total matrix on GPU
mat_op = create_matrix_operator(ws, dims, full_matrix_gpu)
```

These are less efficient than elementary operators for large systems but useful for small systems or pre-computed matrices.
