# Quantum States

CuQuantum.jl provides two state types, both stored as dense tensors on the GPU.

### `DensePureState{T}`

A pure quantum state ``|\psi\rangle`` stored as a rank-``M`` tensor with shape ``(d_1, d_2, \ldots, d_M)``.

```julia
# 2-qubit pure state (d₁=2, d₂=2)
ψ = DensePureState{ComplexF64}(ws, (2, 2); batch_size=1)
allocate_storage!(ψ)
```

### `DenseMixedState{T}`

A density matrix ``\rho`` stored as a rank-``2M`` tensor with shape ``(d_1, \ldots, d_M, d_1, \ldots, d_M)``, where the first ``M`` indices are ket (row) and the last ``M`` are bra (column).

```julia
# 2-cavity mixed state (d₁=3, d₂=3), density matrix is 9×9
ρ = DenseMixedState{ComplexF64}(ws, (3, 3); batch_size=1)
allocate_storage!(ρ)
```

## Supported Data Types

- `ComplexF64` — double precision (recommended)
- `ComplexF32` — single precision (faster, less accurate)

## State Operations

### Initialization

```julia
# Zero-initialize (via cuDensityMat C API)
initialize_zero!(state)

# Copy data from CPU
rho_cpu = zeros(ComplexF64, 9 * 9)
rho_cpu[1] = 1.0  # |0,0⟩⟨0,0|
copyto!(ρ.storage, CUDA.CuVector{ComplexF64}(rho_cpu))
```

### Linear Algebra

```julia
# Scale: ρ ← α * ρ
inplace_scale!(ρ, ComplexF64(0.5))

# Accumulate: ρ_dest ← ρ_dest + α * ρ_src
inplace_accumulate!(ρ_dest, ρ_src, ComplexF64(dt))

# Trace: Tr(ρ)
tr = trace(ρ)  # returns Vector{ComplexF64} of length batch_size

# Norm: ||ρ||²_F (squared Frobenius norm)
n = norm(ρ)  # returns Vector{Float64}

# Inner product: ⟨ρ_left | ρ_right⟩
ip = inner_product(ρ_left, ρ_right)
```

## Batched States

All state types support batched operations — multiple independent states processed in parallel:

```julia
# 16 density matrices in parallel
ρ_batch = DenseMixedState{ComplexF64}(ws, (3, 3); batch_size=16)
allocate_storage!(ρ_batch)
```

The batch dimension is the last index in storage. All compute functions (`trace`, `norm`, `compute_operator_action!`, etc.) operate on all batch members simultaneously.

## Memory Management

State storage is allocated on the GPU via `allocate_storage!`. GPU memory is released when the `WorkStream` is closed.

```julia
ρ = DenseMixedState{ComplexF64}(ws, (3, 3); batch_size=1)
allocate_storage!(ρ)
# ... use ρ ...
close(ws)
```
