# CuQuantum.jl

*Julia bindings for the [NVIDIA cuQuantum SDK](https://docs.nvidia.com/cuda/cuquantum/latest/index.html).*

CuQuantum.jl provides high-level Julia wrappers for NVIDIA's cuQuantum libraries, enabling GPU-accelerated quantum computing simulations. The current focus is on **[cuDensityMat](https://docs.nvidia.com/cuda/cuquantum/latest/cudensitymat/index.html)** — a high-performance library for density matrix simulation of open quantum systems.

## Features

- **Density matrix simulation** — Lindblad master equation dynamics with time-dependent Hamiltonians
- **Time-dependent operators** — CPU callbacks for scalar coefficients and tensor elements
- **Backward differentiation** — parameter gradients for quantum optimal control (single-GPU)
- **Expectation values** — ``\text{Tr}(O \rho)`` for arbitrary operators
- **Eigenspectrum** — extreme eigenvalues of many-body operators
- **MPI/NCCL distributed** — multi-GPU forward-pass computation
- **Tensor network contraction** — operators are never materialized as full superoperator matrices, enabling simulation of systems too large for sparse matrix approaches

## Quick Example

```julia
using CuQuantum
using CuQuantum.CuDensityMat
using CUDA

# Create a workspace (manages GPU handle, workspace memory, CUDA stream)
ws = WorkStream()

# Define a 2-cavity system with Fock truncation d=3
dims = [3, 3]

# Build a sigma_z operator on the first cavity
σz_data = CUDA.CuVector{ComplexF64}([1, 0, 0, 0, -1, 0, 0, 0, 0])
elem = create_elementary_operator(ws, [3], σz_data)

# Assemble into a full operator term
term = create_operator_term(ws, dims)
append_elementary_product!(term, [elem], Int32[0], Int32[0])

# Build the composite operator
operator = create_operator(ws, dims)
append_term!(operator, term; duality=0, coefficient=ComplexF64(0, -1))  # -iHρ
append_term!(operator, term; duality=1, coefficient=ComplexF64(0, +1))  # +iρH

# Create input/output density matrices
ρ_in = DenseMixedState{ComplexF64}(ws, (3, 3); batch_size=1)
ρ_out = DenseMixedState{ComplexF64}(ws, (3, 3); batch_size=1)
allocate_storage!(ρ_in)
allocate_storage!(ρ_out)

# Prepare and compute the action: ρ_out = L[ρ_in]
prepare_operator_action!(ws, operator, ρ_in, ρ_out)
initialize_zero!(ρ_out)
compute_operator_action!(ws, operator, ρ_in, ρ_out; time=0.0, batch_size=1)

close(ws)
```

## Supported Hardware

- **GPU architectures**: Turing, Ampere, Ada, Hopper, Blackwell
- **OS**: Linux (x86_64, ARM64)
- **CUDA Toolkit**: 12.x or 13.x

## Package Status

| Component | Status |
|-----------|--------|
| cuDensityMat | Fully wrapped (58 C API functions) |
| cuStateVec | Planned |
| cuTensorNet | Planned |
| cuPauliProp | Planned |
| cuStabilizer | Planned |

## Related Resources

- [NVIDIA cuQuantum SDK documentation](https://docs.nvidia.com/cuda/cuquantum/latest/index.html)
- [cuDensityMat API reference](https://docs.nvidia.com/cuda/cuquantum/latest/cudensitymat/index.html)
- [cuDensityMat C++ examples](https://github.com/NVIDIA/cuQuantum/tree/main/samples/cudensitymat)
- [Piccolo.jl](https://github.com/harmoniqs/Piccolo.jl) — quantum optimal control built on CuQuantum.jl
