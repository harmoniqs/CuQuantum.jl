# CuQuantum.jl

*Julia bindings for the [NVIDIA cuQuantum SDK](https://docs.nvidia.com/cuda/cuquantum/latest/index.html).*

CuQuantum.jl provides Julia wrappers for NVIDIA's cuQuantum libraries, enabling GPU-accelerated quantum computing simulations. The package wraps **[cuDensityMat](https://docs.nvidia.com/cuda/cuquantum/latest/cudensitymat/index.html)** — a library for density matrix simulation of open quantum systems via tensor network contraction.

## Why CuQuantum.jl?

Standard Lindblad solvers materialize the Liouvillian superoperator as a sparse matrix and compute ``L[\rho]`` via SpMV. This works well for small systems but memory scales as ``O(d^{2M})`` — a 6-cavity system with ``d=3`` already requires a 531K × 531K sparse matrix.

cuDensityMat decomposes ``L[\rho]`` as tensor network contractions over small per-mode operators. It never forms the full superoperator, enabling simulation of systems where sparse approaches are infeasible.

| System size | Sparse matrix approach | cuDensityMat |
|:---:|:---:|:---:|
| M=6 (D=729) | 40 ms (CPU), 0.9 ms (cuSPARSE) | 6.4 ms (A100) |
| M=8 (D=6,561) | infeasible (>77 GB) | **620 ms (A100)** |
| M=9 (D=19,683) | infeasible | **6.7 s (A100)** |

## Features

- **Lindblad master equation** — time-dependent Hamiltonians with dissipation
- **Time-dependent callbacks** — CPU scalar and tensor callbacks for driven systems
- **Backward differentiation** — parameter gradients for quantum optimal control (single-GPU)
- **Expectation values** — ``\text{Tr}(O \rho)`` for arbitrary operators
- **MPI/NCCL distributed** — multi-GPU forward-pass computation
- **Tensor network contraction** — never materializes the full superoperator

## Quick Example

```julia
using CuQuantum, CuQuantum.CuDensityMat, CUDA

ws = WorkStream()
dims = [3, 3]  # 2 cavities, Fock truncation d=3

# Upload a σ_z operator to GPU
σz = CUDA.CuVector{ComplexF64}([1, 0, 0, 0, -1, 0, 0, 0, 0])
elem = create_elementary_operator(ws, [3], σz)

# Build an operator term and attach it to a composite operator
term = create_operator_term(ws, dims)
append_elementary_product!(term, [elem], Int32[0], Int32[0])

op = create_operator(ws, dims)
append_term!(op, term; duality=0, coefficient=ComplexF64(0, -1))  # -iHρ
append_term!(op, term; duality=1, coefficient=ComplexF64(0, +1))  # +iρH

# Allocate input/output density matrices
ρ = DenseMixedState{ComplexF64}(ws, (3, 3); batch_size=1)
ρ̇ = DenseMixedState{ComplexF64}(ws, (3, 3); batch_size=1)
allocate_storage!(ρ); allocate_storage!(ρ̇)

# Compute L[ρ]
prepare_operator_action!(ws, op, ρ, ρ̇)
initialize_zero!(ρ̇)
compute_operator_action!(ws, op, ρ, ρ̇; time=0.0, batch_size=1)

close(ws)
```

## Supported Hardware

- **GPU architectures**: Turing (T4), Ampere (A100), Ada (L4/L40), Hopper (H100), Blackwell (B200)
- **CUDA Toolkit**: 12.x or 13.x
- **OS**: Linux (x86\_64, ARM64)

## Related Resources

- [NVIDIA cuQuantum SDK documentation](https://docs.nvidia.com/cuda/cuquantum/latest/index.html)
- [cuDensityMat C API reference](https://docs.nvidia.com/cuda/cuquantum/latest/cudensitymat/index.html)
- [cuDensityMat C++ examples](https://github.com/NVIDIA/cuQuantum/tree/main/samples/cudensitymat)
- [Piccolo.jl](https://github.com/harmoniqs/Piccolo.jl) — quantum optimal control built on CuQuantum.jl
