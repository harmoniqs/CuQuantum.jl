# CuQuantum.jl

[![CI](https://github.com/harmoniqs/CuQuantum.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/harmoniqs/CuQuantum.jl/actions/workflows/CI.yml)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://harmoniqs.github.io/CuQuantum.jl/dev)

Julia bindings for the [NVIDIA cuQuantum SDK](https://docs.nvidia.com/cuda/cuquantum/latest/index.html), providing GPU-accelerated density matrix simulation for open quantum systems.

## Features

- **Lindblad master equation** — time-dependent Hamiltonians with photon loss, dephasing, and custom dissipation
- **Time-dependent callbacks** — CPU scalar and tensor callbacks for driven/modulated systems
- **Backward differentiation** — parameter gradients for quantum optimal control
- **Expectation values** and **eigenspectrum** computation
- **Tensor network contraction** — operators are never materialized as full superoperator matrices
- **MPI/NCCL distributed** — multi-GPU forward-pass computation

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/harmoniqs/CuQuantum.jl")
```

Requires a CUDA-capable GPU (Turing or newer) and Julia 1.10+.

## Quick Example

```julia
using CuQuantum, CuQuantum.CuDensityMat, CUDA

ws = WorkStream()
dims = [3, 3]  # 2 cavities, Fock truncation d=3

# Build σ_z on cavity 0
σz = CUDA.CuVector{ComplexF64}([1, 0, 0, 0, -1, 0, 0, 0, 0])
elem = create_elementary_operator(ws, [3], σz)
term = create_operator_term(ws, dims)
append_elementary_product!(term, [elem], Int32[0], Int32[0])

# Assemble Hamiltonian commutator: -i[H, ρ]
op = create_operator(ws, dims)
append_term!(op, term; duality=0, coefficient=ComplexF64(0, -1))
append_term!(op, term; duality=1, coefficient=ComplexF64(0, +1))

# Compute L[ρ]
ρ = DenseMixedState{ComplexF64}(ws, (3, 3); batch_size=1)
ρ_dot = DenseMixedState{ComplexF64}(ws, (3, 3); batch_size=1)
allocate_storage!(ρ); allocate_storage!(ρ_dot)

prepare_operator_action!(ws, op, ρ, ρ_dot)
initialize_zero!(ρ_dot)
compute_operator_action!(ws, op, ρ, ρ_dot; time=0.0, batch_size=1)

close(ws)
```

## Performance

cuDensityMat uses tensor network contraction to compute operator actions without materializing the full Liouvillian superoperator. This enables simulation of systems too large for sparse matrix approaches.

### A100: 100-step RK4 Simulation

| Cavities | Hilbert dim | CuQuantum.jl | QT.jl GPU (cuSPARSE) | QT.jl CPU |
|:---:|:---:|:---:|:---:|:---:|
| 4 | 81 | 0.46 s | **0.018 s** | 0.02 s |
| **6** | **729** | 2.62 s | **0.32 s** | 5.9 s |
| **8** | **6,561** | **250 s** | — | — |

### Cross-Framework: 100-step Lindblad Simulation (Tesla T4)

| Cavities | Hilbert dim | CuQuantum.jl | Python cuDensityMat | JAX cuDensityMat | QT.jl CPU | QuTiP CPU |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2 | 9 | 0.22 s | 1.36 s | 0.38 s | **0.0003 s** | 0.005 s |
| 4 | 81 | 1.48 s | 1.01 s | 1.08 s | **0.02 s** | 0.04 s |
| 6 | 729 | 60.0 s | 64.0 s | 64.4 s | **6.7 s** | 12.3 s |

See the [benchmarks](https://harmoniqs.github.io/CuQuantum.jl/dev/benchmarks/) for the full comparison including A100 vs T4, single-action timings, and reproduction scripts.

## Documentation

Full documentation is available at [harmoniqs.github.io/CuQuantum.jl](https://harmoniqs.github.io/CuQuantum.jl/dev).

## License

MIT — see [LICENSE](LICENSE).
