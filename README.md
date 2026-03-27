# CuQuantum.jl

[![CI](https://github.com/harmoniqs/CuQuantum.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/harmoniqs/CuQuantum.jl/actions/workflows/CI.yml)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://harmoniqs.github.io/CuQuantum.jl/dev)

Julia bindings for the [NVIDIA cuQuantum SDK](https://docs.nvidia.com/cuda/cuquantum/latest/index.html), providing GPU-accelerated density matrix simulation for open quantum systems.

## Motivation

Standard Lindblad solvers materialize the Liouvillian as a sparse `D² × D²` matrix and compute `L[ρ]` via sparse matrix-vector multiply. For `M` coupled cavities with Fock truncation `d`, the Hilbert space dimension is `D = dᴹ`. At M=8, D=6,561 and the Liouvillian has 43 million rows — requiring >77 GB just for the sparse matrix.

CuQuantum.jl wraps NVIDIA's [cuDensityMat](https://docs.nvidia.com/cuda/cuquantum/latest/cudensitymat/index.html) library, which decomposes `L[ρ]` into tensor network contractions over small per-mode operators. It never forms the full superoperator, enabling simulation of systems where sparse approaches are infeasible.

### When to use cuDensityMat

| Regime | Hilbert dim | Best approach |
|:---|:---:|:---|
| M ≤ 4 | ≤ 81 | CPU sparse ([QuantumToolbox.jl](https://github.com/qutip/QuantumToolbox.jl)) |
| M = 5–6 | ≤ 729 | GPU cuSPARSE SpMV (fastest per-action) |
| **M ≥ 8** | **≥ 6,561** | **cuDensityMat (only option)** |

cuDensityMat also supports **native batching** — many density matrices evolved with different parameters in a single kernel launch — and **backward differentiation** for parameter gradients, both essential for quantum optimal control.

## Features

- **Lindblad master equation** — time-dependent Hamiltonians with dissipation
- **Time-dependent callbacks** — CPU scalar and tensor callbacks for driven systems
- **Backward differentiation** — parameter gradients via VJP (single-GPU)
- **Native batching** — parallel evolution of many states with different parameters
- **Expectation values** — `Tr(Oρ)` for arbitrary operators
- **MPI/NCCL distributed** — multi-GPU forward-pass computation
- **Tensor network contraction** — never materializes the full superoperator

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/harmoniqs/CuQuantum.jl")
```

Requires a CUDA-capable GPU (Turing or newer) and Julia 1.10+.

## Benchmarks

All benchmarks use M coupled cavities with Fock truncation d=3, Kerr nonlinearity + all-to-all coupling + photon loss. See [Methodology](#methodology) for the full problem specification.

### Single `L[ρ]` Action — A100

Time for one evaluation of the Liouvillian action (median, after warmup).

| M | D | CuQuantum.jl | cuSPARSE SpMV | Sparse feasible? |
|:---:|:---:|:---:|:---:|:---:|
| 2 | 9 | 0.27 ms | **0.039 ms** | yes |
| 4 | 81 | 1.22 ms | **0.048 ms** | yes |
| 6 | 729 | 6.45 ms | **0.90 ms** | yes |
| 8 | 6,561 | **620 ms** | — | no (>77 GB) |
| 9 | 19,683 | **6,742 ms** | — | no |

### Full 100-step RK4 Simulation — A100

| M | D | CuQuantum.jl | cuSPARSE RK4 | QT.jl CPU |
|:---:|:---:|:---:|:---:|:---:|
| 4 | 81 | 0.46 s | **0.018 s** | 0.021 s |
| 6 | 729 | 2.62 s | **0.32 s** | 5.89 s |
| 8 | 6,561 | **250 s** | — | — |

### Batched Evolution — A100

Many density matrices evolved with different coupling strengths in one kernel launch. This is the parameter sweep use case for quantum optimal control.

| M | D | Batch | cuDM batched | cuDM sequential | Speedup |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 2 | 9 | 64 | **0.45 ms** | 20.6 ms | 46× |
| 2 | 9 | 256 | **0.38 ms** | 70.1 ms | 186× |
| 4 | 81 | 64 | **2.79 ms** | 70.1 ms | 25× |
| 4 | 81 | 256 | **8.05 ms** | 280.7 ms | 35× |
| 6 | 729 | 64 | **297 ms** | 497 ms | 1.7× |
| 6 | 729 | 256 | **1,146 ms** | 1,990 ms | 1.7× |

At small system sizes (M=2–4), native batching amortizes kernel launch overhead by up to 186×. At M=6, the per-action cost dominates and batching gives ~1.7× speedup.

### Cross-Framework Comparison — Tesla T4

| M | D | CuQuantum.jl | Python CuPy | JAX ext | QT.jl CPU | QuTiP |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2 | 9 | 0.22 s | 1.36 s | 0.38 s | **0.0003 s** | 0.005 s |
| 4 | 81 | 1.48 s | 1.01 s | 1.08 s | **0.021 s** | 0.043 s |
| 6 | 729 | 60.0 s | 64.0 s | 64.4 s | **6.67 s** | 12.3 s |

Julia wrapper is 7–8% faster than the Python (CuPy) and JAX wrappers over the same C library, due to lower dispatch overhead via direct `ccall`.

### Wrapper Overhead

| Wrapper | Single L[ρ] at M=6 (T4) |
|:---|:---:|
| **CuQuantum.jl** (Julia) | 149 ms |
| Python CuPy | 159 ms (+7%) |
| JAX extension | 161 ms (+8%) |

## Methodology

All benchmarks use the same physical system:

```
H = Σ_m χ n_m(n_m-1) + Σ_{n≠m} κ a_n†a_m
L[ρ] = -i[H,ρ] + γ Σ_m (a_m ρ a_m† - ½{n_m, ρ})
```

- **Fock truncation**: d = 3 per cavity
- **Parameters**: χ = 2π × 0.2, κ = 2π × 0.1, γ = 0.01
- **Initial state**: |1,0,...,0⟩⟨1,0,...,0|
- **Integration**: 100-step RK4 with dt = 0.01 (cuDensityMat, cuSPARSE) or DP5 adaptive with atol=10⁻⁸, rtol=10⁻⁶ (QuantumToolbox.jl, QuTiP)
- **Timing**: wall-clock, excludes JIT/compilation warmup
- **Validation**: CPU and GPU produce bitwise-identical results for static Liouvillian at t=0

### Hardware

| | A100 benchmarks | T4 benchmarks |
|:---|:---|:---|
| **GPU** | A100-SXM4-40GB (2 TB/s, 9.7 TFLOPS FP64) | Tesla T4 (300 GB/s, 0.25 TFLOPS FP64) |
| **CPU** | 12 vCPUs, 85 GB RAM | 8 vCPUs, 30 GB RAM |
| **Instance** | GCE a2-highgpu-1g | GCE n1-standard-8 |
| **CUDA** | 12.8, Driver 570.211.01 | 12.8, Driver 570.211.01 |

### Software

| Component | Version |
|:---|:---|
| Julia | 1.12.5 |
| CUDA.jl | 5.8.x |
| cuQuantum SDK (JLL) | 25.11.0 |
| QuantumToolbox.jl | latest |
| QuTiP | 5.2.3 |
| JAX | 0.6.2 |
| cuquantum-python-jax | 0.0.3 (built from source) |

### Reproduction

Benchmark scripts are in [`benchmark/comparison/`](benchmark/comparison/) and [`benchmark/batched/`](benchmark/batched/). See the [comparison README](benchmark/comparison/README.md) for detailed setup instructions including GCE instance creation, Julia/Python installation, and the JAX extension source build.

## Documentation

Full documentation: [harmoniqs.github.io/CuQuantum.jl](https://harmoniqs.github.io/CuQuantum.jl/dev)

## License

MIT — see [LICENSE](LICENSE).
