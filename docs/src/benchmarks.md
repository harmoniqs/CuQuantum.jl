# Benchmarks

All benchmarks use the same Lindblad master equation for ``M`` coupled cavities with Fock truncation ``d=3``:

```math
\dot{\rho} = -i[H, \rho] + \gamma \sum_m \left( a_m \rho a_m^\dagger - \frac{1}{2}\{a_m^\dagger a_m, \rho\} \right)
```

with ``H = \sum_m \chi\, n_m(n_m - 1) + \sum_{n \ne m} \kappa\, a_n^\dagger a_m``, ``\chi = 2\pi \times 0.2``, ``\kappa = 2\pi \times 0.1``, ``\gamma = 0.01``. Initial state: ``|1,0,\ldots,0\rangle\langle 1,0,\ldots,0|``.

## When to Use cuDensityMat

cuDensityMat uses tensor network contraction — it never materializes the full Liouvillian superoperator. For small systems (``M \le 6``), the per-action overhead makes it slower than sparse matrix approaches. For large systems (``M \ge 8``), it is the **only viable option** because the sparse Liouvillian exceeds GPU memory.

| Regime | Best approach |
|:---|:---|
| ``M \le 4`` (D ≤ 81) | CPU sparse (QuantumToolbox.jl) |
| ``M = 5\text{--}6`` (D ≤ 729) | GPU cuSPARSE SpMV |
| ``M \ge 8`` (D ≥ 6,561) | **cuDensityMat** (only option) |

## A100: CuQuantum.jl vs QuantumToolbox.jl

Same hardware (NVIDIA A100-SXM4-40GB, 12 vCPUs), same RK4 integrator, same time steps.

### Full 100-step RK4 Simulation

| M | D | CuQuantum.jl | QT.jl GPU cuSPARSE | QT.jl CPU |
|:---:|:---:|:---:|:---:|:---:|
| 4 | 81 | 0.46 s | **0.018 s** | 0.021 s |
| 6 | 729 | 2.62 s | **0.32 s** | 5.89 s |
| 8 | 6,561 | **250 s** | infeasible | infeasible |

At ``M=6``, cuSPARSE is 8× faster than cuDensityMat. At ``M=8``, the Liouvillian is ``43\text{M} \times 43\text{M}`` sparse (~77 GB) — only cuDensityMat can run.

### Single ``L[\rho]`` Action

| M | D | CuQuantum.jl | QT.jl GPU cuSPARSE |
|:---:|:---:|:---:|:---:|
| 2 | 9 | 0.267 ms | **0.039 ms** |
| 4 | 81 | 1.22 ms | **0.048 ms** |
| 6 | 729 | 6.45 ms | **0.90 ms** |
| 8 | 6,561 | 620 ms | infeasible |

## Cross-Framework Comparison (Tesla T4)

Five frameworks, same problem. cuDensityMat benchmarks use RK4; QuantumToolbox.jl and QuTiP use adaptive solvers.

### Full 100-step Simulation

| M | D | CuQuantum.jl | Python CuPy | JAX ext | QT.jl CPU | QuTiP |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2 | 9 | 0.22 s | 1.36 s | 0.38 s | **0.0003 s** | 0.005 s |
| 4 | 81 | 1.48 s | 1.01 s | 1.08 s | **0.021 s** | 0.043 s |
| 6 | 729 | 60.0 s | 64.0 s | 64.4 s | **6.67 s** | 12.3 s |

On T4, CPU sparse wins at all sizes. The T4 has only 0.25 TFLOPS FP64 — cuDensityMat's tensor network contraction needs A100-class throughput to be competitive.

### Single ``L[\rho]`` Action (T4)

| M | D | CuQuantum.jl | Python CuPy | JAX ext | QT.jl GPU cuSPARSE | QuTiP |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2 | 9 | 0.48 ms | 0.74 ms | 0.88 ms | **0.046 ms** | 0.005 ms |
| 4 | 81 | 4.93 ms | 5.79 ms | 6.00 ms | **0.072 ms** | 14.0 ms |
| 6 | 729 | 149 ms | 159 ms | 161 ms | **4.06 ms** | OOM |

## Wrapper Overhead

All three cuDensityMat wrappers call the same C library. The overhead at ``M=6`` on T4:

| Wrapper | Single ``L[\rho]`` | vs Julia |
|:---|:---:|:---:|
| **CuQuantum.jl** | 149 ms | — |
| Python CuPy | 159 ms | +7% |
| JAX extension | 161 ms | +8% |

Julia is faster due to direct `ccall` without Python GIL overhead. The gap narrows at large sizes where the cuDensityMat kernel dominates.

## Validation

CPU and GPU produce **bitwise-identical** results for the static Liouvillian at ``t=0``. All frameworks agree on final populations to within solver tolerances across 100 time steps.

## Environment

| | A100 benchmarks | T4 benchmarks |
|:---|:---|:---|
| **GPU** | A100-SXM4-40GB | Tesla T4 (16 GB) |
| **CPU** | 12 vCPUs, 85 GB | 8 vCPUs, 30 GB |
| **Julia** | 1.12.5 | 1.12.5 |
| **CUDA** | 12.8 | 12.8 |
| **Python** | — | 3.10 / 3.11 (JAX) |
| **QuTiP** | — | 5.2.3 |
| **JAX** | — | 0.6.2 |

## Reproduction

Benchmark scripts are in [`benchmark/comparison/`](https://github.com/harmoniqs/CuQuantum.jl/tree/main/benchmark/comparison). See the [README](https://github.com/harmoniqs/CuQuantum.jl/blob/main/benchmark/comparison/README.md) for setup and run instructions.
