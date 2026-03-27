# Benchmarks

We benchmark a Lindblad master equation simulation for ``M`` coupled cavities with Fock truncation ``d=3``:

```math
\dot{\rho} = -i[H, \rho] + \gamma \sum_m \left( a_m \rho a_m^\dagger - \frac{1}{2}\{a_m^\dagger a_m, \rho\} \right)
```

where ``H = \sum_m \chi\, n_m(n_m - 1) + \sum_{n \ne m} \kappa\, a_n^\dagger a_m`` with ``\chi = 2\pi \times 0.2``, ``\kappa = 2\pi \times 0.1``, ``\gamma = 0.01``.

Initial state: ``|1,0,\ldots,0\rangle\langle 1,0,\ldots,0|``.

## A100 Comparison: CuQuantum.jl vs QuantumToolbox.jl

Same hardware, same problem. NVIDIA A100-SXM4-40GB, 12 vCPUs, 85 GB RAM.

### Full Simulation (100 steps)

cuDensityMat uses fixed-step RK4 (``\Delta t = 0.01``). QuantumToolbox.jl uses DP5 adaptive (``\text{atol}=10^{-8}``, ``\text{rtol}=10^{-6}``).

| Cavities (M) | Hilbert dim | CuQuantum.jl (GPU) | QT.jl CPU | QT.jl GPU |
|:---:|:---:|:---:|:---:|:---:|
| 2 | 9 | 0.121 s | **0.0004 s** | ✗ |
| 4 | 81 | 0.434 s | **0.021 s** | ✗ |
| 6 | 729 | **2.61 s** | 5.89 s | ✗ |
| 8 | 6,561 | **249.8 s** | — | — |

At M=6, CuQuantum.jl on A100 is **2.3× faster** than QuantumToolbox.jl CPU. At M=8, CPU sparse is infeasible — only cuDensityMat can run.

### Single Liouvillian Action ``L[\rho]``

| Cavities (M) | Hilbert dim | CuQuantum.jl | QT.jl GPU (cuSPARSE) | Speedup |
|:---:|:---:|:---:|:---:|:---:|
| 2 | 9 | 0.282 ms | **0.037 ms** | cuSPARSE 7.6× |
| 4 | 81 | 1.22 ms | **0.047 ms** | cuSPARSE 26× |
| 6 | 729 | 8.14 ms | **0.901 ms** | cuSPARSE 9.0× |
| 8 | 6,561 | 620 ms | — | GPU only |

QuantumToolbox.jl's cuSPARSE SpMV is faster per-action because it materializes the Liouvillian as a sparse matrix and does a single SpMV call. cuDensityMat uses tensor network contraction — higher per-action overhead but never forms the superoperator, enabling M=8+ where sparse is impossible.

### Notes

- **QT.jl GPU mesolve (✗)**: QuantumToolbox.jl GPU `mesolve` fails on both T4 (sm\_75) and A100 (sm\_80) with a PTX compilation error (`.NaN` modifier in the DP5 kernel). This appears to be a bug in the CUDA.jl/OrdinaryDiffEq.jl GPU codegen, not a hardware limitation. Single-action cuSPARSE benchmarks work fine.
- **M=8 CPU**: the sparse Liouvillian for M=8 (D=6,561) would be a 43M × 43M matrix requiring >100 GB. Only cuDensityMat's tensor network approach can handle this.

## Cross-Framework Comparison (Tesla T4)

### Full Simulation: 100-step RK4 / mesolve

| Cavities (M) | Hilbert dim | CuQuantum.jl | Python CuPy | JAX ext | QT.jl CPU | QuTiP CPU |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2 | 9 | 0.219 s | 1.360 s | 0.382 s | **0.0003 s** | 0.005 s |
| 4 | 81 | 1.484 s | 1.006 s | 1.076 s | **0.021 s** | 0.043 s |
| 6 | 729 | 60.0 s | 64.0 s | 64.4 s | **6.67 s** | 12.3 s |

On T4, CPU sparse is faster than cuDensityMat at all tested sizes — the T4's low FP64 throughput (0.25 TFLOPS) cannot overcome the tensor network contraction overhead. cuDensityMat needs an A100-class GPU to win.

### Single Liouvillian Action ``L[\rho]`` (T4)

| Cavities (M) | Hilbert dim | CuQuantum.jl | Python CuPy | JAX ext | QT.jl GPU (cuSPARSE) | QuTiP CPU |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2 | 9 | 0.478 ms | 0.743 ms | 0.877 ms | **0.046 ms** | 0.005 ms |
| 4 | 81 | 4.93 ms | 5.79 ms | 6.00 ms | **0.072 ms** | 14.0 ms |
| 6 | 729 | **149 ms** | 159 ms | 161 ms | 4.06 ms | OOM |

## Wrapper Overhead

All three cuDensityMat wrappers call the same C library. Wrapper dispatch overhead measured on T4:

| Wrapper | Single L[rho] at M=6 | vs Julia |
|:---|:---:|:---:|
| **CuQuantum.jl** (Julia) | 149.2 ms | baseline |
| Python CuPy | 159.4 ms | 7% slower |
| JAX extension | 160.5 ms | 8% slower |

Julia's advantage comes from lower per-call dispatch overhead (no Python GIL, direct `ccall`). At M=6 the difference is modest because the cuDensityMat kernel dominates. At small sizes (M=2), the gap is larger: Julia 0.478 ms vs Python 0.743 ms (55% faster).

## Why cuDensityMat Scales Better

The CPU sparse approach explicitly constructs the Liouvillian superoperator ``\mathcal{L}`` as a ``D^2 \times D^2`` sparse matrix (where ``D = d^M``), then computes ``\mathcal{L} \cdot \text{vec}(\rho)`` via SpMV. Memory scales as ``O(M \cdot D^2)`` non-zeros.

cuDensityMat decomposes ``\mathcal{L}[\rho]`` as a sum of tensor network contractions over elementary operators. Each contraction involves small tensors (``d \times d`` per mode) and the density matrix, never forming the full superoperator. Computational cost scales more favorably with the number of modes ``M``.

## Validation

CPU and GPU implementations produce **bitwise-identical results** for the static Liouvillian (verified at ``t=0``). All frameworks agree on final populations to within adaptive solver tolerances.

## Environment Details

### A100 benchmarks
- **GPU**: NVIDIA A100-SXM4-40GB (2 TB/s bandwidth, 9.7 TFLOPS FP64)
- **CPU**: 12 vCPUs (GCE a2-highgpu-1g, 85 GB RAM)
- **Julia**: 1.12.5, CUDA.jl, cuQuantum\_jll
- **CUDA**: 12.8, Driver 570.211.01

### T4 cross-framework benchmarks
- **GPU**: NVIDIA Tesla T4 (16 GB, 300 GB/s bandwidth, 0.25 TFLOPS FP64)
- **CPU**: 8 vCPUs (GCE n1-standard-8, 30 GB RAM)
- **Julia**: 1.12.5
- **Python**: 3.10.12 (QuTiP, CuPy), 3.11.15 (JAX ext)
- **QuTiP**: 5.2.3
- **JAX**: 0.6.2
- **cuquantum-python**: 25.3.0 (CuPy), 25.11.1 (JAX ext)
- **cuquantum-python-jax**: 0.0.3 (experimental, built from source)
- **CUDA**: 12.8, Driver 570.211.01

### Reproduction

Benchmark scripts are in [`benchmark/comparison/`](https://github.com/harmoniqs/CuQuantum.jl/tree/main/benchmark/comparison). See the [README](https://github.com/harmoniqs/CuQuantum.jl/blob/main/benchmark/comparison/README.md) for setup instructions.
