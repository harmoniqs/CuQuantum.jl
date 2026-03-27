# Benchmarks

We benchmark a Lindblad master equation simulation for ``M`` coupled cavities with Fock truncation ``d=3``:

```math
\dot{\rho} = -i[H, \rho] + \gamma \sum_m \left( a_m \rho a_m^\dagger - \frac{1}{2}\{a_m^\dagger a_m, \rho\} \right)
```

where ``H = \sum_m \chi\, n_m(n_m - 1) + \sum_{n \ne m} \kappa\, a_n^\dagger a_m`` with ``\chi = 2\pi \times 0.2``, ``\kappa = 2\pi \times 0.1``, ``\gamma = 0.01``.

Initial state: ``|1,0,\ldots,0\rangle\langle 1,0,\ldots,0|``.

## Cross-Framework Comparison (Tesla T4)

### Full Simulation: 100-step RK4 / mesolve

Wall-clock time (seconds) for a full 100-step simulation (``\Delta t = 0.01``, ``t_\text{final} = 1.0``). cuDensityMat benchmarks use fixed-step RK4; QuantumToolbox.jl and QuTiP use adaptive solvers (DP5 / Adams, ``\text{atol}=10^{-8}``, ``\text{rtol}=10^{-6}``).

| Cavities (M) | Hilbert dim | CuQuantum.jl | Python CuPy | JAX ext | QT.jl CPU | QuTiP CPU |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2 | 9 | 0.219 s | 1.360 s | 0.382 s | **0.0003 s** | 0.005 s |
| 4 | 81 | 1.484 s | 1.006 s | 1.076 s | **0.021 s** | 0.043 s |
| 6 | 729 | 60.0 s | 64.0 s | 64.4 s | **6.67 s** | 12.3 s |
| 8 | 6,561 | — | — | — | — | — |

### Single Liouvillian Action ``L[\rho]``

Time for a single evaluation of the Liouvillian action (median of 50 trials, after warmup).

| Cavities (M) | Hilbert dim | CuQuantum.jl | Python CuPy | JAX ext | QT.jl GPU (cuSPARSE) | QuTiP CPU |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2 | 9 | 0.478 ms | 0.743 ms | 0.877 ms | **0.046 ms** | 0.005 ms |
| 4 | 81 | 4.93 ms | 5.79 ms | 6.00 ms | **0.072 ms** | 14.0 ms |
| 6 | 729 | **149 ms** | 159 ms | 161 ms | 4.06 ms | OOM |

### Notes

- M=8 is skipped in the comparison: cuDensityMat `prepare_action` takes >15 minutes, and Python operator construction takes >10 minutes due to O(M²) `OperatorTerm.__add__` overhead.
- QuantumToolbox.jl GPU mesolve could not run on T4 (sm\_75) — the DP5 ODE solver kernel requires sm\_80+ (A100). Single-action benchmarks worked.
- QuTiP single-action at M=6 failed with OOM (tried to allocate the full dense Liouvillian as a 531,441 × 531,441 matrix).

## CuQuantum.jl vs CPU Sparse (A100)

Earlier benchmarks on NVIDIA A100 40GB show the crossover point where cuDensityMat becomes faster:

| Cavities (M) | Hilbert dim | ``\rho`` elements | cuDensityMat (A100) | CPU Sparse | GPU Speedup |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 2 | 9 | 81 | 0.27 ms | 0.001 ms | CPU wins |
| 4 | 81 | 6,561 | 1.2 ms | 0.14 ms | CPU wins |
| **6** | **729** | **531K** | **8.1 ms** | **40 ms** | **4.9×** |
| 8 | 6,561 | 43M | 621 ms | N/A | GPU only |
| 9 | 19,683 | 387M | 6,742 ms | N/A | GPU only |

### Key Findings

- **Crossover at M=6** (729-dimensional Hilbert space): cuDensityMat on A100 is 4.9× faster than CPU sparse matrix-vector multiply.
- **Beyond M=6, CPU is infeasible**: the sparse superoperator for M=8 would require >100 GB to construct. cuDensityMat never materializes the superoperator — it uses tensor network contraction.
- **GPU overhead dominates at small sizes**: for M=2,4, kernel launch overhead (~0.3 ms) exceeds the actual computation time. CPU wins trivially.
- **On T4, cuDensityMat is slower than CPU sparse even at M=6** — the A100's higher memory bandwidth (2 TB/s vs 300 GB/s) and FP64 throughput (9.7 vs 0.25 TFLOPS) make the difference.

## Wrapper Overhead Comparison

All three cuDensityMat wrappers call the same C library. The wrapper overhead:

| Wrapper | Single L[rho] at M=6 | vs Julia |
|:---|:---:|:---:|
| **CuQuantum.jl** (Julia) | 149.2 ms | baseline |
| Python CuPy | 159.4 ms | 7% slower |
| JAX extension | 160.5 ms | 8% slower |

Julia's advantage comes from lower per-call dispatch overhead (no Python GIL, direct `ccall`). At M=6 the difference is modest because the cuDensityMat kernel dominates. At small sizes (M=2), the difference is more pronounced: Julia 0.478 ms vs Python 0.743 ms (55% faster).

## Why cuDensityMat Scales Better

The CPU sparse approach explicitly constructs the Liouvillian superoperator ``\mathcal{L}`` as a ``D^2 \times D^2`` sparse matrix (where ``D = d^M``), then computes ``\mathcal{L} \cdot \text{vec}(\rho)`` via SpMV. Memory scales as ``O(M \cdot D^2)`` non-zeros.

cuDensityMat decomposes ``\mathcal{L}[\rho]`` as a sum of tensor network contractions over elementary operators. Each contraction involves small tensors (``d \times d`` per mode) and the density matrix, never forming the full superoperator. Computational cost scales more favorably with the number of modes ``M``.

## Validation

CPU and GPU implementations produce **bitwise-identical results** for the static Liouvillian (verified at ``t=0``). All frameworks agree on final populations to within adaptive solver tolerances.

## Environment Details

### Cross-framework benchmarks (T4)
- **GPU**: NVIDIA Tesla T4 (16 GB, 300 GB/s bandwidth, 0.25 TFLOPS FP64)
- **CPU**: 8 vCPUs (GCE n1-standard-8, 30 GB RAM)
- **Julia**: 1.12.5
- **Python**: 3.10.12 (QuTiP, CuPy), 3.11.15 (JAX ext)
- **QuTiP**: 5.2.3
- **JAX**: 0.6.2
- **cuquantum-python**: 25.3.0 (CuPy), 25.11.1 (JAX ext)
- **cuquantum-python-jax**: 0.0.3 (experimental, built from source)
- **CUDA**: 12.8, Driver 570.211.01

### A100 benchmarks
- **GPU**: NVIDIA A100-SXM4-40GB (2 TB/s bandwidth, 9.7 TFLOPS FP64)
- **CPU**: 12 vCPUs (GCE a2-highgpu-1g, 85 GB RAM)

### Reproduction

Benchmark scripts are in [`benchmark/comparison/`](https://github.com/harmoniqs/CuQuantum.jl/tree/main/benchmark/comparison). See the [README](https://github.com/harmoniqs/CuQuantum.jl/blob/main/benchmark/comparison/README.md) for setup instructions.
