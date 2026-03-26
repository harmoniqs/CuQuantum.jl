# Benchmarks

## Liouvillian Action: cuDensityMat GPU vs CPU Sparse

We benchmark a single evaluation of the Lindblad Liouvillian action ``L[\rho]`` for a dual-rail qubit system with ``M`` coupled cavities (Fock truncation ``d=3``):

```math
\dot{\rho} = -i[H, \rho] + \gamma \sum_m \left( a_m \rho a_m^\dagger - \frac{1}{2}\{a_m^\dagger a_m, \rho\} \right)
```

### Results: NVIDIA A100 40GB vs 12-core CPU

| Cavities (M) | Hilbert dim | ``\rho`` elements | cuDensityMat (A100) | CPU Sparse | GPU Speedup |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 2 | 9 | 81 | 0.27 ms | 0.001 ms | CPU wins |
| 4 | 81 | 6,561 | 1.2 ms | 0.14 ms | CPU wins |
| **6** | **729** | **531K** | **8.1 ms** | **40 ms** | **4.9x** |
| 8 | 6,561 | 43M | 621 ms | N/A | GPU only option |
| 9 | 19,683 | 387M | 6,742 ms | N/A | GPU only option |

### Key Findings

- **Crossover at M=6** (729-dimensional Hilbert space): cuDensityMat on A100 is 4.9x faster than CPU sparse matrix-vector multiply.
- **Beyond M=6, CPU is infeasible**: the sparse superoperator for M=8 would require >100 GB to construct. cuDensityMat never materializes the superoperator — it uses tensor network contraction.
- **GPU overhead dominates at small sizes**: for M=2,4, kernel launch overhead (~0.3 ms) exceeds the actual computation time. CPU wins trivially.

### Why cuDensityMat Scales Better

The CPU sparse approach explicitly constructs the Liouvillian superoperator ``\mathcal{L}`` as a ``D^2 \times D^2`` sparse matrix (where ``D = d^M``), then computes ``\mathcal{L} \cdot \text{vec}(\rho)`` via SpMV. The memory for the sparse matrix scales as ``O(M \cdot D^2)`` non-zeros.

cuDensityMat instead decomposes ``\mathcal{L}[\rho]`` as a sum of tensor network contractions over the elementary operators. Each contraction involves small tensors (``d \times d`` per mode) and the density matrix, never forming the full superoperator. The computational cost scales more favorably with the number of modes ``M``.

### Validation

The CPU and GPU implementations produce **bitwise-identical results** for the static Liouvillian (verified at ``t=0``). For time-dependent simulations with callbacks, diagonal populations agree to machine precision (``10^{-16}``), with off-diagonal coherences differing by ``< 10^{-3}`` after 50 RK4 steps due to floating-point operation ordering.

### Hardware Details

- **GPU**: NVIDIA A100-SXM4-40GB (2 TB/s bandwidth, 9.7 TFLOPS FP64)
- **CPU**: 12 vCPUs (GCE a2-highgpu-1g instance, 85 GB RAM)
- **Benchmark script**: [`benchmark/run_benchmarks.jl`](https://github.com/harmoniqs/CuQuantum.jl/blob/main/benchmark/run_benchmarks.jl)
