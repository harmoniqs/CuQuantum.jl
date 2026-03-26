# Concepts Overview

CuQuantum.jl wraps the [NVIDIA cuDensityMat](https://docs.nvidia.com/cuda/cuquantum/latest/cudensitymat/index.html) library for GPU-accelerated density matrix simulation. This section explains the key concepts.

## Architecture

The cuDensityMat library represents quantum operators as **sums of tensor products** of elementary operators. Instead of materializing the full Liouvillian superoperator as a matrix (which scales as ``d^{2M} \times d^{2M}``), it uses tensor network contraction to compute the action ``L[\rho]`` directly. This allows simulation of systems far too large for explicit matrix approaches.

```
Elementary Operators  →  Operator Terms  →  Operators  →  Operator Action
    (single-mode)       (tensor products)   (sums of terms)   (L[ρ] computation)
```

## Hilbert Space

The quantum system is defined as a tensor product of ``M`` modes, each with dimension ``d_m``:

```math
\mathcal{H} = \bigotimes_{m=1}^{M} \mathbb{C}^{d_m}
```

For example, ``M=4`` cavities each truncated to ``d=3`` Fock states gives a Hilbert space of dimension ``3^4 = 81``.

## Density Matrix

Density matrices ``\rho`` are stored as dense tensors on the GPU:

- **`DensePureState`**: pure state ``|\psi\rangle`` with shape ``(d_1, d_2, \ldots, d_M)``
- **`DenseMixedState`**: mixed state ``\rho`` with shape ``(d_1, d_2, \ldots, d_M, d_1, d_2, \ldots, d_M)``

Both support batched operations (multiple states in parallel).

## Lindblad Master Equation

The primary use case is the Lindblad master equation:

```math
\dot{\rho} = -i[H(t), \rho] + \sum_k \gamma_k \left( L_k \rho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \rho\} \right)
```

This is decomposed into operator terms and assembled with appropriate duality flags:

| Physical term | Duality | Coefficient |
|--------------|---------|-------------|
| ``-iH\rho`` | 0 (ket/left) | ``-i`` |
| ``+i\rho H`` | 1 (bra/right) | ``+i`` |
| ``\gamma L\rho L^\dagger`` | 0 (sandwich via fused op) | ``\gamma`` |
| ``-\frac{\gamma}{2} L^\dagger L \rho`` | 0 (left) | ``-\gamma/2`` |
| ``-\frac{\gamma}{2} \rho L^\dagger L`` | 1 (right) | ``-\gamma/2`` |

## WorkStream

The `WorkStream` is the central resource manager. It holds:
- A cuDensityMat library **handle** (GPU context)
- A **workspace descriptor** (scratch memory for contractions)
- A **CUDA stream** for asynchronous execution
- An optional **MPI/NCCL communicator** for distributed computation
