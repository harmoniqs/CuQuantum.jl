# Concepts Overview

CuQuantum.jl wraps the [NVIDIA cuDensityMat](https://docs.nvidia.com/cuda/cuquantum/latest/cudensitymat/index.html) library for GPU-accelerated density matrix simulation.

## Architecture

cuDensityMat represents quantum operators as **sums of tensor products** of elementary operators. Instead of materializing the full Liouvillian superoperator (which scales as ``d^{2M} \times d^{2M}``), it uses tensor network contraction to compute ``L[\rho]`` directly.

```
Elementary Operators  →  Operator Terms  →  Operators  →  Operator Action
    (single-mode)       (tensor products)   (sums of terms)   (L[ρ] computation)
```

See [Operators](@ref "Operators") for details on each level of the hierarchy.

## Hilbert Space

The quantum system is a tensor product of ``M`` modes:

```math
\mathcal{H} = \bigotimes_{m=1}^{M} \mathbb{C}^{d_m}
```

For example, ``M=4`` cavities each truncated to ``d=3`` Fock states gives dimension ``3^4 = 81``. Modes are indexed from 0 in the C API (CuQuantum.jl preserves this convention in the low-level wrappers).

## Density Matrix

Density matrices ``\rho`` are stored as dense tensors on the GPU. CuQuantum.jl provides two state types:

- **`DensePureState`**: ``|\psi\rangle`` with shape ``(d_1, d_2, \ldots, d_M)``
- **`DenseMixedState`**: ``\rho`` with shape ``(d_1, \ldots, d_M, d_1, \ldots, d_M)`` — first ``M`` indices are ket, last ``M`` are bra

See [States](@ref "Quantum States") for operations on states.

## Lindblad Master Equation

The primary use case is the Lindblad master equation:

```math
\dot{\rho} = -i[H(t), \rho] + \sum_k \gamma_k \left( L_k \rho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \rho\} \right)
```

This decomposes into operator terms with specific duality flags:

| Physical term | Duality | Coefficient |
|:---|:---:|:---:|
| ``-iH\rho`` | 0 (ket) | ``-i`` |
| ``+i\rho H`` | 1 (bra) | ``+i`` |
| ``\gamma L\rho L^\dagger`` | mixed (0,1) | ``\gamma`` |
| ``-\frac{\gamma}{2} L^\dagger L \rho`` | 0 (ket) | ``-\gamma/2`` |
| ``-\frac{\gamma}{2} \rho L^\dagger L`` | 1 (bra) | ``-\gamma/2`` |

The sandwich term ``L\rho L^\dagger`` uses a fused operator with `mode_action_duality = [0, 1]` — mode 0 on the ket side, mode 1 on the bra side. See [Operators](@ref "Operators") for how to construct this.

## WorkStream

The [`WorkStream`](@ref) is the central resource manager:

- **Library handle** — cuDensityMat GPU context
- **Workspace** — scratch memory for tensor contractions
- **CUDA stream** — for asynchronous GPU execution
- **MPI/NCCL communicator** — optional, for distributed computation

All operator and state objects are created within a `WorkStream` and released when it is closed.

## Backward Differentiation

CuQuantum.jl supports parameter gradients via the cuDensityMat backward differentiation API. This computes the vector-Jacobian product (VJP):

```math
\frac{\partial c}{\partial \theta_n} \mathrel{+}= \text{Re}\!\left(\frac{\partial c}{\partial Q_j} \cdot \frac{\partial Q_j}{\partial \theta_n}\right)
```

where ``Q_j`` are the differentiable quantities (coefficients, operator elements) and ``\theta_n`` are real parameters. The library computes ``\partial c / \partial Q_j`` via tensor contraction and handles the Wirtinger derivative convention. The user's gradient callback provides ``\partial Q_j / \partial \theta_n`` and accumulates into the parameter gradient vector.

See [Callbacks](@ref "Time-Dependent Callbacks") for how to set up gradient callbacks.
