# Time-Dependent Callbacks

## Overview

Many quantum systems have time-dependent Hamiltonians — control pulses, modulated couplings, driven transitions. CuQuantum.jl supports this through **callback functions** invoked by the cuDensityMat library at each time step.

There are two types:

- **Scalar callbacks** — time-dependent scalar coefficients multiplying an operator term
- **Tensor callbacks** — time-dependent matrix elements of an elementary operator

## Scalar Callbacks

A scalar callback provides a complex coefficient ``c(t, \theta)`` evaluated at each time step:

```julia
function detuning(time, params, storage)
    Δ = 2π * 1.0   # detuning amplitude
    ω = 2π * 0.5   # modulation frequency
    for b in eachindex(storage)
        storage[b] = ComplexF64(Δ * sin(ω * time))
    end
end

cb, gcb, refs = wrap_scalar_callback(detuning)

append_elementary_product!(term, [elem_n], Int32[0], Int32[0];
    coefficient=ComplexF64(1.0),
    coefficient_callback=cb,
    coefficient_gradient_callback=gcb)
```

### Callback Signature

```julia
function my_callback(
    time::Float64,           # current simulation time
    params::Matrix{Float64}, # parameters, shape (num_params, batch_size)
    storage::Vector{T}       # output: write coefficient here, length batch_size
)
```

Write the coefficient value into `storage`. For batched simulations, `storage` has one element per batch member. The `params` matrix contains user-defined real parameters passed via `compute_operator_action!`.

## Tensor Callbacks

A tensor callback provides time-dependent matrix elements for an elementary operator:

```julia
function rotation(time, params, storage)
    for b in axes(storage, 3)
        storage[1, 1, b] = complex(cos(time))
        storage[2, 1, b] = complex(sin(time))
        storage[1, 2, b] = complex(-sin(time))
        storage[2, 2, b] = complex(cos(time))
    end
end

tensor_cb, tensor_gcb, refs = wrap_tensor_callback(rotation)
```

## Gradient Callbacks

For backward differentiation, provide a gradient function alongside the coefficient callback. The gradient callback implements the chain rule for parameter gradients:

```math
\frac{\partial c}{\partial \theta_n} \mathrel{+}= \text{Re}\!\left(\text{adjoint} \cdot \frac{\partial f}{\partial \theta_n}\right)
```

The library computes the **adjoint** of the coefficient (``\partial c / \partial f``, passed as `scalar_grad`) via tensor contraction. Your gradient callback multiplies by ``\partial f / \partial \theta_n`` and accumulates into `params_grad`:

```julia
# Forward: f(t, Ω) = exp(-Ω*t)
function my_coeff(time, params, storage)
    for b in eachindex(storage)
        Ω = params[1, b]
        storage[b] = complex(exp(-Ω * time), 0.0)
    end
end

# Backward: df/dΩ = -t * exp(-Ω*t)
function my_gradient(time, params, scalar_grad, params_grad)
    for b in axes(scalar_grad, 1)
        Ω = params[1, b]
        dfdΩ = complex(-time * exp(-Ω * time), 0.0)
        params_grad[1, b] += real(scalar_grad[b] * dfdΩ)
    end
end

cb, gcb, refs = wrap_scalar_callback(my_coeff; gradient=my_gradient)
```

### Gradient Callback Signature

```julia
function my_gradient(
    time::Float64,                # current simulation time
    params::Matrix{Float64},      # parameters, shape (num_params, batch_size)
    scalar_grad::Vector{T},       # INPUT: adjoint ∂c/∂f (from library), length batch_size
    params_grad::Matrix{Float64}  # OUTPUT: accumulate ∂c/∂θ here, shape (num_params, batch_size)
)
```

!!! warning "Accumulation semantics"
    `params_grad` uses **+=** semantics — always add to it, never overwrite. The library may call the gradient callback multiple times for different terms that share parameters. Accumulate ``\text{Re}(\text{scalar\_grad} \cdot \partial f / \partial \theta)``, not ``2\,\text{Re}(\cdots)``: `state_out_adj` is seeded with ``\partial c/\partial \operatorname{Re}\rho - i\,\partial c/\partial \operatorname{Im}\rho = 2\,\partial c/\partial\rho`` (the elementwise conjugate of a ChainRules cotangent; the local cost is ``\operatorname{Re}\sum_{ij} \text{adj}_{ij}\,\rho_{ij}``, seed not conjugated), which is twice the Wirtinger seed ``\partial c/\partial \rho`` that the cuDensityMat header's ``2\,\text{Re}`` assumes. `scalar_grad` is the holomorphic adjoint, so for a complex Jacobian (e.g. ``f = e^{i\Omega t}``) multiply by ``\partial f/\partial\theta`` itself, not its conjugate. A coefficient callback on a term added with `append_term!` is evaluated once per call (batch size 1, `params[:, 1]`) and shared by all batch members; use `append_term_batch!` for per-member coefficients. See `compute_operator_action_backward!` for the full convention; `test/densitymat/test_gradients.jl` pins it against finite differences.

## GPU Callbacks

By default callbacks run on the host (`CUDENSITYMAT_CALLBACK_DEVICE_CPU`): the library
copies parameters and coefficient/tensor storage to host memory around each call. Pass
`device=:gpu` to register a GPU callback (`CUDENSITYMAT_CALLBACK_DEVICE_GPU`) instead.
A GPU callback is still invoked on the host, but receives **device** buffers as
`CuDeviceArray`s and the library's CUDA stream as a `CuStream`, and is expected to
enqueue CUDA.jl kernels on that stream and return without synchronizing:

```julia
function cis_kernel!(storage, params, t)
    i = threadIdx().x
    if i <= length(storage)
        @inbounds storage[i] = cis(params[1, i] * t)
    end
    return
end

function gpu_coeff(time, params, storage, stream)
    @cuda threads=length(storage) stream=stream cis_kernel!(storage, params, time)
    return nothing
end

cb, gcb, refs = wrap_scalar_callback(gpu_coeff; device=:gpu)
```

| | CPU callback | GPU callback |
|:---|:---|:---|
| scalar | `f(t, params::Matrix, storage::Vector)` | `f(t, params::CuDeviceMatrix, storage::CuDeviceVector, stream)` |
| tensor | `f(t, params, storage::Array)` | `f(t, params, storage::CuDeviceArray, stream)` |
| scalar gradient | `g(t, params, scalar_grad, params_grad)` | `g(t, params, scalar_grad, params_grad, stream)` |
| tensor gradient | `g(t, params, tensor_grad, params_grad)` | `g(t, params, tensor_grad, params_grad, stream)` |

The `device` keyword applies to both the callback and its `gradient`.

!!! note "Streams and memory"
    When the library calls back on the task's stream (the stream every high-level
    function passes), `stream` is CUDA.jl's own `CuStream`. Any other stream is wrapped
    with `unsafe_wrap(CuStream, handle)`, which CUDA.jl doesn't own: kernels taking
    `CuDeviceArray`s (as above) can be launched on it, but `CuArray` operations
    (broadcasts, copies) can't. To use `CuArray`s, order work between the borrowed
    stream and the task's stream with `CuEvent`s, as described in the docstring of
    `unsafe_wrap(CuStream, ...)`.

## GC Safety

Callbacks are registered in a global dictionary to prevent garbage collection while the cuDensityMat library holds function pointers. Always unregister after you're done:

```julia
cb, gcb, refs = wrap_scalar_callback(my_func; gradient=my_grad)
# ... use cb/gcb in operators ...
# ... done with operators ...
unregister_callback!(refs)
```
