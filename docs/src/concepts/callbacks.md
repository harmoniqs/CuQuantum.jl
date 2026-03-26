# Time-Dependent Callbacks

## Overview

Many quantum systems have time-dependent Hamiltonians — control pulses, modulated couplings, or driven transitions. CuQuantum.jl supports this through **callback functions** that are called by the GPU library at each time step.

There are two types of callbacks:
- **Scalar callbacks** — time-dependent scalar coefficients multiplying an operator term
- **Tensor callbacks** — time-dependent tensor elements of an elementary operator

## Scalar Callbacks

A scalar callback provides a time-dependent complex coefficient ``c(t, \theta)``:

```julia
# Define a time-dependent detuning: δ(t) = Δ sin(ωt)
function detuning_callback(time, params, storage)
    Δ = 2π * 1.0  # detuning amplitude
    ω = 2π * 0.5  # modulation frequency
    for b in eachindex(storage)
        storage[b] = ComplexF64(Δ * sin(ω * time))
    end
end

# Wrap it for the C API
cb, gcb, refs = wrap_scalar_callback(detuning_callback)

# Use in an elementary product
append_elementary_product!(term, [elem_n], Int32[0], Int32[0];
    coefficient=ComplexF64(1.0),
    coefficient_callback=cb,
    coefficient_gradient_callback=gcb)
```

### Callback Signature

```julia
function my_scalar_callback(
    time::Float64,           # current simulation time
    params::Matrix{Float64}, # user parameters, shape (num_params, batch_size)
    storage::Vector{T}       # output: write coefficients here, length batch_size
)
```

The function must write the coefficient value(s) into `storage`. For batched simulations, `storage` has one element per batch member.

## Tensor Callbacks

A tensor callback provides time-dependent matrix elements for an elementary operator:

```julia
# Time-dependent rotation matrix: R(t) = [[cos(t), -sin(t)], [sin(t), cos(t)]]
function rotation_callback(time, params, storage)
    for b in axes(storage, 3)
        storage[1, 1, b] = complex(cos(time))
        storage[2, 1, b] = complex(sin(time))
        storage[1, 2, b] = complex(-sin(time))
        storage[2, 2, b] = complex(cos(time))
    end
end

tensor_cb, tensor_gcb, refs = wrap_tensor_callback(rotation_callback)

# Create elementary operator with time-dependent elements
elem = create_elementary_operator(ws, [2], static_data;
    tensor_callback=tensor_cb,
    tensor_gradient_callback=tensor_gcb)
```

## Gradient Callbacks

For backward differentiation (parameter gradients), provide a gradient function:

```julia
function my_gradient(time, params, scalar_grad, params_grad)
    # Compute ∂c/∂t and ∂c/∂θ
    for b in axes(scalar_grad, 1)
        scalar_grad[b] = ...  # gradient of coefficient w.r.t. time
        params_grad[1, b] = ...  # gradient w.r.t. first parameter
    end
end

cb, gcb, refs = wrap_scalar_callback(my_coeff; gradient=my_gradient)
```

## GC Safety

Callbacks are registered in a global registry to prevent garbage collection while the C library holds raw pointers. Always call `unregister_callback!(refs)` after destroying the operator that uses the callback:

```julia
cb, gcb, refs = wrap_scalar_callback(my_func)
# ... use cb in operators ...
# ... destroy operators ...
unregister_callback!(refs)
```
