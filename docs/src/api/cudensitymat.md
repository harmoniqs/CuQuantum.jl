# CuDensityMat API Reference

## Module

```@docs
CuQuantum.CuDensityMat
```

## WorkStream

```@docs
CuDensityMat.WorkStream
```

## States

```@docs
CuDensityMat.DensePureState
CuDensityMat.DenseMixedState
```

## Elementary Operators

```@docs
CuDensityMat.ElementaryOperator
CuDensityMat.create_elementary_operator
CuDensityMat.destroy_elementary_operator
```

## Matrix Operators

```@docs
CuDensityMat.MatrixOperator
CuDensityMat.create_matrix_operator
CuDensityMat.destroy_matrix_operator
```

## Operator Terms

```@docs
CuDensityMat.OperatorTerm
CuDensityMat.create_operator_term
CuDensityMat.destroy_operator_term
CuDensityMat.append_elementary_product!
CuDensityMat.append_matrix_product!
```

## Operators

```@docs
CuDensityMat.Operator
CuDensityMat.create_operator
CuDensityMat.destroy_operator
CuDensityMat.append_term!
```

## Operator Action

```@docs
CuDensityMat.OperatorAction
CuDensityMat.create_operator_action
CuDensityMat.destroy_operator_action
CuDensityMat.prepare_operator_action!
CuDensityMat.compute_operator_action!
```

## Backward Differentiation

```@docs
CuDensityMat.prepare_operator_action_backward!
CuDensityMat.compute_operator_action_backward!
```

## Expectation Values

```@docs
CuDensityMat.Expectation
CuDensityMat.create_expectation
CuDensityMat.destroy_expectation
CuDensityMat.prepare_expectation!
CuDensityMat.compute_expectation!
```

## Eigenspectrum

```@docs
CuDensityMat.OperatorSpectrum
CuDensityMat.create_operator_spectrum
CuDensityMat.destroy_operator_spectrum
CuDensityMat.configure_spectrum!
CuDensityMat.prepare_spectrum!
CuDensityMat.compute_spectrum!
```

## Callbacks

```@docs
CuDensityMat.wrap_scalar_callback
CuDensityMat.wrap_tensor_callback
CuDensityMat.unregister_callback!
```

## State Operations

```@docs
CuDensityMat.allocate_storage!
CuDensityMat.initialize_zero!
CuDensityMat.inplace_scale!
CuDensityMat.inplace_accumulate!
CuDensityMat.trace
CuDensityMat.norm
CuDensityMat.inner_product
```
