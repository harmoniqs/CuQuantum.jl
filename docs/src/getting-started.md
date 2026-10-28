# Getting Started

## Installation

CuQuantum.jl requires a CUDA-capable GPU and the NVIDIA cuQuantum SDK (provided automatically via `cuQuantum_jll`).

```julia
using Pkg
Pkg.add(url="https://github.com/harmoniqs/CuQuantum.jl")
```

!!! note "GPU Required"
    CuQuantum.jl requires a CUDA-capable GPU of compute capability 7.5 or newer at runtime: Turing (T4), Ampere (A100), Ada (L4/L40), Hopper (H100), Blackwell (B200). The package loads without a GPU, and `CuDensityMat.version()` works wherever the library can be loaded, but all computation functions need a supported GPU.

## Versions

CuQuantum.jl targets cuDensityMat 0.7 (cuQuantum 26.9, via `cuQuantum_jll` 26.9) and
CUDA.jl 6.5, in lockstep with the `cuStateVec`/`cuTensorNet` packages that ship with
CUDA.jl 6.5. The low-level bindings are generated from the cuDensityMat header with
`gen/generate.jl`; `CuDensityMat.MIN_LIBRARY_VERSION` is the oldest library whose ABI
they match.

## Dependencies

CuQuantum.jl depends on:
- [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) — Julia GPU computing
- [cuQuantum_jll](https://github.com/JuliaBinaryWrappers/cuQuantum_jll.jl) — prebuilt cuQuantum SDK binaries
- [CEnum.jl](https://github.com/JuliaInterop/CEnum.jl) — C enum support

All dependencies are installed automatically.

## Verify Installation

```julia
using CuQuantum
using CuQuantum.CuDensityMat

# Check the cuDensityMat library version
v = CuDensityMat.version()
println("cuDensityMat version: $v")

# Create and close a workspace (verifies GPU + library connectivity)
ws = WorkStream()
close(ws)
println("CuQuantum.jl is working!")
```

## Basic Workflow

Every CuQuantum.jl simulation follows this pattern:

1. **Create a [`WorkStream`](@ref)** — manages the GPU handle, workspace memory, and CUDA stream
2. **Define elementary operators** — single-mode operators (annihilation, number, Pauli) uploaded to GPU
3. **Build operator terms** — tensor products of elementary operators acting on specific modes
4. **Assemble the composite operator** — sum of terms with coefficients and duality flags
5. **Prepare the action** — one-time workspace allocation and contraction planning
6. **Compute** — evaluate ``L[\rho]`` at a given time, as many times as needed
7. **Clean up** — `close(ws)`

See the [Concepts](@ref "Concepts Overview") section for details on each step.
