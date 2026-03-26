# Getting Started

## Installation

CuQuantum.jl requires a CUDA-capable GPU and the NVIDIA cuQuantum SDK (provided automatically via `cuQuantum_jll`).

```julia
using Pkg
Pkg.add(url="https://github.com/harmoniqs/CuQuantum.jl")
```

!!! note "GPU Required"
    CuQuantum.jl requires a CUDA-capable GPU at runtime. The package will load without a GPU, but all computation functions require one. Supported architectures: Turing (T4), Ampere (A100), Ada (L4/L40), Hopper (H100), Blackwell (B200).

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

1. **Create a `WorkStream`** — manages the GPU handle, workspace memory, and CUDA stream
2. **Define elementary operators** — single-mode operators (e.g., annihilation, number, Pauli matrices) uploaded to GPU
3. **Build operator terms** — tensor products of elementary operators acting on specific modes
4. **Assemble the composite operator** — sum of terms with coefficients and duality flags (left/right of ``\rho``)
5. **Prepare the action** — one-time workspace allocation and contraction planning
6. **Compute** — evaluate ``L[\rho]`` at a given time, as many times as needed
7. **Clean up** — close the `WorkStream`

See the [Concepts](@ref) section for detailed explanations of each step.
