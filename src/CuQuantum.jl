"""
    CuQuantum

Julia wrapper package for NVIDIA's cuQuantum SDK.

Currently provides:
- `CuDensityMat` — density matrix simulation (Lindblad dynamics, time-dependent operators, gradients)

Planned:
- `CuStateVec` — state vector simulation
- `CuTensorNet` — tensor network contraction
- `CuPauliProp` — Pauli propagation
- `CuStabilizer` — stabilizer simulation
"""
module CuQuantum

using CUDA

# CuDensityMat submodule
include("densitymat/CuDensityMat.jl")

export CuDensityMat

end # module CuQuantum
