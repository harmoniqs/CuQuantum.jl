# Clang.jl binding generator for cuDensityMat
#
# This script uses Clang.jl to auto-generate Julia bindings from the
# cuDensityMat C headers shipped with cuQuantum_jll.
#
# Usage:
#   julia --project=gen gen/generate.jl
#
# The generated output is written to gen/output/ for inspection.
# Hand-written bindings in src/densitymat/libcudensitymat.jl are authoritative;
# this script is for future re-generation when the C API changes.

using Clang
using Clang.Generators
using cuQuantum_jll

# Locate headers
const CUQUANTUM_INCLUDE = joinpath(cuQuantum_jll.artifact_dir, "include")
const CUDENSITYMAT_HEADER = joinpath(CUQUANTUM_INCLUDE, "cudensitymat.h")

if !isfile(CUDENSITYMAT_HEADER)
    error("cudensitymat.h not found at $CUDENSITYMAT_HEADER")
end

# Output directory
const OUTPUT_DIR = joinpath(@__DIR__, "output")
mkpath(OUTPUT_DIR)

# Generator options
const GENERATOR_TOML = joinpath(@__DIR__, "generator.toml")
options = if isfile(GENERATOR_TOML)
    load_options(GENERATOR_TOML)
else
    @warn "generator.toml not found, using defaults"
    Dict{String,Any}()
end

args = get_default_args()
push!(args, "-I$CUQUANTUM_INCLUDE")

# Find CUDA include path for cuda_runtime.h etc.
try
    using CUDA_Runtime_Discovery
    cuda_path = CUDA_Runtime_Discovery.cuda_toolkit_path()
    push!(args, "-I$(joinpath(cuda_path, "include"))")
catch
    @warn "Could not find CUDA toolkit include path; generation may fail"
end

ctx = create_context([CUDENSITYMAT_HEADER], args, options)

build!(ctx)

println("Generated bindings written to $OUTPUT_DIR")
