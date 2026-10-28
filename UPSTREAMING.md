# Upstreaming cuDensityMat into CUDA.jl

Notes for porting this package's cuDensityMat bindings into CUDA.jl as
`lib/cudensitymat` (JuliaGPU/CUDA.jl#2868, "Wrap cuDensityMat", where the
maintainers have invited us to take it over). Nothing here has been proposed
upstream yet; this is the plan.

## What CUDA.jl expects (as of CUDA.jl 6.5)

CUDA.jl 6 is a monorepo workspace. Each NVIDIA library is its own registered
package under `lib/`, versioned in lockstep with CUDA.jl and depending only on
`CUDACore`. The two existing cuQuantum wrappers are the template:
`lib/custatevec` (package `cuStateVec`) and `lib/cutensornet` (`cuTensorNet`).

```
lib/cudensitymat/
  Project.toml          name = "cuDensityMat", version = "6.5.x" (= CUDA.jl's)
  LICENSE.md            CUDA.jl's MIT license
  README.md
  src/
    cuDensityMat.jl     module, handle cache, library discovery, __init__
    libcudensitymat.jl  generated (do not edit)
    error.jl            CUDENSITYMATError, throw_api_error, check
    types.jl            Julia-side enums/helpers not in the header
    wrappers.jl         thin idiomatic wrappers
    ...                 high-level API files
  test/
    Project.toml        CUDACore, CUDA_Runtime_jll, Test, ParallelTestRunner = "2", cuDensityMat
    runtests.jl         ParallelTestRunner: find_tests(@__DIR__), init_code includes setup.jl
    setup.jl
    *.jl                one file per area
res/wrap/
  cudensitymat.toml             generator options (this repo: gen/cudensitymat.toml)
  libcudensitymat_prologue.jl   (this repo: gen/libcudensitymat_prologue.jl)
  wrap.jl                       add a `cudensitymat` target next to `custatevec`
```

and in CUDA.jl's root `Project.toml`, `"lib/cudensitymat"` in `[workspace].projects`,
plus the `Pkg.develop` lists in `.buildkite/pipeline.yml` (Julia 1.10 doesn't read
`[sources]`).

`lib/custatevec/Project.toml` pins, and `lib/cudensitymat` should copy:

```toml
[deps]
CEnum = "fa961155-64e5-5f13-b03f-caf6b980ea82"
CUDACore = "bd0ed864-bdfe-4181-a5ed-ce625a5fdea2"
CUDA_Runtime_Discovery = "1af6417a-86b4-443c-805f-a4643ffb695f"
cuQuantum_jll = "b75408ef-6fdf-5d74-b65a-7df000ad18e6"

[compat]
CEnum = "0.5, 1"
CUDACore = "=6.5.0"
CUDA_Runtime_Discovery = "2"
cuQuantum_jll = "26.9"
julia = "1.10"
```

## Mapping from this repo

| This repo | CUDA.jl |
|:---|:---|
| `gen/generate.jl` | the `cudensitymat` branch of `res/wrap/wrap.jl`. Our rewriter is a subset of theirs (`@gcsafe_ccall`, `initialize_context()`, `@checked`, `[api.<fn>.argtypes]`), so the TOML carries over as is. Theirs parses against `CUDA_SDK_jll` headers; ours defaults to the stand-ins in `gen/cuda_stub/` (`CUDA_INCLUDE_DIR` switches to real headers). Drop the stubs upstream. |
| `gen/cudensitymat.toml` | `res/wrap/cudensitymat.toml`. Change `output_file_path` to `../../lib/cudensitymat/src/libcudensitymat.jl`, and add `needs_context = false` entries for the logger functions once NVIDIA un-comments them in the header (they are commented out in 0.7.0). |
| `src/densitymat/CuDensityMat.jl` | `lib/cudensitymat/src/cuDensityMat.jl`. It already depends on `CUDACore` (not the `CUDA` umbrella); tighten the compat to `=6.5.x`, and move to their handle pattern: `HandleCache` + a `TaskLocalCache` of `(handle, stream)` with `resource_finalizer` (see `cuStateVec.handle()`). Our `WorkStream` owns its handle; upstream would rather have `handle()` follow the task's stream. |
| `src/densitymat/error.jl` `check` | wrap the call in `CUDACore.capture_submission(...)` like `cuStateVec`, and map `ALLOC_FAILED` to `OutOfGPUMemoryError`. |
| `@gpu_test` / `HAS_GPU` in `test/setup.jl` | not needed: CUDA.jl's CI only runs on supported GPUs. But cuDensityMat needs compute capability ≥ 7.5, which is stricter than CUDA.jl's minimum, so `runtests.jl` should skip the suite (with a message) on older devices instead of failing. |
| `MIN_LIBRARY_VERSION`, version check in `__init__` | keep: `functional()` should be false for a library older than the ABI the bindings were generated from (relevant with `local_toolkit`, where a system cuQuantum can be older than the JLL). |
| GPU callbacks (`callbacks.jl`) | depends on `unsafe_wrap(CuStream, handle)` for non-owned streams (CUDA.jl#3449, in CUDACore 6.5). Keep the CPU callbacks as is. |
| benchmarks, MPI tests | stay here (benchmarks need hardware CUDA.jl CI doesn't schedule for library packages; MPI isn't part of CUDA.jl's test setup). |

## Version-pin coordination

- cuQuantum_jll ships cuStateVec, cuTensorNet and cuDensityMat in one artifact,
  so all three wrappers must agree on its version. CUDA.jl 6.5 pins `26.9`; this
  package now does too. Before upstreaming, and whenever CUDA.jl bumps
  cuQuantum_jll, regenerate the bindings (`julia --project=gen gen/generate.jl`)
  and run the binding tests: `test/densitymat/test_bindings.jl` checks the wrapped
  functions against both the header and the exported symbols, so a changed header
  fails on CPU before any GPU job runs.
- cuDensityMat is pre-1.0 and has broken its ABI in a minor release
  (0.4 → 0.7: `cudensitymatCreateOperatorAction` gained `scopeKind` and
  `approachKind`). The `[api.<fn>.argtypes]` overrides are indexed by argument
  position, so check them on every bump (`generate.jl` only catches renamed or
  removed functions).
- Once upstream, the lib package version is CUDA.jl's (`=6.5.x` on CUDACore), so
  a cuQuantum bump happens in a CUDA.jl release, for all three wrappers together.
- This package then becomes the high-level layer (WorkStream, states, operators,
  callbacks) on top of `cuDensityMat`, or is retired if that layer moves too.
  Deprecate here first; don't break existing users in the same release.

## Licensing

- Our code is MIT, like CUDA.jl, so it can be contributed as is.
- `libcudensitymat.jl` is generated from NVIDIA's header, the same as the
  existing wrappers. CUDA.jl carries those files under its MIT license, and the
  binaries come from `cuQuantum_jll` under NVIDIA's cuQuantum SLA (which allows
  redistribution but not modification). Nothing NVIDIA wrote is vendored here;
  the stand-in headers in `gen/cuda_stub/` are ours and only declare type names.

## Open items before opening the PR

1. Run the GPU tests here on a CC ≥ 7.5 device against 26.9 (none of the
   cuDensityMat GPU tests have run against 0.7 yet).
2. Decide what upstream gets: just the generated layer plus a thin `wrappers.jl`
   (what `cuStateVec` started with), or the high-level API too.
3. Agree on the package name (`cuDensityMat`, following `cuStateVec` /
   `cuTensorNet`) and ask kshyatt for review on #2868.
