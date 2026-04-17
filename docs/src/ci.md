# CI and Benchmarks

CuQuantum.jl uses GitHub Actions for continuous integration. Tests run on
both CPU-only GitHub-hosted runners and on self-hosted GPU runners.

## Workflows

### CI (`CI.yml`)

Triggers on every push to `main` and on pull requests.

**CPU matrix** -- runs on `ubuntu-latest` across Julia 1.10, 1.11, and 1.12.
Tests that require a GPU are automatically skipped via the `@gpu_test` macro
(see `test/setup.jl`). This validates module loading, the test infrastructure
itself, and any non-GPU logic.

**GPU tests** -- the full test suite (`test/runtests.jl`) runs on a
self-hosted GPU runner so that every `@gpu_test` block actually executes.
GPU tests are skipped for pull requests from forks, since self-hosted
runners should not execute untrusted code.

### GPU Benchmark (`gpu-benchmark.yml`)

Triggered manually via `workflow_dispatch` from the Actions tab.

**Inputs:**

| Input | Description |
|-------|-------------|
| `instance-type` | EC2 instance type: `g4dn.xlarge` (T4), `g5.xlarge` (A10G), `p4d.24xlarge` (8x A100), `p5.48xlarge` (8x H100) |
| `benchmark-script` | Julia script to run (default: `benchmark/run_benchmarks.jl`) |

Results are uploaded as a workflow artifact named
`benchmark-results-<instance-type>-<run-number>`.

The default benchmark (`benchmark/run_benchmarks.jl`) compares GPU
(cuDensityMat) vs CPU (dense and sparse) performance for Liouvillian action
$L[\rho]$ on a dual-rail cavity system at increasing numbers of cavities.
Larger systems (M=8+) can only be evaluated on GPU -- CPU methods exceed
memory limits.

## Running locally

```sh
# Run the full test suite (GPU tests skip automatically without a GPU)
julia --project -e 'using Pkg; Pkg.test()'

# Run benchmarks (requires a CUDA-capable GPU)
julia --project=benchmark benchmark/run_benchmarks.jl
```
