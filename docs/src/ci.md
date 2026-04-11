# CI and Benchmarks

CuQuantum.jl uses GitHub Actions for continuous integration. Tests run on
both CPU-only runners (GitHub-hosted) and GPU runners (on-demand EC2
instances via [`machulav/ec2-github-runner`](https://github.com/machulav/ec2-github-runner)).

## Workflows

### CI (`CI.yml`)

Triggers on every push to `main` and on pull requests.

**CPU matrix** -- runs on `ubuntu-latest` across Julia 1.10, 1.11, and 1.12.
Tests that require a GPU are automatically skipped via the `@gpu_test` macro
(see `test/setup.jl`). This validates module loading, the test infrastructure
itself, and any non-GPU logic.

**GPU tests** -- a three-job pattern that spins up a `g4dn.xlarge` (NVIDIA T4)
EC2 instance on demand:

1. **Start GPU Runner** -- authenticates with AWS via OIDC, launches the
   instance using a multi-AZ configuration for capacity resilience, and
   installs Julia via `juliaup`.
2. **GPU Tests** -- runs the full test suite (`test/runtests.jl`) on the EC2
   instance. All `@gpu_test` blocks execute here since `CUDA.functional()`
   returns `true`.
3. **Stop GPU Runner** -- terminates the EC2 instance. This job runs with
   `if: always()` so the instance is cleaned up even when tests fail.

For pull requests from forks, GPU tests are skipped (security constraint --
self-hosted runners should not execute untrusted code).

### GPU Benchmark (`gpu-benchmark.yml`)

Triggered manually via `workflow_dispatch` from the Actions tab.

**Inputs:**

| Input | Description |
|-------|-------------|
| `instance-type` | EC2 instance type: `g4dn.xlarge` (T4), `g5.xlarge` (A10G), `p4d.24xlarge` (8x A100), `p5.48xlarge` (8x H100) |
| `benchmark-script` | Julia script to run (default: `benchmark/run_benchmarks.jl`) |

Uses the same three-job start/work/stop pattern as CI. Results are uploaded
as a workflow artifact named `benchmark-results-<instance-type>-<run-number>`.

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

## Infrastructure

The EC2 runner infrastructure is managed via Terraform in the `aws-infra`
repo (`terraform/modules/gpu_runner/`). It provisions:

- An OIDC role for GitHub Actions (no stored AWS credentials)
- A security group (egress-only)
- An EC2 instance profile with SSM access for debugging
- The Deep Learning Base AMI (Ubuntu 22.04, pre-installed NVIDIA drivers)

## Required secrets

| Secret | Purpose |
|--------|---------|
| `GH_RUNNER_PAT` | Classic PAT with `repo` scope (required by ec2-github-runner) |
| `AWS_GPU_RUNNER_ROLE_ARN` | IAM role ARN for OIDC authentication |
| `AWS_GPU_RUNNER_AZ_CONFIG` | JSON array of availability zone configs (imageId, subnetId, securityGroupId per AZ) |
| `AWS_GPU_RUNNER_INSTANCE_PROFILE` | EC2 instance profile name attached to launched instances |

## Known constraints

- **vCPU quota**: The AWS account has a limited on-demand G-family vCPU
  quota. A `g4dn.xlarge` uses 4 vCPUs, so only one GPU runner can be active
  at a time. Concurrent CI and benchmark runs will fail with capacity errors.
- **No spot fallback**: Instances launch as on-demand. If an AZ lacks
  capacity, the action tries the next AZ in the config before failing.
- **Instance cleanup**: If a start job succeeds but the stop job fails to
  run (e.g., workflow cancelled), check the AWS console for orphaned
  instances tagged `gpu-runner=true`.
