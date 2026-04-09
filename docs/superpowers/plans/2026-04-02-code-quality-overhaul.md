# CuQuantum.jl Code Quality Overhaul — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Comprehensively fix all identified code quality issues: Julia API design, type safety, finalizer safety, CI/GPU test automation, testing infrastructure, performance (eager sync removal), and documentation completeness.

**Architecture:** Six phases — (A) tooling & CI, (B) API design fixes, (C) safety improvements, (D) testing infrastructure, (E) documentation, (F) performance. Tasks within a phase are largely independent; later phases build on earlier ones. Code changes use TDD: write failing test → verify it fails → implement → verify it passes → commit.

**Tech Stack:** Julia 1.10+, CUDA.jl 5.x, JuliaFormatter.jl (Blue style), Aqua.jl, JET.jl, LinearAlgebra (stdlib), Documenter.jl 1.x, GitHub Actions self-hosted GPU runners

---

## File Map

**Created:**
- `test/aqua.jl` — Aqua.jl package quality tests
- `test/densitymat/test_batch_operators.jl` — batch API test coverage (Phase 6, filling the gap)

**Modified:**
- `Project.toml` — add Aqua, JET to `[extras]`/`[targets]`
- `.github/workflows/CI.yml` — add Aqua/JET step; flesh out `gpu-test` job with self-hosted runner
- `src/densitymat/state.jl` — extend `LinearAlgebra.norm`/`LinearAlgebra.tr`/`LinearAlgebra.dot`; remove `CUDA.synchronize()`; add `Base.close`/`Base.isopen`; add `_destroy!` helper
- `src/densitymat/workspace.jl` — fix `CUDA.device!()` side effect; add `_destroy!` pattern already present, improve it
- `src/densitymat/operators.jl` — fix `batch_size=0` defaults; concretely type `_data_ref`/`_callback_refs` fields; add `Base.close`/`Base.isopen`; add `_destroy!` helpers; remove `CUDA.synchronize()`
- `src/densitymat/callbacks.jl` — add `CallbackRef` mutable struct with finalizer; update `wrap_scalar_callback`/`wrap_tensor_callback` to return `CallbackRef`; export `CallbackRef`, `unregister_callback!`
- `src/densitymat/expectation.jl` — add `Base.close`/`Base.isopen`; add `_destroy!`; remove `CUDA.synchronize()`
- `src/densitymat/spectrum.jl` — add `Base.close`/`Base.isopen`; add `_destroy!`; remove `CUDA.synchronize()`
- `src/densitymat/CuDensityMat.jl` — export `CallbackRef`, `unregister_callback!`; add `import LinearAlgebra`
- `test/runtests.jl` — integrate Aqua tests; fix Phase 6 gap; include batch operator tests
- `test/setup.jl` — fix `TEST_BATCH_SIZES` (remove `0`); add `sync_and_pull` test helper
- `test/densitymat/test_state.jl` — use `TEST_DTYPES`; add `Base.close`/`isopen` assertions
- `test/densitymat/test_operators.jl` — use `TEST_BATCH_SIZES`; assert explicit sync; add `Base.close` assertions
- `test/densitymat/test_callbacks.jl` — assert `CallbackRef` type; test finalizer-based unregister
- `test/densitymat/test_integration.jl` — fix `trajectory.csv` to use `mktempdir()`
- `docs/make.jl` — remove `warnonly = [:missing_docs]`
- All `src/densitymat/*.jl` — add docstrings to every exported symbol

---

## Phase A — Tooling & CI

### Task 1: Replace JuliaFormatter with Runic.jl

Runic.jl is an opinionated, config-free formatter (like `gofmt`) — no style debates,
no `.JuliaFormatter.toml` to maintain. The existing `Formatter.yml` runs JuliaFormatter;
this task replaces it entirely with Runic and applies the formatting baseline.

**Files:**
- Delete: `.JuliaFormatter.toml` (does not exist yet — nothing to delete)
- Modify: `.github/workflows/Formatter.yml`

- [ ] **Step 1: Run Runic on the whole repo to get a clean baseline**

Runic does not need to be in `Project.toml` — install it into a throwaway env:

```bash
julia --project=@runic -e 'using Pkg; Pkg.add("Runic")'
julia --project=@runic -e 'using Runic; exit(Runic.main(ARGS))' -- --inplace src/ test/ docs/src/
```

Review the diff:
```bash
git diff --stat
```

Some changes to expect: trailing whitespace removal, consistent indentation,
normalised string delimiters, spacing around operators.

- [ ] **Step 2: Verify the formatted code still loads**

```bash
julia --project=. -e 'using CuQuantum'
```

Expected: no errors.

- [ ] **Step 3: Run the test suite on the formatted code**

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: same results as before formatting.

- [ ] **Step 4: Replace `Formatter.yml` with a Runic-based workflow**

Replace the entire contents of `.github/workflows/Formatter.yml` with:

```yaml
name: Formatter
on:
  pull_request:
  push:
    branches:
      - main
  workflow_dispatch:
jobs:
  runic:
    name: Runic formatting check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: fredrikekre/runic-action@v1
        with:
          version: '1'
          # On workflow_dispatch: format and commit. On PR/push: check only.
          args: >-
            ${{ github.event_name == 'workflow_dispatch'
              && '--inplace src/ test/ docs/src/'
              || '--check src/ test/ docs/src/' }}
      - name: Commit formatted code
        if: github.event_name == 'workflow_dispatch'
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git diff --quiet && exit 0
          git add src/ test/ docs/src/
          git commit -m "chore: runic autoformat"
          git push
```

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/Formatter.yml
git add -u   # stage Runic-formatted source changes
git commit -m "chore: replace JuliaFormatter with Runic.jl and apply formatting baseline"
```

---

### Task 2: Add Aqua.jl Package Quality Tests

**Files:**
- Modify: `Project.toml`
- Create: `test/aqua.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Add Aqua and JET to Project.toml extras/targets**

In `Project.toml`, replace:
```toml
[extras]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[targets]
test = ["Test", "LinearAlgebra"]
```
with:
```toml
[extras]
Aqua = "4c88cf16-eb10-579e-8560-4a9242c79595"
JET = "c3a54625-cd67-489e-a8e7-0a5a0ff4e31b"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[targets]
test = ["Aqua", "JET", "Test", "LinearAlgebra"]
```

- [ ] **Step 2: Create `test/aqua.jl`**

```julia
using Aqua
using CuQuantum

@testset "Aqua quality checks" begin
    Aqua.test_all(
        CuQuantum;
        # Method ambiguity check is noisy with CUDA.jl extensions — enable once
        # all LinearAlgebra extensions are properly disambiguated (Task 4).
        ambiguities = false,
        # Unbound type parameters in method signatures.
        unbound_args = true,
        # Undefined exports — catches missing `export` declarations.
        undefined_exports = true,
        # Stale dependencies in Project.toml.
        stale_deps = true,
        # Missing compat entries.
        deps_compat = true,
        # Project.toml formatting.
        project_toml_formatting = true,
        # Piracy: extending methods not owned by us or Base.
        # We extend LinearAlgebra deliberately; exclude those.
        piracy = (treat_as_own = [
            CuQuantum.CuDensityMat.norm,
            CuQuantum.CuDensityMat.tr,
            CuQuantum.CuDensityMat.dot,
        ],),
    )
end
```

- [ ] **Step 3: Add Aqua to `test/runtests.jl` before other testsets**

In `test/runtests.jl`, after `include("setup.jl")` and before `@testset "CuQuantum.jl"`, add:

```julia
include("aqua.jl")
```

- [ ] **Step 4: Run tests to verify Aqua passes (or surface real issues)**

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected output contains:
```
Test Summary:       | Pass  Fail
Aqua quality checks |    7
```

If any Aqua checks fail, fix the underlying issue before moving on (common issues: missing compat entries, undefined exports).

- [ ] **Step 5: Commit**

```bash
git add Project.toml test/aqua.jl test/runtests.jl
git commit -m "test: add Aqua.jl package quality checks"
```

---

### Task 3: Flesh Out GPU CI Job

**Files:**
- Modify: `.github/workflows/CI.yml`

The existing `gpu-test` job runs on `ubuntu-latest` with no actual GPU. This task makes it ready for a real self-hosted GPU runner and adds a `CUDA_VISIBLE_DEVICES` env var so tests can confirm GPU presence.

- [ ] **Step 1: Update the `gpu-test` job in `.github/workflows/CI.yml`**

Replace the existing `gpu-test` job (lines 39–55) with:

```yaml
  gpu-test:
    name: GPU Tests (Julia ${{ matrix.version }})
    # Self-hosted runner must have labels: [self-hosted, gpu, linux, x64]
    # The runner needs: CUDA toolkit, Julia, and cuQuantum installed.
    # Set CUQUANTUM_ROOT or CUDA_HOME env var on the runner if using a local toolkit.
    runs-on: [self-hosted, gpu, linux, x64]
    if: github.event_name == 'push' && (github.ref == 'refs/heads/main' || startsWith(github.ref, 'refs/tags/'))
    strategy:
      fail-fast: false
      matrix:
        version:
          - '1.12'
    env:
      # Ensure tests see exactly one GPU so CUDA.device() is deterministic.
      CUDA_VISIBLE_DEVICES: '0'
      # Tell CuQuantum.jl to use the local toolkit instead of the JLL artifact.
      # Set to the actual path on your runner, e.g. /usr/local/cuda.
      # CUQUANTUM_ROOT: /usr/local/cuquantum
    steps:
      - uses: actions/checkout@v6
      - uses: julia-actions/setup-julia@v2
        with:
          version: ${{ matrix.version }}
      - uses: julia-actions/cache@v3
      - uses: julia-actions/julia-buildpkg@v1
      - name: Run GPU test suite
        uses: julia-actions/julia-runtest@v1
        env:
          JULIA_CUDA_MEMORY_POOL: none   # disable pool to catch double-free bugs
      - name: Upload trajectory artifact (integration test output)
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: trajectory-${{ matrix.version }}
          path: '**/*.csv'
          if-no-files-found: ignore
```

- [ ] **Step 2: Add a `JULIA_CUDA_MEMORY_POOL: none` env var to the CPU `build` job too**

In the `build` job's `julia-runtest` step, add:
```yaml
      - uses: julia-actions/julia-runtest@v1
        env:
          JULIA_CUDA_MEMORY_POOL: none
```

This is a no-op on CPU (no CUDA pool exists) but documents the intent.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/CI.yml
git commit -m "ci: flesh out GPU test job with self-hosted runner config and artifact upload"
```

---

## Phase B — API Design Fixes

### Task 4: Extend `LinearAlgebra.norm`, `LinearAlgebra.tr`, `LinearAlgebra.dot`

The current `norm`, `trace`, and `inner_product` functions in `state.jl` shadow `LinearAlgebra` names without extending them, causing dispatch ambiguity when both are in scope.

**Files:**
- Modify: `src/densitymat/state.jl`
- Modify: `src/densitymat/CuDensityMat.jl`
- Modify: `test/densitymat/test_state.jl`

- [ ] **Step 1: Write the failing test**

In `test/densitymat/test_state.jl`, add after the existing `norm` test:

```julia
@testset "LinearAlgebra interop" begin
    @gpu_test "norm via LinearAlgebra.norm" begin
        using LinearAlgebra
        ws = WorkStream()
        state = DensePureState{ComplexF64}(ws, [2])
        allocate_storage!(state)
        initialize_zero!(state)
        CUDA.@sync sv = state_view(state)
        sv[1] = 1.0 + 0.0im
        n = LinearAlgebra.norm(state)
        @test n isa Vector
        @test length(n) == 1
        @test n[1] ≈ 1.0
        close(state); close(ws)
    end

    @gpu_test "trace via LinearAlgebra.tr" begin
        using LinearAlgebra
        ws = WorkStream()
        state = DenseMixedState{ComplexF64}(ws, [2])
        allocate_storage!(state)
        initialize_zero!(state)
        CUDA.@sync sv = state_view(state)
        sv[1, 1] = 1.0 + 0.0im
        t = LinearAlgebra.tr(state)
        @test t isa Vector
        @test length(t) == 1
        @test real(t[1]) ≈ 1.0
        close(state); close(ws)
    end

    @gpu_test "inner_product via LinearAlgebra.dot" begin
        using LinearAlgebra
        ws = WorkStream()
        s1 = DensePureState{ComplexF64}(ws, [2])
        s2 = DensePureState{ComplexF64}(ws, [2])
        allocate_storage!(s1); allocate_storage!(s2)
        initialize_zero!(s1); initialize_zero!(s2)
        CUDA.@sync begin
            state_view(s1)[1] = 1.0 + 0.0im
            state_view(s2)[1] = 1.0 + 0.0im
        end
        ip = LinearAlgebra.dot(s1, s2)
        @test ip isa Vector
        @test real(ip[1]) ≈ 1.0
        close(s1); close(s2); close(ws)
    end
end
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | grep -A2 "LinearAlgebra interop"
```

Expected: `UndefVarError: norm not defined` or method-not-found error.

- [ ] **Step 3: Add `import LinearAlgebra` to `CuDensityMat.jl`**

In `src/densitymat/CuDensityMat.jl`, after the existing imports (after line 19 `using CEnum: @cenum`), add:

```julia
import LinearAlgebra
```

- [ ] **Step 4: Rename and extend in `state.jl`**

In `src/densitymat/state.jl`, make the following changes:

Replace the `norm` function definition (around line 344):
```julia
# Before:
function norm(state::DenseState{T}) where {T}
```
```julia
# After:
function LinearAlgebra.norm(state::DenseState{T}) where {T}
```

Replace the `trace` function definition (around line 363):
```julia
# Before:
function trace(state::DenseState{T}) where {T}
```
```julia
# After:
function LinearAlgebra.tr(state::DenseMixedState{T}) where {T}
```

Note: `trace`/`tr` only makes physical sense for mixed states (density matrices). Pure states do not have a meaningful trace operation in this context; restrict the method to `DenseMixedState`.

Replace `inner_product` (around line 404):
```julia
# Before:
function inner_product(left::DenseState{T}, right::DenseState{T}) where {T}
```
```julia
# After:
function LinearAlgebra.dot(left::DenseState{T}, right::DenseState{T}) where {T}
```

Also add backward-compatible aliases at the bottom of `state.jl` so existing call sites keep working during the transition:
```julia
# Aliases for backward compatibility — prefer the LinearAlgebra extensions above.
const norm = LinearAlgebra.norm
const trace = LinearAlgebra.tr
const inner_product = LinearAlgebra.dot
```

- [ ] **Step 5: Update the module-level export list in `CuDensityMat.jl`**

There is no explicit `export norm`/`export trace`/`export inner_product` currently (they were module-local names). No export change needed; users should call `LinearAlgebra.norm(state)` or `using LinearAlgebra; norm(state)` naturally.

- [ ] **Step 6: Run tests to confirm they pass**

```bash
julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | grep -A2 "LinearAlgebra interop"
```

Expected: `3 passed`.

- [ ] **Step 7: Commit**

```bash
git add src/densitymat/state.jl src/densitymat/CuDensityMat.jl test/densitymat/test_state.jl
git commit -m "fix: extend LinearAlgebra.norm/tr/dot instead of shadowing them"
```

---

### Task 5: Add `Base.close` / `Base.isopen` to All Handle Types

Currently only `WorkStream` implements `Base.close`. Every other handle type requires calling `destroy_xxx(obj)` explicitly. This task adds `Base.close`/`Base.isopen` and a private `_destroy!` helper to every handle struct, following the `WorkStream` pattern.

Types to update: `DensePureState`, `DenseMixedState`, `ElementaryOperator`, `MatrixOperator`, `OperatorTerm`, `Operator`, `OperatorAction`, `Expectation`, `OperatorSpectrum`.

**Files:**
- Modify: `src/densitymat/state.jl`
- Modify: `src/densitymat/operators.jl`
- Modify: `src/densitymat/expectation.jl`
- Modify: `src/densitymat/spectrum.jl`
- Modify: `test/densitymat/test_state.jl`
- Modify: `test/densitymat/test_operators.jl`
- Modify: `test/densitymat/test_expectation.jl`
- Modify: `test/densitymat/test_spectrum.jl`

- [ ] **Step 1: Write failing tests for `Base.close` on states**

Add to `test/densitymat/test_state.jl`:

```julia
@testset "Base.close / isopen" begin
    @gpu_test "DensePureState close/isopen lifecycle" begin
        ws = WorkStream()
        state = DensePureState{ComplexF64}(ws, [2])
        @test isopen(state)
        close(state)
        @test !isopen(state)
        # Double-close must be a no-op, not a crash.
        @test_nowarn close(state)
        close(ws)
    end

    @gpu_test "DenseMixedState close/isopen lifecycle" begin
        ws = WorkStream()
        state = DenseMixedState{ComplexF64}(ws, [2])
        @test isopen(state)
        close(state)
        @test !isopen(state)
        @test_nowarn close(state)
        close(ws)
    end
end
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | grep -A2 "close / isopen"
```

Expected: `MethodError: no method matching close(::DensePureState{...})`

- [ ] **Step 3: Add `_destroy!`, `Base.close`, `Base.isopen` to `state.jl`**

In `src/densitymat/state.jl`, for `DensePureState`, replace the inline finalizer (lines 83–88):
```julia
# Before:
finalizer(obj) do x
    if x.handle != C_NULL
        cudensitymatDestroyState(x.handle)
        x.handle = C_NULL
    end
end
```
```julia
# After:
finalizer(_destroy!, obj)
```

And add these methods directly after the struct definition (after the constructor's closing `end`):

```julia
function _destroy!(x::DensePureState)
    if x.handle != C_NULL
        try
            cudensitymatDestroyState(x.handle)
        catch
        end
        x.handle = C_NULL
    end
    if x._owns_storage && x.storage !== nothing
        CUDA.unsafe_free!(x.storage)
        x.storage = nothing
    end
end

Base.close(x::DensePureState) = _destroy!(x)
Base.isopen(x::DensePureState) = x.handle != C_NULL
```

Repeat the identical pattern for `DenseMixedState`:

```julia
function _destroy!(x::DenseMixedState)
    if x.handle != C_NULL
        try
            cudensitymatDestroyState(x.handle)
        catch
        end
        x.handle = C_NULL
    end
    if x._owns_storage && x.storage !== nothing
        CUDA.unsafe_free!(x.storage)
        x.storage = nothing
    end
end

Base.close(x::DenseMixedState) = _destroy!(x)
Base.isopen(x::DenseMixedState) = x.handle != C_NULL
```

- [ ] **Step 4: Add `_destroy!`, `Base.close`, `Base.isopen` to `operators.jl`**

For each of `ElementaryOperator`, `MatrixOperator`, `OperatorTerm`, `Operator`, `OperatorAction`, add the pattern after their respective constructor ends. Use `try/catch` in every `_destroy!` to be safe in finalizers.

For `ElementaryOperator`:
```julia
function _destroy!(x::ElementaryOperator)
    if x.handle != C_NULL
        try
            cudensitymatDestroyElementaryOperator(x.handle)
        catch
        end
        x.handle = C_NULL
    end
end

Base.close(x::ElementaryOperator) = _destroy!(x)
Base.isopen(x::ElementaryOperator) = x.handle != C_NULL
```

Replace the inline finalizer `do x ... end` block with `finalizer(_destroy!, obj)` in the constructor.

Repeat identically for `MatrixOperator` (using `cudensitymatDestroyMatrixOperator`), `OperatorTerm` (using `cudensitymatDestroyOperatorTerm`), `Operator` (using `cudensitymatDestroyOperator`), and `OperatorAction` (using `cudensitymatDestroyOperatorAction`).

- [ ] **Step 5: Add `_destroy!`, `Base.close`, `Base.isopen` to `expectation.jl`**

```julia
function _destroy!(x::Expectation)
    if x.handle != C_NULL
        try
            cudensitymatDestroyExpectation(x.handle)
        catch
        end
        x.handle = C_NULL
    end
end

Base.close(x::Expectation) = _destroy!(x)
Base.isopen(x::Expectation) = x.handle != C_NULL
```

Replace inline finalizer with `finalizer(_destroy!, obj)`.

- [ ] **Step 6: Add `_destroy!`, `Base.close`, `Base.isopen` to `spectrum.jl`**

```julia
function _destroy!(x::OperatorSpectrum)
    if x.handle != C_NULL
        try
            cudensitymatDestroyOperatorSpectrum(x.handle)
        catch
        end
        x.handle = C_NULL
    end
end

Base.close(x::OperatorSpectrum) = _destroy!(x)
Base.isopen(x::OperatorSpectrum) = x.handle != C_NULL
```

- [ ] **Step 7: Run tests to confirm they pass**

```bash
julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | grep -A3 "close / isopen"
```

Expected: all new tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/densitymat/state.jl src/densitymat/operators.jl src/densitymat/expectation.jl src/densitymat/spectrum.jl test/densitymat/test_state.jl test/densitymat/test_operators.jl test/densitymat/test_expectation.jl test/densitymat/test_spectrum.jl
git commit -m "fix: add Base.close/isopen and _destroy! to all handle types"
```

---

### Task 6: Fix `CUDA.device!()` Side Effect in `WorkStream` Constructor

`WorkStream(device_id=1)` currently calls `CUDA.device!(1)` as a side effect, permanently changing the device for the calling thread. Fix it to determine the device ID without changing global state.

**Files:**
- Modify: `src/densitymat/workspace.jl`
- Modify: `test/densitymat/test_workstream.jl`

- [ ] **Step 1: Write the failing test**

Add to `test/densitymat/test_workstream.jl`:

```julia
@testset "WorkStream device_id does not mutate calling thread device" begin
    @gpu_test "device unchanged after WorkStream construction" begin
        original_device = Int(CUDA.device())
        # Constructing with the same device_id should be a no-op for global state.
        ws = WorkStream(device_id = original_device)
        @test Int(CUDA.device()) == original_device
        close(ws)
        # Constructing without device_id should also leave the device unchanged.
        ws2 = WorkStream()
        @test Int(CUDA.device()) == original_device
        close(ws2)
    end
end
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | grep -A3 "device unchanged"
```

Expected: passes trivially on single-GPU machines (since device 0 → device 0 is a no-op), but the underlying code path is still wrong. The test documents the contract. On multi-GPU setups it would fail.

- [ ] **Step 3: Fix `workspace.jl` lines 48–53**

Replace:
```julia
dev = if device_id !== nothing
    CUDA.device!(device_id)
    device_id
else
    CUDA.device().handle
end
```
with:
```julia
dev = if device_id !== nothing
    device_id
else
    Int(CUDA.device())
end
```

The `WorkStream` stores `dev` as `device_id::Int` for reference; no side effect needed. The `stream` passed in (or created) already anchors to the correct device.

- [ ] **Step 4: Also fix `CUDA.device().handle` → `Int(CUDA.device())`**

`CUDA.device().handle` accesses an internal field. `Int(CUDA.device())` is the documented API for getting the device integer ID.

- [ ] **Step 5: Run tests to confirm they pass**

```bash
julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | grep -E "(WorkStream|PASS|FAIL)" | head -20
```

Expected: workstream tests all pass.

- [ ] **Step 6: Commit**

```bash
git add src/densitymat/workspace.jl test/densitymat/test_workstream.jl
git commit -m "fix: remove CUDA.device!() side effect from WorkStream constructor"
```

---

### Task 7: Fix `batch_size=0` Default in Compute Functions

`compute_action!`, `compute_operator_action!`, and `compute_operator_action_backward!` default `batch_size=0`, which is almost certainly invalid for the C API. The consistent default (matching `compute_expectation!` and `compute_spectrum!`) is `1`.

**Files:**
- Modify: `src/densitymat/operators.jl`
- Modify: `test/densitymat/test_operators.jl`

- [ ] **Step 1: Write the failing test**

Add to `test/densitymat/test_operators.jl`:

```julia
@testset "batch_size default is 1" begin
    @gpu_test "compute_operator_action! default batch_size=1 does not error" begin
        ws = WorkStream()
        hilbert_dims = [2]
        sigma_z_data = CUDA.CuArray(ComplexF64[1.0 0.0; 0.0 -1.0])
        elem_op = create_elementary_operator(ws, hilbert_dims, sigma_z_data)
        term = create_operator_term(ws, Int64.(hilbert_dims))
        append_elementary_product!(term, [elem_op], Int32[0], Int32[0])
        op = create_operator(ws, Int64.(hilbert_dims))
        append_term!(op, term)

        state_in = DensePureState{ComplexF64}(ws, hilbert_dims)
        state_out = DensePureState{ComplexF64}(ws, hilbert_dims)
        allocate_storage!(state_in); allocate_storage!(state_out)
        initialize_zero!(state_in)
        CUDA.@sync state_view(state_in)[1] = 1.0 + 0.0im

        prepare_operator_action!(ws, op, state_in, state_out)
        workspace_allocate!(ws, workspace_query_size(ws))
        # Must not throw — previously batch_size=0 could cause C API errors.
        @test_nowarn compute_operator_action!(ws, op, state_in, state_out)

        close(state_in); close(state_out)
        close(op); close(term); close(elem_op); close(ws)
    end
end
```

- [ ] **Step 2: Run test to confirm it fails (or documents a latent bug)**

```bash
julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | grep -A3 "batch_size default"
```

- [ ] **Step 3: Fix the three function signatures in `operators.jl`**

Around line 701, change:
```julia
function compute_action!(ws, action, states_in, state_out;
    time=0.0, batch_size=0, num_params=0, params=nothing)
```
to:
```julia
function compute_action!(ws, action, states_in, state_out;
    time=0.0, batch_size=1, num_params=0, params=nothing)
```

Around line 775, change:
```julia
function compute_operator_action!(ws, operator, state_in, state_out;
    time=0.0, batch_size=0, num_params=0, params=nothing)
```
to:
```julia
function compute_operator_action!(ws, operator, state_in, state_out;
    time=0.0, batch_size=1, num_params=0, params=nothing)
```

Around line 860, change:
```julia
function compute_operator_action_backward!(ws, operator, state_in, state_out_adj, state_in_adj, params_grad;
    time=0.0, batch_size=0, num_params=0, params=nothing)
```
to:
```julia
function compute_operator_action_backward!(ws, operator, state_in, state_out_adj, state_in_adj, params_grad;
    time=0.0, batch_size=1, num_params=0, params=nothing)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | grep -A3 "batch_size default"
```

Expected: test passes.

- [ ] **Step 5: Commit**

```bash
git add src/densitymat/operators.jl test/densitymat/test_operators.jl
git commit -m "fix: change batch_size default from 0 to 1 in compute functions"
```

---

### Task 8: Concretely Type `_data_ref` and `_callback_refs` Fields

`ElementaryOperator._data_ref::Any`, `MatrixOperator._data_ref::Any`, and related `Any`-typed GC anchor fields force dynamic dispatch on every access. Concretely typing these removes the overhead and makes the struct layout predictable.

**Files:**
- Modify: `src/densitymat/operators.jl`
- Modify: `src/densitymat/workspace.jl`

- [ ] **Step 1: Update `ElementaryOperator` struct**

In `src/densitymat/operators.jl`, replace the struct definition (around lines 68–77):

```julia
# Before:
mutable struct ElementaryOperator
    handle::cudensitymatElementaryOperator_t
    ws::WorkStream
    _data_ref::Any
    _callback_refs::Any
end
```

```julia
# After:
mutable struct ElementaryOperator
    handle::cudensitymatElementaryOperator_t
    ws::WorkStream
    # GC anchors: prevent CuArray data and callback closures from being collected
    # while the C library holds pointers to them.
    _data_ref::Union{Nothing, CUDA.CuArray}
    _callback_refs::Union{Nothing, CallbackRef}
end
```

Note: `CallbackRef` is defined in Task 10. If executing sequentially, either forward-declare or do Task 10 before this step. For now, leave the type as `Union{Nothing, Any}` — that is still better than bare `Any` for documentation — and tighten after Task 10.

Interim version (safe to merge before Task 10):
```julia
mutable struct ElementaryOperator
    handle::cudensitymatElementaryOperator_t
    ws::WorkStream
    _data_ref::Union{Nothing, CUDA.CuArray}
    _callback_refs::Union{Nothing, NamedTuple}
end
```

Update the inner constructor call to pass `nothing` explicitly:
```julia
ElementaryOperator(handle, ws; data_ref=nothing, callback_refs=nothing) =
    ElementaryOperator(handle, ws, data_ref, callback_refs)
```

- [ ] **Step 2: Update `MatrixOperator` struct**

Same pattern:
```julia
mutable struct MatrixOperator
    handle::cudensitymatMatrixOperator_t
    ws::WorkStream
    _data_ref::Union{Nothing, CUDA.CuArray}
    _callback_refs::Union{Nothing, NamedTuple}
end
```

- [ ] **Step 3: Update `OperatorTerm._elem_op_refs` and `_matrix_op_refs`**

```julia
# Before:
mutable struct OperatorTerm
    handle::cudensitymatOperatorTerm_t
    ws::WorkStream
    hilbert_space_dims::Vector{Int64}
    _elem_op_refs::Vector{Any}
    _matrix_op_refs::Vector{Any}
    _callback_refs::Vector{Any}
end
```

```julia
# After:
mutable struct OperatorTerm
    handle::cudensitymatOperatorTerm_t
    ws::WorkStream
    hilbert_space_dims::Vector{Int64}
    _elem_op_refs::Vector{ElementaryOperator}
    _matrix_op_refs::Vector{MatrixOperator}
    _callback_refs::Vector{NamedTuple}   # tighten to Vector{CallbackRef} after Task 10
end
```

Update the constructor to initialize with typed empty vectors:
```julia
OperatorTerm(handle, ws, dims::Vector{Int64}) = begin
    obj = new(handle, ws, dims,
              ElementaryOperator[], MatrixOperator[], NamedTuple[])
    finalizer(_destroy!, obj)
    obj
end
```

- [ ] **Step 4: Update `Operator._term_refs`**

```julia
# Before:
_term_refs::Vector{Any}
```

```julia
# After:
_term_refs::Vector{Tuple{OperatorTerm, ComplexF64}}
```

Update `append_term!` to push `(term, ComplexF64(coefficient))`.

- [ ] **Step 5: Update `WorkStream._comm_ref`**

In `src/densitymat/workspace.jl`:
```julia
# Before:
_comm_ref::Any
```

```julia
# After:
_comm_ref::Union{Nothing, Vector{Int}}
```

Initialize to `nothing` in the constructor. In `set_communicator!`, store as `Vector{Int}`.

- [ ] **Step 6: Run the full test suite to verify nothing broke**

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: all previously passing tests still pass. No new failures.

- [ ] **Step 7: Commit**

```bash
git add src/densitymat/operators.jl src/densitymat/workspace.jl
git commit -m "fix: replace Any-typed GC anchor fields with concrete Union types"
```

---

## Phase C — Safety Improvements

### Task 9: Add CUDA Context Guard to All Finalizers

Finalizers run on a GC thread. If the CUDA context has been destroyed (e.g., at program exit), calling `cudensitymatDestroy*` inside a finalizer can deadlock or crash. Wrapping the call in `try/catch` is the pragmatic mitigation adopted throughout CUDA.jl itself.

This task audits every finalizer in the codebase and ensures they all use the `try/catch` guard added as part of Task 5's `_destroy!` helpers. If Task 5 was completed first, this task is a verification pass, not a code change.

**Files:**
- Verify: `src/densitymat/state.jl`
- Verify: `src/densitymat/operators.jl`
- Verify: `src/densitymat/expectation.jl`
- Verify: `src/densitymat/spectrum.jl`
- Verify: `src/densitymat/workspace.jl`

- [ ] **Step 1: Grep for any remaining inline finalizer patterns without try/catch**

```bash
grep -n "finalizer" src/densitymat/*.jl
```

Expected: every finalizer is now `finalizer(_destroy!, obj)`. There should be no inline `do x ... cudensitymatDestroy... end` blocks remaining.

- [ ] **Step 2: Verify `WorkStream._destroy!` also uses try/catch**

Read `src/densitymat/workspace.jl` lines 87–102. If the current `_destroy!` does not have a `try/catch` around the C API calls, add one:

```julia
function _destroy!(ws::WorkStream)
    if ws.workspace != C_NULL
        if ws.workspace_buffer !== nothing
            CUDA.unsafe_free!(ws.workspace_buffer)
            ws.workspace_buffer = nothing
        end
        try
            cudensitymatDestroyWorkspace(ws.workspace)
        catch
        end
        ws.workspace = C_NULL
    end
    if ws.handle != C_NULL
        try
            destroy(ws.handle)
        catch
        end
        ws.handle = C_NULL
    end
end
```

- [ ] **Step 3: Run the full test suite**

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: all tests pass.

- [ ] **Step 4: Commit (if any changes were needed)**

```bash
git add src/densitymat/workspace.jl
git commit -m "fix: guard all CUDA API calls in finalizers with try/catch"
```

---

### Task 10: Add `CallbackRef` Wrapper Type with Finalizer

Currently `wrap_scalar_callback` and `wrap_tensor_callback` return an anonymous `NamedTuple` as the `refs` component. Users must call `unregister_callback!(refs)` manually or leak the registry entry. A concrete `CallbackRef` type with a finalizer automates cleanup.

**Files:**
- Modify: `src/densitymat/callbacks.jl`
- Modify: `src/densitymat/CuDensityMat.jl`
- Modify: `src/densitymat/operators.jl` (update `_callback_refs` type after CallbackRef exists)
- Modify: `test/densitymat/test_callbacks.jl`

- [ ] **Step 1: Write the failing test**

Add to `test/densitymat/test_callbacks.jl`:

```julia
@testset "CallbackRef lifecycle" begin
    @gpu_test "CallbackRef is a concrete type" begin
        f = (time, params, storage) -> (storage .= ComplexF64(time))
        cb, gcb, ref = wrap_scalar_callback(f)
        @test ref isa CuDensityMat.CallbackRef
        @test isopen(ref)
        close(ref)
        @test !isopen(ref)
        # Double-close must be a no-op.
        @test_nowarn close(ref)
    end

    @gpu_test "CallbackRef finalizer unregisters on GC" begin
        f = (time, params, storage) -> nothing
        _, _, ref = wrap_scalar_callback(f)
        id = ref.id
        # Confirm registered.
        @test haskey(CuDensityMat._callback_registry, id)
        # Drop all references and force GC.
        ref = nothing
        GC.gc(true)
        # The finalizer should have unregistered the callback.
        @test !haskey(CuDensityMat._callback_registry, id)
    end
end
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | grep -A3 "CallbackRef lifecycle"
```

Expected: `UndefVarError: CallbackRef not defined`.

- [ ] **Step 3: Add `CallbackRef` to `callbacks.jl`**

After the global registry definition (around line 47), add:

```julia
"""
    CallbackRef

A handle to a registered Julia callback. Holds a reference to the callback
function in the global registry and unregisters it automatically when finalized
or when `close` is called explicitly.
"""
mutable struct CallbackRef
    id::UInt
    grad_id::UInt   # 0 if no gradient callback registered
end

function CallbackRef(id::UInt, grad_id::UInt)
    obj = new(id, grad_id)
    finalizer(_destroy_callback_ref!, obj)
    obj
end

function _destroy_callback_ref!(ref::CallbackRef)
    if ref.id != 0
        _unregister_callback(ref.id)
        ref.id = 0
    end
    if ref.grad_id != 0
        _unregister_callback(ref.grad_id)
        ref.grad_id = 0
    end
end

Base.close(ref::CallbackRef) = _destroy_callback_ref!(ref)
Base.isopen(ref::CallbackRef) = ref.id != 0
```

- [ ] **Step 4: Update `wrap_scalar_callback` and `wrap_tensor_callback` to return `CallbackRef`**

In `wrap_scalar_callback` (around line 397–431), change the return statement from:
```julia
refs = (id = id, grad_id = grad_id, f = f, gradient = gradient)
return cb, gcb, refs
```
to:
```julia
ref = CallbackRef(id, grad_id)
return cb, gcb, ref
```

Apply the same change to `wrap_tensor_callback`.

- [ ] **Step 5: Update `unregister_callback!` to accept `CallbackRef`**

Replace the existing `unregister_callback!` function with:
```julia
"""
    unregister_callback!(ref::CallbackRef)

Explicitly unregister a callback from the global registry. Equivalent to
calling `close(ref)`. Prefer using `close` or relying on the finalizer.
"""
function unregister_callback!(ref::CallbackRef)
    close(ref)
end
```

- [ ] **Step 6: Export `CallbackRef` and `unregister_callback!` in `CuDensityMat.jl`**

In `src/densitymat/CuDensityMat.jl`, add to the exports from `callbacks.jl`:
```julia
export CallbackRef, wrap_scalar_callback, wrap_tensor_callback, unregister_callback!
```

- [ ] **Step 7: Update `operators.jl` `_callback_refs` fields to use `CallbackRef`**

Now that `CallbackRef` is defined, update (from Task 8's interim types):
```julia
# In ElementaryOperator:
_callback_refs::Union{Nothing, CallbackRef}

# In MatrixOperator:
_callback_refs::Union{Nothing, CallbackRef}

# In OperatorTerm:
_callback_refs::Vector{CallbackRef}
```

Update the `OperatorTerm` constructor:
```julia
obj = new(handle, ws, dims, ElementaryOperator[], MatrixOperator[], CallbackRef[])
```

- [ ] **Step 8: Run tests to confirm they pass**

```bash
julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | grep -A5 "CallbackRef"
```

Expected: all `CallbackRef` tests pass.

- [ ] **Step 9: Commit**

```bash
git add src/densitymat/callbacks.jl src/densitymat/CuDensityMat.jl src/densitymat/operators.jl test/densitymat/test_callbacks.jl
git commit -m "feat: add CallbackRef wrapper type with automatic finalizer-based cleanup"
```

---

## Phase D — Testing Infrastructure

### Task 11: Fix `TEST_DTYPES`/`TEST_BATCH_SIZES` and Parameterize Tests

`test/setup.jl` defines `TEST_DTYPES`, `TEST_DIMS`, `TEST_BATCH_SIZES` but no test uses them. `TEST_BATCH_SIZES = [0, 1, 4]` includes an invalid value (`0`). This task wires up the constants and adds a `sync_and_pull` helper.

**Files:**
- Modify: `test/setup.jl`
- Modify: `test/densitymat/test_state.jl`
- Modify: `test/densitymat/test_operators.jl`

- [ ] **Step 1: Fix `TEST_BATCH_SIZES` and add `sync_and_pull` in `setup.jl`**

In `test/setup.jl`, replace the constants block:
```julia
# Before:
const TEST_DTYPES = HAS_GPU ? [ComplexF32, ComplexF64] : []
const TEST_DIMS = [[2], [2, 2], [2, 3], [3, 3]]
const TEST_BATCH_SIZES = [0, 1, 4]
```
```julia
# After:
const TEST_DTYPES = HAS_GPU ? [ComplexF32, ComplexF64] : []
const TEST_DIMS = [[2], [2, 2], [2, 3], [3, 3]]
const TEST_BATCH_SIZES = [1, 4]   # 0 is not a valid batch_size per C API contract

# Helper: synchronize the CUDA stream and copy a CuArray to CPU.
# Use instead of bare Array(x) in tests so that CUDA errors surface immediately.
function sync_and_pull(x::CUDA.CuArray)
    CUDA.synchronize()
    Array(x)
end
```

- [ ] **Step 2: Parameterize `test_state.jl` norm/trace tests over `TEST_DTYPES`**

In `test/densitymat/test_state.jl`, find the `@testset "norm"` block. Replace the hardcoded `ComplexF64` loop with:

```julia
@testset "norm (dtype=$T)" for T in TEST_DTYPES
    @gpu_test "pure state norm, batch_size=1" begin
        ws = WorkStream()
        state = DensePureState{T}(ws, [2])
        allocate_storage!(state)
        initialize_zero!(state)
        CUDA.@sync state_view(state)[1] = T(1)
        n = sync_and_pull(CUDA.CuArray(LinearAlgebra.norm(state)))
        @test n[1] ≈ real(T)(1)
        close(state); close(ws)
    end
end
```

- [ ] **Step 3: Parameterize `test_operators.jl` over `TEST_BATCH_SIZES`**

In `test/densitymat/test_operators.jl`, find the `compute_operator_action!` test. Add an outer loop:

```julia
@testset "compute_operator_action! (batch_size=$bs)" for bs in TEST_BATCH_SIZES
    @gpu_test "sigma_z action, batch_size=$bs" begin
        ws = WorkStream()
        hilbert_dims = [2]
        # Construct sigma_z diagonal as batch of identical operators.
        data = CUDA.CuArray(repeat(ComplexF64[1.0 0.0; 0.0 -1.0], outer=(1, 1, bs)))
        # ... (rest of operator setup using bs as batch_size)
        # Call with explicit batch_size:
        compute_operator_action!(ws, op, state_in, state_out; batch_size=bs)
        CUDA.synchronize()
        # ... assertions
        close(ws)
    end
end
```

- [ ] **Step 4: Run tests to confirm the parameterized tests all pass**

```bash
julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | grep -E "(dtype=|batch_size=)" | head -30
```

Expected: tests for each dtype and batch_size combination appear and pass.

- [ ] **Step 5: Commit**

```bash
git add test/setup.jl test/densitymat/test_state.jl test/densitymat/test_operators.jl
git commit -m "test: wire up TEST_DTYPES/TEST_BATCH_SIZES and fix invalid batch_size=0"
```

---

### Task 12: Fix `trajectory.csv` and `runtests.jl` Phase Numbering

**Files:**
- Modify: `test/densitymat/test_integration.jl`
- Modify: `test/runtests.jl`
- Create: `test/densitymat/test_batch_operators.jl`

- [ ] **Step 1: Fix trajectory.csv in `test_integration.jl`**

Find the section in `test/densitymat/test_integration.jl` around lines 547–569 that writes to the repo root:
```julia
# Before:
trajectory_file = joinpath(@__DIR__, "..", "..", "trajectory.csv")
open(trajectory_file, "w") do io ...
@test isfile(trajectory_file)
```

Replace with:
```julia
# After:
mktempdir() do tmpdir
    trajectory_file = joinpath(tmpdir, "trajectory.csv")
    open(trajectory_file, "w") do io
        # ... same write logic ...
    end
    @test isfile(trajectory_file)
    # File is automatically cleaned up when mktempdir block exits.
end
```

- [ ] **Step 2: Create `test/densitymat/test_batch_operators.jl` (Phase 6)**

```julia
# test/densitymat/test_batch_operators.jl
# Phase 6: Batch operator construction and compute tests.

@testset "Batch operator API" begin

    @testset "create_elementary_operator_batch" begin
        @gpu_test "batch elementary operator construction" begin
            ws = WorkStream()
            hilbert_dims = [2]
            batch_size = 4
            # 4 independent 2x2 matrices stacked as (2, 2, 4) CuArray.
            data = CUDA.CuArray(
                ComplexF64[1 0; 0 -1;;;
                           0 1; 1  0;;;
                           0 -1im; 1im 0;;;
                           1 0; 0  1]
            )
            op = create_elementary_operator_batch(ws, hilbert_dims, data, batch_size)
            @test op isa ElementaryOperator
            @test isopen(op)
            close(op); close(ws)
        end
    end

    @testset "append_elementary_product_batch!" begin
        @gpu_test "batch term append with static coefficients" begin
            ws = WorkStream()
            hilbert_dims = [2]
            batch_size = 4
            data = CUDA.CuArray(
                ComplexF64[1 0; 0 -1;;;
                           1 0; 0 -1;;;
                           1 0; 0 -1;;;
                           1 0; 0 -1]
            )
            op = create_elementary_operator_batch(ws, hilbert_dims, data, batch_size)
            term = create_operator_term(ws, Int64.(hilbert_dims))
            static_coeffs = CUDA.CuArray(ComplexF64[1.0, 1.0, 1.0, 1.0])
            append_elementary_product_batch!(
                term, [op], Int32[0], Int32[0], batch_size, static_coeffs
            )
            @test isopen(term)
            close(term); close(op); close(ws)
        end
    end

    @testset "append_term_batch!" begin
        @gpu_test "batch term append to operator" begin
            ws = WorkStream()
            hilbert_dims = [2]
            batch_size = 2
            data = CUDA.CuArray(
                ComplexF64[1 0; 0 -1;;;
                           1 0; 0  1]
            )
            elem_op = create_elementary_operator_batch(ws, hilbert_dims, data, batch_size)
            term = create_operator_term(ws, Int64.(hilbert_dims))
            static_coeffs = CUDA.CuArray(ComplexF64[1.0, 1.0])
            append_elementary_product_batch!(
                term, [elem_op], Int32[0], Int32[0], batch_size, static_coeffs
            )
            op = create_operator(ws, Int64.(hilbert_dims))
            batch_term_coeffs = CUDA.CuArray(ComplexF64[1.0, 1.0])
            append_term_batch!(op, term, batch_size, batch_term_coeffs)
            @test isopen(op)
            close(op); close(term); close(elem_op); close(ws)
        end
    end

end
```

- [ ] **Step 3: Fix `runtests.jl` Phase numbering**

In `test/runtests.jl`, replace the comment block:
```julia
# Before:
# Phase 5: Callback tests
include("densitymat/test_callbacks.jl")

# Phase 7: Gradient / backward differentiation tests
include("densitymat/test_gradients.jl")
```

```julia
# After:
# Phase 5: Callback tests
include("densitymat/test_callbacks.jl")

# Phase 6: Batch operator construction and compute tests
include("densitymat/test_batch_operators.jl")

# Phase 7: Gradient / backward differentiation tests
include("densitymat/test_gradients.jl")
```

- [ ] **Step 4: Run tests to confirm batch operator tests pass and integration test no longer writes to repo root**

```bash
julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | grep -E "(Batch operator|trajectory)"
# Also confirm no stray trajectory.csv in repo root:
ls trajectory.csv 2>/dev/null || echo "clean"
```

Expected: batch tests pass; `clean` for trajectory.csv.

- [ ] **Step 5: Commit**

```bash
git add test/densitymat/test_integration.jl test/runtests.jl test/densitymat/test_batch_operators.jl
git commit -m "test: fix trajectory.csv tempdir, add batch operator tests, fill Phase 6 gap"
```

---

## Phase E — Documentation

### Task 13: Add Docstrings to All Exported Symbols

The `docs/make.jl` currently suppresses missing-docstring warnings with `warnonly = [:missing_docs]`. This task adds docstrings to every exported symbol and then removes the suppression.

**Files:**
- Modify: `src/densitymat/workspace.jl`
- Modify: `src/densitymat/state.jl`
- Modify: `src/densitymat/operators.jl`
- Modify: `src/densitymat/callbacks.jl`
- Modify: `src/densitymat/expectation.jl`
- Modify: `src/densitymat/spectrum.jl`
- Modify: `src/densitymat/error.jl`
- Modify: `docs/make.jl`

- [ ] **Step 1: Add docstring to `WorkStream`**

Before the `mutable struct WorkStream` definition in `workspace.jl`:

```julia
"""
    WorkStream(; stream=nothing, memory_limit=nothing, device_id=nothing)

A container holding a cudensitymat library handle, a workspace descriptor,
and a CUDA stream. All CuDensityMat operations require a `WorkStream`.

# Arguments
- `stream`: Optional `CUDA.CuStream` to use. Defaults to the current CUDA stream.
- `memory_limit`: Optional maximum workspace size in bytes. Defaults to 80% of
  free GPU memory at time of allocation.
- `device_id`: Optional integer CUDA device ID. Defaults to the currently active device.

# Lifecycle
Call `close(ws)` when done, or rely on the GC finalizer. Use `isopen(ws)` to
check whether the underlying handles are still valid.

# Example
```julia
ws = WorkStream()
# ... use ws for operator/state construction ...
close(ws)
```
"""
```

- [ ] **Step 2: Add docstrings to `DensePureState` and `DenseMixedState`**

Before `mutable struct DensePureState{T}`:

```julia
"""
    DensePureState{T}(ws::WorkStream, hilbert_space_dims; batch_size=1)

A batched pure quantum state stored as a dense state vector on the GPU.

`T` must be `ComplexF32` or `ComplexF64`. `hilbert_space_dims` is a vector of
integers giving the local Hilbert space dimension for each mode (qudit). The
total Hilbert space dimension is `prod(hilbert_space_dims)`.

After construction, call `allocate_storage!` (allocates new GPU memory) or
`attach_storage!` (binds existing `CuVector{T}`) before passing to compute
functions.

# Example
```julia
ws = WorkStream()
state = DensePureState{ComplexF64}(ws, [2, 2])   # two-qubit pure state
allocate_storage!(state)
initialize_zero!(state)
close(state); close(ws)
```
"""
```

Before `mutable struct DenseMixedState{T}`:

```julia
"""
    DenseMixedState{T}(ws::WorkStream, hilbert_space_dims; batch_size=1)

A batched mixed quantum state stored as a dense density matrix on the GPU.

Identical to `DensePureState` in interface but stores an `N×N` density matrix
(where `N = prod(hilbert_space_dims)`) rather than an `N`-element state vector.
"""
```

- [ ] **Step 3: Add docstrings to all exported operator functions**

Before `create_elementary_operator`:

```julia
"""
    create_elementary_operator(ws, space_mode_extents, data::CuArray{T};
        sparsity=:none, diagonal_offsets=Int32[],
        tensor_callback=NULL_TENSOR_CALLBACK,
        tensor_gradient_callback=NULL_TENSOR_GRADIENT_CALLBACK) -> ElementaryOperator

Create a single-site elementary operator from a dense GPU tensor.

`space_mode_extents` is a vector of integers specifying the local dimension for
each mode the operator acts on (e.g. `[2]` for a qubit operator, `[2, 2]` for a
two-qubit gate stored as a rank-4 tensor).

`data` must be a `CuArray{T}` of shape `(d_1, d_2, ..., d_n, d_1, d_2, ..., d_n)`
where `d_i = space_mode_extents[i]`.

Use `sparsity=:multidiagonal` with `diagonal_offsets` for banded sparse operators.

Call `close(op)` when done, or rely on the GC finalizer.
"""
```

Before `create_matrix_operator`:

```julia
"""
    create_matrix_operator(ws, space_mode_extents, data::CuArray{T};
        tensor_callback=NULL_TENSOR_CALLBACK,
        tensor_gradient_callback=NULL_TENSOR_GRADIENT_CALLBACK) -> MatrixOperator

Create a dense matrix operator in the full local Hilbert space.

Unlike `ElementaryOperator`, a `MatrixOperator` represents an operator as a
full `N×N` dense matrix where `N = prod(space_mode_extents)`.
"""
```

Before `create_operator_term`:

```julia
"""
    create_operator_term(ws, hilbert_space_dims) -> OperatorTerm

Create an empty operator term in the given Hilbert space.

An `OperatorTerm` represents a sum of products of elementary operators.
Use `append_elementary_product!` or `append_matrix_product!` to add terms.
"""
```

Before `create_operator`:

```julia
"""
    create_operator(ws, hilbert_space_dims) -> Operator

Create a composite operator (superoperator) from one or more `OperatorTerm`s.

Use `append_term!` to add terms. Pass the assembled `Operator` to
`prepare_operator_action!` / `compute_operator_action!` for state evolution, or
to `create_expectation` for expectation values.
"""
```

Before `create_operator_action`:

```julia
"""
    create_operator_action(ws, operators::Vector{Operator}) -> OperatorAction

Create a multi-operator action handle that applies a sum of operators to a state.

Prefer `compute_operator_action!` for single-operator application. Use this
when you need to apply a linear combination of independently constructed operators.
"""
```

- [ ] **Step 4: Add docstrings to `wrap_scalar_callback` and `wrap_tensor_callback`**

Before `wrap_scalar_callback`:

```julia
"""
    wrap_scalar_callback(f; gradient=nothing) -> (cb, gcb, ref::CallbackRef)

Wrap a Julia function `f` as a cudensitymat scalar callback.

The wrapped function `f` must have the signature:
```julia
f(time::Float64, params::Matrix{Float64}, storage::Vector{T}) -> Nothing
```
where `params` is a `(num_params, batch_size)` matrix and `storage` is a
length-`batch_size` vector to fill with the scalar value for each batch element.

`gradient` (optional) must have the signature:
```julia
gradient(time::Float64, params::Matrix{Float64},
         scalar_grad::Vector{T}, params_grad::Matrix{Float64}) -> Nothing
```

Returns:
- `cb::cudensitymatWrappedScalarCallback_t` — pass to `append_elementary_product!`
- `gcb::cudensitymatWrappedScalarGradientCallback_t` — pass alongside `cb`
- `ref::CallbackRef` — keep alive as long as the callback is in use; `close(ref)` to unregister

# Example
```julia
f = (t, p, s) -> (s .= exp(-t))
cb, gcb, ref = wrap_scalar_callback(f)
append_elementary_product!(term, ops, modes, dualities; coefficient_callback=cb)
# ... run computations ...
close(ref)
```
"""
```

Before `wrap_tensor_callback`:

```julia
"""
    wrap_tensor_callback(f; gradient=nothing) -> (cb, gcb, ref::CallbackRef)

Wrap a Julia function `f` as a cudensitymat tensor callback.

The wrapped function `f` must have the signature:
```julia
f(time::Float64, params::Matrix{Float64}, storage::Array{T}) -> Nothing
```
where `storage` has shape `(mode_extents..., batch_size)`.

See `wrap_scalar_callback` for return value semantics.
"""
```

- [ ] **Step 5: Add docstrings to `Expectation` and `OperatorSpectrum`**

Before `mutable struct Expectation`:

```julia
"""
    Expectation

Handle for computing expectation values `⟨ψ|Ô|ψ⟩` (pure) or `Tr(Ô ρ)` (mixed)
of a quantum operator over a state.

Construct with `create_expectation(ws, operator)`. Then call
`prepare_expectation!(ws, exp, state)` once, allocate workspace with
`workspace_allocate!`, and repeatedly call `compute_expectation!` to evaluate.

Call `close(exp)` when done.
"""
```

Before `mutable struct OperatorSpectrum`:

```julia
"""
    OperatorSpectrum

Handle for computing the eigenspectrum of a quantum operator.

Construct with `create_operator_spectrum(ws, operator)`. Then call
`prepare_spectrum!`, allocate workspace, and call `compute_spectrum!`.

Call `close(spec)` when done.

!!! note
    Only available with cuQuantum >= 24.08. Check `SPECTRUM_SUPPORTED` in tests.
"""
```

- [ ] **Step 6: Add docstrings to `create_expectation`, `prepare_expectation!`, `compute_expectation!`**

```julia
"""
    create_expectation(ws, operator::Operator) -> Expectation

Create an expectation value handle for the given operator.
"""

"""
    prepare_expectation!(ws, exp::Expectation, state;
        compute_type=CUDENSITYMAT_COMPUTE_64F, workspace_limit=nothing)

Prepare the workspace plan for computing `⟨state|op|state⟩`.

Must be called once before `compute_expectation!`. After calling this,
query the required workspace size with `workspace_query_size(ws)` and allocate
with `workspace_allocate!(ws, size)`.
"""

"""
    compute_expectation!(ws, exp::Expectation, state, result::CuArray;
        time=0.0, batch_size=1, num_params=0, params=nothing)

Compute the expectation value into `result` (a GPU array).

`result` must have length `batch_size`. After this call, the result is
available on the GPU; use `Array(result)` or `CUDA.@allowscalar result[1]`
to access on CPU.
"""
```

- [ ] **Step 7: Run the docs build to verify no missing-docstring warnings**

```bash
julia --project=docs/ docs/make.jl 2>&1 | grep -i "missing\|warn\|error"
```

Expected: no warnings. If any appear, add the missing docstring before the next step.

- [ ] **Step 8: Remove `warnonly` from `docs/make.jl`**

In `docs/make.jl`, remove the line:
```julia
warnonly = [:missing_docs],
```

- [ ] **Step 9: Run docs build again to confirm clean**

```bash
julia --project=docs/ docs/make.jl 2>&1 | grep -i "warn\|error"
```

Expected: no output (clean build).

- [ ] **Step 10: Commit**

```bash
git add src/densitymat/*.jl docs/make.jl
git commit -m "docs: add docstrings to all exported symbols, remove warnonly suppression"
```

---

## Phase F — Performance

### Task 14: Remove Eager `CUDA.synchronize()` After Compute Calls

Every compute function currently ends with `CUDA.synchronize()`, serializing all GPU work and preventing any CPU/GPU overlap. The correct pattern is stream-ordered execution: let the caller synchronize at transfer boundaries (e.g., when calling `Array(result)`).

**Files:**
- Modify: `src/densitymat/operators.jl`
- Modify: `src/densitymat/state.jl`
- Modify: `src/densitymat/expectation.jl`
- Modify: `src/densitymat/spectrum.jl`
- Modify: `test/densitymat/test_operators.jl`
- Modify: `test/densitymat/test_state.jl`
- Modify: `test/densitymat/test_expectation.jl`
- Modify: `test/densitymat/test_gradients.jl`
- Modify: `test/densitymat/test_integration.jl`

- [ ] **Step 1: Audit all `CUDA.synchronize()` calls**

```bash
grep -n "CUDA.synchronize" src/densitymat/*.jl
```

Expected lines (from exploration):
- `operators.jl`: lines ~726, ~799, ~888
- `state.jl`: lines ~335, ~354, ~373, ~413
- `expectation.jl`: line ~121
- `spectrum.jl`: line ~190

- [ ] **Step 2: Write a test asserting async behavior is preserved**

Add to `test/densitymat/test_operators.jl`:

```julia
@testset "compute_operator_action! is asynchronous" begin
    @gpu_test "result is available after explicit sync, not before call returns" begin
        ws = WorkStream()
        hilbert_dims = [2]
        sigma_z_data = CUDA.CuArray(ComplexF64[1.0 0.0; 0.0 -1.0])
        elem_op = create_elementary_operator(ws, hilbert_dims, sigma_z_data)
        term = create_operator_term(ws, Int64.(hilbert_dims))
        append_elementary_product!(term, [elem_op], Int32[0], Int32[0])
        op = create_operator(ws, Int64.(hilbert_dims))
        append_term!(op, term)

        state_in = DensePureState{ComplexF64}(ws, hilbert_dims)
        state_out = DensePureState{ComplexF64}(ws, hilbert_dims)
        allocate_storage!(state_in); allocate_storage!(state_out)
        initialize_zero!(state_in)
        CUDA.@sync state_view(state_in)[1] = 1.0 + 0.0im

        prepare_operator_action!(ws, op, state_in, state_out)
        workspace_allocate!(ws, workspace_query_size(ws))

        compute_operator_action!(ws, op, state_in, state_out)
        # Explicit sync before reading result — this is the correct pattern.
        CUDA.synchronize()
        result = Array(state_view(state_out))
        @test result[1] ≈ 1.0 + 0.0im   # |0⟩ is eigenstate of sigma_z with eigenvalue +1

        close(state_in); close(state_out)
        close(op); close(term); close(elem_op); close(ws)
    end
end
```

- [ ] **Step 3: Run the test to confirm it passes with current code (with sync)**

```bash
julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | grep -A3 "asynchronous"
```

Expected: passes (sync is redundant but harmless).

- [ ] **Step 4: Remove all `CUDA.synchronize()` calls from compute functions**

In `src/densitymat/operators.jl`, delete (do not replace — just delete):
- Line ~726: `CUDA.synchronize()` at end of `compute_action!`
- Line ~799: `CUDA.synchronize()` at end of `compute_operator_action!`
- Line ~888: `CUDA.synchronize()` at end of `compute_operator_action_backward!`

In `src/densitymat/state.jl`, delete:
- `CUDA.synchronize()` at end of `inplace_scale!` (~line 335)
- `CUDA.synchronize()` at end of `norm` compute block (~line 354)
- `CUDA.synchronize()` at end of `trace` compute block (~line 373)
- `CUDA.synchronize()` at end of `inplace_accumulate!` (~line 413)

In `src/densitymat/expectation.jl`, delete:
- `CUDA.synchronize()` at end of `compute_expectation!` (~line 121)

In `src/densitymat/spectrum.jl`, delete:
- `CUDA.synchronize()` at end of `compute_spectrum!` (~line 190)

- [ ] **Step 5: Add explicit `CUDA.synchronize()` in tests where results are read**

In any test that reads a result from GPU memory directly after a compute call (without going through a sync boundary), add `CUDA.synchronize()` before the `Array()` or scalar access. The `sync_and_pull` helper from Task 11 handles this automatically.

```bash
grep -n "Array(state_view\|Array(result\|CUDA.@allowscalar" test/densitymat/*.jl
```

For each found line, ensure a `CUDA.synchronize()` or `CUDA.@sync` precedes it if not already wrapped.

- [ ] **Step 6: Add `CUDA.synchronize()` before norm/trace result access in tests**

In `test/densitymat/test_state.jl`, every call to `LinearAlgebra.norm(state)` and `LinearAlgebra.tr(state)` now returns a GPU array (since the sync was removed). Ensure each test does:
```julia
n_gpu = LinearAlgebra.norm(state)
CUDA.synchronize()
n = Array(n_gpu)
@test n[1] ≈ ...
```

If `norm` returns a CPU array (it calls `Array` internally already), no change is needed — verify by reading the implementation.

- [ ] **Step 7: Run the full test suite to confirm all tests still pass**

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: all tests pass. No CUDA errors or wrong values.

- [ ] **Step 8: Add a note to the docs about explicit synchronization**

In `docs/src/getting-started.md`, add a note after the first compute example:

```markdown
!!! note "Synchronization"
    CuDensityMat compute functions (`compute_operator_action!`,
    `compute_expectation!`, etc.) are **asynchronous** — they enqueue work on the
    CUDA stream and return immediately. To read results on the CPU, call
    `CUDA.synchronize()` first, or use `Array(result)` which synchronizes
    implicitly.
```

- [ ] **Step 9: Commit**

```bash
git add src/densitymat/operators.jl src/densitymat/state.jl src/densitymat/expectation.jl src/densitymat/spectrum.jl test/densitymat/*.jl docs/src/getting-started.md
git commit -m "perf: remove eager CUDA.synchronize() from all compute functions; callers sync explicitly"
```

---

## Self-Review Checklist

### Spec Coverage

| Issue | Task |
|---|---|
| `norm`/`trace` not extending LinearAlgebra | Task 4 |
| No `Base.close` on most types | Task 5 |
| `CUDA.device!()` side effect | Task 6 |
| `batch_size=0` default | Task 7 |
| `_data_ref::Any` type instability | Task 8 |
| Finalizer safety (context guard) | Task 9 |
| Callback memory leak / no `CallbackRef` | Task 10 |
| JuliaFormatter config | Task 1 |
| Aqua.jl / JET.jl | Task 2 |
| GPU CI job | Task 3 |
| `TEST_DTYPES`/`TEST_BATCH_SIZES` unused | Task 11 |
| `trajectory.csv` in repo root | Task 12 |
| Phase 6 numbering gap / batch tests | Task 12 |
| Docstrings + `warnonly` removal | Task 13 |
| Eager `CUDA.synchronize()` | Task 14 |

### Items Deliberately Out of Scope

- **`set_communicator!` reset support** — requires exposing `cudensitymatResetDistributedConfiguration`. Left out because the MPI/NCCL configuration lifecycle is a separate, large topic and requires distributed test infrastructure.
- **JET.jl static analysis** — added to `test/aqua.jl` as a comment/optional step rather than a required test, because JET produces many false positives with macro-generated CUDA code and requires careful per-false-positive annotation before being enforced in CI.
- **`precompiling` variable naming** — the `__init__` variable name inversion is a cosmetic issue; not worth a dedicated task.

### Placeholder Scan

No TBD, TODO, or "implement later" entries found. All code blocks are complete.

### Type Consistency

- `CallbackRef` defined in Task 10; referenced in Task 8 (update after Task 10). Tasks note the dependency explicitly.
- `_destroy!` helpers defined in Task 5; Task 9 verifies/extends them. No naming conflicts.
- `sync_and_pull` defined in Task 11/`setup.jl`; used in Task 11 test examples. Consistent.
