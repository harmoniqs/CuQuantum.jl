# Future Benchmarks

Additional benchmark targets to implement when time permits.
These are lower priority than the core 4 (QuantumToolbox.jl CPU/GPU, Python cuDensityMat, QuTiP).

## QuTiP-JAX (`qutip-jax`)

- **What**: QuTiP with JAX backend — replaces NumPy/SciPy internals with JAX arrays.
- **Why**: Tests whether JAX JIT compilation + GPU can compete with cuDensityMat's
  specialized tensor network kernels through a familiar QuTiP API.
- **Status**: Immature. May have install/compatibility issues.
- **Install**: `pip install qutip-jax` (requires `jax[cuda12]`)
- **API sketch**:
  ```python
  import qutip_jax  # registers JAX data layer with QuTiP
  import qutip
  # Convert operators to JAX backend
  H_jax = H.to("jax")
  rho0_jax = rho0.to("jax")
  c_ops_jax = [c.to("jax") for c in c_ops]
  result = qutip.mesolve(H_jax, rho0_jax, tlist, c_ops_jax, e_ops=e_ops)
  ```
- **Concerns**:
  - JIT compilation overhead on first call (exclude from timing)
  - JAX dispatches through XLA — unclear if sparse ops are efficient
  - May fall back to dense on GPU (which could actually be fast for moderate dims)
  - Version compatibility: qutip-jax tracks QuTiP 5.x, may lag behind

## DynamiQs

- **What**: JAX-native open quantum systems library. GPU-accelerated Lindblad solver.
- **Why**: Pure JAX approach — no cuDensityMat dependency. Tests whether generic JAX
  GPU parallelism can match NVIDIA's specialized kernels. Also supports automatic
  differentiation through the dynamics, which is directly relevant to Piccolo.jl.
- **Install**: `pip install dynamiqs jax[cuda12]`
- **API sketch**:
  ```python
  import dynamiqs as dq
  import jax.numpy as jnp

  # Operators
  N = 10
  a = dq.destroy(N)
  H = omega * dq.dag(a) @ a
  jump_ops = [jnp.sqrt(kappa) * a]

  # Time-dependent Hamiltonian
  # DynamiQs uses `dq.modulated(op, fn)` for time-dependent terms
  H_td = H + dq.modulated(a + dq.dag(a), lambda t: A * jnp.cos(omega_d * t))

  # Solve
  result = dq.mesolve(H_td, jump_ops, rho0, tsave)
  # result.states has shape (len(tsave), N, N)
  ```
- **Concerns**:
  - JAX JIT warmup is significant (minutes for large systems)
  - Memory layout: JAX uses row-major by default, may cause issues with large systems
  - Sparse support in JAX is experimental — dense matrices scale as D^2 memory
  - Multi-mode tensor product construction unclear (may need manual `jnp.kron`)
- **Interesting aspects**:
  - Native autodiff through `mesolve` — could benchmark gradient computation too
  - Batched simulations via `jax.vmap` — relevant for parameter sweeps
  - `dq.mesolve` uses Diffrax (JAX ODE solvers) internally

## Raw DifferentialEquations.jl + CUDA.jl

- **What**: Hand-rolled Lindblad ODE on GPU using Julia's DifferentialEquations.jl
  with `DiffEqGPU.jl` and `CuArray` state vectors.
- **Why**: Tests the "no specialized library" baseline — what performance do you get
  with a generic adaptive ODE solver operating on GPU arrays? This is what someone
  would write before cuDensityMat existed.
- **API sketch**:
  ```julia
  using OrdinaryDiffEq, CUDA, SparseArrays, LinearAlgebra

  # Build sparse Liouvillian on CPU, transfer to GPU
  L_cpu = build_liouvillian_superoperator(M, d)  # D^2 x D^2 sparse
  L_gpu = CUDA.CUSPARSE.CuSparseMatrixCSR(L_cpu)

  # Vectorized density matrix on GPU
  rho0_vec = CUDA.CuVector{ComplexF64}(vec(rho0))

  # ODE problem
  function lindblad_rhs!(du, u, p, t)
      mul!(du, L_gpu, u)
  end

  prob = ODEProblem(lindblad_rhs!, rho0_vec, (0.0, t_final))
  sol = solve(prob, Tsit5(); abstol=1e-8, reltol=1e-6, save_everystep=false)
  ```
- **Concerns**:
  - For time-dependent H, need to rebuild L_gpu at each step (expensive transfer)
  - Alternative: use `DiffEqGPU.EnsembleGPUKernel` for parameter sweeps
  - Memory: sparse superoperator on GPU uses cuSPARSE CSR — D^2 x D^2 matrix
    For M=8 (D=6561): CSR with ~50M nnz = ~1.2 GB. Feasible on A100.
    For M=9 (D=19683): CSR would be ~120 GB. Not feasible.
  - This approach scales worse than cuDensityMat's tensor network contraction
    because it materializes the full superoperator
- **Interesting aspects**:
  - Direct comparison: does cuDensityMat's tensor decomposition beat brute-force
    sparse GPU SpMV? (Almost certainly yes at large M, but where's the crossover?)
  - Could also test dense GPU: `CuMatrix` + `mul!` for small D
  - `DiffEqGPU.EnsembleGPUKernel` could be interesting for batched parameter
    optimization — relevant to Piccolo.jl's use case

## Notes on Fair Comparison

All future benchmarks should follow the same protocol as the core benchmarks:

1. **Same physics**: M coupled cavities, Fock truncation d=3, Kerr + coupling + photon loss
2. **Same initial state**: |1,0,...,0><1,0,...,0|
3. **Same time integration**: RK4 with dt=0.01, 100 steps where possible.
   For library-integrated solvers (mesolve), use matched tolerances (atol=1e-8, rtol=1e-6)
4. **Timing**: wall-clock time for the full simulation (setup excluded).
   Also report single L[rho] action time where applicable.
5. **Warmup**: exclude JIT/compilation from timing (1-3 warmup calls)
6. **Validation**: compare final rho diagonal populations against CPU sparse reference
7. **System sizes**: M = 2, 4, 6, 8 (skip 8 if OOM)
