#!/usr/bin/env python3
"""
Benchmark: JAX cuDensityMat extension
=============================================================================

Same dual-rail cavity Lindblad problem using NVIDIA's JAX extension for
cuDensityMat (cuquantum.densitymat.jax). Same physics, same system sizes,
same RK4 integrator as all other benchmarks.

This measures the JAX wrapper overhead vs CuPy and Julia wrappers, and
tests jax.jit compilation of the operator action.

System: M coupled cavities, Fock truncation d=3
  H = sum_m chi * n_m(n_m-1) + sum_{n!=m} kappa * a_n^dag a_m
  L[rho] = -i[H,rho] + gamma * sum_m (a_m rho a_m^dag - 1/2 {n_m, rho})

Usage:
  pip install cuquantum-python-jax jax[cuda12]
  python benchmark/comparison/bench_jax_cudensitymat.py

Output: results_jax_cudensitymat.csv
"""

import csv
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import numpy as np
from cuquantum.densitymat.jax import (
    ElementaryOperator,
    Operator,
    OperatorTerm,
    operator_action,
)


# =============================================================================
# Problem construction
# =============================================================================


def build_system(M: int, d: int):
    """
    Build the M-cavity Lindblad system using JAX cuDensityMat extension.

    Uses the cuquantum.densitymat.jax API:
      - ElementaryOperator for single-mode matrices
      - OperatorTerm.append() with modes and duals for tensor products
      - Operator.append() with dual flag for Liouvillian assembly

    Returns (liouvillian, rho0) where rho0 is a jax.Array.
    """
    dtype = jnp.complex128
    chi = 2 * np.pi * 0.2
    kappa0 = 2 * np.pi * 0.1
    gamma = 0.01

    hilbert_dims = (d,) * M

    # --- Single-cavity operator matrices ---
    a_mat = np.zeros((d, d), dtype=np.complex128)
    for n in range(1, d):
        a_mat[n - 1, n] = np.sqrt(n)
    a_dag_mat = a_mat.conj().T
    n_mat = a_dag_mat @ a_mat
    kerr_mat = n_mat @ (n_mat - np.eye(d, dtype=np.complex128))

    # --- JAX ElementaryOperators ---
    a_op = ElementaryOperator(jnp.array(a_mat))
    a_dag_op = ElementaryOperator(jnp.array(a_dag_mat))
    n_op = ElementaryOperator(jnp.array(n_mat))
    kerr_op = ElementaryOperator(jnp.array(kerr_mat))

    # --- Hamiltonian term ---
    H_term = OperatorTerm(hilbert_dims)

    # Kerr: sum_m chi * n_m(n_m-1)
    for m in range(M):
        H_term.append(
            [kerr_op],
            modes=[m],
            duals=[False],
            coeff=chi,
        )

    # Coupling: sum_{n!=m} kappa0 * a_n^dag a_m
    for ni in range(M):
        for mi in range(M):
            if ni == mi:
                continue
            H_term.append(
                [a_dag_op, a_op],
                modes=[ni, mi],
                duals=[False, False],
                coeff=kappa0,
            )

    # --- Dissipation term ---
    # For each cavity m:
    #   gamma * (a_m rho a_m^dag - 1/2 n_m rho - 1/2 rho n_m)
    diss_term = OperatorTerm(hilbert_dims)
    for m in range(M):
        # Sandwich: a_m rho a_m^dag (ket-side a, bra-side a^dag)
        diss_term.append(
            [a_op, a_dag_op],
            modes=[m, m],
            duals=[False, True],
            coeff=gamma,
        )
        # Anticommutator ket: -gamma/2 * n_m * rho
        diss_term.append(
            [n_op],
            modes=[m],
            duals=[False],
            coeff=-gamma / 2,
        )
        # Anticommutator bra: -gamma/2 * rho * n_m
        diss_term.append(
            [n_op],
            modes=[m],
            duals=[True],
            coeff=-gamma / 2,
        )

    # --- Assemble Liouvillian ---
    liouvillian = Operator(hilbert_dims)
    # Hamiltonian: -i[H, rho] = -iH*rho + i*rho*H
    liouvillian.append(H_term, dual=False, coeff=-1j)
    liouvillian.append(H_term, dual=True, coeff=+1j)
    # Dissipation
    liouvillian.append(diss_term, dual=False, coeff=1.0)

    # --- Initial state: |1,0,...,0><1,0,...,0| ---
    # Shape: (d, d, ..., d, d, ..., d) = 2*M dimensions (ket + bra)
    D = d**M
    rho0_flat = np.zeros(D * D, dtype=np.complex128)
    rho0_flat[1 + D * 1] = 1.0  # |1,0,...,0><1,0,...,0|
    rho0 = jnp.array(rho0_flat.reshape(hilbert_dims + hilbert_dims))

    return liouvillian, rho0


# =============================================================================
# Benchmark: single L[rho] action
# =============================================================================


def bench_single_action(M: int, d: int, n_warmup: int = 5, n_trials: int = 50):
    """Time a single Liouvillian action. Returns (median_ms, min_ms, max_ms)."""
    liouvillian, rho0 = build_system(M, d)

    # JIT-compiled operator action
    @jax.jit
    def apply_L(state):
        return operator_action(liouvillian, 0.0, state)

    # Warmup (triggers JIT compilation)
    for _ in range(n_warmup):
        _ = apply_L(rho0).block_until_ready()

    # Timed runs
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        _ = apply_L(rho0).block_until_ready()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    times.sort()
    median_t = times[len(times) // 2]
    return median_t, times[0], times[-1]


# =============================================================================
# Benchmark: full RK4 simulation
# =============================================================================


def bench_rk4_simulation(M: int, d: int, n_steps: int = 100, dt: float = 0.01):
    """Run full RK4 Lindblad simulation. Returns (wall_time_s, trace)."""
    D = d**M
    liouvillian, rho0 = build_system(M, d)

    # Non-JIT version (JIT of the full loop with lax.fori_loop is complex
    # due to operator_action side effects; benchmark the per-step call instead)
    def eval_L(state, t):
        return operator_action(liouvillian, t, state)

    # Warmup: trigger JIT for eval_L
    _ = jax.jit(lambda s: eval_L(s, 0.0))(rho0).block_until_ready()

    # JIT the single-step function
    @jax.jit
    def eval_L_jit(state):
        return operator_action(liouvillian, 0.0, state)

    # Warmup the JIT
    _ = eval_L_jit(rho0).block_until_ready()

    # Reset state
    rho = rho0

    # Timed RK4 loop
    t_start = time.perf_counter()
    t = 0.0
    for step in range(n_steps):
        # For static Hamiltonian, time argument doesn't matter
        k1 = eval_L_jit(rho)
        k2 = eval_L_jit(rho + (dt / 2) * k1)
        k3 = eval_L_jit(rho + (dt / 2) * k2)
        k4 = eval_L_jit(rho + dt * k3)
        rho = rho + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Block to ensure computation is complete before next step
        rho.block_until_ready()
        t += dt

    rho.block_until_ready()
    t_end = time.perf_counter()
    wall_time = t_end - t_start

    # Compute trace
    rho_np = np.array(rho).reshape(D, D)
    trace = np.trace(rho_np).real

    return wall_time, trace


# =============================================================================
# Main
# =============================================================================


def main():
    d = 3
    M_values = [2, 4, 6]
    # NOTE: M=8 skipped — operator construction overhead is O(M^2) and
    # JAX tracing adds additional overhead on top of cuDensityMat.

    n_steps = 100
    dt = 0.01

    gpu_name = "unknown"
    free_mem = 0
    total_mem = 0
    try:
        devices = jax.devices("gpu")
        if devices:
            gpu_name = devices[0].device_kind
        # Get memory info via CuPy if available
        try:
            import cupy as cp

            free_mem = cp.cuda.Device().mem_info[0]
            total_mem = cp.cuda.Device().mem_info[1]
        except ImportError:
            pass
    except Exception:
        pass

    print("=" * 78)
    print("Benchmark: JAX cuDensityMat extension (cuquantum.densitymat.jax)")
    print("  System: M coupled cavities, d=%d Fock truncation" % d)
    print("  GPU: %s" % gpu_name)
    if total_mem > 0:
        print(
            "  Memory: %.1f GB free / %.1f GB total" % (free_mem / 1e9, total_mem / 1e9)
        )
    print("  JAX version: %s" % jax.__version__)
    print("  Time integration: RK4 fixed-step (dt=%s)" % dt)
    print("  Simulation: %d steps, t_final=%s" % (n_steps, n_steps * dt))
    print("=" * 78)
    print()

    # --- Single action benchmark ---
    print("--- Single Liouvillian action L[rho] (JIT-compiled) ---")
    for M in M_values:
        D = d**M
        try:
            med, mn, mx = bench_single_action(M, d)
            print("  M=%d (D=%d): %.3f ms (min=%.3f, max=%.3f)" % (M, D, med, mn, mx))
        except Exception as e:
            print("  M=%d (D=%d): FAILED: %s" % (M, D, e))
    print()

    # --- Full RK4 benchmark ---
    print("--- Full RK4 simulation (%d steps) ---" % n_steps)
    results = []
    for M in M_values:
        D = d**M
        try:
            wall_time, trace = bench_rk4_simulation(M, d, n_steps=n_steps, dt=dt)
            print("  M=%d (D=%d): %.4f s  (Tr(rho)=%.6f)" % (M, D, wall_time, trace))
            results.append({"M": M, "D": D, "wall_time_s": wall_time, "trace": trace})
        except Exception as e:
            print("  M=%d (D=%d): FAILED: %s" % (M, D, e))
            results.append({"M": M, "D": D, "wall_time_s": float("nan"), "trace": 0})
    print()

    # --- Save CSV ---
    csv_file = Path(__file__).parent / "results_jax_cudensitymat.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "framework",
                "backend",
                "M",
                "D",
                "rho_elements",
                "wall_time_s",
                "n_steps",
                "dt",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    "JAX-cuDensityMat",
                    "GPU-%s" % gpu_name,
                    r["M"],
                    r["D"],
                    r["D"] ** 2,
                    r["wall_time_s"],
                    n_steps,
                    dt,
                ]
            )
    print("Results saved to %s" % csv_file)


if __name__ == "__main__":
    main()
