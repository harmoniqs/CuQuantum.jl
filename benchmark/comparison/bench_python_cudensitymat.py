#!/usr/bin/env python3
"""
Benchmark: Python cuDensityMat (CuPy backend)
=============================================================================

Solves the same dual-rail cavity Lindblad problem using NVIDIA's Python
cuDensityMat bindings. Same physics, same system sizes, same RK4 integrator
as the Julia CuQuantum.jl benchmark.

System: M coupled cavities, Fock truncation d=3
  H = sum_m chi * n_m(n_m-1) + sum_{n!=m} kappa * a_n^dag a_m
  L[rho] = -i[H,rho] + gamma * sum_m (a_m rho a_m^dag - 1/2 {n_m, rho})

Follows the pattern from NVIDIA's dense_operator_example.py:
  - DenseOperator for single-mode matrices
  - tensor_product with explicit duality bools for sandwich terms
  - Single Operator with all Liouvillian terms

Usage:
  pip install cuquantum-python cupy-cuda12x numpy
  python benchmark/comparison/bench_python_cudensitymat.py

Output: results_python_cudensitymat.csv
"""

import csv
import os
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np
from cuquantum.densitymat import (
    DenseMixedState,
    DenseOperator,
    Operator,
    WorkStream,
    tensor_product,
)


# =============================================================================
# Problem construction
# =============================================================================


def build_system(M: int, d: int):
    """
    Build the M-cavity Lindblad system using Python cuDensityMat.

    Constructs a single Operator containing all Liouvillian terms:
      L[rho] = -i[H, rho] + gamma * sum_m (a_m rho a_m^dag - 1/2 {n_m, rho})

    Following NVIDIA's dense_operator_example.py pattern:
      - Sandwich: tensor_product((a, [m], [False]), (a.dag(), [m], [True]))
      - Anticommutator ket: tensor_product((n, [m], [False]))
      - Anticommutator bra: tensor_product((n, [m], [True]))

    Returns (ctx, liouvillian, rho, k1, k2, k3, k4, rho_tmp).
    """
    dtype = "complex128"
    chi = 2 * np.pi * 0.2
    kappa0 = 2 * np.pi * 0.1
    gamma = 0.01

    hilbert_dims = (d,) * M
    batch_size = 1

    # --- Single-cavity operator matrices ---
    a_mat = np.zeros((d, d), dtype=dtype)
    for n in range(1, d):
        a_mat[n - 1, n] = np.sqrt(n)
    a_dag_mat = a_mat.conj().T
    n_mat = a_dag_mat @ a_mat
    kerr_mat = n_mat @ (n_mat - np.eye(d, dtype=dtype))

    # --- Elementary operators (single-mode, d x d) ---
    a_op = DenseOperator(a_mat)
    a_dag_op = DenseOperator(a_dag_mat)
    n_op = DenseOperator(n_mat)
    kerr_op = DenseOperator(kerr_mat)

    # --- Hamiltonian term ---
    # Kerr: sum_m chi * n_m(n_m-1)
    H_term = tensor_product((kerr_op, [0]), coeff=chi)
    for m in range(1, M):
        H_term = H_term + tensor_product((kerr_op, [m]), coeff=chi)

    # Coupling: sum_{n!=m} kappa0 * a_n^dag a_m
    # Use separate single-mode operators per NVIDIA's tensor_product API:
    # tensor_product((a_dag_op, [n]), (a_op, [m])) for each pair
    for ni in range(M):
        for mi in range(M):
            if ni == mi:
                continue
            H_term = H_term + tensor_product(
                (a_dag_op, [ni]), (a_op, [mi]), coeff=kappa0
            )

    # --- Dissipation terms (per NVIDIA dense_operator_example.py pattern) ---
    # For each cavity m:
    #   gamma * (a_m rho a_m^dag - 1/2 n_m rho - 1/2 rho n_m)
    #
    # l_dag_l = a_dag @ a = n_op
    l_dag_l = DenseOperator(n_mat)  # = a^dag a = number operator

    diss_terms = []
    for m in range(M):
        L_term = (
            # Sandwich: a_m rho a_m^dag
            tensor_product(
                (a_op, [m], [False]),  # a acts on ket side
                (a_dag_op, [m], [True]),  # a^dag acts on bra side
            )
            # Anticommutator: -1/2 n_m rho (ket side)
            + tensor_product((-0.5 * l_dag_l, [m], [False]))
            # Anticommutator: -1/2 rho n_m (bra side)
            + tensor_product((-0.5 * l_dag_l, [m], [True]))
        )
        diss_terms.append(gamma * L_term)

    # --- Assemble single Liouvillian Operator ---
    # Hamiltonian part: -i[H, rho] = -iH rho + i rho H
    # Dissipation part: all diss_terms (already include gamma)
    operator_args = [
        (H_term, -1j, False),  # -i H rho (ket side)
        (H_term, +1j, True),  # +i rho H (bra side)
    ]
    for L_term in diss_terms:
        operator_args.append((L_term,))

    liouvillian = Operator(hilbert_dims, *operator_args)

    # --- State setup ---
    ctx = WorkStream()
    rho = DenseMixedState(ctx, hilbert_dims, batch_size, dtype)
    rho.allocate_storage()

    # Initialize to |1,0,...,0><1,0,...,0|
    D = d**M
    rho_cpu = np.zeros(D * D, dtype=dtype)
    # |1,0,...,0> is index 1 in the Fock basis (0-indexed)
    rho_cpu[1 + D * 1] = 1.0
    rho.storage[:] = cp.asarray(rho_cpu.reshape(rho.storage.shape), order="F")

    # RK4 scratch states
    k1 = rho.clone(cp.zeros_like(rho.storage, order="F"))
    k2 = rho.clone(cp.zeros_like(rho.storage, order="F"))
    k3 = rho.clone(cp.zeros_like(rho.storage, order="F"))
    k4 = rho.clone(cp.zeros_like(rho.storage, order="F"))
    rho_tmp = rho.clone(cp.zeros_like(rho.storage, order="F"))

    # Prepare action (single prepare for the unified operator)
    liouvillian.prepare_action(ctx, rho)

    return ctx, liouvillian, rho, k1, k2, k3, k4, rho_tmp


# =============================================================================
# Benchmark: single L[rho] action
# =============================================================================


def bench_single_action(M: int, d: int, n_warmup: int = 5, n_trials: int = 50):
    """Time a single Liouvillian action. Returns (median_ms, min_ms, max_ms)."""
    ctx, liouvillian, rho, k1, *_ = build_system(M, d)

    # Warmup
    for _ in range(n_warmup):
        k1.storage[:] = 0
        liouvillian.compute_action(0.0, None, rho, k1)
    cp.cuda.get_current_stream().synchronize()

    # Timed runs using CUDA events for precise GPU timing
    start_evt = cp.cuda.Event()
    end_evt = cp.cuda.Event()

    times = []
    for _ in range(n_trials):
        k1.storage[:] = 0
        cp.cuda.get_current_stream().synchronize()
        start_evt.record()
        liouvillian.compute_action(0.0, None, rho, k1)
        end_evt.record()
        end_evt.synchronize()
        times.append(cp.cuda.get_elapsed_time(start_evt, end_evt))

    times.sort()
    median_t = times[len(times) // 2]
    return median_t, times[0], times[-1]


# =============================================================================
# Benchmark: full RK4 simulation
# =============================================================================


def bench_rk4_simulation(M: int, d: int, n_steps: int = 100, dt: float = 0.01):
    """Run full RK4 Lindblad simulation. Returns (wall_time_s, trace)."""
    D = d**M
    ctx, liouvillian, rho, k1, k2, k3, k4, rho_tmp = build_system(M, d)

    def eval_L(k_out, rho_in, t):
        k_out.storage[:] = 0
        liouvillian.compute_action(t, None, rho_in, k_out)

    def copy_state(dst, src):
        dst.storage[:] = src.storage

    def accumulate(dst, src, alpha):
        dst.storage[:] += alpha * src.storage

    # Warmup: one full step
    eval_L(k1, rho, 0.0)
    cp.cuda.get_current_stream().synchronize()

    # Reset state
    rho_cpu = np.zeros(D * D, dtype="complex128")
    rho_cpu[1 + D * 1] = 1.0
    rho.storage[:] = cp.asarray(rho_cpu.reshape(rho.storage.shape), order="F")
    cp.cuda.get_current_stream().synchronize()

    # Timed RK4 loop
    t_start = time.perf_counter()
    t = 0.0
    for step in range(n_steps):
        # k1 = L(rho, t)
        eval_L(k1, rho, t)

        # rho_tmp = rho + dt/2 * k1
        copy_state(rho_tmp, rho)
        accumulate(rho_tmp, k1, dt / 2)
        eval_L(k2, rho_tmp, t + dt / 2)

        # rho_tmp = rho + dt/2 * k2
        copy_state(rho_tmp, rho)
        accumulate(rho_tmp, k2, dt / 2)
        eval_L(k3, rho_tmp, t + dt / 2)

        # rho_tmp = rho + dt * k3
        copy_state(rho_tmp, rho)
        accumulate(rho_tmp, k3, dt)
        eval_L(k4, rho_tmp, t + dt)

        # rho += dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        accumulate(rho, k1, dt / 6)
        accumulate(rho, k2, dt / 3)
        accumulate(rho, k3, dt / 3)
        accumulate(rho, k4, dt / 6)

        t += dt

    cp.cuda.get_current_stream().synchronize()
    t_end = time.perf_counter()
    wall_time = t_end - t_start

    # Compute trace
    rho_final = rho.storage.get().ravel()
    trace = sum(rho_final[i + D * i] for i in range(D)).real

    return wall_time, trace


# =============================================================================
# Main
# =============================================================================


def main():
    d = 3
    M_values = [2, 4, 6]
    # NOTE: M=8 skipped — Python operator construction is O(M^2) and takes
    # >10 minutes due to OperatorTerm.__add__ deep copies for 56 coupling products.
    # This is a Python wrapper overhead issue, not a cuDensityMat library issue.

    # Check GPU memory
    free_mem = cp.cuda.Device().mem_info[0]
    total_mem = cp.cuda.Device().mem_info[1]

    n_steps = 100
    dt = 0.01

    gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print("=" * 78)
    print("Benchmark: Python cuDensityMat (CuPy backend)")
    print(f"  System: M coupled cavities, d={d} Fock truncation")
    print(
        f"  GPU: {gpu_name} ({free_mem / 1e9:.1f} GB free / {total_mem / 1e9:.1f} GB total)"
    )
    print(f"  Time integration: RK4 fixed-step (dt={dt})")
    print(f"  Simulation: {n_steps} steps, t_final={n_steps * dt}")
    print("=" * 78)
    print()

    # --- Single action benchmark ---
    print("--- Single Liouvillian action L[rho] ---")
    for M in M_values:
        D = d**M
        try:
            med, mn, mx = bench_single_action(M, d)
            print(f"  M={M} (D={D}): {med:.3f} ms (min={mn:.3f}, max={mx:.3f})")
        except Exception as e:
            print(f"  M={M} (D={D}): FAILED: {e}")
    print()

    # --- Full RK4 benchmark ---
    print(f"--- Full RK4 simulation ({n_steps} steps) ---")
    results = []
    for M in M_values:
        D = d**M
        try:
            wall_time, trace = bench_rk4_simulation(M, d, n_steps=n_steps, dt=dt)
            print(f"  M={M} (D={D}): {wall_time:.4f} s  (Tr(rho)={trace:.6f})")
            results.append({"M": M, "D": D, "wall_time_s": wall_time, "trace": trace})
        except Exception as e:
            print(f"  M={M} (D={D}): FAILED: {e}")
            results.append({"M": M, "D": D, "wall_time_s": float("nan"), "trace": 0})
    print()

    # --- Save CSV ---
    csv_file = Path(__file__).parent / "results_python_cudensitymat.csv"
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
                    "Python-cuDensityMat",
                    f"GPU-{gpu_name}",
                    r["M"],
                    r["D"],
                    r["D"] ** 2,
                    r["wall_time_s"],
                    n_steps,
                    dt,
                ]
            )
    print(f"Results saved to {csv_file}")


if __name__ == "__main__":
    main()
