#!/usr/bin/env python3
"""
Benchmark: QuTiP CPU mesolve
=============================================================================

Solves the same dual-rail cavity Lindblad problem as the cuDensityMat
benchmark using QuTiP's mesolve on CPU.

System: M coupled cavities, Fock truncation d=3
  H = sum_m chi * n_m(n_m-1) + sum_{n!=m} kappa * a_n^dag a_m
  L[rho] = -i[H,rho] + gamma * sum_m (a_m rho a_m^dag - 1/2 {n_m, rho})

Usage:
  pip install qutip numpy
  python benchmark/comparison/bench_qutip.py

Output: results_qutip.csv
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import qutip

# =============================================================================
# Problem construction
# =============================================================================


def build_system(M: int, d: int):
    """
    Build the M-cavity Lindblad problem in QuTiP.
    Returns (H, c_ops, rho0, e_ops).
    """
    chi = 2 * np.pi * 0.2
    kappa0 = 2 * np.pi * 0.1
    gamma = 0.01

    a_single = qutip.destroy(d)
    n_single = qutip.num(d)
    kerr_single = n_single * (n_single - qutip.qeye(d))
    eye_d = qutip.qeye(d)

    def embed(op, mode):
        ops = [eye_d] * M
        ops[mode] = op
        return qutip.tensor(ops)

    a_ops = [embed(a_single, m) for m in range(M)]
    n_ops = [embed(n_single, m) for m in range(M)]

    # Hamiltonian
    H = sum(chi * embed(kerr_single, m) for m in range(M))
    for n in range(M):
        for m in range(M):
            if n != m:
                H += kappa0 * (a_ops[n].dag() * a_ops[m])

    # Collapse operators
    c_ops = [np.sqrt(gamma) * a_ops[m] for m in range(M)]

    # Initial state: |1,0,...,0><1,0,...,0|
    fock_list = [qutip.basis(d, 0)] * M
    fock_list[0] = qutip.basis(d, 1)
    psi0 = qutip.tensor(fock_list)
    rho0 = psi0 * psi0.dag()

    # Expectation operators
    e_ops = [n_ops[m] for m in range(M)]

    return H, c_ops, rho0, e_ops


# =============================================================================
# Benchmark: full mesolve
# =============================================================================


def bench_mesolve(M: int, d: int, n_steps: int = 100, dt: float = 0.01):
    """Run mesolve. Returns (wall_time_seconds, final_populations)."""
    D = d**M
    t_final = n_steps * dt
    tlist = np.linspace(0, t_final, n_steps + 1)

    H, c_ops, rho0, e_ops = build_system(M, d)

    options = {
        "method": "adams",
        "atol": 1e-8,
        "rtol": 1e-6,
        "nsteps": 10000,
        "normalize_output": False,
        "store_states": False,
        "store_final_state": True,
        "progress_bar": "",
    }

    # Warmup
    _ = qutip.mesolve(H, rho0, tlist[:3], c_ops, e_ops=e_ops, options=options)

    # Timed run
    t_start = time.perf_counter()
    result = qutip.mesolve(H, rho0, tlist, c_ops, e_ops=e_ops, options=options)
    t_end = time.perf_counter()
    wall_time = t_end - t_start

    final_pops = [np.real(result.expect[m][-1]) for m in range(M)]
    return wall_time, final_pops


# =============================================================================
# Benchmark: single L*vec(rho) action
# =============================================================================


def bench_single_action(M: int, d: int, n_warmup: int = 5, n_trials: int = 50):
    """Time a single Liouvillian SpMV. Returns (median_ms, min_ms, max_ms)."""
    D = d**M

    H, c_ops, rho0, _ = build_system(M, d)

    # Build Liouvillian superoperator
    L = qutip.liouvillian(H, c_ops)

    # Vectorize initial state
    rho_vec = qutip.operator_to_vector(rho0)
    L_data = L.full()  # dense for fair timing (avoids scipy sparse overhead variance)
    rho_data = rho_vec.full().ravel()

    # For large systems, use sparse
    if D > 100:
        from scipy.sparse import csr_matrix

        L_data = csr_matrix(L.full())

    # Warmup
    for _ in range(n_warmup):
        _ = L_data @ rho_data

    # Timed runs
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        _ = L_data @ rho_data
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    times.sort()
    median_t = times[len(times) // 2]
    return median_t, times[0], times[-1]


# =============================================================================
# Main
# =============================================================================


def main():
    d = 3
    M_values = [2, 4, 6]

    # Check memory for M=8
    import psutil

    total_mem = psutil.virtual_memory().total
    for M_try in [8]:
        D_try = d**M_try
        # Liouvillian is D^2 x D^2 sparse, but scipy needs significant memory
        mem_est = M_try * d**2 * D_try**2 * 24  # rough CSR estimate
        if mem_est < total_mem * 0.5:
            M_values.append(M_try)
        else:
            print(
                f"Skipping M={M_try} (D={D_try}): "
                f"estimated memory ~{mem_est / 1e9:.1f} GB"
            )

    n_steps = 100
    dt = 0.01

    print("=" * 78)
    print(f"Benchmark: QuTiP {qutip.__version__} CPU mesolve")
    print(f"  System: M coupled cavities, d={d} Fock truncation")
    print(f"  Time integration: Adams adaptive (atol=1e-8, rtol=1e-6)")
    print(f"  Simulation: {n_steps} steps, dt={dt}, t_final={n_steps * dt}")
    cpu_count = (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else "?"
    )
    print(f"  CPU: {cpu_count} cores, {total_mem / 1e9:.1f} GB RAM")
    print("=" * 78)
    print()

    # --- Single action benchmark ---
    print("--- Single Liouvillian action L*vec(rho) ---")
    for M in M_values:
        D = d**M
        try:
            med, mn, mx = bench_single_action(M, d)
            print(
                f"  M={M} (D={D}, Liouv={D**2}x{D**2}): "
                f"{med:.3f} ms (min={mn:.3f}, max={mx:.3f})"
            )
        except Exception as e:
            print(f"  M={M} (D={D}): FAILED: {e}")
    print()

    # --- Full mesolve benchmark ---
    print(f"--- Full mesolve simulation ({n_steps} steps) ---")
    results = []
    for M in M_values:
        D = d**M
        try:
            wall_time, pops = bench_mesolve(M, d, n_steps=n_steps, dt=dt)
            pops_str = ", ".join(f"{p:.4f}" for p in pops)
            print(f"  M={M} (D={D}): {wall_time:.4f} s")
            print(f"    Final <n>: {pops_str}")
            results.append({"M": M, "D": D, "wall_time_s": wall_time, "pops": pops})
        except Exception as e:
            print(f"  M={M} (D={D}): FAILED: {e}")
            results.append({"M": M, "D": D, "wall_time_s": float("nan"), "pops": []})
    print()

    # --- Save CSV ---
    csv_file = Path(__file__).parent / "results_qutip.csv"
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
                    "QuTiP",
                    "CPU",
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
    import os

    main()
