#!/usr/bin/env bash
# =============================================================================
# Run all comparison benchmarks and aggregate results
# =============================================================================
#
# Prerequisites:
#   - Julia 1.12+ with CuQuantum.jl, QuantumToolbox.jl installed
#   - Python 3.10+ with qutip, cuquantum-python, cupy-cuda12x installed
#   - NVIDIA GPU with CUDA 12.x
#
# Usage:
#   cd /path/to/cuQuantum.jl
#   bash benchmark/comparison/run_all.sh
#
# Output: benchmark/comparison/results_combined.csv
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR"

echo "=============================================================================="
echo "Comparison Benchmarks: CuQuantum.jl vs QuantumToolbox.jl vs QuTiP vs Python"
echo "=============================================================================="
echo ""
echo "Repo:    $REPO_DIR"
echo "Results: $RESULTS_DIR"
echo ""

# Detect Julia
JULIA="${JULIA:-julia}"
if ! command -v "$JULIA" &>/dev/null; then
    # Try common install locations
    for candidate in "$HOME/julia-1.12.5/bin/julia" "$HOME/julia/bin/julia"; do
        if [[ -x "$candidate" ]]; then
            JULIA="$candidate"
            break
        fi
    done
fi
echo "Julia:   $($JULIA --version 2>/dev/null || echo 'not found')"

# Detect Python
PYTHON="${PYTHON:-python3}"
echo "Python:  $($PYTHON --version 2>/dev/null || echo 'not found')"
echo ""

# GPU info
if command -v nvidia-smi &>/dev/null; then
    echo "GPU:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""
fi

FAILED=()

# ─── 1. CuQuantum.jl (our package) ──────────────────────────────────────────
echo "━━━ [1/4] CuQuantum.jl (cuDensityMat GPU) ━━━"
if "$JULIA" --project="$REPO_DIR" -e "using CuQuantum" &>/dev/null 2>&1; then
    "$JULIA" --project="$REPO_DIR" "$SCRIPT_DIR/bench_cuquantum_jl.jl" || FAILED+=("CuQuantum.jl")
else
    echo "SKIP: CuQuantum.jl not available (run: julia --project=$REPO_DIR -e 'using Pkg; Pkg.instantiate()')"
    FAILED+=("CuQuantum.jl")
fi
echo ""

# ─── 2. QuantumToolbox.jl CPU ────────────────────────────────────────────────
echo "━━━ [2/4] QuantumToolbox.jl CPU ━━━"
# Install QuantumToolbox if not present
"$JULIA" --project="$SCRIPT_DIR" -e '
    using Pkg
    if !haskey(Pkg.project().dependencies, "QuantumToolbox")
        Pkg.add(["QuantumToolbox", "OrdinaryDiffEq", "CUDA"])
    end
    Pkg.instantiate()
' 2>/dev/null || true

if "$JULIA" --project="$SCRIPT_DIR" -e "using QuantumToolbox" &>/dev/null 2>&1; then
    "$JULIA" --project="$SCRIPT_DIR" "$SCRIPT_DIR/bench_quantumtoolbox_cpu.jl" || FAILED+=("QuantumToolbox-CPU")
else
    echo "SKIP: QuantumToolbox.jl not available"
    FAILED+=("QuantumToolbox-CPU")
fi
echo ""

# ─── 3. QuantumToolbox.jl GPU ────────────────────────────────────────────────
echo "━━━ [3/4] QuantumToolbox.jl GPU ━━━"
if "$JULIA" --project="$SCRIPT_DIR" -e "using QuantumToolbox, CUDA; @assert CUDA.functional()" &>/dev/null 2>&1; then
    "$JULIA" --project="$SCRIPT_DIR" "$SCRIPT_DIR/bench_quantumtoolbox_gpu.jl" || FAILED+=("QuantumToolbox-GPU")
else
    echo "SKIP: CUDA not functional for QuantumToolbox.jl GPU benchmark"
    FAILED+=("QuantumToolbox-GPU")
fi
echo ""

# ─── 4. QuTiP (Python) ──────────────────────────────────────────────────────
echo "━━━ [4/4] QuTiP CPU ━━━"
if "$PYTHON" -c "import qutip" &>/dev/null 2>&1; then
    "$PYTHON" "$SCRIPT_DIR/bench_qutip.py" || FAILED+=("QuTiP")
else
    echo "SKIP: QuTiP not installed (run: pip install qutip numpy psutil)"
    FAILED+=("QuTiP")
fi
echo ""

# ─── 5. Python cuDensityMat (optional — needs cuquantum-python) ─────────────
echo "━━━ [5/4] Python cuDensityMat (bonus) ━━━"
if "$PYTHON" -c "from cuquantum.densitymat import WorkStream" &>/dev/null 2>&1; then
    "$PYTHON" "$SCRIPT_DIR/bench_python_cudensitymat.py" || FAILED+=("Python-cuDensityMat")
else
    echo "SKIP: cuquantum-python not installed (run: pip install cuquantum-python cupy-cuda12x)"
    FAILED+=("Python-cuDensityMat")
fi
echo ""

# ─── Aggregate results ──────────────────────────────────────────────────────
echo "━━━ Aggregating results ━━━"
COMBINED="$RESULTS_DIR/results_combined.csv"
echo "framework,backend,M,D,rho_elements,wall_time_s,n_steps,dt" > "$COMBINED"

for csv_file in "$RESULTS_DIR"/results_*.csv; do
    [[ "$(basename "$csv_file")" == "results_combined.csv" ]] && continue
    [[ -f "$csv_file" ]] || continue
    # Skip header, append data
    tail -n +2 "$csv_file" >> "$COMBINED"
    echo "  Added: $(basename "$csv_file")"
done

echo ""
echo "Combined results: $COMBINED"
echo ""

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "WARNING: The following benchmarks failed or were skipped:"
    for f in "${FAILED[@]}"; do
        echo "  - $f"
    done
fi

echo ""
echo "Done."
