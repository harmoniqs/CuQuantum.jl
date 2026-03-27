# Cross-Framework Lindblad Benchmarks

Comparison of CuQuantum.jl against QuantumToolbox.jl, QuTiP, and NVIDIA's own Python cuDensityMat wrappers (CuPy and JAX).

## Benchmark Problem

M coupled cavities with Fock truncation d=3:

- **Hamiltonian**: Kerr nonlinearity + all-to-all hopping
  - `H = Σ_m χ n_m(n_m-1) + Σ_{n≠m} κ a_n†a_m`
  - `χ = 2π × 0.2`, `κ = 2π × 0.1`
- **Dissipation**: photon loss on each cavity, `γ = 0.01`
- **Initial state**: `|1,0,...,0⟩⟨1,0,...,0|`
- **Integration**: 100 steps, dt=0.01, t_final=1.0
- **System sizes**: M = 2, 4, 6

## Scripts

| Script | Framework | Backend | Integration |
|--------|-----------|---------|-------------|
| `bench_cuquantum_jl.jl` | CuQuantum.jl | GPU cuDensityMat | RK4 fixed-step |
| `bench_python_cudensitymat.py` | Python cuDensityMat | GPU CuPy | RK4 fixed-step |
| `bench_jax_cudensitymat.py` | JAX cuDensityMat ext | GPU JAX | RK4 fixed-step |
| `bench_quantumtoolbox_cpu.jl` | QuantumToolbox.jl | CPU sparse | DP5 adaptive |
| `bench_quantumtoolbox_gpu.jl` | QuantumToolbox.jl | GPU cuSPARSE | RK4 fixed-step |
| `bench_qutip.py` | QuTiP | CPU sparse | Adams adaptive |
| `run_all.sh` | all | all | aggregates CSVs |

## Setup

### GCE Instance

```bash
gcloud compute instances create cuquantum-bench \
  --zone=us-central1-a --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=common-cu128-ubuntu-2204-nvidia-570 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=50GB --maintenance-policy=TERMINATE
```

### Julia (CuQuantum.jl + QuantumToolbox.jl)

```bash
# Install Julia 1.12
wget -q https://julialang-s3.julialang.org/bin/linux/x64/1.12/julia-1.12.5-linux-x86_64.tar.gz -O /tmp/julia.tar.gz
tar -xzf /tmp/julia.tar.gz -C $HOME
export PATH="$HOME/julia-1.12.5/bin:$PATH"

# CuQuantum.jl benchmark
cd cuQuantum.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. benchmark/comparison/bench_cuquantum_jl.jl

# QuantumToolbox.jl benchmarks (installs deps into main project)
julia --project=. -e 'using Pkg; Pkg.add(["QuantumToolbox", "OrdinaryDiffEq"])'
julia --project=. -e 'include("benchmark/comparison/bench_quantumtoolbox_cpu.jl")'
julia --project=. -e 'include("benchmark/comparison/bench_quantumtoolbox_gpu.jl")'
```

### Python (QuTiP + CuPy cuDensityMat)

```bash
sudo apt-get install -y python3-pip
pip3 install qutip numpy psutil cupy-cuda12x cuquantum-python

python3 benchmark/comparison/bench_qutip.py
python3 benchmark/comparison/bench_python_cudensitymat.py
```

### Python (JAX cuDensityMat extension)

The JAX extension is experimental (v0.0.3), requires Python >= 3.11, and must be built from source:

```bash
# Install Python 3.11
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get install -y python3.11 python3.11-dev python3.11-venv

# Create venv
python3.11 -m venv ~/jax_env
source ~/jax_env/bin/activate
pip install --upgrade pip setuptools wheel

# Install JAX + cuquantum
pip install "jax[cuda12_local]>=0.5,<0.7"
pip install cuquantum-python-cu12 cupy-cuda12x numpy pybind11 cmake build

# Clone and install the JAX extension from NVIDIA's repo
git clone --depth=1 --filter=blob:none --sparse https://github.com/NVIDIA/cuQuantum.git /tmp/cuQuantum_ext
cd /tmp/cuQuantum_ext && git sparse-checkout set python/extensions
cd python/extensions

# The extension requires cuquantum-python ~= 25.09, but only 25.3 is on PyPI.
# Relax the constraint (the API is compatible):
sed -i 's/cuquantum-python-cu12~=25.09/cuquantum-python-cu12>=25.3.0/g' pyproject.toml

pip install --no-build-isolation .

# Run
export CUDA_PATH=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/local/cuda-12.8/extras/CUPTI/lib64:$LD_LIBRARY_PATH
cd ~/cuQuantum.jl
python benchmark/comparison/bench_jax_cudensitymat.py
```

## Environment (as tested)

| Component | Version |
|-----------|---------|
| GPU | NVIDIA Tesla T4 (16 GB) |
| CUDA | 12.8, Driver 570.211.01 |
| Julia | 1.12.5 |
| Python | 3.10.12 (QuTiP, CuPy), 3.11.15 (JAX) |
| CuQuantum.jl | 0.1.0-dev |
| cuquantum-python | 25.3.0 (CuPy), 25.11.1 (JAX) |
| cuquantum-python-jax | 0.0.3 |
| QuantumToolbox.jl | latest (via Pkg.add) |
| QuTiP | 5.2.3 |
| JAX | 0.6.2 |
| CuPy | 14.0.1 |

## Known Issues

- **M=8 is impractical for benchmarking**: cuDensityMat `prepare_action` takes >15 min; Python `OperatorTerm.__add__` is O(M²) and takes >10 min for 56 coupling products.
- **QuantumToolbox.jl GPU mesolve fails on T4** (sm_75): the DP5 ODE solver emits PTX with `.NaN` modifier requiring sm_80+ (A100). Single-action benchmarks work.
- **QuTiP single-action at M=6**: OOM trying to allocate the full dense 531K×531K Liouvillian.
- **JAX cuDensityMat ext is experimental**: not on PyPI, requires source build, Python >= 3.11, version constraint relaxation.
