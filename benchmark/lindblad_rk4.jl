# Lindbladian builder + batched RK4: timing and agreement.
#
# Two coupled bosonic modes (Fock cutoff c each), beam-splitter coupling, one-body loss and
# two-body loss on each mode:
#
#   dρ/dt = -i[g(a†b + ab†), ρ] + Σ_{m=a,b} ( κ D[m] + γ D[m²] ) ρ,   D[L]ρ = LρL† - ½{L†L, ρ}
#
# The batch holds `batch` Fock-state initial conditions |n_a, n_b⟩⟨n_a, n_b|.
#
# 1. Agreement (small cutoffs): the GPU Operator's action, and a few RK4 steps, against a dense
#    vectorized-superoperator reference (`lindblad_superoperator`, built directly from the spec).
# 2. Timing: warm-cache ms per RK4 step and per operator action vs cutoff and batch size, plus
#    state/workspace bytes and device allocation per step.
#
# Usage: julia --project=<env with CUDA + CuQuantum> benchmark/lindblad_rk4.jl \
#            [cutoffs=16,24,32] [batches=1,16] [steps=20] [agree=3,4,5] [out=lindblad_rk4.json]
using CUDA, CuQuantum, CuQuantum.CuDensityMat, LinearAlgebra, Printf
const CD = CuDensityMat

const KV =
    Dict(split(a, "=")[1] => split(a, "=", limit = 2)[2] for a in ARGS if occursin('=', a))
ints(key, default) = parse.(Int, split(get(KV, key, default), ","))
const CUTOFFS = ints("cutoffs", "16,24,32")
const BATCHES = ints("batches", "1,16")
const AGREE = ints("agree", "3,4,5")
const STEPS = parse(Int, get(KV, "steps", "20"))
const OUT = let o = get(KV, "out", "lindblad_rk4.json")
    isdir(o) ? joinpath(o, "lindblad_rk4.json") : o   # `out=` may name a directory
end

const g, κ, γ = 0.3, 0.05, 0.02

annihilation(c) = Matrix{ComplexF64}(diagm(1 => sqrt.(1.0:(c-1))))

"""Beam splitter + one- and two-body loss on two modes of cutoff `c` (mode 2 = `a`, mode 1 = `b`)."""
function two_mode_spec(c)
    a = annihilation(c)
    spec = LindbladSpec([c, c])
    add_hamiltonian!(spec, g, 1 => a', 2 => a)
    add_hamiltonian!(spec, g, 1 => a, 2 => a')
    for m = 1:2
        add_jump!(spec, κ, m => a)
        add_jump!(spec, γ, m => a * a)
    end
    return spec
end

"""
Batch of Fock-state density matrices |n_a, n_b⟩⟨n_a, n_b|, cycling through occupations
(0, 0), (c-1, 0), (0, c-1), (c-1, c-1), (c÷2, 1), ... so the highest Fock levels (and the
outermost diagonals of the multidiagonal operators) take part at every cutoff.
"""
function fock_batch(c, batch)
    occ = [(0, 0), (c - 1, 0), (0, c - 1), (c - 1, c - 1), (c ÷ 2, 1), (1, c ÷ 2)]
    x = zeros(ComplexF64, c^4 * batch)
    for b = 1:batch
        na, nb = occ[mod1(b, length(occ))]
        k = 1 + nb + c * na                 # mode 1 (b) is the fast index
        x[(b-1)*c^4+k+(k-1)*c^2] = 1
    end
    return x
end

maxrel(x, ref) = maximum(abs, x .- ref) / maximum(abs, ref)

rows_agree = Any[]
rows_time = Any[]

# ── 1. agreement with the dense superoperator ───────────────────────────────
for c in AGREE
    ws = WorkStream()
    spec = two_mode_spec(c)
    n2 = c^2
    B = 4
    L = LindbladOperator(ws, spec; batch_size = B)
    ρ = DenseMixedState{ComplexF64}(ws, [c, c]; batch_size = B)
    out = DenseMixedState{ComplexF64}(ws, [c, c]; batch_size = B)
    CD.allocate_storage!(ρ)
    CD.allocate_storage!(out)
    x0 = fock_batch(c, B)
    copyto!(ρ.storage, x0)
    cache = LindbladRK4Cache(ws, L, ρ)
    lindblad_rhs!(ws, cache, out, ρ, 0.0)
    S = lindblad_superoperator(spec)
    ref_action = reduce(vcat, [S * x0[((b-1)*n2^2+1):(b*n2^2)] for b = 1:B])
    err_action = maxrel(Array(out.storage), ref_action)
    T, N = 0.5, 200
    lindblad_rk4!(ws, L, ρ, 0.0, T; nsteps = N, cache)
    ref_prop = reduce(vcat, [exp(S * T) * x0[((b-1)*n2^2+1):(b*n2^2)] for b = 1:B])
    err_prop = maxrel(Array(ρ.storage), ref_prop)
    @printf(
        "AGREE c=%d  action %.2e  RK4(%d steps, T=%g) vs exp(ST) %.2e\n",
        c,
        err_action,
        N,
        T,
        err_prop
    )
    push!(
        rows_agree,
        Dict(
            "c" => c,
            "action_relerr" => err_action,
            "rk4_vs_expm_relerr" => err_prop,
            "nsteps" => N,
            "T" => T,
        ),
    )
    close(ws)
end

# ── 2. timing ───────────────────────────────────────────────────────────────
for c in CUTOFFS, B in BATCHES
    ws = WorkStream()
    spec = two_mode_spec(c)
    L = LindbladOperator(ws, spec; batch_size = B)
    ρ = DenseMixedState{ComplexF64}(ws, [c, c]; batch_size = B)
    CD.allocate_storage!(ρ)
    copyto!(ρ.storage, fock_batch(c, B))
    cache = LindbladRK4Cache(ws, L, ρ)
    h = 1e-4
    lindblad_rk4!(ws, L, ρ, 0.0, 3h; nsteps = 3, cache)          # warm-up: prepare + compile
    CUDA.synchronize()
    dev_bytes = CD.CUDACore.@allocated lindblad_rk4!(ws, L, ρ, 0.0, 2h; nsteps = 2, cache)
    CUDA.synchronize()
    t = time()
    lindblad_rk4!(ws, L, ρ, 0.0, STEPS * h; nsteps = STEPS, cache)
    CUDA.synchronize()
    ms_step = (time() - t) / STEPS * 1e3
    t = time()
    for _ = 1:STEPS
        lindblad_rhs!(ws, cache, cache.k, ρ, 0.0)
    end
    CUDA.synchronize()
    ms_action = (time() - t) / STEPS * 1e3
    products = sum(length(grp.products) for grp in L.plan)
    @printf(
        "TIME c=%-3d batch=%-3d  %9.3f ms/step  %8.3f ms/action  state %.1f MB  workspace %.1f MB  products %d  device alloc/step %d B\n",
        c,
        B,
        ms_step,
        ms_action,
        sizeof(ρ.storage) / 2^20,
        ws.workspace_size / 2^20,
        products,
        dev_bytes ÷ 2
    )
    push!(
        rows_time,
        Dict(
            "c" => c,
            "batch" => B,
            "ms_per_step" => ms_step,
            "ms_per_action" => ms_action,
            "state_bytes" => sizeof(ρ.storage),
            "workspace_bytes" => ws.workspace_size,
            "products" => products,
            "device_alloc_bytes_per_step" => dev_bytes ÷ 2,
        ),
    )
    close(ws)
end

# ── receipt ─────────────────────────────────────────────────────────────────
function tojson(io, x)
    if x isa AbstractDict
        print(io, "{")
        for (i, k) in enumerate(sort!(collect(keys(x)); by = string))
            i > 1 && print(io, ",")
            print(io, "\"", k, "\":")
            tojson(io, x[k])
        end
        print(io, "}")
    elseif x isa AbstractVector
        print(io, "[")
        for (i, v) in enumerate(x)
            i > 1 && print(io, ",")
            tojson(io, v)
        end
        print(io, "]")
    elseif x isa Real
        print(io, isfinite(x) ? repr(Float64(x)) : "null")
    else
        print(io, "\"", x, "\"")
    end
end
open(OUT, "w") do io
    tojson(
        io,
        Dict(
            "device" => CUDA.name(CUDA.device()),
            "libcudensitymat" => string(CD.version()),
            "model" => "two modes, beam splitter g=$g, one-body loss κ=$κ, two-body loss γ=$γ",
            "agreement" => rows_agree,
            "timing" => rows_time,
        ),
    )
end
println("RECEIPT ", OUT)
