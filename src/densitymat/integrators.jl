# Batched fixed-step integrators and stiffness helpers for dρ/dt = L(t)ρ.
#
# The RK4 core is generic over the array type and the right-hand side: `rhs!(out, x, t)`
# overwrites `out` with L(t)x for flat state vectors (n²·batch, cuDensityMat storage
# order). The GPU entry points wrap a `LindbladOperator` + `DenseMixedState`; the host
# entry points wrap a `CpuLindbladian`. All stage buffers are allocated once (a cache
# that can be reused across calls); the inner loop allocates no device memory.

export RK4_REAL_AXIS_BOUND, RK4_HALF_DISK_BOUND
export rk4_stability_radius, rk4_stable_step
export rk4_step!, rk4_integrate!
export LindbladRK4Cache, lindblad_rk4!, lindblad_rhs!
export estimate_lambda_max, lindblad_lambda_max

# =============================================================================
# RK4 stability region
# =============================================================================

"""
Classic RK4 stability bound on the negative real axis: `|R(z)| ≤ 1` for
`z ∈ [-RK4_REAL_AXIS_BOUND, 0]`, `R(z) = 1 + z + z²/2 + z³/6 + z⁴/24`
(the real root of `R(z) = 1`, `z ≈ -2.785293563405282`). For a real, negative dominant
eigenvalue λ the step must satisfy `h|λ| ≤ RK4_REAL_AXIS_BOUND`.
"""
const RK4_REAL_AXIS_BOUND = 2.785293563405282

"""
Radius of the largest closed half-disk `{|z| ≤ r, Re z ≤ 0}` inside the RK4 stability
region (minimum of [`rk4_stability_radius`](@ref) over angles in [π/2, π], attained near
angle 0.68π). `h · max|λ| ≤ RK4_HALF_DISK_BOUND` is sufficient for every eigenvalue λ of
a Lindbladian (all have Re λ ≤ 0), whatever their phases.
"""
const RK4_HALF_DISK_BOUND = 2.6155879072473387

_rk4_R(z) = 1 + z * (1 + z * (1 / 2 + z * (1 / 6 + z / 24)))

"""
    rk4_stability_radius(θ) -> Float64

Distance from 0 to the boundary of the RK4 stability region along the ray `arg z = θ`
(0 if the ray leaves the region immediately, i.e. |θ| < π/2 up to the imaginary axis).
Equals [`RK4_REAL_AXIS_BOUND`](@ref) at θ = π and `2√2` on the imaginary axis.
"""
function rk4_stability_radius(θ::Real)
    d = cis(Float64(θ))
    abs(_rk4_R(1e-6 * d)) > 1 && return 0.0
    r = 1e-6
    while abs(_rk4_R(r * d)) <= 1
        r += 1e-2
        r > 10 && return Inf
    end
    lo, hi = r - 1e-2, r
    for _ = 1:80
        m = (lo + hi) / 2
        abs(_rk4_R(m * d)) <= 1 ? (lo = m) : (hi = m)
    end
    return lo
end

"""
    rk4_stable_step(λ; safety=1.0) -> Float64

Largest RK4 step keeping `hλ` in the stability region, along the ray of λ:
`safety * rk4_stability_radius(angle(λ)) / |λ|`. For a vector of eigenvalue estimates
(e.g. one per batch member) the minimum over the vector.
"""
rk4_stable_step(λ::Number; safety::Real = 1.0) =
    safety * rk4_stability_radius(angle(ComplexF64(λ))) / abs(λ)
rk4_stable_step(λ::AbstractVector; safety::Real = 1.0) =
    minimum(rk4_stable_step(l; safety) for l in λ)

# =============================================================================
# Generic RK4 core
# =============================================================================

"""
    rk4_step!(rhs!, x, (k, stage, acc), t, h) -> x

One classic RK4 step of `dx/dt = rhs(t, x)` in place, with three work buffers shaped
like `x`. `rhs!(out, y, t)` must overwrite `out`.
"""
function rk4_step!(rhs!::F, x, bufs, t::Real, h::Real) where {F}
    k, s, a = bufs
    h2, h3, h6 = h / 2, h / 3, h / 6
    rhs!(k, x, t)
    a .= x .+ h6 .* k
    s .= x .+ h2 .* k
    rhs!(k, s, t + h2)
    a .+= h3 .* k
    s .= x .+ h2 .* k
    rhs!(k, s, t + h2)
    a .+= h3 .* k
    s .= x .+ h .* k
    rhs!(k, s, t + h)
    x .= a .+ h6 .* k
    return x
end

"""
    rk4_integrate!(rhs!, x, bufs, t0, t1; nsteps, callback=nothing) -> x

`nsteps` RK4 steps of size `h = (t1 - t0)/nsteps` from `t0`; step times are
`t0 + s h` (no accumulated round-off). `callback(x, t, s)` runs after every step;
returning `:stop` ends the integration early.
"""
function rk4_integrate!(
    rhs!::F,
    x,
    bufs,
    t0::Real,
    t1::Real;
    nsteps::Integer,
    callback = nothing,
) where {F}
    nsteps >= 1 || throw(ArgumentError("nsteps must be ≥ 1"))
    h = (Float64(t1) - Float64(t0)) / nsteps
    for s = 1:nsteps
        rk4_step!(rhs!, x, bufs, Float64(t0) + (s - 1) * h, h)
        if callback !== nothing
            callback(x, Float64(t0) + s * h, s) === :stop && break
        end
    end
    return x
end

# =============================================================================
# Host entry points (CpuLindbladian)
# =============================================================================

"""
    lindblad_rk4!(L::CpuLindbladian, x, t0, t1; nsteps, params=nothing, params_fn=nothing,
                  callback=nothing, bufs=nothing) -> x

Host RK4 of a batched state vector `x` (length n²·batch). `params` is a
`(num_params, batch)` matrix, or `params_fn(P, t)` fills it at each stage time.
"""
function lindblad_rk4!(
    L::CpuLindbladian,
    x::AbstractVector,
    t0::Real,
    t1::Real;
    nsteps::Integer,
    params = nothing,
    params_fn = nothing,
    callback = nothing,
    bufs = nothing,
)
    P = params === nothing ? zeros(L.num_params, L.batch_size) : Matrix{Float64}(params)
    rhs! = (out, y, t) -> begin
        params_fn === nothing || params_fn(P, t)
        L(out, y, t, P)
    end
    bufs = bufs === nothing ? (similar(x), similar(x), similar(x)) : bufs
    return rk4_integrate!(rhs!, x, bufs, t0, t1; nsteps, callback)
end

# =============================================================================
# GPU entry points (LindbladOperator)
# =============================================================================

"""
    LindbladRK4Cache(ws, L::LindbladOperator, ρ::DenseMixedState)

Device buffers for [`lindblad_rk4!`](@ref) on states congruent to `ρ`: two
`DenseMixedState`s (RK stage, slope) and the RK accumulator, plus the parameter staging
buffers. Prepares `L`'s action on first use. Reuse it across calls to keep repeated
integrations allocation-free on the device.
"""
mutable struct LindbladRK4Cache
    L::LindbladOperator
    stage::DenseMixedState{ComplexF64}
    k::DenseMixedState{ComplexF64}
    acc::CUDACore.CuVector{ComplexF64}
    params_host::Vector{Float64}
    params_dev::CUDACore.CuVector{Float64}
    prepared::Bool
end

function LindbladRK4Cache(
    ws::WorkStream,
    L::LindbladOperator,
    ρ::DenseMixedState{ComplexF64},
)
    _check_lindblad_state(L, ρ)
    stage = DenseMixedState{ComplexF64}(ws, L.dims; batch_size = L.batch_size)
    k = DenseMixedState{ComplexF64}(ws, L.dims; batch_size = L.batch_size)
    allocate_storage!(stage)
    allocate_storage!(k)
    acc = CUDACore.zeros(ComplexF64, length(ρ.storage))
    np = L.num_params * L.batch_size
    return LindbladRK4Cache(
        L,
        stage,
        k,
        acc,
        zeros(np),
        CUDACore.zeros(Float64, max(np, 1)),
        false,
    )
end

function _check_lindblad_state(L::LindbladOperator, ρ::DenseMixedState)
    ρ.storage === nothing && throw(ArgumentError("state has no storage attached"))
    ρ.batch_size == L.batch_size || throw(
        ArgumentError(
            "state batch size $(ρ.batch_size) != operator batch size $(L.batch_size)",
        ),
    )
    collect(ρ.hilbert_space_dims) == L.dims || throw(
        ArgumentError("state dims $(ρ.hilbert_space_dims) != operator dims $(L.dims)"),
    )
    return nothing
end

"""
    lindblad_rhs!(ws, cache, out::DenseMixedState, x::DenseMixedState, t; params_fn=nothing)

`out = L(t) x` (overwrites `out`) through the prepared cuDensityMat action.
"""
function lindblad_rhs!(
    ws::WorkStream,
    c::LindbladRK4Cache,
    out::DenseMixedState,
    x::DenseMixedState,
    t::Real;
    params_fn = nothing,
)
    if !c.prepared
        prepare_operator_action!(ws, c.L.operator, x, out)
        c.prepared = true
    end
    if params_fn !== nothing
        params_fn(reshape(c.params_host, c.L.num_params, c.L.batch_size), t)
        copyto!(c.params_dev, 1, c.params_host, 1, length(c.params_host))
    end
    initialize_zero!(out)
    compute_operator_action!(
        ws,
        c.L.operator,
        x,
        out;
        time = t,
        batch_size = c.L.batch_size,
        num_params = c.L.num_params,
        params = c.L.num_params == 0 ? nothing : c.params_dev,
    )
    return out
end

"""
    lindblad_rk4!(ws, L::LindbladOperator, ρ::DenseMixedState, t0, t1; nsteps,
                  params=nothing, params_fn=nothing, callback=nothing, cache=nothing) -> ρ

Integrate the batch `ρ` from `t0` to `t1` with `nsteps` classic RK4 steps (FP64) of
the cuDensityMat Lindbladian `L`, in place. Per-member parameters: `params`
(`num_params × batch` matrix, uploaded once) or `params_fn(P, t)`, which fills the host
matrix `P` in place at every stage time (it is then copied to the device). `callback(ρ,
t, step)` runs after every step (return `:stop` to end early). Pass a
[`LindbladRK4Cache`](@ref) to reuse buffers across calls.

Stability: choose `h = (t1-t0)/nsteps` with `h·|λ_max| ≤ RK4_REAL_AXIS_BOUND` for a real
dominant eigenvalue, or [`rk4_stable_step`](@ref) of the estimate from
[`lindblad_lambda_max`](@ref) in general.
"""
function lindblad_rk4!(
    ws::WorkStream,
    L::LindbladOperator,
    ρ::DenseMixedState{ComplexF64},
    t0::Real,
    t1::Real;
    nsteps::Integer,
    params = nothing,
    params_fn = nothing,
    callback = nothing,
    cache::Union{Nothing,LindbladRK4Cache} = nothing,
)
    _check_lindblad_state(L, ρ)
    c = cache === nothing ? LindbladRK4Cache(ws, L, ρ) : cache
    c.L === L || throw(ArgumentError("cache was built for a different LindbladOperator"))
    if params !== nothing
        size(params) == (L.num_params, L.batch_size) ||
            throw(ArgumentError("params must be $(L.num_params)×$(L.batch_size)"))
        c.params_host .= vec(params)
        copyto!(c.params_dev, 1, c.params_host, 1, length(c.params_host))
    end
    x = ρ.storage
    rhs! = (out, y, t) -> lindblad_rhs!(ws, c, c.k, y === x ? ρ : c.stage, t; params_fn)
    rk4_integrate!(
        rhs!,
        x,
        (c.k.storage, c.stage.storage, c.acc),
        t0,
        t1;
        nsteps,
        callback = callback === nothing ? nothing : (_, t, s) -> callback(ρ, t, s),
    )
    return ρ
end

# =============================================================================
# Stiffness: λ_max of the Lindbladian action
# =============================================================================

_colview(x, n2, B) = reshape(x, n2, B)
_coldots(x, y, n2, B) =
    vec(Array(sum(conj.(_colview(x, n2, B)) .* _colview(y, n2, B); dims = 1)))
_colnorms(x, n2, B) = sqrt.(real.(_coldots(x, x, n2, B)))

"""
    estimate_lambda_max(rhs!, v, bufs, batch_size; t=0.0, power_iters=30, krylov_dim=6)
        -> (lambda, abs, ritz, power)

Per-batch-member estimate of the eigenvalue of largest modulus of the linear map
`rhs!(out, x, t)` (out = L(t)x), from matrix-vector products only: `power_iters`
normalized power iterations from `v` (overwritten), then a `krylov_dim`-step Arnoldi
(modified Gram–Schmidt, per member) from the power iterate; `lambda[b]` is the Ritz
value of largest modulus of member b. Power iteration alone does not converge when the
dominant eigenvalues are a complex-conjugate pair (Lindbladians are Hermiticity
preserving, so non-real eigenvalues come in pairs λ, λ̄); the Rayleigh–Ritz step
resolves the pair. `power[b]` is the last power-iteration ratio ‖Lv‖/‖v‖.

`bufs` must hold `krylov_dim + 1` arrays shaped like `v`. Ritz values lie in the field
of values of L, not necessarily inside the convex hull of its spectrum (L is
non-normal), so the estimate may land slightly on either side of |λ_max| (measured
within ±10 % against dense spectra in the tests, typically ≤ 1e-3 relative); keep a
safety factor in the step choice.
"""
function estimate_lambda_max(
    rhs!::F,
    v,
    bufs,
    batch_size::Integer;
    t::Real = 0.0,
    power_iters::Integer = 30,
    krylov_dim::Integer = 6,
) where {F}
    B = batch_size
    n2 = length(v) ÷ B
    m = krylov_dim
    length(bufs) >= m + 1 || throw(ArgumentError("need krylov_dim + 1 = $(m+1) buffers"))
    coef = similar(v, B)
    scalecols!(x, s) =
        (copyto!(coef, ComplexF64.(s)); _colview(x, n2, B) .*= reshape(coef, 1, B); x)
    scalecols!(v, 1 ./ _colnorms(v, n2, B))
    w = bufs[1]
    power = zeros(B)
    for _ = 1:power_iters
        rhs!(w, v, t)
        power = _colnorms(w, n2, B)
        v .= w
        scalecols!(v, 1 ./ power)
    end
    V = bufs
    V[1] .= v
    H = zeros(ComplexF64, m + 1, m, B)
    mm = m
    for j = 1:m
        rhs!(V[j+1], V[j], t)
        for i = 1:j
            h = _coldots(V[i], V[j+1], n2, B)
            H[i, j, :] .= h
            copyto!(coef, -h)
            _colview(V[j+1], n2, B) .+= _colview(V[i], n2, B) .* reshape(coef, 1, B)
        end
        nrm = _colnorms(V[j+1], n2, B)
        H[j+1, j, :] .= nrm
        if any(<(1e-13 * maximum(abs, H[1:j, j, :]) + eps()), nrm)
            mm = j   # (happy) breakdown: the Krylov space is invariant
            break
        end
        scalecols!(V[j+1], 1 ./ nrm)
    end
    ritz = [LinearAlgebra.eigvals(H[1:mm, 1:mm, b]) for b = 1:B]
    lam = [r[argmax(abs.(r))] for r in ritz]
    return (lambda = lam, abs = abs.(lam), ritz = ritz, power = power)
end

"""
    lindblad_lambda_max(L::CpuLindbladian; t=0.0, params=nothing, kwargs...)
    lindblad_lambda_max(ws, L::LindbladOperator; t=0.0, params=nothing, kwargs...)

[`estimate_lambda_max`](@ref) of a Lindbladian (per batch member), from a fixed
pseudo-random Hermitian start; `params` is the `(num_params, batch)` matrix. The GPU
method allocates `krylov_dim + 3` state-sized buffers for the call.
"""
function lindblad_lambda_max(
    L::CpuLindbladian;
    t::Real = 0.0,
    params = nothing,
    krylov_dim::Integer = 6,
    kwargs...,
)
    P = params === nothing ? zeros(L.num_params, L.batch_size) : Matrix{Float64}(params)
    v = _hermitian_start(L.n, L.batch_size)
    bufs = [similar(v) for _ = 1:(krylov_dim+1)]
    return estimate_lambda_max(
        (o, x, τ) -> L(o, x, τ, P),
        v,
        bufs,
        L.batch_size;
        t,
        krylov_dim,
        kwargs...,
    )
end

function lindblad_lambda_max(
    ws::WorkStream,
    L::LindbladOperator;
    t::Real = 0.0,
    params = nothing,
    krylov_dim::Integer = 6,
    kwargs...,
)
    n = prod(L.dims)
    mk() = (
        s = DenseMixedState{ComplexF64}(ws, L.dims; batch_size = L.batch_size);
        allocate_storage!(s);
        s
    )
    states = [mk() for _ = 1:(krylov_dim+2)]
    probe = states[end]
    c = LindbladRK4Cache(
        L,
        states[1],
        states[2],
        CUDACore.zeros(ComplexF64, 1),
        zeros(L.num_params * L.batch_size),
        CUDACore.zeros(Float64, max(1, L.num_params * L.batch_size)),
        false,
    )
    if params !== nothing
        c.params_host .= vec(params)
        copyto!(c.params_dev, 1, c.params_host, 1, length(c.params_host))
    end
    copyto!(probe.storage, _hermitian_start(n, L.batch_size))
    bystorage = IdDict{Any,DenseMixedState{ComplexF64}}(s.storage => s for s in states)
    rhs! = (o, x, τ) -> lindblad_rhs!(ws, c, bystorage[o], bystorage[x], τ)
    return estimate_lambda_max(
        rhs!,
        probe.storage,
        [s.storage for s in states[1:(krylov_dim+1)]],
        L.batch_size;
        t,
        krylov_dim,
        kwargs...,
    )
end

# Deterministic generic Hermitian start vector (no RNG dependency): quasi-random phases.
function _hermitian_start(n::Integer, B::Integer)
    x = zeros(ComplexF64, n * n * B)
    φ = (sqrt(5) - 1) / 2
    for b = 1:B
        X = reshape(view(x, ((b-1)*n*n+1):(b*n*n)), n, n)
        for j = 1:n, i = 1:n
            k = i + n * (j - 1) + n * n * (b - 1)
            X[i, j] = cis(2π * mod(k * φ + k^2 * φ^2, 1)) * (1 + mod(k * φ^3, 1))
        end
        X .= X .+ X'
    end
    return x
end
