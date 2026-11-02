# High-level Lindbladian builder.
#
#   dρ/dt = L(t)ρ = -i[H(t), ρ] + Σ_j γ_j(t) (L_j ρ L_j† - ½{L_j†L_j, ρ})
#
#   H(t) = Σ_k c_k(t, p) P_k,   P_k = ⊗_m O_{k,m}  (single-mode factors on modes m)
#
# A `LindbladSpec` records the model on the host. `lindblad_plan` lowers it to groups of
# elementary products, each product carrying a static coefficient and a list of single-mode
# factors with a ket/bra side; a group shares one (possibly time- and batch-dependent)
# coefficient. `LindbladOperator` uploads a plan as a cuDensityMat `Operator` built from
# MULTIDIAGONAL elementary operators (dense storage only when a factor has no
# multidiagonal advantage). `CpuLindbladian` executes the same plan on the host (sparse
# full-space matrices) and `lindblad_superoperator` builds the dense column-stacked
# superoperator directly from the spec; the two host paths are independent, so the plan
# lowering and the GPU upload can each be checked against the superoperator.
#
# Conventions
# - Modes are 1-based in this API (cuDensityMat's are 0-based). Mode 1 is the fastest-
#   varying index of the state storage: the full-space matrix of `m => O` is
#   kron(I_{d_M}, …, O, …, I_{d_1}) (mode M leftmost), and one batch member of a
#   `DenseMixedState` is the column-major n×n matrix ρ[ket, bra], i.e. `vec(ρ)`.
# - A ket-side factor M on mode m maps ρ ↦ M_m ρ; a bra-side factor maps ρ ↦ ρ M_m.
# - Coefficient functions are `f(t::Float64, p::AbstractVector{Float64}) -> Number`, where
#   `p = params[:, b]` holds batch member b's parameters (`params` is
#   `(num_params, batch_size)`). On the GPU they run as CPU scalar callbacks of BATCHED
#   terms (`append_term_batch!`), so every batch member gets its own value.

export LindbladSpec, add_hamiltonian!, add_jump!
export lindblad_plan, lindblad_superoperator, embed_operator
export CpuLindbladian, LindbladOperator
export multidiagonal_pack, multidiagonal_unpack

# =============================================================================
# Spec
# =============================================================================

"""
    LindbladSpec(dims)

Host-side description of a Lindblad master equation on modes of extents `dims`. Fill it
with [`add_hamiltonian!`](@ref) and [`add_jump!`](@ref), then lower it with
[`LindbladOperator`](@ref) (GPU) or [`CpuLindbladian`](@ref) (host reference).
"""
struct LindbladSpec
    dims::Vector{Int}
    # (coefficient function or nothing, static coefficient, sorted mode => matrix factors)
    hamiltonian::Vector{Tuple{Any,ComplexF64,Vector{Pair{Int,Matrix{ComplexF64}}}}}
    # (rate function or nothing, static rate, factors of L)
    jumps::Vector{Tuple{Any,Float64,Vector{Pair{Int,Matrix{ComplexF64}}}}}
end

LindbladSpec(dims::AbstractVector{<:Integer}) = LindbladSpec(Int.(collect(dims)), [], [])
LindbladSpec(dims::Tuple) = LindbladSpec(collect(dims))

"""
    add_hamiltonian!(spec, coefficient, factors::Pair{Int,<:AbstractMatrix}...) -> spec

Add the Hamiltonian term `coefficient * ⊗ factors` (factors on the same mode are
multiplied in the given order). `coefficient` is a `Number` (static) or a function
`f(t, p) -> Number` of time and the batch member's parameter vector. The terms must add
up to a Hermitian H; individual terms need not be Hermitian (e.g. `g a†b` and `g ab†`).
"""
function add_hamiltonian!(spec::LindbladSpec, coefficient, factors::Pair...)
    fs = _normalize_factors(spec, factors)
    isempty(fs) && return spec   # identity terms drop out of the commutator
    if coefficient isa Number
        push!(spec.hamiltonian, (nothing, ComplexF64(coefficient), fs))
    else
        coefficient isa Function ||
            throw(ArgumentError("coefficient must be a Number or a function f(t, p)"))
        push!(spec.hamiltonian, (coefficient, ComplexF64(1), fs))
    end
    return spec
end

"""
    add_jump!(spec, rate, factors::Pair{Int,<:AbstractMatrix}...) -> spec

Add the dissipator `rate * D[L]`, `D[L]ρ = LρL† - ½{L†L, ρ}`, with jump operator
`L = ⊗ factors`. `rate` is a non-negative `Real` (static) or a function `r(t, p) -> Real`.
(A rate folded into the operator, `L = √γ a`, is equivalent to `rate = γ`, `L = a`.)
"""
function add_jump!(spec::LindbladSpec, rate, factors::Pair...)
    fs = _normalize_factors(spec, factors)
    isempty(fs) && return spec   # D[c·I] = 0
    if rate isa Real
        rate >= 0 || throw(ArgumentError("static rate must be non-negative, got $rate"))
        push!(spec.jumps, (nothing, Float64(rate), fs))
    else
        rate isa Function ||
            throw(ArgumentError("rate must be a Real or a function r(t, p)"))
        push!(spec.jumps, (rate, 1.0, fs))
    end
    return spec
end

function _normalize_factors(spec::LindbladSpec, factors)
    byMode = Dict{Int,Matrix{ComplexF64}}()
    for f in factors
        f isa Pair || throw(ArgumentError("factors must be mode => matrix pairs"))
        m = Int(first(f))
        1 <= m <= length(spec.dims) ||
            throw(ArgumentError("mode $m out of range 1:$(length(spec.dims))"))
        M = Matrix{ComplexF64}(last(f))
        size(M) == (spec.dims[m], spec.dims[m]) || throw(
            ArgumentError(
                "factor on mode $m has size $(size(M)), expected $(spec.dims[m])²",
            ),
        )
        byMode[m] = haskey(byMode, m) ? byMode[m] * M : M
    end
    return [m => byMode[m] for m in sort!(collect(keys(byMode)))]
end

# =============================================================================
# Full-space reference (dense superoperator, independent of the plan)
# =============================================================================

"""
    embed_operator(dims, factors) -> Matrix{ComplexF64}

Full-space matrix of `⊗ factors` (`factors` = `mode => matrix` pairs, identity elsewhere),
in the storage order of cuDensityMat states (mode 1 fastest).
"""
function embed_operator(dims::AbstractVector{<:Integer}, factors)
    ops = Dict{Int,Matrix{ComplexF64}}(
        Int(first(f)) => Matrix{ComplexF64}(last(f)) for f in factors
    )
    M = ones(ComplexF64, 1, 1)
    for m = length(dims):-1:1
        M = kron(
            M,
            get(() -> Matrix{ComplexF64}(LinearAlgebra.I, dims[m], dims[m]), ops, m),
        )
    end
    return M
end

"""
    lindblad_superoperator(spec, t=0.0, p=Float64[]) -> Matrix{ComplexF64}

Dense column-stacked superoperator of `spec` at time `t` for parameter vector `p`
(`vec(L(ρ)) = S * vec(ρ)`), from the textbook form
`S = -i(I⊗H - Hᵀ⊗I) + Σ γ (conj(L)⊗L - ½ I⊗L†L - ½ (L†L)ᵀ⊗I)`. Reference only — the
size is n²×n² for n = prod(dims).
"""
function lindblad_superoperator(spec::LindbladSpec, t::Real = 0.0, p = Float64[])
    n = prod(spec.dims)
    Id = Matrix{ComplexF64}(LinearAlgebra.I, n, n)
    H = zeros(ComplexF64, n, n)
    for (f, s, fs) in spec.hamiltonian
        c = f === nothing ? s : s * ComplexF64(f(Float64(t), p))
        H .+= c .* embed_operator(spec.dims, fs)
    end
    S = -im .* (kron(Id, H) .- kron(transpose(H), Id))
    for (r, γ, fs) in spec.jumps
        rate = r === nothing ? γ : γ * Float64(r(Float64(t), p))
        L = embed_operator(spec.dims, fs)
        K = L' * L
        S .+=
            rate .*
            (kron(conj(L), L) .- 0.5 .* kron(Id, K) .- 0.5 .* kron(transpose(K), Id))
    end
    return S
end

# =============================================================================
# Plan: groups of elementary products
# =============================================================================

"""One single-mode factor of an elementary product: ρ ↦ M_m ρ (ket) or ρ ↦ ρ M_m (bra)."""
struct PlanFactor
    mode::Int
    matrix::Matrix{ComplexF64}
    bra::Bool
end

"""An elementary product `coefficient * ∏ factors` (factors on distinct (mode, side) slots)."""
struct PlanProduct
    coefficient::ComplexF64
    factors::Vector{PlanFactor}
end

"""Products sharing one coefficient: `nothing` (static, 1) or `f(t, p)`."""
struct PlanGroup
    coefficient::Any
    products::Vector{PlanProduct}
end

"""
    lindblad_plan(spec; merge=true) -> Vector{PlanGroup}

Lower `spec` to elementary products. Hamiltonian terms become a ket product (`-i c P`)
and a bra product (`+i c P`); each jump becomes the sandwich `γ (ket L, bra L†)` and the
anticommutator halves `-γ/2 (ket L†L)`, `-γ/2 (bra L†L)` with `L†L = ⊗ L_m†L_m`.
Terms with the same coefficient (`nothing` = static, or the same function object) share
a group. With `merge=true`, single-factor products of a group on the same (mode, side)
are summed into one factor matrix (e.g. Kerr, detuning and all `-½L†L` of one mode
become one ket and one bra operator), which cuts the number of state passes per action.
"""
function lindblad_plan(spec::LindbladSpec; merge::Bool = true)
    groups = Pair{Any,Vector{PlanProduct}}[]
    group_for(f) = begin
        i = findfirst(g -> first(g) === f, groups)
        i === nothing ? last(push!(groups, f => PlanProduct[])[end]) : last(groups[i])
    end
    side(fs, bra) = [PlanFactor(first(f), last(f), bra) for f in fs]
    for (f, s, fs) in spec.hamiltonian
        g = group_for(f)
        push!(g, PlanProduct(-im * s, side(fs, false)))
        push!(g, PlanProduct(im * s, side(fs, true)))
    end
    for (r, γ, fs) in spec.jumps
        g = group_for(r)
        push!(
            g,
            PlanProduct(γ, vcat(side(fs, false), side([m => M' for (m, M) in fs], true))),
        )
        K = [m => M' * M for (m, M) in fs]
        push!(g, PlanProduct(-γ / 2, side(K, false)))
        push!(g, PlanProduct(-γ / 2, side(K, true)))
    end
    return [PlanGroup(first(g), merge ? _merge_single(last(g)) : last(g)) for g in groups]
end

function _merge_single(products::Vector{PlanProduct})
    acc = Dict{Tuple{Int,Bool},Matrix{ComplexF64}}()
    order = Tuple{Int,Bool}[]
    out = PlanProduct[]
    for p in products
        if length(p.factors) == 1
            f = p.factors[1]
            key = (f.mode, f.bra)
            haskey(acc, key) ||
                (push!(order, key); acc[key] = zeros(ComplexF64, size(f.matrix)))
            acc[key] .+= p.coefficient .* f.matrix
        else
            push!(out, p)
        end
    end
    singles = [
        PlanProduct(1, [PlanFactor(k[1], acc[k], k[2])]) for
        k in order if any(!iszero, acc[k])
    ]
    return vcat(singles, out)
end

# =============================================================================
# Multidiagonal storage
# =============================================================================

"""
    multidiagonal_pack(M) -> (offsets::Vector{Int32}, data::Vector{ComplexF64})

cuDensityMat MULTIDIAGONAL storage of the d×d matrix `M`: the non-zero diagonals
(offset k > 0 above, k < 0 below the main one, ascending), each stored as a full column
of length d — `diag(M, k)` from the top-left, padded with trailing zeros — concatenated
in the order of `offsets`. An all-zero matrix keeps the main diagonal.
"""
function multidiagonal_pack(M::AbstractMatrix)
    d = size(M, 1)
    size(M, 2) == d || throw(ArgumentError("matrix must be square"))
    offs = Int32[k for k = (-(d-1)):(d-1) if any(!iszero, LinearAlgebra.diag(M, k))]
    isempty(offs) && push!(offs, 0)
    data = zeros(ComplexF64, d, length(offs))
    for (j, k) in enumerate(offs)
        dg = LinearAlgebra.diag(M, k)
        data[1:length(dg), j] .= dg
    end
    return offs, vec(data)
end

"""
    multidiagonal_unpack(offsets, data, d) -> Matrix{ComplexF64}

Inverse of [`multidiagonal_pack`](@ref).
"""
function multidiagonal_unpack(
    offsets::AbstractVector{<:Integer},
    data::AbstractVector,
    d::Integer,
)
    D = reshape(data, d, length(offsets))
    M = zeros(ComplexF64, d, d)
    for (j, k) in enumerate(offsets)
        len = d - abs(k)
        for i = 1:len
            k >= 0 ? (M[i, i+k] = D[i, j]) : (M[i-k, i] = D[i, j])
        end
    end
    return M
end

# =============================================================================
# Host executor of a plan (reference backend)
# =============================================================================

"""
    CpuLindbladian(spec; batch_size=1, num_params=0, merge=true)

Host executor of `lindblad_plan(spec)` on batched density matrices stored like
cuDensityMat states (a vector of length n²·batch_size, member b = `vec(ρ_b)`). Callable
as `L(out, x, t, params)`: `out = L(t) x` (overwrites `out`), `params` a
`(num_params, batch_size)` matrix. Each product is a pair of sparse full-space matrices
(ket side, bra side); this is the reference backend for small dims and for tests.
"""
struct CpuLindbladian
    dims::Vector{Int}
    n::Int
    batch_size::Int
    num_params::Int
    groups::Vector{Any}
    tmp::Matrix{ComplexF64}
    tmp2::Matrix{ComplexF64}
end

function CpuLindbladian(
    spec::LindbladSpec;
    batch_size::Integer = 1,
    num_params::Integer = 0,
    merge::Bool = true,
)
    n = prod(spec.dims)
    sp(fs) =
        isempty(fs) ? nothing :
        SparseArrays.sparse(embed_operator(spec.dims, [f.mode => f.matrix for f in fs]))
    groups = Any[]
    for g in lindblad_plan(spec; merge)
        push!(
            groups,
            (
                g.coefficient,
                [
                    (
                        p.coefficient,
                        sp(filter(f -> !f.bra, p.factors)),
                        sp(filter(f -> f.bra, p.factors)),
                    ) for p in g.products
                ],
            ),
        )
    end
    return CpuLindbladian(
        spec.dims,
        n,
        batch_size,
        num_params,
        groups,
        zeros(ComplexF64, n, n),
        zeros(ComplexF64, n, n),
    )
end

function (L::CpuLindbladian)(
    out::AbstractVector,
    x::AbstractVector,
    t::Real,
    params = nothing,
)
    n, B = L.n, L.batch_size
    length(x) == n * n * B ||
        throw(DimensionMismatch("state length $(length(x)) != n²·batch $(n*n*B)"))
    fill!(out, 0)
    for b = 1:B
        X = reshape(view(x, ((b-1)*n*n+1):(b*n*n)), n, n)
        Y = reshape(view(out, ((b-1)*n*n+1):(b*n*n)), n, n)
        p = params === nothing ? Float64[] : view(params, :, b)
        for (f, prods) in L.groups
            c = f === nothing ? ComplexF64(1) : ComplexF64(f(Float64(t), p))
            for (s, Lk, Rb) in prods
                w = c * s
                if Rb === nothing
                    LinearAlgebra.mul!(Y, Lk, X, w, true)
                elseif Lk === nothing
                    LinearAlgebra.mul!(L.tmp, X, Rb)
                    Y .+= w .* L.tmp
                else
                    LinearAlgebra.mul!(L.tmp2, Lk, X)
                    LinearAlgebra.mul!(L.tmp, L.tmp2, Rb)
                    Y .+= w .* L.tmp
                end
            end
        end
    end
    return out
end

# If `products` is a commutator — every ket-only product K_j matched, in order, by a bra-only
# product equal to -K_j — return the ket products, else `nothing`.
function _commutator_kets(products::Vector{PlanProduct})
    kets = filter(p -> all(f -> !f.bra, p.factors), products)
    bras = filter(p -> all(f -> f.bra, p.factors), products)
    (
        length(kets) + length(bras) == length(products) &&
        length(kets) == length(bras) &&
        !isempty(kets)
    ) || return nothing
    folded(p) = [
        (f.mode, i == 1 ? p.coefficient .* f.matrix : f.matrix) for
        (i, f) in enumerate(sort(p.factors; by = f -> f.mode))
    ]
    for (k, b) in zip(kets, bras)
        fk, fb = folded(k), folded(b)
        length(fk) == length(fb) || return nothing
        all(
            x[1] == y[1] && x[2] == (i == 1 ? -y[2] : y[2]) for
            (i, (x, y)) in enumerate(zip(fk, fb))
        ) || return nothing
    end
    return kets
end

# =============================================================================
# GPU: cuDensityMat Operator from a plan
# =============================================================================

"""
    LindbladOperator(ws, spec; batch_size=1, num_params=0, sparsity=:auto, merge=true,
                     split_terms=false, factor_order=:by_mode)

The cuDensityMat `Operator` of `dρ/dt = L(t)ρ` for `spec`, for states of batch size
`batch_size` and `num_params` parameters per member (pass `params` as a
`num_params × batch_size` column-major `CuVector`). One `OperatorTerm` per plan group:
the static group is appended with `append_term!`; every coefficient function becomes a
batched CPU scalar callback (`append_term_batch!`), so members can differ. Factors use
MULTIDIAGONAL storage (`sparsity=:auto` unless the factor fills every diagonal, then
dense; `:multidiagonal` / `:dense` force one kind). Each factor owns its elementary
operator (`share_elementary=true` shares identical matrices across products; see the
comment in the constructor for why this is off by default).

`split_terms` / `factor_order` are layout knobs (one `OperatorTerm` per product signature;
factor order `:by_mode` or `:ket_first`) that give the same operator.

Fields: `operator` (the `Operator`), `plan`, `dims`, `batch_size`, `num_params`,
`num_elementary` (distinct elementary operators), `data_bytes` (their device storage).
"""
mutable struct LindbladOperator
    operator::Operator
    ws::WorkStream
    dims::Vector{Int}
    batch_size::Int
    num_params::Int
    plan::Vector{PlanGroup}
    num_elementary::Int
    data_bytes::Int
    _refs::Vector{Any}
end

function LindbladOperator(
    ws::WorkStream,
    spec::LindbladSpec;
    batch_size::Integer = 1,
    num_params::Integer = 0,
    sparsity::Symbol = :auto,
    merge::Bool = true,
    split_terms::Union{Bool,Symbol} = false,
    factor_order::Symbol = :by_mode,
    share_elementary::Bool = false,
    hamiltonian_duality::Bool = true,
)
    factor_order in (:by_mode, :ket_first) ||
        throw(ArgumentError("factor_order must be :by_mode or :ket_first"))
    sparsity in (:auto, :multidiagonal, :dense) ||
        throw(ArgumentError("sparsity must be :auto, :multidiagonal or :dense"))
    plan = lindblad_plan(spec; merge)
    op = create_operator(ws, spec.dims)
    refs = Any[]
    # Elementary operators are not shared by default (one instance per factor occurrence): a
    # conservative choice made while chasing a 0.7.0 prepare abort that was finally traced to
    # batch-size-1 batched callback terms (below). `share_elementary=true` shares identical
    # matrices across products (k-th use inside one product → k-th instance).
    pool = Dict{Matrix{ComplexF64},Vector{ElementaryOperator}}()
    nbytes = 0
    function make_elem(M)
        offs, data = multidiagonal_pack(M)
        d = size(M, 1)
        multi = sparsity == :multidiagonal || (sparsity == :auto && length(offs) < 2d - 1)
        buf = CUDACore.CuVector{ComplexF64}(multi ? data : vec(M))
        nbytes += sizeof(buf)
        return multi ?
               create_elementary_operator(
            ws,
            [d],
            buf;
            sparsity = :multidiagonal,
            diagonal_offsets = offs,
        ) : create_elementary_operator(ws, [d], buf)
    end
    function elems_for(factors)
        uses = Dict{Matrix{ComplexF64},Int}()
        out = ElementaryOperator[]
        for f in factors
            if !share_elementary
                M = f.matrix
                push!(get!(Vector{ElementaryOperator}, pool, M), make_elem(M))
                push!(out, pool[M][end])
                continue
            end
            k = uses[f.matrix] = get(uses, f.matrix, 0) + 1
            inst = get!(Vector{ElementaryOperator}, pool, f.matrix)
            length(inst) < k && push!(inst, make_elem(f.matrix))
            push!(out, inst[k])
        end
        return out
    end
    # Layout knobs (A10 probes, libcudensitymat 0.7.0: all four combinations agree with the
    # superoperator to ≤ 4e-16): `split_terms=true`/`:signature` gives every product
    # signature (set of (mode, side) slots) of a group its own OperatorTerm, `:product` every
    # product; `factor_order` lists a product's
    # factors by (mode, side) (`:by_mode`) or all kets first (`:ket_first`).
    order(fs) =
        factor_order == :by_mode ? sort(fs; by = f -> (f.mode, f.bra)) :
        sort(fs; by = f -> (f.bra, f.mode))
    signature(p) = Tuple(sort([(f.mode, f.bra) for f in p.factors]))
    subterms = Tuple{Any,Vector{PlanProduct}}[]
    mode = split_terms === true ? :signature : split_terms === false ? :none : split_terms
    mode in (:none, :signature, :product) ||
        throw(ArgumentError("split_terms must be false/:none, true/:signature or :product"))
    for g in plan
        if mode == :signature
            sigs = unique(signature.(g.products))
            append!(
                subterms,
                [
                    (g.coefficient, filter(p -> signature(p) == sg, g.products)) for
                    sg in sigs
                ],
            )
        elseif mode == :product
            append!(subterms, [(g.coefficient, [p]) for p in g.products])
        else
            push!(subterms, (g.coefficient, g.products))
        end
    end
    for (coef, products0) in subterms
        isempty(products0) && continue
        # A commutator group (ket products K_j and bra products -K_j) is uploaded as ONE term
        # of the ket products, appended with duality 0 (coefficient +1) and duality 1 (-1) —
        # the "commutator via duality" form. On the A10 (0.7.0) a batched-callback term holding
        # both [a† ket] and [a† bra] products of a one-mode space aborted in prepare.
        kets = hamiltonian_duality ? _commutator_kets(products0) : nothing
        products = kets === nothing ? products0 : kets
        term = create_operator_term(ws, spec.dims)
        for p in products
            fs = order(p.factors)
            append_elementary_product!(
                term,
                elems_for(fs),
                Int32[f.mode - 1 for f in fs],
                Int32[f.bra ? 1 : 0 for f in fs];
                coefficient = p.coefficient,
            )
        end
        g = (coefficient = coef,)
        sides = kets === nothing ? ((0, 1.0),) : ((0, 1.0), (1, -1.0))
        if g.coefficient === nothing
            for (dual, c) in sides
                append_term!(op, term; duality = dual, coefficient = ComplexF64(c))
            end
        else
            f = g.coefficient
            cb, _, cbrefs = wrap_scalar_callback((t, P, s) -> begin
                @inbounds for b in eachindex(s)
                    s[b] = f(t, view(P, :, b))
                end
                nothing
            end)
            for (dual, c) in sides
                if batch_size == 1
                    # libcudensitymat 0.7.0 aborts in prepare (exatn "Connections are invalid")
                    # on a BATCHED callback term of batch size 1; for one member the
                    # non-batched term is the same coefficient (callback sees params[:, 1]).
                    append_term!(
                        op,
                        term;
                        duality = dual,
                        coefficient = ComplexF64(c),
                        coefficient_callback = cb,
                    )
                else
                    static = CUDACore.CuVector{ComplexF64}(fill(ComplexF64(c), batch_size))
                    append_term_batch!(
                        op,
                        term,
                        batch_size,
                        static;
                        duality = dual,
                        coefficient_callback = cb,
                    )
                    push!(refs, static)
                end
            end
            push!(refs, cbrefs)
        end
        push!(refs, term)
    end
    push!(refs, collect(values(pool)))
    L = LindbladOperator(
        op,
        ws,
        spec.dims,
        batch_size,
        num_params,
        plan,
        sum(length, values(pool); init = 0),
        nbytes,
        refs,
    )
    # the callback registry takes a lock: release it from the deferred queue, not the finalizer
    finalizer(L) do x
        cbs = [r for r in x._refs if r isa NamedTuple && haskey(r, :id)]
        _defer_destroy(x, 0, () -> foreach(unregister_callback!, cbs))
    end
    return L
end
