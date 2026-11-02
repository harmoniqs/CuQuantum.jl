# Shared models / helpers for test_lindblad.jl (host only; no GPU calls).

annihilation(d) = Matrix{ComplexF64}(diagm(1 => sqrt.(1.0:(d-1))))

maxrel(x, ref) = maximum(abs, x .- ref) / maximum(abs, ref)

# Deterministic pseudo-random numbers without an RNG dependency (xorshift64*).
function pseudo_randn(N::Integer, seed::Integer)
    s = UInt64(seed) * 0x9E3779B97F4A7C15 + 0x2545F4914F6CDD1D
    u() = (
        s ⊻= s >> 12;
        s ⊻= s << 25;
        s ⊻= s >> 27;
        ((s * 0x2545F4914F6CDD1D) >> 11) / 2.0^53
    )
    return [complex(u() - 0.5, u() - 0.5) for _ = 1:N]
end

"""Batch of B states (vec of n×n each, concatenated); optionally Hermitian PSD with unit trace."""
function random_states(n, B; seed = 1, hermitian = false, unit_trace = false)
    x = pseudo_randn(n * n * B, seed)
    for b = 1:B
        X = reshape(view(x, ((b-1)*n*n+1):(b*n*n)), n, n)
        if hermitian
            X .= X * X'
            unit_trace && (X ./= tr(X))
        end
    end
    return x
end

batch_traces(x, n, B) = [tr(reshape(x[((b-1)*n*n+1):(b*n*n)], n, n)) for b = 1:B]

"""Two-mode model: H = -K(a†²a² + b†²b²) + g(a†b + ab†); rate2·D[a²-shift] (rate or rate(t,p)), rate1·D[a], 2·dephase·D[a†a] per mode."""
function two_mode_model(
    c;
    g = 0.3,
    shift = 1.0,
    rate2 = 1.0,
    rate1 = 0.0,
    dephase = 0.0,
    K = 0.0,
    rate = nothing,
)
    a = annihilation(c)
    spec = LindbladSpec([c, c])
    if K != 0
        add_hamiltonian!(spec, -K, 1 => a' * a' * a * a)
        add_hamiltonian!(spec, -K, 2 => a' * a' * a * a)
    end
    add_hamiltonian!(spec, g, 1 => a', 2 => a)
    add_hamiltonian!(spec, g, 1 => a, 2 => a')
    for m = 1:2
        add_jump!(spec, rate === nothing ? rate2 : rate, m => a * a - shift * I)
        rate1 > 0 && add_jump!(spec, rate1, m => a)
        dephase > 0 && add_jump!(spec, 2dephase, m => a' * a)
    end
    return spec, (g isa Number ? fill(Float64(real(g)), 1, 1) : zeros(1, 1))
end

single_mode_loss(c, shift; rate2 = 1.0) =
    add_jump!(LindbladSpec([c]), rate2, 1 => annihilation(c)^2 - shift * I)

"""Models for the action tests: (name, spec, params (num_params × batch), t)."""
function test_models()
    models = Any[]
    # 1: two modes of different extents, complex time-dependent coupling, non-Hermitian
    #    complex jump, a batched rate function and a two-mode jump.
    a, b = annihilation(4), annihilation(5)
    s1 = LindbladSpec([4, 5])
    add_hamiltonian!(s1, 0.3, 1 => a' * a' * a * a)
    add_hamiltonian!(s1, -0.45, 2 => b' * b)
    f = (t, p) -> p[1] * cis(2t)
    fc = (t, p) -> p[1] * cis(-2t)
    add_hamiltonian!(s1, f, 1 => a', 2 => b)
    add_hamiltonian!(s1, fc, 1 => a, 2 => b')
    add_jump!(s1, 0.7, 1 => a + 0.3im * a' * a)
    add_jump!(s1, (t, p) -> p[2]^2, 2 => b)
    add_jump!(s1, 0.2, 1 => a, 2 => b' * b)
    push!(models, ("two-mode complex [4,5]", s1, [0.4 -0.9; 1.1 0.3], 0.7))
    # 2: two-mode Kerr + beam splitter + losses at c = 6, time-dependent g shared by both coupling products
    s2, _ = two_mode_model(
        6;
        g = (t, p) -> p[1] * cos(t),
        shift = 2.0,
        rate1 = 0.1,
        dephase = 0.05,
        K = 0.2,
    )
    push!(models, ("two-mode Kerr + loss [6,6]", s2, [0.5 -1.3], 0.4))
    # 3: one mode, complex drive c a† + c̄ a
    d = annihilation(5)
    s3 = LindbladSpec([5])
    add_hamiltonian!(s3, (t, p) -> p[1] * cis(t), 1 => d')
    add_hamiltonian!(s3, (t, p) -> p[1] * cis(-t), 1 => d)
    add_jump!(s3, 1.0, 1 => d * d - 1.5I)
    push!(models, ("one mode [5]", s3, fill(0.8, 1, 1), 1.3))
    return models
end

"""Negative control: the same spec with every Hamiltonian coefficient negated (time reversal of the unitary part)."""
function conj_spec(spec::LindbladSpec)
    s = LindbladSpec(spec.dims)
    for (f, c, fs) in spec.hamiltonian
        f === nothing ? push!(s.hamiltonian, (nothing, -c, fs)) :
        push!(s.hamiltonian, ((t, p) -> -f(t, p), c, fs))
    end
    append!(s.jumps, spec.jumps)
    return s
end

"""Negative control (not trace preserving): the host executor of `spec` without the LρL† sandwich products."""
function drop_sandwich(L::CpuLindbladian)
    groups = Any[
        (f, [p for p in prods if p[2] === nothing || p[3] === nothing]) for
        (f, prods) in L.groups
    ]
    return CpuLindbladian(L.dims, L.n, L.batch_size, L.num_params, groups, L.tmp, L.tmp2)
end
