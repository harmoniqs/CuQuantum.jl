# Binding / ABI checks that need no GPU.
#
# The generated low-level layer (src/densitymat/libcudensitymat.jl) must match the
# cuDensityMat shared library and header shipped by the pinned cuQuantum_jll.

using Libdl
import cuQuantum_jll

# All C entry points the module wraps (excluding the `unchecked_` twins of @checked).
const WRAPPED_C_FUNCTIONS = sort!([
    String(n) for n in names(CuDensityMat; all = true) if
    startswith(String(n), "cudensitymat") && getfield(CuDensityMat, n) isa Function
])

# Live (not commented-out) function declarations in a C header.
function declared_c_functions(header)
    src = read(header, String)
    src = replace(src, r"/\*.*?\*/"s => "", r"//[^\n]*" => "")
    return sort!(unique!([m[1] for m in eachmatch(r"\b(cudensitymat\w+)\s*\(", src)]))
end

const LIBRARY_LOADED = CuDensityMat.libcudensitymat !== nothing
const HEADER =
    cuQuantum_jll.is_available() ?
    joinpath(cuQuantum_jll.artifact_dir, "include", "cudensitymat.h") : ""

@testset "Bindings (CPU)" begin

    @testset "dependency lockstep with CUDA.jl" begin
        # CUDA.jl 6.5's lib/custatevec and lib/cutensornet pin cuQuantum_jll 26.9
        @test pkgversion(cuQuantum_jll) >= v"26.9"
        @test pkgversion(CUDA) >= v"6.5"
    end

    @testset "library version" begin
        if LIBRARY_LOADED
            # GetVersion needs no CUDA context: usable on any machine with the library
            @test CuDensityMat.cudensitymatGetVersion() >= 700
            @test CuDensityMat.version() >= v"0.7.0"
            @test CuDensityMat.version() >= CuDensityMat.MIN_LIBRARY_VERSION
        else
            @test_skip "libcudensitymat not loadable on this platform"
        end
    end

    @testset "every wrapped function is exported by libcudensitymat" begin
        @test length(WRAPPED_C_FUNCTIONS) == 110
        if LIBRARY_LOADED
            lib = Libdl.dlopen(CuDensityMat.libcudensitymat)
            missing_syms = filter(WRAPPED_C_FUNCTIONS) do fn
                Libdl.dlsym(lib, fn; throw_error = false) === nothing
            end
            @test isempty(missing_syms)
        else
            @test_skip "libcudensitymat not loadable on this platform"
        end
    end

    @testset "bindings cover the pinned header" begin
        if isfile(HEADER)
            @test declared_c_functions(HEADER) == WRAPPED_C_FUNCTIONS
            # the generated file records the jll it was generated from
            generated = read(
                joinpath(pkgdir(CuQuantum), "src", "densitymat", "libcudensitymat.jl"),
                String,
            )
            @test occursin("cuQuantum_jll v$(pkgversion(cuQuantum_jll))", generated)
        else
            @test_skip "cuQuantum_jll artifact (headers) not available on this platform"
        end
    end

    @testset "cudensitymatCreateOperatorAction ABI (cuDensityMat >= 0.7)" begin
        # (handle, numOperators, operators, scopeKind, approachKind, operatorAction)
        for f in (
            CuDensityMat.cudensitymatCreateOperatorAction,
            CuDensityMat.unchecked_cudensitymatCreateOperatorAction,
        )
            @test only(methods(f)).nargs - 1 == 6
        end
        @test Integer(CuDensityMat.CUDENSITYMAT_FITTING_SCOPE_FULL) == 0
        @test Integer(CuDensityMat.CUDENSITYMAT_FITTING_SCOPE_SPLIT) == 1
        @test Integer(CuDensityMat.CUDENSITYMAT_FITTING_APPROACH_LINSOLVE) == 0
        @test sizeof(CuDensityMat.cudensitymatStateFittingScopeKind_t) == 4
        @test sizeof(CuDensityMat.cudensitymatStateFittingApproachKind_t) == 4
    end

    @testset "high-level operator action takes scope/approach" begin
        m = only(methods(CuDensityMat.create_operator_action))
        @test :scope in Base.kwarg_decl(m)
        @test :approach in Base.kwarg_decl(m)
    end

    @testset "struct layouts" begin
        S = CuDensityMat.cudensitymatWrappedScalarCallback_t
        G = CuDensityMat.cudensitymatWrappedScalarGradientCallback_t
        @test sizeof(S) == 24
        @test fieldoffset.(S, 1:3) == UInt[0, 8, 16]
        @test sizeof(CuDensityMat.cudensitymatWrappedTensorCallback_t) == 24
        @test sizeof(G) == 32
        @test fieldoffset.(G, 1:4) == UInt[0, 8, 16, 24]
        @test sizeof(CuDensityMat.cudensitymatWrappedTensorGradientCallback_t) == 32
        @test sizeof(CuDensityMat.cudensitymatDistributedCommunicator_t) == 16
    end

    @testset "newly wrapped 0.7 API surface" begin
        for fn in (
            # MPS / MPO
            :cudensitymatCreateStateMPS,
            :cudensitymatStateMPSGetCurrentBondExtents,
            :cudensitymatStateMPSSetCurrentBondExtents,
            :cudensitymatCreateMatrixProductOperator,
            :cudensitymatOperatorTermAppendMPOProduct,
            # time propagation (TDVP / Krylov)
            :cudensitymatCreateTimePropagation,
            :cudensitymatCreateTimePropagationScopeSplitTDVPConfig,
            :cudensitymatCreateTimePropagationApproachKrylovConfig,
            :cudensitymatTimePropagationPrepare,
            :cudensitymatTimePropagationCompute,
            # eigen-decomposition (DMRG / Krylov / linear)
            :cudensitymatCreateEigenDecomposition,
            :cudensitymatEigenDecompositionCompute,
            # state fitting + SVD
            :cudensitymatCreateStateFittingScopeSplitALSConfig,
            :cudensitymatCreateSVDConfig,
            :cudensitymatOperatorActionConfigure,
            # batched coefficients + buffer attach
            :cudensitymatAttachBatchedCoefficients,
            :cudensitymatElementaryOperatorAttachBuffer,
            :cudensitymatMatrixOperatorDenseLocalAttachBuffer,
        )
            @test isdefined(CuDensityMat, fn)
        end
    end
end
