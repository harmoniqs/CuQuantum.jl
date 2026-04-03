using Aqua
using CuQuantum

@testset "Aqua quality checks" begin
    Aqua.test_all(
        CuQuantum;
        # Ambiguity check is noisy with CUDA.jl extensions; enable after
        # all LinearAlgebra extensions are properly disambiguated.
        ambiguities = false,
        unbound_args = true,
        undefined_exports = true,
        stale_deps = (ignore = [:CUDA_Runtime_Discovery, Symbol("cuQuantum_jll")],),
        deps_compat = true,
    )
end
