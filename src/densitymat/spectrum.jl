# Operator eigenspectrum computation
#
# Wraps cudensitymatOperatorSpectrum* API for computing eigenvalues and
# eigenstates of operators using a block Krylov algorithm.

export OperatorSpectrum, create_operator_spectrum, destroy_operator_spectrum

"""
    OperatorSpectrum

Wraps a `cudensitymatOperatorSpectrum_t` handle for computing eigenvalues
and eigenstates of an operator.
"""
mutable struct OperatorSpectrum
    handle::cudensitymatOperatorSpectrum_t
    ws::WorkStream
    _operator_ref::Operator

    function OperatorSpectrum(
        handle::cudensitymatOperatorSpectrum_t,
        ws::WorkStream,
        operator::Operator,
    )
        obj = new(handle, ws, operator)
        finalizer(obj) do x
            h = x.handle
            x.handle = C_NULL
            h != C_NULL &&
                _defer_destroy(x, 1, () -> cudensitymatDestroyOperatorSpectrum(h))
        end
        return obj
    end
end

Base.isopen(s::OperatorSpectrum) = s.handle != C_NULL

function destroy_operator_spectrum(s::OperatorSpectrum)
    if s.handle != C_NULL
        cudensitymatDestroyOperatorSpectrum(s.handle)
        s.handle = C_NULL
    end
end

"""
    create_operator_spectrum(ws, operator;
        is_hermitian=true,
        spectrum_kind=CUDENSITYMAT_OPERATOR_SPECTRUM_LARGEST_REAL) -> OperatorSpectrum

Create an extreme-eigenspectrum (block Krylov) solver for the given operator.

# Arguments
- `is_hermitian`: Whether the operator is Hermitian (the library trusts the flag)
- `spectrum_kind`: Which eigenvalues to compute:
  - `CUDENSITYMAT_OPERATOR_SPECTRUM_LARGEST_REAL` — largest real part (default)
  - `CUDENSITYMAT_OPERATOR_SPECTRUM_SMALLEST_REAL` — smallest real part
  - `CUDENSITYMAT_OPERATOR_SPECTRUM_LARGEST` — largest by magnitude
  - `CUDENSITYMAT_OPERATOR_SPECTRUM_SMALLEST` — smallest by magnitude

cuDensityMat 0.7 only implements Hermitian operators with the `*_REAL` kinds (the
same two that cuQuantum Python exposes, as `"LA"`/`"SA"`); the by-magnitude kinds
and `is_hermitian = false` make `cudensitymatCreateOperatorSpectrum` return
`CUDENSITYMAT_STATUS_NOT_SUPPORTED`, which is rethrown here as an `ArgumentError`
naming the supported configuration. The solver further needs non-batched dense
pure states in double precision, and a Hilbert space larger than the Krylov
subspace (see [`prepare_spectrum!`](@ref)).
"""
function create_operator_spectrum(
    ws::WorkStream,
    operator::Operator;
    is_hermitian::Bool = true,
    spectrum_kind::cudensitymatOperatorSpectrumKind_t = CUDENSITYMAT_OPERATOR_SPECTRUM_LARGEST_REAL,
)
    _check_valid(ws)
    spec_ref = Ref{cudensitymatOperatorSpectrum_t}()
    try
        cudensitymatCreateOperatorSpectrum(
            ws.handle,
            operator.handle,
            Int32(is_hermitian ? 1 : 0),
            spectrum_kind,
            spec_ref,
        )
    catch e
        e isa CUDENSITYMATError && e.code == CUDENSITYMAT_STATUS_NOT_SUPPORTED || rethrow()
        throw(
            ArgumentError(
                "cuDensityMat $(version()) does not support an operator spectrum with " *
                "is_hermitian=$is_hermitian, spectrum_kind=$spectrum_kind " *
                "(CUDENSITYMAT_STATUS_NOT_SUPPORTED); supported: is_hermitian=true with " *
                "CUDENSITYMAT_OPERATOR_SPECTRUM_LARGEST_REAL or " *
                "CUDENSITYMAT_OPERATOR_SPECTRUM_SMALLEST_REAL",
            ),
        )
    end
    return OperatorSpectrum(spec_ref[], ws, operator)
end

"""
    configure_spectrum!(ws, spectrum, attribute, value)

Configure a spectrum solver parameter.

# Arguments
- `attribute`: One of:
  - `CUDENSITYMAT_OPERATOR_SPECTRUM_CONFIG_MAX_EXPANSION` — max Krylov expansion ratio (Int32, default 5)
  - `CUDENSITYMAT_OPERATOR_SPECTRUM_CONFIG_MAX_RESTARTS` — max restarts (Int32, default 20)
  - `CUDENSITYMAT_OPERATOR_SPECTRUM_CONFIG_MIN_BLOCK_SIZE` — min block size (Int32, default 1)
- `value`: Int32 configuration value
"""
function configure_spectrum!(
    ws::WorkStream,
    spectrum::OperatorSpectrum,
    attribute::cudensitymatOperatorSpectrumConfig_t,
    value::Integer,
)
    _check_valid(ws)
    val = Ref{Int32}(Int32(value))
    cudensitymatOperatorSpectrumConfigure(
        ws.handle,
        spectrum.handle,
        attribute,
        val,
        Csize_t(sizeof(Int32)),
    )
    return nothing
end

"""
    prepare_spectrum!(ws, spectrum, max_eigenstates, state;
        compute_type=CUDENSITYMAT_COMPUTE_64F, workspace_limit=nothing)

Prepare eigenspectrum computation.

# Arguments
- `max_eigenstates`: Maximum number of eigenvalues/eigenstates to compute
- `state`: A template state (defines the Hilbert space structure); a non-batched
  `DensePureState{ComplexF64}` or `DensePureState{Float64}`

cuDensityMat 0.7 rejects Hilbert spaces that are small relative to the Krylov
subspace (whose size grows with `max_eigenstates` and
`CUDENSITYMAT_OPERATOR_SPECTRUM_CONFIG_MAX_EXPANSION`, default 5); the library logs
"The Hilbert space dimension must be greater than or equal to the sum of twice the
Krylov subspace dimension and the minimum Krylov subspace block size!". A single
qubit is too small; the tests use a 16-level mode.
"""
function prepare_spectrum!(
    ws::WorkStream,
    spectrum::OperatorSpectrum,
    max_eigenstates::Integer,
    state::AbstractState;
    compute_type::cudensitymatComputeType_t = CUDENSITYMAT_COMPUTE_64F,
    workspace_limit::Union{Nothing,Integer} = nothing,
)
    _check_valid(ws)
    mem_limit = _get_workspace_limit(ws, workspace_limit)
    cudensitymatOperatorSpectrumPrepare(
        ws.handle,
        spectrum.handle,
        Int32(max_eigenstates),
        state.handle,
        compute_type,
        Csize_t(mem_limit),
        ws.workspace,
        CUDACore.stream().handle,
    )
    required = workspace_query_size(ws)
    if required > 0
        workspace_allocate!(ws, required)
    end
    return nothing
end

"""
    compute_spectrum!(ws, spectrum, num_eigenstates, eigenstates, eigenvalues;
        time=0.0, batch_size=1, num_params=0, params=nothing,
        tolerances=fill(sqrt(eps(Float64)), num_eigenstates * batch_size)) -> tolerances

Compute eigenvalues and eigenstates.

# Arguments
- `num_eigenstates`: Number of eigenvalues/eigenstates to compute (≤ max from prepare)
- `eigenstates`: Vector of `num_eigenstates` pre-allocated `AbstractState`s. Their
  contents are the initial guesses of the first Krylov block and must be linearly
  independent (in particular, not zero); on return they hold the eigenstates.
- `eigenvalues`: GPU array of length `num_eigenstates * batch_size` with the element
  type of the states (`ComplexF64` for complex states, even for Hermitian operators;
  the library writes values of the state data type).
- `tolerances`: Requested convergence tolerances (CPU, `Float64`), one per eigenstate.

# Returns
- The achieved solver residual norms, one per eigenstate (`Vector{Float64}`).
"""
function compute_spectrum!(
    ws::WorkStream,
    spectrum::OperatorSpectrum,
    num_eigenstates::Integer,
    eigenstates::AbstractVector{<:AbstractState{T}},
    eigenvalues::CUDACore.CuArray;
    time::Real = 0.0,
    batch_size::Integer = 1,
    num_params::Integer = 0,
    params::Union{Nothing,CUDACore.CuVector{Float64}} = nothing,
    tolerances::AbstractVector{<:Real} = fill(
        sqrt(eps(Float64)),
        Int(num_eigenstates) * Int(batch_size),
    ),
) where {T}
    _check_valid(ws)
    eltype(eigenvalues) == T || throw(
        ArgumentError(
            "eigenvalues must have the element type of the states ($T), got $(eltype(eigenvalues))",
        ),
    )
    length(eigenvalues) >= num_eigenstates * batch_size || throw(
        ArgumentError(
            "eigenvalues has $(length(eigenvalues)) elements, need $(num_eigenstates * batch_size)",
        ),
    )
    length(tolerances) == num_eigenstates * batch_size || throw(
        ArgumentError(
            "tolerances has $(length(tolerances)) elements, need $(num_eigenstates * batch_size)",
        ),
    )
    params_ptr = params === nothing ? CUDACore.CU_NULL : pointer(params)
    state_handles = cudensitymatState_t[s.handle for s in eigenstates]
    residuals = Float64.(tolerances)

    cudensitymatOperatorSpectrumCompute(
        ws.handle,
        spectrum.handle,
        Cdouble(time),
        Int64(batch_size),
        Int32(num_params),
        params_ptr,
        Int32(num_eigenstates),
        state_handles,
        pointer(eigenvalues),
        residuals,
        ws.workspace,
        CUDACore.stream().handle,
    )
    CUDACore.synchronize()
    return residuals
end
