# Operator eigenspectrum computation
#
# Wraps cudensitymatOperatorSpectrum* API for computing eigenvalues and
# eigenstates of operators using a block Krylov algorithm.

export OperatorSpectrum, create_operator_spectrum, destroy_operator_spectrum
export configure_spectrum!, prepare_spectrum!, compute_spectrum!

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
        finalizer(_destroy!, obj)
        return obj
    end
end

function _destroy!(x::OperatorSpectrum)
    if x.handle != C_NULL
        try
            cudensitymatDestroyOperatorSpectrum(x.handle)
        catch
        end
        x.handle = C_NULL
    end
end

Base.close(x::OperatorSpectrum) = _destroy!(x)
Base.isopen(s::OperatorSpectrum) = s.handle != C_NULL

"""
    destroy_operator_spectrum(s::OperatorSpectrum)

Explicitly destroy the handle. Prefer `close(s)` instead.
"""
function destroy_operator_spectrum(s::OperatorSpectrum)
    return if s.handle != C_NULL
        cudensitymatDestroyOperatorSpectrum(s.handle)
        s.handle = C_NULL
    end
end

"""
    create_operator_spectrum(ws, operator;
        is_hermitian=true,
        spectrum_kind=CUDENSITYMAT_OPERATOR_SPECTRUM_LARGEST) -> OperatorSpectrum

Create an eigenspectrum solver for the given operator.

# Arguments
- `is_hermitian`: Whether the operator is Hermitian (enables optimized solver)
- `spectrum_kind`: Which eigenvalues to compute:
  - `CUDENSITYMAT_OPERATOR_SPECTRUM_LARGEST` — largest by magnitude
  - `CUDENSITYMAT_OPERATOR_SPECTRUM_SMALLEST` — smallest by magnitude
  - `CUDENSITYMAT_OPERATOR_SPECTRUM_LARGEST_REAL` — largest real part
  - `CUDENSITYMAT_OPERATOR_SPECTRUM_SMALLEST_REAL` — smallest real part
"""
function create_operator_spectrum(
        ws::WorkStream,
        operator::Operator;
        is_hermitian::Bool = true,
        spectrum_kind::cudensitymatOperatorSpectrumKind_t = CUDENSITYMAT_OPERATOR_SPECTRUM_LARGEST,
    )
    _check_valid(ws)
    spec_ref = Ref{cudensitymatOperatorSpectrum_t}()
    cudensitymatCreateOperatorSpectrum(
        ws.handle,
        operator.handle,
        Int32(is_hermitian ? 1 : 0),
        spectrum_kind,
        spec_ref,
    )
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
- `state`: A template state (defines the Hilbert space structure)
"""
function prepare_spectrum!(
        ws::WorkStream,
        spectrum::OperatorSpectrum,
        max_eigenstates::Integer,
        state::AbstractState;
        compute_type::cudensitymatComputeType_t = CUDENSITYMAT_COMPUTE_64F,
        workspace_limit::Union{Nothing, Integer} = nothing,
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
        CUDA.stream().handle,
    )
    required = workspace_query_size(ws)
    if required > 0
        workspace_allocate!(ws, required)
    end
    return nothing
end

"""
    compute_spectrum!(ws, spectrum, num_eigenstates, eigenstates, eigenvalues;
        time=0.0, batch_size=1, num_params=0, params=nothing) -> tolerances

Compute eigenvalues and eigenstates.

# Arguments
- `num_eigenstates`: Number of eigenvalues/eigenstates to compute (≤ max from prepare)
- `eigenstates`: Vector of pre-allocated `AbstractState`s to receive eigenstates
- `eigenvalues`: GPU array to receive eigenvalues. For Hermitian operators, use
  `CuVector{Float64}` of length `num_eigenstates * batch_size`. For non-Hermitian,
  use `CuVector{ComplexF64}`.

# Returns
- `tolerances`: Vector{Float64} of convergence tolerances for each eigenvalue
"""
function compute_spectrum!(
        ws::WorkStream,
        spectrum::OperatorSpectrum,
        num_eigenstates::Integer,
        eigenstates::AbstractVector{<:AbstractState},
        eigenvalues::CUDA.CuArray;
        time::Real = 0.0,
        batch_size::Integer = 1,
        num_params::Integer = 0,
        params::Union{Nothing, CUDA.CuVector{Float64}} = nothing,
    )
    _check_valid(ws)
    params_ptr = params === nothing ? CUDA.CU_NULL : pointer(params)
    state_handles = cudensitymatState_t[s.handle for s in eigenstates]
    tolerances = zeros(Float64, Int(num_eigenstates) * Int(batch_size))

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
        tolerances,
        ws.workspace,
        CUDA.stream().handle,
    )
    CUDA.synchronize()
    return tolerances
end
