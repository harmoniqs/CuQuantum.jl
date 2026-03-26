# Expectation value computation
#
# Wraps cudensitymatExpectation* API for computing expectation values
# of operators with respect to quantum states: Tr(O * ρ).

export Expectation, create_expectation, destroy_expectation

"""
    Expectation

Wraps a `cudensitymatExpectation_t` handle for computing expectation values
of an operator with respect to a quantum state.
"""
mutable struct Expectation
    handle::cudensitymatExpectation_t
    ws::WorkStream
    _operator_ref::Operator

    function Expectation(
        handle::cudensitymatExpectation_t,
        ws::WorkStream,
        operator::Operator,
    )
        obj = new(handle, ws, operator)
        finalizer(obj) do x
            if x.handle != C_NULL
                cudensitymatDestroyExpectation(x.handle)
                x.handle = C_NULL
            end
        end
        return obj
    end
end

Base.isopen(e::Expectation) = e.handle != C_NULL

function destroy_expectation(e::Expectation)
    if e.handle != C_NULL
        cudensitymatDestroyExpectation(e.handle)
        e.handle = C_NULL
    end
end

"""
    create_expectation(ws, operator) -> Expectation

Create an expectation value descriptor for the given operator.
"""
function create_expectation(ws::WorkStream, operator::Operator)
    _check_valid(ws)
    exp_ref = Ref{cudensitymatExpectation_t}()
    cudensitymatCreateExpectation(ws.handle, operator.handle, exp_ref)
    return Expectation(exp_ref[], ws, operator)
end

"""
    prepare_expectation!(ws, expectation, state;
        compute_type=CUDENSITYMAT_COMPUTE_64F, workspace_limit=nothing)

Prepare expectation value computation (one-time setup).
"""
function prepare_expectation!(
    ws::WorkStream,
    expectation::Expectation,
    state::AbstractState;
    compute_type::cudensitymatComputeType_t = CUDENSITYMAT_COMPUTE_64F,
    workspace_limit::Union{Nothing,Integer} = nothing,
)
    _check_valid(ws)
    mem_limit = _get_workspace_limit(ws, workspace_limit)
    cudensitymatExpectationPrepare(
        ws.handle,
        expectation.handle,
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
    compute_expectation!(ws, expectation, state, result;
        time=0.0, batch_size=1, num_params=0, params=nothing)

Compute expectation value: result = Tr(operator * state).

# Arguments
- `result`: GPU array to store the expectation value(s). For complex operators,
  use `CuVector{ComplexF64}` of length `batch_size`.
"""
function compute_expectation!(
    ws::WorkStream,
    expectation::Expectation,
    state::AbstractState,
    result::CUDA.CuArray;
    time::Real = 0.0,
    batch_size::Integer = 1,
    num_params::Integer = 0,
    params::Union{Nothing,CUDA.CuVector{Float64}} = nothing,
)
    _check_valid(ws)
    params_ptr = params === nothing ? CUDA.CU_NULL : pointer(params)
    cudensitymatExpectationCompute(
        ws.handle,
        expectation.handle,
        Cdouble(time),
        Int64(batch_size),
        Int32(num_params),
        params_ptr,
        state.handle,
        pointer(result),
        ws.workspace,
        CUDA.stream().handle,
    )
    CUDA.synchronize()
    return nothing
end
