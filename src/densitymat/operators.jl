# Operator hierarchy: ElementaryOperator → OperatorTerm → Operator → OperatorAction
#
# Wraps the cuDensityMat C API for building quantum operators (Hamiltonians, Lindbladians).
# Operators are composed from elementary single-mode operators via tensor products,
# then assembled into terms and composite operators for action computation.

export ElementaryOperator, create_elementary_operator, destroy_elementary_operator
export MatrixOperator, create_matrix_operator, destroy_matrix_operator
export OperatorTerm, create_operator_term, destroy_operator_term
export Operator, create_operator, destroy_operator
export OperatorAction, create_operator_action, destroy_operator_action

# =============================================================================
# No-callback sentinel (null callback structs)
# =============================================================================

const NULL_TENSOR_CALLBACK =
    cudensitymatWrappedTensorCallback_t(C_NULL, CUDENSITYMAT_CALLBACK_DEVICE_CPU, C_NULL)
const NULL_TENSOR_GRADIENT_CALLBACK = cudensitymatWrappedTensorGradientCallback_t(
    C_NULL,
    CUDENSITYMAT_CALLBACK_DEVICE_CPU,
    C_NULL,
    CUDENSITYMAT_DIFFERENTIATION_DIR_BACKWARD,
)
const NULL_SCALAR_CALLBACK =
    cudensitymatWrappedScalarCallback_t(C_NULL, CUDENSITYMAT_CALLBACK_DEVICE_CPU, C_NULL)
const NULL_SCALAR_GRADIENT_CALLBACK = cudensitymatWrappedScalarGradientCallback_t(
    C_NULL,
    CUDENSITYMAT_CALLBACK_DEVICE_CPU,
    C_NULL,
    CUDENSITYMAT_DIFFERENTIATION_DIR_BACKWARD,
)

# =============================================================================
# Workspace memory limit helper
# =============================================================================

"""
    _get_workspace_limit(ws, user_limit) -> Int

Return the workspace size limit in bytes. If `user_limit` is `nothing`,
returns `ws.memory_limit` if set, or 80% of free GPU memory.
"""
function _get_workspace_limit(ws::WorkStream, user_limit::Union{Nothing, Integer})
    if user_limit !== nothing
        return Int(user_limit)
    end
    if ws.memory_limit !== nothing
        return ws.memory_limit
    end
    # Default: 80% of free GPU memory (matches Python WorkStream default)
    free_mem = CUDA.free_memory()
    return Int(floor(0.8 * free_mem))
end

# =============================================================================
# ElementaryOperator — single-mode operator (dense or multidiagonal)
# =============================================================================

"""
    ElementaryOperator

Wraps a `cudensitymatElementaryOperator_t` handle. Represents a single-mode
operator that can be dense or multidiagonal.

Create via [`create_elementary_operator`](@ref) or [`create_elementary_operator_batch`](@ref).
"""
mutable struct ElementaryOperator
    handle::cudensitymatElementaryOperator_t
    ws::WorkStream
    # Keep references to data to prevent GC
    _data_ref::Any
    _callback_refs::Any

    function ElementaryOperator(
            handle::cudensitymatElementaryOperator_t,
            ws::WorkStream;
            data_ref = nothing,
            callback_refs = nothing,
        )
        obj = new(handle, ws, data_ref, callback_refs)
        finalizer(_destroy!, obj)
        return obj
    end
end

function _destroy!(x::ElementaryOperator)
    if x.handle != C_NULL
        try
            cudensitymatDestroyElementaryOperator(x.handle)
        catch
        end
        x.handle = C_NULL
    end
end

Base.close(x::ElementaryOperator) = _destroy!(x)
Base.isopen(op::ElementaryOperator) = op.handle != C_NULL

"""
    create_elementary_operator(ws, space_mode_extents, data::CuArray{T};
        sparsity=:none, diagonal_offsets=Int32[],
        tensor_callback=nothing, tensor_gradient_callback=nothing) -> ElementaryOperator

Create a dense or multidiagonal elementary operator for a single mode.

# Arguments
- `ws::WorkStream`: Execution context
- `space_mode_extents::Vector{Int}`: Dimensions of the space modes (e.g., `[d]` for single qudit)
- `data::CuArray{T}`: Operator data on GPU (column-major)
- `sparsity`: `:none` for dense, `:multidiagonal` for sparse
- `diagonal_offsets`: Required for multidiagonal sparsity
"""
function create_elementary_operator(
        ws::WorkStream,
        space_mode_extents::AbstractVector{<:Integer},
        data::CUDA.CuArray{T};
        sparsity::Symbol = :none,
        diagonal_offsets::AbstractVector{<:Integer} = Int32[],
        tensor_callback::cudensitymatWrappedTensorCallback_t = NULL_TENSOR_CALLBACK,
        tensor_gradient_callback::cudensitymatWrappedTensorGradientCallback_t = NULL_TENSOR_GRADIENT_CALLBACK,
    ) where {T}
    _check_valid(ws)

    sp =
        sparsity == :none ? CUDENSITYMAT_OPERATOR_SPARSITY_NONE :
        sparsity == :multidiagonal ? CUDENSITYMAT_OPERATOR_SPARSITY_MULTIDIAGONAL :
        error("Unknown sparsity: $sparsity")

    extents = Int64[e for e in space_mode_extents]
    offsets = Int32[o for o in diagonal_offsets]
    num_diags = Int32(length(offsets))

    op_ref = Ref{cudensitymatElementaryOperator_t}()
    cudensitymatCreateElementaryOperator(
        ws.handle,
        Int32(length(extents)),
        extents,
        sp,
        num_diags,
        isempty(offsets) ? C_NULL : offsets,
        julia_to_cuda_dtype(T),
        pointer(data),
        tensor_callback,
        tensor_gradient_callback,
        op_ref,
    )

    return ElementaryOperator(op_ref[], ws; data_ref = data)
end

"""
    create_elementary_operator_batch(ws, space_mode_extents, data::CuArray{T}, batch_size;
        kwargs...) -> ElementaryOperator

Create a batched elementary operator.
"""
function create_elementary_operator_batch(
        ws::WorkStream,
        space_mode_extents::AbstractVector{<:Integer},
        data::CUDA.CuArray{T},
        batch_size::Integer;
        sparsity::Symbol = :none,
        diagonal_offsets::AbstractVector{<:Integer} = Int32[],
        tensor_callback::cudensitymatWrappedTensorCallback_t = NULL_TENSOR_CALLBACK,
        tensor_gradient_callback::cudensitymatWrappedTensorGradientCallback_t = NULL_TENSOR_GRADIENT_CALLBACK,
    ) where {T}
    _check_valid(ws)

    sp =
        sparsity == :none ? CUDENSITYMAT_OPERATOR_SPARSITY_NONE :
        sparsity == :multidiagonal ? CUDENSITYMAT_OPERATOR_SPARSITY_MULTIDIAGONAL :
        error("Unknown sparsity: $sparsity")

    extents = Int64[e for e in space_mode_extents]
    offsets = Int32[o for o in diagonal_offsets]
    num_diags = Int32(length(offsets))

    op_ref = Ref{cudensitymatElementaryOperator_t}()
    cudensitymatCreateElementaryOperatorBatch(
        ws.handle,
        Int32(length(extents)),
        extents,
        Int64(batch_size),
        sp,
        num_diags,
        isempty(offsets) ? C_NULL : offsets,
        julia_to_cuda_dtype(T),
        pointer(data),
        tensor_callback,
        tensor_gradient_callback,
        op_ref,
    )

    return ElementaryOperator(op_ref[], ws; data_ref = data)
end

function destroy_elementary_operator(op::ElementaryOperator)
    return if op.handle != C_NULL
        cudensitymatDestroyElementaryOperator(op.handle)
        op.handle = C_NULL
    end
end

# =============================================================================
# MatrixOperator — full Hilbert-space dense local matrix operator
# =============================================================================

"""
    MatrixOperator

Wraps a `cudensitymatMatrixOperator_t` handle. Represents a full Hilbert-space
dense matrix operator stored locally on a single GPU.
"""
mutable struct MatrixOperator
    handle::cudensitymatMatrixOperator_t
    ws::WorkStream
    _data_ref::Any
    _callback_refs::Any

    function MatrixOperator(
            handle::cudensitymatMatrixOperator_t,
            ws::WorkStream;
            data_ref = nothing,
            callback_refs = nothing,
        )
        obj = new(handle, ws, data_ref, callback_refs)
        finalizer(_destroy!, obj)
        return obj
    end
end

function _destroy!(x::MatrixOperator)
    if x.handle != C_NULL
        try
            cudensitymatDestroyMatrixOperator(x.handle)
        catch
        end
        x.handle = C_NULL
    end
end

Base.close(x::MatrixOperator) = _destroy!(x)
Base.isopen(op::MatrixOperator) = op.handle != C_NULL

"""
    create_matrix_operator(ws, space_mode_extents, data::CuArray{T};
        tensor_callback=nothing, tensor_gradient_callback=nothing) -> MatrixOperator

Create a dense local matrix operator covering the full Hilbert space.
"""
function create_matrix_operator(
        ws::WorkStream,
        space_mode_extents::AbstractVector{<:Integer},
        data::CUDA.CuArray{T};
        tensor_callback::cudensitymatWrappedTensorCallback_t = NULL_TENSOR_CALLBACK,
        tensor_gradient_callback::cudensitymatWrappedTensorGradientCallback_t = NULL_TENSOR_GRADIENT_CALLBACK,
    ) where {T}
    _check_valid(ws)
    extents = Int64[e for e in space_mode_extents]

    op_ref = Ref{cudensitymatMatrixOperator_t}()
    cudensitymatCreateMatrixOperatorDenseLocal(
        ws.handle,
        Int32(length(extents)),
        extents,
        julia_to_cuda_dtype(T),
        pointer(data),
        tensor_callback,
        tensor_gradient_callback,
        op_ref,
    )

    return MatrixOperator(op_ref[], ws; data_ref = data)
end

"""
    create_matrix_operator_batch(ws, space_mode_extents, data::CuArray{T}, batch_size;
        kwargs...) -> MatrixOperator

Create a batched dense local matrix operator.
"""
function create_matrix_operator_batch(
        ws::WorkStream,
        space_mode_extents::AbstractVector{<:Integer},
        data::CUDA.CuArray{T},
        batch_size::Integer;
        tensor_callback::cudensitymatWrappedTensorCallback_t = NULL_TENSOR_CALLBACK,
        tensor_gradient_callback::cudensitymatWrappedTensorGradientCallback_t = NULL_TENSOR_GRADIENT_CALLBACK,
    ) where {T}
    _check_valid(ws)
    extents = Int64[e for e in space_mode_extents]

    op_ref = Ref{cudensitymatMatrixOperator_t}()
    cudensitymatCreateMatrixOperatorDenseLocalBatch(
        ws.handle,
        Int32(length(extents)),
        extents,
        Int64(batch_size),
        julia_to_cuda_dtype(T),
        pointer(data),
        tensor_callback,
        tensor_gradient_callback,
        op_ref,
    )

    return MatrixOperator(op_ref[], ws; data_ref = data)
end

function destroy_matrix_operator(op::MatrixOperator)
    return if op.handle != C_NULL
        cudensitymatDestroyMatrixOperator(op.handle)
        op.handle = C_NULL
    end
end

# =============================================================================
# OperatorTerm — sum of tensor products of elementary operators
# =============================================================================

"""
    OperatorTerm

Wraps a `cudensitymatOperatorTerm_t` handle. Represents one or more tensor products
of elementary operators acting on specified modes.
"""
mutable struct OperatorTerm
    handle::cudensitymatOperatorTerm_t
    ws::WorkStream
    hilbert_space_dims::Vector{Int64}
    # Keep refs to prevent GC of constituent operators
    _elem_op_refs::Vector{Any}
    _matrix_op_refs::Vector{Any}
    _callback_refs::Vector{Any}

    function OperatorTerm(
            handle::cudensitymatOperatorTerm_t,
            ws::WorkStream,
            dims::Vector{Int64},
        )
        obj = new(handle, ws, dims, Any[], Any[], Any[])
        finalizer(_destroy!, obj)
        return obj
    end
end

function _destroy!(x::OperatorTerm)
    if x.handle != C_NULL
        try
            cudensitymatDestroyOperatorTerm(x.handle)
        catch
        end
        x.handle = C_NULL
    end
end

Base.close(x::OperatorTerm) = _destroy!(x)
Base.isopen(t::OperatorTerm) = t.handle != C_NULL

"""
    create_operator_term(ws, hilbert_space_dims) -> OperatorTerm

Create an empty operator term for the given Hilbert space dimensions.
"""
function create_operator_term(ws::WorkStream, hilbert_space_dims::AbstractVector{<:Integer})
    _check_valid(ws)
    dims = Int64[d for d in hilbert_space_dims]
    term_ref = Ref{cudensitymatOperatorTerm_t}()
    cudensitymatCreateOperatorTerm(ws.handle, Int32(length(dims)), dims, term_ref)
    return OperatorTerm(term_ref[], ws, dims)
end

"""
    append_elementary_product!(term, elem_operators, modes_acted_on, mode_action_duality;
        coefficient=1.0+0im, kwargs...)

Append a tensor product of elementary operators to the term.

# Arguments
- `elem_operators`: Vector of `ElementaryOperator`s
- `modes_acted_on`: Flattened vector of mode indices (0-based) each operator acts on
- `mode_action_duality`: Per-mode duality flags (0=ket, nonzero=bra) — same length as `modes_acted_on`
- `coefficient`: Static complex scalar coefficient (default: 1.0+0im)
"""
function append_elementary_product!(
        term::OperatorTerm,
        elem_operators::AbstractVector{ElementaryOperator},
        modes_acted_on::AbstractVector{<:Integer},
        mode_action_duality::AbstractVector{<:Integer};
        coefficient::Number = ComplexF64(1.0),
        coefficient_callback::cudensitymatWrappedScalarCallback_t = NULL_SCALAR_CALLBACK,
        coefficient_gradient_callback::cudensitymatWrappedScalarGradientCallback_t = NULL_SCALAR_GRADIENT_CALLBACK,
    )
    handles = cudensitymatElementaryOperator_t[op.handle for op in elem_operators]
    modes = Int32[m for m in modes_acted_on]
    duality = Int32[d for d in mode_action_duality]

    cudensitymatOperatorTermAppendElementaryProduct(
        term.ws.handle,
        term.handle,
        Int32(length(handles)),
        handles,
        modes,
        duality,
        ComplexF64(coefficient),
        coefficient_callback,
        coefficient_gradient_callback,
    )

    # Keep references
    append!(term._elem_op_refs, elem_operators)
    return nothing
end

"""
    append_elementary_product_batch!(term, elem_operators, modes_acted_on,
        mode_action_duality, batch_size, static_coefficients; kwargs...)

Append a batched tensor product of elementary operators to the term.

# Arguments
- `static_coefficients`: GPU array of ComplexF64 coefficients (length batch_size)
- `total_coefficients`: GPU storage for total coefficients, or `nothing`
"""
function append_elementary_product_batch!(
        term::OperatorTerm,
        elem_operators::AbstractVector{ElementaryOperator},
        modes_acted_on::AbstractVector{<:Integer},
        mode_action_duality::AbstractVector{<:Integer},
        batch_size::Integer,
        static_coefficients::CUDA.CuVector{ComplexF64};
        total_coefficients::Union{Nothing, CUDA.CuVector{ComplexF64}} = nothing,
        coefficient_callback::cudensitymatWrappedScalarCallback_t = NULL_SCALAR_CALLBACK,
        coefficient_gradient_callback::cudensitymatWrappedScalarGradientCallback_t = NULL_SCALAR_GRADIENT_CALLBACK,
    )
    handles = cudensitymatElementaryOperator_t[op.handle for op in elem_operators]
    modes = Int32[m for m in modes_acted_on]
    duality = Int32[d for d in mode_action_duality]
    total_ptr = total_coefficients === nothing ? CUDA.CU_NULL : pointer(total_coefficients)

    cudensitymatOperatorTermAppendElementaryProductBatch(
        term.ws.handle,
        term.handle,
        Int32(length(handles)),
        handles,
        modes,
        duality,
        Int64(batch_size),
        pointer(static_coefficients),
        total_ptr,
        coefficient_callback,
        coefficient_gradient_callback,
    )

    append!(term._elem_op_refs, elem_operators)
    return nothing
end

"""
    append_matrix_product!(term, matrix_operators, conjugations, action_duality;
        coefficient=1.0+0im, kwargs...)

Append a product of matrix operators to the term.

# Arguments
- `matrix_operators`: Vector of `MatrixOperator`s
- `conjugations`: Per-operator conjugation flags (0=normal, nonzero=conjugate-transpose)
- `action_duality`: Per-operator duality flags (0=ket, nonzero=bra)
- `coefficient`: Static complex scalar coefficient (default: 1.0+0im)
"""
function append_matrix_product!(
        term::OperatorTerm,
        matrix_operators::AbstractVector{MatrixOperator},
        conjugations::AbstractVector{<:Integer},
        action_duality::AbstractVector{<:Integer};
        coefficient::Number = ComplexF64(1.0),
        coefficient_callback::cudensitymatWrappedScalarCallback_t = NULL_SCALAR_CALLBACK,
        coefficient_gradient_callback::cudensitymatWrappedScalarGradientCallback_t = NULL_SCALAR_GRADIENT_CALLBACK,
    )
    handles = cudensitymatMatrixOperator_t[op.handle for op in matrix_operators]
    conj = Int32[c for c in conjugations]
    dual = Int32[d for d in action_duality]

    cudensitymatOperatorTermAppendMatrixProduct(
        term.ws.handle,
        term.handle,
        Int32(length(handles)),
        handles,
        conj,
        dual,
        ComplexF64(coefficient),
        coefficient_callback,
        coefficient_gradient_callback,
    )

    append!(term._matrix_op_refs, matrix_operators)
    return nothing
end

function destroy_operator_term(term::OperatorTerm)
    return if term.handle != C_NULL
        cudensitymatDestroyOperatorTerm(term.handle)
        term.handle = C_NULL
    end
end

# =============================================================================
# Operator — composite operator (collection of terms with duality + coefficients)
# =============================================================================

"""
    Operator

Wraps a `cudensitymatOperator_t` handle. A composite operator consisting of
multiple `OperatorTerm`s, each with a duality flag and scalar coefficient.
"""
mutable struct Operator
    handle::cudensitymatOperator_t
    ws::WorkStream
    hilbert_space_dims::Vector{Int64}
    _term_refs::Vector{Any}

    function Operator(handle::cudensitymatOperator_t, ws::WorkStream, dims::Vector{Int64})
        obj = new(handle, ws, dims, Any[])
        finalizer(_destroy!, obj)
        return obj
    end
end

function _destroy!(x::Operator)
    if x.handle != C_NULL
        try
            cudensitymatDestroyOperator(x.handle)
        catch
        end
        x.handle = C_NULL
    end
end

Base.close(x::Operator) = _destroy!(x)
Base.isopen(op::Operator) = op.handle != C_NULL

"""
    create_operator(ws, hilbert_space_dims) -> Operator

Create an empty composite operator for the given Hilbert space.
"""
function create_operator(ws::WorkStream, hilbert_space_dims::AbstractVector{<:Integer})
    _check_valid(ws)
    dims = Int64[d for d in hilbert_space_dims]
    op_ref = Ref{cudensitymatOperator_t}()
    cudensitymatCreateOperator(ws.handle, Int32(length(dims)), dims, op_ref)
    return Operator(op_ref[], ws, dims)
end

"""
    append_term!(operator, term; duality=0, coefficient=1.0+0im, kwargs...)

Append an operator term to the composite operator.

# Arguments
- `duality`: 0 for ket-side (default), nonzero for bra-side
- `coefficient`: Static complex scalar coefficient (default: 1.0+0im)
"""
function append_term!(
        op::Operator,
        term::OperatorTerm;
        duality::Integer = 0,
        coefficient::Number = ComplexF64(1.0),
        coefficient_callback::cudensitymatWrappedScalarCallback_t = NULL_SCALAR_CALLBACK,
        coefficient_gradient_callback::cudensitymatWrappedScalarGradientCallback_t = NULL_SCALAR_GRADIENT_CALLBACK,
    )
    cudensitymatOperatorAppendTerm(
        op.ws.handle,
        op.handle,
        term.handle,
        Int32(duality),
        ComplexF64(coefficient),
        coefficient_callback,
        coefficient_gradient_callback,
    )
    push!(op._term_refs, (term, coefficient))
    return nothing
end

"""
    append_term_batch!(operator, term, batch_size, static_coefficients;
        duality=0, kwargs...)

Append a batched operator term.

# Arguments
- `static_coefficients`: GPU array of ComplexF64 coefficients (length batch_size)
- `total_coefficients`: GPU storage for total coefficients, or `nothing`
"""
function append_term_batch!(
        op::Operator,
        term::OperatorTerm,
        batch_size::Integer,
        static_coefficients::CUDA.CuVector{ComplexF64};
        duality::Integer = 0,
        total_coefficients::Union{Nothing, CUDA.CuVector{ComplexF64}} = nothing,
        coefficient_callback::cudensitymatWrappedScalarCallback_t = NULL_SCALAR_CALLBACK,
        coefficient_gradient_callback::cudensitymatWrappedScalarGradientCallback_t = NULL_SCALAR_GRADIENT_CALLBACK,
    )
    total_ptr = total_coefficients === nothing ? CUDA.CU_NULL : pointer(total_coefficients)
    cudensitymatOperatorAppendTermBatch(
        op.ws.handle,
        op.handle,
        term.handle,
        Int32(duality),
        Int64(batch_size),
        pointer(static_coefficients),
        total_ptr,
        coefficient_callback,
        coefficient_gradient_callback,
    )
    push!(op._term_refs, (term, static_coefficients, batch_size))
    return nothing
end

function destroy_operator(op::Operator)
    return if op.handle != C_NULL
        cudensitymatDestroyOperator(op.handle)
        op.handle = C_NULL
    end
end

# =============================================================================
# OperatorAction — action of operators on states
# =============================================================================

"""
    OperatorAction

Wraps a `cudensitymatOperatorAction_t` handle. Computes the action of one or more
operators on input states, accumulating into an output state.
"""
mutable struct OperatorAction
    handle::cudensitymatOperatorAction_t
    ws::WorkStream
    _operator_refs::Vector{Operator}

    function OperatorAction(
            handle::cudensitymatOperatorAction_t,
            ws::WorkStream,
            operators::Vector{Operator},
        )
        obj = new(handle, ws, operators)
        finalizer(_destroy!, obj)
        return obj
    end
end

function _destroy!(x::OperatorAction)
    if x.handle != C_NULL
        try
            cudensitymatDestroyOperatorAction(x.handle)
        catch
        end
        x.handle = C_NULL
    end
end

Base.close(x::OperatorAction) = _destroy!(x)
Base.isopen(a::OperatorAction) = a.handle != C_NULL

function destroy_operator_action(action::OperatorAction)
    return if action.handle != C_NULL
        cudensitymatDestroyOperatorAction(action.handle)
        action.handle = C_NULL
    end
end

"""
    create_operator_action(ws, operators::Vector{Operator}) -> OperatorAction

Create an operator action descriptor for multiple operators.
"""
function create_operator_action(ws::WorkStream, operators::Vector{Operator})
    _check_valid(ws)
    handles = cudensitymatOperator_t[op.handle for op in operators]
    action_ref = Ref{cudensitymatOperatorAction_t}()
    cudensitymatCreateOperatorAction(ws.handle, Int32(length(handles)), handles, action_ref)
    return OperatorAction(action_ref[], ws, operators)
end

"""
    prepare_action!(ws, action, states_in, state_out;
        compute_type=CUDENSITYMAT_COMPUTE_64F, workspace_limit=nothing)

Prepare the operator action computation (one-time setup).

If `workspace_limit` is `nothing` (default), uses 80% of free GPU memory.
"""
function prepare_action!(
        ws::WorkStream,
        action::OperatorAction,
        states_in::AbstractVector{<:AbstractState},
        state_out::AbstractState;
        compute_type::cudensitymatComputeType_t = CUDENSITYMAT_COMPUTE_64F,
        workspace_limit::Union{Nothing, Integer} = nothing,
    )
    _check_valid(ws)
    mem_limit = _get_workspace_limit(ws, workspace_limit)
    in_handles = cudensitymatState_t[s.handle for s in states_in]
    cudensitymatOperatorActionPrepare(
        ws.handle,
        action.handle,
        in_handles,
        state_out.handle,
        compute_type,
        Csize_t(mem_limit),
        ws.workspace,
        CUDA.stream().handle,
    )
    # Query and allocate workspace
    required = workspace_query_size(ws)
    if required > 0
        workspace_allocate!(ws, required)
    end
    return nothing
end

"""
    compute_action!(ws, action, states_in, state_out;
        time=0.0, batch_size=0, params=nothing)

Execute the operator action: `state_out = sum_i operator_i(states_in[i])`.
"""
function compute_action!(
        ws::WorkStream,
        action::OperatorAction,
        states_in::AbstractVector{<:AbstractState},
        state_out::AbstractState;
        time::Real = 0.0,
        batch_size::Integer = 1,
        num_params::Integer = 0,
        params::Union{Nothing, CUDA.CuVector{Float64}} = nothing,
    )
    _check_valid(ws)
    in_handles = cudensitymatState_t[s.handle for s in states_in]
    params_ptr = params === nothing ? CUDA.CU_NULL : pointer(params)
    cudensitymatOperatorActionCompute(
        ws.handle,
        action.handle,
        Cdouble(time),
        Int64(batch_size),
        Int32(num_params),
        params_ptr,
        in_handles,
        state_out.handle,
        ws.workspace,
        CUDA.stream().handle,
    )
    CUDA.synchronize()
    return nothing
end

# =============================================================================
# Single-operator action convenience (uses Operator directly, not OperatorAction)
# =============================================================================

"""
    prepare_operator_action!(ws, operator, state_in, state_out;
        compute_type=CUDENSITYMAT_COMPUTE_64F, workspace_limit=nothing)

Prepare single-operator action computation.

If `workspace_limit` is `nothing` (default), uses 80% of free GPU memory.
"""
function prepare_operator_action!(
        ws::WorkStream,
        operator::Operator,
        state_in::AbstractState,
        state_out::AbstractState;
        compute_type::cudensitymatComputeType_t = CUDENSITYMAT_COMPUTE_64F,
        workspace_limit::Union{Nothing, Integer} = nothing,
    )
    _check_valid(ws)
    mem_limit = _get_workspace_limit(ws, workspace_limit)
    cudensitymatOperatorPrepareAction(
        ws.handle,
        operator.handle,
        state_in.handle,
        state_out.handle,
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
    compute_operator_action!(ws, operator, state_in, state_out;
        time=0.0, batch_size=0, params=nothing)

Execute single-operator action: `state_out = operator * state_in`.
"""
function compute_operator_action!(
        ws::WorkStream,
        operator::Operator,
        state_in::AbstractState,
        state_out::AbstractState;
        time::Real = 0.0,
        batch_size::Integer = 1,
        num_params::Integer = 0,
        params::Union{Nothing, CUDA.CuVector{Float64}} = nothing,
    )
    _check_valid(ws)
    params_ptr = params === nothing ? CUDA.CU_NULL : pointer(params)
    cudensitymatOperatorComputeAction(
        ws.handle,
        operator.handle,
        Cdouble(time),
        Int64(batch_size),
        Int32(num_params),
        params_ptr,
        state_in.handle,
        state_out.handle,
        ws.workspace,
        CUDA.stream().handle,
    )
    CUDA.synchronize()
    return nothing
end

# =============================================================================
# Backward differentiation (single-GPU only)
# =============================================================================

"""
    prepare_operator_action_backward!(ws, operator, state_in, state_out_adj;
        compute_type=CUDENSITYMAT_COMPUTE_64F, workspace_limit=nothing)

Prepare backward differentiation of a single-operator action.

# Arguments
- `state_in`: The forward-pass input state
- `state_out_adj`: The adjoint of the forward-pass output state (∂L/∂state_out)
"""
function prepare_operator_action_backward!(
        ws::WorkStream,
        operator::Operator,
        state_in::AbstractState,
        state_out_adj::AbstractState;
        compute_type::cudensitymatComputeType_t = CUDENSITYMAT_COMPUTE_64F,
        workspace_limit::Union{Nothing, Integer} = nothing,
    )
    _check_valid(ws)
    mem_limit = _get_workspace_limit(ws, workspace_limit)
    cudensitymatOperatorPrepareActionBackwardDiff(
        ws.handle,
        operator.handle,
        state_in.handle,
        state_out_adj.handle,
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
    compute_operator_action_backward!(ws, operator, state_in, state_out_adj,
        state_in_adj, params_grad; time=0.0, batch_size=0,
        num_params=0, params=nothing)

Compute backward differentiation of a single-operator action.

Computes ∂L/∂state_in (into `state_in_adj`) and ∂L/∂params (into `params_grad`)
given ∂L/∂state_out (`state_out_adj`).

# Arguments
- `state_in`: Forward-pass input state
- `state_out_adj`: Adjoint of forward-pass output (∂L/∂state_out)
- `state_in_adj`: Output — receives ∂L/∂state_in (accumulated, += semantics)
- `params_grad`: GPU vector — receives ∂L/∂params (accumulated, += semantics)
"""
function compute_operator_action_backward!(
        ws::WorkStream,
        operator::Operator,
        state_in::AbstractState,
        state_out_adj::AbstractState,
        state_in_adj::AbstractState,
        params_grad::CUDA.CuVector{Float64};
        time::Real = 0.0,
        batch_size::Integer = 1,
        num_params::Integer = 0,
        params::Union{Nothing, CUDA.CuVector{Float64}} = nothing,
    )
    _check_valid(ws)
    params_ptr = params === nothing ? CUDA.CU_NULL : pointer(params)
    cudensitymatOperatorComputeActionBackwardDiff(
        ws.handle,
        operator.handle,
        Cdouble(time),
        Int64(batch_size),
        Int32(num_params),
        params_ptr,
        state_in.handle,
        state_out_adj.handle,
        state_in_adj.handle,
        pointer(params_grad),
        ws.workspace,
        CUDA.stream().handle,
    )
    CUDA.synchronize()
    return nothing
end
