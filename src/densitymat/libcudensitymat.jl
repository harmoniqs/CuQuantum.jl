# Low-level ccall wrappers for libcudensitymat
#
# Hand-written from the cuDensityMat C API reference.
# All functions use @checked (generating checked + unchecked variants)
# and @gcsafe_ccall (GC-safe foreign calls).
#
# Opaque handles are Ptr{Cvoid}. GPU buffers are CuPtr{Cvoid}.
# Callback wrapper structs are passed by value.
# cudensitymatGetVersion() returns size_t, not a status code.

# ============================================================================
# Library context management
# ============================================================================

@checked function cudensitymatCreate(handle)
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatCreate(
        handle::Ptr{cudensitymatHandle_t},
    )::cudensitymatStatus_t
end

@checked function cudensitymatDestroy(handle)
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatDestroy(
        handle::cudensitymatHandle_t,
    )::cudensitymatStatus_t
end

@checked function cudensitymatResetRandomSeed(handle, randomSeed)
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatResetRandomSeed(
        handle::cudensitymatHandle_t,
        randomSeed::Int32,
    )::cudensitymatStatus_t
end

# ============================================================================
# Distributed parallelization
# ============================================================================

@checked function cudensitymatResetDistributedConfiguration(
        handle,
        provider,
        commPtr,
        commSize,
    )
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatResetDistributedConfiguration(
        handle::cudensitymatHandle_t,
        provider::cudensitymatDistributedProvider_t,
        commPtr::Ptr{Cvoid},
        commSize::Csize_t,
    )::cudensitymatStatus_t
end

@checked function cudensitymatGetNumRanks(handle, numRanks)
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatGetNumRanks(
        handle::cudensitymatHandle_t,
        numRanks::Ptr{Int32},
    )::cudensitymatStatus_t
end

@checked function cudensitymatGetProcRank(handle, procRank)
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatGetProcRank(
        handle::cudensitymatHandle_t,
        procRank::Ptr{Int32},
    )::cudensitymatStatus_t
end

# ============================================================================
# Workspace management
# ============================================================================

@checked function cudensitymatCreateWorkspace(handle, workspaceDescr)
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatCreateWorkspace(
        handle::cudensitymatHandle_t,
        workspaceDescr::Ptr{cudensitymatWorkspaceDescriptor_t},
    )::cudensitymatStatus_t
end

@checked function cudensitymatDestroyWorkspace(workspaceDescr)
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatDestroyWorkspace(
        workspaceDescr::cudensitymatWorkspaceDescriptor_t,
    )::cudensitymatStatus_t
end

@checked function cudensitymatWorkspaceGetMemorySize(
        handle,
        workspaceDescr,
        memSpace,
        workspaceKind,
        memoryBufferSize,
    )
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatWorkspaceGetMemorySize(
        handle::cudensitymatHandle_t,
        workspaceDescr::cudensitymatWorkspaceDescriptor_t,
        memSpace::cudensitymatMemspace_t,
        workspaceKind::cudensitymatWorkspaceKind_t,
        memoryBufferSize::Ptr{Csize_t},
    )::cudensitymatStatus_t
end

@checked function cudensitymatWorkspaceSetMemory(
        handle,
        workspaceDescr,
        memSpace,
        workspaceKind,
        memoryBuffer,
        memoryBufferSize,
    )
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatWorkspaceSetMemory(
        handle::cudensitymatHandle_t,
        workspaceDescr::cudensitymatWorkspaceDescriptor_t,
        memSpace::cudensitymatMemspace_t,
        workspaceKind::cudensitymatWorkspaceKind_t,
        memoryBuffer::CuPtr{Cvoid},
        memoryBufferSize::Csize_t,
    )::cudensitymatStatus_t
end

@checked function cudensitymatWorkspaceGetMemory(
        handle,
        workspaceDescr,
        memSpace,
        workspaceKind,
        memoryBuffer,
        memoryBufferSize,
    )
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatWorkspaceGetMemory(
        handle::cudensitymatHandle_t,
        workspaceDescr::cudensitymatWorkspaceDescriptor_t,
        memSpace::cudensitymatMemspace_t,
        workspaceKind::cudensitymatWorkspaceKind_t,
        memoryBuffer::Ptr{CuPtr{Cvoid}},
        memoryBufferSize::Ptr{Csize_t},
    )::cudensitymatStatus_t
end

# ============================================================================
# Quantum state
# ============================================================================

@checked function cudensitymatCreateState(
        handle,
        purity,
        numSpaceModes,
        spaceModeExtents,
        batchSize,
        dataType,
        state,
    )
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatCreateState(
        handle::cudensitymatHandle_t,
        purity::cudensitymatStatePurity_t,
        numSpaceModes::Int32,
        spaceModeExtents::Ptr{Int64},
        batchSize::Int64,
        dataType::cudaDataType_t,
        state::Ptr{cudensitymatState_t},
    )::cudensitymatStatus_t
end

@checked function cudensitymatDestroyState(state)
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatDestroyState(
        state::cudensitymatState_t,
    )::cudensitymatStatus_t
end

@checked function cudensitymatStateGetNumComponents(handle, state, numStateComponents)
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatStateGetNumComponents(
        handle::cudensitymatHandle_t,
        state::cudensitymatState_t,
        numStateComponents::Ptr{Int32},
    )::cudensitymatStatus_t
end

@checked function cudensitymatStateGetComponentStorageSize(
        handle,
        state,
        numStateComponents,
        componentBufferSize,
    )
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatStateGetComponentStorageSize(
        handle::cudensitymatHandle_t,
        state::cudensitymatState_t,
        numStateComponents::Int32,
        componentBufferSize::Ptr{Csize_t},
    )::cudensitymatStatus_t
end

@checked function cudensitymatStateGetComponentNumModes(
        handle,
        state,
        stateComponentLocalId,
        stateComponentGlobalId,
        stateComponentNumModes,
        batchModeLocation,
    )
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatStateGetComponentNumModes(
        handle::cudensitymatHandle_t,
        state::cudensitymatState_t,
        stateComponentLocalId::Int32,
        stateComponentGlobalId::Ptr{Int32},
        stateComponentNumModes::Ptr{Int32},
        batchModeLocation::Ptr{Int32},
    )::cudensitymatStatus_t
end

@checked function cudensitymatStateGetComponentInfo(
        handle,
        state,
        stateComponentLocalId,
        stateComponentGlobalId,
        stateComponentNumModes,
        stateComponentModeExtents,
        stateComponentModeOffsets,
    )
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatStateGetComponentInfo(
        handle::cudensitymatHandle_t,
        state::cudensitymatState_t,
        stateComponentLocalId::Int32,
        stateComponentGlobalId::Ptr{Int32},
        stateComponentNumModes::Ptr{Int32},
        stateComponentModeExtents::Ptr{Int64},
        stateComponentModeOffsets::Ptr{Int64},
    )::cudensitymatStatus_t
end

@checked function cudensitymatStateAttachComponentStorage(
        handle,
        state,
        numStateComponents,
        componentBuffer,
        componentBufferSize,
    )
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatStateAttachComponentStorage(
        handle::cudensitymatHandle_t,
        state::cudensitymatState_t,
        numStateComponents::Int32,
        componentBuffer::Ptr{CuPtr{Cvoid}},
        componentBufferSize::Ptr{Csize_t},
    )::cudensitymatStatus_t
end

@checked function cudensitymatStateInitializeZero(handle, state, stream)
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatStateInitializeZero(
        handle::cudensitymatHandle_t,
        state::cudensitymatState_t,
        stream::cudaStream_t,
    )::cudensitymatStatus_t
end

@checked function cudensitymatStateComputeScaling(handle, state, scalingFactors, stream)
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatStateComputeScaling(
        handle::cudensitymatHandle_t,
        state::cudensitymatState_t,
        scalingFactors::CuPtr{Cvoid},
        stream::cudaStream_t,
    )::cudensitymatStatus_t
end

@checked function cudensitymatStateComputeNorm(handle, state, norm, stream)
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatStateComputeNorm(
        handle::cudensitymatHandle_t,
        state::cudensitymatState_t,
        norm::CuPtr{Cvoid},
        stream::cudaStream_t,
    )::cudensitymatStatus_t
end

@checked function cudensitymatStateComputeTrace(handle, state, trace, stream)
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatStateComputeTrace(
        handle::cudensitymatHandle_t,
        state::cudensitymatState_t,
        trace::CuPtr{Cvoid},
        stream::cudaStream_t,
    )::cudensitymatStatus_t
end

@checked function cudensitymatStateComputeAccumulation(
        handle,
        stateIn,
        stateOut,
        scalingFactors,
        stream,
    )
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatStateComputeAccumulation(
        handle::cudensitymatHandle_t,
        stateIn::cudensitymatState_t,
        stateOut::cudensitymatState_t,
        scalingFactors::CuPtr{Cvoid},
        stream::cudaStream_t,
    )::cudensitymatStatus_t
end

@checked function cudensitymatStateComputeInnerProduct(
        handle,
        stateLeft,
        stateRight,
        innerProduct,
        stream,
    )
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatStateComputeInnerProduct(
        handle::cudensitymatHandle_t,
        stateLeft::cudensitymatState_t,
        stateRight::cudensitymatState_t,
        innerProduct::CuPtr{Cvoid},
        stream::cudaStream_t,
    )::cudensitymatStatus_t
end

# ============================================================================
# Elementary operators
# ============================================================================

@checked function cudensitymatCreateElementaryOperator(
        handle,
        numSpaceModes,
        spaceModeExtents,
        sparsity,
        numDiagonals,
        diagonalOffsets,
        dataType,
        tensorData,
        tensorCallback,
        tensorGradientCallback,
        elemOperator,
    )
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatCreateElementaryOperator(
        handle::cudensitymatHandle_t,
        numSpaceModes::Int32,
        spaceModeExtents::Ptr{Int64},
        sparsity::cudensitymatElementaryOperatorSparsity_t,
        numDiagonals::Int32,
        diagonalOffsets::Ptr{Int32},
        dataType::cudaDataType_t,
        tensorData::CuPtr{Cvoid},
        tensorCallback::cudensitymatWrappedTensorCallback_t,
        tensorGradientCallback::cudensitymatWrappedTensorGradientCallback_t,
        elemOperator::Ptr{cudensitymatElementaryOperator_t},
    )::cudensitymatStatus_t
end

@checked function cudensitymatCreateElementaryOperatorBatch(
        handle,
        numSpaceModes,
        spaceModeExtents,
        batchSize,
        sparsity,
        numDiagonals,
        diagonalOffsets,
        dataType,
        tensorData,
        tensorCallback,
        tensorGradientCallback,
        elemOperator,
    )
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatCreateElementaryOperatorBatch(
        handle::cudensitymatHandle_t,
        numSpaceModes::Int32,
        spaceModeExtents::Ptr{Int64},
        batchSize::Int64,
        sparsity::cudensitymatElementaryOperatorSparsity_t,
        numDiagonals::Int32,
        diagonalOffsets::Ptr{Int32},
        dataType::cudaDataType_t,
        tensorData::CuPtr{Cvoid},
        tensorCallback::cudensitymatWrappedTensorCallback_t,
        tensorGradientCallback::cudensitymatWrappedTensorGradientCallback_t,
        elemOperator::Ptr{cudensitymatElementaryOperator_t},
    )::cudensitymatStatus_t
end

@checked function cudensitymatDestroyElementaryOperator(elemOperator)
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatDestroyElementaryOperator(
        elemOperator::cudensitymatElementaryOperator_t,
    )::cudensitymatStatus_t
end

# ============================================================================
# Matrix operators (full dense local)
# ============================================================================

@checked function cudensitymatCreateMatrixOperatorDenseLocal(
        handle,
        numSpaceModes,
        spaceModeExtents,
        dataType,
        matrixData,
        tensorCallback,
        tensorGradientCallback,
        matrixOperator,
    )
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatCreateMatrixOperatorDenseLocal(
        handle::cudensitymatHandle_t,
        numSpaceModes::Int32,
        spaceModeExtents::Ptr{Int64},
        dataType::cudaDataType_t,
        matrixData::CuPtr{Cvoid},
        tensorCallback::cudensitymatWrappedTensorCallback_t,
        tensorGradientCallback::cudensitymatWrappedTensorGradientCallback_t,
        matrixOperator::Ptr{cudensitymatMatrixOperator_t},
    )::cudensitymatStatus_t
end

@checked function cudensitymatCreateMatrixOperatorDenseLocalBatch(
        handle,
        numSpaceModes,
        spaceModeExtents,
        batchSize,
        dataType,
        matrixData,
        tensorCallback,
        tensorGradientCallback,
        matrixOperator,
    )
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatCreateMatrixOperatorDenseLocalBatch(
        handle::cudensitymatHandle_t,
        numSpaceModes::Int32,
        spaceModeExtents::Ptr{Int64},
        batchSize::Int64,
        dataType::cudaDataType_t,
        matrixData::CuPtr{Cvoid},
        tensorCallback::cudensitymatWrappedTensorCallback_t,
        tensorGradientCallback::cudensitymatWrappedTensorGradientCallback_t,
        matrixOperator::Ptr{cudensitymatMatrixOperator_t},
    )::cudensitymatStatus_t
end

@checked function cudensitymatDestroyMatrixOperator(matrixOperator)
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatDestroyMatrixOperator(
        matrixOperator::cudensitymatMatrixOperator_t,
    )::cudensitymatStatus_t
end

# ============================================================================
# Operator terms
# ============================================================================

@checked function cudensitymatCreateOperatorTerm(
        handle,
        numSpaceModes,
        spaceModeExtents,
        operatorTerm,
    )
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatCreateOperatorTerm(
        handle::cudensitymatHandle_t,
        numSpaceModes::Int32,
        spaceModeExtents::Ptr{Int64},
        operatorTerm::Ptr{cudensitymatOperatorTerm_t},
    )::cudensitymatStatus_t
end

@checked function cudensitymatDestroyOperatorTerm(operatorTerm)
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatDestroyOperatorTerm(
        operatorTerm::cudensitymatOperatorTerm_t,
    )::cudensitymatStatus_t
end

@checked function cudensitymatOperatorTermAppendElementaryProduct(
        handle,
        operatorTerm,
        numElemOperators,
        elemOperators,
        stateModesActedOn,
        modeActionDuality,
        coefficient,
        coefficientCallback,
        coefficientGradientCallback,
    )
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatOperatorTermAppendElementaryProduct(
        handle::cudensitymatHandle_t,
        operatorTerm::cudensitymatOperatorTerm_t,
        numElemOperators::Int32,
        elemOperators::Ptr{cudensitymatElementaryOperator_t},
        stateModesActedOn::Ptr{Int32},
        modeActionDuality::Ptr{Int32},
        coefficient::ComplexF64,
        coefficientCallback::cudensitymatWrappedScalarCallback_t,
        coefficientGradientCallback::cudensitymatWrappedScalarGradientCallback_t,
    )::cudensitymatStatus_t
end

@checked function cudensitymatOperatorTermAppendElementaryProductBatch(
        handle,
        operatorTerm,
        numElemOperators,
        elemOperators,
        stateModesActedOn,
        modeActionDuality,
        batchSize,
        staticCoefficients,
        totalCoefficients,
        coefficientCallback,
        coefficientGradientCallback,
    )
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatOperatorTermAppendElementaryProductBatch(
        handle::cudensitymatHandle_t,
        operatorTerm::cudensitymatOperatorTerm_t,
        numElemOperators::Int32,
        elemOperators::Ptr{cudensitymatElementaryOperator_t},
        stateModesActedOn::Ptr{Int32},
        modeActionDuality::Ptr{Int32},
        batchSize::Int64,
        staticCoefficients::CuPtr{ComplexF64},
        totalCoefficients::CuPtr{ComplexF64},
        coefficientCallback::cudensitymatWrappedScalarCallback_t,
        coefficientGradientCallback::cudensitymatWrappedScalarGradientCallback_t,
    )::cudensitymatStatus_t
end

@checked function cudensitymatOperatorTermAppendMatrixProduct(
        handle,
        operatorTerm,
        numMatrixOperators,
        matrixOperators,
        matrixConjugation,
        actionDuality,
        coefficient,
        coefficientCallback,
        coefficientGradientCallback,
    )
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatOperatorTermAppendMatrixProduct(
        handle::cudensitymatHandle_t,
        operatorTerm::cudensitymatOperatorTerm_t,
        numMatrixOperators::Int32,
        matrixOperators::Ptr{cudensitymatMatrixOperator_t},
        matrixConjugation::Ptr{Int32},
        actionDuality::Ptr{Int32},
        coefficient::ComplexF64,
        coefficientCallback::cudensitymatWrappedScalarCallback_t,
        coefficientGradientCallback::cudensitymatWrappedScalarGradientCallback_t,
    )::cudensitymatStatus_t
end

@checked function cudensitymatOperatorTermAppendMatrixProductBatch(
        handle,
        operatorTerm,
        numMatrixOperators,
        matrixOperators,
        matrixConjugation,
        actionDuality,
        batchSize,
        staticCoefficients,
        totalCoefficients,
        coefficientCallback,
        coefficientGradientCallback,
    )
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatOperatorTermAppendMatrixProductBatch(
        handle::cudensitymatHandle_t,
        operatorTerm::cudensitymatOperatorTerm_t,
        numMatrixOperators::Int32,
        matrixOperators::Ptr{cudensitymatMatrixOperator_t},
        matrixConjugation::Ptr{Int32},
        actionDuality::Ptr{Int32},
        batchSize::Int64,
        staticCoefficients::CuPtr{ComplexF64},
        totalCoefficients::CuPtr{ComplexF64},
        coefficientCallback::cudensitymatWrappedScalarCallback_t,
        coefficientGradientCallback::cudensitymatWrappedScalarGradientCallback_t,
    )::cudensitymatStatus_t
end

# ============================================================================
# Composite operators
# ============================================================================

@checked function cudensitymatCreateOperator(
        handle,
        numSpaceModes,
        spaceModeExtents,
        operator,
    )
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatCreateOperator(
        handle::cudensitymatHandle_t,
        numSpaceModes::Int32,
        spaceModeExtents::Ptr{Int64},
        operator::Ptr{cudensitymatOperator_t},
    )::cudensitymatStatus_t
end

@checked function cudensitymatDestroyOperator(operator)
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatDestroyOperator(
        operator::cudensitymatOperator_t,
    )::cudensitymatStatus_t
end

@checked function cudensitymatOperatorAppendTerm(
        handle,
        superoperator,
        operatorTerm,
        duality,
        coefficient,
        coefficientCallback,
        coefficientGradientCallback,
    )
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatOperatorAppendTerm(
        handle::cudensitymatHandle_t,
        superoperator::cudensitymatOperator_t,
        operatorTerm::cudensitymatOperatorTerm_t,
        duality::Int32,
        coefficient::ComplexF64,
        coefficientCallback::cudensitymatWrappedScalarCallback_t,
        coefficientGradientCallback::cudensitymatWrappedScalarGradientCallback_t,
    )::cudensitymatStatus_t
end

@checked function cudensitymatOperatorAppendTermBatch(
        handle,
        superoperator,
        operatorTerm,
        duality,
        batchSize,
        staticCoefficients,
        totalCoefficients,
        coefficientCallback,
        coefficientGradientCallback,
    )
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatOperatorAppendTermBatch(
        handle::cudensitymatHandle_t,
        superoperator::cudensitymatOperator_t,
        operatorTerm::cudensitymatOperatorTerm_t,
        duality::Int32,
        batchSize::Int64,
        staticCoefficients::CuPtr{ComplexF64},
        totalCoefficients::CuPtr{ComplexF64},
        coefficientCallback::cudensitymatWrappedScalarCallback_t,
        coefficientGradientCallback::cudensitymatWrappedScalarGradientCallback_t,
    )::cudensitymatStatus_t
end

# ============================================================================
# Single-operator action (prepare/compute on a single operator)
# ============================================================================

@checked function cudensitymatOperatorPrepareAction(
        handle,
        superoperator,
        stateIn,
        stateOut,
        computeType,
        workspaceSizeLimit,
        workspace,
        stream,
    )
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatOperatorPrepareAction(
        handle::cudensitymatHandle_t,
        superoperator::cudensitymatOperator_t,
        stateIn::cudensitymatState_t,
        stateOut::cudensitymatState_t,
        computeType::cudensitymatComputeType_t,
        workspaceSizeLimit::Csize_t,
        workspace::cudensitymatWorkspaceDescriptor_t,
        stream::cudaStream_t,
    )::cudensitymatStatus_t
end

@checked function cudensitymatOperatorComputeAction(
        handle,
        superoperator,
        time,
        batchSize,
        numParams,
        params,
        stateIn,
        stateOut,
        workspace,
        stream,
    )
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatOperatorComputeAction(
        handle::cudensitymatHandle_t,
        superoperator::cudensitymatOperator_t,
        time::Cdouble,
        batchSize::Int64,
        numParams::Int32,
        params::CuPtr{Cdouble},
        stateIn::cudensitymatState_t,
        stateOut::cudensitymatState_t,
        workspace::cudensitymatWorkspaceDescriptor_t,
        stream::cudaStream_t,
    )::cudensitymatStatus_t
end

# ============================================================================
# Single-operator backward differentiation
# ============================================================================

@checked function cudensitymatOperatorPrepareActionBackwardDiff(
        handle,
        superoperator,
        stateIn,
        stateOutAdj,
        computeType,
        workspaceSizeLimit,
        workspace,
        stream,
    )
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatOperatorPrepareActionBackwardDiff(
        handle::cudensitymatHandle_t,
        superoperator::cudensitymatOperator_t,
        stateIn::cudensitymatState_t,
        stateOutAdj::cudensitymatState_t,
        computeType::cudensitymatComputeType_t,
        workspaceSizeLimit::Csize_t,
        workspace::cudensitymatWorkspaceDescriptor_t,
        stream::cudaStream_t,
    )::cudensitymatStatus_t
end

@checked function cudensitymatOperatorComputeActionBackwardDiff(
        handle,
        superoperator,
        time,
        batchSize,
        numParams,
        params,
        stateIn,
        stateOutAdj,
        stateInAdj,
        paramsGrad,
        workspace,
        stream,
    )
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatOperatorComputeActionBackwardDiff(
        handle::cudensitymatHandle_t,
        superoperator::cudensitymatOperator_t,
        time::Cdouble,
        batchSize::Int64,
        numParams::Int32,
        params::CuPtr{Cdouble},
        stateIn::cudensitymatState_t,
        stateOutAdj::cudensitymatState_t,
        stateInAdj::cudensitymatState_t,
        paramsGrad::CuPtr{Cdouble},
        workspace::cudensitymatWorkspaceDescriptor_t,
        stream::cudaStream_t,
    )::cudensitymatStatus_t
end

# ============================================================================
# Multi-operator action (aggregate action descriptor)
# ============================================================================

@checked function cudensitymatCreateOperatorAction(
        handle,
        numOperators,
        operators,
        operatorAction,
    )
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatCreateOperatorAction(
        handle::cudensitymatHandle_t,
        numOperators::Int32,
        operators::Ptr{cudensitymatOperator_t},
        operatorAction::Ptr{cudensitymatOperatorAction_t},
    )::cudensitymatStatus_t
end

@checked function cudensitymatDestroyOperatorAction(operatorAction)
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatDestroyOperatorAction(
        operatorAction::cudensitymatOperatorAction_t,
    )::cudensitymatStatus_t
end

@checked function cudensitymatOperatorActionPrepare(
        handle,
        operatorAction,
        stateIn,
        stateOut,
        computeType,
        workspaceSizeLimit,
        workspace,
        stream,
    )
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatOperatorActionPrepare(
        handle::cudensitymatHandle_t,
        operatorAction::cudensitymatOperatorAction_t,
        stateIn::Ptr{cudensitymatState_t},
        stateOut::cudensitymatState_t,
        computeType::cudensitymatComputeType_t,
        workspaceSizeLimit::Csize_t,
        workspace::cudensitymatWorkspaceDescriptor_t,
        stream::cudaStream_t,
    )::cudensitymatStatus_t
end

@checked function cudensitymatOperatorActionCompute(
        handle,
        operatorAction,
        time,
        batchSize,
        numParams,
        params,
        stateIn,
        stateOut,
        workspace,
        stream,
    )
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatOperatorActionCompute(
        handle::cudensitymatHandle_t,
        operatorAction::cudensitymatOperatorAction_t,
        time::Cdouble,
        batchSize::Int64,
        numParams::Int32,
        params::CuPtr{Cdouble},
        stateIn::Ptr{cudensitymatState_t},
        stateOut::cudensitymatState_t,
        workspace::cudensitymatWorkspaceDescriptor_t,
        stream::cudaStream_t,
    )::cudensitymatStatus_t
end

# ============================================================================
# Expectation values
# ============================================================================

@checked function cudensitymatCreateExpectation(handle, superoperator, expectation)
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatCreateExpectation(
        handle::cudensitymatHandle_t,
        superoperator::cudensitymatOperator_t,
        expectation::Ptr{cudensitymatExpectation_t},
    )::cudensitymatStatus_t
end

@checked function cudensitymatDestroyExpectation(expectation)
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatDestroyExpectation(
        expectation::cudensitymatExpectation_t,
    )::cudensitymatStatus_t
end

@checked function cudensitymatExpectationPrepare(
        handle,
        expectation,
        state,
        computeType,
        workspaceSizeLimit,
        workspace,
        stream,
    )
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatExpectationPrepare(
        handle::cudensitymatHandle_t,
        expectation::cudensitymatExpectation_t,
        state::cudensitymatState_t,
        computeType::cudensitymatComputeType_t,
        workspaceSizeLimit::Csize_t,
        workspace::cudensitymatWorkspaceDescriptor_t,
        stream::cudaStream_t,
    )::cudensitymatStatus_t
end

@checked function cudensitymatExpectationCompute(
        handle,
        expectation,
        time,
        batchSize,
        numParams,
        params,
        state,
        expectationValue,
        workspace,
        stream,
    )
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatExpectationCompute(
        handle::cudensitymatHandle_t,
        expectation::cudensitymatExpectation_t,
        time::Cdouble,
        batchSize::Int64,
        numParams::Int32,
        params::CuPtr{Cdouble},
        state::cudensitymatState_t,
        expectationValue::CuPtr{Cvoid},
        workspace::cudensitymatWorkspaceDescriptor_t,
        stream::cudaStream_t,
    )::cudensitymatStatus_t
end

# ============================================================================
# Operator eigenspectrum
# ============================================================================

@checked function cudensitymatCreateOperatorSpectrum(
        handle,
        superoperator,
        isHermitian,
        spectrumKind,
        spectrum,
    )
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatCreateOperatorSpectrum(
        handle::cudensitymatHandle_t,
        superoperator::cudensitymatOperator_t,
        isHermitian::Int32,
        spectrumKind::cudensitymatOperatorSpectrumKind_t,
        spectrum::Ptr{cudensitymatOperatorSpectrum_t},
    )::cudensitymatStatus_t
end

@checked function cudensitymatDestroyOperatorSpectrum(spectrum)
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatDestroyOperatorSpectrum(
        spectrum::cudensitymatOperatorSpectrum_t,
    )::cudensitymatStatus_t
end

@checked function cudensitymatOperatorSpectrumConfigure(
        handle,
        spectrum,
        attribute,
        attributeValue,
        attributeValueSize,
    )
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatOperatorSpectrumConfigure(
        handle::cudensitymatHandle_t,
        spectrum::cudensitymatOperatorSpectrum_t,
        attribute::cudensitymatOperatorSpectrumConfig_t,
        attributeValue::Ptr{Cvoid},
        attributeValueSize::Csize_t,
    )::cudensitymatStatus_t
end

@checked function cudensitymatOperatorSpectrumPrepare(
        handle,
        spectrum,
        maxEigenStates,
        state,
        computeType,
        workspaceSizeLimit,
        workspace,
        stream,
    )
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatOperatorSpectrumPrepare(
        handle::cudensitymatHandle_t,
        spectrum::cudensitymatOperatorSpectrum_t,
        maxEigenStates::Int32,
        state::cudensitymatState_t,
        computeType::cudensitymatComputeType_t,
        workspaceSizeLimit::Csize_t,
        workspace::cudensitymatWorkspaceDescriptor_t,
        stream::cudaStream_t,
    )::cudensitymatStatus_t
end

@checked function cudensitymatOperatorSpectrumCompute(
        handle,
        spectrum,
        time,
        batchSize,
        numParams,
        params,
        numEigenStates,
        eigenstates,
        eigenvalues,
        tolerances,
        workspace,
        stream,
    )
    initialize_context()
    @gcsafe_ccall libcudensitymat.cudensitymatOperatorSpectrumCompute(
        handle::cudensitymatHandle_t,
        spectrum::cudensitymatOperatorSpectrum_t,
        time::Cdouble,
        batchSize::Int64,
        numParams::Int32,
        params::CuPtr{Cdouble},
        numEigenStates::Int32,
        eigenstates::Ptr{cudensitymatState_t},
        eigenvalues::CuPtr{Cvoid},
        tolerances::Ptr{Cdouble},
        workspace::cudensitymatWorkspaceDescriptor_t,
        stream::cudaStream_t,
    )::cudensitymatStatus_t
end

# ============================================================================
# Version query (NOT @checked — returns size_t, not status)
# ============================================================================

function cudensitymatGetVersion()
    return @gcsafe_ccall libcudensitymat.cudensitymatGetVersion()::Csize_t
end
