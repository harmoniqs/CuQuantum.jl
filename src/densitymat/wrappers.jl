# Convenience wrappers for cuDensityMat C API
#
# These provide Julia-idiomatic interfaces on top of the raw ccall bindings.
# Pattern: create() → handle, destroy(handle) → nothing, version() → VersionNumber

"""
    version()

Return the CuDensityMat library version as a `VersionNumber`.
"""
function version()
    v = cudensitymatGetVersion()
    major = div(v, 10000)
    minor = div(v % 10000, 100)
    patch = v % 100
    return VersionNumber(major, minor, patch)
end

"""
    create() -> cudensitymatHandle_t

Create a CuDensityMat library context handle.
"""
function create()
    handle_ref = Ref{cudensitymatHandle_t}()
    cudensitymatCreate(handle_ref)
    return handle_ref[]
end

"""
    destroy(handle::cudensitymatHandle_t)

Destroy a CuDensityMat library context handle.
"""
function destroy(handle::cudensitymatHandle_t)
    return cudensitymatDestroy(handle)
end

"""
    create_workspace(handle) -> cudensitymatWorkspaceDescriptor_t

Create a workspace descriptor.
"""
function create_workspace(handle::cudensitymatHandle_t)
    workspace_ref = Ref{cudensitymatWorkspaceDescriptor_t}()
    cudensitymatCreateWorkspace(handle, workspace_ref)
    return workspace_ref[]
end

"""
    destroy_workspace(workspace)

Destroy a workspace descriptor.
"""
function destroy_workspace(workspace::cudensitymatWorkspaceDescriptor_t)
    return cudensitymatDestroyWorkspace(workspace)
end
