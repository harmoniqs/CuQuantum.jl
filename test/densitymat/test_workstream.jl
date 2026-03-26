# Phase 1: WorkStream tests (local, single-GPU)
#
# Mirrors Python's test_work_stream.py
# 7 tests covering handle lifecycle, workspace management, configuration

@testset "WorkStream (local)" begin

    @gpu_test "default construction" begin
        ws = WorkStream()
        @test isopen(ws)
        @test ws.handle != C_NULL
        @test ws.workspace != C_NULL
        @test ws.device_id >= 0
        @test ws.comm_set == false
        @test ws.comm_provider == CuDensityMat.CUDENSITYMAT_DISTRIBUTED_PROVIDER_NONE
        close(ws)
    end

    @gpu_test "explicit stream" begin
        s = CUDA.CuStream()
        ws = WorkStream(stream=s)
        @test isopen(ws)
        @test ws.stream === s
        close(ws)
    end

    @gpu_test "close releases resources" begin
        ws = WorkStream()
        @test isopen(ws)
        close(ws)
        @test !isopen(ws)
        # Double close should be safe (no-op)
        close(ws)
        @test !isopen(ws)
    end

    @gpu_test "finalizer cleanup" begin
        # Create and immediately lose reference — finalizer should clean up
        ws_handle = let
            ws = WorkStream()
            h = ws.handle
            @test h != C_NULL
            h
        end
        GC.gc(true)
        # Can't directly assert the handle was destroyed, but no crash = success
        @test true
    end

    @gpu_test "random seed" begin
        ws = WorkStream()
        # Setting random seed should not error
        CuDensityMat.set_random_seed!(ws, 42)
        CuDensityMat.set_random_seed!(ws, 0)
        CuDensityMat.set_random_seed!(ws, typemax(Int32))
        close(ws)
    end

    @gpu_test "workspace release and recreate" begin
        ws = WorkStream()
        old_workspace = ws.workspace
        @test old_workspace != C_NULL

        # Release workspace — should create a fresh descriptor
        CuDensityMat.release_workspace!(ws)
        @test isopen(ws)
        @test ws.workspace != C_NULL
        @test ws.workspace_size == 0
        @test ws.workspace_buffer === nothing

        close(ws)
    end

    @gpu_test "version query" begin
        v = CuDensityMat.version()
        @test v isa VersionNumber
        @test v.major >= 0
        # The version should be reasonable (cuQuantum SDK 24.x or 25.x)
        @test v.major >= 24 || v.major == 0  # 0 for dev builds
    end

end
