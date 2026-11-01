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
        ws = WorkStream(stream = s)
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
        # Can't directly assert the handle was destroyed; no crash, and a fresh
        # WorkStream is still created after the finalizer ran
        ws2 = WorkStream()
        @test isopen(ws2)
        close(ws2)
    end

    @gpu_test "random seed" begin
        ws = WorkStream()
        # Setting random seed should not error (previously this testset had no
        # assertion at all and was reported with 0 tests)
        for seed in (42, 0, typemax(Int32))
            @test CuDensityMat.set_random_seed!(ws, seed) === nothing
        end
        @test isopen(ws)
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
        # libcudensitymat's own version (0.7.x with cuQuantum_jll 26.9), not the SDK's
        @test v >= v"0.7.0"
    end

end
