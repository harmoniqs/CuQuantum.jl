# Phase 1: WorkStream MPI/NCCL distributed tests
#
# Mirrors Python's test_work_stream_mpi.py
# 8 tests covering MPI and NCCL communicator setup
#
# These tests are designed to be run with:
#   mpiexecjl -n 2 julia --project test/mpi_runtests.jl
#
# They will be skipped when run in non-MPI mode.

# Detect MPI availability once
const HAS_MPI = try
    @eval import MPI as _MPI_mod
    true
catch
    false
end

@testset "WorkStream (MPI distributed)" begin

    @gpu_test "MPI communicator from comm object" begin
        if !HAS_MPI
            @test_skip "MPI.jl not available"
        else
            if !_MPI_mod.Initialized()
                _MPI_mod.Init()
            end

            comm = _MPI_mod.COMM_WORLD
            rank = _MPI_mod.Comm_rank(comm)
            nranks = _MPI_mod.Comm_size(comm)

            ws = WorkStream()

            comm_ptr = Ptr{Cvoid}(_MPI_mod.API.MPI_Comm_c2f(comm))
            comm_size = sizeof(_MPI_mod.MPI_Comm)

            CuDensityMat.set_communicator!(
                ws,
                :mpi;
                comm_ptr = comm_ptr,
                comm_size = comm_size,
            )

            @test CuDensityMat.get_num_ranks(ws) == nranks
            @test CuDensityMat.get_proc_rank(ws) == rank
            @test ws.comm_set == true

            close(ws)
        end
    end

    @gpu_test "MPI communicator from integer pointer" begin
        if !HAS_MPI
            @test_skip "MPI.jl not available"
        else
            if !_MPI_mod.Initialized()
                _MPI_mod.Init()
            end

            comm = _MPI_mod.COMM_WORLD
            rank = _MPI_mod.Comm_rank(comm)
            nranks = _MPI_mod.Comm_size(comm)

            ws = WorkStream()

            comm_ptr = UInt(_MPI_mod.API.MPI_Comm_c2f(comm))
            comm_size = sizeof(_MPI_mod.MPI_Comm)

            CuDensityMat.set_communicator!(
                ws,
                :mpi;
                comm_ptr = comm_ptr,
                comm_size = comm_size,
            )

            @test CuDensityMat.get_num_ranks(ws) == nranks
            @test CuDensityMat.get_proc_rank(ws) == rank

            close(ws)
        end
    end

    @gpu_test "communicator cannot be set twice" begin
        if !HAS_MPI
            @test_skip "MPI.jl not available"
        else
            if !_MPI_mod.Initialized()
                _MPI_mod.Init()
            end

            comm = _MPI_mod.COMM_WORLD
            ws = WorkStream()

            comm_ptr = Ptr{Cvoid}(_MPI_mod.API.MPI_Comm_c2f(comm))
            comm_size = sizeof(_MPI_mod.MPI_Comm)

            CuDensityMat.set_communicator!(
                ws,
                :mpi;
                comm_ptr = comm_ptr,
                comm_size = comm_size,
            )

            # Second set should error
            @test_throws ErrorException CuDensityMat.set_communicator!(
                ws,
                :mpi;
                comm_ptr = comm_ptr,
                comm_size = comm_size,
            )

            close(ws)
        end
    end

    @gpu_test "no-provider communicator" begin
        ws = WorkStream()
        CuDensityMat.set_communicator!(ws, :none)
        @test ws.comm_set == true
        close(ws)
    end

    @gpu_test "invalid provider" begin
        ws = WorkStream()
        @test_throws ErrorException CuDensityMat.set_communicator!(ws, :invalid)
        close(ws)
    end

    @gpu_test "operations on closed WorkStream" begin
        ws = WorkStream()
        close(ws)
        @test_throws ErrorException CuDensityMat.get_num_ranks(ws)
        @test_throws ErrorException CuDensityMat.get_proc_rank(ws)
        @test_throws ErrorException CuDensityMat.set_communicator!(ws, :none)
    end

    @gpu_test "NCCL communicator from integer pointer" begin
        if !HAS_MPI
            @test_skip "MPI.jl not available for NCCL bootstrap"
        else
            # NCCL communicator setup would require NCCL.jl bindings
            # For now, test the error path (missing comm_ptr)
            ws = WorkStream()
            @test_throws ErrorException CuDensityMat.set_communicator!(ws, :nccl)
            close(ws)
        end
    end

    @gpu_test "rank queries without communicator" begin
        ws = WorkStream()
        # Without a communicator set, should return 1 rank, rank 0
        @test CuDensityMat.get_num_ranks(ws) == 1
        @test CuDensityMat.get_proc_rank(ws) == 0
        close(ws)
    end

end
