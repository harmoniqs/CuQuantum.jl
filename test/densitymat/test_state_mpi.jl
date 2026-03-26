# Phase 2: State MPI distributed tests
#
# Mirrors Python's test_state_mpi.py
# Tests state creation, storage, and compute operations in distributed mode.
#
# These tests require MPI and are skipped when MPI.jl is not available.

@testset "State (MPI distributed)" begin

    @gpu_test "distributed pure state creation" begin
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

            for T in [ComplexF64, ComplexF32]
                for dims in [(2, 2, 2)]
                    for bs in [1, 2]
                        psi = DensePureState{T}(ws, dims; batch_size = bs)
                        shape, offsets = CuDensityMat.local_info(psi)
                        @test length(shape) == length(offsets)
                        # Pure: ndims = len(dims) + batch
                        expected_modes = length(dims) + 1
                        @test length(shape) == expected_modes

                        sz = CuDensityMat.storage_size(psi)
                        buf = CUDA.zeros(T, sz)
                        CuDensityMat.attach_storage!(psi, buf)
                        @test psi.storage !== nothing
                    end
                end
            end

            close(ws)
        end
    end

    @gpu_test "distributed mixed state creation" begin
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

            for T in [ComplexF64, ComplexF32]
                for dims in [(2, 2, 2)]
                    for bs in [1, 2]
                        rho = DenseMixedState{T}(ws, dims; batch_size = bs)
                        shape, offsets = CuDensityMat.local_info(rho)
                        @test length(shape) == length(offsets)
                        # Mixed: ndims = 2*len(dims) + batch
                        expected_modes = 2 * length(dims) + 1
                        @test length(shape) == expected_modes

                        sz = CuDensityMat.storage_size(rho)
                        buf = CUDA.zeros(T, sz)
                        CuDensityMat.attach_storage!(rho, buf)
                        @test rho.storage !== nothing
                    end
                end
            end

            close(ws)
        end
    end

    @gpu_test "distributed allocate_storage!" begin
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

            psi = DensePureState{ComplexF64}(ws, (2, 2, 2); batch_size = 1)
            CuDensityMat.allocate_storage!(psi)
            @test psi.storage !== nothing
            @test length(psi.storage) == CuDensityMat.storage_size(psi)

            close(ws)
        end
    end

    @gpu_test "distributed state view" begin
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

            psi = DensePureState{ComplexF64}(ws, (2, 2, 2); batch_size = 2)
            CuDensityMat.allocate_storage!(psi)
            v = CuDensityMat.state_view(psi)
            shape, _ = CuDensityMat.local_info(psi)
            @test size(v) == shape

            close(ws)
        end
    end

    @gpu_test "distributed norm" begin
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

            psi = DensePureState{ComplexF64}(ws, (2, 2, 2); batch_size = 1)
            CuDensityMat.allocate_storage!(psi)
            # Fill with ones for predictable norm
            psi.storage .= one(ComplexF64)
            n = CuDensityMat.norm(psi)
            @test length(n) == 1
            # Each rank has a local slice; norm is global sum
            @test n[1] > 0

            close(ws)
        end
    end

    @gpu_test "distributed trace" begin
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

            psi = DensePureState{ComplexF64}(ws, (2, 2, 2); batch_size = 1)
            CuDensityMat.allocate_storage!(psi)
            psi.storage .= one(ComplexF64)
            t = CuDensityMat.trace(psi)
            @test length(t) == 1
            @test real(t[1]) > 0

            close(ws)
        end
    end

    @gpu_test "distributed scale" begin
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

            psi = DensePureState{ComplexF64}(ws, (2, 2, 2); batch_size = 1)
            CuDensityMat.allocate_storage!(psi)
            psi.storage .= one(ComplexF64)
            original = Array(psi.storage)
            CuDensityMat.inplace_scale!(psi, ComplexF64(2.0))
            scaled = Array(psi.storage)
            @test scaled ≈ original .* 2.0

            close(ws)
        end
    end

    @gpu_test "distributed accumulate" begin
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

            psi1 = DensePureState{ComplexF64}(ws, (2, 2, 2); batch_size = 1)
            CuDensityMat.allocate_storage!(psi1)
            psi1.storage .= ComplexF64(1.0)

            sz = CuDensityMat.storage_size(psi1)
            buf2 = CUDA.fill(ComplexF64(2.0), sz)
            psi2 = CuDensityMat.clone(psi1, buf2)

            arr1 = Array(psi1.storage)
            arr2 = Array(psi2.storage)
            CuDensityMat.inplace_accumulate!(psi1, psi2, ComplexF64(3.0))
            result = Array(psi1.storage)
            @test result ≈ arr1 .+ 3.0 .* arr2

            close(ws)
        end
    end

end
