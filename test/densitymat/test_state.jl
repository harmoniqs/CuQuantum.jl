# Phase 2: State tests (local, single-GPU)
#
# Mirrors Python's test_state.py
# Tests cover creation, storage, scale, accumulate, inner_product, norm, trace

# Helper: create a random state with storage attached
function make_state(ws, T, dims, batch_size; mixed = false, init = :random)
    state = if mixed
        DenseMixedState{T}(ws, dims; batch_size = batch_size)
    else
        DensePureState{T}(ws, dims; batch_size = batch_size)
    end
    shape, _ = CuDensityMat.local_info(state)
    n = prod(shape)
    if init == :random
        data = CUDA.rand(real(T), n)
        if T <: Complex
            data_c = CUDA.CuVector{T}(
                complex.(
                    CUDA.CuVector{real(T)}(CUDA.rand(real(T), n) .- real(T)(0.5)),
                    CUDA.CuVector{real(T)}(CUDA.rand(real(T), n) .- real(T)(0.5)),
                ),
            )
            CuDensityMat.attach_storage!(state, data_c)
        else
            data_r = CUDA.CuVector{T}(CUDA.rand(T, n) .- T(0.5))
            CuDensityMat.attach_storage!(state, data_r)
        end
    elseif init == :zeros
        CuDensityMat.allocate_storage!(state)
    end
    return state
end

@testset "State (local)" begin

    # ---- Creation & storage ----

    @gpu_test "pure state creation" begin
        ws = WorkStream()
        for T in [ComplexF32, ComplexF64, Float32, Float64]
            for dims in [(2,), (2, 2), (2, 3)]
                psi = DensePureState{T}(ws, dims)
                @test isopen(psi)
                @test CuDensityMat.num_components(psi) == 1
                sz = CuDensityMat.storage_size(psi)
                @test sz == prod(dims)  # pure: product of dims (batch=1 default)
            end
        end
        close(ws)
    end

    @gpu_test "mixed state creation" begin
        ws = WorkStream()
        for T in [ComplexF32, ComplexF64]
            for dims in [(2,), (2, 2)]
                rho = DenseMixedState{T}(ws, dims)
                @test isopen(rho)
                sz = CuDensityMat.storage_size(rho)
                @test sz == prod(dims)^2  # mixed: dims^2
            end
        end
        close(ws)
    end

    @gpu_test "batch state creation" begin
        ws = WorkStream()
        for bs in [1, 2, 4]
            psi = DensePureState{ComplexF64}(ws, (2, 2); batch_size = bs)
            @test psi.batch_size == bs
            sz = CuDensityMat.storage_size(psi)
            @test sz == 4 * max(bs, 1)  # 2*2 * batch_size
        end
        close(ws)
    end

    @gpu_test "allocate_storage!" begin
        ws = WorkStream()
        psi = DensePureState{ComplexF64}(ws, (2, 2))
        CuDensityMat.allocate_storage!(psi)
        @test psi.storage !== nothing
        @test length(psi.storage) >= CuDensityMat.storage_size(psi)
        close(ws)
    end

    @gpu_test "attach_storage!" begin
        ws = WorkStream()
        psi = DensePureState{ComplexF64}(ws, (2, 2))
        sz = CuDensityMat.storage_size(psi)
        buf = CUDA.zeros(ComplexF64, sz)
        CuDensityMat.attach_storage!(psi, buf)
        @test psi.storage === buf
        close(ws)
    end

    @gpu_test "local_info shape" begin
        ws = WorkStream()
        # Pure state: shape should have length(dims) + 1 (batch dim) modes
        psi = DensePureState{ComplexF64}(ws, (2, 3); batch_size = 1)
        CuDensityMat.allocate_storage!(psi)
        shape, offs = CuDensityMat.local_info(psi)
        @test length(shape) == 3  # 2 space modes + 1 batch
        @test shape[end] == 1    # batch dim

        # Mixed state: 2*length(dims) + 1 modes
        rho = DenseMixedState{ComplexF64}(ws, (2, 3); batch_size = 2)
        CuDensityMat.allocate_storage!(rho)
        shape_m, offs_m = CuDensityMat.local_info(rho)
        @test length(shape_m) == 5  # 4 space modes + 1 batch
        @test shape_m[end] == 2    # batch dim

        close(ws)
    end

    @gpu_test "state_view" begin
        ws = WorkStream()
        psi = DensePureState{ComplexF64}(ws, (2, 3); batch_size = 2)
        CuDensityMat.allocate_storage!(psi)
        v = CuDensityMat.state_view(psi)
        shape, _ = CuDensityMat.local_info(psi)
        @test size(v) == shape
        close(ws)
    end

    @gpu_test "initialize_zero!" begin
        ws = WorkStream()
        psi = make_state(ws, ComplexF64, (2, 2), 1; init = :random)
        CuDensityMat.initialize_zero!(psi)
        @test all(Array(psi.storage) .== 0)
        close(ws)
    end

    @gpu_test "clone" begin
        ws = WorkStream()
        psi = make_state(ws, ComplexF64, (2, 2), 1; init = :random)
        buf = CUDA.zeros(ComplexF64, length(psi.storage))
        psi2 = CuDensityMat.clone(psi, buf)
        @test typeof(psi2) == typeof(psi)
        @test psi2.hilbert_space_dims == psi.hilbert_space_dims
        @test psi2.batch_size == psi.batch_size
        close(ws)
    end

    # ---- Computations ----

    @gpu_test "inplace_scale! scalar" begin
        ws = WorkStream()
        for T in [ComplexF64, ComplexF32, Float64, Float32]
            psi = make_state(ws, T, (2,), 1; init = :random)
            original = Array(psi.storage)
            CuDensityMat.inplace_scale!(psi, T(2))
            scaled = Array(psi.storage)
            @test scaled ≈ original .* T(2) rtol=1e-5
        end
        close(ws)
    end

    @gpu_test "inplace_scale! batched" begin
        ws = WorkStream()
        psi = make_state(ws, ComplexF64, (2,), 2; init = :random)
        original = Array(psi.storage)
        shape, _ = CuDensityMat.local_info(psi)
        orig_view = reshape(original, shape)

        factors = ComplexF64[2.0, 3.0]
        CuDensityMat.inplace_scale!(psi, factors)
        scaled = reshape(Array(psi.storage), shape)

        for i = 1:2
            @test scaled[:, i] ≈ orig_view[:, i] .* factors[i] rtol=1e-5
        end
        close(ws)
    end

    @gpu_test "norm" begin
        ws = WorkStream()
        for T in [ComplexF64, ComplexF32, Float64, Float32]
            for mixed in [false, true]
                (real(T) == Float32 && mixed) && continue  # skip for brevity
                psi = make_state(ws, T, (2,), 1; mixed = mixed, init = :random)
                n = CuDensityMat.norm(psi)
                @test length(n) == 1
                # Manual: squared frobenius norm
                arr = Array(psi.storage)
                expected = real(dot(arr, arr))
                @test n[1] ≈ expected rtol=1e-4
            end
        end
        close(ws)
    end

    @gpu_test "norm batched" begin
        ws = WorkStream()
        psi = make_state(ws, ComplexF64, (2, 2), 2; init = :random)
        n = CuDensityMat.norm(psi)
        @test length(n) == 2
        shape, _ = CuDensityMat.local_info(psi)
        arr = reshape(Array(psi.storage), shape)
        for i = 1:2
            col = vec(arr[ntuple(_->:, length(shape)-1)..., i])
            expected = real(dot(col, col))
            @test n[i] ≈ expected rtol=1e-5
        end
        close(ws)
    end

    @gpu_test "trace pure" begin
        ws = WorkStream()
        for T in [ComplexF64, ComplexF32]
            psi = make_state(ws, T, (2, 2), 1; mixed = false, init = :random)
            t = CuDensityMat.trace(psi)
            # For pure states, trace = <psi|psi>
            arr = Array(psi.storage)
            expected = dot(arr, arr)
            @test t[1] ≈ expected rtol=1e-4
        end
        close(ws)
    end

    @gpu_test "trace mixed" begin
        ws = WorkStream()
        rho = make_state(ws, ComplexF64, (2,), 1; mixed = true, init = :random)
        t = CuDensityMat.trace(rho)
        # For mixed states, trace = tr(rho)
        arr = Array(rho.storage)
        d = prod(rho.hilbert_space_dims)
        mat = reshape(arr[1:(d^2)], (d, d))
        expected = tr(mat)
        @test t[1] ≈ expected rtol=1e-5
        close(ws)
    end

    @gpu_test "inplace_accumulate!" begin
        ws = WorkStream()
        for T in [ComplexF64, ComplexF32, Float64, Float32]
            psi1 = make_state(ws, T, (2,), 1; init = :random)
            psi2 = make_state(ws, T, (2,), 1; init = :random)
            # Match storage sizes
            sz = CuDensityMat.storage_size(psi1)
            buf2 = CUDA.rand(real(T), sz)
            if T <: Complex
                buf2c = CUDA.CuVector{T}(
                    complex.(
                        CUDA.CuVector{real(T)}(CUDA.rand(real(T), sz)),
                        CUDA.CuVector{real(T)}(CUDA.rand(real(T), sz)),
                    ),
                )
                psi2_clone = CuDensityMat.clone(psi1, buf2c)
            else
                buf2r = CUDA.CuVector{T}(CUDA.rand(T, sz))
                psi2_clone = CuDensityMat.clone(psi1, buf2r)
            end

            arr1 = Array(psi1.storage)
            arr2 = Array(psi2_clone.storage)

            factor = T(2)
            CuDensityMat.inplace_accumulate!(psi1, psi2_clone, factor)
            result = Array(psi1.storage)
            expected = arr1 .+ factor .* arr2
            @test result ≈ expected rtol=1e-4
        end
        close(ws)
    end

    @gpu_test "inner_product" begin
        ws = WorkStream()
        for T in [ComplexF64, ComplexF32]
            psi1 = make_state(ws, T, (2, 2), 1; init = :random)
            sz = CuDensityMat.storage_size(psi1)
            buf = CUDA.CuVector{T}(
                complex.(
                    CUDA.CuVector{real(T)}(CUDA.rand(real(T), sz)),
                    CUDA.CuVector{real(T)}(CUDA.rand(real(T), sz)),
                ),
            )
            psi2 = CuDensityMat.clone(psi1, buf)

            ip = CuDensityMat.inner_product(psi1, psi2)
            arr1 = Array(psi1.storage)
            arr2 = Array(psi2.storage)
            expected = dot(arr1, arr2)
            @test ip[1] ≈ expected rtol=1e-4
        end
        close(ws)
    end

    @gpu_test "inner_product batched" begin
        ws = WorkStream()
        psi1 = make_state(ws, ComplexF64, (2,), 2; init = :random)
        sz = CuDensityMat.storage_size(psi1)
        buf = CUDA.CuVector{ComplexF64}(
            complex.(
                CUDA.CuVector{Float64}(CUDA.rand(Float64, sz)),
                CUDA.CuVector{Float64}(CUDA.rand(Float64, sz)),
            ),
        )
        psi2 = CuDensityMat.clone(psi1, buf)

        ip = CuDensityMat.inner_product(psi1, psi2)
        @test length(ip) == 2

        shape, _ = CuDensityMat.local_info(psi1)
        arr1 = reshape(Array(psi1.storage), shape)
        arr2 = reshape(Array(psi2.storage), shape)
        for i = 1:2
            col1 = vec(arr1[ntuple(_->:, length(shape)-1)..., i])
            col2 = vec(arr2[ntuple(_->:, length(shape)-1)..., i])
            expected = dot(col1, col2)
            @test ip[i] ≈ expected rtol=1e-5
        end
        close(ws)
    end

    @gpu_test "state finalizer" begin
        ws = WorkStream()
        handle = let
            psi = DensePureState{ComplexF64}(ws, (2,))
            CuDensityMat.allocate_storage!(psi)
            psi.handle
        end
        GC.gc(true)
        @test true  # no crash = success
        close(ws)
    end

    @gpu_test "incompatible states error" begin
        ws = WorkStream()
        psi1 = make_state(ws, ComplexF64, (2,), 1; init = :random)
        psi2 = make_state(ws, ComplexF64, (3,), 1; init = :random)
        @test_throws ErrorException CuDensityMat.inner_product(psi1, psi2)

        # Different batch sizes
        psi3 = make_state(ws, ComplexF64, (2,), 2; init = :random)
        @test_throws ErrorException CuDensityMat.inner_product(psi1, psi3)
        close(ws)
    end

end
