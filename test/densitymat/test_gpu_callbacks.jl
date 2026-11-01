# GPU-side (CUDENSITYMAT_CALLBACK_DEVICE_GPU) callbacks.
#
# A GPU callback is invoked on the host with device pointers and the library's
# CUDA stream; it enqueues CUDA.jl kernels on that (non-owned) stream. The
# trampoline tests below drive the C-callable wrappers exactly as cuDensityMat
# does, so they need CUDA.jl but not a cuDensityMat-capable device.

using CUDA: CuDeviceArray, CuDeviceVector, CuDeviceMatrix
import CUDA.CUDACore: CUstream, C_64F

function _fill_cis_kernel!(storage, params, t)
    i = threadIdx().x
    if i <= length(storage)
        @inbounds storage[i] = cis(params[1, i] * t)
    end
    return nothing
end

function gpu_cis_coefficient(t, params, storage, stream)
    @test stream isa CuStream
    @cuda threads = length(storage) stream = stream _fill_cis_kernel!(storage, params, t)
    return nothing
end

function _rotation_kernel!(storage, t)
    b = threadIdx().x
    if b <= size(storage, 3)
        c, s = cos(t), sin(t)
        @inbounds storage[1, 1, b] = c
        @inbounds storage[2, 1, b] = s
        @inbounds storage[1, 2, b] = -s
        @inbounds storage[2, 2, b] = c
    end
    return nothing
end

function gpu_rotation(t, params, storage, stream)
    @cuda threads = size(storage, 3) stream = stream _rotation_kernel!(storage, t)
    return nothing
end

function _cis_grad_kernel!(scalar_grad, params, params_grad, t)
    b = threadIdx().x
    if b <= length(scalar_grad)
        # d/dω cis(ω t) = i t cis(ω t); same accumulation convention as the CPU tests
        @inbounds params_grad[1, b] += real(scalar_grad[b] * im * t * cis(params[1, b] * t))
    end
    return nothing
end

function gpu_cis_gradient(t, params, scalar_grad, params_grad, stream)
    @cuda threads = length(scalar_grad) stream = stream _cis_grad_kernel!(
        scalar_grad,
        params,
        params_grad,
        t,
    )
    return nothing
end

host_ptr(a) = reinterpret(Ptr{Cvoid}, pointer(a))

@testset "GPU callbacks" begin

    @testset "wrapping selects the GPU device and trampolines" begin
        cb, gcb, refs = CuDensityMat.wrap_scalar_callback(
            gpu_cis_coefficient;
            gradient = gpu_cis_gradient,
            device = :gpu,
        )
        @test cb.device == CuDensityMat.CUDENSITYMAT_CALLBACK_DEVICE_GPU
        @test gcb.device == CuDensityMat.CUDENSITYMAT_CALLBACK_DEVICE_GPU
        @test cb.wrapper != C_NULL && gcb.wrapper != C_NULL
        cpu_cb, _, cpu_refs = CuDensityMat.wrap_scalar_callback((t, p, s) -> nothing)
        @test cb.wrapper != cpu_cb.wrapper
        @test cpu_cb.device == CuDensityMat.CUDENSITYMAT_CALLBACK_DEVICE_CPU
        CuDensityMat.unregister_callback!(refs)
        CuDensityMat.unregister_callback!(cpu_refs)

        tcb, tgcb, trefs = CuDensityMat.wrap_tensor_callback(
            gpu_rotation;
            gradient = (t, p, g, pg, s) -> nothing,
            device = :gpu,
        )
        @test tcb.device == CuDensityMat.CUDENSITYMAT_CALLBACK_DEVICE_GPU
        @test tgcb.device == CuDensityMat.CUDENSITYMAT_CALLBACK_DEVICE_GPU
        CuDensityMat.unregister_callback!(trefs)

        @test_throws ArgumentError CuDensityMat.wrap_scalar_callback(
            gpu_cis_coefficient;
            device = :tpu,
        )
    end

    @testset "CPU scalar trampoline (host buffers)" begin
        f(t, params, storage) = (storage .= cis.(params[1, :] .* t); nothing)
        cb, _, refs = CuDensityMat.wrap_scalar_callback(f)
        params = [1.0 2.0 3.0]
        storage = zeros(ComplexF64, 3)
        rc = ccall(
            cb.wrapper,
            Int32,
            (
                Ptr{Cvoid},
                Cdouble,
                Int64,
                Int32,
                Ptr{Cdouble},
                CuDensityMat.cudaDataType_t,
                Ptr{Cvoid},
                CUstream,
            ),
            cb.callback,
            0.5,
            3,
            1,
            params,
            C_64F,
            storage,
            C_NULL,
        )
        @test rc == 0
        @test storage ≈ cis.([0.5, 1.0, 1.5])

        bad, _, bad_refs = CuDensityMat.wrap_scalar_callback((t, p, s) -> error("boom"))
        rc = @test_logs (:error,) match_mode = :any ccall(
            bad.wrapper,
            Int32,
            (
                Ptr{Cvoid},
                Cdouble,
                Int64,
                Int32,
                Ptr{Cdouble},
                CuDensityMat.cudaDataType_t,
                Ptr{Cvoid},
                CUstream,
            ),
            bad.callback,
            0.5,
            3,
            1,
            params,
            C_64F,
            storage,
            C_NULL,
        )
        @test rc != 0
        CuDensityMat.unregister_callback!(refs)
        CuDensityMat.unregister_callback!(bad_refs)
    end

    @cuda_test "GPU scalar trampoline launches on the library's stream" begin
        cb, _, refs = CuDensityMat.wrap_scalar_callback(gpu_cis_coefficient; device = :gpu)
        params = CuArray([1.0 2.0 3.0])
        storage = CUDA.zeros(ComplexF64, 3)
        CUDA.synchronize()
        lib_stream = CuStream()  # stands in for the stream cuDensityMat passes
        rc = ccall(
            cb.wrapper,
            Int32,
            (
                Ptr{Cvoid},
                Cdouble,
                Int64,
                Int32,
                Ptr{Cdouble},
                CuDensityMat.cudaDataType_t,
                Ptr{Cvoid},
                CUstream,
            ),
            cb.callback,
            0.5,
            3,
            1,
            reinterpret(Ptr{Cdouble}, pointer(params)),
            C_64F,
            host_ptr(storage),
            lib_stream.handle,
        )
        @test rc == 0
        synchronize(lib_stream)
        @test Array(storage) ≈ cis.([0.5, 1.0, 1.5])

        # the default (NULL) stream is accepted as well
        storage .= 0
        CUDA.synchronize()
        rc = ccall(
            cb.wrapper,
            Int32,
            (
                Ptr{Cvoid},
                Cdouble,
                Int64,
                Int32,
                Ptr{Cdouble},
                CuDensityMat.cudaDataType_t,
                Ptr{Cvoid},
                CUstream,
            ),
            cb.callback,
            1.0,
            3,
            1,
            reinterpret(Ptr{Cdouble}, pointer(params)),
            C_64F,
            host_ptr(storage),
            C_NULL,
        )
        @test rc == 0
        CUDA.device_synchronize()
        @test Array(storage) ≈ cis.([1.0, 2.0, 3.0])
        CuDensityMat.unregister_callback!(refs)
    end

    @cuda_test "GPU tensor trampoline" begin
        cb, _, refs = CuDensityMat.wrap_tensor_callback(gpu_rotation; device = :gpu)
        storage = CUDA.zeros(ComplexF64, 2, 2, 2)
        extents = Int64[2, 2]
        CUDA.synchronize()
        lib_stream = CuStream()
        rc = ccall(
            cb.wrapper,
            Int32,
            (
                Ptr{Cvoid},
                CuDensityMat.cudensitymatElementaryOperatorSparsity_t,
                Int32,
                Ptr{Int64},
                Ptr{Int32},
                Cdouble,
                Int64,
                Int32,
                Ptr{Cdouble},
                CuDensityMat.cudaDataType_t,
                Ptr{Cvoid},
                CUstream,
            ),
            cb.callback,
            CuDensityMat.CUDENSITYMAT_OPERATOR_SPARSITY_NONE,
            2,
            extents,
            C_NULL,
            0.3,
            2,
            0,
            C_NULL,
            C_64F,
            host_ptr(storage),
            lib_stream.handle,
        )
        @test rc == 0
        synchronize(lib_stream)
        R = ComplexF64[cos(0.3) -sin(0.3); sin(0.3) cos(0.3)]
        @test Array(storage)[:, :, 1] ≈ R
        @test Array(storage)[:, :, 2] ≈ R
        CuDensityMat.unregister_callback!(refs)
    end

    @cuda_test "GPU scalar gradient trampoline" begin
        _, gcb, refs = CuDensityMat.wrap_scalar_callback(
            gpu_cis_coefficient;
            gradient = gpu_cis_gradient,
            device = :gpu,
        )
        params = CuArray([1.0 2.0])
        scalar_grad = CuArray(ComplexF64[1.0, 1.0im])
        params_grad = CUDA.zeros(Float64, 1, 2)
        CUDA.synchronize()
        lib_stream = CuStream()
        rc = ccall(
            gcb.wrapper,
            Int32,
            (
                Ptr{Cvoid},
                Cdouble,
                Int64,
                Int32,
                Ptr{Cdouble},
                CuDensityMat.cudaDataType_t,
                Ptr{Cvoid},
                Ptr{Cdouble},
                CUstream,
            ),
            gcb.callback,
            0.5,
            2,
            1,
            reinterpret(Ptr{Cdouble}, pointer(params)),
            C_64F,
            host_ptr(scalar_grad),
            reinterpret(Ptr{Cdouble}, pointer(params_grad)),
            lib_stream.handle,
        )
        @test rc == 0
        synchronize(lib_stream)
        expected =
            [real(a * im * 0.5 * cis(w * 0.5)) for (a, w) in zip([1.0, 1.0im], [1.0, 2.0])]
        @test vec(Array(params_grad)) ≈ expected
        CuDensityMat.unregister_callback!(refs)
    end

    # ---- end-to-end through cuDensityMat (compute capability >= 7.5) ----

    @gpu_test "GPU scalar callback matches CPU callback in operator action" begin
        T = ComplexF64
        dims = [2]
        function run(cb, gcb)
            ws = WorkStream()
            sz = CUDA.CuVector{T}([1.0 + 0im, 0.0, 0.0, -1.0 + 0im])
            elem = CuDensityMat.create_elementary_operator(ws, [2], sz)
            term = CuDensityMat.create_operator_term(ws, dims)
            CuDensityMat.append_elementary_product!(term, [elem], Int32[0], Int32[0])
            op = CuDensityMat.create_operator(ws, dims)
            CuDensityMat.append_term!(
                op,
                term;
                duality = 0,
                coefficient = 1.0 + 0im,
                coefficient_callback = cb,
                coefficient_gradient_callback = gcb,
            )
            ρin = DenseMixedState{T}(ws, (2,); batch_size = 1)
            ρout = DenseMixedState{T}(ws, (2,); batch_size = 1)
            CuDensityMat.allocate_storage!(ρin)
            CuDensityMat.allocate_storage!(ρout)
            copyto!(ρin.storage, CUDA.CuVector{T}([0.5, 0.5, 0.5, 0.5]))
            CuDensityMat.prepare_operator_action!(ws, op, ρin, ρout)
            CuDensityMat.initialize_zero!(ρout)
            CuDensityMat.compute_operator_action!(
                ws,
                op,
                ρin,
                ρout;
                time = 0.5,
                batch_size = 1,
                num_params = 1,
                params = CUDA.CuVector{Float64}([2.0]),
            )
            out = Array(ρout.storage)
            close(ws)
            return out
        end
        cpu_cb, cpu_gcb, cpu_refs = CuDensityMat.wrap_scalar_callback(
            (t, p, s) -> (s .= cis.(p[1, :] .* t); nothing),
        )
        gpu_cb, gpu_gcb, gpu_refs =
            CuDensityMat.wrap_scalar_callback(gpu_cis_coefficient; device = :gpu)
        expected = run(cpu_cb, cpu_gcb)
        # dense oracle: cis(ω t)·σz ρ with ω t = 1, ρ = |+⟩⟨+|
        ρ = fill(0.5 + 0im, 2, 2)
        @test expected ≈ vec(cis(1.0) * ComplexF64[1 0; 0 -1] * ρ) rtol = 1e-12
        @test run(gpu_cb, gpu_gcb) ≈ expected rtol = 1e-12
        CuDensityMat.unregister_callback!(cpu_refs)
        CuDensityMat.unregister_callback!(gpu_refs)
    end

    @gpu_test "GPU tensor callback matches CPU callback in operator action" begin
        T = ComplexF64
        dims = [2]
        function run(tcb, tgcb)
            ws = WorkStream()
            elem = CuDensityMat.create_elementary_operator(
                ws,
                [2],
                CUDA.zeros(T, 4);
                tensor_callback = tcb,
                tensor_gradient_callback = tgcb,
            )
            term = CuDensityMat.create_operator_term(ws, dims)
            CuDensityMat.append_elementary_product!(term, [elem], Int32[0], Int32[0])
            op = CuDensityMat.create_operator(ws, dims)
            CuDensityMat.append_term!(op, term; duality = 0)
            ρin = DenseMixedState{T}(ws, (2,); batch_size = 1)
            ρout = DenseMixedState{T}(ws, (2,); batch_size = 1)
            CuDensityMat.allocate_storage!(ρin)
            CuDensityMat.allocate_storage!(ρout)
            copyto!(ρin.storage, CUDA.CuVector{T}([1.0, 0.0, 0.0, 0.0]))
            CuDensityMat.prepare_operator_action!(ws, op, ρin, ρout)
            CuDensityMat.initialize_zero!(ρout)
            CuDensityMat.compute_operator_action!(
                ws,
                op,
                ρin,
                ρout;
                time = 0.3,
                batch_size = 1,
            )
            out = Array(ρout.storage)
            close(ws)
            return out
        end
        function cpu_rotation(t, p, s)
            for b in axes(s, 3)
                s[:, :, b] .= [cos(t) -sin(t); sin(t) cos(t)]
            end
        end
        c1, g1, r1 = CuDensityMat.wrap_tensor_callback(cpu_rotation)
        c2, g2, r2 = CuDensityMat.wrap_tensor_callback(gpu_rotation; device = :gpu)
        expected = run(c1, g1)
        # dense oracle: R(0.3)·|0⟩⟨0|
        R = ComplexF64[cos(0.3) -sin(0.3); sin(0.3) cos(0.3)]
        @test expected ≈ vec(R * ComplexF64[1 0; 0 0]) rtol = 1e-12
        @test run(c2, g2) ≈ expected rtol = 1e-12
        CuDensityMat.unregister_callback!(r1)
        CuDensityMat.unregister_callback!(r2)
    end

    @gpu_test "GPU gradient callback matches CPU gradient callback (backward diff)" begin
        T = ComplexF64
        dims = [2]
        function run(cb, gcb)
            ws = WorkStream()
            sz = CUDA.CuVector{T}([1.0 + 0im, 0.0, 0.0, -1.0 + 0im])
            elem = CuDensityMat.create_elementary_operator(ws, [2], sz)
            term = CuDensityMat.create_operator_term(ws, dims)
            CuDensityMat.append_elementary_product!(term, [elem], Int32[0], Int32[0])
            op = CuDensityMat.create_operator(ws, dims)
            CuDensityMat.append_term!(
                op,
                term;
                duality = 0,
                coefficient = 1.0 + 0im,
                coefficient_callback = cb,
                coefficient_gradient_callback = gcb,
            )
            ρin = DenseMixedState{T}(ws, (2,); batch_size = 1)
            ρout_adj = DenseMixedState{T}(ws, (2,); batch_size = 1)
            ρin_adj = DenseMixedState{T}(ws, (2,); batch_size = 1)
            foreach(CuDensityMat.allocate_storage!, (ρin, ρout_adj, ρin_adj))
            copyto!(ρin.storage, CUDA.CuVector{T}([0.5, 0.5, 0.5, 0.5]))
            copyto!(ρout_adj.storage, CUDA.CuVector{T}([1.0, 0.0, 0.0, 0.0]))
            params = CUDA.CuVector{Float64}([2.0])
            params_grad = CUDA.zeros(Float64, 1)
            CuDensityMat.prepare_operator_action_backward!(ws, op, ρin, ρout_adj)
            CuDensityMat.initialize_zero!(ρin_adj)
            CuDensityMat.compute_operator_action_backward!(
                ws,
                op,
                ρin,
                ρout_adj,
                ρin_adj,
                params_grad;
                time = 0.5,
                batch_size = 1,
                num_params = 1,
                params = params,
            )
            out = Array(params_grad)
            close(ws)
            return out
        end
        function cpu_grad(t, p, sg, pg)
            for b in axes(sg, 1)
                pg[1, b] += real(sg[b] * im * t * cis(p[1, b] * t))
            end
        end
        c1, g1, r1 = CuDensityMat.wrap_scalar_callback(
            (t, p, s) -> (s .= cis.(p[1, :] .* t); nothing);
            gradient = cpu_grad,
        )
        c2, g2, r2 = CuDensityMat.wrap_scalar_callback(
            gpu_cis_coefficient;
            gradient = gpu_cis_gradient,
            device = :gpu,
        )
        expected = run(c1, g1)
        # analytic oracle: scalar_grad = ⟨|0⟩⟨0|, σz ρ⟩ = 1/2, so
        # dc/dω = Re(1/2 · i t cis(ω t)) = -sin(1)/4 at t = 0.5, ω = 2 (no factor 2)
        @test expected[1] ≈ -sin(1.0) / 4 rtol = 1e-12
        @test run(c2, g2) ≈ expected rtol = 1e-12
        CuDensityMat.unregister_callback!(r1)
        CuDensityMat.unregister_callback!(r2)
    end

    @gpu_test "operator action with explicit dense (FULL) scope" begin
        T = ComplexF64
        ws = WorkStream()
        dims = [2]
        sx = CUDA.CuVector{T}([0.0, 1.0, 1.0, 0.0])
        elem = CuDensityMat.create_elementary_operator(ws, [2], sx)
        term = CuDensityMat.create_operator_term(ws, dims)
        CuDensityMat.append_elementary_product!(term, [elem], Int32[0], Int32[0])
        op = CuDensityMat.create_operator(ws, dims)
        CuDensityMat.append_term!(op, term; duality = 0)
        action = CuDensityMat.create_operator_action(
            ws,
            [op];
            scope = CuDensityMat.CUDENSITYMAT_FITTING_SCOPE_FULL,
            approach = CuDensityMat.CUDENSITYMAT_FITTING_APPROACH_LINSOLVE,
        )
        @test isopen(action)
        ψin = DensePureState{T}(ws, (2,); batch_size = 1)
        ψout = DensePureState{T}(ws, (2,); batch_size = 1)
        CuDensityMat.allocate_storage!(ψin)
        CuDensityMat.allocate_storage!(ψout)
        copyto!(ψin.storage, CUDA.CuVector{T}([1.0, 0.0]))
        CuDensityMat.prepare_action!(ws, action, [ψin], ψout)
        CuDensityMat.initialize_zero!(ψout)
        CuDensityMat.compute_action!(ws, action, [ψin], ψout)
        @test Array(ψout.storage) ≈ T[0, 1]
        close(ws)
    end
end
