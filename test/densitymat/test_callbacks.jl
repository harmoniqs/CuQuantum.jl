# Phase 5: Callback tests
#
# Tests time-dependent scalar and tensor callbacks:
# wrapping, registration, GC safety, and end-to-end usage
# with operator action computation.

@testset "Callbacks (local)" begin

    # ---- Scalar callback wrapping ----

    @gpu_test "wrap scalar callback" begin
        # Simple callback: f(t) = exp(iΩt) where Ω=1
        function my_coeff(time, params, storage)
            for i in eachindex(storage)
                storage[i] = complex(cos(time), sin(time))
            end
        end

        cb, gcb, refs = CuDensityMat.wrap_scalar_callback(my_coeff)
        @test cb.callback != C_NULL
        @test cb.device == CuDensityMat.CUDENSITYMAT_CALLBACK_DEVICE_CPU
        @test cb.wrapper != C_NULL  # our trampoline
        # Gradient should be null (no gradient provided)
        @test gcb.callback == C_NULL
        CuDensityMat.unregister_callback!(refs)
    end

    @gpu_test "wrap scalar callback with gradient" begin
        function my_coeff(time, params, storage)
            for i in eachindex(storage)
                storage[i] = complex(cos(time), sin(time))
            end
        end
        function my_grad(time, params, scalar_grad, params_grad)
            for i in eachindex(scalar_grad)
                scalar_grad[i] = complex(-sin(time), cos(time))
            end
        end

        cb, gcb, refs = CuDensityMat.wrap_scalar_callback(my_coeff; gradient=my_grad)
        @test cb.callback != C_NULL
        @test gcb.callback != C_NULL
        @test gcb.wrapper != C_NULL
        @test gcb.direction == CuDensityMat.CUDENSITYMAT_DIFFERENTIATION_DIR_BACKWARD
        CuDensityMat.unregister_callback!(refs)
    end

    # ---- Tensor callback wrapping ----

    @gpu_test "wrap tensor callback" begin
        function my_tensor(time, params, storage)
            # Fill with rotation matrix: [[cos(t), -sin(t)], [sin(t), cos(t)]]
            # Column-major: storage[1]=cos(t), storage[2]=sin(t), storage[3]=-sin(t), storage[4]=cos(t)
            storage[1, 1, 1] = complex(cos(time))
            storage[2, 1, 1] = complex(sin(time))
            storage[1, 2, 1] = complex(-sin(time))
            storage[2, 2, 1] = complex(cos(time))
        end

        cb, gcb, refs = CuDensityMat.wrap_tensor_callback(my_tensor)
        @test cb.callback != C_NULL
        @test cb.device == CuDensityMat.CUDENSITYMAT_CALLBACK_DEVICE_CPU
        @test cb.wrapper != C_NULL
        CuDensityMat.unregister_callback!(refs)
    end

    # ---- Callback registry ----

    @gpu_test "callback registry lifecycle" begin
        f1(t, p, s) = nothing
        f2(t, p, s) = nothing

        id1 = CuDensityMat._register_callback(f1)
        id2 = CuDensityMat._register_callback(f2)
        @test id1 != id2
        @test CuDensityMat._get_callback(id1) === f1
        @test CuDensityMat._get_callback(id2) === f2

        CuDensityMat._unregister_callback(id1)
        @test_throws KeyError CuDensityMat._get_callback(id1)
        @test CuDensityMat._get_callback(id2) === f2

        CuDensityMat._unregister_callback(id2)
    end

    # ---- End-to-end: scalar callback with operator action ----

    @gpu_test "time-dependent scalar coefficient end-to-end" begin
        ws = WorkStream()
        dims = [2]
        T = ComplexF64

        # sigma_z operator
        sigma_z_data = CUDA.CuVector{T}([1.0+0im, 0.0, 0.0, -1.0+0im])
        elem = CuDensityMat.create_elementary_operator(ws, [2], sigma_z_data)

        # Create term with sigma_z on mode 0
        term = CuDensityMat.create_operator_term(ws, dims)
        CuDensityMat.append_elementary_product!(term, [elem], Int32[0], Int32[0])

        # Time-dependent coefficient: f(t) = exp(iΩt), Ω = params[1,b]
        function td_coeff(time, params, storage)
            for b in eachindex(storage)
                ω = params[1, b]
                storage[b] = complex(cos(ω * time), sin(ω * time))
            end
        end

        cb, gcb, cb_refs = CuDensityMat.wrap_scalar_callback(td_coeff)

        # Assemble operator with time-dependent coefficient callback
        operator = CuDensityMat.create_operator(ws, dims)
        CuDensityMat.append_term!(operator, term;
            duality=0, coefficient=1.0+0im,
            coefficient_callback=cb,
            coefficient_gradient_callback=gcb)

        # Create states
        psi_in = DenseMixedState{T}(ws, (2,); batch_size=1)
        psi_out = DenseMixedState{T}(ws, (2,); batch_size=1)
        CuDensityMat.allocate_storage!(psi_in)
        CuDensityMat.allocate_storage!(psi_out)

        # Set input: rho = |0><0| = [[1,0],[0,0]]
        copyto!(psi_in.storage, CUDA.CuVector{T}([1.0, 0.0, 0.0, 0.0]))

        # Prepare
        CuDensityMat.prepare_operator_action!(ws, operator, psi_in, psi_out)

        # Compute at t=0.5 with param Ω=2.0
        CuDensityMat.initialize_zero!(psi_out)
        params = CUDA.CuVector{Float64}([2.0])  # Ω = 2.0
        CuDensityMat.compute_operator_action!(ws, operator, psi_in, psi_out;
            time=0.5, batch_size=1, num_params=1, params=params)

        result = Array(psi_out.storage)
        @test any(x -> abs(x) > 0, result)

        # Cleanup
        CuDensityMat.unregister_callback!(cb_refs)
        close(ws)
    end

    # ---- End-to-end: time-dependent tensor callback ----

    @gpu_test "time-dependent tensor callback end-to-end" begin
        ws = WorkStream()
        dims = [2]
        T = ComplexF64

        # Create elementary operator with tensor callback (time-dependent matrix)
        # The callback fills in a rotation matrix R(t) = [[cos(t), -sin(t)], [sin(t), cos(t)]]
        function td_tensor(time, params, storage)
            # storage shape: (2, 2, batch_size) for dense 1-mode operator
            for b in axes(storage, 3)
                storage[1, 1, b] = complex(cos(time))
                storage[2, 1, b] = complex(sin(time))
                storage[1, 2, b] = complex(-sin(time))
                storage[2, 2, b] = complex(cos(time))
            end
        end

        tensor_cb, tensor_gcb, tcb_refs = CuDensityMat.wrap_tensor_callback(td_tensor)

        # Create elementary op with callback (pass zeros for initial static data)
        static_data = CUDA.zeros(T, 4)
        elem = CuDensityMat.create_elementary_operator(ws, [2], static_data;
            tensor_callback=tensor_cb,
            tensor_gradient_callback=tensor_gcb)

        # Build operator
        term = CuDensityMat.create_operator_term(ws, dims)
        CuDensityMat.append_elementary_product!(term, [elem], Int32[0], Int32[0])

        operator = CuDensityMat.create_operator(ws, dims)
        CuDensityMat.append_term!(operator, term; duality=0)

        # States
        psi_in = DenseMixedState{T}(ws, (2,); batch_size=1)
        psi_out = DenseMixedState{T}(ws, (2,); batch_size=1)
        CuDensityMat.allocate_storage!(psi_in)
        CuDensityMat.allocate_storage!(psi_out)
        copyto!(psi_in.storage, CUDA.CuVector{T}([1.0, 0.0, 0.0, 0.0]))

        # Prepare + compute
        CuDensityMat.prepare_operator_action!(ws, operator, psi_in, psi_out)
        CuDensityMat.initialize_zero!(psi_out)
        CuDensityMat.compute_operator_action!(ws, operator, psi_in, psi_out;
            time=0.3, batch_size=1)

        result = Array(psi_out.storage)
        @test any(x -> abs(x) > 0, result)

        CuDensityMat.unregister_callback!(tcb_refs)
        close(ws)
    end

end
