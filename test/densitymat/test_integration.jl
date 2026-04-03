# =============================================================================
# Phase 10: Integration Test — Dual-Rail Qubit Lindblad Dynamics
# =============================================================================
#
# This test simulates the full Lindblad master equation for a dual-rail qubit
# encoded as a single photon across M=2 coupled cavities with photon loss:
#
#   ρ̇ = -i[H(t), ρ] + γ Σ_m (a_m ρ a_m† - ½{a_m†a_m, ρ})
#
# where the Hamiltonian is:
#
#   H(t) = Σ_m  a_m†² a_m²                           (self-Kerr)
#        + Σ_m  δ_m(t) a_m†a_m                       (time-dependent detuning)
#        + Σ_{n≠m} √(κ_n(t)κ_m(t)) a_n†a_m           (time-dependent coupling)
#
# We use forward Euler time-stepping: ρ(t+dt) = ρ(t) + dt * L[ρ(t)]
# and record the trajectory for post-hoc plotting.
#
# System parameters (in MHz, times in μs):
#   - M = 2 cavities, Fock truncation dim = 3 (|0⟩, |1⟩, |2⟩)
#   - Self-Kerr: χ = 2π × 0.2 MHz
#   - Detuning: δ_m(t) = Δ_m sin(ω_m t), Δ₁ = 2π×1.0, Δ₂ = 2π×0.8 MHz
#   - Coupling: κ_m(t) = κ₀(1 + ε cos(ω_κ t)), κ₀ = 2π×0.1, ε = 0.3
#   - Photon loss rate: γ = 0.01 MHz
#   - Simulation: t ∈ [0, 2.0] μs, dt = 0.02 μs, RK4 integrator (100 steps)

@testset "Integration: Dual-Rail Lindblad (local)" begin

    @gpu_test "full Lindblad simulation with trajectory" begin

        # =====================================================================
        # 1. System parameters
        # =====================================================================

        M = 2           # number of cavities
        d = 3           # Fock space truncation per cavity (|0⟩, |1⟩, |2⟩)
        T = ComplexF64   # data type

        # Physical parameters (MHz and μs)
        χ = 2π * 0.2    # self-Kerr nonlinearity
        Δ = [
            2π * 1.0,   # detuning amplitude, cavity 1
            2π * 0.8,
        ]   # detuning amplitude, cavity 2
        ω_δ = [
            2π * 0.5,  # detuning modulation frequency, cavity 1
            2π * 0.4,
        ]  # detuning modulation frequency, cavity 2
        κ₀ = 2π * 0.1    # base coupling strength
        ε = 0.3          # coupling modulation depth
        ω_κ = 2π * 0.2   # coupling modulation frequency
        γ = 0.01         # photon loss rate

        # Simulation (RK4 allows much larger dt than Euler)
        t_final = 2.0     # total time (μs)
        dt = 0.02         # time step (μs)
        n_steps = Int(t_final / dt)  # 100 steps

        # =====================================================================
        # 2. Create the WorkStream
        # =====================================================================

        ws = WorkStream()
        dims = [d, d]  # composite Hilbert space: cavity₁ ⊗ cavity₂

        # =====================================================================
        # 3. Build single-cavity operator matrices (3×3, column-major)
        # =====================================================================

        # Annihilation operator: a|n⟩ = √n |n-1⟩
        #   a = [[0, 1, 0],
        #        [0, 0, √2],
        #        [0, 0, 0]]   (column-major storage)
        a_mat = T[0, 0, 0, 1, 0, 0, 0, √2, 0]

        # Number operator: n = a†a = diag(0, 1, 2)
        n_mat = T[0, 0, 0, 0, 1, 0, 0, 0, 2]

        # Self-Kerr operator: n(n-1) = a†²a² = diag(0, 0, 2)
        kerr_mat = T[0, 0, 0, 0, 0, 0, 0, 0, 2]

        # =====================================================================
        # 4. Build fused 2-mode operators for Lindblad dissipation
        # =====================================================================

        # Fused operator for a_m ρ a_m†:
        #   F[i0, i1; j0, j1] = a[i0; j0] × a†[i1; j1]
        #   = a[i0; j0] × conj(a[j1; i1])
        #
        # This 4-index tensor has shape (d, d, d, d) = (3, 3, 3, 3) = 81 elements.
        # Index ordering (column-major): i0 fastest, then i1, then j0, then j1.
        #
        # When applied with mode_action_duality = [0, 1] on modes (m, m),
        # the first index pair (i0, j0) acts on the ket side (left of ρ)
        # and the second pair (i1, j1) acts on the bra side (right of ρ),
        # giving: a_m ρ a_m†
        aa_dag_fused = zeros(T, d, d, d, d)
        a_matrix = [
            0.0 1.0 0.0;
            0.0 0.0 √2;
            0.0 0.0 0.0
        ]  # a as a proper matrix
        a_dag_matrix = transpose(a_matrix)  # a† (real operator, so transpose = adjoint)
        for j1 in 1:d, j0 in 1:d, i1 in 1:d, i0 in 1:d
            # F[i0, i1, j0, j1] = a[i0, j0] * a†[i1, j1]
            aa_dag_fused[i0, i1, j0, j1] = a_matrix[i0, j0] * a_dag_matrix[i1, j1]
        end

        # =====================================================================
        # 5. Upload operator data to GPU
        # =====================================================================

        a_gpu = CUDA.CuVector{T}(a_mat)
        n_gpu = CUDA.CuVector{T}(n_mat)
        kerr_gpu = CUDA.CuVector{T}(kerr_mat)
        aa_dag_gpu = CUDA.CuVector{T}(vec(aa_dag_fused))  # flatten to 81-element vector

        # =====================================================================
        # 6. Create elementary operators
        # =====================================================================

        # Single-mode operators (act on one cavity, d=3)
        elem_n = CuDensityMat.create_elementary_operator(ws, [d], n_gpu)
        elem_kerr = CuDensityMat.create_elementary_operator(ws, [d], kerr_gpu)

        # Annihilation operator — needed for the beam-splitter coupling a_n†a_m.
        # Since a_n†a_m is a 2-mode operator, we fuse it into a single (d,d,d,d) tensor.
        # For coupling between cavities 0↔1:
        #   C_01[i0, i1; j0, j1] = a†[i0; j0] × a[i1; j1]  (cavity 0 creation, cavity 1 annihilation)
        #   C_10[i0, i1; j0, j1] = a[i0; j0] × a†[i1; j1]   (cavity 0 annihilation, cavity 1 creation)
        coupling_01 = zeros(T, d, d, d, d)  # a₀†a₁
        coupling_10 = zeros(T, d, d, d, d)  # a₁†a₀ = (a₀†a₁)†
        for j1 in 1:d, j0 in 1:d, i1 in 1:d, i0 in 1:d
            coupling_01[i0, i1, j0, j1] = a_dag_matrix[i0, j0] * a_matrix[i1, j1]
            coupling_10[i0, i1, j0, j1] = a_matrix[i0, j0] * a_dag_matrix[i1, j1]
        end
        coupling_01_gpu = CUDA.CuVector{T}(vec(coupling_01))
        coupling_10_gpu = CUDA.CuVector{T}(vec(coupling_10))

        # 2-mode fused elementary operators
        elem_aa_dag = CuDensityMat.create_elementary_operator(ws, [d, d], aa_dag_gpu)
        elem_coupling01 =
            CuDensityMat.create_elementary_operator(ws, [d, d], coupling_01_gpu)
        elem_coupling10 =
            CuDensityMat.create_elementary_operator(ws, [d, d], coupling_10_gpu)

        # =====================================================================
        # 7. Build Hamiltonian operator terms
        # =====================================================================

        # --- 7a. Self-Kerr term: H_kerr = χ Σ_m a_m†²a_m² ---
        # This is a static (time-independent) term.
        # We create one OperatorTerm and add the Kerr operator on each mode.
        kerr_term = CuDensityMat.create_operator_term(ws, dims)
        for m in 0:(M - 1)
            CuDensityMat.append_elementary_product!(
                kerr_term,
                [elem_kerr],        # single Kerr operator
                Int32[m],           # acts on cavity m
                Int32[0];           # ket-side (duality = 0)
                coefficient = ComplexF64(χ),  # static coefficient χ
            )
        end

        # --- 7b. Detuning terms: H_det = Σ_m δ_m(t) n_m ---
        # Each cavity has its own time-dependent detuning δ_m(t) = Δ_m sin(ω_m t).
        # We need separate OperatorTerms because each has a different scalar callback.

        # Detuning callback for cavity 1: δ₁(t) = Δ₁ sin(ω₁ t)
        function detuning_1_callback(time, params, storage)
            for b in eachindex(storage)
                storage[b] = ComplexF64(Δ[1] * sin(ω_δ[1] * time))
            end
        end
        det1_cb, det1_gcb, det1_refs =
            CuDensityMat.wrap_scalar_callback(detuning_1_callback)

        detuning_term_1 = CuDensityMat.create_operator_term(ws, dims)
        CuDensityMat.append_elementary_product!(
            detuning_term_1,
            [elem_n],       # number operator
            Int32[0],       # cavity 0
            Int32[0];       # ket-side
            coefficient = ComplexF64(1.0),      # static part = 1 (callback provides δ₁(t))
            coefficient_callback = det1_cb,
            coefficient_gradient_callback = det1_gcb,
        )

        # Detuning callback for cavity 2: δ₂(t) = Δ₂ sin(ω₂ t)
        function detuning_2_callback(time, params, storage)
            for b in eachindex(storage)
                storage[b] = ComplexF64(Δ[2] * sin(ω_δ[2] * time))
            end
        end
        det2_cb, det2_gcb, det2_refs =
            CuDensityMat.wrap_scalar_callback(detuning_2_callback)

        detuning_term_2 = CuDensityMat.create_operator_term(ws, dims)
        CuDensityMat.append_elementary_product!(
            detuning_term_2,
            [elem_n],
            Int32[1],       # cavity 1
            Int32[0];
            coefficient = ComplexF64(1.0),
            coefficient_callback = det2_cb,
            coefficient_gradient_callback = det2_gcb,
        )

        # --- 7c. Coupling term: H_coup = √(κ₁(t)κ₂(t)) (a₀†a₁ + a₁†a₀) ---
        # Time-dependent coupling with scalar callback.
        # κ_m(t) = κ₀(1 + ε cos(ω_κ t)), so √(κ₁κ₂) = κ₀(1 + ε cos(ω_κ t))
        # (both cavities have the same κ(t) in our symmetric setup).
        function coupling_callback(time, params, storage)
            κ_t = κ₀ * (1.0 + ε * cos(ω_κ * time))
            # √(κ₁(t) × κ₂(t)) = √(κ(t)²) = κ(t) since κ(t) > 0
            for b in eachindex(storage)
                storage[b] = ComplexF64(κ_t)
            end
        end
        coup_cb, coup_gcb, coup_refs = CuDensityMat.wrap_scalar_callback(coupling_callback)

        # Both coupling directions (a₀†a₁ and a₁†a₀) share the same OperatorTerm
        # and the same time-dependent coefficient.
        coupling_term = CuDensityMat.create_operator_term(ws, dims)
        # a₀†a₁: fused 2-mode operator acting on modes (0, 1), ket-side
        CuDensityMat.append_elementary_product!(
            coupling_term,
            [elem_coupling01],
            Int32[0, 1],    # acts on cavity 0 and cavity 1
            Int32[0, 0];    # both ket-side
            coefficient = ComplexF64(1.0),
            coefficient_callback = coup_cb,
            coefficient_gradient_callback = coup_gcb,
        )
        # a₁†a₀: Hermitian conjugate direction, same coefficient
        CuDensityMat.append_elementary_product!(
            coupling_term,
            [elem_coupling10],
            Int32[0, 1],
            Int32[0, 0];
            coefficient = ComplexF64(1.0),
            coefficient_callback = coup_cb,
            coefficient_gradient_callback = coup_gcb,
        )

        # =====================================================================
        # 8. Build Lindblad dissipation terms
        # =====================================================================

        # The Lindblad dissipator for photon loss on mode m is:
        #   D_m[ρ] = γ (a_m ρ a_m† - ½ a_m†a_m ρ - ½ ρ a_m†a_m)
        #
        # This decomposes into:
        #   (a) Sandwich term: a_m ρ a_m†  →  fused aa_dag on modes (m,m), duality [0,1]
        #   (b) Left anticommutator: -½ n_m ρ  →  n on mode m, duality 0, coeff -½
        #   (c) Right anticommutator: -½ ρ n_m  →  n on mode m, duality 1, coeff -½

        # --- 8a. Sandwich term: Σ_m a_m ρ a_m† ---
        # Uses the fused aa_dag operator with mode_action_duality = [0, 1].
        # duality [0, 1] means: first index pair acts on ket (left of ρ),
        # second index pair acts on bra (right of ρ).
        # modes_acted_on = [m, m]: both "modes" in the fused op map to the same
        # physical cavity m. This is the same pattern as the YY dissipation in
        # NVIDIA's C++ sample.
        dissipation_sandwich = CuDensityMat.create_operator_term(ws, dims)
        for m in 0:(M - 1)
            CuDensityMat.append_elementary_product!(
                dissipation_sandwich,
                [elem_aa_dag],
                Int32[m, m],     # both indices map to cavity m
                Int32[0, 1];     # ket-side and bra-side (sandwich)
                coefficient = ComplexF64(1.0),  # coefficient γ applied when appending to Operator
            )
        end

        # --- 8b. Anticommutator term: Σ_m n_m ---
        # This term will be appended twice to the Operator:
        #   once with duality=0, coeff=-γ/2  (left: -½ n_m ρ)
        #   once with duality=1, coeff=-γ/2  (right: -½ ρ n_m)
        dissipation_number = CuDensityMat.create_operator_term(ws, dims)
        for m in 0:(M - 1)
            CuDensityMat.append_elementary_product!(
                dissipation_number,
                [elem_n],
                Int32[m],
                Int32[0];
                coefficient = ComplexF64(1.0),
            )
        end

        # =====================================================================
        # 9. Assemble the full Liouvillian super-operator
        # =====================================================================
        #
        # The Liouvillian L[ρ] = -i[H, ρ] + D[ρ] is built by appending terms
        # to the Operator with appropriate duality flags and coefficients:
        #
        #   -i[H, ρ] = (-i)(H ρ) + (i)(ρ H)
        #            = coeff=-i, duality=0  +  coeff=+i, duality=1
        #
        #   D[ρ]     = γ(a ρ a†)            → coeff=γ, duality=0 (sandwich term)
        #            + (-γ/2)(n ρ)           → coeff=-γ/2, duality=0
        #            + (-γ/2)(ρ n)           → coeff=-γ/2, duality=1

        liouvillian = CuDensityMat.create_operator(ws, dims)

        # Hamiltonian: -i[H, ρ]
        # Each Hamiltonian term is appended twice (left and right of ρ)
        for (term, label) in [
                (kerr_term, "Kerr"),
                (detuning_term_1, "Detuning₁"),
                (detuning_term_2, "Detuning₂"),
                (coupling_term, "Coupling"),
            ]
            # -i × H × ρ  (acts from the left)
            CuDensityMat.append_term!(
                liouvillian,
                term;
                duality = 0,
                coefficient = ComplexF64(0, -1),
            )
            # +i × ρ × H  (acts from the right)
            CuDensityMat.append_term!(
                liouvillian,
                term;
                duality = 1,
                coefficient = ComplexF64(0, +1),
            )
        end

        # Lindblad dissipation
        # Sandwich: γ × (a_m ρ a_m†) — duality=0 because the duality is already
        # encoded in the mode_action_duality of the elementary product
        CuDensityMat.append_term!(
            liouvillian,
            dissipation_sandwich;
            duality = 0,
            coefficient = ComplexF64(γ),
        )

        # Anticommutator: -γ/2 × n_m ρ  (left side)
        CuDensityMat.append_term!(
            liouvillian,
            dissipation_number;
            duality = 0,
            coefficient = ComplexF64(-γ / 2),
        )

        # Anticommutator: -γ/2 × ρ n_m  (right side)
        CuDensityMat.append_term!(
            liouvillian,
            dissipation_number;
            duality = 1,
            coefficient = ComplexF64(-γ / 2),
        )

        # =====================================================================
        # 10. Prepare the initial state
        # =====================================================================
        #
        # Initial state: |1,0⟩⟨1,0| — one photon in cavity 1, vacuum in cavity 2.
        # In the 9×9 density matrix (d₁=3, d₂=3), the state |1,0⟩ has index
        # corresponding to (n₁=1, n₂=0).
        #
        # Column-major flattened index for |n₁, n₂⟩ in a (3,3) matrix:
        #   index = n₁ + d * n₂ + 1  (1-based Julia)
        # So |1,0⟩ is at index 2.
        # The density matrix ρ = |1,0⟩⟨1,0| has ρ[2,2] = 1 in the 9×9 matrix.
        # Flattened column-major: element (2,2) is at position 2 + 9*(2-1) = 11.

        d_total = prod(dims)  # 9
        rho_init = zeros(T, d_total * d_total)
        rho_init[2 + d_total * (2 - 1)] = 1.0 + 0im  # ρ[|1,0⟩, |1,0⟩] = 1
        rho_init_gpu = CUDA.CuVector{T}(rho_init)

        rho = DenseMixedState{T}(ws, Tuple(dims); batch_size = 1)
        CuDensityMat.allocate_storage!(rho)
        copyto!(rho.storage, rho_init_gpu)

        # Output state for ρ̇ = L[ρ]
        rho_dot = DenseMixedState{T}(ws, Tuple(dims); batch_size = 1)
        CuDensityMat.allocate_storage!(rho_dot)

        # =====================================================================
        # 11. Allocate RK4 scratch states
        # =====================================================================
        #
        # RK4 requires 4 evaluations of L[ρ] per step. We need scratch states
        # to store the intermediate k-values and a temporary state ρ_tmp.
        #
        # The RK4 algorithm for ρ̇ = L(t, ρ):
        #   k1 = L(t,       ρ)
        #   k2 = L(t + dt/2, ρ + dt/2 × k1)
        #   k3 = L(t + dt/2, ρ + dt/2 × k2)
        #   k4 = L(t + dt,   ρ + dt   × k3)
        #   ρ(t+dt) = ρ(t) + (dt/6)(k1 + 2k2 + 2k3 + k4)

        k1 = DenseMixedState{T}(ws, Tuple(dims); batch_size = 1)
        k2 = DenseMixedState{T}(ws, Tuple(dims); batch_size = 1)
        k3 = DenseMixedState{T}(ws, Tuple(dims); batch_size = 1)
        k4 = DenseMixedState{T}(ws, Tuple(dims); batch_size = 1)
        rho_tmp = DenseMixedState{T}(ws, Tuple(dims); batch_size = 1)
        CuDensityMat.allocate_storage!(k1)
        CuDensityMat.allocate_storage!(k2)
        CuDensityMat.allocate_storage!(k3)
        CuDensityMat.allocate_storage!(k4)
        CuDensityMat.allocate_storage!(rho_tmp)

        # =====================================================================
        # 12. Prepare the Liouvillian action
        # =====================================================================
        #
        # prepare_operator_action! must be called once. It plans the contraction
        # strategy for L[ρ] and allocates internal workspace. After this, we can
        # call compute_operator_action! many times with different time values.

        CuDensityMat.prepare_operator_action!(ws, liouvillian, rho, k1)

        # =====================================================================
        # 13. Time-stepping loop (RK4)
        # =====================================================================
        #
        # We record observables at each step for plotting:
        #   - Tr(ρ): Lindblad preserves trace — this is our primary sanity check
        #   - Diagonal populations P(|n₁,n₂⟩) = ⟨n₁,n₂|ρ|n₁,n₂⟩

        times = Float64[]
        traces = ComplexF64[]
        pop_10 = Float64[]  # P(|1,0⟩): photon in cavity 1
        pop_01 = Float64[]  # P(|0,1⟩): photon in cavity 2
        pop_00 = Float64[]  # P(|0,0⟩): vacuum (photon lost)
        pop_20 = Float64[]  # P(|2,0⟩): leakage into 2-photon sector
        pop_02 = Float64[]  # P(|0,2⟩): leakage into 2-photon sector

        function record_observables!(t, rho_state)
            rho_cpu = Array(rho_state.storage)
            d_tot = d_total

            push!(times, t)
            tr = sum(rho_cpu[k + d_tot * (k - 1)] for k in 1:d_tot)
            push!(traces, tr)

            # Fock state indices: |n₁,n₂⟩ → flat index = n₁ + d*n₂ + 1 (1-based)
            idx_10 = 1 + 1 + d * 0  # |1,0⟩
            idx_01 = 1 + 0 + d * 1  # |0,1⟩
            idx_00 = 1 + 0 + d * 0  # |0,0⟩
            idx_20 = 1 + 2 + d * 0  # |2,0⟩
            idx_02 = 1 + 0 + d * 2  # |0,2⟩

            push!(pop_10, real(rho_cpu[idx_10 + d_tot * (idx_10 - 1)]))
            push!(pop_01, real(rho_cpu[idx_01 + d_tot * (idx_01 - 1)]))
            push!(pop_00, real(rho_cpu[idx_00 + d_tot * (idx_00 - 1)]))
            push!(pop_20, real(rho_cpu[idx_20 + d_tot * (idx_20 - 1)]))
            push!(pop_02, real(rho_cpu[idx_02 + d_tot * (idx_02 - 1)]))
        end

        # Helper: copy state a into state b (GPU-to-GPU)
        function copy_state!(dst, src)
            copyto!(dst.storage, src.storage)
        end

        # Helper: compute L[rho_in] at time t → store in k_out
        function eval_liouvillian!(k_out, rho_in, t)
            CuDensityMat.initialize_zero!(k_out)
            CuDensityMat.compute_operator_action!(
                ws,
                liouvillian,
                rho_in,
                k_out;
                time = t,
                batch_size = 1,
            )
        end

        # Record initial state
        record_observables!(0.0, rho)

        for step in 1:n_steps
            t = (step - 1) * dt

            # --- k1 = L(t, ρ) ---
            eval_liouvillian!(k1, rho, t)

            # --- k2 = L(t + dt/2, ρ + dt/2 × k1) ---
            copy_state!(rho_tmp, rho)
            CuDensityMat.inplace_accumulate!(rho_tmp, k1, ComplexF64(dt / 2))
            eval_liouvillian!(k2, rho_tmp, t + dt / 2)

            # --- k3 = L(t + dt/2, ρ + dt/2 × k2) ---
            copy_state!(rho_tmp, rho)
            CuDensityMat.inplace_accumulate!(rho_tmp, k2, ComplexF64(dt / 2))
            eval_liouvillian!(k3, rho_tmp, t + dt / 2)

            # --- k4 = L(t + dt, ρ + dt × k3) ---
            copy_state!(rho_tmp, rho)
            CuDensityMat.inplace_accumulate!(rho_tmp, k3, ComplexF64(dt))
            eval_liouvillian!(k4, rho_tmp, t + dt)

            # --- ρ(t+dt) = ρ(t) + (dt/6)(k1 + 2k2 + 2k3 + k4) ---
            CuDensityMat.inplace_accumulate!(rho, k1, ComplexF64(dt / 6))
            CuDensityMat.inplace_accumulate!(rho, k2, ComplexF64(dt / 3))
            CuDensityMat.inplace_accumulate!(rho, k3, ComplexF64(dt / 3))
            CuDensityMat.inplace_accumulate!(rho, k4, ComplexF64(dt / 6))

            # Record at every step (100 total data points — enough for smooth plots)
            record_observables!(step * dt, rho)
        end

        # =====================================================================
        # 13. Assertions
        # =====================================================================

        # (a) Output trajectory has data
        @test length(times) > 10

        # (b) Initial state was |1,0⟩⟨1,0|
        @test abs(pop_10[1] - 1.0) < 1.0e-10
        @test abs(pop_01[1]) < 1.0e-10
        @test abs(pop_00[1]) < 1.0e-10

        # (c) Trace should be preserved (≈ 1) throughout — Lindblad preserves trace.
        # RK4 is much more accurate than Euler, so tighter tolerance is reasonable.
        for (i, tr) in enumerate(traces)
            @test abs(real(tr) - 1.0) < 0.02
        end

        # (d) Population has transferred: |1,0⟩ decreased, |0,1⟩ increased (coupling)
        @test pop_10[end] < pop_10[1]   # photon left cavity 1
        @test pop_01[end] > pop_01[1]   # photon arrived in cavity 2

        # (e) Photon loss: vacuum population increased
        @test pop_00[end] > pop_00[1]

        # (f) Populations are physical (non-negative, within [0,1])
        for p in [pop_10; pop_01; pop_00]
            @test p >= -0.01  # allow tiny numerical error
            @test p <= 1.01
        end

        # =====================================================================
        # 14. Save trajectory data for plotting
        # =====================================================================
        #
        # Write a simple CSV that can be plotted with any tool.
        # Columns: time, trace_real, pop_10, pop_01, pop_00, pop_20, pop_02

        mktempdir() do tmpdir
            trajectory_file = joinpath(tmpdir, "trajectory.csv")
            open(trajectory_file, "w") do io
                println(io, "time,trace_real,trace_imag,pop_10,pop_01,pop_00,pop_20,pop_02")
                for i in eachindex(times)
                    println(
                        io,
                        join(
                            [
                                times[i],
                                real(traces[i]),
                                imag(traces[i]),
                                pop_10[i],
                                pop_01[i],
                                pop_00[i],
                                pop_20[i],
                                pop_02[i],
                            ],
                            ",",
                        ),
                    )
                end
            end
            @test isfile(trajectory_file)
        end

        # =====================================================================
        # 15. Cleanup
        # =====================================================================

        CuDensityMat.unregister_callback!(det1_refs)
        CuDensityMat.unregister_callback!(det2_refs)
        CuDensityMat.unregister_callback!(coup_refs)
        close(ws)

        # Print summary for visual inspection
        r6(x) = round(x; digits = 6)
        println("\n  Dual-Rail Lindblad Simulation Complete")
        println("  System: M=2 cavities, dim=3 each, RK4 integrator")
        println("  Steps:  $(n_steps) x dt=$(dt) us = $(t_final) us")
        println("  Tr(rho) final: $(r6(real(traces[end])))")
        println("  P(|1,0>):      $(r6(pop_10[1])) -> $(r6(pop_10[end]))")
        println("  P(|0,1>):      $(r6(pop_01[1])) -> $(r6(pop_01[end]))")
        println("  P(|0,0>):      $(r6(pop_00[1])) -> $(r6(pop_00[end]))")
        println("  Trajectory saved to temp file (cleaned up)")
    end

end
