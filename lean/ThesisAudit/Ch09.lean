import Mathlib

/-!
Audit of display-math formulas in ch09-em-kernel.tex
(chapter "Learning the Kernel, and What Convergence Costs").
Four display equations: eq:em-fixedpoint, eq:em-missing-info,
eq:em-settle, eq:em-cumulant-damping.
-/

namespace ThesisAudit

/- ------------------------------------------------------------------ -/

-- eq:em-fixedpoint (ch09-em-kernel.tex lines 29-36), part 1:
-- the general first-order linearization of the EM map about a fixed point.
-- For any map M differentiable at θ* with M θ* = θ*,
-- M θ - θ* - J (θ - θ*) is little-o of (θ - θ*), which is the content of
-- "θ^{k+1} - θ* = J (θ^k - θ*) + (higher order)".
theorem ch09_em_fixedpoint_littleO (M : ℝ → ℝ) (θs J : ℝ)
    (hfix : M θs = θs) (hd : HasDerivAt M J θs) :
    (fun θ => M θ - θs - J * (θ - θs)) =o[nhds θs] (fun θ => θ - θs) := by
  have h := hasDerivAt_iff_isLittleO.mp hd
  -- h : (fun θ => M θ - M θs - (θ - θs) • J) =o[𝓝 θs] fun θ => θ - θs
  have heq : (fun θ => M θ - θs - J * (θ - θs))
      = fun θ => M θ - M θs - (θ - θs) • J := by
    funext θ
    rw [hfix, smul_eq_mul]
    ring
  rw [heq]
  exact h

-- eq:em-fixedpoint (ch09-em-kernel.tex lines 29-36), part 2:
-- the stated O(‖θ - θ*‖²) remainder, verified exactly on a generic
-- one-dimensional quadratic map M(θ) = θ* + J(θ-θ*) + c(θ-θ*)²:
-- θ* is a fixed point and the linearization error is bounded by |c|·|θ-θ*|².
theorem ch09_em_fixedpoint_quadratic_instance (θs J c θ : ℝ) :
    (θs + J * (θs - θs) + c * (θs - θs) ^ 2 = θs) ∧
    |(θs + J * (θ - θs) + c * (θ - θs) ^ 2) - θs - J * (θ - θs)|
      = |c| * |θ - θs| ^ 2 := by
  refine ⟨by ring, ?_⟩
  have h : (θs + J * (θ - θs) + c * (θ - θs) ^ 2) - θs - J * (θ - θs)
      = c * (θ - θs) ^ 2 := by ring
  rw [h, abs_mul, abs_pow]

/- ------------------------------------------------------------------ -/

-- eq:em-missing-info (ch09-em-kernel.tex lines 42-49):
-- given I_com = I_obs + I_mis (the first display, which defines I_mis as the
-- difference) and I_com invertible, the Dempster-Laird-Rubin Jacobian identity
-- I_com⁻¹ I_mis = 𝕀 - I_com⁻¹ I_obs, proved for arbitrary n×n real matrices.
theorem ch09_em_missing_info (n : ℕ)
    (Icom Iobs Imis : Matrix (Fin n) (Fin n) ℝ)
    [Invertible Icom] (h : Icom = Iobs + Imis) :
    Icom⁻¹ * Imis = 1 - Icom⁻¹ * Iobs := by
  have hmis : Imis = Icom - Iobs := by rw [h]; abel
  rw [hmis, mul_sub, Matrix.inv_mul_of_invertible]

/- ------------------------------------------------------------------ -/

-- eq:em-settle (ch09-em-kernel.tex lines 61-70), part 1:
-- k(τ,λ) = log τ / log λ is exactly the iteration count at which λ^k = τ:
-- for 0 < λ ≠ 1 and τ > 0, λ^(log τ / log λ) = τ.
theorem ch09_em_settle_count (lam tau : ℝ)
    (hlam : 0 < lam) (hne : lam ≠ 1) (htau : 0 < tau) :
    lam ^ (Real.log tau / Real.log lam) = tau := by
  have hlog : Real.log lam ≠ 0 := by
    intro h0
    have hexp := Real.exp_log hlam
    rw [h0, Real.exp_zero] at hexp
    exact hne hexp.symm
  rw [Real.rpow_def_of_pos hlam, mul_comm, div_mul_cancel₀ _ hlog,
    Real.exp_log htau]

-- eq:em-settle (ch09-em-kernel.tex lines 61-70), part 2:
-- the ratio factorization k₂/k₁ = (log τ₂ / log τ₁) · (log λ₁ / log λ₂),
-- with k_i = log τ_i / log λ_i, an exact algebraic identity whenever the
-- logs in the denominators are nonzero.
theorem ch09_em_settle_ratio (tau1 tau2 lam1 lam2 : ℝ)
    (h1 : Real.log tau1 ≠ 0) (h2 : Real.log lam1 ≠ 0)
    (h3 : Real.log lam2 ≠ 0) :
    (Real.log tau2 / Real.log lam2) / (Real.log tau1 / Real.log lam1)
      = (Real.log tau2 / Real.log tau1) * (Real.log lam1 / Real.log lam2) := by
  field_simp

/- ------------------------------------------------------------------ -/

-- eq:em-cumulant-damping (ch09-em-kernel.tex lines 108-112):
-- for the variance-preserving OU channel X_t = e^{-t} A + √(1-e^{-2t}) Z
-- with Z standard Gaussian independent of A and the variance-matched prior
-- κ₂(A) = 1, the second cumulant is preserved exactly, κ₂(X_t) = κ₂(A),
-- while the fourth is damped, κ₄(X_t) = e^{-4t} κ₄(A).
-- The cumulant calculus (additivity over independent summands, n-th order
-- homogeneity κ_n(cY) = cⁿ κ_n(Y), Gaussian κ₄ = 0) enters as the two
-- hypotheses h2 and h4; the noise variance s2 = 1 - e^{-2t}.
theorem ch09_em_cumulant_damping (t k2A k4A k2X k4X s2 : ℝ)
    (hs2 : s2 = 1 - Real.exp (-(2 * t)))
    (h2A : k2A = 1)
    (h2 : k2X = Real.exp (-t) ^ 2 * k2A + s2 * 1)
    (h4 : k4X = Real.exp (-t) ^ 4 * k4A + s2 ^ 2 * 0) :
    k2X = k2A ∧ k4X = Real.exp (-(4 * t)) * k4A := by
  have e2 : Real.exp (-t) ^ 2 = Real.exp (-(2 * t)) := by
    rw [← Real.exp_nat_mul]
    congr 1
    push_cast
    ring
  have e4 : Real.exp (-t) ^ 4 = Real.exp (-(4 * t)) := by
    rw [← Real.exp_nat_mul]
    congr 1
    push_cast
    ring
  constructor
  · rw [h2, h2A, e2, hs2]; ring
  · rw [h4, e4]; ring

-- Supporting identity for the h4 hypothesis above: fourth-cumulant additivity
-- derived from the raw-moment expansion. For X = c·A + s·Z with A, Z centered
-- and independent (so mixed moments factor and odd cross terms vanish),
-- E X² = c² E A² + s² E Z² and
-- E X⁴ = c⁴ E A⁴ + 6 c² s² (E A²)(E Z²) + s⁴ E Z⁴, and then
-- κ₄(X) = E X⁴ - 3 (E X²)² decomposes exactly as c⁴ κ₄(A) + s⁴ κ₄(Z):
-- the 6c²s² cross terms cancel identically.
theorem ch09_em_kappa4_additive (c s a2 a4 z2 z4 : ℝ) :
    (c ^ 4 * a4 + 6 * c ^ 2 * s ^ 2 * a2 * z2 + s ^ 4 * z4)
      - 3 * (c ^ 2 * a2 + s ^ 2 * z2) ^ 2
    = c ^ 4 * (a4 - 3 * a2 ^ 2) + s ^ 4 * (z4 - 3 * z2 ^ 2) := by
  ring

-- Same statement with the channel written exactly as in the thesis figure,
-- x = e^{-t} a + √(Δ_t) ε with Δ_t = 1 - e^{-2t}: the square of the noise
-- amplitude √(Δ_t) is Δ_t (needs t ≥ 0 so that Δ_t ≥ 0), so the previous
-- theorem applies with s2 = (√Δ_t)².
theorem ch09_em_cumulant_damping_sqrt (t : ℝ) (ht : 0 ≤ t) :
    Real.sqrt (1 - Real.exp (-(2 * t))) ^ 2 = 1 - Real.exp (-(2 * t)) := by
  have hnn : (0 : ℝ) ≤ 1 - Real.exp (-(2 * t)) := by
    have : Real.exp (-(2 * t)) ≤ 1 := by
      rw [Real.exp_le_one_iff]
      linarith
    linarith
  exact Real.sq_sqrt hnn

end ThesisAudit
