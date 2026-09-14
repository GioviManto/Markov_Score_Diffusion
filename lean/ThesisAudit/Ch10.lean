import Mathlib

/-!
Audit of the display-math formulas in
`thesis/chapters/ch10-comparison.tex`
("What Structure Buys, and Where It Stops Paying"), whole file (lines 1-1287).

Conventions used throughout this file:

* The variance-preserving channel is `x = e^{-t} a + √(Δ_t) ε` with
  `Δ_t = 1 - e^{-2t}` (`ch10_Delta`), the Chapter 5 convention carried into this
  chapter; `Σ_t = e^{-2t} Σ_0 + Δ_t I` (`ch10_Sig`).
* Expectations of quadratic forms are checked at the level of the algebraic
  identity they reduce to: `E‖A X‖² = tr(A Σ Aᵀ)` when `Cov X = Σ` is proved as
  the pure sum/trace identity `ch10_energy_trace`; the statistical step
  `E[X Xᵀ] = Σ` is not re-proved.
* Cumulant claims are formalised with the cumulant calculus (additivity over
  independent summands, order-`n` homogeneity, Gaussian `κ₄ = 0`) as explicit
  hypotheses, in the style of `ThesisAudit.Ch09`.
* "Bulk"/infinite-chain statements are formalised as the exact Green's-function
  relation for the bi-infinite tridiagonal Toeplitz operator, which is what the
  toolbox derivation actually establishes.
-/

namespace ThesisAudit

open Matrix

noncomputable section

/-! ## Shared channel objects -/

/-- `Δ_t = 1 - e^{-2t}`, the variance-preserving channel noise variance. -/
def ch10_Delta (t : ℝ) : ℝ := 1 - Real.exp (-(2 * t))

theorem ch10_Delta_pos {t : ℝ} (ht : 0 < t) : 0 < ch10_Delta t := by
  have h : Real.exp (-(2 * t)) < 1 := Real.exp_lt_one_iff.mpr (by linarith)
  unfold ch10_Delta; linarith

theorem ch10_one_sub_Delta (t : ℝ) : 1 - ch10_Delta t = Real.exp (-(2 * t)) := by
  unfold ch10_Delta; ring

/-- `Σ_t = e^{-2t} Σ₀ + Δ_t I`, the marginal covariance of the noised chain. -/
def ch10_Sig {n : ℕ} (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  Real.exp (-(2 * t)) • S0 + ch10_Delta t • (1 : Matrix (Fin n) (Fin n) ℝ)

/-- `Σ_t = Σ₀ + Δ_t (I - Σ₀)`: the interpolation form used for the small-`t`
expansion of `tr Σ_t⁻¹`. -/
theorem ch10_Sig_interp {n : ℕ} (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ) :
    ch10_Sig S0 t = S0 + ch10_Delta t • ((1 : Matrix (Fin n) (Fin n) ℝ) - S0) := by
  unfold ch10_Sig
  rw [smul_sub, ← ch10_one_sub_Delta t, sub_smul, one_smul]
  abel

/-- The all-ones vector. -/
def ch10_ones (n : ℕ) : Fin n → ℝ := fun _ => 1

/-! ## eq:em-relerr (lines 35-43) -/

-- eq:em-relerr (ch10-comparison.tex lines 35-43): the relative L² score error
-- E[ŝ;t] = (Σ_j‖ŝ(x^j)-s*(x^j)‖² / Σ_j‖s*(x^j)‖²)^{1/2} over a test set of J
-- sequences of n sites.  A definition, not a claim.
def ch10_relErr {J n : ℕ} (shat sstar : Fin J → Fin n → ℝ) : ℝ :=
  Real.sqrt ((∑ j, ∑ i, (shat j i - sstar j i) ^ 2) / (∑ j, ∑ i, (sstar j i) ^ 2))

theorem ch10_relErr_nonneg {J n : ℕ} (shat sstar : Fin J → Fin n → ℝ) :
    0 ≤ ch10_relErr shat sstar := Real.sqrt_nonneg _

-- eq:em-relerr, the stated property "It is scale free: multiplying the data by a
-- constant leaves it unchanged."
theorem ch10_relErr_scale_invariant {J n : ℕ} (c : ℝ) (hc : c ≠ 0)
    (shat sstar : Fin J → Fin n → ℝ) :
    ch10_relErr (fun j i => c * shat j i) (fun j i => c * sstar j i)
      = ch10_relErr shat sstar := by
  unfold ch10_relErr
  congr 1
  have hn : (∑ j, ∑ i, (c * shat j i - c * sstar j i) ^ 2)
      = c ^ 2 * ∑ j, ∑ i, (shat j i - sstar j i) ^ 2 := by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun i _ => by ring
  have hd : (∑ j, ∑ i, (c * sstar j i) ^ 2) = c ^ 2 * ∑ j, ∑ i, (sstar j i) ^ 2 := by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun i _ => by ring
  rw [hn, hd]
  rcases eq_or_ne (∑ j, ∑ i, (sstar j i) ^ 2) 0 with h0 | h0
  · rw [h0]; simp
  · have hc2 : c ^ 2 ≠ 0 := pow_ne_zero 2 hc
    field_simp

/-! ## eq:em-pooling (lines 48-56) -/

-- eq:em-pooling: "Pooling over the noise schedule means pooling the two sums, not
-- averaging the ratios": (Σ_t N_t)/(Σ_t D_t) = Σ_t w_t (N_t/D_t) with
-- w_t = D_t/Σ_s D_s.  N_t is the level-t squared error and D_t the level-t
-- reference score energy (the E‖s*‖² of the displayed w_t).
theorem ch10_em_pooling {ι : Type*} (s : Finset ι) (hs : s.Nonempty) (N D : ι → ℝ)
    (hD : ∀ i ∈ s, 0 < D i) :
    (∑ i ∈ s, N i) / (∑ i ∈ s, D i)
      = ∑ i ∈ s, (D i / ∑ j ∈ s, D j) * (N i / D i) := by
  have hpos : 0 < ∑ j ∈ s, D j := Finset.sum_pos hD hs
  rw [Finset.sum_div]
  refine Finset.sum_congr rfl fun i hi => ?_
  have h1 : D i ≠ 0 := ne_of_gt (hD i hi)
  have h2 : (∑ j ∈ s, D j) ≠ 0 := ne_of_gt hpos
  field_simp

-- eq:em-pooling: the weights are a probability vector.
theorem ch10_em_pooling_weights_sum {ι : Type*} (s : Finset ι) (hs : s.Nonempty) (D : ι → ℝ)
    (hD : ∀ i ∈ s, 0 < D i) :
    (∑ i ∈ s, D i / ∑ j ∈ s, D j) = 1 := by
  have hpos : 0 < ∑ j ∈ s, D j := Finset.sum_pos hD hs
  rw [← Finset.sum_div, div_self (ne_of_gt hpos)]

/-! ## eq:em-weights (lines 62-67) -/

-- Algebraic core of "E‖A X‖² = tr(A Σ Aᵀ) when E[X Xᵀ] = Σ".
theorem ch10_energy_trace {n : ℕ} (A S : Matrix (Fin n) (Fin n) ℝ) :
    (∑ i, ∑ j, ∑ k, A i j * S j k * A i k) = Matrix.trace (A * S * Aᵀ) := by
  simp only [Matrix.trace, Matrix.diag_apply, Matrix.mul_apply, Matrix.transpose_apply]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [Finset.sum_mul]

-- eq:em-weights: for the Gaussian chain with score s*(x,t) = -Σ_t⁻¹ x and
-- E[X Xᵀ] = Σ_t, E‖s*(X_t,t)‖² = tr(Σ_t⁻¹ Σ_t Σ_t⁻ᵀ) = tr Σ_t⁻¹.
theorem ch10_em_weights {n : ℕ} (S : Matrix (Fin n) (Fin n) ℝ)
    (hsym : Sᵀ = S) (hdet : IsUnit S.det) :
    Matrix.trace (S⁻¹ * S * (S⁻¹)ᵀ) = Matrix.trace S⁻¹ := by
  have ht : (S⁻¹)ᵀ = S⁻¹ := by rw [Matrix.transpose_nonsing_inv, hsym]
  rw [ht, Matrix.nonsing_inv_mul _ hdet, Matrix.one_mul]

theorem ch10_exp_neg_two_tendsto :
    Filter.Tendsto (fun t : ℝ => Real.exp (-(2 * t))) Filter.atTop (nhds 0) := by
  have h2 : Filter.Tendsto (fun t : ℝ => 2 * t) Filter.atTop Filter.atTop := by
    apply Filter.tendsto_atTop_atTop.2
    intro b
    refine ⟨max b 0, fun a ha => ?_⟩
    have hb : b ≤ a := le_trans (le_max_left _ _) ha
    have h0 : (0:ℝ) ≤ a := le_trans (le_max_right _ _) ha
    linarith
  have h3 : Filter.Tendsto (fun t : ℝ => -(2 * t)) Filter.atTop Filter.atBot :=
    Filter.tendsto_neg_atTop_atBot.comp h2
  exact Real.tendsto_exp_atBot.comp h3

-- eq:em-weights, second half: "Σ_t⁻¹ → I (t → ∞)" — checked as Σ_t → I entrywise
-- (the channel washes out the prior), whence tr Σ_t⁻¹ → n = 32.
theorem ch10_em_weights_Sig_tendsto_one {n : ℕ} (S0 : Matrix (Fin n) (Fin n) ℝ) (i j : Fin n) :
    Filter.Tendsto (fun t : ℝ => ch10_Sig S0 t i j) Filter.atTop
      (nhds ((1 : Matrix (Fin n) (Fin n) ℝ) i j)) := by
  have h0 := ch10_exp_neg_two_tendsto
  have hd : Filter.Tendsto (fun t : ℝ => ch10_Delta t) Filter.atTop (nhds 1) := by
    have h := Filter.Tendsto.const_sub (1 : ℝ) h0
    simpa [ch10_Delta] using h
  have h1 : Filter.Tendsto (fun t : ℝ => Real.exp (-(2 * t)) * S0 i j)
      Filter.atTop (nhds 0) := by
    simpa using h0.mul_const (S0 i j)
  have h2 : Filter.Tendsto (fun t : ℝ => ch10_Delta t * ((1 : Matrix (Fin n) (Fin n) ℝ) i j))
      Filter.atTop (nhds ((1 : Matrix (Fin n) (Fin n) ℝ) i j)) := by
    simpa using hd.mul_const ((1 : Matrix (Fin n) (Fin n) ℝ) i j)
  have h3 := h1.add h2
  simpa [ch10_Sig] using h3

/-! ## eq:em-weights-smallt (lines 70-77) -/

-- eq:em-weights-smallt: tr Σ_t⁻¹ = tr Q₀ + 2t[tr Q₀ - tr Q₀²] + O(t²) as t ↓ 0,
-- with Q₀ = Σ₀⁻¹.  Exact algebraic content, with ε = Δ_t playing the role of 2t
-- (see ch10_Delta_approx_two_t): writing Σ_t = Σ₀ + ε(I - Σ₀), the truncated
-- inverse Q₀ - ε(Q₀² - Q₀) inverts Σ_t up to an explicit ε² term, so the
-- expansion is correct to first order with a genuinely quadratic remainder.
theorem ch10_em_weights_smallt_resolvent {n : ℕ} (A B : Matrix (Fin n) (Fin n) ℝ)
    (hA : IsUnit A.det) (ε : ℝ) :
    (A + ε • B) * (A⁻¹ - ε • (A⁻¹ * B * A⁻¹))
      = 1 - ε ^ 2 • (B * A⁻¹ * (B * A⁻¹)) := by
  have hQ : A * A⁻¹ = 1 := Matrix.mul_nonsing_inv _ hA
  have hAQBQ : A * (A⁻¹ * B * A⁻¹) = B * A⁻¹ := by
    rw [← Matrix.mul_assoc, ← Matrix.mul_assoc, hQ, Matrix.one_mul]
  have hBQBQ : B * (A⁻¹ * B * A⁻¹) = B * A⁻¹ * (B * A⁻¹) := by
    rw [Matrix.mul_assoc A⁻¹ B A⁻¹, ← Matrix.mul_assoc]
  simp only [Matrix.add_mul, Matrix.mul_sub, Matrix.smul_mul, Matrix.mul_smul, smul_smul,
    hQ, hAQBQ, hBQBQ, pow_two]
  match_scalars <;> ring

-- eq:em-weights-smallt: the middle factor at B = I - Σ₀ is exactly Q₀² - Q₀, so
-- the first-order term of the resolvent expansion is -ε(Q₀² - Q₀) and its trace is
-- +ε(tr Q₀ - tr Q₀²), as displayed.
theorem ch10_em_weights_smallt_middle {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ)
    (hA : IsUnit A.det) :
    A⁻¹ * ((1 : Matrix (Fin n) (Fin n) ℝ) - A) * A⁻¹ = A⁻¹ * A⁻¹ - A⁻¹ := by
  have hQA : A⁻¹ * A = 1 := Matrix.nonsing_inv_mul _ hA
  rw [Matrix.mul_sub, Matrix.mul_one, hQA, Matrix.sub_mul, Matrix.one_mul]

-- eq:em-weights-smallt: the trace of the truncated inverse is exactly
-- tr Q₀ + ε (tr Q₀ - tr Q₀²), the displayed expansion with ε = 2t.
theorem ch10_em_weights_smallt_trace {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) (ε : ℝ) :
    Matrix.trace (A⁻¹ - ε • (A⁻¹ * A⁻¹ - A⁻¹))
      = Matrix.trace A⁻¹ + ε * (Matrix.trace A⁻¹ - Matrix.trace (A⁻¹ * A⁻¹)) := by
  rw [Matrix.trace_sub, Matrix.trace_smul, Matrix.trace_sub, smul_eq_mul]
  ring

-- eq:em-weights-smallt: Δ_t = 2t + O(t²), so the ε-expansion above is the stated
-- t-expansion; two-sided bound 2t - 4t² ≤ Δ_t ≤ 2t for t ≥ 0.
theorem ch10_Delta_approx_two_t {t : ℝ} (ht : 0 ≤ t) :
    ch10_Delta t ≤ 2 * t ∧ 2 * t - 4 * t ^ 2 ≤ ch10_Delta t := by
  have hx : 0 < Real.exp (-(2 * t)) := Real.exp_pos _
  have he : Real.exp (-(2 * t)) * Real.exp (2 * t) = 1 := by
    rw [← Real.exp_add]; simp
  constructor
  · have h := Real.add_one_le_exp (-(2 * t))
    unfold ch10_Delta; linarith
  · have hE : 0 < Real.exp (2 * t) := Real.exp_pos _
    have h1 : 1 + 2 * t ≤ Real.exp (2 * t) := by
      have := Real.add_one_le_exp (2 * t); linarith
    have h2 : Real.exp (-(2 * t)) * (1 + 2 * t) ≤ 1 := by
      calc Real.exp (-(2 * t)) * (1 + 2 * t)
          ≤ Real.exp (-(2 * t)) * Real.exp (2 * t) := by
            exact mul_le_mul_of_nonneg_left h1 hx.le
        _ = 1 := he
    have ht3 : 0 ≤ t ^ 3 := by positivity
    have h3 : Real.exp (-(2 * t)) * (1 + 2 * t) ≤ (1 - 2 * t + 4 * t ^ 2) * (1 + 2 * t) := by
      have hrw : (1 - 2 * t + 4 * t ^ 2) * (1 + 2 * t) = 1 + 8 * t ^ 3 := by ring
      rw [hrw]; linarith
    have h4 : (0:ℝ) < 1 + 2 * t := by linarith
    have h5 : Real.exp (-(2 * t)) ≤ 1 - 2 * t + 4 * t ^ 2 :=
      le_of_mul_le_mul_right h3 h4
    unfold ch10_Delta; linarith

/-- Closed form for `tr Q₀` on the AR(1) chain of `n` sites with correlation `r`:
`n-2` interior rows contribute `(1+r²)/(1-r²)` and the two boundary rows `1/(1-r²)`
(entries as in eq:em-prior-precision, proved for every `n ≥ 2` in
`ch10_em_prior_precision` and `ch10_em_prior_precision_boundary` below). -/
def ch10_trQ0 (n : ℕ) (r : ℝ) : ℝ := (((n : ℝ) - 2) * (1 + r ^ 2) + 2) / (1 - r ^ 2)

/-- Closed form for `tr Q₀²` = the sum of squares of the entries of the symmetric
tridiagonal `Q₀`: `n-2` interior diagonal entries, 2 boundary diagonal entries and
`2(n-1)` off-diagonal entries. -/
def ch10_trQ0sq (n : ℕ) (r : ℝ) : ℝ :=
  (((n : ℝ) - 2) * (1 + r ^ 2) ^ 2 + 2 + 2 * ((n : ℝ) - 1) * r ^ 2) / (1 - r ^ 2) ^ 2

-- eq:em-weights-smallt, the quoted numbers at ρ = 0.85, n = 32: "tr Q₀ = 193.4"
-- and "the linear coefficient in eq:em-weights-smallt is -3140".
theorem ch10_trQ0_numeric : ch10_trQ0 32 (17 / 20) = 21470 / 111 := by
  unfold ch10_trQ0; norm_num

theorem ch10_trQ0_rounds : |ch10_trQ0 32 (17 / 20) - 193.4| < 0.03 := by
  rw [ch10_trQ0_numeric, abs_lt]
  constructor <;> norm_num

theorem ch10_smallt_coefficient :
    2 * (ch10_trQ0 32 (17 / 20) - ch10_trQ0sq 32 (17 / 20)) = -(38691320 / 12321) := by
  unfold ch10_trQ0 ch10_trQ0sq; norm_num

theorem ch10_smallt_coefficient_rounds :
    |2 * (ch10_trQ0 32 (17 / 20) - ch10_trQ0sq 32 (17 / 20)) - (-3140)| < 0.3 := by
  rw [ch10_smallt_coefficient, abs_lt]
  constructor <;> norm_num

/-! ## eq:em-ratio (lines 124-128) -/

-- eq:em-ratio: R = E_base/E_EM, and "R > 1 means the structured estimator is more
-- accurate".
def ch10_R (Ebase EEM : ℝ) : ℝ := Ebase / EEM

theorem ch10_R_gt_one_iff {Ebase EEM : ℝ} (h : 0 < EEM) :
    1 < ch10_R Ebase EEM ↔ EEM < Ebase := by
  unfold ch10_R
  rw [lt_div_iff₀ h, one_mul]

/-! ## eq:em-parameterisation (lines 192-197) -/

-- eq:em-parameterisation: ŝ_ε = -ε̂/√Δ_t and ŝ_a = (e^{-t} â - x)/Δ_t "agree
-- identically under the change of variable â = e^{t}(x - √Δ_t ε̂)".  Stated per
-- site (both heads act coordinatewise on the sequence).
theorem ch10_em_parameterisation (t x eps : ℝ) (hΔ : 0 < ch10_Delta t) :
    (Real.exp (-t) * (Real.exp t * (x - Real.sqrt (ch10_Delta t) * eps)) - x) / ch10_Delta t
      = -(eps / Real.sqrt (ch10_Delta t)) := by
  have h1 : Real.exp (-t) * Real.exp t = 1 := by
    rw [← Real.exp_add]; simp
  have hs : 0 < Real.sqrt (ch10_Delta t) := Real.sqrt_pos.mpr hΔ
  have hsq : Real.sqrt (ch10_Delta t) * Real.sqrt (ch10_Delta t) = ch10_Delta t :=
    Real.mul_self_sqrt (le_of_lt hΔ)
  set s := Real.sqrt (ch10_Delta t) with hsdef
  have hcollapse : Real.exp (-t) * (Real.exp t * (x - s * eps)) = x - s * eps := by
    rw [← mul_assoc, h1, one_mul]
  have hne : s ≠ 0 := ne_of_gt hs
  rw [hcollapse, ← hsq]
  field_simp
  ring

-- eq:em-parameterisation: the companion claim that the substitution "rescales the
-- residual by e^t √Δ_t", so the two parameterisations are the same function class
-- but different optimisation problems.
theorem ch10_em_parameterisation_residual (t x e1 e2 : ℝ) :
    (Real.exp t * (x - Real.sqrt (ch10_Delta t) * e1))
        - (Real.exp t * (x - Real.sqrt (ch10_Delta t) * e2))
      = -(Real.exp t * Real.sqrt (ch10_Delta t)) * (e1 - e2) := by
  ring

/-! ## eq:em-dim (lines 223-226) -/

-- eq:em-dim: dim θ = (C-1) + C + C + 1 = 3C, and 24 at C = 8.
theorem ch10_em_dim (C : ℕ) (hC : 1 ≤ C) : (C - 1) + C + C + 1 = 3 * C := by omega

theorem ch10_em_dim_at_eight : (8 - 1) + 8 + 8 + 1 = 24 := by norm_num

/-! ## eq:em-mlp (lines 236-239) -/

/-- eq:em-mlp: the unstructured denoiser
`f_φ(x,t) = W₃ tanh(W₂ tanh(W₁[x;τ(t)] + b₁) + b₂) + b₃`. -/
def ch10_mlp {n h : ℕ} (W1 : Matrix (Fin h) (Fin (n + 3)) ℝ) (b1 : Fin h → ℝ)
    (W2 : Matrix (Fin h) (Fin h) ℝ) (b2 : Fin h → ℝ)
    (W3 : Matrix (Fin n) (Fin h) ℝ) (b3 : Fin n → ℝ)
    (z : Fin (n + 3) → ℝ) : Fin n → ℝ :=
  fun i => (W3 *ᵥ fun k => Real.tanh ((W2 *ᵥ fun l => Real.tanh ((W1 *ᵥ z) l + b1 l)) k + b2 k)) i
    + b3 i

/-- Free-parameter count of `ch10_mlp`: `W₁, b₁, W₂, b₂, W₃, b₃`. -/
def ch10_mlp_params (n h : ℕ) : ℕ := h * (n + 3) + h + h * h + h + n * h + n

-- eq:em-mlp: "W₁ ∈ ℝ^{128×(n+3)}, W₂ ∈ ℝ^{128×128}, W₃ ∈ ℝ^{n×128}, giving 25,248
-- free parameters" at n = 32.
theorem ch10_mlp_params_value : ch10_mlp_params 32 128 = 25248 := by
  unfold ch10_mlp_params; norm_num

-- eq:em-mlp: "Nothing in eq:em-mlp knows that the sites are ordered: permuting the
-- coordinates of x and the rows of the output is absorbed by permuting the columns
-- of W₁ and the rows of W₃."  Proved for an arbitrary permutation π of the input
-- coordinates and σ of the sites; the thesis's claim is the case where π fixes the
-- three time-embedding coordinates.
theorem ch10_mlp_permutation_blind {n h : ℕ} (π : Equiv.Perm (Fin (n + 3)))
    (σ : Equiv.Perm (Fin n)) (W1 : Matrix (Fin h) (Fin (n + 3)) ℝ) (b1 : Fin h → ℝ)
    (W2 : Matrix (Fin h) (Fin h) ℝ) (b2 : Fin h → ℝ)
    (W3 : Matrix (Fin n) (Fin h) ℝ) (b3 : Fin n → ℝ) (z : Fin (n + 3) → ℝ) :
    ch10_mlp (W1.submatrix id π) b1 W2 b2 (W3.submatrix σ id) (fun i => b3 (σ i))
        (fun j => z (π j))
      = fun i => ch10_mlp W1 b1 W2 b2 W3 b3 z (σ i) := by
  have hfirst : ((W1.submatrix id π) *ᵥ fun j => z (π j)) = W1 *ᵥ z := by
    funext l
    simp only [Matrix.mulVec, dotProduct, Matrix.submatrix_apply, id_eq]
    exact Equiv.sum_comp π (fun k => W1 l k * z k)
  have hrow : ∀ (H : Fin h → ℝ) (i : Fin n),
      ((W3.submatrix σ id) *ᵥ H) i = (W3 *ᵥ H) (σ i) := by
    intro H i
    simp [Matrix.mulVec, dotProduct, Matrix.submatrix_apply]
  funext i
  simp only [ch10_mlp, hfirst, hrow]

/-! ## eq:em-window (lines 257-261) -/

/-- eq:em-window: the weight-shared window head `ŷ_k = g_φ(x_{k-r},…,x_{k+r}; τ(t))`.
The sequence is a function on `ℤ` (zero-padded outside `1..n`), the same `g` at
every site. -/
def ch10_windowHead (r : ℕ) (g : (Fin (2 * r + 1) → ℝ) → (Fin 3 → ℝ) → ℝ)
    (x : ℤ → ℝ) (tau : Fin 3 → ℝ) (k : ℤ) : ℝ :=
  g (fun i => x (k - r + (i : ℕ))) tau

-- eq:em-window: "Its receptive radius is r by construction: no amount of data lets
-- it propagate information further."  Two sequences agreeing on [k-r, k+r] give the
-- same prediction at k, whatever g is.
theorem ch10_window_receptive (r : ℕ) (g : (Fin (2 * r + 1) → ℝ) → (Fin 3 → ℝ) → ℝ)
    (x y : ℤ → ℝ) (tau : Fin 3 → ℝ) (k : ℤ)
    (h : ∀ d : ℤ, |d| ≤ (r : ℤ) → x (k + d) = y (k + d)) :
    ch10_windowHead r g x tau k = ch10_windowHead r g y tau k := by
  unfold ch10_windowHead
  congr 1
  funext i
  have hi : ((i : ℕ) : ℤ) ≤ 2 * (r : ℤ) := by
    have := i.isLt
    omega
  have hnn : (0 : ℤ) ≤ ((i : ℕ) : ℤ) := Int.natCast_nonneg _
  have hd := h (((i : ℕ) : ℤ) - (r : ℤ)) (by rw [abs_le]; omega)
  have he : k - (r : ℤ) + ((i : ℕ) : ℤ) = k + (((i : ℕ) : ℤ) - (r : ℤ)) := by ring
  rw [he, hd]

/-- Free-parameter count of the window head at radius `r`, width `W`: input layer
`(2r+4)W + W`, hidden layer `W² + W`, scalar readout `W + 1`. -/
def ch10_window_params (r W : ℕ) : ℕ := ((2 * r + 4) * W + W) + (W * W + W) + (W + 1)

-- eq:em-window: the stated total `(2r+4)W + W² + 3W + 1`.
theorem ch10_window_params_formula (r W : ℕ) :
    ch10_window_params r W = (2 * r + 4) * W + W ^ 2 + 3 * W + 1 := by
  unfold ch10_window_params; ring

-- eq:em-window / tab:em-architectures: "the window head at its selected setting
-- (r = 2, W = 64) carries 4,801".
theorem ch10_window_params_selected : ch10_window_params 2 64 = 4801 := by
  unfold ch10_window_params; norm_num

/-! ## eq:em-dilated (lines 273-278) -/

/-- eq:em-dilated: one residual layer of the dilated stack,
`h^{(ℓ)}_k = h^{(ℓ-1)}_k + P_ℓ tanh(Σ_{s∈{-d,0,d}} W_{ℓ,s} h^{(ℓ-1)}_{k+s} + b_ℓ)`. -/
def ch10_dilatedLayer {w : ℕ} (P Wm W0 Wp : Matrix (Fin w) (Fin w) ℝ) (b : Fin w → ℝ) (d : ℤ)
    (h : ℤ → Fin w → ℝ) : ℤ → Fin w → ℝ :=
  fun k i => h k i +
    (P *ᵥ fun j => Real.tanh
      ((Wm *ᵥ h (k - d)) j + (W0 *ᵥ h k) j + (Wp *ᵥ h (k + d)) j + b j)) i

-- eq:em-dilated: "Its receptive radius is Σ_ℓ d_ℓ = 15".
theorem ch10_dilated_radius : (1 : ℕ) + 2 + 4 + 8 = 15 := by norm_num

-- eq:em-dilated, the companion claim "which spans an n = 32 chain end to end": a
-- radius-15 field covers 2·15+1 = 31 of the 32 sites, so it spans the chain only
-- from sites near the middle; sites 1 and 32 are 31 apart, further than 15.
theorem ch10_dilated_field_width : 2 * 15 + 1 = 31 ∧ (31 : ℕ) < 32 ∧ (15 : ℕ) < 31 := by
  refine ⟨by norm_num, by norm_num, by norm_num⟩

/-! ## eq:em-bimp (lines 286-292) -/

/-- eq:em-bimp: the forward recurrence `h⃗_k = tanh(A z_k + F h⃗_{k-1} + a)`, started
from `h⃗_{-1} = 0`. -/
def ch10_bimp_fwd {w m : ℕ} (A : Matrix (Fin w) (Fin m) ℝ) (F : Matrix (Fin w) (Fin w) ℝ)
    (a : Fin w → ℝ) (z : ℕ → Fin m → ℝ) : ℕ → Fin w → ℝ
  | 0 => fun i => Real.tanh ((A *ᵥ z 0) i + a i)
  | (k + 1) => fun i =>
      Real.tanh ((A *ᵥ z (k + 1)) i + (F *ᵥ ch10_bimp_fwd A F a z k) i + a i)

/-- eq:em-bimp: the joint readout `ŷ_k = C[h⃗_k; h⃖_k; z_k] + c`, written as three
blocks applied to the forward state, the backward state and the input. -/
def ch10_bimp_readout {w m p : ℕ} (C1 C2 : Matrix (Fin p) (Fin w) ℝ)
    (C3 : Matrix (Fin p) (Fin m) ℝ) (c : Fin p → ℝ)
    (hf hb : Fin w → ℝ) (z : Fin m → ℝ) : Fin p → ℝ :=
  fun i => (C1 *ᵥ hf) i + (C2 *ᵥ hb) i + (C3 *ᵥ z) i + c i

-- eq:em-bimp: the forward state at site k depends only on z_0,…,z_k — the
-- recurrence is causal, and it is the joint readout that makes the head
-- bidirectional ("run once in each direction, then combined").
theorem ch10_bimp_fwd_causal {w m : ℕ} (A : Matrix (Fin w) (Fin m) ℝ)
    (F : Matrix (Fin w) (Fin w) ℝ) (a : Fin w → ℝ) (z z' : ℕ → Fin m → ℝ) :
    ∀ k : ℕ, (∀ j ≤ k, z j = z' j) →
      ch10_bimp_fwd A F a z k = ch10_bimp_fwd A F a z' k := by
  intro k
  induction k with
  | zero => intro h; simp [ch10_bimp_fwd, h 0 (le_refl 0)]
  | succ k ih =>
      intro h
      have hz : z (k + 1) = z' (k + 1) := h (k + 1) (le_refl _)
      have hprev : ch10_bimp_fwd A F a z k = ch10_bimp_fwd A F a z' k :=
        ih fun j hj => h j (Nat.le_succ_of_le hj)
      simp [ch10_bimp_fwd, hz, hprev]

/-! ## eq:em-rate-ratio (lines 388-392) -/

-- eq:em-rate-ratio: if each arm's error is E ≈ c N^{-ϱ} then
-- R(N) = (c_base/c_EM) N^{ϱ_EM - ϱ_base}, "so the ratio grows precisely when the
-- structured arm converges faster in data".
theorem ch10_em_rate_ratio (cb ce rhob rhoe N : ℝ) (hN : 0 < N) (hce : ce ≠ 0) :
    (cb * N ^ (-rhob)) / (ce * N ^ (-rhoe)) = (cb / ce) * N ^ (rhoe - rhob) := by
  have hb : (0:ℝ) < N ^ rhob := Real.rpow_pos_of_pos hN _
  have he : (0:ℝ) < N ^ rhoe := Real.rpow_pos_of_pos hN _
  rw [Real.rpow_neg hN.le, Real.rpow_neg hN.le, Real.rpow_sub hN]
  have hb' : (N : ℝ) ^ rhob ≠ 0 := ne_of_gt hb
  have he' : (N : ℝ) ^ rhoe ≠ 0 := ne_of_gt he
  field_simp

-- eq:em-rate-ratio: "a 128-fold increase in data multiplies the ratio by
-- 128^{ϱ_EM - ϱ_base}".
theorem ch10_em_rate_ratio_growth (K rhob rhoe N : ℝ) (hN : 0 < N) (hK : 0 < K) :
    (K * N) ^ (rhoe - rhob) = K ^ (rhoe - rhob) * N ^ (rhoe - rhob) :=
  Real.mul_rpow hK.le hN.le

/-! ## eq:em-prior-precision (lines 431-438) -/

/-- The AR(1) covariance `Σ₀ = (ρ^{|i-j|})`. -/
def ch10_arSig (n : ℕ) (r : ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  Matrix.of fun i j => r ^ (((i : ℤ) - (j : ℤ)).natAbs)

-- eq:em-prior-precision: Q₀ = Σ₀⁻¹ is tridiagonal with interior diagonal
-- (1+ρ²)/σ_η², off-diagonal -ρ/σ_η² and zero beyond, σ_η² = 1-ρ².
-- Checked here at n = 2, 3, 4 with symbolic ρ by verifying Q₀ Σ₀ = I; these
-- instances are subsumed by the general-n theorem `ch10_em_prior_precision`
-- further down.  The boundary diagonal entries 1/σ_η² (not stated in the
-- display, which is explicitly the interior formula) are visible in the
-- matrices and are proved in general in `ch10_em_prior_precision_boundary`.
theorem ch10_arQ2_inv (r : ℝ) (h : 1 - r ^ 2 ≠ 0) :
    ((1 / (1 - r ^ 2)) • !![(1:ℝ), -r; -r, 1]) * !![(1:ℝ), r; r, 1] = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [Matrix.mul_apply, Fin.sum_univ_two, Matrix.one_apply] <;> field_simp <;> ring

theorem ch10_arQ3_inv (r : ℝ) (h : 1 - r ^ 2 ≠ 0) :
    ((1 / (1 - r ^ 2)) • !![(1:ℝ), -r, 0; -r, 1 + r ^ 2, -r; 0, -r, 1])
        * !![(1:ℝ), r, r ^ 2; r, 1, r; r ^ 2, r, 1] = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [Matrix.mul_apply, Fin.sum_univ_three, Matrix.one_apply] <;> field_simp <;> ring

theorem ch10_arQ4_inv (r : ℝ) (h : 1 - r ^ 2 ≠ 0) :
    ((1 / (1 - r ^ 2)) •
        !![(1:ℝ), -r, 0, 0; -r, 1 + r ^ 2, -r, 0; 0, -r, 1 + r ^ 2, -r; 0, 0, -r, 1])
        * !![(1:ℝ), r, r ^ 2, r ^ 3; r, 1, r, r ^ 2; r ^ 2, r, 1, r; r ^ 3, r ^ 2, r, 1]
      = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [Matrix.mul_apply, Fin.sum_univ_four, Matrix.one_apply] <;> field_simp <;> ring


/-! ### eq:em-prior-precision at every chain length

The instance checks above are superseded by the following general-`n` theorem:
for every `n ≥ 2` and every `r` with `1 - r² ≠ 0`, the inverse of the AR(1)
covariance `Σ₀ = (r^{|i-j|})` is *exactly* the tridiagonal matrix displayed in
eq:em-prior-precision.  (`n ≥ 2` is forced by the display itself, which speaks of
the neighbours `k ± 1`; at `n = 1` one has `Σ₀ = (1)` and `Q₀ = (1)`, not
`1/(1-r²)`.) -/

/-- The tridiagonal AR(1) precision matrix on a chain of `n` sites: diagonal
`(1+r²)/(1-r²)` in the interior and `1/(1-r²)` at the two ends, nearest-neighbour
off-diagonal `-r/(1-r²)`, and `0` at distance `≥ 2`. -/
def ch10_arQ (n : ℕ) (r : ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  Matrix.of fun i j =>
    if (i : ℕ) = (j : ℕ) then
      (if (i : ℕ) = 0 ∨ (i : ℕ) + 1 = n then 1 else 1 + r ^ 2) / (1 - r ^ 2)
    else if ((i : ℤ) - (j : ℤ)).natAbs = 1 then -r / (1 - r ^ 2)
    else 0

/-- `ch10_arQ` entry by entry (definitional). -/
theorem ch10_arQ_apply {n : ℕ} (r : ℝ) (i j : Fin n) :
    ch10_arQ n r i j =
      if (i : ℕ) = (j : ℕ) then
        (if (i : ℕ) = 0 ∨ (i : ℕ) + 1 = n then 1 else 1 + r ^ 2) / (1 - r ^ 2)
      else if ((i : ℤ) - (j : ℤ)).natAbs = 1 then -r / (1 - r ^ 2)
      else 0 := rfl

theorem ch10_arSig_apply_le {n : ℕ} (r : ℝ) (i j : Fin n) (hij : (j : ℕ) ≤ (i : ℕ)) :
    ch10_arSig n r i j = r ^ ((i : ℕ) - (j : ℕ)) := by
  unfold ch10_arSig
  simp only [Matrix.of_apply]
  exact congrArg (fun m : ℕ => r ^ m) (by omega)

theorem ch10_arSig_apply_ge {n : ℕ} (r : ℝ) (i j : Fin n) (hij : (i : ℕ) ≤ (j : ℕ)) :
    ch10_arSig n r i j = r ^ ((j : ℕ) - (i : ℕ)) := by
  unfold ch10_arSig
  simp only [Matrix.of_apply]
  exact congrArg (fun m : ℕ => r ^ m) (by omega)

/-- A sum over `Fin n` of a term supported on the single index `k₀`. -/
theorem ch10_sum_pick {n : ℕ} (P : ℕ → Prop) [DecidablePred P] (g : Fin n → ℝ)
    (k₀ : Fin n) (h₀ : P (k₀ : ℕ)) (huniq : ∀ k : Fin n, P (k : ℕ) → k = k₀) :
    ∑ k : Fin n, (if P (k : ℕ) then g k else 0) = g k₀ := by
  refine (Finset.sum_eq_single_of_mem k₀ (Finset.mem_univ _) ?_).trans (if_pos h₀)
  intro b _ hb
  exact if_neg fun hp => hb (huniq b hp)

/-- A sum over `Fin n` of a term with empty support. -/
theorem ch10_sum_none {n : ℕ} (P : ℕ → Prop) [DecidablePred P] (g : Fin n → ℝ)
    (hP : ∀ k : Fin n, ¬ P (k : ℕ)) :
    ∑ k : Fin n, (if P (k : ℕ) then g k else 0) = 0 :=
  Finset.sum_eq_zero fun k _ => if_neg (hP k)

/-- Row `i` of `Q₀ Σ₀`: only the diagonal and the two neighbours of `i` survive. -/
theorem ch10_arQ_row {n : ℕ} (r : ℝ) (i j : Fin n) :
    ∑ k : Fin n, ch10_arQ n r i k * ch10_arSig n r k j
      = (if (i : ℕ) = 0 ∨ (i : ℕ) + 1 = n then 1 else 1 + r ^ 2) / (1 - r ^ 2)
            * ch10_arSig n r i j
        + (∑ k : Fin n, (if (k : ℕ) + 1 = (i : ℕ)
            then (-r / (1 - r ^ 2)) * ch10_arSig n r k j else 0))
        + (∑ k : Fin n, (if (i : ℕ) + 1 = (k : ℕ)
            then (-r / (1 - r ^ 2)) * ch10_arSig n r k j else 0)) := by
  have hsplit : ∀ k : Fin n, ch10_arQ n r i k * ch10_arSig n r k j
      = (if (i : ℕ) = (k : ℕ)
          then (if (i : ℕ) = 0 ∨ (i : ℕ) + 1 = n then 1 else 1 + r ^ 2) / (1 - r ^ 2)
                * ch10_arSig n r k j else 0)
        + (if (k : ℕ) + 1 = (i : ℕ) then (-r / (1 - r ^ 2)) * ch10_arSig n r k j else 0)
        + (if (i : ℕ) + 1 = (k : ℕ) then (-r / (1 - r ^ 2)) * ch10_arSig n r k j else 0) := by
    intro k
    rw [ch10_arQ_apply]
    split_ifs <;> first | ring1 | (exfalso; omega)
  have hdiag : (∑ k : Fin n, (if (i : ℕ) = (k : ℕ)
      then (if (i : ℕ) = 0 ∨ (i : ℕ) + 1 = n then 1 else 1 + r ^ 2) / (1 - r ^ 2)
            * ch10_arSig n r k j else 0))
      = (if (i : ℕ) = 0 ∨ (i : ℕ) + 1 = n then 1 else 1 + r ^ 2) / (1 - r ^ 2)
          * ch10_arSig n r i j :=
    ch10_sum_pick (fun m => (i : ℕ) = m) _ i rfl fun k hk => Fin.val_injective hk.symm
  rw [Finset.sum_congr rfl fun k _ => hsplit k]
  simp only [Finset.sum_add_distrib]
  rw [hdiag]

/-- **eq:em-prior-precision, general `n`.**  `Q₀ Σ₀ = I` for the AR(1) chain of
any length `n ≥ 2`, with `Q₀` the tridiagonal matrix `ch10_arQ`. -/
theorem ch10_arQ_mul_arSig {n : ℕ} (hn : 2 ≤ n) (r : ℝ) (h : 1 - r ^ 2 ≠ 0) :
    ch10_arQ n r * ch10_arSig n r = 1 := by
  ext i j
  rw [Matrix.mul_apply, ch10_arQ_row, Matrix.one_apply]
  rcases Nat.eq_zero_or_pos (i : ℕ) with hi0 | hipos
  · -- `i` is the first site: only the right neighbour contributes
    obtain ⟨k1, hk1⟩ : ∃ k : Fin n, (k : ℕ) = 1 := ⟨⟨1, by omega⟩, rfl⟩
    have hL : (∑ k : Fin n, (if (k : ℕ) + 1 = (i : ℕ)
        then (-r / (1 - r ^ 2)) * ch10_arSig n r k j else 0)) = 0 :=
      ch10_sum_none (fun m => m + 1 = (i : ℕ)) _ fun k => by omega
    have hR : (∑ k : Fin n, (if (i : ℕ) + 1 = (k : ℕ)
        then (-r / (1 - r ^ 2)) * ch10_arSig n r k j else 0))
        = (-r / (1 - r ^ 2)) * ch10_arSig n r k1 j :=
      ch10_sum_pick (fun m => (i : ℕ) + 1 = m) _ k1 (by omega)
        fun k hk => Fin.val_injective (by omega)
    rw [hL, hR, if_pos (show (i : ℕ) = 0 ∨ (i : ℕ) + 1 = n from Or.inl hi0)]
    rcases Nat.eq_zero_or_pos (j : ℕ) with hj0 | hjpos
    · have hij : i = j := Fin.val_injective (by omega)
      rw [ch10_arSig_apply_le r i j (by omega), ch10_arSig_apply_le r k1 j (by omega),
        if_pos hij, (show (i : ℕ) - (j : ℕ) = 0 by omega),
        (show (k1 : ℕ) - (j : ℕ) = 1 by omega)]
      field_simp
      try ring
    · have hij : ¬ (i = j) := fun e => by subst e; omega
      obtain ⟨p, hp⟩ : ∃ p : ℕ, (j : ℕ) = p + 1 := ⟨(j : ℕ) - 1, by omega⟩
      rw [ch10_arSig_apply_ge r i j (by omega), ch10_arSig_apply_ge r k1 j (by omega),
        if_neg hij, (show (j : ℕ) - (i : ℕ) = p + 1 by omega),
        (show (j : ℕ) - (k1 : ℕ) = p by omega)]
      field_simp
      try ring
  · -- `i` has a left neighbour
    obtain ⟨km, hkm⟩ : ∃ k : Fin n, (k : ℕ) = (i : ℕ) - 1 :=
      ⟨⟨(i : ℕ) - 1, by have := i.isLt; omega⟩, rfl⟩
    have hL : (∑ k : Fin n, (if (k : ℕ) + 1 = (i : ℕ)
        then (-r / (1 - r ^ 2)) * ch10_arSig n r k j else 0))
        = (-r / (1 - r ^ 2)) * ch10_arSig n r km j :=
      ch10_sum_pick (fun m => m + 1 = (i : ℕ)) _ km (by omega)
        fun k hk => Fin.val_injective (by omega)
    rw [hL]
    rcases Nat.lt_or_ge ((i : ℕ) + 1) n with hint | hlast
    · -- interior site: both neighbours present
      obtain ⟨kp, hkp⟩ : ∃ k : Fin n, (k : ℕ) = (i : ℕ) + 1 := ⟨⟨(i : ℕ) + 1, hint⟩, rfl⟩
      have hR : (∑ k : Fin n, (if (i : ℕ) + 1 = (k : ℕ)
          then (-r / (1 - r ^ 2)) * ch10_arSig n r k j else 0))
          = (-r / (1 - r ^ 2)) * ch10_arSig n r kp j :=
        ch10_sum_pick (fun m => (i : ℕ) + 1 = m) _ kp (by omega)
          fun k hk => Fin.val_injective (by omega)
      rw [hR, if_neg (show ¬ ((i : ℕ) = 0 ∨ (i : ℕ) + 1 = n) by omega)]
      rcases lt_trichotomy (j : ℕ) (i : ℕ) with hji | hji | hji
      · have hij : ¬ (i = j) := fun e => by subst e; omega
        obtain ⟨p, hp⟩ : ∃ p : ℕ, (i : ℕ) = (j : ℕ) + p + 1 :=
          ⟨(i : ℕ) - (j : ℕ) - 1, by omega⟩
        rw [ch10_arSig_apply_le r i j (by omega), ch10_arSig_apply_le r km j (by omega),
          ch10_arSig_apply_le r kp j (by omega), if_neg hij,
          (show (i : ℕ) - (j : ℕ) = p + 1 by omega),
          (show (km : ℕ) - (j : ℕ) = p by omega),
          (show (kp : ℕ) - (j : ℕ) = p + 2 by omega)]
        field_simp
        try ring
      · have hij : i = j := Fin.val_injective hji.symm
        rw [ch10_arSig_apply_le r i j (by omega), ch10_arSig_apply_ge r km j (by omega),
          ch10_arSig_apply_le r kp j (by omega), if_pos hij,
          (show (i : ℕ) - (j : ℕ) = 0 by omega),
          (show (j : ℕ) - (km : ℕ) = 1 by omega),
          (show (kp : ℕ) - (j : ℕ) = 1 by omega)]
        field_simp
        try ring
      · have hij : ¬ (i = j) := fun e => by subst e; omega
        obtain ⟨p, hp⟩ : ∃ p : ℕ, (j : ℕ) = (i : ℕ) + p + 1 :=
          ⟨(j : ℕ) - (i : ℕ) - 1, by omega⟩
        rw [ch10_arSig_apply_ge r i j (by omega), ch10_arSig_apply_ge r km j (by omega),
          ch10_arSig_apply_ge r kp j (by omega), if_neg hij,
          (show (j : ℕ) - (i : ℕ) = p + 1 by omega),
          (show (j : ℕ) - (km : ℕ) = p + 2 by omega),
          (show (j : ℕ) - (kp : ℕ) = p by omega)]
        field_simp
        try ring
    · -- last site: no right neighbour
      have hlastEq : (i : ℕ) + 1 = n := by have := i.isLt; omega
      have hR : (∑ k : Fin n, (if (i : ℕ) + 1 = (k : ℕ)
          then (-r / (1 - r ^ 2)) * ch10_arSig n r k j else 0)) = 0 :=
        ch10_sum_none (fun m => (i : ℕ) + 1 = m) _ fun k => by have := k.isLt; omega
      rw [hR, if_pos (show (i : ℕ) = 0 ∨ (i : ℕ) + 1 = n from Or.inr hlastEq)]
      rcases lt_trichotomy (j : ℕ) (i : ℕ) with hji | hji | hji
      · have hij : ¬ (i = j) := fun e => by subst e; omega
        obtain ⟨p, hp⟩ : ∃ p : ℕ, (i : ℕ) = (j : ℕ) + p + 1 :=
          ⟨(i : ℕ) - (j : ℕ) - 1, by omega⟩
        rw [ch10_arSig_apply_le r i j (by omega), ch10_arSig_apply_le r km j (by omega),
          if_neg hij, (show (i : ℕ) - (j : ℕ) = p + 1 by omega),
          (show (km : ℕ) - (j : ℕ) = p by omega)]
        field_simp
        try ring
      · have hij : i = j := Fin.val_injective hji.symm
        rw [ch10_arSig_apply_le r i j (by omega), ch10_arSig_apply_ge r km j (by omega),
          if_pos hij, (show (i : ℕ) - (j : ℕ) = 0 by omega),
          (show (j : ℕ) - (km : ℕ) = 1 by omega)]
        field_simp
        try ring
      · exact absurd j.isLt (by omega)

/-- The AR(1) precision matrix is the inverse of the AR(1) covariance, at every
chain length `n ≥ 2`. -/
theorem ch10_arSig_inv {n : ℕ} (hn : 2 ≤ n) (r : ℝ) (h : 1 - r ^ 2 ≠ 0) :
    (ch10_arSig n r)⁻¹ = ch10_arQ n r :=
  Matrix.inv_eq_left_inv (ch10_arQ_mul_arSig hn r h)

/-- **eq:em-prior-precision, exactly as displayed, for every `n ≥ 2`.**
`Q₀ = Σ₀⁻¹` has interior diagonal `(1+r²)/(1-r²)`, nearest-neighbour entries
`-r/(1-r²)`, and vanishes at distance `≥ 2`. -/
theorem ch10_em_prior_precision {n : ℕ} (hn : 2 ≤ n) (r : ℝ) (h : 1 - r ^ 2 ≠ 0) :
    (∀ k : Fin n, 0 < (k : ℕ) → (k : ℕ) + 1 < n →
        (ch10_arSig n r)⁻¹ k k = (1 + r ^ 2) / (1 - r ^ 2)) ∧
    (∀ k j : Fin n, (k : ℕ) + 1 = (j : ℕ) ∨ (j : ℕ) + 1 = (k : ℕ) →
        (ch10_arSig n r)⁻¹ k j = -r / (1 - r ^ 2)) ∧
    (∀ k j : Fin n, 2 ≤ ((k : ℤ) - (j : ℤ)).natAbs →
        (ch10_arSig n r)⁻¹ k j = 0) := by
  have hinv := ch10_arSig_inv hn r h
  refine ⟨fun k hk1 hk2 => ?_, fun k j hkj => ?_, fun k j hkj => ?_⟩
  · rw [hinv, ch10_arQ_apply, if_pos (show (k : ℕ) = (k : ℕ) from rfl),
      if_neg (show ¬ ((k : ℕ) = 0 ∨ (k : ℕ) + 1 = n) by rintro (hc | hc) <;> omega)]
  · rcases hkj with hkj | hkj
    · rw [hinv, ch10_arQ_apply, if_neg (show ¬ ((k : ℕ) = (j : ℕ)) by omega),
        if_pos (show ((k : ℤ) - (j : ℤ)).natAbs = 1 by omega)]
    · rw [hinv, ch10_arQ_apply, if_neg (show ¬ ((k : ℕ) = (j : ℕ)) by omega),
        if_pos (show ((k : ℤ) - (j : ℤ)).natAbs = 1 by omega)]
  · rw [hinv, ch10_arQ_apply, if_neg (show ¬ ((k : ℕ) = (j : ℕ)) by omega),
      if_neg (show ¬ (((k : ℤ) - (j : ℤ)).natAbs = 1) by omega)]

/-- The two boundary diagonal entries of `Q₀` are `1/(1-r²)`, not the interior
value `(1+r²)/(1-r²)`; the display in the thesis is explicitly the interior one,
and this is the value the trace closed forms `ch10_trQ0`, `ch10_trQ0sq` use. -/
theorem ch10_em_prior_precision_boundary {n : ℕ} (hn : 2 ≤ n) (r : ℝ) (h : 1 - r ^ 2 ≠ 0)
    (k : Fin n) (hk : (k : ℕ) = 0 ∨ (k : ℕ) + 1 = n) :
    (ch10_arSig n r)⁻¹ k k = 1 / (1 - r ^ 2) := by
  rw [ch10_arSig_inv hn r h, ch10_arQ_apply,
    if_pos (show (k : ℕ) = (k : ℕ) from rfl), if_pos hk]

/-! ## eq:em-posterior-precision (lines 440-448) -/

-- eq:em-posterior-precision: -log p(a|x) = ½ aᵀJa - hᵀa + const with
-- J = Q₀ + (e^{-2t}/Δ_t) I and h = (e^{-t}/Δ_t) x.  Here c = e^{-t}, so
-- e^{-2t} = c²: "the channel enters the diagonal alone" — it adds c²/Δ to the
-- diagonal of J and nothing off it.
theorem ch10_em_posterior_precision {n : ℕ} (Q0 : Matrix (Fin n) (Fin n) ℝ)
    (c Δ : ℝ) (hΔ : Δ ≠ 0) (x a : Fin n → ℝ) :
    (1 / 2) * (a ⬝ᵥ (Q0 *ᵥ a)) + (1 / (2 * Δ)) * ((x - c • a) ⬝ᵥ (x - c • a))
      = (1 / 2) * (a ⬝ᵥ ((Q0 + (c ^ 2 / Δ) • (1 : Matrix (Fin n) (Fin n) ℝ)) *ᵥ a))
        - ((c / Δ) • x) ⬝ᵥ a + (1 / (2 * Δ)) * (x ⬝ᵥ x) := by
  have hmul : ((Q0 + (c ^ 2 / Δ) • (1 : Matrix (Fin n) (Fin n) ℝ)) *ᵥ a)
      = (Q0 *ᵥ a) + (c ^ 2 / Δ) • a := by
    rw [Matrix.add_mulVec, Matrix.smul_mulVec, Matrix.one_mulVec]
  rw [hmul]
  simp only [dotProduct_add, dotProduct_sub, sub_dotProduct, dotProduct_smul, smul_dotProduct,
    smul_eq_mul]
  rw [dotProduct_comm a x]
  field_simp
  ring

-- eq:em-posterior-precision, second half: "the posterior is Gaussian with mean
-- m = E[a|x] = J⁻¹h" — completing the square, with m characterised by J m = h.
theorem ch10_em_posterior_mean {n : ℕ} (J : Matrix (Fin n) (Fin n) ℝ) (hsym : Jᵀ = J)
    (m h a : Fin n → ℝ) (hm : J *ᵥ m = h) :
    (1 / 2) * (a ⬝ᵥ (J *ᵥ a)) - h ⬝ᵥ a
      = (1 / 2) * ((a - m) ⬝ᵥ (J *ᵥ (a - m))) - (1 / 2) * (m ⬝ᵥ h) := by
  have hswap : m ⬝ᵥ (J *ᵥ a) = h ⬝ᵥ a := by
    rw [Matrix.dotProduct_mulVec, ← hsym, Matrix.vecMul_transpose, hm]
  have hmm : m ⬝ᵥ (J *ᵥ m) = m ⬝ᵥ h := by rw [hm]
  have ham : a ⬝ᵥ (J *ᵥ m) = a ⬝ᵥ h := by rw [hm]
  rw [Matrix.mulVec_sub]
  simp only [dotProduct_sub, sub_dotProduct]
  rw [hswap, hmm, ham, dotProduct_comm h a]
  ring

-- eq:em-posterior-precision: with J positive definite the completed square makes
-- m the unique minimiser, which is what "the posterior mean is J⁻¹h" asserts.
theorem ch10_em_posterior_mean_min {n : ℕ} (J : Matrix (Fin n) (Fin n) ℝ) (hsym : Jᵀ = J)
    (hpd : ∀ v : Fin n → ℝ, v ≠ 0 → 0 < v ⬝ᵥ (J *ᵥ v))
    (m h a : Fin n → ℝ) (hm : J *ᵥ m = h) (hne : a ≠ m) :
    (1 / 2) * (m ⬝ᵥ (J *ᵥ m)) - h ⬝ᵥ m < (1 / 2) * (a ⬝ᵥ (J *ᵥ a)) - h ⬝ᵥ a := by
  have h1 := ch10_em_posterior_mean J hsym m h a hm
  have h2 := ch10_em_posterior_mean J hsym m h m hm
  have hpos : 0 < (a - m) ⬝ᵥ (J *ᵥ (a - m)) := hpd _ (sub_ne_zero.mpr hne)
  rw [h1, h2]
  have hz : ((m - m) ⬝ᵥ (J *ᵥ (m - m))) = 0 := by
    simp
  rw [hz]
  linarith

/-! ## eq:em-bulk-params (lines 456-464) -/

/-- The evidence part of the bulk diagonal, `e^{-2t}/Δ_t`. -/
def ch10_Jevid (t : ℝ) : ℝ := Real.exp (-(2 * t)) / ch10_Delta t

-- eq:em-bulk-params: "Only J_d depends on t", through e^{-2t}/Δ_t = 1/(e^{2t}-1).
theorem ch10_Jevid_eq {t : ℝ} (ht : 0 < t) : ch10_Jevid t = 1 / (Real.exp (2 * t) - 1) := by
  have hgt : 1 < Real.exp (2 * t) := by
    have h := Real.exp_lt_exp.mpr (show (0:ℝ) < 2 * t by linarith)
    simpa using h
  have hinv : Real.exp (-(2 * t)) = (Real.exp (2 * t))⁻¹ := Real.exp_neg _
  have hne : Real.exp (2 * t) ≠ 0 := ne_of_gt (Real.exp_pos _)
  unfold ch10_Jevid ch10_Delta
  rw [hinv]
  field_simp

-- eq:em-bulk-params: J_d "is monotone" in t — strictly decreasing on t > 0.
theorem ch10_Jevid_strictAnti {s t : ℝ} (hs : 0 < s) (hst : s < t) :
    ch10_Jevid t < ch10_Jevid s := by
  have h1 : 1 < Real.exp (2 * s) := by
    have h := Real.exp_lt_exp.mpr (show (0:ℝ) < 2 * s by linarith)
    simpa using h
  have h2 : Real.exp (2 * s) < Real.exp (2 * t) := Real.exp_lt_exp.mpr (by linarith)
  rw [ch10_Jevid_eq hs, ch10_Jevid_eq (lt_trans hs hst)]
  exact one_div_lt_one_div_of_lt (by linarith) (by linarith)

-- eq:em-bulk-params: "it diverges like 1/(2t) as t ↓ 0", as the two-sided bound
-- 1/(2t e^{2t}) ≤ e^{-2t}/Δ_t ≤ 1/(2t) for t > 0.
theorem ch10_Jevid_bounds {t : ℝ} (ht : 0 < t) :
    1 / (2 * t * Real.exp (2 * t)) ≤ ch10_Jevid t ∧ ch10_Jevid t ≤ 1 / (2 * t) := by
  have hE : 0 < Real.exp (2 * t) := Real.exp_pos _
  have hgt : 1 < Real.exp (2 * t) := by
    have h := Real.exp_lt_exp.mpr (show (0:ℝ) < 2 * t by linarith)
    simpa using h
  have hlin : 1 + 2 * t ≤ Real.exp (2 * t) := by
    have := Real.add_one_le_exp (2 * t); linarith
  have he : Real.exp (-(2 * t)) * Real.exp (2 * t) = 1 := by
    rw [← Real.exp_add]; simp
  have hupper : Real.exp (2 * t) - 1 ≤ 2 * t * Real.exp (2 * t) := by
    have h := Real.add_one_le_exp (-(2 * t))
    have h2 := mul_le_mul_of_nonneg_right h hE.le
    rw [he] at h2
    nlinarith [h2]
  rw [ch10_Jevid_eq ht]
  constructor
  · exact one_div_le_one_div_of_le (by linarith) hupper
  · exact one_div_le_one_div_of_le (by linarith) (by linarith)

/-! ## eq:em-bulk-cov (prop:em-bulk, lines 472-515) -/

/-- The decay root of the bulk transfer equation `β q² + J_d q + β = 0` inside the
unit disc, written so that it is valid for either sign of `β`:
`-(J_d - √(J_d²-4β²))/(2β)`.  For `β < 0` (the operating regime `ρ > 0`) this is
the displayed `(J_d - √(J_d²-4β²))/(2|β|) > 0`; for `β > 0` it is its negative,
the sign alternation noted parenthetically in the toolbox. -/
def ch10_qroot (Jd b : ℝ) : ℝ := -(Jd - Real.sqrt (Jd ^ 2 - 4 * b ^ 2)) / (2 * b)

/-- The displayed prefactor `V = 1/√(J_d² - 4β²)`. -/
def ch10_V (Jd b : ℝ) : ℝ := 1 / Real.sqrt (Jd ^ 2 - 4 * b ^ 2)

/-- The bulk Green's function `G(d) = V q^{|d|}` of eq:em-bulk-cov. -/
def ch10_G (Jd b : ℝ) (d : ℕ) : ℝ := ch10_V Jd b * (ch10_qroot Jd b) ^ d

section Bulk

variable {Jd b : ℝ}

theorem ch10_bulk_sq_abs_pos (hb : b ≠ 0) : 0 < b ^ 2 := by
  have h := abs_pos.mpr hb
  nlinarith [mul_pos h h, sq_abs b]

-- toolbox display (lines 507-512): the reality margin
-- J_d - 2|β| = e^{-2t}/Δ_t + (1-|ρ|)²/σ_η² > 0, both terms non-negative and the
-- first strictly positive for t > 0.  Checked in the stated closed form.
theorem ch10_noname_2 (r t : ℝ) (ht : 0 < t) (hr : |r| < 1) :
    (ch10_Jevid t + (1 + r ^ 2) / (1 - r ^ 2)) - 2 * |(-(r / (1 - r ^ 2)))|
      = ch10_Jevid t + (1 - |r|) ^ 2 / (1 - r ^ 2)
    ∧ 0 < (ch10_Jevid t + (1 + r ^ 2) / (1 - r ^ 2)) - 2 * |(-(r / (1 - r ^ 2)))| := by
  have hr2 : r ^ 2 < 1 := by
    have := abs_nonneg r
    nlinarith [sq_abs r, abs_nonneg r]
  have hden : 0 < 1 - r ^ 2 := by linarith
  have habs : |(-(r / (1 - r ^ 2)))| = |r| / (1 - r ^ 2) := by
    rw [abs_neg, abs_div, abs_of_pos hden]
  have hsq : |r| ^ 2 = r ^ 2 := sq_abs r
  have hJ : 0 < ch10_Jevid t := by
    rw [ch10_Jevid_eq ht]
    have hgt : 1 < Real.exp (2 * t) := by
      have h := Real.exp_lt_exp.mpr (show (0:ℝ) < 2 * t by linarith)
      simpa using h
    positivity
  constructor
  · rw [habs]
    field_simp
    nlinarith [hsq]
  · rw [habs]
    have hpos : 0 ≤ (1 - |r|) ^ 2 / (1 - r ^ 2) := by
      apply div_nonneg (sq_nonneg _) hden.le
    have heq : (1 + r ^ 2) / (1 - r ^ 2) - 2 * (|r| / (1 - r ^ 2))
        = (1 - |r|) ^ 2 / (1 - r ^ 2) := by
      field_simp
      nlinarith [hsq]
    linarith [heq ▸ hpos]

theorem ch10_bulk_disc_pos (hb : b ≠ 0) (hJ : 2 * |b| < Jd) : 0 < Jd ^ 2 - 4 * b ^ 2 := by
  have habs : 0 ≤ |b| := abs_nonneg b
  have h1 : 0 < Jd - 2 * |b| := by linarith
  have h2 : 0 < Jd + 2 * |b| := by linarith
  have h3 : 0 < (Jd - 2 * |b|) * (Jd + 2 * |b|) := mul_pos h1 h2
  have h4 : (Jd - 2 * |b|) * (Jd + 2 * |b|) = Jd ^ 2 - 4 * |b| ^ 2 := by ring
  rw [h4, sq_abs] at h3
  linarith

theorem ch10_bulk_sqrt_pos (hb : b ≠ 0) (hJ : 2 * |b| < Jd) :
    0 < Real.sqrt (Jd ^ 2 - 4 * b ^ 2) := Real.sqrt_pos.mpr (ch10_bulk_disc_pos hb hJ)

theorem ch10_bulk_sqrt_sq (hb : b ≠ 0) (hJ : 2 * |b| < Jd) :
    Real.sqrt (Jd ^ 2 - 4 * b ^ 2) ^ 2 = Jd ^ 2 - 4 * b ^ 2 :=
  Real.sq_sqrt (le_of_lt (ch10_bulk_disc_pos hb hJ))

theorem ch10_bulk_sqrt_lt (hb : b ≠ 0) (hJ : 2 * |b| < Jd) :
    Real.sqrt (Jd ^ 2 - 4 * b ^ 2) < Jd := by
  have hJpos : 0 < Jd := lt_of_le_of_lt (by positivity) hJ
  have hb2 : 0 < b ^ 2 := ch10_bulk_sq_abs_pos hb
  have h1 : Jd ^ 2 - 4 * b ^ 2 < Jd ^ 2 := by linarith
  have h2 : Real.sqrt (Jd ^ 2 - 4 * b ^ 2) < Real.sqrt (Jd ^ 2) :=
    Real.sqrt_lt_sqrt (le_of_lt (ch10_bulk_disc_pos hb hJ)) h1
  rwa [Real.sqrt_sq hJpos.le] at h2

-- toolbox display (lines 491-495): with the geometric ansatz the three-term
-- relation at every d ≥ 1 collapses to the single quadratic β q² + J_d q + β = 0,
-- "a quadratic that is the same for every d — which is why one geometric sequence
-- suffices".
theorem ch10_noname_1 (V q : ℝ) (hV : V ≠ 0) (hq : q ≠ 0) (e : ℕ) :
    b * (V * q ^ (e + 2)) + Jd * (V * q ^ (e + 1)) + b * (V * q ^ e) = 0
      ↔ b * q ^ 2 + Jd * q + b = 0 := by
  have hfac : b * (V * q ^ (e + 2)) + Jd * (V * q ^ (e + 1)) + b * (V * q ^ e)
      = (V * q ^ e) * (b * q ^ 2 + Jd * q + b) := by ring
  rw [hfac]
  constructor
  · intro h
    rcases mul_eq_zero.mp h with h1 | h1
    · exact absurd h1 (mul_ne_zero hV (pow_ne_zero _ hq))
    · exact h1
  · intro h; rw [h, mul_zero]

-- toolbox: "Its two roots multiply to β/β = 1, so they are q and 1/q" — the
-- displayed root solves the quadratic exactly.
theorem ch10_qroot_quadratic (hb : b ≠ 0) (hJ : 2 * |b| < Jd) :
    b * (ch10_qroot Jd b) ^ 2 + Jd * (ch10_qroot Jd b) + b = 0 := by
  have hs := ch10_bulk_sqrt_sq hb hJ
  have h4b : (4 : ℝ) * b ≠ 0 := by
    simpa using hb
  have key : (b * (ch10_qroot Jd b) ^ 2 + Jd * (ch10_qroot Jd b) + b) * (4 * b)
      = Real.sqrt (Jd ^ 2 - 4 * b ^ 2) ^ 2 - (Jd ^ 2 - 4 * b ^ 2) := by
    unfold ch10_qroot
    field_simp
    ring
  rw [hs, sub_self] at key
  exact (mul_eq_zero.mp key).resolve_right h4b

-- eq:em-bulk-cov: 0 < |q| < 1, "exactly one lies inside the unit disc".
theorem ch10_qroot_abs (hb : b ≠ 0) (hJ : 2 * |b| < Jd) :
    |ch10_qroot Jd b| = (Jd - Real.sqrt (Jd ^ 2 - 4 * b ^ 2)) / (2 * |b|) := by
  have hslt := ch10_bulk_sqrt_lt hb hJ
  have hbpos : 0 < |b| := abs_pos.mpr hb
  unfold ch10_qroot
  rw [abs_div, abs_neg, abs_of_pos (by linarith), abs_mul,
    abs_of_pos (by norm_num : (0:ℝ) < 2)]

theorem ch10_qroot_abs_lt_one (hb : b ≠ 0) (hJ : 2 * |b| < Jd) :
    0 < |ch10_qroot Jd b| ∧ |ch10_qroot Jd b| < 1 := by
  have hslt := ch10_bulk_sqrt_lt hb hJ
  have hspos := ch10_bulk_sqrt_pos hb hJ
  have hs := ch10_bulk_sqrt_sq hb hJ
  have hbpos : 0 < |b| := abs_pos.mpr hb
  rw [ch10_qroot_abs hb hJ]
  refine ⟨div_pos (by linarith) (by linarith), ?_⟩
  rw [div_lt_one (by linarith)]
  by_contra hcon
  push_neg at hcon
  have h6 : Real.sqrt (Jd ^ 2 - 4 * b ^ 2) ≤ Jd - 2 * |b| := by linarith
  have h7 : Real.sqrt (Jd ^ 2 - 4 * b ^ 2) ^ 2 ≤ (Jd - 2 * |b|) ^ 2 := by
    nlinarith [hspos, h6]
  rw [hs] at h7
  nlinarith [mul_pos hbpos (show (0:ℝ) < Jd - 2 * |b| by linarith), sq_abs b]

-- eq:em-bulk-cov: the prefactor relation at d = 0, V √(J_d² - 4β²) = 1.
theorem ch10_V_normalisation (hb : b ≠ 0) (hJ : 2 * |b| < Jd) :
    ch10_V Jd b * Real.sqrt (Jd ^ 2 - 4 * b ^ 2) = 1 := by
  have hspos := ch10_bulk_sqrt_pos hb hJ
  have hne : Real.sqrt (Jd ^ 2 - 4 * b ^ 2) ≠ 0 := ne_of_gt hspos
  unfold ch10_V
  field_simp

-- prop:em-bulk / eq:em-bulk-cov: the full statement.  G(d) = V q^{|d|} is the
-- Green's function of the bulk tridiagonal Toeplitz operator:
-- β G(d+1) + J_d G(d) + β G(d-1) = δ_{d,0} at every separation d ≥ 0.
theorem ch10_em_bulk_cov (hb : b ≠ 0) (hJ : 2 * |b| < Jd) :
    (∀ d : ℕ, b * ch10_G Jd b (d + 2) + Jd * ch10_G Jd b (d + 1) + b * ch10_G Jd b d = 0)
      ∧ (b * ch10_G Jd b 1 + Jd * ch10_G Jd b 0 + b * ch10_G Jd b 1 = 1) := by
  have hquad := ch10_qroot_quadratic hb hJ
  have hVnorm := ch10_V_normalisation hb hJ
  constructor
  · intro d
    have hfac : b * ch10_G Jd b (d + 2) + Jd * ch10_G Jd b (d + 1) + b * ch10_G Jd b d
        = (ch10_V Jd b * (ch10_qroot Jd b) ^ d)
          * (b * (ch10_qroot Jd b) ^ 2 + Jd * (ch10_qroot Jd b) + b) := by
      unfold ch10_G; ring
    rw [hfac, hquad, mul_zero]
  · have hq' : ch10_qroot Jd b = (Real.sqrt (Jd ^ 2 - 4 * b ^ 2) - Jd) / (2 * b) := by
      unfold ch10_qroot; ring
    have hcancel : 2 * b * ((Real.sqrt (Jd ^ 2 - 4 * b ^ 2) - Jd) / (2 * b))
        = Real.sqrt (Jd ^ 2 - 4 * b ^ 2) - Jd := by
      field_simp
    have h2 : 2 * b * ch10_qroot Jd b + Jd = Real.sqrt (Jd ^ 2 - 4 * b ^ 2) := by
      rw [hq', hcancel]; ring
    have hfac : b * ch10_G Jd b 1 + Jd * ch10_G Jd b 0 + b * ch10_G Jd b 1
        = ch10_V Jd b * (2 * b * ch10_qroot Jd b + Jd) := by
      unfold ch10_G; ring
    rw [hfac, h2, hVnorm]

-- eq:em-q-limits (lines 524-536), the t → 0 end: q ≈ |β|/J_d → 0, here as the
-- exact bound |q| ≤ 2|β|/J_d (so q → 0 as J_d → ∞ with β fixed).
theorem ch10_em_q_limits_small_t (hb : b ≠ 0) (hJ : 2 * |b| < Jd) :
    |ch10_qroot Jd b| ≤ 2 * |b| / Jd := by
  have hJpos : 0 < Jd := lt_of_le_of_lt (by positivity) hJ
  have hslt := ch10_bulk_sqrt_lt hb hJ
  have hspos := ch10_bulk_sqrt_pos hb hJ
  have hs := ch10_bulk_sqrt_sq hb hJ
  have hbpos : 0 < |b| := abs_pos.mpr hb
  rw [ch10_qroot_abs hb hJ, div_le_div_iff₀ (by linarith) hJpos]
  nlinarith [hs, mul_pos hspos (show (0:ℝ) < Jd - Real.sqrt (Jd ^ 2 - 4 * b ^ 2) by linarith),
    sq_abs b]

-- eq:em-q-limits, the t → 0 end: V ≤ 1/√(J_d²-4β²) → 0 likewise; the prefactor is
-- positive and shrinks as the disc grows.
theorem ch10_V_pos (hb : b ≠ 0) (hJ : 2 * |b| < Jd) : 0 < ch10_V Jd b := by
  unfold ch10_V
  exact div_pos one_pos (ch10_bulk_sqrt_pos hb hJ)

end Bulk

-- eq:em-q-limits, the t → ∞ end: once the evidence term vanishes,
-- J_d = (1+ρ²)/σ_η², β = -ρ/σ_η² with σ_η² = 1-ρ², so
-- J_d² - 4β² = (1-ρ²)²/σ_η⁴ = 1, hence V → 1 and q → ρ (= |ρ| for ρ > 0).
theorem ch10_em_q_limits_prior_disc (r : ℝ) (hr : r ^ 2 < 1) :
    ((1 + r ^ 2) / (1 - r ^ 2)) ^ 2 - 4 * (-(r / (1 - r ^ 2))) ^ 2 = 1 := by
  have h : (1 : ℝ) - r ^ 2 ≠ 0 := by
    intro h0
    linarith
  field_simp
  ring

theorem ch10_em_q_limits_prior (r : ℝ) (hr0 : 0 < r) (hr : r < 1) :
    ch10_qroot ((1 + r ^ 2) / (1 - r ^ 2)) (-(r / (1 - r ^ 2))) = r
      ∧ ch10_V ((1 + r ^ 2) / (1 - r ^ 2)) (-(r / (1 - r ^ 2))) = 1 := by
  have hr2 : r ^ 2 < 1 := by nlinarith
  have hden : (0:ℝ) < 1 - r ^ 2 := by linarith
  have hdisc := ch10_em_q_limits_prior_disc r hr2
  have hsqrt :
      Real.sqrt (((1 + r ^ 2) / (1 - r ^ 2)) ^ 2 - 4 * (-(r / (1 - r ^ 2))) ^ 2) = 1 := by
    rw [hdisc, Real.sqrt_one]
  constructor
  · unfold ch10_qroot
    rw [hsqrt]
    field_simp
    ring
  · unfold ch10_V
    rw [hsqrt]
    norm_num

/-! ## eq:em-influence (lines 545-553) -/

-- eq:em-influence, the linearity step: because m = (e^{-t}/Δ_t) J⁻¹ x is linear in
-- x, the sensitivity of m_k to x_j is exactly the coefficient (e^{-t}/Δ_t)(J⁻¹)_{kj}.
theorem ch10_em_influence_linear {n : ℕ} (Jinv : Matrix (Fin n) (Fin n) ℝ) (c : ℝ)
    (x : Fin n → ℝ) (k : Fin n) :
    c * ((Jinv *ᵥ x) k) = ∑ j, (c * Jinv k j) * x j := by
  simp only [Matrix.mulVec, dotProduct, Finset.mul_sum]
  exact Finset.sum_congr rfl fun j _ => by ring

-- eq:em-influence: the geometric decay is a pure exponential in the separation,
-- q^d = e^{-d/ξ} with ξ = 1/log(1/q), for 0 < q < 1.
theorem ch10_em_influence_exp (q : ℝ) (hq0 : 0 < q) (hq1 : q < 1) (d : ℕ) :
    q ^ d = Real.exp (-(d : ℝ) / (1 / Real.log (1 / q))) := by
  have hlogq : Real.log q < 0 := Real.log_neg hq0 hq1
  have hL : Real.log (1 / q) = -Real.log q := by
    rw [one_div, Real.log_inv]
  have hrw : -(d : ℝ) * -Real.log q = (d : ℝ) * Real.log q := by ring
  rw [one_div (Real.log (1 / q)), div_inv_eq_mul, hL, hrw, Real.exp_nat_mul,
    Real.exp_log hq0]

/-! ## eq:em-discarded (lines 564-570) and eq:em-truncation (lines 575-577) -/

-- eq:em-discarded: the two geometric sums, "exactly".
theorem ch10_em_discarded_sums (q : ℝ) (hq0 : 0 ≤ q) (hq1 : q < 1) (r : ℕ) :
    (∑' d : ℕ, (q ^ 2) ^ d) = 1 / (1 - q ^ 2)
      ∧ (2 * ∑' d : ℕ, (q ^ 2) ^ (d + (r + 1))) = 2 * q ^ (2 * (r + 1)) / (1 - q ^ 2)
      ∧ (1 + 2 * ∑' d : ℕ, (q ^ 2) ^ (d + 1)) = (1 + q ^ 2) / (1 - q ^ 2) := by
  have hq2 : q ^ 2 < 1 := by nlinarith
  have hq2nn : (0:ℝ) ≤ q ^ 2 := sq_nonneg q
  have hne : (1 : ℝ) - q ^ 2 ≠ 0 := by linarith
  have hgeo : (∑' d : ℕ, (q ^ 2) ^ d) = (1 - q ^ 2)⁻¹ := tsum_geometric_of_lt_one hq2nn hq2
  refine ⟨by rw [hgeo, one_div], ?_, ?_⟩
  · have hsplit : (∑' d : ℕ, (q ^ 2) ^ (d + (r + 1)))
        = (q ^ 2) ^ (r + 1) * ∑' d : ℕ, (q ^ 2) ^ d := by
      rw [← tsum_mul_left]
      exact tsum_congr fun d => by rw [pow_add]; ring
    rw [hsplit, hgeo, ← pow_mul]
    field_simp
  · have hsplit : (∑' d : ℕ, (q ^ 2) ^ (d + 1)) = (q ^ 2) * ∑' d : ℕ, (q ^ 2) ^ d := by
      rw [← tsum_mul_left]
      exact tsum_congr fun d => by rw [pow_add]; ring
    rw [hsplit, hgeo]
    field_simp
    ring

-- eq:em-discarded: the stated ratio, 2q^{2(r+1)}/(1+q²).
theorem ch10_em_discarded (q : ℝ) (hq0 : 0 ≤ q) (hq1 : q < 1) (r : ℕ) :
    (2 * q ^ (2 * (r + 1)) / (1 - q ^ 2)) / ((1 + q ^ 2) / (1 - q ^ 2))
      = 2 * q ^ (2 * (r + 1)) / (1 + q ^ 2) := by
  have hq2 : q ^ 2 < 1 := by nlinarith
  have hne : (1 : ℝ) - q ^ 2 ≠ 0 := by linarith
  have hpos : (0 : ℝ) < 1 + q ^ 2 := by positivity
  have hne2 : (1 : ℝ) + q ^ 2 ≠ 0 := ne_of_gt hpos
  field_simp

/-- eq:em-truncation: the coefficient-tail fraction
`T(r) = Σ_t w_t · 2 q(ρ,t)^{2(r+1)}/(1+q(ρ,t)²)`.  A definition. -/
def ch10_truncation {m : ℕ} (w q : Fin m → ℝ) (r : ℕ) : ℝ :=
  ∑ t, w t * (2 * (q t) ^ (2 * (r + 1)) / (1 + (q t) ^ 2))

-- eq:em-truncation: the tail fraction falls by a factor q² per unit radius at
-- every level, which is the "sets the geometric exponent of everything local"
-- reading (and, as the text stresses, not a risk).
theorem ch10_truncation_ratio {m : ℕ} (q : Fin m → ℝ) (r : ℕ) (t : Fin m) :
    2 * (q t) ^ (2 * (r + 1 + 1)) / (1 + (q t) ^ 2)
      = (q t) ^ 2 * (2 * (q t) ^ (2 * (r + 1)) / (1 + (q t) ^ 2)) := by
  have hexp : 2 * (r + 1 + 1) = 2 * (r + 1) + 2 := by ring
  rw [hexp, pow_add]
  ring

/-! ## eq:em-oracle (lines 595-603) -/

/-- eq:em-oracle: the population relative score error of the Bayes-optimal
radius-`r` estimator `-M_r(t)x` of the Gaussian design model, pooled over the
schedule. -/
def ch10_W {m n : ℕ} (M Q Sig : Fin m → Matrix (Fin n) (Fin n) ℝ) : ℝ :=
  Real.sqrt ((∑ t, Matrix.trace ((M t - Q t) * Sig t * (M t - Q t)ᵀ))
    / (∑ t, Matrix.trace (Q t)))

-- eq:em-oracle: the numerator is the error energy E‖(M_r - Q_t)X‖² of
-- ch10_energy_trace and vanishes exactly when the windowed estimator is the exact
-- score; the denominator Σ_t tr Q_t is the pooled reference energy of
-- eq:em-weights, so the two are commensurable.
theorem ch10_em_oracle_zero {m n : ℕ} (Q Sig : Fin m → Matrix (Fin n) (Fin n) ℝ) :
    ch10_W Q Q Sig = 0 := by
  unfold ch10_W
  have hzero : (∑ t, Matrix.trace ((Q t - Q t) * Sig t * (Q t - Q t)ᵀ)) = 0 := by
    apply Finset.sum_eq_zero
    intro t _
    simp
  rw [hzero, zero_div, Real.sqrt_zero]

/-! ## eq:em-floor (lines 791-794) -/

/-- eq:em-floor: the two-parameter fit `E(N)² = E_∞² + c/N`, "a floor plus a 1/N
estimation term". -/
def ch10_floorFit (Einf c N : ℝ) : ℝ := Einf ^ 2 + c / N

theorem ch10_floorFit_antitone {Einf c N₁ N₂ : ℝ} (hc : 0 < c) (h1 : 0 < N₁) (h12 : N₁ < N₂) :
    ch10_floorFit Einf c N₂ < ch10_floorFit Einf c N₁ := by
  unfold ch10_floorFit
  have h : c / N₂ < c / N₁ := div_lt_div_of_pos_left hc h1 h12
  linarith

theorem ch10_floorFit_tendsto (Einf c : ℝ) :
    Filter.Tendsto (fun N : ℝ => ch10_floorFit Einf c N) Filter.atTop (nhds (Einf ^ 2)) := by
  have h : Filter.Tendsto (fun N : ℝ => c / N) Filter.atTop (nhds 0) :=
    Filter.Tendsto.div_atTop tendsto_const_nhds Filter.tendsto_id
  have h2 := (tendsto_const_nhds (α := ℝ) (x := Einf ^ 2) (f := Filter.atTop)).add h
  simpa [ch10_floorFit] using h2

/-! ## eq:em-global (lines 876-881) -/

/-- eq:em-global: the covariance of the shared-global-latent prior
`a = (y + βg)/√(1+β²)`, namely `Σ_β = (Σ₀ + β² 𝟏𝟏ᵀ)/(1+β²)`. -/
def ch10_Sigbeta {n : ℕ} (S0 : Matrix (Fin n) (Fin n) ℝ) (beta : ℝ) :
    Matrix (Fin n) (Fin n) ℝ :=
  (1 / (1 + beta ^ 2)) • (S0 + beta ^ 2 • Matrix.vecMulVec (ch10_ones n) (ch10_ones n))

-- eq:em-global: "variance preserving for every β" — the diagonal stays 1.
theorem ch10_em_global_variance_preserving {n : ℕ} (S0 : Matrix (Fin n) (Fin n) ℝ)
    (hdiag : ∀ i, S0 i i = 1) (beta : ℝ) (i : Fin n) :
    ch10_Sigbeta S0 beta i i = 1 := by
  have hne : (1 : ℝ) + beta ^ 2 ≠ 0 := by positivity
  unfold ch10_Sigbeta ch10_ones
  simp only [Matrix.smul_apply, Matrix.add_apply, Matrix.vecMulVec_apply, smul_eq_mul,
    hdiag i, mul_one]
  field_simp

-- eq:em-global / the proof of prop:em-rankone (lines 936-937): "the covariance of
-- eq:em-global at separation d ≥ 1 is (ρ^d + β²)/(1+β²) and the marginal variance
-- is 1".
theorem ch10_em_global_entry {n : ℕ} (r beta : ℝ) (i j : Fin n) :
    ch10_Sigbeta (ch10_arSig n r) beta i j
      = (r ^ (((i : ℤ) - (j : ℤ)).natAbs) + beta ^ 2) / (1 + beta ^ 2) := by
  unfold ch10_Sigbeta ch10_arSig ch10_ones
  simp only [Matrix.smul_apply, Matrix.add_apply, Matrix.of_apply, Matrix.vecMulVec_apply,
    smul_eq_mul, mul_one]
  ring

-- eq:em-global: "At β = 1 half the marginal variance of every site is one shared
-- constant."
theorem ch10_em_global_half : (1:ℝ) ^ 2 / (1 + (1:ℝ) ^ 2) = 1 / 2 := by norm_num

/-! ## eq:em-longrange (lines 893-899) -/

/-- eq:em-longrange: the long-range precision perturbation,
`[Q_long]_{ij} = -e^{-|i-j|/ℓ}` for `|i-j| ≥ 2`, zero at `|i-j| = 1`, and
`[Q_long]_{ii} = 1.05 Σ_{j≠i} e^{-|i-j|/ℓ}`. -/
def ch10_Qlong (n : ℕ) (l : ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  Matrix.of fun i j =>
    if i = j then
      1.05 * ∑ k ∈ Finset.univ.erase i, Real.exp (-((((i : ℤ) - (k : ℤ)).natAbs : ℝ)) / l)
    else if 2 ≤ ((i : ℤ) - (j : ℤ)).natAbs then
      -Real.exp (-((((i : ℤ) - (j : ℤ)).natAbs : ℝ)) / l)
    else 0

-- eq:em-longrange: "the diagonal compensation making Q_γ strictly diagonally
-- dominant, hence positive definite".  The absolute off-diagonal row sum is at
-- most S_i = Σ_{j≠i} e^{-|i-j|/ℓ} (the |i-j| = 1 terms are dropped from the
-- off-diagonal but kept in the compensation), while the diagonal is 1.05 S_i, so
-- the dominance margin is at least 0.05 S_i > 0.
theorem ch10_Qlong_strictly_dominant (n : ℕ) (l : ℝ) (i : Fin n)
    (hne : (Finset.univ.erase i).Nonempty) :
    (∑ j ∈ Finset.univ.erase i, |ch10_Qlong n l i j|) < ch10_Qlong n l i i := by
  set S := ∑ k ∈ Finset.univ.erase i, Real.exp (-((((i : ℤ) - (k : ℤ)).natAbs : ℝ)) / l)
    with hS
  have hSpos : 0 < S := by
    rw [hS]
    exact Finset.sum_pos (fun k _ => Real.exp_pos _) hne
  have hbound : (∑ j ∈ Finset.univ.erase i, |ch10_Qlong n l i j|) ≤ S := by
    rw [hS]
    apply Finset.sum_le_sum
    intro j hj
    have hij : ¬ (i = j) := fun hcon => (Finset.ne_of_mem_erase hj) hcon.symm
    by_cases hcase : 2 ≤ ((i : ℤ) - (j : ℤ)).natAbs
    · simp only [ch10_Qlong, Matrix.of_apply, if_neg hij, if_pos hcase, abs_neg,
        abs_of_pos (Real.exp_pos _)]
      exact le_rfl
    · simp only [ch10_Qlong, Matrix.of_apply, if_neg hij, if_neg hcase, abs_zero]
      exact le_of_lt (Real.exp_pos _)
  have hdiag : ch10_Qlong n l i i = 1.05 * S := by
    rw [hS]
    simp [ch10_Qlong]
  rw [hdiag]
  linarith

/-! ## eq:em-woodbury (prop:em-rankone, lines 913-919) -/

-- Rank-one algebra used by the Sherman-Morrison step.
theorem ch10_mul_vecMulVec {n : ℕ} (M : Matrix (Fin n) (Fin n) ℝ) (x y : Fin n → ℝ) :
    M * Matrix.vecMulVec x y = Matrix.vecMulVec (M *ᵥ x) y := by
  ext i j
  simp only [Matrix.mul_apply, Matrix.vecMulVec_apply, Matrix.mulVec, dotProduct, Finset.sum_mul]
  exact Finset.sum_congr rfl fun k _ => by ring

theorem ch10_vecMulVec_mul {n : ℕ} (M : Matrix (Fin n) (Fin n) ℝ) (x y : Fin n → ℝ) :
    Matrix.vecMulVec x y * M = Matrix.vecMulVec x (y ᵥ* M) := by
  ext i j
  simp only [Matrix.mul_apply, Matrix.vecMulVec_apply, Matrix.vecMul, dotProduct, Finset.mul_sum]
  exact Finset.sum_congr rfl fun k _ => by ring

theorem ch10_vecMulVec_mul_vecMulVec {n : ℕ} (a b x y : Fin n → ℝ) :
    Matrix.vecMulVec a b * Matrix.vecMulVec x y = (b ⬝ᵥ x) • Matrix.vecMulVec a y := by
  ext i j
  simp only [Matrix.mul_apply, Matrix.vecMulVec_apply, Matrix.smul_apply, smul_eq_mul,
    dotProduct, Finset.sum_mul]
  exact Finset.sum_congr rfl fun k _ => by ring

/-- Sherman-Morrison for the rank-one update `Σ₀ + β² 𝟏𝟏ᵀ`, the step the proof of
prop:em-rankone invokes.  `c` is `β²/(1 + β² 𝟏ᵀQ𝟏)`, supplied through the
hypothesis `hc` so the statement carries no division. -/
theorem ch10_sherman_morrison {n : ℕ} (S0 : Matrix (Fin n) (Fin n) ℝ) (hsym : S0ᵀ = S0)
    (hS : IsUnit S0.det) (beta c : ℝ)
    (hc : c * (1 + beta ^ 2 * (ch10_ones n ⬝ᵥ (S0⁻¹ *ᵥ ch10_ones n))) = beta ^ 2) :
    (S0 + beta ^ 2 • Matrix.vecMulVec (ch10_ones n) (ch10_ones n))
        * (S0⁻¹ - c • Matrix.vecMulVec (S0⁻¹ *ᵥ ch10_ones n) (S0⁻¹ *ᵥ ch10_ones n))
      = 1 := by
  have hSQ : S0 * S0⁻¹ = 1 := Matrix.mul_nonsing_inv _ hS
  have hQsym : (S0⁻¹)ᵀ = S0⁻¹ := by rw [Matrix.transpose_nonsing_inv, hsym]
  have hS0u : S0 *ᵥ (S0⁻¹ *ᵥ ch10_ones n) = ch10_ones n := by
    rw [Matrix.mulVec_mulVec, hSQ, Matrix.one_mulVec]
  have hQvecMul : ch10_ones n ᵥ* S0⁻¹ = S0⁻¹ *ᵥ ch10_ones n := by
    rw [← Matrix.vecMul_transpose, hQsym]
  have hmul1 : S0 * Matrix.vecMulVec (S0⁻¹ *ᵥ ch10_ones n) (S0⁻¹ *ᵥ ch10_ones n)
      = Matrix.vecMulVec (ch10_ones n) (S0⁻¹ *ᵥ ch10_ones n) := by
    rw [ch10_mul_vecMulVec, hS0u]
  have hmul2 : Matrix.vecMulVec (ch10_ones n) (ch10_ones n) * S0⁻¹
      = Matrix.vecMulVec (ch10_ones n) (S0⁻¹ *ᵥ ch10_ones n) := by
    rw [ch10_vecMulVec_mul, hQvecMul]
  have hmul3 : Matrix.vecMulVec (ch10_ones n) (ch10_ones n)
        * Matrix.vecMulVec (S0⁻¹ *ᵥ ch10_ones n) (S0⁻¹ *ᵥ ch10_ones n)
      = (ch10_ones n ⬝ᵥ (S0⁻¹ *ᵥ ch10_ones n))
        • Matrix.vecMulVec (ch10_ones n) (S0⁻¹ *ᵥ ch10_ones n) := by
    rw [ch10_vecMulVec_mul_vecMulVec]
  have hzero : beta ^ 2 - c - beta ^ 2 * c * (ch10_ones n ⬝ᵥ (S0⁻¹ *ᵥ ch10_ones n)) = 0 := by
    linear_combination -hc
  have key : (S0 + beta ^ 2 • Matrix.vecMulVec (ch10_ones n) (ch10_ones n))
        * (S0⁻¹ - c • Matrix.vecMulVec (S0⁻¹ *ᵥ ch10_ones n) (S0⁻¹ *ᵥ ch10_ones n))
      = 1 + (beta ^ 2 - c - beta ^ 2 * c * (ch10_ones n ⬝ᵥ (S0⁻¹ *ᵥ ch10_ones n)))
          • Matrix.vecMulVec (ch10_ones n) (S0⁻¹ *ᵥ ch10_ones n) := by
    simp only [Matrix.add_mul, Matrix.mul_sub, Matrix.smul_mul, Matrix.mul_smul, smul_smul,
      hSQ, hmul1, hmul2, hmul3]
    match_scalars <;> ring
  rw [key, hzero, zero_smul, add_zero]

-- eq:em-woodbury: Q_β = (1+β²) Q_AR - c_β u uᵀ with u = Q_AR 𝟏 and
-- c_β = (1+β²)β²/(1 + β² 𝟏ᵀ Q_AR 𝟏), "so the departure from a rescaled chain
-- precision has rank exactly one, in the single direction u".  Verified by
-- checking that this matrix really is Σ_β⁻¹.
theorem ch10_em_woodbury {n : ℕ} (S0 : Matrix (Fin n) (Fin n) ℝ) (hsym : S0ᵀ = S0)
    (hS : IsUnit S0.det) (beta c : ℝ)
    (hc : c * (1 + beta ^ 2 * (ch10_ones n ⬝ᵥ (S0⁻¹ *ᵥ ch10_ones n))) = beta ^ 2) :
    ch10_Sigbeta S0 beta
        * ((1 + beta ^ 2) • S0⁻¹
          - ((1 + beta ^ 2) * c)
            • Matrix.vecMulVec (S0⁻¹ *ᵥ ch10_ones n) (S0⁻¹ *ᵥ ch10_ones n))
      = 1 := by
  have core := ch10_sherman_morrison S0 hsym hS beta c hc
  have hbne : (1 : ℝ) + beta ^ 2 ≠ 0 := by positivity
  have hsplit : ((1 + beta ^ 2) • S0⁻¹
        - ((1 + beta ^ 2) * c)
          • Matrix.vecMulVec (S0⁻¹ *ᵥ ch10_ones n) (S0⁻¹ *ᵥ ch10_ones n))
      = (1 + beta ^ 2)
        • (S0⁻¹ - c • Matrix.vecMulVec (S0⁻¹ *ᵥ ch10_ones n) (S0⁻¹ *ᵥ ch10_ones n)) := by
    rw [smul_sub, smul_smul]
  unfold ch10_Sigbeta
  rw [hsplit, Matrix.smul_mul, Matrix.mul_smul, core, smul_smul, one_div,
    inv_mul_cancel₀ hbne, one_smul]

/-! ## eq:em-chowliu (lines 924-928) -/

-- eq:em-chowliu: the lag-one correlation of the global-latent prior,
-- ρ_CL(β) = (ρ + β²)/(1+β²), read off eq:em-global's covariance at separation one
-- (the marginal variance being 1).
theorem ch10_em_chowliu {n : ℕ} (r beta : ℝ) (i j : Fin n)
    (hadj : ((i : ℤ) - (j : ℤ)).natAbs = 1) :
    ch10_Sigbeta (ch10_arSig n r) beta i j = (r + beta ^ 2) / (1 + beta ^ 2) := by
  rw [ch10_em_global_entry, hadj, pow_one]

-- eq:em-chowliu, the quoted value 0.9250 at ρ = 0.85, β = 1.
theorem ch10_em_chowliu_numeric : ((17:ℝ) / 20 + 1 ^ 2) / (1 + 1 ^ 2) = 0.925 := by norm_num

/-! ## eq:em-delta (lines 973-979) -/

/-- eq:em-delta: the per-site KL divergence between covariance-matched centred
Gaussians, `δ = (1/2n)[tr(Σ_CL⁻¹ Σ) - n + log(det Σ_CL/det Σ)]`. -/
def ch10_delta {n : ℕ} (S Scl : Matrix (Fin n) (Fin n) ℝ) : ℝ :=
  (1 / (2 * (n : ℝ))) *
    (Matrix.trace (Scl⁻¹ * S) - (n : ℝ) + Real.log (Scl.det / S.det))

-- eq:em-delta: δ = 0 when the law already is its own best chain (the "none" row of
-- tab:em-misspecification, δ = 0).
theorem ch10_delta_self {n : ℕ} (S : Matrix (Fin n) (Fin n) ℝ) (hS : IsUnit S.det) :
    ch10_delta S S = 0 := by
  unfold ch10_delta
  rw [Matrix.nonsing_inv_mul _ hS, Matrix.trace_one, div_self (IsUnit.ne_zero hS), Real.log_one]
  simp

/-! ## eq:em-breakeven (lines 1078-1083) -/

/-- Linear interpolation through two points: the "straight line through those
endpoints" of ch10-comparison.tex line 1085. -/
def ch10_lin (x0 y0 x1 y1 x : ℝ) : ℝ := y0 + (x - x0) * ((y1 - y0) / (x1 - x0))

-- eq:em-breakeven: the reported brackets agree with tab:em-misspecification — the
-- CNN ratio crosses 1 between δ = 1.1e-2 (ratio 1.2) and δ = 2.2e-2 (ratio 0.97),
-- the MLP ratio between δ = 2.2e-2 (ratio 1.2) and δ = 4.0e-2 (ratio 0.84).
theorem ch10_em_breakeven_brackets :
    (0.97 : ℝ) < 1 ∧ (1 : ℝ) < 1.2 ∧ (0.84 : ℝ) < 1 := by
  refine ⟨by norm_num, by norm_num, by norm_num⟩

-- eq:em-breakeven, the CNN interpolation: the straight line through (1.1, 1.2) and
-- (2.2, 0.97) crosses 1 at δ = 1.1 + 22/23 = 2.0565…, which the text rounds to 2.1.
theorem ch10_em_breakeven_cnn_crossing :
    ch10_lin 1.1 1.2 2.2 0.97 (1.1 + 22 / 23) = 1 := by
  unfold ch10_lin; norm_num

theorem ch10_em_breakeven_cnn_rounds : |(1.1 + 22 / 23 : ℝ) - 2.1| < 0.05 := by
  rw [abs_lt]
  constructor <;> norm_num

-- eq:em-breakeven, the MLP interpolation: the straight line through (2.2, 1.2) and
-- (4.0, 0.84) crosses 1 at δ = 3.2 exactly, NOT at the 3.0 stated in the text; at
-- δ = 3.0 the interpolant is still 1.04, i.e. above break-even.
theorem ch10_em_breakeven_mlp_crossing :
    ch10_lin 2.2 1.2 4.0 0.84 3.2 = 1 := by
  unfold ch10_lin; norm_num

theorem ch10_em_breakeven_mlp_discrepancy :
    ch10_lin 2.2 1.2 4.0 0.84 3.0 = 1.04 ∧ ch10_lin 2.2 1.2 4.0 0.84 3.0 ≠ 1 := by
  constructor
  · unfold ch10_lin; norm_num
  · unfold ch10_lin; norm_num

/-! ## eq:em-cumulants (lines 1123-1130) -/

-- eq:em-cumulants: κ₂(X_t) = e^{-2t}κ₂(A) + Δ_t = 1, κ₄(X_t) = e^{-4t}κ₄(A) and
-- κ₁₁(X_{t,i},X_{t,i+1}) = e^{-2t}ρ, for X_t = e^{-t}A + √Δ_t ε with ε standard
-- Gaussian independent of A.  The cumulant calculus (additivity over independent
-- summands, order-n homogeneity, Gaussian κ₄ = 0, independence of the noise across
-- sites) enters through the hypotheses h2, h4, h11.
theorem ch10_em_cumulants (t k2A k4A k2X k4X k11A k11X rho : ℝ)
    (h2A : k2A = 1) (h11A : k11A = rho)
    (h2 : k2X = Real.exp (-t) ^ 2 * k2A + ch10_Delta t * 1)
    (h4 : k4X = Real.exp (-t) ^ 4 * k4A + ch10_Delta t ^ 2 * 0)
    (h11 : k11X = Real.exp (-t) ^ 2 * k11A + ch10_Delta t * 0) :
    k2X = 1 ∧ k4X = Real.exp (-(4 * t)) * k4A ∧ k11X = Real.exp (-(2 * t)) * rho := by
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
  refine ⟨?_, ?_, ?_⟩
  · rw [h2, h2A, e2]
    unfold ch10_Delta
    ring
  · rw [h4, e4]; ring
  · rw [h11, h11A, e2]; ring

/-! ## eq:em-info-prediction (lines 1148-1157) and the heuristic display (1139-1142) -/

/-- The unlabelled display of ch10-comparison.tex lines 1139-1142: the delta-method
reading of information as squared sensitivity over sampling variance,
`I_θ ≍ (∂_θ κ(X_t))²/Var κ̂`. -/
def ch10_infoHeuristic (dkappa varkappa : ℝ) : ℝ := dkappa ^ 2 / varkappa

-- eq:em-info-prediction: with ∂_ρ κ₁₁ ∝ e^{-2t}, ∂_p κ₄ ∝ e^{-4t} and
-- t-independent denominators, I_ρ ≍ e^{-4t}, I_p ≍ e^{-8t}, and the contrast
-- I_p/I_ρ ≍ e^{-4t} — "the robust one: whatever common factors the leading-order
-- reading omits cancel from it".
theorem ch10_em_info_prediction (t c1 c2 v1 v2 : ℝ) (hv1 : v1 ≠ 0) (hv2 : v2 ≠ 0)
    (hc1 : c1 ≠ 0) :
    ch10_infoHeuristic (c1 * Real.exp (-(2 * t))) v1 = (c1 ^ 2 / v1) * Real.exp (-(4 * t))
      ∧ ch10_infoHeuristic (c2 * Real.exp (-(4 * t))) v2
          = (c2 ^ 2 / v2) * Real.exp (-(8 * t))
      ∧ ch10_infoHeuristic (c2 * Real.exp (-(4 * t))) v2
          / ch10_infoHeuristic (c1 * Real.exp (-(2 * t))) v1
        = ((c2 ^ 2 / v2) / (c1 ^ 2 / v1)) * Real.exp (-(4 * t)) := by
  have hexp2 : Real.exp (-(2 * t)) ^ 2 = Real.exp (-(4 * t)) := by
    rw [← Real.exp_nat_mul]; congr 1; push_cast; ring
  have hexp4 : Real.exp (-(4 * t)) ^ 2 = Real.exp (-(8 * t)) := by
    rw [← Real.exp_nat_mul]; congr 1; push_cast; ring
  have h8 : Real.exp (-(8 * t)) = Real.exp (-(4 * t)) * Real.exp (-(4 * t)) := by
    rw [← Real.exp_add]; congr 1; ring
  have p1 : ch10_infoHeuristic (c1 * Real.exp (-(2 * t))) v1
      = (c1 ^ 2 / v1) * Real.exp (-(4 * t)) := by
    unfold ch10_infoHeuristic
    rw [mul_pow, hexp2]
    ring
  have p2 : ch10_infoHeuristic (c2 * Real.exp (-(4 * t))) v2
      = (c2 ^ 2 / v2) * Real.exp (-(8 * t)) := by
    unfold ch10_infoHeuristic
    rw [mul_pow, hexp4]
    ring
  refine ⟨p1, p2, ?_⟩
  have hE : Real.exp (-(4 * t)) ≠ 0 := ne_of_gt (Real.exp_pos _)
  have hcv : c1 ^ 2 / v1 ≠ 0 := div_ne_zero (pow_ne_zero 2 hc1) hv1
  rw [p1, p2, h8]
  field_simp

-- eq:em-info-prediction / tab:em-information: the predicted contrast between
-- t = 0.05 and t = 1.6 is e^{4·1.55} = e^{6.2} (the exponent arithmetic; the value
-- 493 quoted in the caption is the decimal evaluation e^{6.2} = 492.75).
theorem ch10_em_info_contrast_exponent : 4 * (1.6 - 0.05 : ℝ) = 6.2 := by norm_num

-- tab:em-information: "its geometric mean over the five shapes is 465".  The five
-- measured contrasts are 112, 236, 776, 867, 1222; their product lies strictly
-- between 464.5^5 and 465^5, so the geometric mean is in (464.5, 465) and rounds
-- to 465 as quoted.
theorem ch10_em_info_geomean :
    (464.5 : ℝ) ^ 5 < 112 * 236 * 776 * 867 * 1222
      ∧ (112 * 236 * 776 * 867 * 1222 : ℝ) < 465 ^ 5 := by
  constructor <;> norm_num

/-! ## Strengthening pass: positive definiteness for eq:em-longrange -/

-- The implication the text invokes for eq:em-longrange: a symmetric matrix that is
-- strictly diagonally dominant (each row's absolute off-diagonal sum below its
-- diagonal entry) is positive definite.  Proved in general over ℝ.
theorem ch10_posdef_of_strictly_dominant {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ)
    (hsym : ∀ i j, A i j = A j i)
    (hdd : ∀ i, ∑ j ∈ Finset.univ.erase i, |A i j| < A i i)
    (v : Fin n → ℝ) (hv : v ≠ 0) :
    0 < v ⬝ᵥ (A *ᵥ v) := by
  classical
  -- the quadratic form, written with the diagonal split off
  have hrow : ∀ i, (∑ j, v i * (A i j * v j))
      = A i i * v i ^ 2 + ∑ j ∈ Finset.univ.erase i, A i j * v i * v j := by
    intro i
    rw [← Finset.add_sum_erase _ (fun j => v i * (A i j * v j)) (Finset.mem_univ i)]
    congr 1
    · ring
    · exact Finset.sum_congr rfl fun j _ => by ring
  have hexp : v ⬝ᵥ (A *ᵥ v)
      = ∑ i, (A i i * v i ^ 2 + ∑ j ∈ Finset.univ.erase i, A i j * v i * v j) := by
    simp only [dotProduct, Matrix.mulVec, Finset.mul_sum]
    exact Finset.sum_congr rfl fun i _ => hrow i
  -- termwise AM-GM bound on the off-diagonal contribution
  have hterm : ∀ i j, -((1 / 2) * |A i j| * v i ^ 2 + (1 / 2) * |A i j| * v j ^ 2)
      ≤ A i j * v i * v j := by
    intro i j
    have habs : |A i j * v i * v j| = |A i j| * (|v i| * |v j|) := by
      rw [abs_mul, abs_mul, mul_assoc]
    have hamgm : |v i| * |v j| ≤ (v i ^ 2 + v j ^ 2) / 2 := by
      nlinarith [sq_nonneg (|v i| - |v j|), sq_abs (v i), sq_abs (v j)]
    have h1 : |A i j * v i * v j| ≤ |A i j| * ((v i ^ 2 + v j ^ 2) / 2) := by
      rw [habs]
      exact mul_le_mul_of_nonneg_left hamgm (abs_nonneg _)
    have h2 := neg_abs_le (A i j * v i * v j)
    linarith
  have hrowbound : ∀ i,
      -(∑ j ∈ Finset.univ.erase i, ((1 / 2) * |A i j| * v i ^ 2 + (1 / 2) * |A i j| * v j ^ 2))
        ≤ ∑ j ∈ Finset.univ.erase i, A i j * v i * v j := by
    intro i
    rw [← Finset.sum_neg_distrib]
    exact Finset.sum_le_sum fun j _ => hterm i j
  -- the two halves of the bound are equal, by symmetry of |A|
  have hfull : (∑ i, ∑ j, (1 / 2) * |A i j| * v j ^ 2)
      = ∑ i, ∑ j, (1 / 2) * |A i j| * v i ^ 2 := by
    rw [Finset.sum_comm]
    exact Finset.sum_congr rfl fun i _ =>
      Finset.sum_congr rfl fun j _ => by rw [hsym j i]
  have herase : ∀ (g : Fin n → Fin n → ℝ) (i : Fin n),
      (∑ j ∈ Finset.univ.erase i, g i j) = (∑ j, g i j) - g i i := fun g i =>
    Finset.sum_erase_eq_sub (Finset.mem_univ i)
  have hswap :
      (∑ i, ∑ j ∈ Finset.univ.erase i, (1 / 2) * |A i j| * v j ^ 2)
        = ∑ i, ∑ j ∈ Finset.univ.erase i, (1 / 2) * |A i j| * v i ^ 2 := by
    have h1 : (∑ i, ∑ j ∈ Finset.univ.erase i, (1 / 2) * |A i j| * v j ^ 2)
        = (∑ i, ∑ j, (1 / 2) * |A i j| * v j ^ 2)
          - ∑ i, (1 / 2) * |A i i| * v i ^ 2 := by
      rw [← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun i _ => herase (fun i j => (1 / 2) * |A i j| * v j ^ 2) i
    have h2 : (∑ i, ∑ j ∈ Finset.univ.erase i, (1 / 2) * |A i j| * v i ^ 2)
        = (∑ i, ∑ j, (1 / 2) * |A i j| * v i ^ 2)
          - ∑ i, (1 / 2) * |A i i| * v i ^ 2 := by
      rw [← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun i _ => herase (fun i j => (1 / 2) * |A i j| * v i ^ 2) i
    rw [h1, h2, hfull]
  -- assemble: the form dominates Σ_i (A_ii - S_i) v_i²
  have hsplit : ∀ i,
      (∑ j ∈ Finset.univ.erase i, ((1 / 2) * |A i j| * v i ^ 2 + (1 / 2) * |A i j| * v j ^ 2))
        = (∑ j ∈ Finset.univ.erase i, (1 / 2) * |A i j| * v i ^ 2)
          + ∑ j ∈ Finset.univ.erase i, (1 / 2) * |A i j| * v j ^ 2 := fun i =>
    Finset.sum_add_distrib
  have hdiag : ∀ i, (∑ j ∈ Finset.univ.erase i, (1 / 2) * |A i j| * v i ^ 2)
      = (1 / 2) * (∑ j ∈ Finset.univ.erase i, |A i j|) * v i ^ 2 := by
    intro i
    rw [Finset.mul_sum, Finset.sum_mul]
  -- the total off-diagonal penalty, summed over rows, is exactly Σ_i S_i v_i²
  have hpen : (∑ i, ∑ j ∈ Finset.univ.erase i,
        ((1 / 2) * |A i j| * v i ^ 2 + (1 / 2) * |A i j| * v j ^ 2))
      = ∑ i, (∑ j ∈ Finset.univ.erase i, |A i j|) * v i ^ 2 := by
    have e1 : (∑ i, ∑ j ∈ Finset.univ.erase i,
          ((1 / 2) * |A i j| * v i ^ 2 + (1 / 2) * |A i j| * v j ^ 2))
        = (∑ i, ∑ j ∈ Finset.univ.erase i, (1 / 2) * |A i j| * v i ^ 2)
          + ∑ i, ∑ j ∈ Finset.univ.erase i, (1 / 2) * |A i j| * v j ^ 2 := by
      rw [← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun i _ => hsplit i
    rw [e1, hswap, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun i _ => by rw [hdiag i]; ring
  have hlower : ∑ i, (A i i - ∑ j ∈ Finset.univ.erase i, |A i j|) * v i ^ 2
      ≤ v ⬝ᵥ (A *ᵥ v) := by
    have l1 : (∑ i, (A i i - ∑ j ∈ Finset.univ.erase i, |A i j|) * v i ^ 2)
        = (∑ i, A i i * v i ^ 2)
          - ∑ i, (∑ j ∈ Finset.univ.erase i, |A i j|) * v i ^ 2 := by
      rw [← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun i _ => by ring
    have l2 : (∑ i, (A i i * v i ^ 2
          - ∑ j ∈ Finset.univ.erase i,
              ((1 / 2) * |A i j| * v i ^ 2 + (1 / 2) * |A i j| * v j ^ 2)))
        = (∑ i, A i i * v i ^ 2)
          - ∑ i, ∑ j ∈ Finset.univ.erase i,
              ((1 / 2) * |A i j| * v i ^ 2 + (1 / 2) * |A i j| * v j ^ 2) :=
      Finset.sum_sub_distrib _ _
    have l3 : (∑ i, (A i i - ∑ j ∈ Finset.univ.erase i, |A i j|) * v i ^ 2)
        = ∑ i, (A i i * v i ^ 2
          - ∑ j ∈ Finset.univ.erase i,
              ((1 / 2) * |A i j| * v i ^ 2 + (1 / 2) * |A i j| * v j ^ 2)) := by
      rw [l1, l2, hpen]
    rw [l3, hexp]
    refine Finset.sum_le_sum fun i _ => ?_
    linarith [hrowbound i]
  -- v ≠ 0 supplies a strictly positive term; strict dominance makes all terms ≥ 0
  obtain ⟨k, hk⟩ : ∃ k, v k ≠ 0 := by
    by_contra h
    push_neg at h
    exact hv (funext fun i => h i)
  have hvk : 0 < v k ^ 2 := lt_of_le_of_ne (sq_nonneg _) (Ne.symm (pow_ne_zero 2 hk))
  have hpos : 0 < ∑ i, (A i i - ∑ j ∈ Finset.univ.erase i, |A i j|) * v i ^ 2 :=
    Finset.sum_pos' (fun i _ => mul_nonneg (sub_pos.mpr (hdd i)).le (sq_nonneg _))
      ⟨k, Finset.mem_univ k, mul_pos (sub_pos.mpr (hdd k)) hvk⟩
  exact lt_of_lt_of_le hpos hlower

/-! ### eq:em-weights-smallt: `tr Q₀` and `tr Q₀²` at every chain length

The two closed forms `ch10_trQ0` and `ch10_trQ0sq`, from which the quoted numbers
`tr Q₀ = 193.4` and the linear coefficient `-3140` are computed, are *derived* here
from the general-`n` inverse `ch10_arSig_inv` — for every chain length `n ≥ 2` and
every `r` with `1 - r² ≠ 0` — rather than typed in and checked at small `n`.  The
`n = 32`, `ρ = 0.85` numbers are then instances of a theorem about the matrix. -/

/-- A sum over `Fin n` of a term supported on exactly two indices. -/
theorem ch10_sum_pick2 {n : ℕ} (P : ℕ → Prop) [DecidablePred P] (g : Fin n → ℝ)
    (k₁ k₂ : Fin n) (h₁ : P (k₁ : ℕ)) (h₂ : P (k₂ : ℕ)) (hne : k₁ ≠ k₂)
    (huniq : ∀ k : Fin n, P (k : ℕ) → k = k₁ ∨ k = k₂) :
    ∑ k : Fin n, (if P (k : ℕ) then g k else 0) = g k₁ + g k₂ := by
  have hpt : ∀ k : Fin n, (if P (k : ℕ) then g k else 0)
      = (if k = k₁ then g k₁ else 0) + (if k = k₂ then g k₂ else 0) := by
    intro k
    by_cases hk : P (k : ℕ)
    · rcases huniq k hk with rfl | rfl
      · simp [hk, hne, hne.symm]
      · simp [hk, hne, hne.symm]
    · have hk1 : k ≠ k₁ := by rintro rfl; exact hk h₁
      have hk2 : k ≠ k₂ := by rintro rfl; exact hk h₂
      simp [hk, hk1, hk2]
  rw [Finset.sum_congr rfl fun k _ => hpt k, Finset.sum_add_distrib]
  simp

/-- **eq:em-weights-smallt, `tr Q₀` at every chain length.**  The trace of the AR(1)
precision matrix on `n ≥ 2` sites: `n - 2` interior rows contribute `(1+r²)/(1-r²)`
and the two ends `1/(1-r²)`. -/
theorem ch10_arQ_trace {n : ℕ} (hn : 2 ≤ n) (r : ℝ) (h : 1 - r ^ 2 ≠ 0) :
    Matrix.trace (ch10_arQ n r) = ch10_trQ0 n r := by
  obtain ⟨i0, hi0⟩ : ∃ k : Fin n, (k : ℕ) = 0 := ⟨⟨0, by omega⟩, rfl⟩
  obtain ⟨i1, hi1⟩ : ∃ k : Fin n, (k : ℕ) = n - 1 := ⟨⟨n - 1, by omega⟩, rfl⟩
  have hne : i0 ≠ i1 := by
    intro hEq
    rw [hEq] at hi0
    omega
  have huniq : ∀ k : Fin n, ((k : ℕ) = 0 ∨ (k : ℕ) + 1 = n) → k = i0 ∨ k = i1 := by
    intro k hk
    have := k.isLt
    rcases hk with hk | hk
    · exact Or.inl (Fin.val_injective (by omega))
    · exact Or.inr (Fin.val_injective (by omega))
  have hsplit : ∀ i : Fin n, ch10_arQ n r i i
      = (1 + r ^ 2) / (1 - r ^ 2)
        + (if (i : ℕ) = 0 ∨ (i : ℕ) + 1 = n then (-(r ^ 2)) / (1 - r ^ 2) else 0) := by
    intro i
    rw [ch10_arQ_apply]
    split_ifs <;> first | ring1 | (exfalso; omega)
  have hbdry : (∑ i : Fin n, (if (i : ℕ) = 0 ∨ (i : ℕ) + 1 = n
        then (-(r ^ 2)) / (1 - r ^ 2) else 0))
      = (-(r ^ 2)) / (1 - r ^ 2) + (-(r ^ 2)) / (1 - r ^ 2) :=
    ch10_sum_pick2 (fun m => m = 0 ∨ m + 1 = n) (fun _ => (-(r ^ 2)) / (1 - r ^ 2))
      i0 i1 (Or.inl hi0) (Or.inr (by omega)) hne huniq
  have htr : Matrix.trace (ch10_arQ n r) = ∑ i : Fin n, ch10_arQ n r i i := rfl
  rw [htr, Finset.sum_congr rfl fun i _ => hsplit i, Finset.sum_add_distrib, hbdry,
    Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  unfold ch10_trQ0
  field_simp
  ring

/-- The diagonal of `Q₀²`: because `Q₀` is symmetric and tridiagonal, `(Q₀²)ᵢᵢ` is the
sum of the squares of the entries of row `i` — the diagonal entry plus one
off-diagonal entry at an end of the chain, two in the interior. -/
theorem ch10_arQ_rowsq {n : ℕ} (hn : 2 ≤ n) (r : ℝ) (i : Fin n) :
    ∑ k : Fin n, ch10_arQ n r i k * ch10_arQ n r k i
      = ((if (i : ℕ) = 0 ∨ (i : ℕ) + 1 = n then 1 else 1 + r ^ 2) / (1 - r ^ 2)) ^ 2
        + (if (i : ℕ) = 0 ∨ (i : ℕ) + 1 = n then 1 else 2) * (-r / (1 - r ^ 2)) ^ 2 := by
  have hsplit : ∀ k : Fin n, ch10_arQ n r i k * ch10_arQ n r k i
      = (if (i : ℕ) = (k : ℕ)
          then ((if (i : ℕ) = 0 ∨ (i : ℕ) + 1 = n then 1 else 1 + r ^ 2) / (1 - r ^ 2)) ^ 2
          else 0)
        + (if (k : ℕ) + 1 = (i : ℕ) then (-r / (1 - r ^ 2)) ^ 2 else 0)
        + (if (i : ℕ) + 1 = (k : ℕ) then (-r / (1 - r ^ 2)) ^ 2 else 0) := by
    intro k
    rw [ch10_arQ_apply, ch10_arQ_apply]
    split_ifs <;> first | ring1 | (exfalso; omega)
  have hdiag : (∑ k : Fin n, (if (i : ℕ) = (k : ℕ)
      then ((if (i : ℕ) = 0 ∨ (i : ℕ) + 1 = n then 1 else 1 + r ^ 2) / (1 - r ^ 2)) ^ 2
      else 0))
      = ((if (i : ℕ) = 0 ∨ (i : ℕ) + 1 = n then 1 else 1 + r ^ 2) / (1 - r ^ 2)) ^ 2 :=
    ch10_sum_pick (fun m => (i : ℕ) = m)
      (fun _ => ((if (i : ℕ) = 0 ∨ (i : ℕ) + 1 = n then 1 else 1 + r ^ 2) / (1 - r ^ 2)) ^ 2)
      i rfl fun k hk => Fin.val_injective hk.symm
  rw [Finset.sum_congr rfl fun k _ => hsplit k, Finset.sum_add_distrib, Finset.sum_add_distrib,
    hdiag]
  rcases Nat.eq_zero_or_pos (i : ℕ) with hi0 | hipos
  · -- first site: right neighbour only
    obtain ⟨k1, hk1⟩ : ∃ k : Fin n, (k : ℕ) = 1 := ⟨⟨1, by omega⟩, rfl⟩
    have hL : (∑ k : Fin n, (if (k : ℕ) + 1 = (i : ℕ) then (-r / (1 - r ^ 2)) ^ 2 else 0)) = 0 :=
      ch10_sum_none (fun m => m + 1 = (i : ℕ)) _ fun k => by omega
    have hR : (∑ k : Fin n, (if (i : ℕ) + 1 = (k : ℕ) then (-r / (1 - r ^ 2)) ^ 2 else 0))
        = (-r / (1 - r ^ 2)) ^ 2 :=
      ch10_sum_pick (fun m => (i : ℕ) + 1 = m) _ k1 (by omega)
        fun k hk => Fin.val_injective (by omega)
    rw [hL, hR]
    split_ifs <;> first | ring1 | (exfalso; omega)
  · obtain ⟨km, hkm⟩ : ∃ k : Fin n, (k : ℕ) = (i : ℕ) - 1 :=
      ⟨⟨(i : ℕ) - 1, by have := i.isLt; omega⟩, rfl⟩
    have hL : (∑ k : Fin n, (if (k : ℕ) + 1 = (i : ℕ) then (-r / (1 - r ^ 2)) ^ 2 else 0))
        = (-r / (1 - r ^ 2)) ^ 2 :=
      ch10_sum_pick (fun m => m + 1 = (i : ℕ)) _ km (by omega)
        fun k hk => Fin.val_injective (by omega)
    rw [hL]
    rcases Nat.lt_or_ge ((i : ℕ) + 1) n with hint | hlast
    · -- interior site: both neighbours
      obtain ⟨kp, hkp⟩ : ∃ k : Fin n, (k : ℕ) = (i : ℕ) + 1 := ⟨⟨(i : ℕ) + 1, hint⟩, rfl⟩
      have hR : (∑ k : Fin n, (if (i : ℕ) + 1 = (k : ℕ) then (-r / (1 - r ^ 2)) ^ 2 else 0))
          = (-r / (1 - r ^ 2)) ^ 2 :=
        ch10_sum_pick (fun m => (i : ℕ) + 1 = m) _ kp (by omega)
          fun k hk => Fin.val_injective (by omega)
      rw [hR]
      split_ifs <;> first | ring1 | (exfalso; omega)
    · -- last site: left neighbour only
      have hlastEq : (i : ℕ) + 1 = n := by have := i.isLt; omega
      have hR : (∑ k : Fin n, (if (i : ℕ) + 1 = (k : ℕ) then (-r / (1 - r ^ 2)) ^ 2 else 0))
          = 0 :=
        ch10_sum_none (fun m => (i : ℕ) + 1 = m) _ fun k => by have := k.isLt; omega
      rw [hR]
      split_ifs <;> first | ring1 | (exfalso; omega)

/-- **eq:em-weights-smallt, `tr Q₀²` at every chain length.** -/
theorem ch10_arQ_trace_sq {n : ℕ} (hn : 2 ≤ n) (r : ℝ) (h : 1 - r ^ 2 ≠ 0) :
    Matrix.trace (ch10_arQ n r * ch10_arQ n r) = ch10_trQ0sq n r := by
  obtain ⟨i0, hi0⟩ : ∃ k : Fin n, (k : ℕ) = 0 := ⟨⟨0, by omega⟩, rfl⟩
  obtain ⟨i1, hi1⟩ : ∃ k : Fin n, (k : ℕ) = n - 1 := ⟨⟨n - 1, by omega⟩, rfl⟩
  have hne : i0 ≠ i1 := by
    intro hEq
    rw [hEq] at hi0
    omega
  have huniq : ∀ k : Fin n, ((k : ℕ) = 0 ∨ (k : ℕ) + 1 = n) → k = i0 ∨ k = i1 := by
    intro k hk
    have := k.isLt
    rcases hk with hk | hk
    · exact Or.inl (Fin.val_injective (by omega))
    · exact Or.inr (Fin.val_injective (by omega))
  have htr : Matrix.trace (ch10_arQ n r * ch10_arQ n r)
      = ∑ i : Fin n, ∑ k : Fin n, ch10_arQ n r i k * ch10_arQ n r k i := by
    simp only [Matrix.trace, Matrix.diag_apply, Matrix.mul_apply]
  have hcorr : ∀ i : Fin n,
      ((if (i : ℕ) = 0 ∨ (i : ℕ) + 1 = n then 1 else 1 + r ^ 2) / (1 - r ^ 2)) ^ 2
        + (if (i : ℕ) = 0 ∨ (i : ℕ) + 1 = n then 1 else 2) * (-r / (1 - r ^ 2)) ^ 2
      = (((1 + r ^ 2) / (1 - r ^ 2)) ^ 2 + 2 * (-r / (1 - r ^ 2)) ^ 2)
        + (if (i : ℕ) = 0 ∨ (i : ℕ) + 1 = n
            then (1 / (1 - r ^ 2)) ^ 2 + (-r / (1 - r ^ 2)) ^ 2
                  - ((1 + r ^ 2) / (1 - r ^ 2)) ^ 2 - 2 * (-r / (1 - r ^ 2)) ^ 2
            else 0) := by
    intro i
    split_ifs <;> ring1
  have hbdry : (∑ i : Fin n, (if (i : ℕ) = 0 ∨ (i : ℕ) + 1 = n
        then (1 / (1 - r ^ 2)) ^ 2 + (-r / (1 - r ^ 2)) ^ 2
              - ((1 + r ^ 2) / (1 - r ^ 2)) ^ 2 - 2 * (-r / (1 - r ^ 2)) ^ 2
        else 0))
      = ((1 / (1 - r ^ 2)) ^ 2 + (-r / (1 - r ^ 2)) ^ 2
            - ((1 + r ^ 2) / (1 - r ^ 2)) ^ 2 - 2 * (-r / (1 - r ^ 2)) ^ 2)
        + ((1 / (1 - r ^ 2)) ^ 2 + (-r / (1 - r ^ 2)) ^ 2
            - ((1 + r ^ 2) / (1 - r ^ 2)) ^ 2 - 2 * (-r / (1 - r ^ 2)) ^ 2) :=
    ch10_sum_pick2 (fun m => m = 0 ∨ m + 1 = n)
      (fun _ => (1 / (1 - r ^ 2)) ^ 2 + (-r / (1 - r ^ 2)) ^ 2
            - ((1 + r ^ 2) / (1 - r ^ 2)) ^ 2 - 2 * (-r / (1 - r ^ 2)) ^ 2)
      i0 i1 (Or.inl hi0) (Or.inr (by omega)) hne huniq
  rw [htr, Finset.sum_congr rfl fun i _ => ch10_arQ_rowsq hn r i,
    Finset.sum_congr rfl fun i _ => hcorr i, Finset.sum_add_distrib, hbdry,
    Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  unfold ch10_trQ0sq
  field_simp
  ring

/-- `tr Σ₀⁻¹ = tr Q₀` in closed form, every `n ≥ 2`. -/
theorem ch10_em_trQ0_eq {n : ℕ} (hn : 2 ≤ n) (r : ℝ) (h : 1 - r ^ 2 ≠ 0) :
    Matrix.trace (ch10_arSig n r)⁻¹ = ch10_trQ0 n r := by
  rw [ch10_arSig_inv hn r h]
  exact ch10_arQ_trace hn r h

/-- `tr (Σ₀⁻¹)² = tr Q₀²` in closed form, every `n ≥ 2`. -/
theorem ch10_em_trQ0sq_eq {n : ℕ} (hn : 2 ≤ n) (r : ℝ) (h : 1 - r ^ 2 ≠ 0) :
    Matrix.trace ((ch10_arSig n r)⁻¹ * (ch10_arSig n r)⁻¹) = ch10_trQ0sq n r := by
  rw [ch10_arSig_inv hn r h]
  exact ch10_arQ_trace_sq hn r h

/-- **eq:em-weights-smallt on the AR(1) chain, every `n ≥ 2`.**  The first-order
truncation of `tr Σ_t⁻¹` at `ε = Δ_t` has exactly the displayed closed-form
coefficients: constant term `tr Q₀` and linear coefficient `tr Q₀ - tr Q₀²`, both in
closed form in the chain length and the correlation. -/
theorem ch10_em_weights_smallt_ar {n : ℕ} (hn : 2 ≤ n) (r : ℝ) (h : 1 - r ^ 2 ≠ 0) (ε : ℝ) :
    Matrix.trace ((ch10_arSig n r)⁻¹
        - ε • ((ch10_arSig n r)⁻¹ * (ch10_arSig n r)⁻¹ - (ch10_arSig n r)⁻¹))
      = ch10_trQ0 n r + ε * (ch10_trQ0 n r - ch10_trQ0sq n r) := by
  rw [ch10_em_weights_smallt_trace, ch10_em_trQ0_eq hn r h, ch10_em_trQ0sq_eq hn r h]

-- eq:em-weights-smallt, the quoted numbers, now as instances of the general-`n`
-- theorems about the matrix `Σ₀` itself rather than of stand-alone closed forms.
theorem ch10_trQ0_matrix32 :
    Matrix.trace (ch10_arSig 32 (17 / 20 : ℝ))⁻¹ = 21470 / 111 := by
  rw [ch10_em_trQ0_eq (by norm_num) _ (by norm_num), ch10_trQ0_numeric]

theorem ch10_smallt_coefficient_matrix32 :
    2 * (Matrix.trace (ch10_arSig 32 (17 / 20 : ℝ))⁻¹
          - Matrix.trace ((ch10_arSig 32 (17 / 20 : ℝ))⁻¹ * (ch10_arSig 32 (17 / 20 : ℝ))⁻¹))
      = -(38691320 / 12321) := by
  rw [ch10_em_trQ0_eq (by norm_num) _ (by norm_num),
    ch10_em_trQ0sq_eq (by norm_num) _ (by norm_num)]
  exact ch10_smallt_coefficient

/-! ### eq:em-q-limits as genuine limits (lines 524-536)

The display asserts two limits.  `ch10_em_q_limits_small_t` and
`ch10_em_q_limits_prior` record the algebra at a fixed operating point; the
statements below turn both ends into `Filter.Tendsto` facts, sharpen the `t → 0`
rate from `|q| ≤ 2|β|/J_d` to the displayed `q ≈ |β|/J_d`, and cover `ρ < 0`. -/

/-- `J_d → ∞` as `t ↓ 0`: the evidence precision `e^{-2t}/Δ_t` diverges. -/
theorem ch10_Jevid_tendsto_atTop :
    Filter.Tendsto ch10_Jevid (nhdsWithin 0 (Set.Ioi (0 : ℝ))) Filter.atTop := by
  have hc : Continuous (fun t : ℝ => Real.exp (2 * t) - 1) :=
    (Real.continuous_exp.comp (continuous_const.mul continuous_id)).sub continuous_const
  have h0 : Filter.Tendsto (fun t : ℝ => Real.exp (2 * t) - 1)
      (nhdsWithin 0 (Set.Ioi (0 : ℝ))) (nhds 0) := by
    have h := hc.tendsto 0
    simp only [mul_zero, Real.exp_zero, sub_self] at h
    exact h.mono_left nhdsWithin_le_nhds
  have hw : Filter.Tendsto (fun t : ℝ => Real.exp (2 * t) - 1)
      (nhdsWithin 0 (Set.Ioi (0 : ℝ))) (nhdsWithin 0 (Set.Ioi (0 : ℝ))) := by
    refine tendsto_nhdsWithin_of_tendsto_nhds_of_eventually_within _ h0 ?_
    filter_upwards [self_mem_nhdsWithin] with t ht
    have h1 := Real.add_one_le_exp (2 * t)
    have ht' : (0 : ℝ) < t := ht
    simp only [Set.mem_Ioi]
    linarith
  refine Filter.Tendsto.congr' ?_ (tendsto_inv_nhdsGT_zero.comp hw)
  filter_upwards [self_mem_nhdsWithin] with t ht
  have ht' : (0 : ℝ) < t := ht
  simp only [Function.comp_apply]
  rw [ch10_Jevid_eq ht', one_div]

/-- `J_d → 0` as `t → ∞`: the evidence term vanishes and `J_d` falls to its prior
value, which is the hypothesis of the second line of eq:em-q-limits. -/
theorem ch10_Jevid_tendsto_zero : Filter.Tendsto ch10_Jevid Filter.atTop (nhds 0) := by
  have h1 : Filter.Tendsto (fun t : ℝ => Real.exp (2 * t)) Filter.atTop Filter.atTop := by
    refine Filter.tendsto_atTop_mono' _ ?_ Real.tendsto_exp_atTop
    filter_upwards [Filter.eventually_ge_atTop (0 : ℝ)] with t ht
    exact Real.exp_le_exp.mpr (by linarith)
  have hexp : Filter.Tendsto (fun t : ℝ => Real.exp (2 * t) - 1) Filter.atTop Filter.atTop := by
    simpa [sub_eq_add_neg] using Filter.tendsto_atTop_add_const_right Filter.atTop (-1 : ℝ) h1
  refine Filter.Tendsto.congr' ?_ hexp.inv_tendsto_atTop
  filter_upwards [Filter.eventually_gt_atTop (0 : ℝ)] with t ht
  rw [Pi.inv_apply, ch10_Jevid_eq ht, one_div]

/-- eq:em-q-limits, `t → 0`: `V = 1/√(J_d² - 4β²) → 0` as `J_d → ∞` at fixed
coupling `β` — "sites pinned by their observations". -/
theorem ch10_V_tendsto_zero (b : ℝ) :
    Filter.Tendsto (fun Jd : ℝ => ch10_V Jd b) Filter.atTop (nhds 0) := by
  have h1 : Filter.Tendsto (fun Jd : ℝ => Jd ^ 2) Filter.atTop Filter.atTop :=
    Filter.tendsto_pow_atTop (by norm_num)
  have h2 : Filter.Tendsto (fun Jd : ℝ => Jd ^ 2 - 4 * b ^ 2) Filter.atTop Filter.atTop := by
    simpa [sub_eq_add_neg] using
      Filter.tendsto_atTop_add_const_right Filter.atTop (-(4 * b ^ 2)) h1
  have h3 : Filter.Tendsto (fun Jd : ℝ => Real.sqrt (Jd ^ 2 - 4 * b ^ 2))
      Filter.atTop Filter.atTop := Real.tendsto_sqrt_atTop.comp h2
  simpa [ch10_V, one_div, Pi.inv_def] using h3.inv_tendsto_atTop

section QRate

variable {Jd b : ℝ}

/-- The exact gap identity behind the `t → 0` rate:
`J_d (J_d - √(J_d²-4β²)) = 2β² + (J_d - √(J_d²-4β²))²/2`. -/
theorem ch10_qroot_gap (hb : b ≠ 0) (hJ : 2 * |b| < Jd) :
    Jd * (Jd - Real.sqrt (Jd ^ 2 - 4 * b ^ 2))
      = 2 * b ^ 2 + (Jd - Real.sqrt (Jd ^ 2 - 4 * b ^ 2)) ^ 2 / 2 := by
  have hs := ch10_bulk_sqrt_sq hb hJ
  linear_combination (-(1 : ℝ) / 2) * hs

/-- **eq:em-q-limits, the `t → 0` rate, sharp.**  `J_d|q| ∈ [|β|, |β| + 4|β|³/J_d²]`,
so `|q| = |β|/J_d (1 + O(J_d⁻²))` — the displayed `q ≈ |β|/J_d`, with the constant,
not the factor-of-two bound `|q| ≤ 2|β|/J_d`. -/
theorem ch10_qroot_rate (hb : b ≠ 0) (hJ : 2 * |b| < Jd) :
    |b| ≤ Jd * |ch10_qroot Jd b|
      ∧ (Jd * |ch10_qroot Jd b| - |b|) * Jd ^ 2 ≤ 4 * |b| ^ 3 := by
  have hbpos : 0 < |b| := abs_pos.mpr hb
  have hbne : |b| ≠ 0 := ne_of_gt hbpos
  have hJpos : 0 < Jd := lt_of_le_of_lt (by positivity) hJ
  have hs := ch10_bulk_sqrt_sq hb hJ
  have hspos := ch10_bulk_sqrt_pos hb hJ
  have hslt := ch10_bulk_sqrt_lt hb hJ
  have hkey := ch10_qroot_gap hb hJ
  have habs2 : |b| ^ 2 = b ^ 2 := sq_abs b
  have habs4 : |b| ^ 4 = b ^ 4 := by
    rw [show |b| ^ 4 = (|b| ^ 2) ^ 2 by ring, habs2]; ring
  have hq2 : 2 * |b| * |ch10_qroot Jd b| = Jd - Real.sqrt (Jd ^ 2 - 4 * b ^ 2) := by
    rw [ch10_qroot_abs hb hJ]
    field_simp
  have h1 : 2 * |b| * (Jd * |ch10_qroot Jd b|)
      = Jd * (Jd - Real.sqrt (Jd ^ 2 - 4 * b ^ 2)) := by
    rw [← hq2]; ring
  constructor
  · have h3 : 2 * |b| * |b| ≤ 2 * |b| * (Jd * |ch10_qroot Jd b|) := by
      rw [h1, hkey]
      nlinarith [sq_nonneg (Jd - Real.sqrt (Jd ^ 2 - 4 * b ^ 2)), habs2]
    exact le_of_mul_le_mul_left h3 (by positivity)
  · have hgap : (Jd - Real.sqrt (Jd ^ 2 - 4 * b ^ 2)) * Jd ≤ 4 * b ^ 2 := by
      nlinarith [mul_pos hspos (sub_pos.mpr hslt)]
    have hgap0 : 0 ≤ Jd - Real.sqrt (Jd ^ 2 - 4 * b ^ 2) := by linarith
    have huJd : 0 ≤ (Jd - Real.sqrt (Jd ^ 2 - 4 * b ^ 2)) * Jd :=
      mul_nonneg hgap0 hJpos.le
    have hsq : ((Jd - Real.sqrt (Jd ^ 2 - 4 * b ^ 2)) * Jd) ^ 2 ≤ 16 * b ^ 4 := by
      nlinarith [hgap, huJd]
    have h4 : 2 * |b| * ((Jd * |ch10_qroot Jd b| - |b|) * Jd ^ 2)
        = ((Jd - Real.sqrt (Jd ^ 2 - 4 * b ^ 2)) * Jd) ^ 2 / 2 := by
      linear_combination (Jd ^ 2) * h1 + (Jd ^ 2) * hkey - (2 * Jd ^ 2) * habs2
    have h5 : 2 * |b| * ((Jd * |ch10_qroot Jd b| - |b|) * Jd ^ 2)
        ≤ 2 * |b| * (4 * |b| ^ 3) := by
      rw [h4]
      nlinarith [hsq, habs4]
    exact le_of_mul_le_mul_left h5 (by positivity)

/-- eq:em-q-limits, `t → 0`: `q → 0` as `J_d → ∞`. -/
theorem ch10_qroot_tendsto_zero (b : ℝ) (hb : b ≠ 0) :
    Filter.Tendsto (fun Jd : ℝ => ch10_qroot Jd b) Filter.atTop (nhds 0) := by
  have hbound : Filter.Tendsto (fun Jd : ℝ => 2 * |b| / Jd) Filter.atTop (nhds 0) :=
    Filter.Tendsto.div_atTop tendsto_const_nhds Filter.tendsto_id
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le' (g := fun Jd : ℝ => -(2 * |b| / Jd))
    (h := fun Jd : ℝ => 2 * |b| / Jd) (by simpa using hbound.neg) hbound ?_ ?_
  · filter_upwards [Filter.eventually_gt_atTop (2 * |b|)] with Jd hJd
    exact (abs_le.mp (ch10_em_q_limits_small_t hb hJd)).1
  · filter_upwards [Filter.eventually_gt_atTop (2 * |b|)] with Jd hJd
    exact (abs_le.mp (ch10_em_q_limits_small_t hb hJd)).2

/-- **eq:em-q-limits, `q ≈ |β|/J_d` as a limit:** `J_d |q| → |β|` as `J_d → ∞`. -/
theorem ch10_qroot_asymptotic (b : ℝ) (hb : b ≠ 0) :
    Filter.Tendsto (fun Jd : ℝ => Jd * |ch10_qroot Jd b|) Filter.atTop (nhds |b|) := by
  have hup : Filter.Tendsto (fun Jd : ℝ => |b| + 4 * |b| ^ 3 / Jd ^ 2) Filter.atTop
      (nhds |b|) := by
    have h : Filter.Tendsto (fun Jd : ℝ => 4 * |b| ^ 3 / Jd ^ 2) Filter.atTop (nhds 0) :=
      Filter.Tendsto.div_atTop tendsto_const_nhds (Filter.tendsto_pow_atTop (by norm_num))
    simpa using tendsto_const_nhds.add h
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds hup ?_ ?_
  · filter_upwards [Filter.eventually_gt_atTop (2 * |b|)] with Jd hJd
    exact (ch10_qroot_rate hb hJd).1
  · filter_upwards [Filter.eventually_gt_atTop (2 * |b|),
      Filter.eventually_gt_atTop (0 : ℝ)] with Jd hJd hJ0
    rw [← sub_le_iff_le_add', le_div_iff₀ (by positivity)]
    exact (ch10_qroot_rate hb hJd).2

end QRate

/-- **eq:em-q-limits, the `t → 0` end, as limits** in the chapter's own parameters:
`J_d(t) → ∞`, `q → 0` and `V → 0` as `t ↓ 0`, at the prior coupling
`β = -ρ/σ_η²` held fixed. -/
theorem ch10_em_q_limits_small_t_full (r : ℝ) (hr : r ^ 2 < 1) (hr0 : r ≠ 0) :
    Filter.Tendsto (fun t : ℝ => ch10_Jevid t + (1 + r ^ 2) / (1 - r ^ 2))
        (nhdsWithin 0 (Set.Ioi (0 : ℝ))) Filter.atTop
      ∧ Filter.Tendsto (fun t : ℝ => ch10_qroot (ch10_Jevid t + (1 + r ^ 2) / (1 - r ^ 2))
          (-(r / (1 - r ^ 2)))) (nhdsWithin 0 (Set.Ioi (0 : ℝ))) (nhds 0)
      ∧ Filter.Tendsto (fun t : ℝ => ch10_V (ch10_Jevid t + (1 + r ^ 2) / (1 - r ^ 2))
          (-(r / (1 - r ^ 2)))) (nhdsWithin 0 (Set.Ioi (0 : ℝ))) (nhds 0) := by
  have hden : (0 : ℝ) < 1 - r ^ 2 := by linarith
  have hb : (-(r / (1 - r ^ 2))) ≠ 0 := by
    have h := div_ne_zero hr0 (ne_of_gt hden)
    simpa using h
  have hJ : Filter.Tendsto (fun t : ℝ => ch10_Jevid t + (1 + r ^ 2) / (1 - r ^ 2))
      (nhdsWithin 0 (Set.Ioi (0 : ℝ))) Filter.atTop :=
    Filter.tendsto_atTop_add_const_right _ _ ch10_Jevid_tendsto_atTop
  exact ⟨hJ, (ch10_qroot_tendsto_zero _ hb).comp hJ, (ch10_V_tendsto_zero _).comp hJ⟩

/-- **eq:em-q-limits, the `t → ∞` end, for either sign of `ρ`.**  At the prior
operating point the decay root is exactly `ρ` (so `|q| = |ρ|`, the displayed value,
with the sign alternation of the toolbox for `ρ < 0`) and `V = 1`. -/
theorem ch10_em_q_limits_prior' (r : ℝ) (hr : r ^ 2 < 1) (hr0 : r ≠ 0) :
    ch10_qroot ((1 + r ^ 2) / (1 - r ^ 2)) (-(r / (1 - r ^ 2))) = r
      ∧ |ch10_qroot ((1 + r ^ 2) / (1 - r ^ 2)) (-(r / (1 - r ^ 2)))| = |r|
      ∧ ch10_V ((1 + r ^ 2) / (1 - r ^ 2)) (-(r / (1 - r ^ 2))) = 1 := by
  have hden : (0 : ℝ) < 1 - r ^ 2 := by linarith
  have hden' : (1 : ℝ) - r ^ 2 ≠ 0 := ne_of_gt hden
  have hsqrt :
      Real.sqrt (((1 + r ^ 2) / (1 - r ^ 2)) ^ 2 - 4 * (-(r / (1 - r ^ 2))) ^ 2) = 1 := by
    rw [ch10_em_q_limits_prior_disc r hr, Real.sqrt_one]
  have hq : ch10_qroot ((1 + r ^ 2) / (1 - r ^ 2)) (-(r / (1 - r ^ 2))) = r := by
    unfold ch10_qroot
    rw [hsqrt]
    field_simp
    ring
  refine ⟨hq, by rw [hq], ?_⟩
  unfold ch10_V
  rw [hsqrt]
  norm_num

/-- `J_d(t) → (1+ρ²)/σ_η²` as `t → ∞`: the first limit of the second line. -/
theorem ch10_em_q_limits_prior_Jd (r : ℝ) :
    Filter.Tendsto (fun t : ℝ => ch10_Jevid t + (1 + r ^ 2) / (1 - r ^ 2)) Filter.atTop
      (nhds ((1 + r ^ 2) / (1 - r ^ 2))) := by
  simpa using ch10_Jevid_tendsto_zero.add (tendsto_const_nhds (x := (1 + r ^ 2) / (1 - r ^ 2)))

/-- `q` is continuous in `J_d` at fixed coupling. -/
theorem ch10_qroot_continuous (b : ℝ) : Continuous (fun Jd : ℝ => ch10_qroot Jd b) := by
  have h : Continuous (fun Jd : ℝ => -(Jd - Real.sqrt (Jd ^ 2 - 4 * b ^ 2))) :=
    (continuous_id.sub
      (Real.continuous_sqrt.comp ((continuous_pow 2).sub continuous_const))).neg
  simpa [ch10_qroot] using h.div_const (2 * b)

/-- **eq:em-q-limits, the `t → ∞` end, as limits:** `q(t) → ρ` (hence `|q| → |ρ|`)
and `V(t) → 1` as diffusion time runs out, for either sign of `ρ`. -/
theorem ch10_em_q_limits_prior_limit (r : ℝ) (hr : r ^ 2 < 1) (hr0 : r ≠ 0) :
    Filter.Tendsto (fun t : ℝ => ch10_qroot (ch10_Jevid t + (1 + r ^ 2) / (1 - r ^ 2))
        (-(r / (1 - r ^ 2)))) Filter.atTop (nhds r)
      ∧ Filter.Tendsto (fun t : ℝ => ch10_V (ch10_Jevid t + (1 + r ^ 2) / (1 - r ^ 2))
        (-(r / (1 - r ^ 2)))) Filter.atTop (nhds 1) := by
  have hden : (0 : ℝ) < 1 - r ^ 2 := by linarith
  obtain ⟨hq, -, hV⟩ := ch10_em_q_limits_prior' r hr hr0
  have hJ := ch10_em_q_limits_prior_Jd r
  have hsqrt :
      Real.sqrt (((1 + r ^ 2) / (1 - r ^ 2)) ^ 2 - 4 * (-(r / (1 - r ^ 2))) ^ 2) = 1 := by
    rw [ch10_em_q_limits_prior_disc r hr, Real.sqrt_one]
  constructor
  · have hcomp := ((ch10_qroot_continuous (-(r / (1 - r ^ 2)))).tendsto _).comp hJ
    rw [hq] at hcomp
    exact hcomp
  · have hVcont : ContinuousAt (fun J : ℝ => ch10_V J (-(r / (1 - r ^ 2))))
        ((1 + r ^ 2) / (1 - r ^ 2)) := by
      unfold ch10_V
      refine ContinuousAt.div continuousAt_const ?_ ?_
      · exact (Real.continuous_sqrt.comp
          ((continuous_pow 2).sub continuous_const)).continuousAt
      · show Real.sqrt (((1 + r ^ 2) / (1 - r ^ 2)) ^ 2
            - 4 * (-(r / (1 - r ^ 2))) ^ 2) ≠ 0
        rw [hsqrt]
        norm_num
    have hcomp := hVcont.tendsto.comp hJ
    rw [hV] at hcomp
    exact hcomp

/-! ### eq:em-breakeven, the reported brackets (lines 1078-1083)

The display reports `δ*` as a *bracket* between adjacent tested strengths rather
than a point estimate.  What makes such a bracket a break-even bracket is a sign
change of the advantage ratio across it: given the two measured endpoint ratios and
continuity of the ratio in `δ` (which is what speaking of a crossing presupposes;
the thesis itself declines to attach an interval to it), a `δ*` exists strictly
inside.  The numbers below are the entries of tab:em-misspecification. -/

/-- **The bracket claim, in general.**  If the advantage ratio `R` varies
continuously with the misspecification `δ` between two adjacent tested strengths
and straddles `1` there, a break-even `δ*` lies strictly inside the bracket. -/
theorem ch10_breakeven_bracket {a b : ℝ} (hab : a ≤ b) (R : ℝ → ℝ)
    (hcont : ContinuousOn R (Set.Icc a b)) (ha : 1 < R a) (hb : R b < 1) :
    ∃ d ∈ Set.Ioo a b, R d = 1 := by
  obtain ⟨d, hd, hRd⟩ := intermediate_value_Ioo' hab hcont (Set.mem_Ioo.mpr ⟨hb, ha⟩)
  exact ⟨d, hd, hRd⟩

/-- A break-even point is unique wherever the ratio is strictly decreasing in `δ`,
which is the monotonicity the text reports ("the advantage is monotone in `δ`"). -/
theorem ch10_breakeven_unique {a b : ℝ} (R : ℝ → ℝ) (hanti : StrictAntiOn R (Set.Icc a b))
    {d₁ d₂ : ℝ} (h₁ : d₁ ∈ Set.Icc a b) (h₂ : d₂ ∈ Set.Icc a b)
    (hR₁ : R d₁ = 1) (hR₂ : R d₂ = 1) : d₁ = d₂ :=
  hanti.injOn h₁ h₂ (hR₁.trans hR₂.symm)

/-- **eq:em-breakeven, the CNN bracket `[1.1, 2.2] × 10⁻²`.**  Ratios `1.2` at
`δ = 1.1e-2` and `0.97` at `δ = 2.2e-2` (tab:em-misspecification). -/
theorem ch10_em_breakeven_cnn_bracket (R : ℝ → ℝ)
    (hcont : ContinuousOn R (Set.Icc (0.011 : ℝ) 0.022))
    (h1 : R 0.011 = 1.2) (h2 : R 0.022 = 0.97) :
    ∃ d ∈ Set.Ioo (0.011 : ℝ) 0.022, R d = 1 :=
  ch10_breakeven_bracket (by norm_num) R hcont (by rw [h1]; norm_num) (by rw [h2]; norm_num)

/-- **eq:em-breakeven, the MLP bracket `[2.2, 4.0] × 10⁻²`.**  Ratios `1.2` at
`δ = 2.2e-2` and `0.84` at `δ = 4.0e-2` (tab:em-misspecification). -/
theorem ch10_em_breakeven_mlp_bracket (R : ℝ → ℝ)
    (hcont : ContinuousOn R (Set.Icc (0.022 : ℝ) 0.040))
    (h1 : R 0.022 = 1.2) (h2 : R 0.040 = 0.84) :
    ∃ d ∈ Set.Ioo (0.022 : ℝ) 0.040, R d = 1 :=
  ch10_breakeven_bracket (by norm_num) R hcont (by rw [h1]; norm_num) (by rw [h2]; norm_num)

/-- **Why the MLP bracket starts at `2.2 × 10⁻²` and not at `1.1 × 10⁻²`.**  The MLP
ratio is still `1.7` at `δ = 1.1e-2` and `1.2` at `2.2e-2`; wherever the ratio is
antitone in `δ` it stays strictly above break-even on the whole earlier interval, so
no crossing is bracketed there. -/
theorem ch10_em_breakeven_mlp_no_earlier_crossing {c : ℝ} (R : ℝ → ℝ) (hc : c ≤ 0.022)
    (hanti : AntitoneOn R (Set.Icc c (0.022 : ℝ))) (h2 : R 0.022 = 1.2) :
    ∀ d ∈ Set.Icc c (0.022 : ℝ), 1 < R d := by
  intro d hd
  have hmem : (0.022 : ℝ) ∈ Set.Icc c (0.022 : ℝ) := Set.right_mem_Icc.mpr hc
  have hle : R (0.022 : ℝ) ≤ R d := hanti hd hmem hd.2
  rw [h2] at hle
  linarith

end

end ThesisAudit
