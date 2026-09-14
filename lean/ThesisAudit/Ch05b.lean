import Mathlib

/-!
# Audit of `thesis/chapters/ch05-gaussian-matrix.tex`, lines 785-1686

Conventions used throughout this file.

* The clean covariance `Σ₀` is kept **abstract** (an invertible symmetric
  matrix) wherever the text's argument only uses invertibility; this is
  strictly stronger than the AR(1) instance the thesis works with.  The
  concrete AR(1) Toeplitz matrix `ch05b_Sig0` is used for the instance
  checks.
* `Σ_t := e^{-2t} Σ₀ + Δ_t I` (eq:g-Sigt, tex 617) and `Q_t := Σ_t⁻¹`
  (eq:g-Qt-def, tex 686).  Matrix inverses are mathlib's `Matrix.inv`
  (`⁻¹`), and the identities are proved through `Matrix.inv_eq_right_inv`,
  so no invertibility side conditions are hidden.
* Densities are handled through their (negative log) exponents: the thesis'
  "∝" statements are exactly statements about the exponent up to an additive
  constant, and that is what is formalised.  Conditional expectations of a
  Gaussian posterior are characterised as the unique minimiser of the
  posterior exponent, which is the standard characterisation and the one the
  "completing the square" toolbox uses.
* Asymptotic statements (`O(t^d)`, Neumann series) are formalised by the
  *exact finite identity with explicit remainder* that the text's expansion
  is shorthand for; the remainder is displayed in the statement.
-/

namespace ThesisAudit

open Matrix Finset

noncomputable section

variable {n : ℕ}

/-! ## Shared scalar definitions -/

/-- `Δ_t = 1 - e^{-2t}` (tex line 592). -/
def ch05b_Delta (t : ℝ) : ℝ := 1 - Real.exp (-2 * t)

theorem ch05b_Delta_pos {t : ℝ} (ht : 0 < t) : 0 < ch05b_Delta t := by
  have : Real.exp (-2 * t) < 1 := Real.exp_lt_one_iff.mpr (by linarith)
  simp only [ch05b_Delta, sub_pos]
  exact this

theorem ch05b_Delta_ne {t : ℝ} (ht : 0 < t) : ch05b_Delta t ≠ 0 :=
  ne_of_gt (ch05b_Delta_pos ht)

/-- `γ_t := e^{-2t}/Δ_t` — eq:g-gamma-def, tex 1276-1281. -/
def ch05b_gamma (t : ℝ) : ℝ := Real.exp (-2 * t) / ch05b_Delta t

/-- `c_t := e^{2t} - 1` — eq:g-ct-def, tex 1356-1359. -/
def ch05b_c (t : ℝ) : ℝ := Real.exp (2 * t) - 1

theorem ch05b_c_pos {t : ℝ} (ht : 0 < t) : 0 < ch05b_c t := by
  have : (1 : ℝ) < Real.exp (2 * t) := Real.one_lt_exp_iff.mpr (by linarith)
  simp only [ch05b_c, sub_pos]; exact this

/-- `Δ_t γ_t = e^{-2t}`. -/
theorem ch05b_Delta_mul_gamma {t : ℝ} (hΔ : ch05b_Delta t ≠ 0) :
    ch05b_Delta t * ch05b_gamma t = Real.exp (-2 * t) := by
  simp only [ch05b_gamma]
  field_simp

/-- `e^{-t} e^{-t} = e^{-2t}`. -/
theorem ch05b_exp_sq (t : ℝ) : Real.exp (-t) * Real.exp (-t) = Real.exp (-2 * t) := by
  rw [← Real.exp_add]; congr 1; ring

/-! ## Shared matrix definitions -/

/-- `Σ_t = e^{-2t} Σ₀ + Δ_t I` (eq:g-Sigt, tex 617). -/
def ch05b_Sigt (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  Real.exp (-2 * t) • S0 + ch05b_Delta t • (1 : Matrix (Fin n) (Fin n) ℝ)

/-- `Q_t := Σ_t⁻¹` (eq:g-Qt-def, tex 686). -/
def ch05b_Qt (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  (ch05b_Sigt S0 t)⁻¹

/-- The posterior precision `J = Q₀ + (e^{-2t}/Δ_t) I` (eq:g-posterior-precision). -/
def ch05b_J (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  S0⁻¹ + ch05b_gamma t • (1 : Matrix (Fin n) (Fin n) ℝ)

/-- The posterior field `h = (e^{-t}/Δ_t) x` (eq:g-posterior). -/
def ch05b_h (t : ℝ) (x : Fin n → ℝ) : Fin n → ℝ :=
  (Real.exp (-t) / ch05b_Delta t) • x

/-- The joint score `s(x,t) = -Q_t x` (eq:g-score-matrix). -/
def ch05b_score (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ) (x : Fin n → ℝ) : Fin n → ℝ :=
  -(ch05b_Qt S0 t *ᵥ x)

/-- Stationary AR(1) covariance `(Σ₀)_{ij} = α^{|i-j|}` (eq:g-cov-lag, tex 230). -/
def ch05b_Sig0 (α : ℝ) (n : ℕ) : Matrix (Fin n) (Fin n) ℝ :=
  fun i j => α ^ (Int.natAbs ((i : ℤ) - (j : ℤ)))

/-! ## The score, two ways (tex 785-1120) -/

-- eq:g-score-matrix (ch05 lines 791-795): the joint score of a centred Gaussian
-- is `-Q_t x`.  The gradient claim `∇(xᵀQx) = 2Qx` is formalised by the exact
-- first-order expansion: the increment is `2(Qx)·v` plus a term quadratic in `v`.
theorem ch05b_g_score_matrix_grad (Q : Matrix (Fin n) (Fin n) ℝ) (hQ : Q.IsSymm)
    (x v : Fin n → ℝ) :
    (x + v) ⬝ᵥ (Q *ᵥ (x + v)) - x ⬝ᵥ (Q *ᵥ x)
      = 2 * ((Q *ᵥ x) ⬝ᵥ v) + v ⬝ᵥ (Q *ᵥ v) := by
  have hsym : ∀ u w : Fin n → ℝ, u ⬝ᵥ (Q *ᵥ w) = (Q *ᵥ u) ⬝ᵥ w := by
    intro u w
    rw [dotProduct_mulVec, ← Matrix.mulVec_transpose, hQ.eq]
  have h1 : v ⬝ᵥ (Q *ᵥ x) = (Q *ᵥ x) ⬝ᵥ v := dotProduct_comm _ _
  have h2 : x ⬝ᵥ (Q *ᵥ v) = (Q *ᵥ x) ⬝ᵥ v := hsym x v
  simp only [Matrix.mulVec_add, dotProduct_add, add_dotProduct]
  rw [h1, h2]
  ring

-- eq:g-score-matrix (ch05 lines 791-795): definitional form of the score.
theorem ch05b_g_score_matrix (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ) (x : Fin n → ℝ) :
    ch05b_score S0 t x = -((ch05b_Sigt S0 t)⁻¹ *ᵥ x) := rfl

/-- At `L = 2` the noisy covariance is `!![1, r; r, 1]` with `r = α e^{-2t}`
(tex 798-800). -/
theorem ch05b_Sigt_two (α t : ℝ) :
    ch05b_Sigt (ch05b_Sig0 α 2) t = !![1, Real.exp (-2 * t) * α; Real.exp (-2 * t) * α, 1] := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [ch05b_Sigt, ch05b_Sig0, ch05b_Delta]

/-- The `2 × 2` inverse used at `L = 2`. -/
theorem ch05b_inv_two (r : ℝ) (hne : (1 : ℝ) - r ^ 2 ≠ 0) :
    (!![(1 : ℝ), r; r, 1])⁻¹ = (1 / (1 - r ^ 2)) • !![(1 : ℝ), -r; -r, 1] := by
  refine Matrix.inv_eq_right_inv ?_
  rw [Matrix.mul_smul]
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [Matrix.mul_fin_two, Matrix.one_apply] <;> field_simp <;> ring

/-- Tex 798-800: at `L = 2`, `s₀(x,t) = -(x₀ - r x₁)/(1 - r²)` with `r = α e^{-2t}`. -/
theorem ch05b_g_score_matrix_two (α t : ℝ) (x : Fin 2 → ℝ)
    (hne : (1 : ℝ) - (Real.exp (-2 * t) * α) ^ 2 ≠ 0) :
    ch05b_score (ch05b_Sig0 α 2) t x 0
      = -(x 0 - (Real.exp (-2 * t) * α) * x 1) / (1 - (Real.exp (-2 * t) * α) ^ 2) := by
  have hinv : (ch05b_Sigt (ch05b_Sig0 α 2) t)⁻¹
      = (1 / (1 - (Real.exp (-2 * t) * α) ^ 2)) •
          !![(1 : ℝ), -(Real.exp (-2 * t) * α); -(Real.exp (-2 * t) * α), 1] := by
    rw [ch05b_Sigt_two]; exact ch05b_inv_two _ hne
  simp only [ch05b_score, ch05b_Qt, hinv, Pi.neg_apply, Matrix.smul_mulVec,
    Pi.smul_apply, smul_eq_mul]
  simp [Matrix.mulVec, dotProduct, Fin.sum_univ_two]
  field_simp
  ring

/-! ### The posterior of the clean chain (tex 828-1025) -/

/-- Gaussian prior exponent, `P₀(a) ∝ exp(-½ aᵀQ₀a)` (tex 876-882). -/
def ch05b_prior (Q0 : Matrix (Fin n) (Fin n) ℝ) (a : Fin n → ℝ) : ℝ :=
  Real.exp (-(1 / 2) * (a ⬝ᵥ (Q0 *ᵥ a)))

/-- Channel likelihood, `P(x|a) ∝ exp(-‖x - e^{-t}a‖²/(2Δ_t))` (tex 884-891). -/
def ch05b_lik (t : ℝ) (x a : Fin n → ℝ) : ℝ :=
  Real.exp (-(1 / (2 * ch05b_Delta t)) * (∑ k, (x k - Real.exp (-t) * a k) ^ 2))

/-- `-2 log` of the unnormalised posterior (tex 894-903). -/
def ch05b_negLogPost (Q0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ) (x a : Fin n → ℝ) : ℝ :=
  a ⬝ᵥ (Q0 *ᵥ a) + (1 / ch05b_Delta t) * (∑ k, (x k - Real.exp (-t) * a k) ^ 2)

/-- Gaussian information-form exponent `-½(aᵀJa - 2hᵀa)` (tex 939-944). -/
def ch05b_infoForm (J : Matrix (Fin n) (Fin n) ℝ) (h a : Fin n → ℝ) : ℝ :=
  -(1 / 2) * (a ⬝ᵥ (J *ᵥ a) - 2 * (h ⬝ᵥ a))

theorem ch05b_prior_pos (Q0 : Matrix (Fin n) (Fin n) ℝ) (a : Fin n → ℝ) :
    0 < ch05b_prior Q0 a := Real.exp_pos _

theorem ch05b_lik_pos (t : ℝ) (x a : Fin n → ℝ) : 0 < ch05b_lik t x a := Real.exp_pos _

-- noname-2 (ch05 lines 870-874): Bayes' rule `P(a|x) ∝ P₀(a) P(x|a)`; the content
-- used by the text is that negative logs add.
theorem ch05b_noname_bayes (Q0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ) (x a : Fin n → ℝ) :
    -2 * Real.log (ch05b_prior Q0 a * ch05b_lik t x a)
      = -2 * Real.log (ch05b_prior Q0 a) + -2 * Real.log (ch05b_lik t x a) := by
  rw [Real.log_mul (ne_of_gt (ch05b_prior_pos Q0 a)) (ne_of_gt (ch05b_lik_pos t x a))]
  ring

-- noname-4 (ch05 lines 884-891): the vector form of the channel likelihood is the
-- product of the `L` scalar Gaussian channel densities.
theorem ch05b_noname_lik_prod (t : ℝ) (x a : Fin n → ℝ) :
    ch05b_lik t x a
      = ∏ k, Real.exp (-((x k - Real.exp (-t) * a k) ^ 2) / (2 * ch05b_Delta t)) := by
  rw [← Real.exp_sum]
  unfold ch05b_lik
  congr 1
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl ?_
  intro k _
  ring

-- eq:g-post-expand-1 (ch05 lines 894-903): `-2 log P(a|x)` is the prior quadratic
-- form plus the scaled channel residual, up to a constant in `a`.
theorem ch05b_g_post_expand_1 (Q0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ) (x a : Fin n → ℝ) :
    -2 * Real.log (ch05b_prior Q0 a * ch05b_lik t x a)
      = ch05b_negLogPost Q0 t x a := by
  rw [ch05b_noname_bayes]
  unfold ch05b_prior ch05b_lik ch05b_negLogPost
  rw [Real.log_exp, Real.log_exp]
  ring

-- noname-5 (ch05 lines 905-911): expansion of the channel residual.
theorem ch05b_noname_residual (t : ℝ) (x a : Fin n → ℝ) :
    (∑ k, (x k - Real.exp (-t) * a k) ^ 2)
      = x ⬝ᵥ x - 2 * Real.exp (-t) * (x ⬝ᵥ a) + Real.exp (-2 * t) * (a ⬝ᵥ a) := by
  have hexp : Real.exp (-2 * t) = Real.exp (-t) * Real.exp (-t) := by
    rw [← Real.exp_add]; ring_nf
  simp only [dotProduct, Finset.mul_sum, ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl ?_
  intro k _
  rw [hexp]; ring

/-- `a ⬝ᵥ ((c • 1) *ᵥ a) = c * (a ⬝ᵥ a)`. -/
theorem ch05b_dot_smul_one (c : ℝ) (a : Fin n → ℝ) :
    a ⬝ᵥ ((c • (1 : Matrix (Fin n) (Fin n) ℝ)) *ᵥ a) = c * (a ⬝ᵥ a) := by
  rw [Matrix.smul_mulVec, Matrix.one_mulVec, dotProduct_smul, smul_eq_mul]

-- eq:g-post-expand-2 / eq:g-J-identified / eq:g-h-identified
-- (ch05 lines 914-936, 946-955): matching the posterior exponent against the
-- information form identifies `J = Q₀ + (e^{-2t}/Δ_t) I` and `h = (e^{-t}/Δ_t) x`.
theorem ch05b_g_post_expand_2 (t : ℝ) (S0 : Matrix (Fin n) (Fin n) ℝ)
    (x a : Fin n → ℝ) :
    ch05b_negLogPost S0⁻¹ t x a
      = a ⬝ᵥ (ch05b_J S0 t *ᵥ a) - 2 * (ch05b_h t x ⬝ᵥ a)
        + (1 / ch05b_Delta t) * (x ⬝ᵥ x) := by
  have h1 : a ⬝ᵥ (ch05b_J S0 t *ᵥ a)
      = a ⬝ᵥ (S0⁻¹ *ᵥ a) + ch05b_gamma t * (a ⬝ᵥ a) := by
    simp only [ch05b_J, Matrix.add_mulVec, dotProduct_add, ch05b_dot_smul_one]
  have h2 : ch05b_h t x ⬝ᵥ a = (Real.exp (-t) / ch05b_Delta t) * (x ⬝ᵥ a) := by
    simp only [ch05b_h, smul_dotProduct, smul_eq_mul]
  simp only [ch05b_negLogPost, ch05b_noname_residual, h1, h2, ch05b_gamma]
  ring

-- eq:g-J-identified (ch05 lines 946-950): the quadratic part of the posterior
-- exponent is `J = Q₀ + (e^{-2t}/Δ_t) I`.
theorem ch05b_g_J_identified (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ) :
    ch05b_J S0 t = S0⁻¹ + ch05b_gamma t • (1 : Matrix (Fin n) (Fin n) ℝ) := rfl

-- eq:g-h-identified (ch05 lines 951-955): the linear part is `h = (e^{-t}/Δ_t) x`.
theorem ch05b_g_h_identified (t : ℝ) (x : Fin n → ℝ) :
    ch05b_h t x = (Real.exp (-t) / ch05b_Delta t) • x := rfl

-- eq:g-posterior-precision (ch05 lines 837-844) and
-- eq:g-information-addition (ch05 lines 986-995): the posterior precision is the
-- sum of prior information and channel information, the latter a pure diagonal.
theorem ch05b_g_information_addition (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ) :
    ch05b_J S0 t - S0⁻¹
      = (Real.exp (-2 * t) / ch05b_Delta t) • (1 : Matrix (Fin n) (Fin n) ℝ) := by
  simp [ch05b_J, ch05b_gamma]

-- noname-7 (ch05 lines 957-965): completing the square.
theorem ch05b_noname_complete_square (J : Matrix (Fin n) (Fin n) ℝ) (hJ : J.IsSymm)
    (hu : IsUnit J.det) (h a : Fin n → ℝ) :
    a ⬝ᵥ (J *ᵥ a) - 2 * (h ⬝ᵥ a)
      = (a - J⁻¹ *ᵥ h) ⬝ᵥ (J *ᵥ (a - J⁻¹ *ᵥ h)) - h ⬝ᵥ (J⁻¹ *ᵥ h) := by
  have hsym : ∀ u w : Fin n → ℝ, u ⬝ᵥ (J *ᵥ w) = (J *ᵥ u) ⬝ᵥ w := by
    intro u w
    rw [dotProduct_mulVec, ← Matrix.mulVec_transpose, hJ.eq]
  have hJm : J *ᵥ (J⁻¹ *ᵥ h) = h := by
    rw [Matrix.mulVec_mulVec, Matrix.mul_nonsing_inv _ hu, Matrix.one_mulVec]
  have key : (J⁻¹ *ᵥ h) ⬝ᵥ (J *ᵥ (J⁻¹ *ᵥ h)) = h ⬝ᵥ (J⁻¹ *ᵥ h) := by
    rw [hJm]; exact dotProduct_comm _ _
  simp only [Matrix.mulVec_sub, dotProduct_sub, sub_dotProduct]
  rw [key, hJm, hsym (J⁻¹ *ᵥ h) a, hJm]
  have : a ⬝ᵥ h = h ⬝ᵥ a := dotProduct_comm _ _
  rw [this]
  ring

-- noname-1 (ch05 lines 831-835) and noname-8 (ch05 lines 967-975):
-- `P(a|x) = N(a; m, J⁻¹)` with `m = J⁻¹h`: the posterior exponent is the
-- centred quadratic form of precision `J` about `m`, up to a constant in `a`.
theorem ch05b_noname_posterior_gaussian (t : ℝ)
    (S0 : Matrix (Fin n) (Fin n) ℝ) (hJ : (ch05b_J S0 t).IsSymm)
    (hu : IsUnit (ch05b_J S0 t).det) (x a : Fin n → ℝ) :
    ch05b_negLogPost S0⁻¹ t x a
      = (a - (ch05b_J S0 t)⁻¹ *ᵥ ch05b_h t x)
          ⬝ᵥ (ch05b_J S0 t *ᵥ (a - (ch05b_J S0 t)⁻¹ *ᵥ ch05b_h t x))
        + ((1 / ch05b_Delta t) * (x ⬝ᵥ x)
            - ch05b_h t x ⬝ᵥ ((ch05b_J S0 t)⁻¹ *ᵥ ch05b_h t x)) := by
  rw [ch05b_g_post_expand_2 t,
    ch05b_noname_complete_square (ch05b_J S0 t) hJ hu (ch05b_h t x) a]
  ring

-- eq:g-posterior-mean (ch05 lines 857-864), noname-9 (ch05 lines 977-981) and
-- eq:g-posterior (ch05 lines 846-855, the equation `Jm = h`):
-- the posterior mean `m = J⁻¹h` is the unique minimiser of the posterior exponent.
theorem ch05b_g_posterior_mean (t : ℝ)
    (S0 : Matrix (Fin n) (Fin n) ℝ) (hJ : (ch05b_J S0 t).IsSymm)
    (hu : IsUnit (ch05b_J S0 t).det)
    (hpos : ∀ v : Fin n → ℝ, v ≠ 0 → 0 < v ⬝ᵥ (ch05b_J S0 t *ᵥ v))
    (x a : Fin n → ℝ) (ha : a ≠ (ch05b_J S0 t)⁻¹ *ᵥ ch05b_h t x) :
    ch05b_negLogPost S0⁻¹ t x ((ch05b_J S0 t)⁻¹ *ᵥ ch05b_h t x)
      < ch05b_negLogPost S0⁻¹ t x a := by
  have hm := ch05b_noname_posterior_gaussian t S0 hJ hu x
    ((ch05b_J S0 t)⁻¹ *ᵥ ch05b_h t x)
  have hA := ch05b_noname_posterior_gaussian t S0 hJ hu x a
  have hv : a - (ch05b_J S0 t)⁻¹ *ᵥ ch05b_h t x ≠ 0 := sub_ne_zero_of_ne ha
  have := hpos _ hv
  rw [hm, hA]
  simp only [sub_self, Matrix.mulVec_zero, dotProduct_zero, zero_add]
  linarith

-- eq:g-posterior (ch05 lines 846-855): `J m = h` for `m = J⁻¹ h`.
theorem ch05b_g_posterior (J : Matrix (Fin n) (Fin n) ℝ) (hu : IsUnit J.det)
    (h : Fin n → ℝ) : J *ᵥ (J⁻¹ *ᵥ h) = h := by
  rw [Matrix.mulVec_mulVec, Matrix.mul_nonsing_inv _ hu, Matrix.one_mulVec]

-- noname-6 (ch05 lines 939-944): the information form of a Gaussian exponent.
theorem ch05b_noname_infoForm (J : Matrix (Fin n) (Fin n) ℝ) (h a : Fin n → ℝ) :
    ch05b_infoForm J h a = -(1 / 2) * (a ⬝ᵥ (J *ᵥ a) - 2 * (h ⬝ᵥ a)) := rfl

-- noname-3 (ch05 lines 876-882): the prior exponent; positivity sanity check.
theorem ch05b_noname_prior (Q0 : Matrix (Fin n) (Fin n) ℝ) (a : Fin n → ℝ) :
    ch05b_prior Q0 a = Real.exp (-(1 / 2) * (a ⬝ᵥ (Q0 *ᵥ a))) := rfl

-- noname-10 (ch05 lines 1003-1005): `Q_t = Σ_t⁻¹`, the marginal precision.
theorem ch05b_noname_Qt_def (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ) :
    ch05b_Qt S0 t = (ch05b_Sigt S0 t)⁻¹ := rfl

-- noname-11 (ch05 lines 1009-1021): the boxed contrast `J` tridiagonal vs `Q_t`
-- generally dense.  Checked as an instance: `L = 3`, AR(1) with `α = 1/2` and
-- `e^{-2t} = 1/2`, where `J₀₂ = 0` (the prior precision is tridiagonal and the
-- channel only adds a diagonal) but `(Q_t)₀₂ = -1/14 ≠ 0`.
theorem ch05b_noname_J_tridiag_Qt_dense :
    (ch05b_Sig0 (1/2 : ℝ) 3)⁻¹ 0 2 = 0 ∧
      ((1/2 : ℝ) • ch05b_Sig0 (1/2 : ℝ) 3
        + (1/2 : ℝ) • (1 : Matrix (Fin 3) (Fin 3) ℝ))⁻¹ 0 2 = -(1/14 : ℝ) := by
  constructor
  · have hS : (ch05b_Sig0 (1/2 : ℝ) 3)⁻¹
        = !![(4/3 : ℝ), -2/3, 0; -2/3, 5/3, -2/3; 0, -2/3, 4/3] := by
      refine Matrix.inv_eq_right_inv ?_
      ext i j
      fin_cases i <;> fin_cases j <;>
        simp [ch05b_Sig0, Matrix.mul_apply, Fin.sum_univ_three, Matrix.one_apply] <;> norm_num
    rw [hS]
    norm_num [Matrix.cons_val_two, Matrix.vecHead, Matrix.vecTail]
  · have hM : ((1/2 : ℝ) • ch05b_Sig0 (1/2 : ℝ) 3
        + (1/2 : ℝ) • (1 : Matrix (Fin 3) (Fin 3) ℝ))⁻¹
        = !![(15/14 : ℝ), -1/4, -1/14; -1/4, 9/8, -1/4; -1/14, -1/4, 15/14] := by
      refine Matrix.inv_eq_right_inv ?_
      ext i j
      fin_cases i <;> fin_cases j <;>
        simp [ch05b_Sig0, Matrix.mul_apply, Fin.sum_univ_three, Matrix.one_apply] <;> norm_num
    rw [hM]
    norm_num [Matrix.cons_val_two, Matrix.vecHead, Matrix.vecTail]

/-! ### The two routes to the score agree (tex 1028-1120) -/

/-- Posterior mean `m = J⁻¹h` (eq:g-posterior-mean). -/
def ch05b_m (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ) (x : Fin n → ℝ) : Fin n → ℝ :=
  (ch05b_J S0 t)⁻¹ *ᵥ ch05b_h t x

-- noname-12 (ch05 lines 1029-1035): `m = J⁻¹h = (e^{-t}/Δ_t) J⁻¹x`.
theorem ch05b_noname_m_eq (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ) (x : Fin n → ℝ) :
    ch05b_m S0 t x = (Real.exp (-t) / ch05b_Delta t) • ((ch05b_J S0 t)⁻¹ *ᵥ x) := by
  simp [ch05b_m, ch05b_h, Matrix.mulVec_smul]

-- noname-13 (ch05 lines 1051-1066): `Δ_t J = Σ₀⁻¹ Σ_t`, i.e. `J = (1/Δ_t) Σ₀⁻¹Σ_t`.
theorem ch05b_noname_J_factor (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ)
    (hS0 : IsUnit S0.det) (hΔ : ch05b_Delta t ≠ 0) :
    ch05b_Delta t • ch05b_J S0 t = S0⁻¹ * ch05b_Sigt S0 t := by
  have h1 : S0⁻¹ * ch05b_Sigt S0 t
      = Real.exp (-2 * t) • (1 : Matrix (Fin n) (Fin n) ℝ) + ch05b_Delta t • S0⁻¹ := by
    simp [ch05b_Sigt, Matrix.mul_add, Matrix.mul_smul, Matrix.nonsing_inv_mul _ hS0]
  rw [h1, ch05b_J, smul_add, smul_smul, ch05b_Delta_mul_gamma hΔ, add_comm]

theorem ch05b_noname_J_factor' (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ)
    (hS0 : IsUnit S0.det) (hΔ : ch05b_Delta t ≠ 0) :
    ch05b_J S0 t = (1 / ch05b_Delta t) • (S0⁻¹ * ch05b_Sigt S0 t) := by
  rw [← ch05b_noname_J_factor S0 t hS0 hΔ, smul_smul, one_div, inv_mul_cancel₀ hΔ, one_smul]

-- eq:g-Jinv-relation (ch05 lines 1068-1075): `J⁻¹ = Δ_t Σ_t⁻¹Σ₀ = Δ_t Q_tΣ₀`.
theorem ch05b_g_Jinv_relation (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ)
    (hS0 : IsUnit S0.det) (hSt : IsUnit (ch05b_Sigt S0 t).det) (hΔ : ch05b_Delta t ≠ 0) :
    (ch05b_J S0 t)⁻¹ = ch05b_Delta t • (ch05b_Qt S0 t * S0) := by
  refine Matrix.inv_eq_right_inv ?_
  have key : (S0⁻¹ * ch05b_Sigt S0 t) * ((ch05b_Sigt S0 t)⁻¹ * S0) = 1 := by
    rw [Matrix.mul_assoc, ← Matrix.mul_assoc (ch05b_Sigt S0 t),
      Matrix.mul_nonsing_inv _ hSt, Matrix.one_mul, Matrix.nonsing_inv_mul _ hS0]
  calc ch05b_J S0 t * (ch05b_Delta t • (ch05b_Qt S0 t * S0))
      = ch05b_Delta t • (ch05b_J S0 t * (ch05b_Qt S0 t * S0)) := Matrix.mul_smul _ _ _
    _ = (ch05b_Delta t • ch05b_J S0 t) * (ch05b_Qt S0 t * S0) := (Matrix.smul_mul _ _ _).symm
    _ = (S0⁻¹ * ch05b_Sigt S0 t) * ((ch05b_Sigt S0 t)⁻¹ * S0) := by
          rw [ch05b_noname_J_factor S0 t hS0 hΔ]; rfl
    _ = 1 := key

-- noname-14 (ch05 lines 1077-1106): the operator of eq:g-score-posterior-form
-- collapses to `-Q_t`.
theorem ch05b_noname_score_op (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ)
    (hS0 : IsUnit S0.det) (hSt : IsUnit (ch05b_Sigt S0 t).det) (hΔ : ch05b_Delta t ≠ 0) :
    (Real.exp (-2 * t) / (ch05b_Delta t) ^ 2) • (ch05b_J S0 t)⁻¹
        - (1 / ch05b_Delta t) • (1 : Matrix (Fin n) (Fin n) ℝ)
      = -(ch05b_Qt S0 t) := by
  have h1 : ch05b_Qt S0 t * (Real.exp (-2 * t) • S0)
      = 1 - ch05b_Delta t • ch05b_Qt S0 t := by
    have hsplit : Real.exp (-2 * t) • S0
        = ch05b_Sigt S0 t - ch05b_Delta t • (1 : Matrix (Fin n) (Fin n) ℝ) := by
      simp [ch05b_Sigt]
    rw [hsplit, Matrix.mul_sub, Matrix.mul_smul, Matrix.mul_one, ch05b_Qt,
      Matrix.nonsing_inv_mul _ hSt]
  have hcoef : Real.exp (-2 * t) / (ch05b_Delta t) ^ 2 * ch05b_Delta t
      = 1 / ch05b_Delta t * Real.exp (-2 * t) := by
    field_simp
  have h2 : (Real.exp (-2 * t) / (ch05b_Delta t) ^ 2) • (ch05b_J S0 t)⁻¹
      = (1 / ch05b_Delta t) • (1 : Matrix (Fin n) (Fin n) ℝ) - ch05b_Qt S0 t := by
    rw [ch05b_g_Jinv_relation S0 t hS0 hSt hΔ, smul_smul, hcoef, ← smul_smul,
      ← Matrix.mul_smul, h1, smul_sub, smul_smul, one_div, inv_mul_cancel₀ hΔ, one_smul]
  rw [h2]; abel

-- eq:g-score-posterior-form (ch05 lines 1037-1048): `s = (e^{-t}m - x)/Δ_t`
-- equals `((e^{-2t}/Δ_t²)J⁻¹ - (1/Δ_t)I)x`.
theorem ch05b_g_score_posterior_form (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ)
    (x : Fin n → ℝ) :
    (1 / ch05b_Delta t) • (Real.exp (-t) • ch05b_m S0 t x - x)
      = ((Real.exp (-2 * t) / (ch05b_Delta t) ^ 2) • (ch05b_J S0 t)⁻¹
          - (1 / ch05b_Delta t) • (1 : Matrix (Fin n) (Fin n) ℝ)) *ᵥ x := by
  have hc : 1 / ch05b_Delta t * Real.exp (-t) * (Real.exp (-t) / ch05b_Delta t)
      = Real.exp (-2 * t) / (ch05b_Delta t) ^ 2 := by
    rw [← ch05b_exp_sq]; ring
  rw [ch05b_noname_m_eq, Matrix.sub_mulVec, Matrix.smul_mulVec, Matrix.smul_mulVec,
    Matrix.one_mulVec, smul_sub, smul_smul, smul_smul, hc]

-- eq:g-two-score-routes (ch05 lines 1108-1115): `(e^{-t}E[a|x] - x)/Δ_t = -Σ_t⁻¹x`.
theorem ch05b_g_two_score_routes (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ) (x : Fin n → ℝ)
    (hS0 : IsUnit S0.det) (hSt : IsUnit (ch05b_Sigt S0 t).det) (hΔ : ch05b_Delta t ≠ 0) :
    (1 / ch05b_Delta t) • (Real.exp (-t) • ch05b_m S0 t x - x)
      = -((ch05b_Sigt S0 t)⁻¹ *ᵥ x) := by
  rw [ch05b_g_score_posterior_form, ch05b_noname_score_op S0 t hS0 hSt hΔ,
    Matrix.neg_mulVec, ch05b_Qt]

-- eq:g-tweedie (ch05 lines 818-825): componentwise,
-- `s_k(x,t) = (e^{-t}E[a_k|x] - x_k)/Δ_t`.
theorem ch05b_g_tweedie (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ) (x : Fin n → ℝ)
    (k : Fin n) (hS0 : IsUnit S0.det) (hSt : IsUnit (ch05b_Sigt S0 t).det)
    (hΔ : ch05b_Delta t ≠ 0) :
    ch05b_score S0 t x k
      = (Real.exp (-t) * ch05b_m S0 t x k - x k) / ch05b_Delta t := by
  have hk := congrFun (ch05b_g_two_score_routes S0 t x hS0 hSt hΔ) k
  simp only [Pi.smul_apply, Pi.sub_apply, Pi.neg_apply, smul_eq_mul, one_div] at hk
  simp only [ch05b_score, ch05b_Qt, Pi.neg_apply]
  rw [← hk]; ring

/-! ## The spectral form (tex 1123-1250) -/

-- eq:g-spectral (ch05 lines 1136-1140), rank-one form: `U diag(ω) Uᵀ = Σ ωᵢ uᵢuᵢᵀ`.
theorem ch05b_g_spectral_rank_one (U : Matrix (Fin n) (Fin n) ℝ) (w : Fin n → ℝ) :
    U * Matrix.diagonal w * Uᵀ
      = ∑ i, w i • Matrix.vecMulVec (fun k => U k i) (fun k => U k i) := by
  ext a b
  rw [Matrix.mul_apply]
  simp only [Matrix.mul_diagonal, Matrix.transpose_apply, Matrix.sum_apply, Matrix.smul_apply,
    Matrix.vecMulVec_apply, smul_eq_mul]
  exact Finset.sum_congr rfl fun i _ => by ring

-- eq:g-spectral (ch05 lines 1136-1140), existence: every real symmetric matrix has
-- an orthonormal eigenbasis.
theorem ch05b_g_spectral (M : Matrix (Fin n) (Fin n) ℝ) (hM : M.IsSymm) :
    ∃ (U : Matrix (Fin n) (Fin n) ℝ) (w : Fin n → ℝ),
      Uᵀ * U = 1 ∧ M = U * Matrix.diagonal w * Uᵀ
        ∧ M = ∑ i, w i • Matrix.vecMulVec (fun k => U k i) (fun k => U k i) := by
  have hH : M.IsHermitian := Matrix.isHermitian_iff_isSymm.mpr hM
  refine ⟨(hH.eigenvectorUnitary : Matrix (Fin n) (Fin n) ℝ), hH.eigenvalues, ?_, ?_, ?_⟩
  · have h := (hH.eigenvectorUnitary).2.1
    simpa [Matrix.star_eq_conjTranspose] using h
  · have h := hH.spectral_theorem
    rw [Unitary.conjStarAlgAut_apply] at h
    simpa [Matrix.star_eq_conjTranspose] using h
  · rw [← ch05b_g_spectral_rank_one]
    have h := hH.spectral_theorem
    rw [Unitary.conjStarAlgAut_apply] at h
    simpa [Matrix.star_eq_conjTranspose] using h

-- eq:g-affine (ch05 lines 1146-1150): `cM + dI = U diag(cωᵢ + d) Uᵀ`.
theorem ch05b_g_affine (U : Matrix (Fin n) (Fin n) ℝ) (w : Fin n → ℝ)
    (hU : Uᵀ * U = 1) (c d : ℝ) :
    c • (U * Matrix.diagonal w * Uᵀ) + d • (1 : Matrix (Fin n) (Fin n) ℝ)
      = U * Matrix.diagonal (fun i => c * w i + d) * Uᵀ := by
  have hd : IsUnit U.det := by
    have h := congrArg Matrix.det hU
    rw [Matrix.det_mul, Matrix.det_transpose, Matrix.det_one] at h
    exact isUnit_iff_exists_inv.mpr ⟨U.det, h⟩
  have hU' : U * Uᵀ = 1 := by
    rw [← Matrix.inv_eq_left_inv hU, Matrix.mul_nonsing_inv _ hd]
  have hD : Matrix.diagonal (fun i => c * w i + d)
      = c • Matrix.diagonal w + d • (1 : Matrix (Fin n) (Fin n) ℝ) := by
    ext i j
    by_cases h : i = j <;>
      simp [Matrix.diagonal_apply, Matrix.one_apply, h]
  rw [hD]
  simp only [Matrix.mul_add, Matrix.add_mul, Matrix.mul_smul, Matrix.smul_mul,
    Matrix.mul_one, hU']

-- eq:g-affine (ch05 lines 1151-1153), inverse clause.
theorem ch05b_g_affine_inv (U : Matrix (Fin n) (Fin n) ℝ) (v : Fin n → ℝ)
    (hU : Uᵀ * U = 1) (hv : ∀ i, v i ≠ 0) :
    (U * Matrix.diagonal v * Uᵀ)⁻¹ = U * Matrix.diagonal (fun i => (v i)⁻¹) * Uᵀ := by
  have hd : IsUnit U.det := by
    have h := congrArg Matrix.det hU
    rw [Matrix.det_mul, Matrix.det_transpose, Matrix.det_one] at h
    exact isUnit_iff_exists_inv.mpr ⟨U.det, h⟩
  have hU' : U * Uᵀ = 1 := by
    rw [← Matrix.inv_eq_left_inv hU, Matrix.mul_nonsing_inv _ hd]
  refine Matrix.inv_eq_right_inv ?_
  have hdiag : Matrix.diagonal v * Matrix.diagonal (fun i => (v i)⁻¹)
      = (1 : Matrix (Fin n) (Fin n) ℝ) := by
    rw [Matrix.diagonal_mul_diagonal]
    have hfun : (fun i => v i * (v i)⁻¹) = (fun _ : Fin n => (1 : ℝ)) :=
      funext fun i => mul_inv_cancel₀ (hv i)
    rw [hfun]
    exact Matrix.diagonal_one
  calc U * Matrix.diagonal v * Uᵀ * (U * Matrix.diagonal (fun i => (v i)⁻¹) * Uᵀ)
      = U * Matrix.diagonal v * (Uᵀ * U) * Matrix.diagonal (fun i => (v i)⁻¹) * Uᵀ := by
        simp only [Matrix.mul_assoc]
    _ = U * (Matrix.diagonal v * Matrix.diagonal (fun i => (v i)⁻¹)) * Uᵀ := by
        rw [hU]; simp only [Matrix.mul_one, Matrix.mul_assoc]
    _ = 1 := by rw [hdiag, Matrix.mul_one, hU']

/-- The spectral eigenvalue path `λᵢ(t) = e^{-2t}ωᵢ + Δ_t` (eq:g-Qt-spec). -/
def ch05b_lam (ω : ℝ) (t : ℝ) : ℝ := Real.exp (-2 * t) * ω + ch05b_Delta t

theorem ch05b_lam_pos {ω t : ℝ} (hω : 0 < ω) (ht : 0 ≤ t) : 0 < ch05b_lam ω t := by
  have h1 : 0 < Real.exp (-2 * t) * ω := mul_pos (Real.exp_pos _) hω
  have h2 : (0 : ℝ) ≤ ch05b_Delta t := by
    simp only [ch05b_Delta, sub_nonneg]
    exact Real.exp_le_one_iff.mpr (by linarith)
  simp only [ch05b_lam]; linarith

-- eq:g-Qt-spec (ch05 lines 1169-1178): `Σ_t = U diag(λᵢ(t)) Uᵀ`,
-- `Q_t = U diag(1/λᵢ(t)) Uᵀ`, `λᵢ(t) = e^{-2t}ωᵢ + Δ_t`.
theorem ch05b_g_Qt_spec (U : Matrix (Fin n) (Fin n) ℝ) (ω : Fin n → ℝ)
    (hU : Uᵀ * U = 1) (S0 : Matrix (Fin n) (Fin n) ℝ)
    (hS0 : S0 = U * Matrix.diagonal ω * Uᵀ) (t : ℝ) (ht : 0 ≤ t)
    (hω : ∀ i, 0 < ω i) :
    ch05b_Sigt S0 t = U * Matrix.diagonal (fun i => ch05b_lam (ω i) t) * Uᵀ ∧
      ch05b_Qt S0 t = U * Matrix.diagonal (fun i => (ch05b_lam (ω i) t)⁻¹) * Uᵀ := by
  have hSig : ch05b_Sigt S0 t = U * Matrix.diagonal (fun i => ch05b_lam (ω i) t) * Uᵀ := by
    rw [ch05b_Sigt, hS0]
    exact ch05b_g_affine U ω hU _ _
  refine ⟨hSig, ?_⟩
  rw [ch05b_Qt, hSig]
  exact ch05b_g_affine_inv U _ hU (fun i => ne_of_gt (ch05b_lam_pos (hω i) ht))

/-! ## Two computational corollaries (tex 1253-1515) -/

-- noname-15 (ch05 lines 1256-1260): `Σ_t = e^{-2t}Σ₀ + Δ_t I`.
theorem ch05b_noname_Sigt_restate (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ) :
    ch05b_Sigt S0 t
      = Real.exp (-2 * t) • S0 + ch05b_Delta t • (1 : Matrix (Fin n) (Fin n) ℝ) := rfl

-- noname-16 (ch05 lines 1268-1275): `Σ_t = Δ_t(I + γ_tΣ₀)`.
theorem ch05b_noname_Sigt_snr (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ)
    (hΔ : ch05b_Delta t ≠ 0) :
    ch05b_Sigt S0 t
      = ch05b_Delta t • ((1 : Matrix (Fin n) (Fin n) ℝ) + ch05b_gamma t • S0) := by
  rw [ch05b_Sigt, smul_add, smul_smul, ch05b_Delta_mul_gamma hΔ, add_comm]

-- eq:g-gamma-def (ch05 lines 1277-1282): `γ_t := e^{-2t}/Δ_t`.
theorem ch05b_g_gamma_def (t : ℝ) : ch05b_gamma t = Real.exp (-2 * t) / ch05b_Delta t := rfl

-- eq:g-Qt-snr (ch05 lines 1284-1292): `Q_t = (1/Δ_t)(I + γ_tΣ₀)⁻¹`.
theorem ch05b_g_Qt_snr (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ)
    (hΔ : ch05b_Delta t ≠ 0)
    (hB : IsUnit ((1 : Matrix (Fin n) (Fin n) ℝ) + ch05b_gamma t • S0).det) :
    ch05b_Qt S0 t
      = (1 / ch05b_Delta t) • ((1 : Matrix (Fin n) (Fin n) ℝ) + ch05b_gamma t • S0)⁻¹ := by
  rw [ch05b_Qt]
  refine Matrix.inv_eq_right_inv ?_
  rw [ch05b_noname_Sigt_snr S0 t hΔ, Matrix.smul_mul, Matrix.mul_smul, smul_smul,
    Matrix.mul_nonsing_inv _ hB]
  rw [show ch05b_Delta t * (1 / ch05b_Delta t) = 1 by field_simp, one_smul]

-- noname-17 (ch05 lines 1296-1300): `γ_t = e^{-2t}/(1 - e^{-2t})`.
theorem ch05b_noname_gamma_explicit (t : ℝ) :
    ch05b_gamma t = Real.exp (-2 * t) / (1 - Real.exp (-2 * t)) := rfl

private theorem ch05b_aux_gamma_c {A B : ℝ} (h1 : A * B = 1) (hA : 1 - A ≠ 0) :
    A / (1 - A) * (B - 1) = 1 := by
  field_simp
  linear_combination h1

/-- `γ_t c_t = 1`: the SNR is the reciprocal of `c_t = e^{2t} - 1`. -/
theorem ch05b_gamma_mul_c {t : ℝ} (ht : 0 < t) : ch05b_gamma t * ch05b_c t = 1 := by
  have h1 : Real.exp (-2 * t) * Real.exp (2 * t) = 1 := by
    rw [← Real.exp_add, show -2 * t + 2 * t = 0 by ring, Real.exp_zero]
  have hΔ : (1 : ℝ) - Real.exp (-2 * t) ≠ 0 := ch05b_Delta_ne ht
  show Real.exp (-2 * t) / (1 - Real.exp (-2 * t)) * (Real.exp (2 * t) - 1) = 1
  exact ch05b_aux_gamma_c h1 hΔ

theorem ch05b_gamma_pos {t : ℝ} (ht : 0 < t) : 0 < ch05b_gamma t := by
  have := ch05b_Delta_pos ht
  simp only [ch05b_gamma]
  exact div_pos (Real.exp_pos _) this

-- noname-18a (ch05 lines 1302-1308): `γ_t → ∞` as `t ↓ 0`, in ε-δ form.
theorem ch05b_noname_gamma_limit_zero (M : ℝ) (hM : 0 < M) :
    ∃ δ > 0, ∀ t, 0 < t → t < δ → M < ch05b_gamma t := by
  have hone : (1 : ℝ) < 1 + 1 / M := by
    have : 0 < 1 / M := by positivity
    linarith
  have hlog : 0 < Real.log (1 + 1 / M) := Real.log_pos hone
  refine ⟨Real.log (1 + 1 / M) / 2, by linarith, ?_⟩
  intro t ht hlt
  have hpos : (0 : ℝ) < 1 + 1 / M := by positivity
  have h2t : 2 * t < Real.log (1 + 1 / M) := by linarith
  have hexp : Real.exp (2 * t) < 1 + 1 / M := by
    calc Real.exp (2 * t) < Real.exp (Real.log (1 + 1 / M)) := Real.exp_lt_exp.mpr h2t
      _ = 1 + 1 / M := Real.exp_log hpos
  have hc : ch05b_c t < 1 / M := by simp only [ch05b_c]; linarith
  have hcpos := ch05b_c_pos ht
  have hg := ch05b_gamma_mul_c ht
  have hgpos := ch05b_gamma_pos ht
  have hMc : M * ch05b_c t < 1 := by
    have h := mul_lt_mul_of_pos_left hc hM
    rwa [mul_one_div, div_self (ne_of_gt hM)] at h
  nlinarith [hg, hcpos, hMc, hgpos]

-- noname-18b (ch05 lines 1302-1308): `γ_t → 0` as `t → ∞`, in ε-δ form.
theorem ch05b_noname_gamma_limit_infty (ε : ℝ) (hε : 0 < ε) :
    ∃ T, ∀ t, T < t → ch05b_gamma t < ε := by
  have hone : (1 : ℝ) < 1 + 1 / ε := by
    have : 0 < 1 / ε := by positivity
    linarith
  have hpos : (0 : ℝ) < 1 + 1 / ε := by positivity
  refine ⟨max 1 (Real.log (1 + 1 / ε) / 2), ?_⟩
  intro t ht
  have h1t : (1 : ℝ) < t := lt_of_le_of_lt (le_max_left _ _) ht
  have ht0 : 0 < t := by linarith
  have hlt : Real.log (1 + 1 / ε) / 2 < t := lt_of_le_of_lt (le_max_right _ _) ht
  have h2t : Real.log (1 + 1 / ε) < 2 * t := by linarith
  have hexp : 1 + 1 / ε < Real.exp (2 * t) := by
    calc (1 : ℝ) + 1 / ε = Real.exp (Real.log (1 + 1 / ε)) := (Real.exp_log hpos).symm
      _ < Real.exp (2 * t) := Real.exp_lt_exp.mpr h2t
  have hc : 1 / ε < ch05b_c t := by simp only [ch05b_c]; linarith
  have hcpos := ch05b_c_pos ht0
  have hg := ch05b_gamma_mul_c ht0
  have hgpos := ch05b_gamma_pos ht0
  have hεc : 1 < ε * ch05b_c t := by
    have h := mul_lt_mul_of_pos_left hc hε
    rwa [mul_one_div, div_self (ne_of_gt hε)] at h
  nlinarith [hg, hcpos, hεc, hgpos]

-- eq:g-Qt-smallt (ch05 lines 1312-1322): the exact content of the small-`t`
-- approximation chain, `(1/Δ_t)(γ_tΣ₀)⁻¹ = (1/(Δ_tγ_t))Q₀ = e^{2t}Q₀`.
theorem ch05b_g_Qt_smallt (S0 : Matrix (Fin n) (Fin n) ℝ) {t : ℝ} (ht : 0 < t)
    (hS0 : IsUnit S0.det) :
    (1 / ch05b_Delta t) • (ch05b_gamma t • S0)⁻¹
      = (1 / (ch05b_Delta t * ch05b_gamma t)) • S0⁻¹
      ∧ (1 / (ch05b_Delta t * ch05b_gamma t)) • S0⁻¹
        = Real.exp (2 * t) • S0⁻¹ := by
  have hΔ := ch05b_Delta_ne ht
  have hexp : Real.exp (-2 * t) ≠ 0 := Real.exp_ne_zero _
  have hγ : ch05b_gamma t ≠ 0 := by
    simp only [ch05b_gamma]; exact div_ne_zero hexp hΔ
  have hinv : (ch05b_gamma t • S0)⁻¹ = (ch05b_gamma t)⁻¹ • S0⁻¹ := by
    refine Matrix.inv_eq_right_inv ?_
    rw [Matrix.smul_mul, Matrix.mul_smul, smul_smul, mul_inv_cancel₀ hγ, one_smul,
      Matrix.mul_nonsing_inv _ hS0]
  constructor
  · rw [hinv, smul_smul]; congr 1; field_simp
  · congr 1
    rw [ch05b_Delta_mul_gamma hΔ]
    rw [show (1 : ℝ) / Real.exp (-2 * t) = Real.exp (2 * t) by
      rw [div_eq_iff hexp, ← Real.exp_add, show 2 * t + -2 * t = 0 by ring, Real.exp_zero]]

-- eq:g-Qt-larget (ch05 lines 1327-1332): the exact remainder behind `Q_t ≈ (1/Δ_t)I`.
theorem ch05b_g_Qt_larget (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ)
    (hΔ : ch05b_Delta t ≠ 0)
    (hB : IsUnit ((1 : Matrix (Fin n) (Fin n) ℝ) + ch05b_gamma t • S0).det) :
    ch05b_Qt S0 t - (1 / ch05b_Delta t) • (1 : Matrix (Fin n) (Fin n) ℝ)
      = -((ch05b_gamma t / ch05b_Delta t) •
          (S0 * ((1 : Matrix (Fin n) (Fin n) ℝ) + ch05b_gamma t • S0)⁻¹)) := by
  set B : Matrix (Fin n) (Fin n) ℝ := (1 : Matrix (Fin n) (Fin n) ℝ) + ch05b_gamma t • S0 with hBdef
  have hBinv : B * B⁻¹ = 1 := Matrix.mul_nonsing_inv _ hB
  have key : B⁻¹ - 1 = -(ch05b_gamma t • (S0 * B⁻¹)) := by
    have h1 : (1 : Matrix (Fin n) (Fin n) ℝ) - B * B⁻¹
        = (1 : Matrix (Fin n) (Fin n) ℝ) - B⁻¹ - ch05b_gamma t • (S0 * B⁻¹) := by
      rw [hBdef, Matrix.add_mul, Matrix.one_mul, Matrix.smul_mul]; abel
    rw [hBinv] at h1
    have h2 : (0 : Matrix (Fin n) (Fin n) ℝ)
        = 1 - B⁻¹ - ch05b_gamma t • (S0 * B⁻¹) := by simpa using h1
    have := h2.symm
    linear_combination (norm := abel) -this
  rw [ch05b_g_Qt_snr S0 t hΔ hB, ← hBdef, ← smul_sub, key, smul_neg, smul_smul]
  congr 2
  field_simp

-- noname-19 (ch05 lines 1340-1347): `Σ_t = e^{-2t}(Σ₀ + e^{2t}Δ_t I)`.
theorem ch05b_noname_Sigt_res (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ) :
    ch05b_Sigt S0 t
      = Real.exp (-2 * t) • (S0 + (Real.exp (2 * t) * ch05b_Delta t)
          • (1 : Matrix (Fin n) (Fin n) ℝ)) := by
  have h1 : Real.exp (-2 * t) * (Real.exp (2 * t) * ch05b_Delta t) = ch05b_Delta t := by
    rw [← mul_assoc, ← Real.exp_add]; simp
  rw [ch05b_Sigt, smul_add, smul_smul, h1]

-- noname-20 (ch05 lines 1349-1355): `e^{2t}Δ_t = e^{2t}(1 - e^{-2t}) = e^{2t} - 1`.
theorem ch05b_noname_c_eq (t : ℝ) :
    Real.exp (2 * t) * ch05b_Delta t = Real.exp (2 * t) - 1 := by
  have h1 : Real.exp (2 * t) * Real.exp (-2 * t) = 1 := by rw [← Real.exp_add]; simp
  simp only [ch05b_Delta]
  nlinarith [h1]

-- eq:g-ct-def (ch05 lines 1357-1360): `c_t := e^{2t} - 1`.
theorem ch05b_g_ct_def (t : ℝ) : ch05b_c t = Real.exp (2 * t) - 1 := rfl

-- noname-21 (ch05 lines 1362-1367): `Σ_t = e^{-2t}(Σ₀ + c_t I)`.
theorem ch05b_noname_Sigt_ct (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ) :
    ch05b_Sigt S0 t
      = Real.exp (-2 * t) • (S0 + ch05b_c t • (1 : Matrix (Fin n) (Fin n) ℝ)) := by
  rw [ch05b_noname_Sigt_res, ch05b_noname_c_eq, ch05b_c]

-- eq:g-res-step1 (ch05 lines 1369-1375): `Q_t = e^{2t}(Σ₀ + c_t I)⁻¹`.
theorem ch05b_g_res_step1 (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ)
    (hA : IsUnit (S0 + ch05b_c t • (1 : Matrix (Fin n) (Fin n) ℝ)).det) :
    ch05b_Qt S0 t
      = Real.exp (2 * t) • (S0 + ch05b_c t • (1 : Matrix (Fin n) (Fin n) ℝ))⁻¹ := by
  rw [ch05b_Qt]
  refine Matrix.inv_eq_right_inv ?_
  rw [ch05b_noname_Sigt_ct, Matrix.smul_mul, Matrix.mul_smul, smul_smul,
    Matrix.mul_nonsing_inv _ hA, ← Real.exp_add]
  simp

-- noname-22 (ch05 lines 1377-1384): `Σ₀ + c_t I = Q₀⁻¹(I + c_tQ₀)` with `Q₀ = Σ₀⁻¹`.
theorem ch05b_noname_resolvent_factor (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ)
    (hS0 : IsUnit S0.det) :
    S0 + ch05b_c t • (1 : Matrix (Fin n) (Fin n) ℝ)
      = S0 * ((1 : Matrix (Fin n) (Fin n) ℝ) + ch05b_c t • S0⁻¹) := by
  rw [Matrix.mul_add, Matrix.mul_one, Matrix.mul_smul, Matrix.mul_nonsing_inv _ hS0]

-- eq:g-Qt-resolvent (ch05 lines 1386-1395): `Q_t = e^{2t}Q₀(I + c_tQ₀)⁻¹`.
theorem ch05b_g_Qt_resolvent (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ)
    (hS0 : IsUnit S0.det)
    (hB : IsUnit ((1 : Matrix (Fin n) (Fin n) ℝ) + ch05b_c t • S0⁻¹).det) :
    ch05b_Qt S0 t
      = Real.exp (2 * t) • (S0⁻¹ * ((1 : Matrix (Fin n) (Fin n) ℝ) + ch05b_c t • S0⁻¹)⁻¹) := by
  set B : Matrix (Fin n) (Fin n) ℝ := (1 : Matrix (Fin n) (Fin n) ℝ) + ch05b_c t • S0⁻¹ with hBdef
  have hcomm : S0 * B = B * S0 := by
    rw [hBdef, Matrix.mul_add, Matrix.add_mul, Matrix.mul_one, Matrix.one_mul,
      Matrix.mul_smul, Matrix.smul_mul, Matrix.mul_nonsing_inv _ hS0,
      Matrix.nonsing_inv_mul _ hS0]
  have hcomm' : S0⁻¹ * B = B * S0⁻¹ := by
    calc S0⁻¹ * B = S0⁻¹ * (B * S0) * S0⁻¹ := by
          rw [Matrix.mul_assoc, Matrix.mul_assoc, Matrix.mul_nonsing_inv _ hS0, Matrix.mul_one]
      _ = S0⁻¹ * (S0 * B) * S0⁻¹ := by rw [hcomm]
      _ = B * S0⁻¹ := by
          rw [← Matrix.mul_assoc, Matrix.nonsing_inv_mul _ hS0, Matrix.one_mul]
  rw [ch05b_Qt]
  refine Matrix.inv_eq_right_inv ?_
  rw [ch05b_noname_Sigt_ct, ch05b_noname_resolvent_factor S0 t hS0, ← hBdef,
    Matrix.smul_mul, Matrix.mul_smul, smul_smul, ← Real.exp_add]
  have : S0 * B * (S0⁻¹ * B⁻¹) = 1 := by
    calc S0 * B * (S0⁻¹ * B⁻¹) = S0 * (B * S0⁻¹) * B⁻¹ := by
          simp only [Matrix.mul_assoc]
      _ = S0 * (S0⁻¹ * B) * B⁻¹ := by rw [hcomm']
      _ = (S0 * S0⁻¹) * (B * B⁻¹) := by simp only [Matrix.mul_assoc]
      _ = 1 := by
          rw [Matrix.mul_nonsing_inv _ hS0, Matrix.mul_nonsing_inv _ hB, Matrix.mul_one]
  rw [this, show -2 * t + 2 * t = 0 by ring, Real.exp_zero, one_smul]

-- ch05 lines 1396-1397: `Q₀` commutes with `(I + c_tQ₀)⁻¹`, so the order of the
-- two factors in eq:g-Qt-resolvent is immaterial.
theorem ch05b_noname_resolvent_comm (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ)
    (hB : IsUnit ((1 : Matrix (Fin n) (Fin n) ℝ) + ch05b_c t • S0⁻¹).det) :
    S0⁻¹ * ((1 : Matrix (Fin n) (Fin n) ℝ) + ch05b_c t • S0⁻¹)⁻¹
      = ((1 : Matrix (Fin n) (Fin n) ℝ) + ch05b_c t • S0⁻¹)⁻¹ * S0⁻¹ := by
  set B : Matrix (Fin n) (Fin n) ℝ := (1 : Matrix (Fin n) (Fin n) ℝ) + ch05b_c t • S0⁻¹ with hBdef
  have hcomm : S0⁻¹ * B = B * S0⁻¹ := by
    rw [hBdef, Matrix.mul_add, Matrix.add_mul, Matrix.mul_one, Matrix.one_mul,
      Matrix.mul_smul, Matrix.smul_mul]
  calc S0⁻¹ * B⁻¹ = B⁻¹ * (B * S0⁻¹) * B⁻¹ := by
        rw [← Matrix.mul_assoc, Matrix.nonsing_inv_mul _ hB, Matrix.one_mul]
    _ = B⁻¹ * (S0⁻¹ * B) * B⁻¹ := by rw [hcomm]
    _ = B⁻¹ * S0⁻¹ := by
        rw [Matrix.mul_assoc, Matrix.mul_assoc, Matrix.mul_nonsing_inv _ hB, Matrix.mul_one]

/-! ### The Neumann series (tex 1399-1462) -/

-- noname-23 (ch05 lines 1402-1407): the scalar geometric series `1/(1+z) = Σ(-z)^m`.
theorem ch05b_noname_geometric (z : ℝ) (hz : |z| < 1) :
    ∑' m : ℕ, (-z) ^ m = 1 / (1 + z) := by
  have h : ‖(-z : ℝ)‖ < 1 := by rw [Real.norm_eq_abs, abs_neg]; exact hz
  rw [tsum_geometric_of_norm_lt_one h, one_div]
  congr 1
  ring

/-- Truncated geometric identity for matrices: `(I - X)(I + X + ⋯ + X^{N-1}) = I - X^N`. -/
theorem ch05b_geom_trunc (X : Matrix (Fin n) (Fin n) ℝ) (N : ℕ) :
    ((1 : Matrix (Fin n) (Fin n) ℝ) - X) * (∑ m ∈ Finset.range N, X ^ m)
      = 1 - X ^ N := by
  induction N with
  | zero => simp
  | succ k ih =>
      have hstep : ((1 : Matrix (Fin n) (Fin n) ℝ) - X) * X ^ k = X ^ k - X ^ (k + 1) := by
        rw [Matrix.sub_mul, Matrix.one_mul, ← pow_succ']
      rw [Finset.sum_range_succ, Matrix.mul_add, ih, hstep]
      abel

-- eq:g-neumann-general (ch05 lines 1409-1416): the exact finite form of the Neumann
-- series, `(I + A)(Σ_{m<N}(-A)^m) = I - (-A)^N`.
theorem ch05b_g_neumann_general (A : Matrix (Fin n) (Fin n) ℝ) (N : ℕ) :
    ((1 : Matrix (Fin n) (Fin n) ℝ) + A) * (∑ m ∈ Finset.range N, (-A) ^ m)
      = 1 - (-A) ^ N := by
  have h : (1 : Matrix (Fin n) (Fin n) ℝ) + A = 1 - (-A) := by abel
  rw [h]
  exact ch05b_geom_trunc (-A) N

/-- Neumann expansion of the inverse with explicit remainder. -/
theorem ch05b_g_neumann_general_inv (A : Matrix (Fin n) (Fin n) ℝ) (N : ℕ)
    (hA : IsUnit ((1 : Matrix (Fin n) (Fin n) ℝ) + A).det) :
    ((1 : Matrix (Fin n) (Fin n) ℝ) + A)⁻¹
      = (∑ m ∈ Finset.range N, (-A) ^ m)
        + ((1 : Matrix (Fin n) (Fin n) ℝ) + A)⁻¹ * (-A) ^ N := by
  have hBS := ch05b_g_neumann_general A N
  have h := congrArg (fun X => ((1 : Matrix (Fin n) (Fin n) ℝ) + A)⁻¹ * X) hBS
  rw [← Matrix.mul_assoc, Matrix.nonsing_inv_mul _ hA, Matrix.one_mul, Matrix.mul_sub,
    Matrix.mul_one] at h
  rw [h]
  abel

theorem ch05b_neg_smul_pow (c : ℝ) (M : Matrix (Fin n) (Fin n) ℝ) (m : ℕ) :
    (-(c • M)) ^ m = (-c) ^ m • M ^ m := by
  rw [← neg_smul, smul_pow]

-- noname-24 (ch05 lines 1419-1427): the Neumann series applied to `A = c_t Q₀`.
theorem ch05b_noname_neumann_apply (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ) (N : ℕ)
    (hB : IsUnit ((1 : Matrix (Fin n) (Fin n) ℝ) + ch05b_c t • S0⁻¹).det) :
    ((1 : Matrix (Fin n) (Fin n) ℝ) + ch05b_c t • S0⁻¹)⁻¹
      = (∑ m ∈ Finset.range N, (-(ch05b_c t)) ^ m • S0⁻¹ ^ m)
        + ((1 : Matrix (Fin n) (Fin n) ℝ) + ch05b_c t • S0⁻¹)⁻¹
            * ((-(ch05b_c t)) ^ N • S0⁻¹ ^ N) := by
  have h := ch05b_g_neumann_general_inv (ch05b_c t • S0⁻¹) N hB
  simpa only [ch05b_neg_smul_pow] using h

-- eq:g-Qt-res (ch05 lines 1429-1455) and noname-27 (ch05 lines 1485-1495): the
-- resolvent expansion of `Q_t` with explicit remainder after `N` terms.
theorem ch05b_g_Qt_res (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ) (N : ℕ)
    (hS0 : IsUnit S0.det)
    (hB : IsUnit ((1 : Matrix (Fin n) (Fin n) ℝ) + ch05b_c t • S0⁻¹).det) :
    ch05b_Qt S0 t
      = Real.exp (2 * t) • ((∑ m ∈ Finset.range N, (-(ch05b_c t)) ^ m • S0⁻¹ ^ (m + 1))
          + (-(ch05b_c t)) ^ N
              • (S0⁻¹ * ((1 : Matrix (Fin n) (Fin n) ℝ) + ch05b_c t • S0⁻¹)⁻¹ * S0⁻¹ ^ N)) := by
  have hres := ch05b_g_Qt_resolvent S0 t hS0 hB
  have hneu := ch05b_noname_neumann_apply S0 t N hB
  have hQsum : S0⁻¹ * (∑ m ∈ Finset.range N, (-(ch05b_c t)) ^ m • S0⁻¹ ^ m)
      = ∑ m ∈ Finset.range N, (-(ch05b_c t)) ^ m • S0⁻¹ ^ (m + 1) := by
    rw [Matrix.mul_sum]
    refine Finset.sum_congr rfl fun m _ => ?_
    rw [Matrix.mul_smul, ← pow_succ']
  rw [hres]
  congr 1
  conv_lhs => rw [hneu]
  rw [Matrix.mul_add, hQsum]
  simp only [Matrix.mul_smul, Matrix.mul_assoc]

/-! ### Bandwidth of powers (tex 1465-1515) -/

/-- `A` has bandwidth `p`: `A i j = 0` whenever `|i - j| > p`. -/
def ch05b_Banded (A : Matrix (Fin n) (Fin n) ℝ) (p : ℕ) : Prop :=
  ∀ i j : Fin n, p < Int.natAbs ((i : ℤ) - (j : ℤ)) → A i j = 0

theorem ch05b_banded_one : ch05b_Banded (1 : Matrix (Fin n) (Fin n) ℝ) 0 := by
  intro i j hij
  have hne : i ≠ j := by
    intro h
    subst h
    simp at hij
  exact Matrix.one_apply_ne hne

theorem ch05b_banded_mul {A B : Matrix (Fin n) (Fin n) ℝ} {p q : ℕ}
    (hA : ch05b_Banded A p) (hB : ch05b_Banded B q) : ch05b_Banded (A * B) (p + q) := by
  intro i j hij
  rw [Matrix.mul_apply]
  refine Finset.sum_eq_zero ?_
  intro k _
  by_cases hk : p < Int.natAbs ((i : ℤ) - (k : ℤ))
  · rw [hA i k hk, zero_mul]
  · have hk2 : q < Int.natAbs ((k : ℤ) - (j : ℤ)) := by omega
    rw [hB k j hk2, mul_zero]

-- eq:g-power-bandwidth (ch05 lines 1478-1483): `(Q₀^m)_{ij} = 0` when `|i-j| > m`,
-- for tridiagonal `Q₀`.
theorem ch05b_g_power_bandwidth {A : Matrix (Fin n) (Fin n) ℝ} (hA : ch05b_Banded A 1)
    (m : ℕ) : ch05b_Banded (A ^ m) m := by
  induction m with
  | zero => simpa using ch05b_banded_one
  | succ k ih =>
      rw [pow_succ]
      have h := ch05b_banded_mul ih hA
      exact h

-- noname-25 (ch05 lines 1468-1471): `Q₀` couples frames at distance at most 1 --
-- the tridiagonality hypothesis, checked on the AR(1) instance `α = 1/2`, `L = 3`.
theorem ch05b_noname_Q0_banded_instance : ch05b_Banded ((ch05b_Sig0 (1/2 : ℝ) 3)⁻¹) 1 := by
  have hS : (ch05b_Sig0 (1/2 : ℝ) 3)⁻¹
      = !![(4/3 : ℝ), -2/3, 0; -2/3, 5/3, -2/3; 0, -2/3, 4/3] := by
    refine Matrix.inv_eq_right_inv ?_
    ext i j
    fin_cases i <;> fin_cases j <;>
      simp [ch05b_Sig0, Matrix.mul_apply, Fin.sum_univ_three, Matrix.one_apply] <;> norm_num
  intro i j hij
  rw [hS]
  fin_cases i <;> fin_cases j <;> simp_all <;> omega

-- noname-26 (ch05 lines 1473-1476): `Q₀²` couples frames at distance at most 2.
theorem ch05b_noname_Q0_sq_banded {A : Matrix (Fin n) (Fin n) ℝ} (hA : ch05b_Banded A 1) :
    ch05b_Banded (A ^ 2) 2 := ch05b_g_power_bandwidth hA 2

-- eq:g-band-fill-order (ch05 lines 1504-1512): at frame distance `d ≥ 1` the whole
-- entry carries the factor `(-c_t)^{d-1}`, so `(Q_t)_{ij} = O(c_t^{d-1}) = O(t^{d-1})`.
theorem ch05b_g_band_fill_order (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ) (d : ℕ)
    (hd : 1 ≤ d) (i j : Fin n) (hij : Int.natAbs ((i : ℤ) - (j : ℤ)) = d)
    (hQ : ch05b_Banded S0⁻¹ 1) (hS0 : IsUnit S0.det)
    (hB : IsUnit ((1 : Matrix (Fin n) (Fin n) ℝ) + ch05b_c t • S0⁻¹).det) :
    ch05b_Qt S0 t i j
      = (-(ch05b_c t)) ^ (d - 1) * (Real.exp (2 * t)
          * (S0⁻¹ * ((1 : Matrix (Fin n) (Fin n) ℝ) + ch05b_c t • S0⁻¹)⁻¹
              * S0⁻¹ ^ (d - 1)) i j) := by
  have hres := ch05b_g_Qt_res S0 t (d - 1) hS0 hB
  have hsum : (∑ m ∈ Finset.range (d - 1), (-(ch05b_c t)) ^ m • S0⁻¹ ^ (m + 1)) i j = 0 := by
    rw [Matrix.sum_apply]
    refine Finset.sum_eq_zero ?_
    intro m hm
    rw [Matrix.smul_apply, smul_eq_mul]
    have hb : ch05b_Banded (S0⁻¹ ^ (m + 1)) (m + 1) := ch05b_g_power_bandwidth hQ (m + 1)
    have hlt : m + 1 < Int.natAbs ((i : ℤ) - (j : ℤ)) := by
      rw [hij]
      simp only [Finset.mem_range] at hm
      omega
    rw [hb i j hlt, mul_zero]
  rw [hres]
  simp only [Matrix.smul_apply, Matrix.add_apply, smul_eq_mul, hsum, zero_add]
  ring

-- eq:g-bandfill (ch05 lines 1584-1591): at frame distance `d`, the leading term is
-- `e^{2t}(-c_t)^{d-1}(Q₀^d)_{i,i+d}` and the remainder carries `(-c_t)^d`.
-- Since `c_t = e^{2t} - 1` satisfies `2t ≤ c_t ≤ 4t` for `0 ≤ t ≤ 1/4`
-- (`ch05b_c_bounds`), this is exactly `(-1)^{d-1}(2t)^{d-1}(Q₀^d)_{i,i+d} + O(t^d)`.
theorem ch05b_g_bandfill (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ) (d : ℕ)
    (hd : 1 ≤ d) (i j : Fin n) (hij : Int.natAbs ((i : ℤ) - (j : ℤ)) = d)
    (hQ : ch05b_Banded S0⁻¹ 1) (hS0 : IsUnit S0.det)
    (hB : IsUnit ((1 : Matrix (Fin n) (Fin n) ℝ) + ch05b_c t • S0⁻¹).det) :
    ch05b_Qt S0 t i j
      = Real.exp (2 * t) * ((-(ch05b_c t)) ^ (d - 1) * (S0⁻¹ ^ d) i j)
        + Real.exp (2 * t) * ((-(ch05b_c t)) ^ d
            * (S0⁻¹ * ((1 : Matrix (Fin n) (Fin n) ℝ) + ch05b_c t • S0⁻¹)⁻¹
                * S0⁻¹ ^ d) i j) := by
  have hdd : d - 1 + 1 = d := by omega
  have hres := ch05b_g_Qt_res S0 t d hS0 hB
  have hsum : (∑ m ∈ Finset.range d, (-(ch05b_c t)) ^ m • S0⁻¹ ^ (m + 1)) i j
      = (-(ch05b_c t)) ^ (d - 1) * (S0⁻¹ ^ d) i j := by
    rw [Matrix.sum_apply, Finset.sum_eq_single (d - 1)]
    · rw [Matrix.smul_apply, smul_eq_mul, hdd]
    · intro m hm hne
      rw [Matrix.smul_apply, smul_eq_mul]
      have hb : ch05b_Banded (S0⁻¹ ^ (m + 1)) (m + 1) := ch05b_g_power_bandwidth hQ (m + 1)
      have hlt : m + 1 < Int.natAbs ((i : ℤ) - (j : ℤ)) := by
        rw [hij]
        simp only [Finset.mem_range] at hm
        omega
      rw [hb i j hlt, mul_zero]
    · intro hnm
      exact absurd (Finset.mem_range.mpr (by omega)) hnm
  rw [hres]
  simp only [Matrix.smul_apply, Matrix.add_apply, smul_eq_mul, hsum]
  ring

-- noname-28 (ch05 lines 1597-1600): `c_t = 2t + O(t²)`, in the two-sided form
-- `2t ≤ c_t ≤ 4t` on `0 ≤ t ≤ 1/4`.
theorem ch05b_c_bounds {t : ℝ} (ht : 0 ≤ t) (ht4 : t ≤ 1/4) :
    2 * t ≤ ch05b_c t ∧ ch05b_c t ≤ 4 * t := by
  constructor
  · have h := Real.add_one_le_exp (2 * t)
    simp only [ch05b_c]; linarith
  · have h := Real.add_one_le_exp (-(2 * t))
    have hexp : Real.exp (-(2 * t)) * Real.exp (2 * t) = 1 := by
      rw [← Real.exp_add]; simp
    have hpos : 0 < Real.exp (2 * t) := Real.exp_pos _
    simp only [ch05b_c]
    nlinarith [h, hexp, hpos]

-- eq:g-neumann-condition (ch05 lines 1458-1461): in each eigen-coordinate the
-- expansion converges exactly under `c_t ω < 1`, and `c_t λ_max(Q₀) < 1` is
-- equivalent to that condition holding for every eigenvalue.
theorem ch05b_g_neumann_condition (c q : ℝ) (hq : 0 < q) (hc : 0 < c) (h : c * q < 1) :
    ∑' m : ℕ, (-(c * q)) ^ m = 1 / (1 + c * q) := by
  refine ch05b_noname_geometric _ ?_
  rw [abs_lt]
  constructor <;> nlinarith

theorem ch05b_g_neumann_condition_spec (c : ℝ) (hc : 0 < c) (ω : Fin n → ℝ)
    (hω : ∀ i, 0 < ω i) (i0 : Fin n) (hmax : ∀ i, ω i ≤ ω i0) :
    c * ω i0 < 1 ↔ ∀ i, |c * ω i| < 1 := by
  constructor
  · intro h i
    rw [abs_lt]
    have h1 : 0 < c * ω i := mul_pos hc (hω i)
    have h2 : c * ω i ≤ c * ω i0 := by nlinarith [hmax i]
    constructor <;> linarith
  · intro h
    have := h i0
    rw [abs_lt] at this
    exact this.2

/-! ### Losing tridiagonality (tex 1518-1607) -/

-- eq:g-Qt-inverse-form (ch05 lines 1549-1554):
-- `Q_t = (e^{2t}/c_t)[(I + c_tQ₀) - I](I + c_tQ₀)⁻¹ = (e^{2t}/c_t)[I - (I + c_tQ₀)⁻¹]`.
theorem ch05b_g_Qt_inverse_form (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ)
    (hct : ch05b_c t ≠ 0) (hS0 : IsUnit S0.det)
    (hB : IsUnit ((1 : Matrix (Fin n) (Fin n) ℝ) + ch05b_c t • S0⁻¹).det) :
    ch05b_Qt S0 t
      = (Real.exp (2 * t) / ch05b_c t)
        • ((1 : Matrix (Fin n) (Fin n) ℝ)
            - ((1 : Matrix (Fin n) (Fin n) ℝ) + ch05b_c t • S0⁻¹)⁻¹) := by
  set B : Matrix (Fin n) (Fin n) ℝ := (1 : Matrix (Fin n) (Fin n) ℝ) + ch05b_c t • S0⁻¹ with hBdef
  have hsub : ch05b_c t • S0⁻¹ = B - 1 := by rw [hBdef]; abel
  have hkey : ch05b_c t • (S0⁻¹ * B⁻¹) = 1 - B⁻¹ := by
    rw [← Matrix.smul_mul, hsub, Matrix.sub_mul, Matrix.one_mul,
      Matrix.mul_nonsing_inv _ hB]
  rw [ch05b_g_Qt_resolvent S0 t hS0 hB, ← hBdef, ← hkey, smul_smul]
  congr 1
  field_simp

-- eq:g-tridiag-inverse (ch05 lines 1560-1566): the explicit inverse of a symmetric
-- tridiagonal matrix, `(B⁻¹)_{ij} = (-1)^{i+j} θ_{i-1}φ_{j+1}/θ_{L-1} Π_{k=i}^{j-1} b_k`
-- (0-based: `θ_m = det B[0..m]`, `θ_{-1} = 1`; `φ_m = det B[m..L-1]`, `φ_L = 1`).
theorem ch05b_g_tridiag_inverse_two (a0 a1 b0 : ℝ) (hdet : a0 * a1 - b0 ^ 2 ≠ 0) :
    (!![a0, b0; b0, a1])⁻¹ 0 0 = (-1 : ℝ) ^ (0 + 0) * (1 * a1) / (a0 * a1 - b0 ^ 2) ∧
      (!![a0, b0; b0, a1])⁻¹ 0 1
        = (-1 : ℝ) ^ (0 + 1) * (1 * 1) / (a0 * a1 - b0 ^ 2) * b0 ∧
      (!![a0, b0; b0, a1])⁻¹ 1 1
        = (-1 : ℝ) ^ (1 + 1) * (a0 * 1) / (a0 * a1 - b0 ^ 2) := by
  have hadj : (!![a0, b0; b0, a1]) * !![a1, -b0; -b0, a0]
      = (a0 * a1 - b0 ^ 2) • (1 : Matrix (Fin 2) (Fin 2) ℝ) := by
    ext i j
    fin_cases i <;> fin_cases j <;>
      simp [Matrix.mul_apply, Fin.sum_univ_two, Matrix.one_apply] <;> ring
  have hinv : (!![a0, b0; b0, a1])⁻¹
      = (a0 * a1 - b0 ^ 2)⁻¹ • !![a1, -b0; -b0, a0] := by
    refine Matrix.inv_eq_right_inv ?_
    rw [Matrix.mul_smul, hadj, smul_smul, inv_mul_cancel₀ hdet, one_smul]
  refine ⟨?_, ?_, ?_⟩ <;>
    · rw [hinv]
      simp only [Matrix.smul_apply, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
        Matrix.head_fin_const, smul_eq_mul, Matrix.of_apply]
      field_simp
      ring

theorem ch05b_g_tridiag_inverse_three (a0 a1 a2 b0 b1 : ℝ)
    (hdet : a0 * a1 * a2 - a0 * b1 ^ 2 - b0 ^ 2 * a2 ≠ 0) :
    (!![a0, b0, 0; b0, a1, b1; 0, b1, a2])⁻¹
      = (a0 * a1 * a2 - a0 * b1 ^ 2 - b0 ^ 2 * a2)⁻¹ •
        !![a1 * a2 - b1 ^ 2, -(a2 * b0), b0 * b1;
           -(a2 * b0), a0 * a2, -(a0 * b1);
           b0 * b1, -(a0 * b1), a0 * a1 - b0 ^ 2] := by
  have hadj : (!![a0, b0, 0; b0, a1, b1; 0, b1, a2]) *
      !![a1 * a2 - b1 ^ 2, -(a2 * b0), b0 * b1;
         -(a2 * b0), a0 * a2, -(a0 * b1);
         b0 * b1, -(a0 * b1), a0 * a1 - b0 ^ 2]
      = (a0 * a1 * a2 - a0 * b1 ^ 2 - b0 ^ 2 * a2) • (1 : Matrix (Fin 3) (Fin 3) ℝ) := by
    ext i j
    fin_cases i <;> fin_cases j <;>
      simp [Matrix.mul_apply, Fin.sum_univ_three, Matrix.one_apply] <;> ring
  refine Matrix.inv_eq_right_inv ?_
  rw [Matrix.mul_smul, hadj, smul_smul, inv_mul_cancel₀ hdet, one_smul]

/-! ### Frobenius bookkeeping (tex 1610-1662) -/

-- eq:g-frob (ch05 lines 1612-1616): `‖A‖_F² = Σ A_{ij}² = tr(AᵀA)`.
def ch05b_frobSq (A : Matrix (Fin n) (Fin n) ℝ) : ℝ := ∑ i, ∑ j, (A i j) ^ 2

theorem ch05b_g_frob (A : Matrix (Fin n) (Fin n) ℝ) :
    ch05b_frobSq A = Matrix.trace (Aᵀ * A) := by
  have h : Matrix.trace (Aᵀ * A) = ∑ j : Fin n, ∑ i : Fin n, A i j * A i j := by
    simp [Matrix.trace, Matrix.diag, Matrix.mul_apply, Matrix.transpose_apply]
  rw [h]
  simp only [ch05b_frobSq, sq]
  exact Finset.sum_comm

theorem ch05b_g_frob_norm (A : Matrix (Fin n) (Fin n) ℝ) :
    Real.sqrt (∑ i, ∑ j, (A i j) ^ 2) = Real.sqrt (Matrix.trace (Aᵀ * A)) := by
  show Real.sqrt (ch05b_frobSq A) = _
  rw [ch05b_g_frob]

/-- Diagonal part of a matrix. -/
def ch05b_diagPart (A : Matrix (Fin n) (Fin n) ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  fun i j => if i = j then A i j else 0

-- eq:g-bandproj (ch05 lines 1625-1628): `(ΠA)_{ij} = A_{ij} 1{|i-j| ≤ 1}`.
def ch05b_Pi (A : Matrix (Fin n) (Fin n) ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  fun i j => if Int.natAbs ((i : ℤ) - (j : ℤ)) ≤ 1 then A i j else 0

theorem ch05b_g_bandproj (A : Matrix (Fin n) (Fin n) ℝ) (i j : Fin n) :
    ch05b_Pi A i j = if Int.natAbs ((i : ℤ) - (j : ℤ)) ≤ 1 then A i j else 0 := rfl

theorem ch05b_Pi_banded (A : Matrix (Fin n) (Fin n) ℝ) : ch05b_Banded (ch05b_Pi A) 1 := by
  intro i j hij
  simp only [ch05b_Pi]
  rw [if_neg (by omega)]

-- eq:g-frob-split (ch05 lines 1633-1639): `‖A‖_F²` splits orthogonally into
-- diagonal, nearest-neighbour band, and off-tridiagonal parts.
theorem ch05b_g_frob_split (A : Matrix (Fin n) (Fin n) ℝ) :
    ch05b_frobSq A
      = ch05b_frobSq (ch05b_diagPart A)
        + ch05b_frobSq (ch05b_Pi A - ch05b_diagPart A)
        + ch05b_frobSq (A - ch05b_Pi A) := by
  have key : ∀ i j : Fin n, (A i j) ^ 2
      = (ch05b_diagPart A i j) ^ 2
        + ((ch05b_Pi A - ch05b_diagPart A) i j) ^ 2
        + ((A - ch05b_Pi A) i j) ^ 2 := by
    intro i j
    by_cases hij : i = j
    · subst hij
      simp [ch05b_diagPart, ch05b_Pi, Matrix.sub_apply]
    · have hne : ¬ (Int.natAbs ((i : ℤ) - (j : ℤ)) = 0) := by
        intro h
        exact hij (Fin.ext (by omega))
      by_cases hd : Int.natAbs ((i : ℤ) - (j : ℤ)) ≤ 1
      · simp [ch05b_diagPart, ch05b_Pi, Matrix.sub_apply, hij, hd]
      · simp [ch05b_diagPart, ch05b_Pi, Matrix.sub_apply, hij, hd]
  simp only [ch05b_frobSq]
  rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun j _ => key i j

-- eq:g-offmass (ch05 lines 1645-1649): `M_⊥(t)² = Σ_{|i-j| ≥ 2}(Q_t)_{ij}²`.
theorem ch05b_g_offmass (A : Matrix (Fin n) (Fin n) ℝ) :
    ch05b_frobSq (A - ch05b_Pi A)
      = ∑ i : Fin n, ∑ j : Fin n,
          (if 2 ≤ Int.natAbs ((i : ℤ) - (j : ℤ)) then (A i j) ^ 2 else 0) := by
  simp only [ch05b_frobSq]
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
  by_cases hd : Int.natAbs ((i : ℤ) - (j : ℤ)) ≤ 1
  · rw [if_neg (by omega)]
    simp [ch05b_Pi, Matrix.sub_apply, hd]
  · rw [if_pos (by omega)]
    simp [ch05b_Pi, Matrix.sub_apply, hd]

-- ch05 line 1656: `Σ_t = I + e^{-2t}(Σ₀ - I)`.
theorem ch05b_noname_Sigt_larget (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ) :
    ch05b_Sigt S0 t
      = (1 : Matrix (Fin n) (Fin n) ℝ)
        + Real.exp (-2 * t) • (S0 - (1 : Matrix (Fin n) (Fin n) ℝ)) := by
  rw [ch05b_Sigt, smul_sub, ch05b_Delta]
  rw [sub_smul, one_smul]
  abel

-- eq:g-larget (ch05 lines 1659-1662):
-- `Q_t = I - e^{-2t}(Σ₀ - I) + e^{-4t}(Σ₀ - I)² + O(e^{-6t})`, as the exact
-- three-term Neumann expansion with its remainder.
theorem ch05b_g_larget (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ)
    (hSt : IsUnit (ch05b_Sigt S0 t).det) :
    ch05b_Qt S0 t
      = (1 : Matrix (Fin n) (Fin n) ℝ)
        - Real.exp (-2 * t) • (S0 - (1 : Matrix (Fin n) (Fin n) ℝ))
        + (Real.exp (-2 * t)) ^ 2 • (S0 - (1 : Matrix (Fin n) (Fin n) ℝ)) ^ 2
        + (-(Real.exp (-2 * t))) ^ 3
            • (ch05b_Qt S0 t * (S0 - (1 : Matrix (Fin n) (Fin n) ℝ)) ^ 3) := by
  set E : Matrix (Fin n) (Fin n) ℝ := S0 - (1 : Matrix (Fin n) (Fin n) ℝ) with hE
  set u : ℝ := Real.exp (-2 * t) with hu
  have hSig : ch05b_Sigt S0 t = 1 + u • E := ch05b_noname_Sigt_larget S0 t
  have hunit : IsUnit ((1 : Matrix (Fin n) (Fin n) ℝ) + u • E).det := by rw [← hSig]; exact hSt
  have hneu := ch05b_g_neumann_general_inv (u • E) 3 hunit
  have hQ : ch05b_Qt S0 t = ((1 : Matrix (Fin n) (Fin n) ℝ) + u • E)⁻¹ := by
    rw [ch05b_Qt, hSig]
  rw [hQ]
  conv_lhs => rw [hneu]
  rw [Finset.sum_range_succ, Finset.sum_range_succ, Finset.sum_range_succ, Finset.sum_range_zero]
  simp only [ch05b_neg_smul_pow, pow_zero, pow_one, one_smul, zero_add]
  rw [← hQ, Matrix.mul_smul, neg_sq]
  abel


/-! ### Pass-2 strengthenings -/

-- eq:g-tridiag-inverse (ch05 lines 1560-1566): the text's formula itself,
-- `(B⁻¹)_{ij} = (-1)^{i+j} (θ_{i-1} φ_{j+1} / θ_{L-1}) ∏_{k=i}^{j-1} b_k`, checked
-- entry by entry at `L = 3` with symbolic `a₀,a₁,a₂,b₀,b₁`.  In 0-based indexing
-- θ₋₁ = 1, θ₀ = a₀, θ₁ = a₀a₁ - b₀² are the leading principal minors,
-- φ₃ = 1, φ₂ = a₂, φ₁ = a₁a₂ - b₁² the trailing ones, and `D = θ₂ = det B`.
-- The lower triangle is the `i ↔ j` mirror, as it must be for symmetric `B`.
theorem ch05b_g_tridiag_inverse_three_usmani (a0 a1 a2 b0 b1 D : ℝ)
    (hD : D = a0 * a1 * a2 - a0 * b1 ^ 2 - b0 ^ 2 * a2) (hdet : D ≠ 0) :
    (!![a0, b0, 0; b0, a1, b1; 0, b1, a2])⁻¹ 0 0
        = (-1 : ℝ) ^ (0 + 0) * (1 * (a1 * a2 - b1 ^ 2)) / D ∧
      (!![a0, b0, 0; b0, a1, b1; 0, b1, a2])⁻¹ 0 1
        = (-1 : ℝ) ^ (0 + 1) * (1 * a2) / D * b0 ∧
      (!![a0, b0, 0; b0, a1, b1; 0, b1, a2])⁻¹ 0 2
        = (-1 : ℝ) ^ (0 + 2) * (1 * 1) / D * (b0 * b1) ∧
      (!![a0, b0, 0; b0, a1, b1; 0, b1, a2])⁻¹ 1 1
        = (-1 : ℝ) ^ (1 + 1) * (a0 * a2) / D ∧
      (!![a0, b0, 0; b0, a1, b1; 0, b1, a2])⁻¹ 1 2
        = (-1 : ℝ) ^ (1 + 2) * (a0 * 1) / D * b1 ∧
      (!![a0, b0, 0; b0, a1, b1; 0, b1, a2])⁻¹ 2 2
        = (-1 : ℝ) ^ (2 + 2) * ((a0 * a1 - b0 ^ 2) * 1) / D := by
  subst hD
  have hinv := ch05b_g_tridiag_inverse_three a0 a1 a2 b0 b1 hdet
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    rw [hinv] <;>
    simp only [Matrix.smul_apply, smul_eq_mul, Matrix.cons_val_zero, Matrix.cons_val_one,
      Matrix.head_cons, Matrix.cons_val_two, Matrix.tail_cons, Matrix.head_fin_const,
      Matrix.of_apply] <;>
    field_simp <;>
    ring

-- eq:g-neumann-condition (ch05 lines 1458-1461): under `c λ_max(Q₀) < 1` the
-- Neumann series converges in every eigen-coordinate, and its sums assemble
-- exactly into the resolvent `(I + cQ₀)⁻¹`.
theorem ch05b_g_neumann_condition_matrix (U : Matrix (Fin n) (Fin n) ℝ) (ω : Fin n → ℝ)
    (hU : Uᵀ * U = 1) (Q0 : Matrix (Fin n) (Fin n) ℝ)
    (hQ0 : Q0 = U * Matrix.diagonal ω * Uᵀ) (c : ℝ) (hc : 0 < c)
    (hω : ∀ i, 0 < ω i) (i0 : Fin n) (hmax : ∀ i, ω i ≤ ω i0) (h : c * ω i0 < 1) :
    ((1 : Matrix (Fin n) (Fin n) ℝ) + c • Q0)⁻¹
      = U * Matrix.diagonal (fun i => ∑' m : ℕ, (-(c * ω i)) ^ m) * Uᵀ := by
  have haff : (1 : Matrix (Fin n) (Fin n) ℝ) + c • Q0
      = U * Matrix.diagonal (fun i => c * ω i + 1) * Uᵀ := by
    rw [← ch05b_g_affine U ω hU c 1, hQ0, one_smul]; abel
  have hpos : ∀ i, 0 < c * ω i + 1 := by
    intro i
    have := mul_pos hc (hω i)
    linarith
  have hfun : (fun i => (c * ω i + 1)⁻¹) = fun i => ∑' m : ℕ, (-(c * ω i)) ^ m := by
    funext i
    have hlt : |c * ω i| < 1 := by
      rw [abs_lt]
      have h1 : 0 < c * ω i := mul_pos hc (hω i)
      have h2 : c * ω i ≤ c * ω i0 := by nlinarith [hmax i]
      constructor <;> linarith
    rw [ch05b_noname_geometric _ hlt, one_div]
    congr 1
    ring
  rw [haff, ch05b_g_affine_inv U _ hU (fun i => ne_of_gt (hpos i)), hfun]

/-- `e^{2t}(1 - 2t) ≤ 1`, i.e. `e^{2t} ≤ 1/(1-2t)` when `2t < 1`. -/
theorem ch05b_exp_le_inv (t : ℝ) : Real.exp (2 * t) * (1 - 2 * t) ≤ 1 := by
  have h := Real.add_one_le_exp (-(2 * t))
  have hmul : Real.exp (-(2 * t)) * Real.exp (2 * t) = 1 := by
    rw [← Real.exp_add]; simp
  have hpos : 0 < Real.exp (2 * t) := Real.exp_pos _
  nlinarith [h, hmul, hpos]

/-- `e^{2t} ≤ 2` on `0 ≤ t ≤ 1/8`. -/
theorem ch05b_exp_le_two {t : ℝ} (ht : 0 ≤ t) (ht8 : t ≤ 1/8) :
    Real.exp (2 * t) ≤ 2 := by
  have hle := ch05b_exp_le_inv t
  have hpos : 0 < Real.exp (2 * t) := Real.exp_pos _
  nlinarith [hle, hpos, ht, ht8]

-- noname-28 (ch05 lines 1596-1600), quantitative form of `c_t = 2t + O(t²)` as it is
-- actually used: `2t ≤ e^{2t}c_t ≤ 2t + 15t²` for `0 ≤ t ≤ 1/8`.
theorem ch05b_c_exp_bridge {t : ℝ} (ht : 0 ≤ t) (ht8 : t ≤ 1/8) :
    2 * t ≤ Real.exp (2 * t) * ch05b_c t ∧
      Real.exp (2 * t) * ch05b_c t ≤ 2 * t + 15 * t ^ 2 := by
  have hE : 0 < Real.exp (2 * t) := Real.exp_pos _
  have hE1 : 1 ≤ Real.exp (2 * t) := by
    have := Real.add_one_le_exp (2 * t); linarith
  have hc := ch05b_c_bounds ht (by linarith)
  have hle := ch05b_exp_le_inv t
  have hu : (0 : ℝ) < 1 - 2 * t := by linarith
  refine ⟨by nlinarith [hc.1, hE1], ?_⟩
  have hcE : ch05b_c t = Real.exp (2 * t) - 1 := rfl
  rw [hcE]
  -- with `v = 1/(1-2t)`: `E ≤ v`, and `v(v-1) ≤ 2t + 15t²`.
  set v : ℝ := 1 / (1 - 2 * t) with hvdef
  have hv : v * (1 - 2 * t) = 1 := by rw [hvdef]; field_simp
  have hEv : Real.exp (2 * t) ≤ v := by
    rw [hvdef, le_div_iff₀ hu]; exact hle
  have hv1 : 1 ≤ v := le_trans hE1 hEv
  have hsq : v * (v - 1) * (1 - 2 * t) ^ 2 = 2 * t := by
    linear_combination (v * (1 - 2 * t) + 2 * t) * hv
  have hpoly : 0 ≤ t * (7 - 52 * t + 60 * t ^ 2) := by nlinarith [ht, ht8, sq_nonneg t]
  have hsqpos : (0 : ℝ) < (1 - 2 * t) ^ 2 := by positivity
  have hstep : v * (v - 1) * (1 - 2 * t) ^ 2
      ≤ (2 * t + 15 * t ^ 2) * (1 - 2 * t) ^ 2 := by
    rw [hsq]; nlinarith [hpoly]
  have hvv : v * (v - 1) ≤ 2 * t + 15 * t ^ 2 :=
    le_of_mul_le_mul_right (by linarith [hstep]) hsqpos
  nlinarith [hEv, hE1, hv1, hvv]

-- eq:g-bandfill (ch05 lines 1584-1591) at `d = 2`, the instance the text uses at
-- line 1653: `(Q_t)_{i,i+2} = -2t (Q₀²)_{i,i+2} + O(t²)`, with explicit constants
-- valid on `0 ≤ t ≤ 1/8`.
theorem ch05b_g_bandfill_two (S0 : Matrix (Fin n) (Fin n) ℝ) {t : ℝ}
    (ht : 0 ≤ t) (ht8 : t ≤ 1/8) (i j : Fin n)
    (hij : Int.natAbs ((i : ℤ) - (j : ℤ)) = 2)
    (hQ : ch05b_Banded S0⁻¹ 1) (hS0 : IsUnit S0.det)
    (hB : IsUnit ((1 : Matrix (Fin n) (Fin n) ℝ) + ch05b_c t • S0⁻¹).det) :
    |ch05b_Qt S0 t i j + 2 * t * (S0⁻¹ ^ 2) i j|
      ≤ 15 * t ^ 2 * |(S0⁻¹ ^ 2) i j|
        + 32 * t ^ 2 * |(S0⁻¹ * ((1 : Matrix (Fin n) (Fin n) ℝ) + ch05b_c t • S0⁻¹)⁻¹
            * S0⁻¹ ^ 2) i j| := by
  have hbf := ch05b_g_bandfill S0 t 2 (by norm_num) i j hij hQ hS0 hB
  simp only [show (2 : ℕ) - 1 = 1 from rfl, pow_one] at hbf
  set A : ℝ := (S0⁻¹ ^ 2) i j with hA
  set R : ℝ := (S0⁻¹ * ((1 : Matrix (Fin n) (Fin n) ℝ) + ch05b_c t • S0⁻¹)⁻¹
      * S0⁻¹ ^ 2) i j with hR
  have hEpos : 0 < Real.exp (2 * t) := Real.exp_pos _
  have hE2 := ch05b_exp_le_two ht ht8
  have hc := ch05b_c_bounds ht (by linarith)
  have hbridge := ch05b_c_exp_bridge ht ht8
  have heq : ch05b_Qt S0 t i j + 2 * t * A
      = -(Real.exp (2 * t) * ch05b_c t - 2 * t) * A
        + Real.exp (2 * t) * ch05b_c t ^ 2 * R := by
    rw [hbf]; ring
  rw [heq]
  have h1 : |(-(Real.exp (2 * t) * ch05b_c t - 2 * t)) * A| ≤ 15 * t ^ 2 * |A| := by
    rw [abs_mul, abs_neg]
    have hb : |Real.exp (2 * t) * ch05b_c t - 2 * t| ≤ 15 * t ^ 2 := by
      rw [abs_le]
      exact ⟨by linarith [hbridge.1, sq_nonneg t], by linarith [hbridge.2]⟩
    exact mul_le_mul_of_nonneg_right hb (abs_nonneg _)
  have h2 : |Real.exp (2 * t) * ch05b_c t ^ 2 * R| ≤ 32 * t ^ 2 * |R| := by
    rw [abs_mul]
    have hnn : 0 ≤ Real.exp (2 * t) * ch05b_c t ^ 2 := by positivity
    have hcsq : ch05b_c t ^ 2 ≤ 16 * t ^ 2 := by nlinarith [hc.1, hc.2, ht]
    have hle : Real.exp (2 * t) * ch05b_c t ^ 2 ≤ 32 * t ^ 2 := by
      nlinarith [hE2, hEpos, hcsq, sq_nonneg (ch05b_c t), sq_nonneg t]
    calc |Real.exp (2 * t) * ch05b_c t ^ 2| * |R|
        = Real.exp (2 * t) * ch05b_c t ^ 2 * |R| := by rw [abs_of_nonneg hnn]
      _ ≤ 32 * t ^ 2 * |R| := mul_le_mul_of_nonneg_right hle (abs_nonneg _)
  calc |(-(Real.exp (2 * t) * ch05b_c t - 2 * t)) * A
          + Real.exp (2 * t) * ch05b_c t ^ 2 * R|
      ≤ |(-(Real.exp (2 * t) * ch05b_c t - 2 * t)) * A|
        + |Real.exp (2 * t) * ch05b_c t ^ 2 * R| := abs_add_le _ _
    _ ≤ 15 * t ^ 2 * |A| + 32 * t ^ 2 * |R| := add_le_add h1 h2

/-! ## Pass-3 strengthenings: general-`L` theorems

The two matrix facts the chapter uses at `L = 2, 3` are proved here for **every** chain
length, with no size-specific computation:

* Usmani's closed form for the inverse of a symmetric tridiagonal matrix
  (`eq:g-tridiag-inverse`), with the leading and trailing principal minors identified with
  the determinants of the corresponding principal blocks (`ch05b_det_tri`,
  `ch05b_ps_eq_trailing`), and
* the tridiagonality of the AR(1) precision matrix `Q₀ = Σ₀⁻¹` (`noname-25`), through the
  explicit inverse `ch05b_Q0`.
-/

/-! ## General symmetric tridiagonal matrices and Usmani's inverse formula -/

/-- Entries of the symmetric tridiagonal matrix with diagonal `a` and off-diagonal `b`,
as a function of natural-number indices. -/
def ch05b_triE (a b : ℕ → ℝ) (i j : ℕ) : ℝ :=
  if i = j then a i else if i + 1 = j then b i else if j + 1 = i then b j else 0

/-- The `L × L` symmetric tridiagonal matrix with diagonal `a` and off-diagonal `b`. -/
def ch05b_tri (a b : ℕ → ℝ) (L : ℕ) : Matrix (Fin L) (Fin L) ℝ :=
  fun i j => ch05b_triE a b (i : ℕ) (j : ℕ)

theorem ch05b_tri_apply (a b : ℕ → ℝ) (L : ℕ) (i j : Fin L) :
    ch05b_tri a b L i j = ch05b_triE a b (i : ℕ) (j : ℕ) := rfl

theorem ch05b_triE_self (a b : ℕ → ℝ) (i : ℕ) : ch05b_triE a b i i = a i := by
  unfold ch05b_triE; split_ifs <;> first | rfl | (exfalso; omega)

theorem ch05b_triE_up (a b : ℕ → ℝ) (i k : ℕ) (h : i + 1 = k) : ch05b_triE a b i k = b i := by
  subst h; unfold ch05b_triE; split_ifs <;> first | rfl | (exfalso; omega)

theorem ch05b_triE_down (a b : ℕ → ℝ) (i k : ℕ) (h : k + 1 = i) : ch05b_triE a b i k = b k := by
  subst h; unfold ch05b_triE; split_ifs <;> first | rfl | (exfalso; omega)

theorem ch05b_prod_Ico_single (b : ℕ → ℝ) (p q : ℕ) (h : p + 1 = q) :
    ∏ k ∈ Finset.Ico p q, b k = b p := by
  subst h
  rw [Finset.prod_Ico_succ_top (le_refl p), Finset.Ico_self, Finset.prod_empty, one_mul]

theorem ch05b_triE_eq_zero (a b : ℕ → ℝ) {i k : ℕ} (h1 : i ≠ k) (h2 : i + 1 ≠ k)
    (h3 : k + 1 ≠ i) : ch05b_triE a b i k = 0 := by
  simp [ch05b_triE, h1, h2, h3]

theorem ch05b_triE_shift (a b : ℕ → ℝ) (m i j : ℕ) :
    ch05b_triE (fun k => a (m + k)) (fun k => b (m + k)) i j = ch05b_triE a b (m + i) (m + j) := by
  unfold ch05b_triE
  split_ifs <;> first | rfl | (exfalso; omega)

theorem ch05b_triE_symm (a b : ℕ → ℝ) (i j : ℕ) :
    ch05b_triE a b j i = ch05b_triE a b i j := by
  unfold ch05b_triE
  split_ifs <;> first | rfl | (exfalso; omega) | (congr 1 <;> omega)

theorem ch05b_tri_isSymm (a b : ℕ → ℝ) (L : ℕ) : (ch05b_tri a b L).IsSymm := by
  unfold Matrix.IsSymm
  ext i j
  exact ch05b_triE_symm a b (i : ℕ) (j : ℕ)

/-- Leading principal minors of the tridiagonal matrix, by the three-term recursion:
`ch05b_th a b (m+1)` is the order-`m` leading minor, the text's `θ_{m-1}`
(so `ch05b_th a b 1 = 1 = θ_{-1}`).  The extra slot `ch05b_th a b 0 = 0` stands for
`θ_{-2} = 0` and makes the recursion uniform at the top-left corner. -/
def ch05b_th (a b : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | 1 => 1
  | (m + 2) => a m * ch05b_th a b (m + 1) - (b (m - 1)) ^ 2 * ch05b_th a b m

/-- Trailing principal minors, indexed from the bottom: `ch05b_ps a b L (k+1)` is the
determinant of the trailing `k × k` block, the text's `φ_{L-k}`.  `ch05b_ps a b L 0 = 0`
is the analogous phantom slot at the bottom-right corner. -/
def ch05b_ps (a b : ℕ → ℝ) (L : ℕ) : ℕ → ℝ
  | 0 => 0
  | 1 => 1
  | (k + 2) => a (L - 1 - k) * ch05b_ps a b L (k + 1) - (b (L - 1 - k)) ^ 2 * ch05b_ps a b L k

theorem ch05b_th_zero (a b : ℕ → ℝ) : ch05b_th a b 0 = 0 := rfl
theorem ch05b_th_one (a b : ℕ → ℝ) : ch05b_th a b 1 = 1 := rfl
theorem ch05b_th_two (a b : ℕ → ℝ) : ch05b_th a b 2 = a 0 := by
  show a 0 * ch05b_th a b 1 - (b (0 - 1)) ^ 2 * ch05b_th a b 0 = a 0
  rw [ch05b_th_one, ch05b_th_zero]; ring
theorem ch05b_th_rec (a b : ℕ → ℝ) (p : ℕ) :
    ch05b_th a b (p + 3) = a (p + 1) * ch05b_th a b (p + 2) - (b p) ^ 2 * ch05b_th a b (p + 1) :=
  rfl

theorem ch05b_ps_zero (a b : ℕ → ℝ) (L : ℕ) : ch05b_ps a b L 0 = 0 := rfl
theorem ch05b_ps_one (a b : ℕ → ℝ) (L : ℕ) : ch05b_ps a b L 1 = 1 := rfl
theorem ch05b_ps_rec (a b : ℕ → ℝ) (L k : ℕ) :
    ch05b_ps a b L (k + 2)
      = a (L - 1 - k) * ch05b_ps a b L (k + 1) - (b (L - 1 - k)) ^ 2 * ch05b_ps a b L k := rfl

/-- The unnormalised Usmani cofactor matrix entries. -/
def ch05b_usmE (a b : ℕ → ℝ) (L : ℕ) (i j : ℕ) : ℝ :=
  (-1) ^ (i + j) * ch05b_th a b (min i j + 1) * ch05b_ps a b L (L - max i j)
    * ∏ k ∈ Finset.Ico (min i j) (max i j), b k

def ch05b_usm (a b : ℕ → ℝ) (L : ℕ) : Matrix (Fin L) (Fin L) ℝ :=
  fun i j => ch05b_usmE a b L (i : ℕ) (j : ℕ)

theorem ch05b_usm_apply (a b : ℕ → ℝ) (L : ℕ) (i j : Fin L) :
    ch05b_usm a b L i j = ch05b_usmE a b L (i : ℕ) (j : ℕ) := rfl

theorem ch05b_usmE_le (a b : ℕ → ℝ) (L i j : ℕ) (h : i ≤ j) :
    ch05b_usmE a b L i j
      = (-1) ^ (i + j) * ch05b_th a b (i + 1) * ch05b_ps a b L (L - j)
        * ∏ k ∈ Finset.Ico i j, b k := by
  simp [ch05b_usmE, min_eq_left h, max_eq_right h]

theorem ch05b_usmE_ge (a b : ℕ → ℝ) (L i j : ℕ) (h : j ≤ i) :
    ch05b_usmE a b L i j
      = (-1) ^ (i + j) * ch05b_th a b (j + 1) * ch05b_ps a b L (L - i)
        * ∏ k ∈ Finset.Ico j i, b k := by
  simp [ch05b_usmE, min_eq_right h, max_eq_left h]

theorem ch05b_usmE_out (a b : ℕ → ℝ) (L i j : ℕ) (h : L ≤ i) : ch05b_usmE a b L i j = 0 := by
  have hm : L - max i j = 0 := by omega
  simp [ch05b_usmE, hm, ch05b_ps_zero]

/-! ### Summing a banded row -/

theorem ch05b_sum_band0 (L : ℕ) (g : ℕ → ℝ)
    (h2 : ∀ k, 1 < k → g k = 0) (h3 : ∀ k, L ≤ k → g k = 0) :
    ∑ k ∈ Finset.range L, g k = g 0 + g 1 := by
  have hss : Finset.range L ⊆ Finset.range (max L 2) := by
    intro x hx; simp only [Finset.mem_range] at hx ⊢; omega
  have hss2 : Finset.range 2 ⊆ Finset.range (max L 2) := by
    intro x hx; simp only [Finset.mem_range] at hx ⊢; omega
  have hext : ∑ k ∈ Finset.range L, g k = ∑ k ∈ Finset.range (max L 2), g k := by
    refine Finset.sum_subset hss ?_
    intro x _ hx'
    simp only [Finset.mem_range, not_lt] at hx'
    exact h3 x hx'
  have hvan : ∀ x ∈ Finset.range (max L 2), x ∉ Finset.range 2 → g x = 0 := by
    intro x _ hx
    simp only [Finset.mem_range, not_lt] at hx
    exact h2 x (by omega)
  rw [hext, ← Finset.sum_subset hss2 hvan, show (2 : ℕ) = 1 + 1 from rfl,
    Finset.sum_range_succ, Finset.sum_range_one]

theorem ch05b_sum_band (L p : ℕ) (g : ℕ → ℝ)
    (h1 : ∀ k, k < p → g k = 0) (h2 : ∀ k, p + 2 < k → g k = 0) (h3 : ∀ k, L ≤ k → g k = 0) :
    ∑ k ∈ Finset.range L, g k = g p + g (p + 1) + g (p + 2) := by
  have hss : Finset.range L ⊆ Finset.range (max L (p + 3)) := by
    intro x hx; simp only [Finset.mem_range] at hx ⊢; omega
  have hext : ∑ k ∈ Finset.range L, g k = ∑ k ∈ Finset.range (max L (p + 3)), g k := by
    refine Finset.sum_subset hss ?_
    intro x _ hx'
    simp only [Finset.mem_range, not_lt] at hx'
    exact h3 x hx'
  have hsub : Finset.Ico p (p + 3) ⊆ Finset.range (max L (p + 3)) := by
    intro x hx
    simp only [Finset.mem_Ico] at hx
    simp only [Finset.mem_range]
    have h := le_max_right L (p + 3)
    omega
  have hvan : ∀ x ∈ Finset.range (max L (p + 3)), x ∉ Finset.Ico p (p + 3) → g x = 0 := by
    intro x _ hx
    simp only [Finset.mem_Ico, not_and_or, not_lt, not_le] at hx
    rcases hx with hx | hx
    · exact h1 x hx
    · exact h2 x (by omega)
  have hIco : ∑ k ∈ Finset.Ico p (p + 3), g k = g p + g (p + 1) + g (p + 2) := by
    rw [Finset.sum_Ico_eq_sum_range, show p + 3 - p = 3 from by omega,
      show (3 : ℕ) = 2 + 1 from rfl, Finset.sum_range_succ, show (2 : ℕ) = 1 + 1 from rfl,
      Finset.sum_range_succ, Finset.sum_range_one]
    simp only [Nat.add_zero]
  rw [hext, ← Finset.sum_subset hsub hvan, hIco]

/-! ### The minor invariant -/

/-- The quantity `θ_i φ_{i+1} - b_i² θ_{i-1} φ_{i+2}` does not depend on `i`: it is the
determinant `θ_{L-1}` throughout.  This is the identity behind the diagonal entries of
`B B⁻¹ = I`. -/
theorem ch05b_minor_invariant (a b : ℕ → ℝ) (L : ℕ) :
    ∀ r p : ℕ, p + r + 1 = L →
      ch05b_th a b (p + 2) * ch05b_ps a b L (r + 1)
        - (b p) ^ 2 * ch05b_th a b (p + 1) * ch05b_ps a b L r = ch05b_th a b (L + 1) := by
  intro r
  induction r with
  | zero =>
      intro p hp
      have hL : L = p + 1 := by omega
      subst hL
      rw [ch05b_ps_one, ch05b_ps_zero, show p + 1 + 1 = p + 2 from rfl]
      ring
  | succ r ih =>
      intro p hp
      have hIH := ih (p + 1) (by omega)
      have hps : ch05b_ps a b L (r + 1 + 1)
          = a (p + 1) * ch05b_ps a b L (r + 1) - (b (p + 1)) ^ 2 * ch05b_ps a b L r := by
        have h0 := ch05b_ps_rec a b L r
        rw [show L - 1 - r = p + 1 from by omega] at h0
        exact h0
      rw [hps, ← hIH, show p + 1 + 2 = p + 3 from rfl, show p + 1 + 1 = p + 2 from rfl,
        ch05b_th_rec]
      ring

/-! ### `B · U = θ_{L-1} · I` -/

theorem ch05b_tri_usm_row (a b : ℕ → ℝ) (L i j : ℕ) (hi : i < L) (hj : j < L) :
    ∑ k ∈ Finset.range L, ch05b_triE a b i k * ch05b_usmE a b L k j
      = if i = j then ch05b_th a b (L + 1) else 0 := by
  have hout : ∀ k, L ≤ k → ch05b_triE a b i k * ch05b_usmE a b L k j = 0 := by
    intro k hk
    rw [ch05b_usmE_out a b L k j hk, mul_zero]
  rcases Nat.eq_zero_or_pos i with hi0 | hi0
  · -- first row
    subst hi0
    obtain ⟨r, hr⟩ : ∃ r, L = r + 1 := ⟨L - 1, by omega⟩
    have h2 : ∀ k, 1 < k → ch05b_triE a b 0 k * ch05b_usmE a b L k j = 0 := by
      intro k hk
      rw [ch05b_triE_eq_zero a b (by omega) (by omega) (by omega), zero_mul]
    rw [ch05b_sum_band0 L _ h2 hout, ch05b_triE_self, ch05b_triE_up a b 0 1 rfl]
    rcases Nat.eq_zero_or_pos j with hj0 | hj0
    · -- diagonal entry (0,0)
      subst hj0
      have hinv := ch05b_minor_invariant a b L r 0 (by omega)
      rw [show (0 : ℕ) + 2 = 2 from rfl, show (0 : ℕ) + 1 = 1 from rfl, ch05b_th_two,
        ch05b_th_one] at hinv
      rw [if_pos rfl, ch05b_usmE_le a b L 0 0 (le_refl 0), ch05b_usmE_ge a b L 1 0 (by omega),
        show L - 0 = r + 1 from by omega, show L - 1 = r from by omega, ch05b_th_one]
      simp only [Finset.Ico_self, Finset.prod_empty, ch05b_prod_Ico_single b 0 1 rfl]
      linear_combination hinv
    · -- off-diagonal entry (0,j), j > 0
      have hP : ∏ k ∈ Finset.Ico 0 j, b k = b 0 * ∏ k ∈ Finset.Ico (0 + 1) j, b k :=
        Finset.prod_eq_prod_Ico_succ_bot (by omega) b
      rw [if_neg (by omega), ch05b_usmE_le a b L 0 j (by omega),
        ch05b_usmE_le a b L 1 j (by omega), hP, ch05b_th_one, ch05b_th_two,
        show (0 : ℕ) + 1 = 1 from rfl, show (1 : ℕ) + j = j + 1 from by omega, pow_succ,
        show (0 : ℕ) + j = j from by omega]
      ring
  · -- row i = p+1
    obtain ⟨p, rfl⟩ : ∃ p, i = p + 1 := ⟨i - 1, by omega⟩
    obtain ⟨r, hr⟩ : ∃ r, L = p + 2 + r := ⟨L - (p + 2), by omega⟩
    have h1 : ∀ k, k < p → ch05b_triE a b (p + 1) k * ch05b_usmE a b L k j = 0 := by
      intro k hk
      rw [ch05b_triE_eq_zero a b (by omega) (by omega) (by omega), zero_mul]
    have h2 : ∀ k, p + 2 < k → ch05b_triE a b (p + 1) k * ch05b_usmE a b L k j = 0 := by
      intro k hk
      rw [ch05b_triE_eq_zero a b (by omega) (by omega) (by omega), zero_mul]
    rw [ch05b_sum_band L p _ h1 h2 hout, ch05b_triE_down a b (p + 1) p rfl, ch05b_triE_self,
      ch05b_triE_up a b (p + 1) (p + 2) rfl]
    rcases lt_trichotomy j (p + 1) with hlt | heq | hgt
    · -- j < i : killed by the trailing-minor recursion
      have hps : ch05b_ps a b L (r + 2)
          = a (p + 1) * ch05b_ps a b L (r + 1) - (b (p + 1)) ^ 2 * ch05b_ps a b L r := by
        have h0 := ch05b_ps_rec a b L r
        rw [show L - 1 - r = p + 1 from by omega] at h0
        exact h0
      have hP1 : ∏ k ∈ Finset.Ico j (p + 1), b k = (∏ k ∈ Finset.Ico j p, b k) * b p :=
        Finset.prod_Ico_succ_top (by omega) b
      have hP2 : ∏ k ∈ Finset.Ico j (p + 2), b k
          = (∏ k ∈ Finset.Ico j (p + 1), b k) * b (p + 1) :=
        Finset.prod_Ico_succ_top (by omega) b
      have e1 : ((-1 : ℝ)) ^ (p + 1 + j) = -(-1 : ℝ) ^ (p + j) := by
        rw [show p + 1 + j = p + j + 1 from by omega, pow_succ]; ring
      have e2 : ((-1 : ℝ)) ^ (p + 2 + j) = (-1 : ℝ) ^ (p + j) := by
        rw [show p + 2 + j = p + j + 1 + 1 from by omega, pow_succ, pow_succ]; ring
      rw [if_neg (by omega), ch05b_usmE_ge a b L p j (by omega),
        ch05b_usmE_ge a b L (p + 1) j (by omega), ch05b_usmE_ge a b L (p + 2) j (by omega),
        show L - p = r + 2 from by omega, show L - (p + 1) = r + 1 from by omega,
        show L - (p + 2) = r from by omega, hP2, hP1, hps, e1, e2]
      ring
    · -- diagonal entry
      subst heq
      have hinv := ch05b_minor_invariant a b L r (p + 1) (by omega)
      rw [show p + 1 + 2 = p + 3 from rfl, show p + 1 + 1 = p + 2 from rfl,
        ch05b_th_rec a b p] at hinv
      have s1 : ((-1 : ℝ)) ^ (p + (p + 1)) = -1 := Odd.neg_one_pow ⟨p, by ring⟩
      have s2 : ((-1 : ℝ)) ^ (p + 1 + (p + 1)) = 1 := Even.neg_one_pow ⟨p + 1, by ring⟩
      have s3 : ((-1 : ℝ)) ^ (p + 2 + (p + 1)) = -1 := Odd.neg_one_pow ⟨p + 1, by ring⟩
      rw [if_pos rfl, ch05b_usmE_le a b L p (p + 1) (by omega),
        ch05b_usmE_le a b L (p + 1) (p + 1) (le_refl _),
        ch05b_usmE_ge a b L (p + 2) (p + 1) (by omega),
        show L - (p + 1) = r + 1 from by omega, show L - (p + 2) = r from by omega,
        show p + 1 + 1 = p + 2 from rfl, s1, s2, s3,
        ch05b_prod_Ico_single b p (p + 1) rfl, ch05b_prod_Ico_single b (p + 1) (p + 2) rfl]
      simp only [Finset.Ico_self, Finset.prod_empty]
      linear_combination hinv
    · -- j > i : killed by the leading-minor recursion
      have hQ1 : ∏ k ∈ Finset.Ico p j, b k = b p * ∏ k ∈ Finset.Ico (p + 1) j, b k :=
        Finset.prod_eq_prod_Ico_succ_bot (by omega) b
      have hQ2 : ∏ k ∈ Finset.Ico (p + 1) j, b k
          = b (p + 1) * ∏ k ∈ Finset.Ico (p + 2) j, b k :=
        Finset.prod_eq_prod_Ico_succ_bot (by omega) b
      have e1 : ((-1 : ℝ)) ^ (p + 1 + j) = -(-1 : ℝ) ^ (p + j) := by
        rw [show p + 1 + j = p + j + 1 from by omega, pow_succ]; ring
      have e2 : ((-1 : ℝ)) ^ (p + 2 + j) = (-1 : ℝ) ^ (p + j) := by
        rw [show p + 2 + j = p + j + 1 + 1 from by omega, pow_succ, pow_succ]; ring
      rw [if_neg (by omega), ch05b_usmE_le a b L p j (by omega),
        ch05b_usmE_le a b L (p + 1) j (by omega), ch05b_usmE_le a b L (p + 2) j (by omega),
        hQ1, hQ2, e1, e2, ch05b_th_rec]
      ring

/-- **Usmani's identity, general `L`.**  `B · U = θ_{L-1} · I`, where `U` is the
unnormalised cofactor matrix of the text's formula. -/
theorem ch05b_tri_mul_usm (a b : ℕ → ℝ) (L : ℕ) :
    ch05b_tri a b L * ch05b_usm a b L
      = ch05b_th a b (L + 1) • (1 : Matrix (Fin L) (Fin L) ℝ) := by
  ext i j
  rw [Matrix.mul_apply]
  simp only [ch05b_tri_apply, ch05b_usm_apply]
  rw [Fin.sum_univ_eq_sum_range
      (fun k => ch05b_triE a b (i : ℕ) k * ch05b_usmE a b L k (j : ℕ)) L,
    ch05b_tri_usm_row a b L i j i.isLt j.isLt, Matrix.smul_apply, Matrix.one_apply, smul_eq_mul]
  by_cases hij : (i : ℕ) = (j : ℕ)
  · rw [if_pos hij, if_pos (Fin.eq_of_val_eq hij), mul_one]
  · rw [if_neg hij, if_neg (fun h : i = j => hij (by rw [h])), mul_zero]


/-! ### The recursions are the principal minors -/

/-- Expansion of the determinant along the first row: the tridiagonal recursion. -/
theorem ch05b_det_tri_succ (a b : ℕ → ℝ) (m n : ℕ) :
    (ch05b_tri (fun i => a (m + i)) (fun i => b (m + i)) (n + 2)).det
      = a m * (ch05b_tri (fun i => a (m + 1 + i)) (fun i => b (m + 1 + i)) (n + 1)).det
        - (b m) ^ 2 * (ch05b_tri (fun i => a (m + 2 + i)) (fun i => b (m + 2 + i)) n).det := by
  have hA : ∀ (i j : Fin (n + 2)),
      ch05b_tri (fun k => a (m + k)) (fun k => b (m + k)) (n + 2) i j
        = ch05b_triE a b (m + (i : ℕ)) (m + (j : ℕ)) := by
    intro i j; rw [ch05b_tri_apply, ch05b_triE_shift]
  have hsub1 : (ch05b_tri (fun k => a (m + k)) (fun k => b (m + k)) (n + 2)).submatrix
        Fin.succ (Fin.succAbove 0)
      = ch05b_tri (fun k => a (m + 1 + k)) (fun k => b (m + 1 + k)) (n + 1) := by
    ext i j
    rw [Matrix.submatrix_apply, Fin.succAbove_zero, hA, ch05b_tri_apply, ch05b_triE_shift]
    simp only [Fin.val_succ]
    congr 1 <;> omega
  have hsub2 : ((ch05b_tri (fun k => a (m + k)) (fun k => b (m + k)) (n + 2)).submatrix
        Fin.succ (Fin.succAbove ((0 : Fin (n + 1)).succ))).submatrix Fin.succ Fin.succ
      = ch05b_tri (fun k => a (m + 2 + k)) (fun k => b (m + 2 + k)) n := by
    ext i j
    rw [Matrix.submatrix_apply, Matrix.submatrix_apply,
      Fin.succAbove_succ_of_lt (0 : Fin (n + 1)) j.succ (Fin.succ_pos j), hA,
      ch05b_tri_apply, ch05b_triE_shift]
    simp only [Fin.val_succ]
    congr 1 <;> omega
  have hrow : ∀ j : Fin n,
      ch05b_tri (fun k => a (m + k)) (fun k => b (m + k)) (n + 2) 0 j.succ.succ = 0 := by
    intro j
    rw [hA]
    refine ch05b_triE_eq_zero _ _ ?_ ?_ ?_ <;> simp only [Fin.val_succ, Fin.val_zero] <;> omega
  have hM0 : ∀ i : Fin n,
      (ch05b_tri (fun k => a (m + k)) (fun k => b (m + k)) (n + 2)).submatrix
        Fin.succ (Fin.succAbove ((0 : Fin (n + 1)).succ)) i.succ 0 = 0 := by
    intro i
    rw [Matrix.submatrix_apply,
      Fin.succAbove_succ_of_le (0 : Fin (n + 1)) 0 (le_refl _), hA]
    refine ch05b_triE_eq_zero _ _ ?_ ?_ ?_ <;>
      simp only [Fin.val_succ, Fin.val_zero, Fin.coe_castSucc] <;> omega
  have hdetM : ((ch05b_tri (fun k => a (m + k)) (fun k => b (m + k)) (n + 2)).submatrix
        Fin.succ (Fin.succAbove ((0 : Fin (n + 1)).succ))).det
      = b m * (ch05b_tri (fun k => a (m + 2 + k)) (fun k => b (m + 2 + k)) n).det := by
    rw [Matrix.det_succ_column_zero, Fin.sum_univ_succ]
    simp only [hM0, mul_zero, zero_mul, Finset.sum_const_zero, add_zero, Fin.val_zero,
      pow_zero, one_mul, Fin.succAbove_zero]
    rw [Matrix.submatrix_apply, Fin.succAbove_succ_of_le (0 : Fin (n + 1)) 0 (le_refl _), hA,
      hsub2]
    simp only [Fin.val_succ, Fin.val_zero, Fin.coe_castSucc]
    rw [ch05b_triE_down a b (m + (0 + 1)) (m + 0) (by omega)]
    norm_num
  rw [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.sum_univ_succ]
  simp only [hrow, mul_zero, zero_mul, Finset.sum_const_zero, add_zero]
  rw [hA, hA, hsub1, hdetM]
  simp only [Fin.val_zero, Fin.val_succ, pow_zero, one_mul]
  rw [ch05b_triE_self a b (m + 0), ch05b_triE_up a b (m + 0) (m + (0 + 1)) (by omega)]
  norm_num
  ring

/-- `ch05b_ps a b L (c+1)` is the determinant of the trailing `c × c` block of the
tridiagonal matrix, i.e. the text's trailing principal minor `φ_m` with `m + c = L`. -/
theorem ch05b_ps_eq_det (a b : ℕ → ℝ) (L : ℕ) (c : ℕ) : ∀ m : ℕ, m + c = L →
    ch05b_ps a b L (c + 1)
      = (ch05b_tri (fun i => a (m + i)) (fun i => b (m + i)) c).det := by
  induction c using Nat.twoStepInduction with
  | zero =>
      intro m _
      rw [ch05b_ps_one, Matrix.det_fin_zero]
  | one =>
      intro m hm
      rw [ch05b_ps_rec, ch05b_ps_one, ch05b_ps_zero, show L - 1 - 0 = m from by omega,
        Matrix.det_fin_one, ch05b_tri_apply, ch05b_triE_self]
      simp
  | more c ih1 ih2 =>
      intro m hm
      rw [ch05b_ps_rec, show L - 1 - (c + 1) = m from by omega, ih2 (m + 1) (by omega),
        ih1 (m + 2) (by omega), ch05b_det_tri_succ]

theorem ch05b_tri_shift_zero (a b : ℕ → ℝ) (L : ℕ) :
    ch05b_tri (fun i => a (0 + i)) (fun i => b (0 + i)) L = ch05b_tri a b L := by
  ext i j
  rw [ch05b_tri_apply, ch05b_tri_apply, ch05b_triE_shift]
  congr 1 <;> omega

theorem ch05b_ps_eq_th (a b : ℕ → ℝ) (L : ℕ) :
    ch05b_ps a b L (L + 1) = ch05b_th a b (L + 1) := by
  rcases Nat.eq_zero_or_pos L with hL | hL
  · subst hL; rw [ch05b_ps_one, ch05b_th_one]
  · obtain ⟨r, rfl⟩ : ∃ r, L = r + 1 := ⟨L - 1, by omega⟩
    have hinv := ch05b_minor_invariant a b (r + 1) r 0 (by omega)
    rw [show (0 : ℕ) + 2 = 2 from rfl, show (0 : ℕ) + 1 = 1 from rfl, ch05b_th_two,
      ch05b_th_one] at hinv
    rw [ch05b_ps_rec, show r + 1 - 1 - r = 0 from by omega]
    linear_combination hinv

/-- `θ_{L-1} = det B`: the recursion computes the determinant of the tridiagonal matrix. -/
theorem ch05b_det_tri (a b : ℕ → ℝ) (L : ℕ) :
    (ch05b_tri a b L).det = ch05b_th a b (L + 1) := by
  rw [← ch05b_tri_shift_zero a b L, ← ch05b_ps_eq_det a b L L 0 (by omega), ch05b_ps_eq_th]

/-- The leading `k × k` block of the tridiagonal matrix is the tridiagonal matrix of size `k`,
so `ch05b_th a b (k+1)` is the order-`k` leading principal minor. -/
theorem ch05b_tri_leading_block (a b : ℕ → ℝ) (L k : ℕ) (h : k ≤ L) :
    (ch05b_tri a b L).submatrix (Fin.castLE h) (Fin.castLE h) = ch05b_tri a b k := by
  ext i j; rfl

/-- The trailing `c × c` block of the tridiagonal matrix, starting at row `m`. -/
theorem ch05b_tri_trailing_block (a b : ℕ → ℝ) (L m c : ℕ) (h : m + c = L) :
    (ch05b_tri a b L).submatrix (fun i : Fin c => ⟨m + (i : ℕ), by have := i.isLt; omega⟩)
        (fun j : Fin c => ⟨m + (j : ℕ), by have := j.isLt; omega⟩)
      = ch05b_tri (fun i => a (m + i)) (fun i => b (m + i)) c := by
  ext i j
  rw [Matrix.submatrix_apply, ch05b_tri_apply, ch05b_tri_apply, ch05b_triE_shift]

theorem ch05b_ps_eq_trailing (a b : ℕ → ℝ) (L j : ℕ) (hj : j < L) :
    ch05b_ps a b L (L - j)
      = (ch05b_tri (fun k => a (j + 1 + k)) (fun k => b (j + 1 + k)) (L - j - 1)).det := by
  have h := ch05b_ps_eq_det a b L (L - j - 1) (j + 1) (by omega)
  rwa [show L - j - 1 + 1 = L - j from by omega] at h

/-! ### Usmani's formula for the inverse -/

theorem ch05b_g_tridiag_inverse (a b : ℕ → ℝ) (L : ℕ) (hdet : ch05b_th a b (L + 1) ≠ 0)
    (i j : Fin L) (hij : (i : ℕ) ≤ (j : ℕ)) :
    (ch05b_tri a b L)⁻¹ i j
      = (-1) ^ ((i : ℕ) + (j : ℕ))
          * (ch05b_th a b ((i : ℕ) + 1) * ch05b_ps a b L (L - (j : ℕ)))
          / ch05b_th a b (L + 1)
        * ∏ k ∈ Finset.Ico (i : ℕ) (j : ℕ), b k := by
  have hinv : (ch05b_tri a b L)⁻¹ = (ch05b_th a b (L + 1))⁻¹ • ch05b_usm a b L := by
    refine Matrix.inv_eq_right_inv ?_
    rw [Matrix.mul_smul, ch05b_tri_mul_usm, smul_smul, inv_mul_cancel₀ hdet, one_smul]
  rw [hinv, Matrix.smul_apply, smul_eq_mul, ch05b_usm_apply, ch05b_usmE_le a b L _ _ hij]
  field_simp <;> ring

/-- **eq:g-tridiag-inverse, for every size `L`.**  For an invertible symmetric tridiagonal
`B` with diagonal `a` and off-diagonal `b`, and `i ≤ j`,
`(B⁻¹)_{ij} = (-1)^{i+j} (θ_{i-1} φ_{j+1} / θ_{L-1}) ∏_{k=i}^{j-1} b_k`,
where `θ_{i-1}` is the order-`i` leading principal minor of `B`, `φ_{j+1}` the trailing
principal minor of order `L-j-1`, and `θ_{L-1} = det B`. -/
theorem ch05b_g_tridiag_inverse_minors (a b : ℕ → ℝ) (L : ℕ)
    (hdet : (ch05b_tri a b L).det ≠ 0) (i j : Fin L) (hij : (i : ℕ) ≤ (j : ℕ)) :
    (ch05b_tri a b L)⁻¹ i j
      = (-1) ^ ((i : ℕ) + (j : ℕ))
          * ((ch05b_tri a b (i : ℕ)).det
              * (ch05b_tri (fun k => a ((j : ℕ) + 1 + k)) (fun k => b ((j : ℕ) + 1 + k))
                  (L - (j : ℕ) - 1)).det)
          / (ch05b_tri a b L).det
        * ∏ k ∈ Finset.Ico (i : ℕ) (j : ℕ), b k := by
  rw [ch05b_det_tri a b L, ch05b_det_tri a b (i : ℕ),
    ← ch05b_ps_eq_trailing a b L (j : ℕ) j.isLt]
  exact ch05b_g_tridiag_inverse a b L (by rwa [ch05b_det_tri] at hdet) i j hij

/-! ## The AR(1) precision matrix, for every chain length -/

/-- Entries of the AR(1) covariance `Σ₀`, as a function of natural-number indices. -/
def ch05b_Sig0E (α : ℝ) (i j : ℕ) : ℝ := α ^ (Int.natAbs ((i : ℤ) - (j : ℤ)))

theorem ch05b_Sig0_apply (α : ℝ) (L : ℕ) (i j : Fin L) :
    ch05b_Sig0 α L i j = ch05b_Sig0E α (i : ℕ) (j : ℕ) := rfl

theorem ch05b_Sig0E_eq (α : ℝ) (i j d : ℕ) (h : Int.natAbs ((i : ℤ) - (j : ℤ)) = d) :
    ch05b_Sig0E α i j = α ^ d := by rw [ch05b_Sig0E, h]

/-- The AR(1) precision matrix: tridiagonal, with diagonal `(1, 1+α², …, 1+α², 1)/(1-α²)`
and off-diagonal `-α/(1-α²)`. -/
def ch05b_Q0E (α : ℝ) (L : ℕ) (i j : ℕ) : ℝ :=
  if L ≤ i ∨ L ≤ j then 0
  else if i = j then (if j = 0 ∨ j + 1 = L then 1 else 1 + α ^ 2) / (1 - α ^ 2)
  else if i + 1 = j ∨ j + 1 = i then -α / (1 - α ^ 2)
  else 0

def ch05b_Q0 (α : ℝ) (L : ℕ) : Matrix (Fin L) (Fin L) ℝ :=
  fun i j => ch05b_Q0E α L (i : ℕ) (j : ℕ)

theorem ch05b_Q0_apply (α : ℝ) (L : ℕ) (i j : Fin L) :
    ch05b_Q0 α L i j = ch05b_Q0E α L (i : ℕ) (j : ℕ) := rfl

theorem ch05b_Q0E_out (α : ℝ) (L i j : ℕ) (h : L ≤ i ∨ L ≤ j) : ch05b_Q0E α L i j = 0 := by
  unfold ch05b_Q0E; rw [if_pos h]

theorem ch05b_Q0E_far (α : ℝ) (L i j : ℕ) (h1 : i ≠ j) (h2 : i + 1 ≠ j) (h3 : j + 1 ≠ i) :
    ch05b_Q0E α L i j = 0 := by
  unfold ch05b_Q0E
  split_ifs <;> first | rfl | (exfalso; omega)

theorem ch05b_Q0E_diag_bd (α : ℝ) (L j : ℕ) (hj : j < L) (h : j = 0 ∨ j + 1 = L) :
    ch05b_Q0E α L j j = 1 / (1 - α ^ 2) := by
  unfold ch05b_Q0E; rw [if_neg (by omega), if_pos rfl, if_pos h]

theorem ch05b_Q0E_diag_int (α : ℝ) (L j : ℕ) (hj : j < L) (h1 : j ≠ 0) (h2 : j + 1 ≠ L) :
    ch05b_Q0E α L j j = (1 + α ^ 2) / (1 - α ^ 2) := by
  unfold ch05b_Q0E; rw [if_neg (by omega), if_pos rfl, if_neg (by omega)]

theorem ch05b_Q0E_up (α : ℝ) (L i j : ℕ) (hi : i < L) (hj : j < L) (h : i + 1 = j) :
    ch05b_Q0E α L i j = -α / (1 - α ^ 2) := by
  unfold ch05b_Q0E
  rw [if_neg (by omega), if_neg (by omega), if_pos (Or.inl h)]

theorem ch05b_Q0E_down (α : ℝ) (L i j : ℕ) (hi : i < L) (hj : j < L) (h : j + 1 = i) :
    ch05b_Q0E α L i j = -α / (1 - α ^ 2) := by
  unfold ch05b_Q0E
  rw [if_neg (by omega), if_neg (by omega), if_pos (Or.inr h)]

theorem ch05b_Sig0_Q0_row (α : ℝ) (L : ℕ) (hα : 1 - α ^ 2 ≠ 0) (hL : 2 ≤ L) (i j : ℕ)
    (hi : i < L) (hj : j < L) :
    ∑ k ∈ Finset.range L, ch05b_Sig0E α i k * ch05b_Q0E α L k j = if i = j then 1 else 0 := by
  have hout : ∀ k, L ≤ k → ch05b_Sig0E α i k * ch05b_Q0E α L k j = 0 := by
    intro k hk
    rw [ch05b_Q0E_out α L k j (Or.inl hk), mul_zero]
  rcases Nat.eq_zero_or_pos j with hj0 | hj0
  · -- first column
    subst hj0
    have h2 : ∀ k, 1 < k → ch05b_Sig0E α i k * ch05b_Q0E α L k 0 = 0 := by
      intro k hk
      rw [ch05b_Q0E_far α L k 0 (by omega) (by omega) (by omega), mul_zero]
    rw [ch05b_sum_band0 L _ h2 hout, ch05b_Q0E_diag_bd α L 0 (by omega) (Or.inl rfl),
      ch05b_Q0E_down α L 1 0 (by omega) (by omega) rfl]
    rcases Nat.eq_zero_or_pos i with hi0 | hi0
    · rw [if_pos (by omega), ch05b_Sig0E_eq α i 0 0 (by omega),
        ch05b_Sig0E_eq α i 1 1 (by omega)]
      field_simp <;> ring
    · obtain ⟨d, hd⟩ : ∃ d, i = d + 1 := ⟨i - 1, by omega⟩
      rw [if_neg (by omega), ch05b_Sig0E_eq α i 0 (d + 1) (by omega),
        ch05b_Sig0E_eq α i 1 d (by omega)]
      field_simp <;> ring
  · -- column j = q+1
    obtain ⟨q, hq⟩ : ∃ q, j = q + 1 := ⟨j - 1, by omega⟩
    subst hq
    have h1 : ∀ k, k < q → ch05b_Sig0E α i k * ch05b_Q0E α L k (q + 1) = 0 := by
      intro k hk
      rw [ch05b_Q0E_far α L k (q + 1) (by omega) (by omega) (by omega), mul_zero]
    have h2 : ∀ k, q + 2 < k → ch05b_Sig0E α i k * ch05b_Q0E α L k (q + 1) = 0 := by
      intro k hk
      rw [ch05b_Q0E_far α L k (q + 1) (by omega) (by omega) (by omega), mul_zero]
    rw [ch05b_sum_band L q _ h1 h2 hout, ch05b_Q0E_up α L q (q + 1) (by omega) (by omega) rfl]
    rcases Nat.lt_or_ge (q + 2) L with hqL | hqL
    · -- interior column
      rw [ch05b_Q0E_diag_int α L (q + 1) (by omega) (by omega) (by omega),
        ch05b_Q0E_down α L (q + 2) (q + 1) (by omega) (by omega) rfl]
      rcases Nat.lt_or_ge i q with hlt | hge
      · obtain ⟨e, he⟩ : ∃ e, q = i + 1 + e := ⟨q - i - 1, by omega⟩
        rw [if_neg (by omega), ch05b_Sig0E_eq α i q (1 + e) (by omega),
          ch05b_Sig0E_eq α i (q + 1) (2 + e) (by omega),
          ch05b_Sig0E_eq α i (q + 2) (3 + e) (by omega)]
        field_simp <;> ring
      · rcases Nat.lt_or_ge i (q + 3) with hs | hs
        · have hcase : i = q ∨ i = q + 1 ∨ i = q + 2 := by omega
          rcases hcase with h | h | h
          · rw [if_neg (by omega), ch05b_Sig0E_eq α i q 0 (by omega),
              ch05b_Sig0E_eq α i (q + 1) 1 (by omega),
              ch05b_Sig0E_eq α i (q + 2) 2 (by omega)]
            field_simp <;> ring
          · rw [if_pos (by omega), ch05b_Sig0E_eq α i q 1 (by omega),
              ch05b_Sig0E_eq α i (q + 1) 0 (by omega),
              ch05b_Sig0E_eq α i (q + 2) 1 (by omega)]
            field_simp <;> ring
          · rw [if_neg (by omega), ch05b_Sig0E_eq α i q 2 (by omega),
              ch05b_Sig0E_eq α i (q + 1) 1 (by omega),
              ch05b_Sig0E_eq α i (q + 2) 0 (by omega)]
            field_simp <;> ring
        · obtain ⟨e, he⟩ : ∃ e, i = q + 3 + e := ⟨i - q - 3, by omega⟩
          rw [if_neg (by omega), ch05b_Sig0E_eq α i q (3 + e) (by omega),
            ch05b_Sig0E_eq α i (q + 1) (2 + e) (by omega),
            ch05b_Sig0E_eq α i (q + 2) (1 + e) (by omega)]
          field_simp <;> ring
    · -- last column
      rw [ch05b_Q0E_diag_bd α L (q + 1) (by omega) (Or.inr (by omega)),
        ch05b_Q0E_out α L (q + 2) (q + 1) (Or.inl (by omega))]
      rcases Nat.lt_or_ge i q with hlt | hge
      · obtain ⟨e, he⟩ : ∃ e, q = i + 1 + e := ⟨q - i - 1, by omega⟩
        rw [if_neg (by omega), ch05b_Sig0E_eq α i q (1 + e) (by omega),
          ch05b_Sig0E_eq α i (q + 1) (2 + e) (by omega)]
        field_simp <;> ring
      · have hcase : i = q ∨ i = q + 1 := by omega
        rcases hcase with h | h
        · rw [if_neg (by omega), ch05b_Sig0E_eq α i q 0 (by omega),
            ch05b_Sig0E_eq α i (q + 1) 1 (by omega)]
          field_simp <;> ring
        · rw [if_pos (by omega), ch05b_Sig0E_eq α i q 1 (by omega),
            ch05b_Sig0E_eq α i (q + 1) 0 (by omega)]
          field_simp <;> ring

theorem ch05b_Sig0_mul_Q0 (α : ℝ) (L : ℕ) (hα : 1 - α ^ 2 ≠ 0) (hL : 2 ≤ L) :
    ch05b_Sig0 α L * ch05b_Q0 α L = (1 : Matrix (Fin L) (Fin L) ℝ) := by
  ext i j
  rw [Matrix.mul_apply]
  simp only [ch05b_Sig0_apply, ch05b_Q0_apply]
  rw [Fin.sum_univ_eq_sum_range
      (fun k => ch05b_Sig0E α (i : ℕ) k * ch05b_Q0E α L k (j : ℕ)) L,
    ch05b_Sig0_Q0_row α L hα hL i j i.isLt j.isLt, Matrix.one_apply]
  by_cases hij : (i : ℕ) = (j : ℕ)
  · rw [if_pos hij, if_pos (Fin.eq_of_val_eq hij)]
  · rw [if_neg hij, if_neg (fun h : i = j => hij (by rw [h]))]

/-- `Q₀ = Σ₀⁻¹` in closed form, for every chain length `L ≥ 2`. -/
theorem ch05b_Sig0_inv_eq_Q0 (α : ℝ) (L : ℕ) (hα : 1 - α ^ 2 ≠ 0) (hL : 2 ≤ L) :
    (ch05b_Sig0 α L)⁻¹ = ch05b_Q0 α L :=
  Matrix.inv_eq_right_inv (ch05b_Sig0_mul_Q0 α L hα hL)

theorem ch05b_Q0_banded (α : ℝ) (L : ℕ) : ch05b_Banded (ch05b_Q0 α L) 1 := by
  intro i j hij
  rw [ch05b_Q0_apply]
  exact ch05b_Q0E_far α L _ _ (by omega) (by omega) (by omega)

/-- **noname-25, general `L`.**  `Q₀ = Σ₀⁻¹` couples frames at distance at most one:
the AR(1) precision matrix is tridiagonal for every chain length. -/
theorem ch05b_noname_Q0_banded (α : ℝ) (L : ℕ) (hα : 1 - α ^ 2 ≠ 0) :
    ch05b_Banded ((ch05b_Sig0 α L)⁻¹) 1 := by
  rcases Nat.lt_or_ge L 2 with hL | hL
  · intro i j hij
    exfalso
    have h1 := i.isLt
    have h2 := j.isLt
    omega
  · rw [ch05b_Sig0_inv_eq_Q0 α L hα hL]
    exact ch05b_Q0_banded α L

/-! ## Losing tridiagonality immediately: `Q_t` is dense for every chain length -/

theorem ch05b_abs_lt_one {α : ℝ} (h : α ^ 2 < 1) : |α| < 1 := by
  nlinarith [sq_abs α, abs_nonneg α]

theorem ch05b_dom_aux (c x : ℝ) (hc : 0 < c) (hx : 0 ≤ x) (hx1 : x < 1) :
    c * (x / (1 - x ^ 2)) + c * (x / (1 - x ^ 2)) < 1 + c * ((1 + x ^ 2) / (1 - x ^ 2)) := by
  have hd : (0:ℝ) < 1 - x ^ 2 := by nlinarith
  have h : (1 + c * ((1 + x ^ 2) / (1 - x ^ 2))) - (c * (x / (1 - x ^ 2)) + c * (x / (1 - x ^ 2)))
      = ((1 - x ^ 2) + c * (1 - x) ^ 2) / (1 - x ^ 2) := by
    field_simp
    ring
  have hnum : (0:ℝ) < (1 - x ^ 2) + c * (1 - x) ^ 2 := by
    nlinarith [mul_nonneg hc.le (sq_nonneg (1 - x))]
  have hq := div_pos hnum hd
  linarith

theorem ch05b_dom_aux' (c x : ℝ) (hc : 0 < c) (hx : 0 ≤ x) (hx1 : x < 1) :
    c * (x / (1 - x ^ 2)) < 1 + c * (1 / (1 - x ^ 2)) := by
  have hd : (0:ℝ) < 1 - x ^ 2 := by nlinarith
  have h : (1 + c * (1 / (1 - x ^ 2))) - c * (x / (1 - x ^ 2))
      = ((1 - x ^ 2) + c * (1 - x)) / (1 - x ^ 2) := by
    field_simp
    ring
  have hnum : (0:ℝ) < (1 - x ^ 2) + c * (1 - x) := by nlinarith
  have hq := div_pos hnum hd
  linarith

/-! ### Strict diagonal dominance forces all principal minors to be positive -/

theorem ch05b_th_pos (a b : ℕ → ℝ) (L : ℕ) (h0 : |b 0| < a 0)
    (hdom : ∀ r, 0 < r → r < L → |b (r - 1)| + |b r| < a r) :
    ∀ k, k < L → 0 < ch05b_th a b (k + 1) ∧
      |b k| * ch05b_th a b (k + 1) < ch05b_th a b (k + 2) := by
  intro k
  induction k with
  | zero =>
      intro _
      rw [ch05b_th_one, ch05b_th_two]
      exact ⟨one_pos, by rw [mul_one]; exact h0⟩
  | succ k ih =>
      intro hk
      obtain ⟨hpos, hlt⟩ := ih (by omega)
      have hpos2 : 0 < ch05b_th a b (k + 2) :=
        lt_of_le_of_lt (mul_nonneg (abs_nonneg _) hpos.le) hlt
      refine ⟨hpos2, ?_⟩
      have hd := hdom (k + 1) (by omega) (by omega)
      rw [show k + 1 - 1 = k from rfl] at hd
      rw [show k + 1 + 2 = k + 3 from rfl, ch05b_th_rec, show k + 1 + 1 = k + 2 from rfl]
      nlinarith [mul_pos (sub_pos.2 hd) hpos2, mul_nonneg (abs_nonneg (b k)) (sub_pos.2 hlt).le,
        sq_abs (b k), abs_nonneg (b (k + 1))]

theorem ch05b_th_pos_top (a b : ℕ → ℝ) (L : ℕ) (hL : 0 < L) (h0 : |b 0| < a 0)
    (hdom : ∀ r, 0 < r → r < L → |b (r - 1)| + |b r| < a r) :
    0 < ch05b_th a b (L + 1) := by
  obtain ⟨r, hr⟩ : ∃ r, L = r + 1 := ⟨L - 1, by omega⟩
  subst hr
  obtain ⟨hpos, hlt⟩ := ch05b_th_pos a b (r + 1) h0 hdom r (by omega)
  exact lt_of_le_of_lt (mul_nonneg (abs_nonneg _) hpos.le) hlt

theorem ch05b_ps_pos (a b : ℕ → ℝ) (L : ℕ) (h0 : |b 0| < a 0)
    (hdom : ∀ r, 0 < r → r < L → |b (r - 1)| + |b r| < a r) :
    ∀ k, k < L → 0 ≤ ch05b_ps a b L k ∧
      |b (L - k - 1)| * ch05b_ps a b L k < ch05b_ps a b L (k + 1) := by
  intro k
  induction k with
  | zero =>
      intro _
      rw [ch05b_ps_zero, ch05b_ps_one, mul_zero]
      exact ⟨le_refl 0, one_pos⟩
  | succ k ih =>
      intro hk
      obtain ⟨hnn, hlt⟩ := ih (by omega)
      have hpos : 0 < ch05b_ps a b L (k + 1) :=
        lt_of_le_of_lt (mul_nonneg (abs_nonneg _) hnn) hlt
      refine ⟨hpos.le, ?_⟩
      obtain ⟨s, hs⟩ : ∃ s, L = k + 2 + s := ⟨L - k - 2, by omega⟩
      rw [show L - (k + 1) - 1 = s from by omega, ch05b_ps_rec,
        show L - 1 - k = s + 1 from by omega]
      rw [show L - k - 1 = s + 1 from by omega] at hlt
      have hd := hdom (s + 1) (by omega) (by omega)
      rw [show s + 1 - 1 = s from rfl] at hd
      nlinarith [mul_pos (sub_pos.2 hd) hpos,
        mul_nonneg (abs_nonneg (b (s + 1))) (sub_pos.2 hlt).le, sq_abs (b (s + 1)),
        abs_nonneg (b s)]

theorem ch05b_ps_pos_succ (a b : ℕ → ℝ) (L : ℕ) (h0 : |b 0| < a 0)
    (hdom : ∀ r, 0 < r → r < L → |b (r - 1)| + |b r| < a r) (k : ℕ) (hk : k < L) :
    0 < ch05b_ps a b L (k + 1) := by
  obtain ⟨hnn, hlt⟩ := ch05b_ps_pos a b L h0 hdom k hk
  exact lt_of_le_of_lt (mul_nonneg (abs_nonneg _) hnn) hlt

/-! ### The resolvent `B = I + c Q₀` of the AR(1) chain is tridiagonal and dominant -/

/-- Diagonal of `B = I + c Q₀`. -/
def ch05b_aB (α c : ℝ) (L : ℕ) : ℕ → ℝ :=
  fun k => 1 + c * ((if k = 0 ∨ k + 1 = L then 1 else 1 + α ^ 2) / (1 - α ^ 2))

/-- Off-diagonal of `B = I + c Q₀` (the unused slot `k = L-1` is set to `0`). -/
def ch05b_bB (α c : ℝ) (L : ℕ) : ℕ → ℝ :=
  fun k => if k + 1 = L then 0 else c * (-α / (1 - α ^ 2))

theorem ch05b_bB_zero (α c : ℝ) (L k : ℕ) (h : k + 1 = L) : ch05b_bB α c L k = 0 := by
  unfold ch05b_bB; rw [if_pos h]

theorem ch05b_bB_abs (α c : ℝ) (L k : ℕ) (hc : 0 < c) (hα : (0:ℝ) < 1 - α ^ 2) (h : k + 1 ≠ L) :
    |ch05b_bB α c L k| = c * (|α| / (1 - α ^ 2)) := by
  unfold ch05b_bB
  rw [if_neg h, abs_mul, abs_div, abs_neg, abs_of_pos hc, abs_of_pos hα]

theorem ch05b_bB_ne (α c : ℝ) (L k : ℕ) (hc : 0 < c) (hα0 : α ≠ 0) (hα : (1:ℝ) - α ^ 2 ≠ 0)
    (h : k + 1 ≠ L) : ch05b_bB α c L k ≠ 0 := by
  unfold ch05b_bB
  rw [if_neg h]
  exact mul_ne_zero (ne_of_gt hc) (div_ne_zero (neg_ne_zero.2 hα0) hα)

theorem ch05b_B_eq_tri (α c : ℝ) (L : ℕ) :
    (1 : Matrix (Fin L) (Fin L) ℝ) + c • ch05b_Q0 α L
      = ch05b_tri (ch05b_aB α c L) (ch05b_bB α c L) L := by
  ext i j
  have hi : (i : ℕ) < L := i.isLt
  have hj : (j : ℕ) < L := j.isLt
  have hone : (1 : Matrix (Fin L) (Fin L) ℝ) i j = if (i : ℕ) = (j : ℕ) then 1 else 0 := by
    rw [Matrix.one_apply]
    by_cases h : i = j
    · rw [if_pos h, if_pos (by rw [h])]
    · rw [if_neg h, if_neg (fun hh => h (Fin.eq_of_val_eq hh))]
  rw [Matrix.add_apply, Matrix.smul_apply, smul_eq_mul, hone, ch05b_Q0_apply, ch05b_tri_apply]
  rcases eq_or_ne (i : ℕ) (j : ℕ) with h | h
  · rw [h, if_pos rfl, ch05b_triE_self]
    by_cases hb : (j : ℕ) = 0 ∨ (j : ℕ) + 1 = L
    · rw [ch05b_Q0E_diag_bd α L (j : ℕ) hj hb]
      unfold ch05b_aB
      rw [if_pos hb]
    · rw [ch05b_Q0E_diag_int α L (j : ℕ) hj (by omega) (by omega)]
      unfold ch05b_aB
      rw [if_neg hb]
  · rcases eq_or_ne ((i : ℕ) + 1) (j : ℕ) with h2 | h2
    · rw [if_neg h, ch05b_triE_up _ _ _ _ h2, ch05b_Q0E_up α L _ _ hi hj h2]
      unfold ch05b_bB
      rw [if_neg (by omega)]
      ring
    · rcases eq_or_ne ((j : ℕ) + 1) (i : ℕ) with h3 | h3
      · rw [if_neg h, ch05b_triE_down _ _ _ _ h3, ch05b_Q0E_down α L _ _ hi hj h3]
        unfold ch05b_bB
        rw [if_neg (by omega)]
        ring
      · rw [if_neg h, ch05b_triE_eq_zero _ _ h h2 h3, ch05b_Q0E_far α L _ _ h h2 h3]
        ring

theorem ch05b_B_dom0 (α c : ℝ) (L : ℕ) (hc : 0 < c) (hα2 : α ^ 2 < 1) :
    |ch05b_bB α c L 0| < ch05b_aB α c L 0 := by
  have hab : |α| < 1 := ch05b_abs_lt_one hα2
  have hd : (0:ℝ) < 1 - α ^ 2 := by linarith
  by_cases h : (0 : ℕ) + 1 = L
  · rw [ch05b_bB_zero α c L 0 h, abs_zero]
    unfold ch05b_aB
    rw [if_pos (Or.inl rfl)]
    have := mul_pos hc (div_pos one_pos hd)
    linarith
  · rw [ch05b_bB_abs α c L 0 hc hd h]
    unfold ch05b_aB
    rw [if_pos (Or.inl rfl), ← sq_abs α]
    exact ch05b_dom_aux' c |α| hc (abs_nonneg α) hab

theorem ch05b_B_dom (α c : ℝ) (L : ℕ) (hc : 0 < c) (hα2 : α ^ 2 < 1) :
    ∀ r, 0 < r → r < L →
      |ch05b_bB α c L (r - 1)| + |ch05b_bB α c L r| < ch05b_aB α c L r := by
  intro r hr hrL
  have hab : |α| < 1 := ch05b_abs_lt_one hα2
  have hd : (0:ℝ) < 1 - α ^ 2 := by linarith
  rw [ch05b_bB_abs α c L (r - 1) hc hd (by omega)]
  by_cases h : r + 1 = L
  · rw [ch05b_bB_zero α c L r h, abs_zero, add_zero]
    unfold ch05b_aB
    rw [if_pos (Or.inr h), ← sq_abs α]
    exact ch05b_dom_aux' c |α| hc (abs_nonneg α) hab
  · rw [ch05b_bB_abs α c L r hc hd h]
    unfold ch05b_aB
    rw [if_neg (by omega), ← sq_abs α]
    exact ch05b_dom_aux c |α| hc (abs_nonneg α) hab

/-! ### Every off-diagonal entry of `Q_t` is non-zero -/

/-- **The "immediately" claim of tex 1543-1580, for every chain length `L`.**  For a
non-degenerate AR(1) chain (`α ≠ 0`, `α² < 1`) and every `t > 0`, *every* off-diagonal
entry of the marginal precision `Q_t = Σ_t⁻¹` is non-zero: the tridiagonal structure of
`Q₀` is lost completely at every positive diffusion time.  This is the general-`L` form of
the boxed contrast `J` tridiagonal vs `Q_t` dense. -/
theorem ch05b_Qt_offdiag_ne_zero (α t : ℝ) (L : ℕ) (ht : 0 < t) (hα2 : α ^ 2 < 1)
    (hα0 : α ≠ 0) (i j : Fin L) (hij : i ≠ j) :
    ch05b_Qt (ch05b_Sig0 α L) t i j ≠ 0 := by
  have hne : (i : ℕ) ≠ (j : ℕ) := fun h => hij (Fin.eq_of_val_eq h)
  have hi : (i : ℕ) < L := i.isLt
  have hj : (j : ℕ) < L := j.isLt
  have hL : 2 ≤ L := by omega
  have hc : 0 < ch05b_c t := ch05b_c_pos ht
  have hd : (0:ℝ) < 1 - α ^ 2 := by linarith
  have hdne : (1:ℝ) - α ^ 2 ≠ 0 := ne_of_gt hd
  have hQ0 : (ch05b_Sig0 α L)⁻¹ = ch05b_Q0 α L := ch05b_Sig0_inv_eq_Q0 α L hdne hL
  have hBtri : (1 : Matrix (Fin L) (Fin L) ℝ) + ch05b_c t • (ch05b_Sig0 α L)⁻¹
      = ch05b_tri (ch05b_aB α (ch05b_c t) L) (ch05b_bB α (ch05b_c t) L) L := by
    rw [hQ0]; exact ch05b_B_eq_tri α (ch05b_c t) L
  have hdom0 := ch05b_B_dom0 α (ch05b_c t) L hc hα2
  have hdom := ch05b_B_dom α (ch05b_c t) L hc hα2
  have hthpos : 0 < ch05b_th (ch05b_aB α (ch05b_c t) L) (ch05b_bB α (ch05b_c t) L) (L + 1) :=
    ch05b_th_pos_top _ _ L (by omega) hdom0 hdom
  have hdet : IsUnit ((1 : Matrix (Fin L) (Fin L) ℝ) + ch05b_c t • (ch05b_Sig0 α L)⁻¹).det := by
    rw [hBtri, ch05b_det_tri]
    exact isUnit_iff_ne_zero.2 (ne_of_gt hthpos)
  have hS0 : IsUnit (ch05b_Sig0 α L).det :=
    Matrix.isUnit_det_of_right_inverse (ch05b_Sig0_mul_Q0 α L hdne hL)
  -- every entry of `B⁻¹` above the diagonal is non-zero
  have hmain : ∀ p q : Fin L, (p : ℕ) ≤ (q : ℕ) →
      (ch05b_tri (ch05b_aB α (ch05b_c t) L) (ch05b_bB α (ch05b_c t) L) L)⁻¹ p q ≠ 0 := by
    intro p q hpq
    rw [ch05b_g_tridiag_inverse _ _ L (ne_of_gt hthpos) p q hpq]
    have hthp : 0 < ch05b_th (ch05b_aB α (ch05b_c t) L) (ch05b_bB α (ch05b_c t) L)
        ((p : ℕ) + 1) := (ch05b_th_pos _ _ L hdom0 hdom (p : ℕ) p.isLt).1
    have hpsq : 0 < ch05b_ps (ch05b_aB α (ch05b_c t) L) (ch05b_bB α (ch05b_c t) L) L
        (L - (q : ℕ)) := by
      have h := ch05b_ps_pos_succ _ _ L hdom0 hdom (L - (q : ℕ) - 1) (by omega)
      rwa [show L - (q : ℕ) - 1 + 1 = L - (q : ℕ) from by omega] at h
    refine mul_ne_zero (div_ne_zero (mul_ne_zero (pow_ne_zero _ (by norm_num)) ?_)
      (ne_of_gt hthpos)) ?_
    · exact mul_ne_zero (ne_of_gt hthp) (ne_of_gt hpsq)
    · refine Finset.prod_ne_zero_iff.2 ?_
      intro k hk
      simp only [Finset.mem_Ico] at hk
      have hq := q.isLt
      exact ch05b_bB_ne α (ch05b_c t) L k hc hα0 hdne (by omega)
  have hkey : (ch05b_tri (ch05b_aB α (ch05b_c t) L) (ch05b_bB α (ch05b_c t) L) L)⁻¹ i j ≠ 0 := by
    rcases le_total (i : ℕ) (j : ℕ) with h | h
    · exact hmain i j h
    · have h1 := Matrix.transpose_nonsing_inv
        (ch05b_tri (ch05b_aB α (ch05b_c t) L) (ch05b_bB α (ch05b_c t) L) L)
      have h2 : (ch05b_tri (ch05b_aB α (ch05b_c t) L) (ch05b_bB α (ch05b_c t) L) L)ᵀ
          = ch05b_tri (ch05b_aB α (ch05b_c t) L) (ch05b_bB α (ch05b_c t) L) L :=
        ch05b_tri_isSymm _ _ L
      have hsym : (ch05b_tri (ch05b_aB α (ch05b_c t) L) (ch05b_bB α (ch05b_c t) L) L)⁻¹ i j
          = (ch05b_tri (ch05b_aB α (ch05b_c t) L) (ch05b_bB α (ch05b_c t) L) L)⁻¹ j i := by
        calc (ch05b_tri (ch05b_aB α (ch05b_c t) L) (ch05b_bB α (ch05b_c t) L) L)⁻¹ i j
            = ((ch05b_tri (ch05b_aB α (ch05b_c t) L) (ch05b_bB α (ch05b_c t) L) L)⁻¹)ᵀ j i := rfl
          _ = (((ch05b_tri (ch05b_aB α (ch05b_c t) L) (ch05b_bB α (ch05b_c t) L) L)ᵀ)⁻¹) j i := by
              rw [h1]
          _ = (ch05b_tri (ch05b_aB α (ch05b_c t) L) (ch05b_bB α (ch05b_c t) L) L)⁻¹ j i := by
              rw [h2]
      rw [hsym]
      exact hmain j i h
  rw [ch05b_g_Qt_inverse_form (ch05b_Sig0 α L) t (ne_of_gt hc) hS0 hdet, Matrix.smul_apply,
    Matrix.sub_apply, Matrix.one_apply_ne hij, smul_eq_mul, hBtri, zero_sub]
  exact mul_ne_zero (div_ne_zero (ne_of_gt (Real.exp_pos _)) (ne_of_gt hc))
    (neg_ne_zero.2 hkey)

/-- `J = Q₀ + γ_t I` is tridiagonal for every chain length: the observation channel only
adds to the diagonal, so the local Markov graph of the clean chain survives. -/
theorem ch05b_J_banded (α t : ℝ) (L : ℕ) (hα : (1:ℝ) - α ^ 2 ≠ 0) :
    ch05b_Banded (ch05b_J (ch05b_Sig0 α L) t) 1 := by
  intro i j hij
  have hne : i ≠ j := by
    intro h
    subst h
    omega
  simp only [ch05b_J, Matrix.add_apply, Matrix.smul_apply, Matrix.one_apply_ne hne, smul_zero,
    add_zero]
  exact ch05b_noname_Q0_banded α L hα i j hij

/-- **noname-11 (tex 1009-1021), for every chain length.**  The boxed contrast: the
posterior precision `J` of `a | x` is tridiagonal, while the marginal precision `Q_t` of
`x` has *no* vanishing off-diagonal entry at all — for every `L`, every `t > 0` and every
non-degenerate AR(1) coupling. -/
theorem ch05b_noname_J_tridiag_Qt_dense_gen (α t : ℝ) (L : ℕ) (ht : 0 < t) (hα2 : α ^ 2 < 1)
    (hα0 : α ≠ 0) :
    ch05b_Banded (ch05b_J (ch05b_Sig0 α L) t) 1 ∧
      ∀ i j : Fin L, i ≠ j → ch05b_Qt (ch05b_Sig0 α L) t i j ≠ 0 :=
  ⟨ch05b_J_banded α t L (ne_of_gt (by linarith : (0:ℝ) < 1 - α ^ 2)),
    fun i j hij => ch05b_Qt_offdiag_ne_zero α t L ht hα2 hα0 i j hij⟩

/-! ## Pass-4 gap closing

Three additions, all at arbitrary chain length `L = n`:

* **thm:g-decouple** (tex 1220-1255), the chapter's central Decoupling theorem;
* quantitative forms of **eq:g-Qt-smallt**, **eq:g-band-fill-order** and **eq:g-bandfill**,
  whose Neumann remainders were previously exact but unbounded;
* **noname-frob-normalisation** (tex 1641-1643), the clean nearest-neighbour Frobenius mass.

Throughout, the spectral data `(U, ω)` is exactly what `ch05b_g_spectral` produces for a
symmetric `Σ₀`, and `0 < m ≤ ωᵢ` is the thesis' standing `Σ₀ ≻ 0`.
-/

/-! ### Orthogonal conjugation toolkit -/

theorem ch05b_orth_right (U : Matrix (Fin n) (Fin n) ℝ) (hU : Uᵀ * U = 1) : U * Uᵀ = 1 := by
  have hd : IsUnit U.det := by
    have h := congrArg Matrix.det hU
    rw [Matrix.det_mul, Matrix.det_transpose, Matrix.det_one] at h
    exact isUnit_iff_exists_inv.mpr ⟨U.det, h⟩
  rw [← Matrix.inv_eq_left_inv hU, Matrix.mul_nonsing_inv _ hd]

theorem ch05b_orth_cancel (U : Matrix (Fin n) (Fin n) ℝ) (hU : Uᵀ * U = 1)
    (X : Matrix (Fin n) (Fin n) ℝ) : Uᵀ * (U * X) = X := by
  rw [← Matrix.mul_assoc, hU, Matrix.one_mul]

/-- An orthogonal map preserves the inner product: this is the "rigid rotation" of
tex 1216-1218. -/
theorem ch05b_orth_dot (U : Matrix (Fin n) (Fin n) ℝ) (hU : Uᵀ * U = 1) (a b : Fin n → ℝ) :
    (U *ᵥ a) ⬝ᵥ (U *ᵥ b) = a ⬝ᵥ b := by
  first
  | rw [dotProduct_mulVec, ← Matrix.mulVec_transpose, Matrix.mulVec_mulVec, hU,
      Matrix.one_mulVec]
  | simp [dotProduct_mulVec, ← Matrix.mulVec_transpose, hU]

theorem ch05b_conj_mul (U : Matrix (Fin n) (Fin n) ℝ) (hU : Uᵀ * U = 1) (v w : Fin n → ℝ) :
    (U * Matrix.diagonal v * Uᵀ) * (U * Matrix.diagonal w * Uᵀ)
      = U * Matrix.diagonal (fun i => v i * w i) * Uᵀ := by
  calc (U * Matrix.diagonal v * Uᵀ) * (U * Matrix.diagonal w * Uᵀ)
      = U * (Matrix.diagonal v * (Uᵀ * (U * (Matrix.diagonal w * Uᵀ)))) := by
        simp only [Matrix.mul_assoc]
    _ = U * (Matrix.diagonal v * (Matrix.diagonal w * Uᵀ)) := by rw [ch05b_orth_cancel U hU]
    _ = U * (Matrix.diagonal v * Matrix.diagonal w * Uᵀ) := by simp only [Matrix.mul_assoc]
    _ = U * Matrix.diagonal (fun i => v i * w i) * Uᵀ := by
        rw [Matrix.diagonal_mul_diagonal, Matrix.mul_assoc]

theorem ch05b_conj_smul (U : Matrix (Fin n) (Fin n) ℝ) (c : ℝ) (v : Fin n → ℝ) :
    c • (U * Matrix.diagonal v * Uᵀ) = U * Matrix.diagonal (fun i => c * v i) * Uᵀ := by
  have hd : Matrix.diagonal (fun i => c * v i) = c • Matrix.diagonal v := by
    ext i j
    by_cases h : i = j <;> simp [Matrix.diagonal_apply, h]
  rw [hd, Matrix.mul_smul, Matrix.smul_mul]

theorem ch05b_conj_sub (U : Matrix (Fin n) (Fin n) ℝ) (v w : Fin n → ℝ) :
    U * Matrix.diagonal v * Uᵀ - U * Matrix.diagonal w * Uᵀ
      = U * Matrix.diagonal (fun i => v i - w i) * Uᵀ := by
  have hd : Matrix.diagonal (fun i => v i - w i) = Matrix.diagonal v - Matrix.diagonal w := by
    ext i j
    by_cases h : i = j <;> simp [Matrix.diagonal_apply, h]
  rw [hd, Matrix.mul_sub, Matrix.sub_mul]

theorem ch05b_conj_pow (U : Matrix (Fin n) (Fin n) ℝ) (hU : Uᵀ * U = 1) (v : Fin n → ℝ) :
    ∀ k : ℕ, (U * Matrix.diagonal v * Uᵀ) ^ k = U * Matrix.diagonal (fun i => v i ^ k) * Uᵀ := by
  intro k
  induction k with
  | zero =>
      have hone : (fun i => v i ^ 0) = (1 : Fin n → ℝ) := by funext i; simp
      have hd1 : Matrix.diagonal (1 : Fin n → ℝ) = (1 : Matrix (Fin n) (Fin n) ℝ) := by
        ext a b
        by_cases h : a = b <;> simp [Matrix.diagonal_apply, Matrix.one_apply, h]
      rw [pow_zero, hone, hd1, Matrix.mul_one]
      exact (ch05b_orth_right U hU).symm
  | succ k ih =>
      have hfun : (fun i => v i ^ k * v i) = (fun i => v i ^ (k + 1)) := by
        funext i; rw [pow_succ]
      rw [pow_succ, ih, ch05b_conj_mul U hU, hfun]

theorem ch05b_conj_det (U : Matrix (Fin n) (Fin n) ℝ) (hU : Uᵀ * U = 1) (v : Fin n → ℝ) :
    (U * Matrix.diagonal v * Uᵀ).det = ∏ i, v i := by
  have hdetU : U.det * U.det = 1 := by
    have h := congrArg Matrix.det hU
    rw [Matrix.det_mul, Matrix.det_transpose, Matrix.det_one] at h
    exact h
  rw [Matrix.det_mul, Matrix.det_mul, Matrix.det_diagonal, Matrix.det_transpose]
  linear_combination (∏ i, v i) * hdetU

/-- Every entry of an orthogonal matrix is bounded by `1` (its rows are unit vectors). -/
theorem ch05b_orth_entry_le_one (U : Matrix (Fin n) (Fin n) ℝ) (hU : Uᵀ * U = 1) (i k : Fin n) :
    |U i k| ≤ 1 := by
  have hUU : U * Uᵀ = 1 := ch05b_orth_right U hU
  have hrow : ∑ l, U i l * U i l = 1 := by
    have h : (U * Uᵀ) i i = (1 : Matrix (Fin n) (Fin n) ℝ) i i := by rw [hUU]
    rw [Matrix.mul_apply] at h
    simpa [Matrix.transpose_apply, Matrix.one_apply_eq] using h
  have hle : U i k * U i k ≤ ∑ l, U i l * U i l :=
    Finset.single_le_sum (f := fun l => U i l * U i l) (fun l _ => mul_self_nonneg _)
      (Finset.mem_univ k)
  rw [hrow] at hle
  nlinarith [abs_nonneg (U i k), sq_abs (U i k), hle]

/-- Entrywise bound for a matrix diagonal in an orthonormal basis. -/
theorem ch05b_conj_entry_bound (U : Matrix (Fin n) (Fin n) ℝ) (hU : Uᵀ * U = 1)
    (v : Fin n → ℝ) (M : ℝ) (hv : ∀ k, |v k| ≤ M) (i j : Fin n) :
    |(U * Matrix.diagonal v * Uᵀ) i j| ≤ (n : ℝ) * M := by
  have hentry : (U * Matrix.diagonal v * Uᵀ) i j = ∑ k, U i k * v k * U j k := by
    rw [Matrix.mul_apply]
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [Matrix.mul_diagonal, Matrix.transpose_apply]
  have hterm : ∀ k : Fin n, |U i k * v k * U j k| ≤ M := by
    intro k
    have h1 := ch05b_orth_entry_le_one U hU i k
    have h2 := ch05b_orth_entry_le_one U hU j k
    have h3 := hv k
    have hM : (0 : ℝ) ≤ M := le_trans (abs_nonneg _) h3
    have habs : |U i k * v k * U j k| = |U i k| * |v k| * |U j k| := by rw [abs_mul, abs_mul]
    rw [habs]
    have t1 : |U i k| * |v k| ≤ 1 * M := mul_le_mul h1 h3 (abs_nonneg _) zero_le_one
    have t2 : |U i k| * |v k| * |U j k| ≤ 1 * M * 1 :=
      mul_le_mul t1 h2 (abs_nonneg _) (by linarith)
    linarith
  rw [hentry]
  calc |∑ k, U i k * v k * U j k| ≤ ∑ k, |U i k * v k * U j k| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _k : Fin n, M := Finset.sum_le_sum fun k _ => hterm k
    _ = (n : ℝ) * M := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

/-- The Frobenius norm is unitarily invariant. -/
theorem ch05b_frobSq_conj (U : Matrix (Fin n) (Fin n) ℝ) (hU : Uᵀ * U = 1) (v : Fin n → ℝ) :
    ch05b_frobSq (U * Matrix.diagonal v * Uᵀ) = ∑ i, (v i) ^ 2 := by
  rw [ch05b_g_frob]
  have hsymm : (U * Matrix.diagonal v * Uᵀ)ᵀ = U * Matrix.diagonal v * Uᵀ := by
    simp [Matrix.transpose_mul, Matrix.mul_assoc]
  rw [hsymm, ch05b_conj_mul U hU]
  calc Matrix.trace (U * Matrix.diagonal (fun i => v i * v i) * Uᵀ)
      = Matrix.trace (U * (Matrix.diagonal (fun i => v i * v i) * Uᵀ)) := by
        rw [Matrix.mul_assoc]
    _ = Matrix.trace (Matrix.diagonal (fun i => v i * v i) * Uᵀ * U) := Matrix.trace_mul_comm _ _
    _ = Matrix.trace (Matrix.diagonal (fun i => v i * v i)) := by
        rw [Matrix.mul_assoc, hU, Matrix.mul_one]
    _ = ∑ i, (v i) ^ 2 := by
        rw [Matrix.trace_diagonal]
        exact Finset.sum_congr rfl fun i _ => (pow_two (v i)).symm

/-! ### thm:g-decouple (tex 1220-1255): the Decoupling theorem -/

/-- `U` orthogonal has `|det U| = 1`: the change of coordinates `x̃ = Uᵀx` of tex 1216-1218
carries no Jacobian factor, so the density of `x̃` at `x̃` is the density of `x` at `Ux̃`. -/
theorem ch05b_orth_abs_det (U : Matrix (Fin n) (Fin n) ℝ) (hU : Uᵀ * U = 1) : |U.det| = 1 := by
  have h : U.det * U.det = 1 := by
    have h0 := congrArg Matrix.det hU
    rw [Matrix.det_mul, Matrix.det_transpose, Matrix.det_one] at h0
    exact h0
  have h2 : |U.det| ^ 2 = 1 := by rw [sq_abs, pow_two]; exact h
  have h3 : (0 : ℝ) ≤ |U.det| := abs_nonneg _
  have h4 : (|U.det| - 1) * (|U.det| + 1) = 0 := by linear_combination h2
  rcases mul_eq_zero.mp h4 with h5 | h5
  · linarith
  · linarith

/-- The centred multivariate Gaussian density with covariance `S`,
`P(x) = ((2π)ⁿ det S)^{-1/2} exp(-½ xᵀS⁻¹x)`. -/
def ch05b_gaussPDF (S : Matrix (Fin n) (Fin n) ℝ) (x : Fin n → ℝ) : ℝ :=
  (Real.sqrt ((2 * Real.pi) ^ n * S.det))⁻¹ * Real.exp (-((x ⬝ᵥ (S⁻¹ *ᵥ x)) / 2))

/-- First step of the theorem's proof (tex 1236-1238): `Cov(x̃) = UᵀΣ_tU = diag(λᵢ(t))`. -/
theorem ch05b_g_decouple_cov (U : Matrix (Fin n) (Fin n) ℝ) (ω : Fin n → ℝ) (hU : Uᵀ * U = 1)
    (S0 : Matrix (Fin n) (Fin n) ℝ) (hS0 : S0 = U * Matrix.diagonal ω * Uᵀ)
    (t : ℝ) (ht : 0 ≤ t) (hω : ∀ i, 0 < ω i) :
    Uᵀ * ch05b_Sigt S0 t * U = Matrix.diagonal (fun i => ch05b_lam (ω i) t) := by
  rw [(ch05b_g_Qt_spec U ω hU S0 hS0 t ht hω).1]
  calc Uᵀ * (U * Matrix.diagonal (fun i => ch05b_lam (ω i) t) * Uᵀ) * U
      = Uᵀ * (U * (Matrix.diagonal (fun i => ch05b_lam (ω i) t) * (Uᵀ * U))) := by
        simp only [Matrix.mul_assoc]
    _ = Matrix.diagonal (fun i => ch05b_lam (ω i) t) * (Uᵀ * U) := ch05b_orth_cancel U hU _
    _ = Matrix.diagonal (fun i => ch05b_lam (ω i) t) := by rw [hU, Matrix.mul_one]

/-- **thm:g-decouple (i)** (tex 1222-1224), for every chain length.  In the eigen-coordinates
`x̃ = Uᵀx` the noisy law factorises into independent scalar Gaussians,
`P_t(x̃) = ∏ᵢ N(x̃ᵢ; 0, λᵢ(t))`.  `U` is orthogonal, so `|det U| = 1` and the density of
`x̃` at `x̃` is the density of `x` at `Ux̃`; the scalar factors are mathlib's
`gaussianPDFReal`. -/
theorem ch05b_g_decouple_density (U : Matrix (Fin n) (Fin n) ℝ) (ω : Fin n → ℝ)
    (hU : Uᵀ * U = 1) (S0 : Matrix (Fin n) (Fin n) ℝ) (hS0 : S0 = U * Matrix.diagonal ω * Uᵀ)
    (t : ℝ) (ht : 0 ≤ t) (hω : ∀ i, 0 < ω i) (xt : Fin n → ℝ) :
    ch05b_gaussPDF (ch05b_Sigt S0 t) (U *ᵥ xt)
      = ∏ i, ProbabilityTheory.gaussianPDFReal 0 (ch05b_lam (ω i) t).toNNReal (xt i) := by
  have hspec := ch05b_g_Qt_spec U ω hU S0 hS0 t ht hω
  have hpos : ∀ i, 0 < ch05b_lam (ω i) t := fun i => ch05b_lam_pos (hω i) ht
  have hdet : (ch05b_Sigt S0 t).det = ∏ i, ch05b_lam (ω i) t := by
    rw [hspec.1, ch05b_conj_det U hU]
  have hquad : (U *ᵥ xt) ⬝ᵥ ((ch05b_Sigt S0 t)⁻¹ *ᵥ (U *ᵥ xt))
      = ∑ i, xt i ^ 2 / ch05b_lam (ω i) t := by
    have hQ : (ch05b_Sigt S0 t)⁻¹
        = U * Matrix.diagonal (fun i => (ch05b_lam (ω i) t)⁻¹) * Uᵀ := hspec.2
    have hmm : U * Matrix.diagonal (fun i => (ch05b_lam (ω i) t)⁻¹) * Uᵀ * U
        = U * Matrix.diagonal (fun i => (ch05b_lam (ω i) t)⁻¹) := by
      calc U * Matrix.diagonal (fun i => (ch05b_lam (ω i) t)⁻¹) * Uᵀ * U
          = U * Matrix.diagonal (fun i => (ch05b_lam (ω i) t)⁻¹) * (Uᵀ * U) := by
            simp only [Matrix.mul_assoc]
        _ = U * Matrix.diagonal (fun i => (ch05b_lam (ω i) t)⁻¹) := by
            rw [hU, Matrix.mul_one]
    rw [hQ, Matrix.mulVec_mulVec, hmm, ← Matrix.mulVec_mulVec, ch05b_orth_dot U hU]
    simp only [dotProduct, Matrix.mulVec_diagonal]
    exact Finset.sum_congr rfl fun i _ => by rw [div_eq_mul_inv]; ring
  have hterm : ∀ i : Fin n,
      ProbabilityTheory.gaussianPDFReal 0 (ch05b_lam (ω i) t).toNNReal (xt i)
        = (Real.sqrt (2 * Real.pi * ch05b_lam (ω i) t))⁻¹
          * Real.exp (-((xt i ^ 2 / ch05b_lam (ω i) t) / 2)) := by
    intro i
    have harg : -(xt i - 0) ^ 2 / (2 * ch05b_lam (ω i) t)
        = -((xt i ^ 2 / ch05b_lam (ω i) t) / 2) := by
      rw [sub_zero]
      simp only [div_eq_mul_inv, mul_inv]
      ring
    simp only [ProbabilityTheory.gaussianPDFReal, Real.coe_toNNReal _ (hpos i).le, harg]
  have hnn : ∀ i ∈ (Finset.univ : Finset (Fin n)), 0 ≤ 2 * Real.pi * ch05b_lam (ω i) t := by
    intro i _
    have h1 : (0 : ℝ) ≤ 2 * Real.pi := by positivity
    exact mul_nonneg h1 (hpos i).le
  have hconst : (Real.sqrt ((2 * Real.pi) ^ n * ∏ i, ch05b_lam (ω i) t))⁻¹
      = ∏ i, (Real.sqrt (2 * Real.pi * ch05b_lam (ω i) t))⁻¹ := by
    rw [Finset.prod_inv_distrib, ← Real.sqrt_prod _ hnn]
    congr 2
    rw [Finset.prod_mul_distrib, Finset.prod_const, Finset.card_univ, Fintype.card_fin]
  have hexpprod : Real.exp (-((∑ i, xt i ^ 2 / ch05b_lam (ω i) t) / 2))
      = ∏ i, Real.exp (-((xt i ^ 2 / ch05b_lam (ω i) t) / 2)) := by
    rw [← Real.exp_sum]
    congr 1
    rw [Finset.sum_neg_distrib, ← Finset.sum_div]
  rw [ch05b_gaussPDF, hdet, hquad, hconst, hexpprod, ← Finset.prod_mul_distrib]
  exact Finset.prod_congr rfl fun i _ => (hterm i).symm

/-- **thm:g-decouple (ii)** (tex 1225-1227), for every chain length: in the eigenbasis the
score splits into independent scalar components, `s̃ᵢ(x̃,t) = -x̃ᵢ/λᵢ(t)`. -/
theorem ch05b_g_decouple_score (U : Matrix (Fin n) (Fin n) ℝ) (ω : Fin n → ℝ)
    (hU : Uᵀ * U = 1) (S0 : Matrix (Fin n) (Fin n) ℝ) (hS0 : S0 = U * Matrix.diagonal ω * Uᵀ)
    (t : ℝ) (ht : 0 ≤ t) (hω : ∀ i, 0 < ω i) (xt : Fin n → ℝ) (i : Fin n) :
    (Uᵀ *ᵥ ch05b_score S0 t (U *ᵥ xt)) i = -(xt i) / ch05b_lam (ω i) t := by
  have hQ : ch05b_Qt S0 t = U * Matrix.diagonal (fun k => (ch05b_lam (ω k) t)⁻¹) * Uᵀ :=
    (ch05b_g_Qt_spec U ω hU S0 hS0 t ht hω).2
  have hmat : Uᵀ * (U * Matrix.diagonal (fun k => (ch05b_lam (ω k) t)⁻¹) * Uᵀ) * U
      = Matrix.diagonal (fun k => (ch05b_lam (ω k) t)⁻¹) := by
    calc Uᵀ * (U * Matrix.diagonal (fun k => (ch05b_lam (ω k) t)⁻¹) * Uᵀ) * U
        = Uᵀ * (U * (Matrix.diagonal (fun k => (ch05b_lam (ω k) t)⁻¹) * (Uᵀ * U))) := by
          simp only [Matrix.mul_assoc]
      _ = Matrix.diagonal (fun k => (ch05b_lam (ω k) t)⁻¹) * (Uᵀ * U) := ch05b_orth_cancel U hU _
      _ = Matrix.diagonal (fun k => (ch05b_lam (ω k) t)⁻¹) := by rw [hU, Matrix.mul_one]
  have hkey : Uᵀ *ᵥ (ch05b_Qt S0 t *ᵥ (U *ᵥ xt))
      = Matrix.diagonal (fun k => (ch05b_lam (ω k) t)⁻¹) *ᵥ xt := by
    rw [hQ, Matrix.mulVec_mulVec, Matrix.mulVec_mulVec, hmat]
  rw [ch05b_score, Matrix.mulVec_neg, Pi.neg_apply, hkey, Matrix.mulVec_diagonal]
  rw [div_eq_mul_inv]
  ring

/-- **thm:g-decouple (ii)**, "each depending on its own coordinate only": the `i`-th
eigen-component of the score is a function of `x̃ᵢ` alone. -/
theorem ch05b_g_decouple_score_local (U : Matrix (Fin n) (Fin n) ℝ) (ω : Fin n → ℝ)
    (hU : Uᵀ * U = 1) (S0 : Matrix (Fin n) (Fin n) ℝ) (hS0 : S0 = U * Matrix.diagonal ω * Uᵀ)
    (t : ℝ) (ht : 0 ≤ t) (hω : ∀ i, 0 < ω i) (y z : Fin n → ℝ) (i : Fin n) (h : y i = z i) :
    (Uᵀ *ᵥ ch05b_score S0 t (U *ᵥ y)) i = (Uᵀ *ᵥ ch05b_score S0 t (U *ᵥ z)) i := by
  rw [ch05b_g_decouple_score U ω hU S0 hS0 t ht hω y i,
    ch05b_g_decouple_score U ω hU S0 hS0 t ht hω z i, h]

/-- The reverse-time drift of eq:sm-reverse, `b(x,t) = f(x,t) - g(t)²s(x,t)`, for the
linear forward drift `f(x,t) = f_c(t)·x` of the chapter's OU process. -/
def ch05b_revDrift (fc g : ℝ → ℝ) (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ) (x : Fin n → ℝ) :
    Fin n → ℝ :=
  fc t • x - (g t) ^ 2 • ch05b_score S0 t x

/-- The per-mode reverse drift coefficient `κᵢ(t) = f_c(t) + g(t)²/λᵢ(t)`. -/
def ch05b_revCoef (fc g : ℝ → ℝ) (ω t : ℝ) : ℝ := fc t + (g t) ^ 2 / ch05b_lam ω t

/-- **thm:g-decouple (iii)**, drift half (tex 1229-1233): in the eigenbasis the reverse-time
drift field is coordinatewise scalar and linear, `b̃ᵢ(x̃,t) = κᵢ(t)·x̃ᵢ` — one scalar
OU-type drift per mode, coupled to no other coordinate. -/
theorem ch05b_g_decouple_reverse (U : Matrix (Fin n) (Fin n) ℝ) (ω : Fin n → ℝ)
    (hU : Uᵀ * U = 1) (S0 : Matrix (Fin n) (Fin n) ℝ) (hS0 : S0 = U * Matrix.diagonal ω * Uᵀ)
    (t : ℝ) (ht : 0 ≤ t) (hω : ∀ i, 0 < ω i) (fc g : ℝ → ℝ) (xt : Fin n → ℝ) (i : Fin n) :
    (Uᵀ *ᵥ ch05b_revDrift fc g S0 t (U *ᵥ xt)) i = ch05b_revCoef fc g (ω i) t * xt i := by
  have hsc := ch05b_g_decouple_score U ω hU S0 hS0 t ht hω xt i
  have hUU : Uᵀ *ᵥ (U *ᵥ xt) = xt := by
    rw [Matrix.mulVec_mulVec, hU, Matrix.one_mulVec]
  simp only [ch05b_revDrift, ch05b_revCoef, Matrix.mulVec_sub, Matrix.mulVec_smul, Pi.sub_apply,
    Pi.smul_apply, smul_eq_mul, hUU, hsc]
  ring

/-- **thm:g-decouple (iii)**, noise half (tex 1241-1244): "an orthogonal image of white noise
is again white" — `Uᵀ(σ²I)U = σ²I`, so the transformed driving noise is still isotropic,
with no cross-coordinate covariance. -/
theorem ch05b_g_decouple_noise (U : Matrix (Fin n) (Fin n) ℝ) (hU : Uᵀ * U = 1) (σ : ℝ) :
    Uᵀ * (σ ^ 2 • (1 : Matrix (Fin n) (Fin n) ℝ)) * U = σ ^ 2 • (1 : Matrix (Fin n) (Fin n) ℝ) := by
  rw [Matrix.mul_smul, Matrix.mul_one, Matrix.smul_mul, hU]

/-- **thm:g-decouple (iii)**, time-inhomogeneity (tex 1230-1233).  With the chapter's VP
coefficients (`f(x,t) = -x`, `g ≡ √2`) the reverse drift coefficient of mode `i` is
injective in `t ≥ 0` whenever `ωᵢ ≠ 1`: the scalar diffusions really are
time-inhomogeneous, the `t`-dependence entering only through `λᵢ(t)`. -/
theorem ch05b_g_decouple_inhomog (ω : ℝ) (hω : 0 < ω) (hω1 : ω ≠ 1) {t₁ t₂ : ℝ}
    (ht₁ : 0 ≤ t₁) (ht₂ : 0 ≤ t₂)
    (h : ch05b_revCoef (fun _ => -1) (fun _ => Real.sqrt 2) ω t₁
        = ch05b_revCoef (fun _ => -1) (fun _ => Real.sqrt 2) ω t₂) :
    t₁ = t₂ := by
  have h2 : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  simp only [ch05b_revCoef, h2] at h
  have hl1 : 0 < ch05b_lam ω t₁ := ch05b_lam_pos hω ht₁
  have hl2 : 0 < ch05b_lam ω t₂ := ch05b_lam_pos hω ht₂
  have hl1' : ch05b_lam ω t₁ ≠ 0 := ne_of_gt hl1
  have hl2' : ch05b_lam ω t₂ ≠ 0 := ne_of_gt hl2
  have hlam : ch05b_lam ω t₁ = ch05b_lam ω t₂ := by
    have h' : (2 : ℝ) / ch05b_lam ω t₁ = 2 / ch05b_lam ω t₂ := by linarith
    field_simp at h'
    linarith
  have hexp : Real.exp (-2 * t₁) = Real.exp (-2 * t₂) := by
    simp only [ch05b_lam, ch05b_Delta] at hlam
    have hne : ω - 1 ≠ 0 := sub_ne_zero.mpr hω1
    have hz : (Real.exp (-2 * t₁) - Real.exp (-2 * t₂)) * (ω - 1) = 0 := by
      linear_combination hlam
    rcases mul_eq_zero.mp hz with h' | h'
    · linarith
    · exact absurd h' hne
  have h3 : -2 * t₁ = -2 * t₂ := Real.exp_eq_exp.mp hexp
  linarith

/-- **thm:g-decouple (tex 1220-1255)**, all parts together, at arbitrary chain length. -/
theorem ch05b_g_decouple (U : Matrix (Fin n) (Fin n) ℝ) (ω : Fin n → ℝ) (hU : Uᵀ * U = 1)
    (S0 : Matrix (Fin n) (Fin n) ℝ) (hS0 : S0 = U * Matrix.diagonal ω * Uᵀ)
    (t : ℝ) (ht : 0 ≤ t) (hω : ∀ i, 0 < ω i) (fc g : ℝ → ℝ) :
    (Uᵀ * ch05b_Sigt S0 t * U = Matrix.diagonal (fun i => ch05b_lam (ω i) t))
    ∧ (∀ xt, ch05b_gaussPDF (ch05b_Sigt S0 t) (U *ᵥ xt)
        = ∏ i, ProbabilityTheory.gaussianPDFReal 0 (ch05b_lam (ω i) t).toNNReal (xt i))
    ∧ (∀ xt i, (Uᵀ *ᵥ ch05b_score S0 t (U *ᵥ xt)) i = -(xt i) / ch05b_lam (ω i) t)
    ∧ (∀ xt i, (Uᵀ *ᵥ ch05b_revDrift fc g S0 t (U *ᵥ xt)) i
        = ch05b_revCoef fc g (ω i) t * xt i)
    ∧ (∀ σ : ℝ, Uᵀ * (σ ^ 2 • (1 : Matrix (Fin n) (Fin n) ℝ)) * U
        = σ ^ 2 • (1 : Matrix (Fin n) (Fin n) ℝ)) :=
  ⟨ch05b_g_decouple_cov U ω hU S0 hS0 t ht hω,
   fun xt => ch05b_g_decouple_density U ω hU S0 hS0 t ht hω xt,
   fun xt i => ch05b_g_decouple_score U ω hU S0 hS0 t ht hω xt i,
   fun xt i => ch05b_g_decouple_reverse U ω hU S0 hS0 t ht hω fc g xt i,
   fun σ => ch05b_g_decouple_noise U hU σ⟩

/-! ### eq:g-score-matrix: the score as the gradient of the log density (tex 785-796) -/

theorem ch05b_Sigt_isSymm (S0 : Matrix (Fin n) (Fin n) ℝ) (hS0 : S0.IsSymm) (t : ℝ) :
    (ch05b_Sigt S0 t).IsSymm := by
  show (ch05b_Sigt S0 t)ᵀ = ch05b_Sigt S0 t
  simp only [ch05b_Sigt, Matrix.transpose_add, Matrix.transpose_smul, hS0.eq, Matrix.transpose_one]

theorem ch05b_Sigt_inv_isSymm (S0 : Matrix (Fin n) (Fin n) ℝ) (hS0 : S0.IsSymm) (t : ℝ) :
    ((ch05b_Sigt S0 t)⁻¹).IsSymm := by
  show ((ch05b_Sigt S0 t)⁻¹)ᵀ = (ch05b_Sigt S0 t)⁻¹
  rw [Matrix.transpose_nonsing_inv, (ch05b_Sigt_isSymm S0 hS0 t).eq]

/-- `log P_t(x) = -½ xᵀQ_t x + const` (tex 790): the log of the Gaussian density is exactly the
quadratic form, up to the additive normalising constant. -/
theorem ch05b_log_gaussPDF (S : Matrix (Fin n) (Fin n) ℝ)
    (hdet : 0 < (2 * Real.pi) ^ n * S.det) (x : Fin n → ℝ) :
    Real.log (ch05b_gaussPDF S x)
      = -Real.log (Real.sqrt ((2 * Real.pi) ^ n * S.det)) - (x ⬝ᵥ (S⁻¹ *ᵥ x)) / 2 := by
  have hpos : 0 < Real.sqrt ((2 * Real.pi) ^ n * S.det) := Real.sqrt_pos.mpr hdet
  simp only [ch05b_gaussPDF]
  rw [Real.log_mul (inv_ne_zero (ne_of_gt hpos)) (Real.exp_ne_zero _), Real.log_inv, Real.log_exp]
  ring

/-- **eq:g-score-matrix (tex 792-796)**, the gradient claim in exact first-order form: for every
increment `v`,
`log P(x+v) - log P(x) = ⟨-Q x, v⟩ - ½ vᵀQv`,
so the linear part of the increment of the log density is exactly the inner product with `-Qx`
and the entire remainder is the displayed quadratic term.  Stated multiplicatively, so that no
positivity of `det S` is needed. -/
theorem ch05b_g_score_matrix_density (S : Matrix (Fin n) (Fin n) ℝ) (hS : (S⁻¹).IsSymm)
    (x v : Fin n → ℝ) :
    ch05b_gaussPDF S (x + v)
      = ch05b_gaussPDF S x
        * Real.exp ((-(S⁻¹ *ᵥ x)) ⬝ᵥ v - (v ⬝ᵥ (S⁻¹ *ᵥ v)) / 2) := by
  have hgrad := ch05b_g_score_matrix_grad S⁻¹ hS x v
  have hneg : (-(S⁻¹ *ᵥ x)) ⬝ᵥ v = -((S⁻¹ *ᵥ x) ⬝ᵥ v) := by
    first
    | exact neg_dotProduct _ _
    | simp
  simp only [ch05b_gaussPDF]
  rw [mul_assoc, ← Real.exp_add, hneg]
  congr 2
  linarith [hgrad]

/-- **eq:g-score-matrix (tex 792-796)** for the chapter's marginal law: with `P_t` the centred
Gaussian of covariance `Σ_t` (eq:g-Sigt), the linear part of the increment of `log P_t` at `x` is
the inner product with the score `s(x,t) = -Q_t x` of `ch05b_score`. -/
theorem ch05b_g_score_matrix_grad_log (S0 : Matrix (Fin n) (Fin n) ℝ) (hS0 : S0.IsSymm) (t : ℝ)
    (x v : Fin n → ℝ) :
    ch05b_gaussPDF (ch05b_Sigt S0 t) (x + v)
      = ch05b_gaussPDF (ch05b_Sigt S0 t) x
        * Real.exp ((ch05b_score S0 t x) ⬝ᵥ v - (v ⬝ᵥ (ch05b_Qt S0 t *ᵥ v)) / 2) := by
  have h := ch05b_g_score_matrix_density (ch05b_Sigt S0 t) (ch05b_Sigt_inv_isSymm S0 hS0 t) x v
  first
  | exact h
  | simpa [ch05b_score, ch05b_Qt] using h

/-! ### Spectral form of the resolvent, and entrywise bounds for its remainder

The hypotheses below — `S0 = U diag(ω) Uᵀ` with `UᵀU = I` and `0 < m ≤ ωᵢ` — are exactly the
spectral data `ch05b_g_spectral` returns for a symmetric positive definite `Σ₀`, which is the
chapter's standing assumption (tex 1180-1184).  Nothing is assumed about `t` beyond
`0 ≤ t ≤ 1/8`. -/

theorem ch05b_inv_anti (m x : ℝ) (hm : 0 < m) (h : m ≤ x) : x⁻¹ ≤ m⁻¹ := inv_anti₀ hm h

theorem ch05b_spec_inv (U : Matrix (Fin n) (Fin n) ℝ) (ω : Fin n → ℝ) (hU : Uᵀ * U = 1)
    (S0 : Matrix (Fin n) (Fin n) ℝ) (hS0 : S0 = U * Matrix.diagonal ω * Uᵀ)
    (hω : ∀ i, 0 < ω i) :
    S0⁻¹ = U * Matrix.diagonal (fun i => (ω i)⁻¹) * Uᵀ := by
  rw [hS0]
  exact ch05b_g_affine_inv U ω hU fun i => ne_of_gt (hω i)

theorem ch05b_spec_det_unit (U : Matrix (Fin n) (Fin n) ℝ) (ω : Fin n → ℝ) (hU : Uᵀ * U = 1)
    (S0 : Matrix (Fin n) (Fin n) ℝ) (hS0 : S0 = U * Matrix.diagonal ω * Uᵀ)
    (hω : ∀ i, 0 < ω i) : IsUnit S0.det := by
  rw [hS0, ch05b_conj_det U hU]
  exact isUnit_iff_ne_zero.mpr (Finset.prod_ne_zero_iff.mpr fun i _ => ne_of_gt (hω i))

theorem ch05b_spec_res (U : Matrix (Fin n) (Fin n) ℝ) (ω : Fin n → ℝ) (hU : Uᵀ * U = 1)
    (S0 : Matrix (Fin n) (Fin n) ℝ) (hS0 : S0 = U * Matrix.diagonal ω * Uᵀ)
    (hω : ∀ i, 0 < ω i) (c : ℝ) :
    (1 : Matrix (Fin n) (Fin n) ℝ) + c • S0⁻¹
      = U * Matrix.diagonal (fun i => c * (ω i)⁻¹ + 1) * Uᵀ := by
  have h := ch05b_g_affine U (fun i => (ω i)⁻¹) hU c 1
  rw [ch05b_spec_inv U ω hU S0 hS0 hω, ← h, one_smul, add_comm]

theorem ch05b_spec_res_pos (ω : Fin n → ℝ) (hω : ∀ i, 0 < ω i) (c : ℝ) (hc : 0 ≤ c) (i : Fin n) :
    0 < c * (ω i)⁻¹ + 1 := by
  have hinv : 0 < (ω i)⁻¹ := inv_pos.mpr (hω i)
  have : 0 ≤ c * (ω i)⁻¹ := mul_nonneg hc hinv.le
  linarith

theorem ch05b_spec_res_det_unit (U : Matrix (Fin n) (Fin n) ℝ) (ω : Fin n → ℝ) (hU : Uᵀ * U = 1)
    (S0 : Matrix (Fin n) (Fin n) ℝ) (hS0 : S0 = U * Matrix.diagonal ω * Uᵀ)
    (hω : ∀ i, 0 < ω i) (c : ℝ) (hc : 0 ≤ c) :
    IsUnit ((1 : Matrix (Fin n) (Fin n) ℝ) + c • S0⁻¹).det := by
  rw [ch05b_spec_res U ω hU S0 hS0 hω c, ch05b_conj_det U hU]
  exact isUnit_iff_ne_zero.mpr
    (Finset.prod_ne_zero_iff.mpr fun i _ => ne_of_gt (ch05b_spec_res_pos ω hω c hc i))

theorem ch05b_spec_resinv (U : Matrix (Fin n) (Fin n) ℝ) (ω : Fin n → ℝ) (hU : Uᵀ * U = 1)
    (S0 : Matrix (Fin n) (Fin n) ℝ) (hS0 : S0 = U * Matrix.diagonal ω * Uᵀ)
    (hω : ∀ i, 0 < ω i) (c : ℝ) (hc : 0 ≤ c) :
    ((1 : Matrix (Fin n) (Fin n) ℝ) + c • S0⁻¹)⁻¹
      = U * Matrix.diagonal (fun i => (c * (ω i)⁻¹ + 1)⁻¹) * Uᵀ := by
  rw [ch05b_spec_res U ω hU S0 hS0 hω c]
  exact ch05b_g_affine_inv U _ hU fun i => ne_of_gt (ch05b_spec_res_pos ω hω c hc i)

/-- The Neumann remainder matrix `Q₀(I + cQ₀)⁻¹Q₀ᵏ` is diagonal in the eigenbasis. -/
theorem ch05b_spec_rem (U : Matrix (Fin n) (Fin n) ℝ) (ω : Fin n → ℝ) (hU : Uᵀ * U = 1)
    (S0 : Matrix (Fin n) (Fin n) ℝ) (hS0 : S0 = U * Matrix.diagonal ω * Uᵀ)
    (hω : ∀ i, 0 < ω i) (c : ℝ) (hc : 0 ≤ c) (k : ℕ) :
    S0⁻¹ * ((1 : Matrix (Fin n) (Fin n) ℝ) + c • S0⁻¹)⁻¹ * S0⁻¹ ^ k
      = U * Matrix.diagonal
          (fun i => (ω i)⁻¹ * (c * (ω i)⁻¹ + 1)⁻¹ * ((ω i)⁻¹) ^ k) * Uᵀ := by
  rw [ch05b_spec_resinv U ω hU S0 hS0 hω c hc, ch05b_spec_inv U ω hU S0 hS0 hω,
    ch05b_conj_pow U hU, ch05b_conj_mul U hU, ch05b_conj_mul U hU]

/-- Entrywise bound on the Neumann remainder, uniform in `t`: `|(Q₀(I+cQ₀)⁻¹Q₀ᵏ)ᵢⱼ| ≤ n/m^{k+1}`
for every `c ≥ 0`. -/
theorem ch05b_spec_rem_bound (U : Matrix (Fin n) (Fin n) ℝ) (ω : Fin n → ℝ) (hU : Uᵀ * U = 1)
    (S0 : Matrix (Fin n) (Fin n) ℝ) (hS0 : S0 = U * Matrix.diagonal ω * Uᵀ)
    (hω : ∀ i, 0 < ω i) (m : ℝ) (hm : 0 < m) (hmω : ∀ i, m ≤ ω i)
    (c : ℝ) (hc : 0 ≤ c) (k : ℕ) (i j : Fin n) :
    |(S0⁻¹ * ((1 : Matrix (Fin n) (Fin n) ℝ) + c • S0⁻¹)⁻¹ * S0⁻¹ ^ k) i j|
      ≤ (n : ℝ) * (m⁻¹) ^ (k + 1) := by
  rw [ch05b_spec_rem U ω hU S0 hS0 hω c hc k]
  refine ch05b_conj_entry_bound U hU _ _ (fun l => ?_) i j
  have hwl : 0 < ω l := hω l
  have hinv : 0 < (ω l)⁻¹ := inv_pos.mpr hwl
  have hinvm : 0 < m⁻¹ := inv_pos.mpr hm
  have hle : (ω l)⁻¹ ≤ m⁻¹ := ch05b_inv_anti m (ω l) hm (hmω l)
  have hres : 0 < c * (ω l)⁻¹ + 1 := ch05b_spec_res_pos ω hω c hc l
  have hcw : 0 ≤ c * (ω l)⁻¹ := mul_nonneg hc hinv.le
  have hres1 : (c * (ω l)⁻¹ + 1)⁻¹ ≤ 1 := (inv_le_one₀ hres).mpr (by linarith)
  have hres0 : 0 < (c * (ω l)⁻¹ + 1)⁻¹ := inv_pos.mpr hres
  have hpk : ((ω l)⁻¹) ^ k ≤ (m⁻¹) ^ k := pow_le_pow_left₀ hinv.le hle k
  have hnn : 0 ≤ (ω l)⁻¹ * (c * (ω l)⁻¹ + 1)⁻¹ * ((ω l)⁻¹) ^ k :=
    mul_nonneg (mul_nonneg hinv.le hres0.le) (pow_nonneg hinv.le k)
  rw [abs_of_nonneg hnn]
  have step1 : (ω l)⁻¹ * (c * (ω l)⁻¹ + 1)⁻¹ ≤ m⁻¹ * 1 :=
    mul_le_mul hle hres1 hres0.le hinvm.le
  have step2 : (ω l)⁻¹ * (c * (ω l)⁻¹ + 1)⁻¹ * ((ω l)⁻¹) ^ k ≤ m⁻¹ * 1 * (m⁻¹) ^ k :=
    mul_le_mul step1 hpk (pow_nonneg hinv.le k) (by rw [mul_one]; exact hinvm.le)
  calc (ω l)⁻¹ * (c * (ω l)⁻¹ + 1)⁻¹ * ((ω l)⁻¹) ^ k ≤ m⁻¹ * 1 * (m⁻¹) ^ k := step2
    _ = (m⁻¹) ^ (k + 1) := by rw [pow_succ]; ring

/-- Entrywise bound on the powers of the clean precision: `|(Q₀ᵈ)ᵢⱼ| ≤ n/mᵈ`. -/
theorem ch05b_spec_pow_bound (U : Matrix (Fin n) (Fin n) ℝ) (ω : Fin n → ℝ) (hU : Uᵀ * U = 1)
    (S0 : Matrix (Fin n) (Fin n) ℝ) (hS0 : S0 = U * Matrix.diagonal ω * Uᵀ)
    (hω : ∀ i, 0 < ω i) (m : ℝ) (hm : 0 < m) (hmω : ∀ i, m ≤ ω i) (d : ℕ) (i j : Fin n) :
    |(S0⁻¹ ^ d) i j| ≤ (n : ℝ) * (m⁻¹) ^ d := by
  rw [ch05b_spec_inv U ω hU S0 hS0 hω, ch05b_conj_pow U hU]
  refine ch05b_conj_entry_bound U hU _ _ (fun l => ?_) i j
  have hinv : 0 < (ω l)⁻¹ := inv_pos.mpr (hω l)
  have hle : (ω l)⁻¹ ≤ m⁻¹ := ch05b_inv_anti m (ω l) hm (hmω l)
  rw [abs_of_nonneg (pow_nonneg hinv.le d)]
  exact pow_le_pow_left₀ hinv.le hle d

/-! ### The `O(t^{d-1})` and `O(t^d)` statements -/

/-- Elementary: `b^{k+1} - a^{k+1} ≤ (k+1)Mᵏ(b - a)` for `0 ≤ a ≤ b ≤ M`. -/
theorem ch05b_pow_sub_le (a b M : ℝ) (ha : 0 ≤ a) (hab : a ≤ b) (hbM : b ≤ M) :
    ∀ k : ℕ, b ^ (k + 1) - a ^ (k + 1) ≤ ((k : ℝ) + 1) * M ^ k * (b - a) := by
  have hb0 : 0 ≤ b := le_trans ha hab
  have hM0 : 0 ≤ M := le_trans hb0 hbM
  have hba : 0 ≤ b - a := by linarith
  have haM : a ≤ M := le_trans hab hbM
  intro k
  induction k with
  | zero => norm_num
  | succ k ih =>
      have hX : 0 ≤ b ^ (k + 1) - a ^ (k + 1) := by
        have h := pow_le_pow_left₀ ha hab (k + 1)
        linarith
      have t1 : b * (b ^ (k + 1) - a ^ (k + 1)) ≤ M * (((k : ℝ) + 1) * M ^ k * (b - a)) :=
        mul_le_mul hbM ih hX hM0
      have t2 : a ^ (k + 1) * (b - a) ≤ M ^ (k + 1) * (b - a) :=
        mul_le_mul_of_nonneg_right (pow_le_pow_left₀ ha haM (k + 1)) hba
      have hid : M * (((k : ℝ) + 1) * M ^ k * (b - a)) + M ^ (k + 1) * (b - a)
          = ((k : ℝ) + 1 + 1) * M ^ (k + 1) * (b - a) := by ring
      have hsplit : b ^ (k + 1 + 1) - a ^ (k + 1 + 1)
          = b * (b ^ (k + 1) - a ^ (k + 1)) + a ^ (k + 1) * (b - a) := by ring
      push_cast
      linarith [t1, t2, hid, hsplit]

/-- `c_tᵏ - (2t)ᵏ = O(t^{k+1})`: the two differ by one order in `t`, with explicit constant. -/
theorem ch05b_c_pow_sub {t : ℝ} (ht : 0 ≤ t) (ht8 : t ≤ 1/8) (k : ℕ) :
    ch05b_c t ^ k - (2 * t) ^ k ≤ 15 * (k : ℝ) * 4 ^ k * t ^ (k + 1) := by
  have ht4 : t ≤ 1/4 := by linarith
  have hcb := ch05b_c_bounds ht ht4
  have hc0 : 0 ≤ ch05b_c t := by linarith [hcb.1]
  have h2t : (0 : ℝ) ≤ 2 * t := by linarith
  have h4t : (0 : ℝ) ≤ 4 * t := by linarith
  have hE1 : 1 ≤ Real.exp (2 * t) := by
    have := Real.add_one_le_exp (2 * t); linarith
  have hbridge := ch05b_c_exp_bridge ht ht8
  have hcsmall : ch05b_c t ≤ 2 * t + 15 * t ^ 2 := by nlinarith [hbridge.2, hE1, hc0]
  cases k with
  | zero => norm_num
  | succ j =>
      have h1 := ch05b_pow_sub_le (2 * t) (ch05b_c t) (4 * t) h2t hcb.1 hcb.2 j
      have hco : (0 : ℝ) ≤ ((j : ℝ) + 1) * (4 * t) ^ j :=
        mul_nonneg (by positivity) (pow_nonneg h4t j)
      have hX0 : (0 : ℝ) ≤ ((j : ℝ) + 1) * (4 * t) ^ j * (15 * t ^ 2) :=
        mul_nonneg hco (by positivity)
      have h3 : ((j : ℝ) + 1) * (4 * t) ^ j * (ch05b_c t - 2 * t)
          ≤ ((j : ℝ) + 1) * (4 * t) ^ j * (15 * t ^ 2) :=
        mul_le_mul_of_nonneg_left (by linarith) hco
      have hXeq : 15 * ((j : ℝ) + 1) * 4 ^ (j + 1) * t ^ (j + 1 + 1)
          = 4 * (((j : ℝ) + 1) * (4 * t) ^ j * (15 * t ^ 2)) := by
        rw [mul_pow]; ring
      push_cast
      linarith [h1, h3, hX0, hXeq]

/-- The quantitative bridge from the thesis' `(2t)^{d-1}` to the exact coefficient
`e^{2t}c_t^{d-1}`: they differ by `O(t^d)`, with an explicit constant, on `[0, 1/8]`. -/
theorem ch05b_exp_c_pow_bridge {t : ℝ} (ht : 0 ≤ t) (ht8 : t ≤ 1/8) (k : ℕ) :
    |Real.exp (2 * t) * ch05b_c t ^ k - (2 * t) ^ k| ≤ 4 ^ k * (4 + 15 * (k : ℝ)) * t ^ (k + 1) := by
  have ht4 : t ≤ 1/4 := by linarith
  have hcb := ch05b_c_bounds ht ht4
  have hc0 : 0 ≤ ch05b_c t := by linarith [hcb.1]
  have h2t : (0 : ℝ) ≤ 2 * t := by linarith
  have hE1 : 1 ≤ Real.exp (2 * t) := by
    have := Real.add_one_le_exp (2 * t); linarith
  have hEc : Real.exp (2 * t) = 1 + ch05b_c t := by simp only [ch05b_c]; ring
  have hbridge := ch05b_c_exp_bridge ht ht8
  have hcsmall : ch05b_c t ≤ 2 * t + 15 * t ^ 2 := by nlinarith [hbridge.2, hE1, hc0]
  have hpow : (2 * t) ^ k ≤ ch05b_c t ^ k := pow_le_pow_left₀ h2t hcb.1 k
  have hlow : 0 ≤ Real.exp (2 * t) * ch05b_c t ^ k - (2 * t) ^ k := by
    nlinarith [hpow, hE1, pow_nonneg hc0 k]
  rw [abs_of_nonneg hlow]
  have hexp : Real.exp (2 * t) * ch05b_c t ^ k = ch05b_c t ^ k + ch05b_c t ^ (k + 1) := by
    rw [hEc, pow_succ]; ring
  rw [hexp]
  have hA : ch05b_c t ^ (k + 1) ≤ 4 * 4 ^ k * t ^ (k + 1) := by
    have h := pow_le_pow_left₀ hc0 hcb.2 (k + 1)
    rw [mul_pow] at h
    calc ch05b_c t ^ (k + 1) ≤ 4 ^ (k + 1) * t ^ (k + 1) := h
      _ = 4 * 4 ^ k * t ^ (k + 1) := by rw [pow_succ]; ring
  have hB := ch05b_c_pow_sub ht ht8 k
  linarith [hA, hB]

/-- **eq:g-band-fill-order (tex 1504-1512), as a genuine `O(t^{d-1})` bound.**  At frame
distance `d = k+1` the whole entry of `Q_t` is bounded by an explicit constant times
`t^{d-1}`, uniformly on `0 ≤ t ≤ 1/8`: the band really does fill at order `t^{d-1}`. -/
theorem ch05b_g_band_fill_order_bound (U : Matrix (Fin n) (Fin n) ℝ) (ω : Fin n → ℝ)
    (hU : Uᵀ * U = 1) (S0 : Matrix (Fin n) (Fin n) ℝ) (hS0 : S0 = U * Matrix.diagonal ω * Uᵀ)
    (hω : ∀ i, 0 < ω i) (m : ℝ) (hm : 0 < m) (hmω : ∀ i, m ≤ ω i)
    (hQ : ch05b_Banded S0⁻¹ 1) {t : ℝ} (ht : 0 ≤ t) (ht8 : t ≤ 1/8)
    (k : ℕ) (i j : Fin n) (hij : Int.natAbs ((i : ℤ) - (j : ℤ)) = k + 1) :
    |ch05b_Qt S0 t i j| ≤ (2 * (n : ℝ) * 4 ^ k * (m⁻¹) ^ (k + 1)) * t ^ k := by
  have ht4 : t ≤ 1/4 := by linarith
  have hcb := ch05b_c_bounds ht ht4
  have hc0 : 0 ≤ ch05b_c t := by linarith [hcb.1]
  have hEpos : 0 < Real.exp (2 * t) := Real.exp_pos _
  have hE2 := ch05b_exp_le_two ht ht8
  have hS0det := ch05b_spec_det_unit U ω hU S0 hS0 hω
  have hBdet := ch05b_spec_res_det_unit U ω hU S0 hS0 hω (ch05b_c t) hc0
  have hfac := ch05b_g_band_fill_order S0 t (k + 1) (by omega) i j hij hQ hS0det hBdet
  simp only [Nat.add_sub_cancel] at hfac
  have hR := ch05b_spec_rem_bound U ω hU S0 hS0 hω m hm hmω (ch05b_c t) hc0 k i j
  rw [hfac, abs_mul, abs_pow, abs_neg, abs_of_nonneg hc0, abs_mul, abs_of_pos hEpos]
  have hck : ch05b_c t ^ k ≤ 4 ^ k * t ^ k := by
    have h := pow_le_pow_left₀ hc0 hcb.2 k
    rwa [mul_pow] at h
  have h2 : Real.exp (2 * t) * |(S0⁻¹ * ((1 : Matrix (Fin n) (Fin n) ℝ) + ch05b_c t • S0⁻¹)⁻¹
      * S0⁻¹ ^ k) i j| ≤ 2 * ((n : ℝ) * (m⁻¹) ^ (k + 1)) :=
    mul_le_mul hE2 hR (abs_nonneg _) (by norm_num)
  calc ch05b_c t ^ k * (Real.exp (2 * t)
        * |(S0⁻¹ * ((1 : Matrix (Fin n) (Fin n) ℝ) + ch05b_c t • S0⁻¹)⁻¹ * S0⁻¹ ^ k) i j|)
      ≤ (4 ^ k * t ^ k) * (2 * ((n : ℝ) * (m⁻¹) ^ (k + 1))) :=
        mul_le_mul hck h2 (mul_nonneg hEpos.le (abs_nonneg _))
          (mul_nonneg (by positivity) (pow_nonneg ht k))
    _ = (2 * (n : ℝ) * 4 ^ k * (m⁻¹) ^ (k + 1)) * t ^ k := by ring

/-- **eq:g-bandfill (tex 1584-1591), the boxed theorem with its `O(t^d)` remainder.**  At frame
distance `d = k+1`,
`(Q_t)_{i,i+d} = (-1)^{d-1}(2t)^{d-1}(Q₀^d)_{i,i+d} + R`, with `|R| ≤ K t^d` for an explicit
`K` depending only on `(n, m, d)` — uniformly on `0 ≤ t ≤ 1/8`.  This is the thesis' constant
`(2t)^{d-1}` (not merely `c_t^{d-1}`) and a genuinely bounded remainder. -/
theorem ch05b_g_bandfill_bound (U : Matrix (Fin n) (Fin n) ℝ) (ω : Fin n → ℝ)
    (hU : Uᵀ * U = 1) (S0 : Matrix (Fin n) (Fin n) ℝ) (hS0 : S0 = U * Matrix.diagonal ω * Uᵀ)
    (hω : ∀ i, 0 < ω i) (m : ℝ) (hm : 0 < m) (hmω : ∀ i, m ≤ ω i)
    (hQ : ch05b_Banded S0⁻¹ 1) {t : ℝ} (ht : 0 ≤ t) (ht8 : t ≤ 1/8)
    (k : ℕ) (i j : Fin n) (hij : Int.natAbs ((i : ℤ) - (j : ℤ)) = k + 1) :
    |ch05b_Qt S0 t i j - (-1 : ℝ) ^ k * (2 * t) ^ k * (S0⁻¹ ^ (k + 1)) i j|
      ≤ (4 ^ k * (4 + 15 * (k : ℝ)) * ((n : ℝ) * (m⁻¹) ^ (k + 1))
          + 2 * 4 ^ (k + 1) * ((n : ℝ) * (m⁻¹) ^ (k + 2))) * t ^ (k + 1) := by
  have ht4 : t ≤ 1/4 := by linarith
  have hcb := ch05b_c_bounds ht ht4
  have hc0 : 0 ≤ ch05b_c t := by linarith [hcb.1]
  have hEpos : 0 < Real.exp (2 * t) := Real.exp_pos _
  have hE2 := ch05b_exp_le_two ht ht8
  have hS0det := ch05b_spec_det_unit U ω hU S0 hS0 hω
  have hBdet := ch05b_spec_res_det_unit U ω hU S0 hS0 hω (ch05b_c t) hc0
  have hbf := ch05b_g_bandfill S0 t (k + 1) (by omega) i j hij hQ hS0det hBdet
  simp only [Nat.add_sub_cancel] at hbf
  have hA := ch05b_spec_pow_bound U ω hU S0 hS0 hω m hm hmω (k + 1) i j
  have hR : |(S0⁻¹ * ((1 : Matrix (Fin n) (Fin n) ℝ) + ch05b_c t • S0⁻¹)⁻¹ * S0⁻¹ ^ (k + 1)) i j|
      ≤ (n : ℝ) * (m⁻¹) ^ (k + 2) := by
    have h := ch05b_spec_rem_bound U ω hU S0 hS0 hω m hm hmω (ch05b_c t) hc0 (k + 1) i j
    first
    | exact h
    | (rw [show k + 1 + 1 = k + 2 from rfl] at h; exact h)
    | simpa using h
  set A : ℝ := (S0⁻¹ ^ (k + 1)) i j with hAdef
  set R : ℝ := (S0⁻¹ * ((1 : Matrix (Fin n) (Fin n) ℝ) + ch05b_c t • S0⁻¹)⁻¹
      * S0⁻¹ ^ (k + 1)) i j with hRdef
  have hneg : (-(ch05b_c t)) ^ k = (-1 : ℝ) ^ k * ch05b_c t ^ k := by
    rw [← neg_one_mul, mul_pow]
  have hsplit : ch05b_Qt S0 t i j - (-1 : ℝ) ^ k * (2 * t) ^ k * A
      = (Real.exp (2 * t) * ch05b_c t ^ k - (2 * t) ^ k) * ((-1 : ℝ) ^ k * A)
        + Real.exp (2 * t) * (-(ch05b_c t)) ^ (k + 1) * R := by
    rw [hbf, hneg]; ring
  rw [hsplit]
  have hcoef1 : (0 : ℝ) ≤ 4 ^ k * (4 + 15 * (k : ℝ)) * t ^ (k + 1) :=
    mul_nonneg (mul_nonneg (by positivity) (by positivity)) (pow_nonneg ht _)
  have hp1 : |(Real.exp (2 * t) * ch05b_c t ^ k - (2 * t) ^ k) * ((-1 : ℝ) ^ k * A)|
      ≤ (4 ^ k * (4 + 15 * (k : ℝ)) * t ^ (k + 1)) * ((n : ℝ) * (m⁻¹) ^ (k + 1)) := by
    rw [abs_mul, abs_mul, abs_pow, abs_neg, abs_one, one_pow, one_mul]
    exact mul_le_mul (ch05b_exp_c_pow_bridge ht ht8 k) hA (abs_nonneg _) hcoef1
  have hc1 : ch05b_c t ^ (k + 1) ≤ 4 ^ (k + 1) * t ^ (k + 1) := by
    have h := pow_le_pow_left₀ hc0 hcb.2 (k + 1)
    rwa [mul_pow] at h
  have hstep : Real.exp (2 * t) * ch05b_c t ^ (k + 1) ≤ 2 * (4 ^ (k + 1) * t ^ (k + 1)) :=
    mul_le_mul hE2 hc1 (pow_nonneg hc0 _) (by norm_num)
  have hp2 : |Real.exp (2 * t) * (-(ch05b_c t)) ^ (k + 1) * R|
      ≤ (2 * 4 ^ (k + 1) * t ^ (k + 1)) * ((n : ℝ) * (m⁻¹) ^ (k + 2)) := by
    rw [abs_mul, abs_mul, abs_of_pos hEpos, abs_pow, abs_neg, abs_of_nonneg hc0]
    calc Real.exp (2 * t) * ch05b_c t ^ (k + 1) * |R|
        ≤ (2 * (4 ^ (k + 1) * t ^ (k + 1))) * ((n : ℝ) * (m⁻¹) ^ (k + 2)) :=
          mul_le_mul hstep hR (abs_nonneg _)
            (mul_nonneg (by norm_num) (mul_nonneg (by positivity) (pow_nonneg ht _)))
      _ = (2 * 4 ^ (k + 1) * t ^ (k + 1)) * ((n : ℝ) * (m⁻¹) ^ (k + 2)) := by ring
  calc |(Real.exp (2 * t) * ch05b_c t ^ k - (2 * t) ^ k) * ((-1 : ℝ) ^ k * A)
          + Real.exp (2 * t) * (-(ch05b_c t)) ^ (k + 1) * R|
      ≤ |(Real.exp (2 * t) * ch05b_c t ^ k - (2 * t) ^ k) * ((-1 : ℝ) ^ k * A)|
        + |Real.exp (2 * t) * (-(ch05b_c t)) ^ (k + 1) * R| := abs_add_le _ _
    _ ≤ (4 ^ k * (4 + 15 * (k : ℝ)) * t ^ (k + 1)) * ((n : ℝ) * (m⁻¹) ^ (k + 1))
        + (2 * 4 ^ (k + 1) * t ^ (k + 1)) * ((n : ℝ) * (m⁻¹) ^ (k + 2)) := add_le_add hp1 hp2
    _ = (4 ^ k * (4 + 15 * (k : ℝ)) * ((n : ℝ) * (m⁻¹) ^ (k + 1))
          + 2 * 4 ^ (k + 1) * ((n : ℝ) * (m⁻¹) ^ (k + 2))) * t ^ (k + 1) := by ring

/-! ### eq:g-Qt-smallt with an explicit remainder (tex 1312-1322) -/

/-- The exact scalar identity behind the small-`t` approximation: `e^{2t}λᵢ(t) = ωᵢ + c_t`. -/
theorem ch05b_lam_exp_id (ω t : ℝ) : Real.exp (2 * t) * ch05b_lam ω t = ω + ch05b_c t := by
  simp only [ch05b_lam, ch05b_Delta, ch05b_c]
  have he : Real.exp (2 * t) * Real.exp (-2 * t) = 1 := by
    rw [← Real.exp_add, show 2 * t + -2 * t = 0 by ring, Real.exp_zero]
  linear_combination (ω - 1) * he

/-- **eq:g-Qt-smallt, per mode, exactly.**  The error of the thesis' `Q_t ≈ e^{2t}Q₀` is, in
eigen-coordinate `i`, exactly `-c_t/(λᵢ(t)ωᵢ)`. -/
theorem ch05b_g_Qt_smallt_mode (ω t : ℝ) (hω : 0 < ω) (ht : 0 ≤ t) :
    (ch05b_lam ω t)⁻¹ - Real.exp (2 * t) * ω⁻¹ = -(ch05b_c t / (ch05b_lam ω t * ω)) := by
  have hlam : 0 < ch05b_lam ω t := ch05b_lam_pos hω ht
  have hl : ch05b_lam ω t ≠ 0 := ne_of_gt hlam
  have hw : ω ≠ 0 := ne_of_gt hω
  have key : Real.exp (2 * t) * ch05b_lam ω t = ω + ch05b_c t := ch05b_lam_exp_id ω t
  field_simp
  first
  | linarith [key]
  | linear_combination key
  | linear_combination -key
  | nlinarith [key]

theorem ch05b_lam_ge {ω t m : ℝ} (hm : 0 < m) (hm1 : m ≤ 1) (hmω : m ≤ ω) (ht : 0 ≤ t) :
    m ≤ ch05b_lam ω t := by
  have hE : 0 < Real.exp (-2 * t) := Real.exp_pos _
  have hE1 : Real.exp (-2 * t) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith)
  simp only [ch05b_lam, ch05b_Delta]
  rcases le_total 1 ω with h | h
  · nlinarith
  · nlinarith

/-- **eq:g-Qt-smallt, per mode, quantitatively**: the approximation error is `O(t)`, with an
explicit constant, on `0 ≤ t ≤ 1/4`. -/
theorem ch05b_g_Qt_smallt_bound (ω t m : ℝ) (hm : 0 < m) (hm1 : m ≤ 1) (hmω : m ≤ ω)
    (ht : 0 ≤ t) (ht4 : t ≤ 1/4) :
    |(ch05b_lam ω t)⁻¹ - Real.exp (2 * t) * ω⁻¹| ≤ 4 * t / m ^ 2 := by
  have hω : 0 < ω := lt_of_lt_of_le hm hmω
  have hlam : 0 < ch05b_lam ω t := ch05b_lam_pos hω ht
  have hlm : m ≤ ch05b_lam ω t := ch05b_lam_ge hm hm1 hmω ht
  have hcb := ch05b_c_bounds ht ht4
  have hc0 : 0 ≤ ch05b_c t := by linarith [hcb.1]
  have hm2 : m ^ 2 ≤ ch05b_lam ω t * ω := by
    have h := mul_le_mul hlm hmω hm.le (le_trans hm.le hlm)
    rw [pow_two]; exact h
  rw [ch05b_g_Qt_smallt_mode ω t hω ht, abs_neg,
    abs_of_nonneg (div_nonneg hc0 (mul_nonneg hlam.le hω.le)),
    div_le_div_iff₀ (mul_pos hlam hω) (pow_pos hm 2)]
  nlinarith [mul_nonneg (by linarith : (0:ℝ) ≤ 4 * t - ch05b_c t) (sq_nonneg m),
    mul_nonneg (by linarith : (0:ℝ) ≤ 4 * t)
      (by linarith : (0:ℝ) ≤ ch05b_lam ω t * ω - m ^ 2)]

/-- **eq:g-Qt-smallt, matrix form, exactly**: the remainder of `Q_t ≈ e^{2t}Q₀` is the `N = 1`
Neumann remainder `-e^{2t}c_t Q₀(I + c_tQ₀)⁻¹Q₀`. -/
theorem ch05b_g_Qt_smallt_remainder (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ)
    (hS0 : IsUnit S0.det)
    (hB : IsUnit ((1 : Matrix (Fin n) (Fin n) ℝ) + ch05b_c t • S0⁻¹).det) :
    ch05b_Qt S0 t - Real.exp (2 * t) • S0⁻¹
      = -((Real.exp (2 * t) * ch05b_c t)
          • (S0⁻¹ * ((1 : Matrix (Fin n) (Fin n) ℝ) + ch05b_c t • S0⁻¹)⁻¹ * S0⁻¹)) := by
  have h := ch05b_g_Qt_res S0 t 1 hS0 hB
  rw [Finset.sum_range_one] at h
  simp only [pow_zero, one_smul, zero_add, pow_one] at h
  have hmul : Real.exp (2 * t) * -(ch05b_c t) = -(Real.exp (2 * t) * ch05b_c t) := by ring
  rw [h, smul_add, smul_smul, hmul, neg_smul]
  abel

/-- **eq:g-Qt-smallt, matrix form, quantitatively**: `‖Q_t - e^{2t}Q₀‖_F² ≤ n(4t/m²)²`, so the
thesis' small-`t` approximation is correct to `O(t)` in Frobenius norm. -/
theorem ch05b_g_Qt_smallt_frob (U : Matrix (Fin n) (Fin n) ℝ) (ω : Fin n → ℝ) (hU : Uᵀ * U = 1)
    (S0 : Matrix (Fin n) (Fin n) ℝ) (hS0 : S0 = U * Matrix.diagonal ω * Uᵀ)
    (hω : ∀ i, 0 < ω i) (m : ℝ) (hm : 0 < m) (hm1 : m ≤ 1) (hmω : ∀ i, m ≤ ω i)
    {t : ℝ} (ht : 0 ≤ t) (ht4 : t ≤ 1/4) :
    ch05b_frobSq (ch05b_Qt S0 t - Real.exp (2 * t) • S0⁻¹) ≤ (n : ℝ) * (4 * t / m ^ 2) ^ 2 := by
  have hQ : ch05b_Qt S0 t = U * Matrix.diagonal (fun i => (ch05b_lam (ω i) t)⁻¹) * Uᵀ :=
    (ch05b_g_Qt_spec U ω hU S0 hS0 t ht hω).2
  have hQ0 : Real.exp (2 * t) • S0⁻¹
      = U * Matrix.diagonal (fun i => Real.exp (2 * t) * (ω i)⁻¹) * Uᵀ := by
    rw [ch05b_spec_inv U ω hU S0 hS0 hω, ch05b_conj_smul]
  rw [hQ, hQ0, ch05b_conj_sub, ch05b_frobSq_conj U hU]
  have hb : ∀ i : Fin n,
      ((ch05b_lam (ω i) t)⁻¹ - Real.exp (2 * t) * (ω i)⁻¹) ^ 2 ≤ (4 * t / m ^ 2) ^ 2 := by
    intro i
    have h := ch05b_g_Qt_smallt_bound (ω i) t m hm hm1 (hmω i) ht ht4
    calc ((ch05b_lam (ω i) t)⁻¹ - Real.exp (2 * t) * (ω i)⁻¹) ^ 2
        = |(ch05b_lam (ω i) t)⁻¹ - Real.exp (2 * t) * (ω i)⁻¹| ^ 2 := (sq_abs _).symm
      _ ≤ (4 * t / m ^ 2) ^ 2 := pow_le_pow_left₀ (abs_nonneg _) h 2
  calc ∑ i, ((ch05b_lam (ω i) t)⁻¹ - Real.exp (2 * t) * (ω i)⁻¹) ^ 2
      ≤ ∑ _i : Fin n, (4 * t / m ^ 2) ^ 2 := Finset.sum_le_sum fun i _ => hb i
    _ = (n : ℝ) * (4 * t / m ^ 2) ^ 2 := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

/-! ### noname-frob-normalisation (tex 1641-1643) -/

theorem ch05b_count_edge (L : ℕ) :
    ∑ i ∈ Finset.range L, (if Int.natAbs ((i : ℤ) - (L : ℤ)) = 1 then (1:ℝ) else 0)
      = if L = 0 then 0 else 1 := by
  rcases Nat.eq_zero_or_pos L with h | h
  · subst h; simp
  · rw [if_neg (by omega), Finset.sum_eq_single (L - 1)]
    · rw [if_pos (by omega)]
    · intro b hb hne
      simp only [Finset.mem_range] at hb
      rw [if_neg (by omega)]
    · intro hnot
      exact absurd (Finset.mem_range.mpr (by omega)) hnot

theorem ch05b_count_edge' (L : ℕ) :
    ∑ j ∈ Finset.range L, (if Int.natAbs ((L : ℤ) - (j : ℤ)) = 1 then (1:ℝ) else 0)
      = if L = 0 then 0 else 1 := by
  rcases Nat.eq_zero_or_pos L with h | h
  · subst h; simp
  · rw [if_neg (by omega), Finset.sum_eq_single (L - 1)]
    · rw [if_pos (by omega)]
    · intro b hb hne
      simp only [Finset.mem_range] at hb
      rw [if_neg (by omega)]
    · intro hnot
      exact absurd (Finset.mem_range.mpr (by omega)) hnot

/-- There are exactly `2(L-1)` ordered pairs of frames at distance one. -/
theorem ch05b_count_nn (L : ℕ) :
    ∑ i ∈ Finset.range L, ∑ j ∈ Finset.range L,
        (if Int.natAbs ((i : ℤ) - (j : ℤ)) = 1 then (1:ℝ) else 0)
      = 2 * ((L - 1 : ℕ) : ℝ) := by
  induction L with
  | zero => simp
  | succ L ih =>
      have hrow : ∀ i : ℕ, ∑ j ∈ Finset.range (L + 1),
          (if Int.natAbs ((i : ℤ) - (j : ℤ)) = 1 then (1:ℝ) else 0)
          = (∑ j ∈ Finset.range L, (if Int.natAbs ((i : ℤ) - (j : ℤ)) = 1 then (1:ℝ) else 0))
            + (if Int.natAbs ((i : ℤ) - (L : ℤ)) = 1 then (1:ℝ) else 0) :=
        fun i => Finset.sum_range_succ _ _
      have hLL : (if Int.natAbs ((L : ℤ) - (L : ℤ)) = 1 then (1:ℝ) else 0) = 0 := by
        rw [if_neg (by omega)]
      rw [Finset.sum_range_succ]
      simp only [hrow]
      rw [Finset.sum_add_distrib, ih, ch05b_count_edge L, ch05b_count_edge' L, hLL,
        Nat.add_sub_cancel]
      rcases Nat.eq_zero_or_pos L with hL | hL
      · subst hL; norm_num
      · rw [if_neg (by omega)]
        have hc : ((L - 1 : ℕ) : ℝ) = (L : ℝ) - 1 := by
          rw [Nat.cast_sub hL]; norm_num
        rw [hc]; ring

theorem ch05b_frob_nn_sq (α : ℝ) (L : ℕ) :
    ch05b_frobSq (ch05b_Pi (ch05b_Q0 α L) - ch05b_diagPart (ch05b_Q0 α L))
      = 2 * ((L - 1 : ℕ) : ℝ) * (α ^ 2 / (1 - α ^ 2) ^ 2) := by
  have hentry : ∀ i j : Fin L,
      ((ch05b_Pi (ch05b_Q0 α L) - ch05b_diagPart (ch05b_Q0 α L)) i j) ^ 2
        = (if Int.natAbs ((i : ℤ) - (j : ℤ)) = 1 then (1:ℝ) else 0)
          * (α ^ 2 / (1 - α ^ 2) ^ 2) := by
    intro i j
    have hi : (i : ℕ) < L := i.isLt
    have hj : (j : ℕ) < L := j.isLt
    by_cases hd : Int.natAbs ((i : ℤ) - (j : ℤ)) = 1
    · have hne : ¬ (i = j) := by
        intro hh
        rw [hh] at hd
        omega
      have hQ : ch05b_Q0 α L i j = -α / (1 - α ^ 2) := by
        rw [ch05b_Q0_apply]
        rcases (by omega : (i : ℕ) + 1 = (j : ℕ) ∨ (j : ℕ) + 1 = (i : ℕ)) with hup | hdn
        · exact ch05b_Q0E_up α L _ _ hi hj hup
        · exact ch05b_Q0E_down α L _ _ hi hj hdn
      have hPi : ch05b_Pi (ch05b_Q0 α L) i j = ch05b_Q0 α L i j := by
        simp only [ch05b_Pi]; rw [if_pos (le_of_eq hd)]
      have hDg : ch05b_diagPart (ch05b_Q0 α L) i j = 0 := by
        simp only [ch05b_diagPart]; rw [if_neg hne]
      rw [if_pos hd, one_mul, Matrix.sub_apply, hPi, hDg, sub_zero, hQ]
      first
      | rw [div_pow, neg_sq]
      | (rw [div_pow]; norm_num)
      | ring
    · rw [if_neg hd, zero_mul]
      by_cases hij : i = j
      · have hPi : ch05b_Pi (ch05b_Q0 α L) i j = ch05b_Q0 α L i j := by
          simp only [ch05b_Pi]
          rw [if_pos (by rw [hij]; omega)]
        have hDg : ch05b_diagPart (ch05b_Q0 α L) i j = ch05b_Q0 α L i j := by
          simp only [ch05b_diagPart]; rw [if_pos hij]
        rw [Matrix.sub_apply, hPi, hDg, sub_self]
        norm_num
      · have hval : (i : ℕ) ≠ (j : ℕ) := fun hh => hij (Fin.ext hh)
        have hgt : ¬ (Int.natAbs ((i : ℤ) - (j : ℤ)) ≤ 1) := by omega
        have hPi : ch05b_Pi (ch05b_Q0 α L) i j = 0 := by
          simp only [ch05b_Pi]; rw [if_neg hgt]
        have hDg : ch05b_diagPart (ch05b_Q0 α L) i j = 0 := by
          simp only [ch05b_diagPart]; rw [if_neg hij]
        rw [Matrix.sub_apply, hPi, hDg, sub_zero]
        norm_num
  have hrow : ∀ i : Fin L,
      ∑ j : Fin L, ((ch05b_Pi (ch05b_Q0 α L) - ch05b_diagPart (ch05b_Q0 α L)) i j) ^ 2
        = (∑ j : Fin L, (if Int.natAbs ((i : ℤ) - (j : ℤ)) = 1 then (1:ℝ) else 0))
          * (α ^ 2 / (1 - α ^ 2) ^ 2) := by
    intro i
    rw [Finset.sum_mul]
    exact Finset.sum_congr rfl fun j _ => hentry i j
  have hcast : ∑ i : Fin L, ∑ j : Fin L,
      (if Int.natAbs ((i : ℤ) - (j : ℤ)) = 1 then (1:ℝ) else 0)
      = ∑ i ∈ Finset.range L, ∑ j ∈ Finset.range L,
        (if Int.natAbs ((i : ℤ) - (j : ℤ)) = 1 then (1:ℝ) else 0) := by
    rw [← Fin.sum_univ_eq_sum_range
      (fun i => ∑ j ∈ Finset.range L,
        (if Int.natAbs ((i : ℤ) - (j : ℤ)) = 1 then (1:ℝ) else 0)) L]
    exact Finset.sum_congr rfl fun i _ =>
      Fin.sum_univ_eq_sum_range
        (fun j => (if Int.natAbs ((i : ℤ) - (j : ℤ)) = 1 then (1:ℝ) else 0)) L
  simp only [ch05b_frobSq]
  rw [Finset.sum_congr rfl fun i _ => hrow i, ← Finset.sum_mul, hcast, ch05b_count_nn L]

/-- **noname-frob-normalisation (tex 1641-1643), for every chain length `L`.**  The clean
nearest-neighbour coupling mass by which Figure fig:g-markov-survival normalises both curves is
`‖ΠQ₀ - diag Q₀‖_F = √(2(L-1))·|α|/σ_η²`, with `σ_η² = 1 - α²`. -/
theorem ch05b_noname_frob_normalisation (α : ℝ) (L : ℕ) (hα2 : α ^ 2 < 1) :
    Real.sqrt (ch05b_frobSq (ch05b_Pi (ch05b_Q0 α L) - ch05b_diagPart (ch05b_Q0 α L)))
      = Real.sqrt (2 * ((L - 1 : ℕ) : ℝ)) * (|α| / (1 - α ^ 2)) := by
  have hpos : (0:ℝ) < 1 - α ^ 2 := by linarith
  rw [ch05b_frob_nn_sq, Real.sqrt_mul (by positivity)]
  congr 1
  rw [show α ^ 2 / (1 - α ^ 2) ^ 2 = (α / (1 - α ^ 2)) ^ 2 by rw [div_pow],
    Real.sqrt_sq_eq_abs, abs_div, abs_of_pos hpos]

end

end ThesisAudit
