import Mathlib

/-!
Audit of display-math formulas in ch01-introduction.tex.

The chapter contains exactly one display equation, the definition of the score
function (eq:intro-score),

    s(x, t) = ∇_x log p_t(x).

In the thesis the argument `x` is the *whole noised sequence* -- "The joint score
of the whole noisy sequence is then available as an exact functional recursion"
-- so `x` ranges over ℝ^L and `∇_x` is a genuine gradient, not a one-dimensional
derivative.  The faithful encoding is therefore `ch01_intro_score_multi`, the
gradient of `log p_t` on `EuclideanSpace ℝ (Fin n)` for an *arbitrary* dimension
`n : ℕ`.  The one-dimensional `ch01_intro_score` is retained as the `n = 1`
special case.

Everything below is proved for a general dimension `n`; nothing is checked only
at small `n`.  The Gaussian sanity check is likewise general: it is proved for an
arbitrary nondegenerate Gaussian `C · exp(-½ ⟪y - μ, A (y - μ)⟫)` with `A` any
self-adjoint operator (in the thesis's Gaussian chapters, `A = Σ⁻¹`), and the
hypothesis is exhibited as non-vacuous by showing that an arbitrary symmetric
precision *matrix* in dimension `n` realises it.
-/

namespace ThesisAudit

open scoped RealInnerProductSpace
open InnerProductSpace

/-! ### eq:intro-score, one-dimensional case -/

-- eq:intro-score (ch01-introduction.tex lines 16-19): s(x,t) = ∇_x log p_t(x),
-- the score function, in one dimension the spatial derivative of the log-density.
noncomputable def ch01_intro_score (p : ℝ → ℝ → ℝ) (x t : ℝ) : ℝ :=
  deriv (fun y => Real.log (p y t)) x

-- Sanity check for the definition: for the standard Gaussian density
-- p_t(x) = exp(-x²/2)/√(2π) the score is -x, the classical value.
theorem ch01_intro_score_gaussian (x t : ℝ) :
    ch01_intro_score (fun y _ => Real.exp (-(y ^ 2 / 2)) / Real.sqrt (2 * Real.pi)) x t
      = -x := by
  unfold ch01_intro_score
  have hsqrt : Real.sqrt (2 * Real.pi) ≠ 0 := by
    have : (0 : ℝ) < 2 * Real.pi := by positivity
    exact ne_of_gt (Real.sqrt_pos.mpr this)
  have hfun : (fun y => Real.log (Real.exp (-(y ^ 2 / 2)) / Real.sqrt (2 * Real.pi)))
      = fun y => -(y ^ 2 / 2) - Real.log (Real.sqrt (2 * Real.pi)) := by
    funext y
    rw [Real.log_div (Real.exp_ne_zero _) hsqrt, Real.log_exp]
  rw [hfun]
  have h1 : HasDerivAt (fun y : ℝ => y ^ 2 / 2) x x := by
    have h : HasDerivAt (fun y : ℝ => y ^ 2) (2 * x) x := by
      simpa using hasDerivAt_pow 2 x
    have h' := h.div_const 2
    have hx : 2 * x / 2 = x := by ring
    rw [hx] at h'
    exact h'
  have h2 : HasDerivAt
      (fun y : ℝ => -(y ^ 2 / 2) - Real.log (Real.sqrt (2 * Real.pi))) (-x) x :=
    (h1.neg).sub_const _
  exact h2.deriv

/-! ### eq:intro-score in a general dimension

`x` ranges over ℝⁿ for an arbitrary `n`, which is the form the equation is
actually used in (the joint score of a length-`L` sequence). -/

section GeneralDimension

variable {n : ℕ}

-- eq:intro-score (ch01-introduction.tex lines 16-19), general dimension:
-- s(x,t) = ∇_x log p_t(x), the gradient of the log-density on ℝⁿ.
noncomputable def ch01_intro_score_multi
    (p : EuclideanSpace ℝ (Fin n) → ℝ → ℝ)
    (x : EuclideanSpace ℝ (Fin n)) (t : ℝ) : EuclideanSpace ℝ (Fin n) :=
  gradient (fun y => Real.log (p y t)) x

-- The definition really is "the gradient", not some other vector attached to the
-- derivative: `g` is the score of `p_t` at `x` exactly when the Fréchet
-- derivative of `log p_t` at `x` is the linear form `v ↦ ⟪g, v⟫`.  This pins the
-- encoding down and rules out an off-by-a-transpose reading.
theorem ch01_intro_score_multi_spec
    (p : EuclideanSpace ℝ (Fin n) → ℝ → ℝ) (x g : EuclideanSpace ℝ (Fin n)) (t : ℝ) :
    (HasGradientAt (fun y => Real.log (p y t)) g x
        ↔ HasFDerivAt (fun y => Real.log (p y t))
            (toDual ℝ (EuclideanSpace ℝ (Fin n)) g) x)
      ∧ ∀ v : EuclideanSpace ℝ (Fin n),
          toDual ℝ (EuclideanSpace ℝ (Fin n)) g v = ⟪g, v⟫ := by
  exact ⟨hasGradientAt_iff_hasFDerivAt, fun _ => rfl⟩

-- Consequently the score of a log-density differentiable at `x` is that gradient.
theorem ch01_intro_score_multi_eq
    (p : EuclideanSpace ℝ (Fin n) → ℝ → ℝ) (x : EuclideanSpace ℝ (Fin n)) (t : ℝ)
    {g : EuclideanSpace ℝ (Fin n)}
    (hg : HasGradientAt (fun y => Real.log (p y t)) g x) :
    ch01_intro_score_multi p x t = g :=
  hg.gradient

/-! #### The gradient of a quadratic form -/

-- For a self-adjoint `A`, the quadratic form `q(y) = ⟪y - μ, A (y - μ)⟫` has
-- gradient `2 A (x - μ)` at every point `x`, in every dimension `n`.
theorem ch01_gradient_quadratic
    (A : EuclideanSpace ℝ (Fin n) →L[ℝ] EuclideanSpace ℝ (Fin n))
    (hA : ∀ u v, ⟪A u, v⟫ = ⟪u, A v⟫)
    (μ x : EuclideanSpace ℝ (Fin n)) :
    HasGradientAt (fun y => ⟪y - μ, A (y - μ)⟫) ((2 : ℝ) • A (x - μ)) x := by
  have hsub : HasFDerivAt (fun y : EuclideanSpace ℝ (Fin n) => y - μ)
      (ContinuousLinearMap.id ℝ (EuclideanSpace ℝ (Fin n))) x :=
    (hasFDerivAt_id x).sub_const μ
  have hAy : HasFDerivAt (fun y : EuclideanSpace ℝ (Fin n) => A (y - μ))
      (A.comp (ContinuousLinearMap.id ℝ (EuclideanSpace ℝ (Fin n)))) x :=
    A.hasFDerivAt.comp x hsub
  have hinner := hsub.inner ℝ hAy
  rw [hasGradientAt_iff_hasFDerivAt]
  refine hinner.congr_fderiv (ContinuousLinearMap.ext fun v => ?_)
  have hlhs : ((fderivInnerCLM ℝ ((x - μ), A (x - μ))).comp
        ((ContinuousLinearMap.id ℝ (EuclideanSpace ℝ (Fin n))).prod
          (A.comp (ContinuousLinearMap.id ℝ (EuclideanSpace ℝ (Fin n)))))) v
      = ⟪x - μ, A v⟫ + ⟪v, A (x - μ)⟫ := rfl
  have hrhs : (toDual ℝ (EuclideanSpace ℝ (Fin n)) ((2 : ℝ) • A (x - μ))) v
      = ⟪(2 : ℝ) • A (x - μ), v⟫ := rfl
  rw [hlhs, hrhs, real_inner_smul_left, ← hA (x - μ) v, real_inner_comm (A (x - μ)) v]
  ring

/-! #### The score of a general Gaussian in a general dimension -/

-- The log-density of `C · exp(-½ ⟪y - μ, A (y - μ)⟫)` has gradient `-A (x - μ)`.
theorem ch01_gaussian_log_gradient
    (A : EuclideanSpace ℝ (Fin n) →L[ℝ] EuclideanSpace ℝ (Fin n))
    (hA : ∀ u v, ⟪A u, v⟫ = ⟪u, A v⟫)
    (μ : EuclideanSpace ℝ (Fin n)) (C : ℝ) (hC : 0 < C)
    (x : EuclideanSpace ℝ (Fin n)) :
    HasGradientAt
      (fun y => Real.log (C * Real.exp (-(1 / 2 : ℝ) * ⟪y - μ, A (y - μ)⟫)))
      (-(A (x - μ))) x := by
  have hlog : (fun y : EuclideanSpace ℝ (Fin n) =>
        Real.log (C * Real.exp (-(1 / 2 : ℝ) * ⟪y - μ, A (y - μ)⟫)))
      = fun y : EuclideanSpace ℝ (Fin n) =>
          Real.log C + (-(1 / 2 : ℝ)) * ⟪y - μ, A (y - μ)⟫ := by
    funext y
    rw [Real.log_mul (ne_of_gt hC) (Real.exp_ne_zero _), Real.log_exp]
  rw [hlog]
  have hq : HasFDerivAt (fun y : EuclideanSpace ℝ (Fin n) => ⟪y - μ, A (y - μ)⟫)
      (toDual ℝ (EuclideanSpace ℝ (Fin n)) ((2 : ℝ) • A (x - μ))) x :=
    (ch01_gradient_quadratic A hA μ x).hasFDerivAt
  have hfin : HasFDerivAt (fun y : EuclideanSpace ℝ (Fin n) =>
        Real.log C + (-(1 / 2 : ℝ)) * ⟪y - μ, A (y - μ)⟫)
      ((-(1 / 2 : ℝ)) • toDual ℝ (EuclideanSpace ℝ (Fin n)) ((2 : ℝ) • A (x - μ))) x := by
    exact (hq.const_mul (-(1 / 2 : ℝ))).const_add (Real.log C)
  rw [hasGradientAt_iff_hasFDerivAt]
  refine hfin.congr_fderiv (ContinuousLinearMap.ext fun v => ?_)
  have hlhs : (((-(1 / 2 : ℝ)) •
        toDual ℝ (EuclideanSpace ℝ (Fin n)) ((2 : ℝ) • A (x - μ))) v)
      = (-(1 / 2 : ℝ)) * ⟪(2 : ℝ) • A (x - μ), v⟫ := rfl
  have hrhs : (toDual ℝ (EuclideanSpace ℝ (Fin n)) (-(A (x - μ)))) v
      = ⟪-(A (x - μ)), v⟫ := rfl
  rw [hlhs, hrhs, real_inner_smul_left, inner_neg_left]
  ring

-- **eq:intro-score, Gaussian case, general dimension and general covariance.**
-- For the nondegenerate Gaussian density p(y) = C · exp(-½ ⟪y-μ, A (y-μ)⟫) with
-- `A` self-adjoint (in the thesis `A = Σ⁻¹`), the score is `-A (x - μ)` -- the
-- classical value `-Σ⁻¹(x - μ)`, for every dimension `n`.
theorem ch01_gaussian_score
    (A : EuclideanSpace ℝ (Fin n) →L[ℝ] EuclideanSpace ℝ (Fin n))
    (hA : ∀ u v, ⟪A u, v⟫ = ⟪u, A v⟫)
    (μ : EuclideanSpace ℝ (Fin n)) (C : ℝ) (hC : 0 < C)
    (x : EuclideanSpace ℝ (Fin n)) (t : ℝ) :
    ch01_intro_score_multi
      (fun y _ => C * Real.exp (-(1 / 2 : ℝ) * ⟪y - μ, A (y - μ)⟫)) x t
      = -(A (x - μ)) :=
  (ch01_gaussian_log_gradient A hA μ C hC x).gradient

-- Specialisation to the standard Gaussian on ℝⁿ, for every `n`: this is the
-- general-dimension replacement for the `n = 1` check `ch01_intro_score_gaussian`.
theorem ch01_intro_score_gaussian_multi (C : ℝ) (hC : 0 < C)
    (x : EuclideanSpace ℝ (Fin n)) (t : ℝ) :
    ch01_intro_score_multi (fun y _ => C * Real.exp (-(‖y‖ ^ 2 / 2))) x t = -x := by
  have hid : ∀ u v : EuclideanSpace ℝ (Fin n),
      ⟪(ContinuousLinearMap.id ℝ (EuclideanSpace ℝ (Fin n))) u, v⟫
        = ⟪u, (ContinuousLinearMap.id ℝ (EuclideanSpace ℝ (Fin n))) v⟫ := by
    intro u v
    simp
  have key := ch01_gaussian_score (ContinuousLinearMap.id ℝ (EuclideanSpace ℝ (Fin n)))
      hid (0 : EuclideanSpace ℝ (Fin n)) C hC x t
  have hfun : (fun (y : EuclideanSpace ℝ (Fin n)) (_ : ℝ) =>
        C * Real.exp (-(1 / 2 : ℝ) * ⟪y - (0 : EuclideanSpace ℝ (Fin n)),
          (ContinuousLinearMap.id ℝ (EuclideanSpace ℝ (Fin n))) (y - 0)⟫))
      = fun (y : EuclideanSpace ℝ (Fin n)) (_ : ℝ) => C * Real.exp (-(‖y‖ ^ 2 / 2)) := by
    funext y _
    have harg : (-(1 / 2 : ℝ)) * ⟪y - (0 : EuclideanSpace ℝ (Fin n)),
        (ContinuousLinearMap.id ℝ (EuclideanSpace ℝ (Fin n))) (y - 0)⟫
        = -(‖y‖ ^ 2 / 2) := by
      rw [sub_zero, ContinuousLinearMap.id_apply, real_inner_self_eq_norm_sq]
      ring
    rw [harg]
  rw [hfun] at key
  simpa using key

-- The same with the explicit normalising constant (2π)^(-n/2).
theorem ch01_intro_score_gaussian_multi_normalised
    (x : EuclideanSpace ℝ (Fin n)) (t : ℝ) :
    ch01_intro_score_multi
      (fun y _ => (2 * Real.pi) ^ (-(n : ℝ) / 2) * Real.exp (-(‖y‖ ^ 2 / 2))) x t = -x :=
  ch01_intro_score_gaussian_multi _ (Real.rpow_pos_of_pos (by positivity) _) x t

/-! #### The self-adjointness hypothesis is realised by every symmetric precision
matrix, in every dimension -- so `ch01_gaussian_score` is not vacuous. -/

noncomputable def ch01_precisionCLM (S : Matrix (Fin n) (Fin n) ℝ) :
    EuclideanSpace ℝ (Fin n) →L[ℝ] EuclideanSpace ℝ (Fin n) :=
  LinearMap.toContinuousLinearMap (Matrix.toEuclideanLin S)

theorem ch01_precisionCLM_apply (S : Matrix (Fin n) (Fin n) ℝ)
    (y : EuclideanSpace ℝ (Fin n)) (i : Fin n) :
    ch01_precisionCLM S y i = ∑ j, S i j * y j := by
  simp [ch01_precisionCLM, Matrix.mulVec, dotProduct]

theorem ch01_euclidean_inner_eq_sum (a b : EuclideanSpace ℝ (Fin n)) :
    ⟪a, b⟫ = ∑ i, a i * b i := by
  have h : ⟪a, b⟫ = ∑ i, b i * a i := by
    simp [PiLp.inner_apply, RCLike.inner_apply]
  rw [h]
  exact Finset.sum_congr rfl fun i _ => mul_comm _ _

theorem ch01_precisionCLM_selfAdjoint (S : Matrix (Fin n) (Fin n) ℝ)
    (hS : ∀ i j, S i j = S j i) (u v : EuclideanSpace ℝ (Fin n)) :
    ⟪ch01_precisionCLM S u, v⟫ = ⟪u, ch01_precisionCLM S v⟫ := by
  rw [ch01_euclidean_inner_eq_sum, ch01_euclidean_inner_eq_sum]
  have hL : ∑ i, (ch01_precisionCLM S u) i * v i
      = ∑ i, ∑ j, S i j * u j * v i := by
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [ch01_precisionCLM_apply, Finset.sum_mul]
  have hR : ∑ i, u i * (ch01_precisionCLM S v) i
      = ∑ i, ∑ j, u i * (S i j * v j) := by
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [ch01_precisionCLM_apply, Finset.mul_sum]
  rw [hL, hR]
  conv_rhs => rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
  rw [hS j i]
  ring

-- **eq:intro-score for a Gaussian given by a symmetric precision matrix**, in
-- every dimension `n`: the i-th component of the score is -(S (x - μ))_i.
theorem ch01_gaussian_score_precision_matrix (S : Matrix (Fin n) (Fin n) ℝ)
    (hS : ∀ i j, S i j = S j i) (μ : EuclideanSpace ℝ (Fin n)) (C : ℝ) (hC : 0 < C)
    (x : EuclideanSpace ℝ (Fin n)) (t : ℝ) (i : Fin n) :
    ch01_intro_score_multi
      (fun y _ => C * Real.exp (-(1 / 2 : ℝ) * ⟪y - μ, ch01_precisionCLM S (y - μ)⟫)) x t i
      = -∑ j, S i j * (x j - μ j) := by
  rw [ch01_gaussian_score (ch01_precisionCLM S) (ch01_precisionCLM_selfAdjoint S hS)
    μ C hC x t]
  simp [ch01_precisionCLM_apply, mul_sub, Finset.sum_sub_distrib]

end GeneralDimension

/-! ## eq:em-fixedpoint (ch09-em-kernel.tex lines 29-36) in the generality the chapter uses it

The display is

    θ^{(k+1)} - θ* = J(θ*) (θ^{(k)} - θ*) + O(‖θ^{(k)} - θ*‖²),   J = ∂M/∂θ|_{θ*},

for a parameter *vector* θ (note the norm bars) with a Jacobian *matrix* J, and the
chapter's argument is spectral throughout: "the spectrum of J", "a coordinate aligned
with an eigenvector of J with eigenvalue λ", and the two-rate prediction
λ_{κ₄} > λ_α.  Everything below is therefore stated on ℝ^d for an *arbitrary* d, with
`J` an arbitrary continuous linear map -- equivalently, via `ch09_matrixCLM`, an
arbitrary d × d matrix -- and the remainder is proved in the stated `O(‖·‖²)` form,
not only in the weaker Peano form `o(‖·‖)`.
-/

section EMFixedPoint

variable {d : ℕ}

/-- The continuous linear map on ℝ^d determined by a `d × d` real matrix.  This is how
the Jacobian `J = ∂M/∂θ` of eq:em-fixedpoint, "written for a map acting on column
vectors", enters. -/
noncomputable def ch09_matrixCLM (Jm : Matrix (Fin d) (Fin d) ℝ) :
    EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d) :=
  LinearMap.toContinuousLinearMap (Matrix.toEuclideanLin Jm)

theorem ch09_matrixCLM_apply (Jm : Matrix (Fin d) (Fin d) ℝ)
    (v : EuclideanSpace ℝ (Fin d)) (i : Fin d) :
    ch09_matrixCLM Jm v i = ∑ j, Jm i j * v j := by
  simp [ch09_matrixCLM, Matrix.mulVec, dotProduct]

/-- **eq:em-fixedpoint, Peano form, arbitrary dimension.**  For an EM map `M` on ℝ^d
differentiable at a fixed point `θ*` with derivative `J`,
`M θ - θ* - J (θ - θ*)` is `o(θ - θ*)`. -/
theorem ch09_em_fixedpoint_littleO_vec
    (M : EuclideanSpace ℝ (Fin d) → EuclideanSpace ℝ (Fin d))
    (θs : EuclideanSpace ℝ (Fin d))
    (J : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
    (hfix : M θs = θs) (hJ : HasFDerivAt M J θs) :
    (fun θ => M θ - θs - J (θ - θs)) =o[nhds θs] fun θ => θ - θs := by
  have h := hasFDerivAt_iff_isLittleO.mp hJ
  rw [hfix] at h
  exact h

/-- The same with the Jacobian presented as a `d × d` matrix, as in the thesis. -/
theorem ch09_em_fixedpoint_littleO_matrix
    (M : EuclideanSpace ℝ (Fin d) → EuclideanSpace ℝ (Fin d))
    (θs : EuclideanSpace ℝ (Fin d)) (Jm : Matrix (Fin d) (Fin d) ℝ)
    (hfix : M θs = θs) (hJ : HasFDerivAt M (ch09_matrixCLM Jm) θs) :
    (fun θ => M θ - θs - ch09_matrixCLM Jm (θ - θs)) =o[nhds θs] fun θ => θ - θs :=
  ch09_em_fixedpoint_littleO_vec M θs _ hfix hJ

/-! ### The spectral consequence the chapter actually uses

"A coordinate aligned with an eigenvector of `J` with eigenvalue `λ` has its error
contract as `λ^k`" is a statement about the *linearised* recursion `e_{k+1} = J e_k`,
and it is proved here at that face value in arbitrary dimension.  A scalar `J` has a
single eigenvalue, so this content is invisible in dimension one. -/

theorem ch09_em_linear_error_eigen
    (J : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d)) (lam : ℝ)
    (v : EuclideanSpace ℝ (Fin d)) (hv : J v = lam • v)
    (e : ℕ → EuclideanSpace ℝ (Fin d)) (h0 : e 0 = v) (hrec : ∀ k, e (k + 1) = J (e k)) :
    ∀ k, e k = (lam ^ k) • v := by
  intro k
  induction k with
  | zero => simpa using h0
  | succ k ih =>
      rw [hrec k, ih, ContinuousLinearMap.map_smul, hv, smul_smul, pow_succ]

/-- ... so its norm is exactly `|λ|^k ‖v‖`: geometric (linear) convergence at rate `λ`,
which is what "EM converges linearly, at a rate set by the spectrum of J" asserts. -/
theorem ch09_em_linear_error_eigen_norm
    (J : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d)) (lam : ℝ)
    (v : EuclideanSpace ℝ (Fin d)) (hv : J v = lam • v)
    (e : ℕ → EuclideanSpace ℝ (Fin d)) (h0 : e 0 = v) (hrec : ∀ k, e (k + 1) = J (e k))
    (k : ℕ) : ‖e k‖ = |lam| ^ k * ‖v‖ := by
  rw [ch09_em_linear_error_eigen J lam v hv e h0 hrec k, norm_smul, Real.norm_eq_abs,
    abs_pow]

/-! ### The stated `O(‖·‖²)` remainder

The Peano form above is what differentiability alone gives.  The remainder stated in
the thesis is `O(‖θ - θ*‖²)`, which needs one more degree of regularity.  The exact
hypothesis used here is that the derivative of `M` is Lipschitz at `θ*` with constant
`K` on a ball -- `‖M'(x) - J‖ ≤ K ‖x - θ*‖` -- which is what C² regularity supplies
(see `ch09_em_fixedpoint_bigO_sq_of_contDiffAt` below, where it is *derived* from
`ContDiffAt ℝ 2 M θ*`).  The bound obtained is explicit, with the same constant `K`. -/

theorem ch09_em_fixedpoint_remainder_sq
    (M : EuclideanSpace ℝ (Fin d) → EuclideanSpace ℝ (Fin d))
    (M' : EuclideanSpace ℝ (Fin d) →
      (EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d)))
    (θs : EuclideanSpace ℝ (Fin d))
    (J : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
    (r K : ℝ) (hK : 0 ≤ K) (hfix : M θs = θs)
    (hM : ∀ x ∈ Metric.closedBall θs r, HasFDerivAt M (M' x) x)
    (hlip : ∀ x ∈ Metric.closedBall θs r, ‖M' x - J‖ ≤ K * ‖x - θs‖)
    (θ : EuclideanSpace ℝ (Fin d)) (hθ : θ ∈ Metric.closedBall θs r) :
    ‖M θ - θs - J (θ - θs)‖ ≤ K * ‖θ - θs‖ ^ 2 := by
  have hθnorm : ‖θ - θs‖ ≤ r := by
    simpa [Metric.mem_closedBall, dist_eq_norm] using hθ
  set s : Set (EuclideanSpace ℝ (Fin d)) := Metric.closedBall θs ‖θ - θs‖ with hs
  have hsub : s ⊆ Metric.closedBall θs r := Metric.closedBall_subset_closedBall hθnorm
  have hθs_mem : θs ∈ s := by simp [hs]
  have hθ_mem : θ ∈ s := by simp [hs, Metric.mem_closedBall, dist_eq_norm]
  have hgderiv : ∀ x ∈ s,
      HasFDerivWithinAt (fun y => M y - θs - J (y - θs)) (M' x - J) s x := by
    intro x hx
    have h1 : HasFDerivAt M (M' x) x := hM x (hsub hx)
    have h2 : HasFDerivAt (fun y : EuclideanSpace ℝ (Fin d) => y - θs)
        (ContinuousLinearMap.id ℝ (EuclideanSpace ℝ (Fin d))) x :=
      (hasFDerivAt_id x).sub_const θs
    have h3 : HasFDerivAt (fun y : EuclideanSpace ℝ (Fin d) => J (y - θs))
        (J.comp (ContinuousLinearMap.id ℝ (EuclideanSpace ℝ (Fin d)))) x :=
      J.hasFDerivAt.comp x h2
    have h4 : HasFDerivAt (fun y => M y - θs - J (y - θs))
        (M' x - J.comp (ContinuousLinearMap.id ℝ (EuclideanSpace ℝ (Fin d)))) x :=
      (h1.sub_const θs).sub h3
    rw [ContinuousLinearMap.comp_id] at h4
    exact h4.hasFDerivWithinAt
  have hbound : ∀ x ∈ s, ‖M' x - J‖ ≤ K * ‖θ - θs‖ := by
    intro x hx
    refine (hlip x (hsub hx)).trans ?_
    have : ‖x - θs‖ ≤ ‖θ - θs‖ := by
      simpa [hs, Metric.mem_closedBall, dist_eq_norm] using hx
    exact mul_le_mul_of_nonneg_left this hK
  have hMVT := (convex_closedBall θs ‖θ - θs‖).norm_image_sub_le_of_norm_hasFDerivWithin_le
    hgderiv hbound hθs_mem hθ_mem
  have hzero : M θs - θs - J (θs - θs) = 0 := by
    rw [hfix, sub_self, map_zero, sub_zero]
  rw [hzero, sub_zero] at hMVT
  calc ‖M θ - θs - J (θ - θs)‖ ≤ K * ‖θ - θs‖ * ‖θ - θs‖ := hMVT
    _ = K * ‖θ - θs‖ ^ 2 := by ring

/-- **eq:em-fixedpoint with the remainder exactly as stated**: under the same
hypotheses, on a ball of positive radius, `M θ - θ* - J (θ - θ*) = O(‖θ - θ*‖²)`. -/
theorem ch09_em_fixedpoint_bigO_sq
    (M : EuclideanSpace ℝ (Fin d) → EuclideanSpace ℝ (Fin d))
    (M' : EuclideanSpace ℝ (Fin d) →
      (EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d)))
    (θs : EuclideanSpace ℝ (Fin d))
    (J : EuclideanSpace ℝ (Fin d) →L[ℝ] EuclideanSpace ℝ (Fin d))
    (r K : ℝ) (hr : 0 < r) (hK : 0 ≤ K) (hfix : M θs = θs)
    (hM : ∀ x ∈ Metric.closedBall θs r, HasFDerivAt M (M' x) x)
    (hlip : ∀ x ∈ Metric.closedBall θs r, ‖M' x - J‖ ≤ K * ‖x - θs‖) :
    (fun θ => M θ - θs - J (θ - θs)) =O[nhds θs] fun θ => ‖θ - θs‖ ^ 2 := by
  refine Asymptotics.IsBigO.of_bound K ?_
  filter_upwards [Metric.closedBall_mem_nhds θs hr] with θ hθ
  have h := ch09_em_fixedpoint_remainder_sq M M' θs J r K hK hfix hM hlip θ hθ
  calc ‖M θ - θs - J (θ - θs)‖ ≤ K * ‖θ - θs‖ ^ 2 := h
    _ = K * ‖(‖θ - θs‖ ^ 2)‖ := by
        rw [Real.norm_eq_abs, abs_of_nonneg (by positivity)]

/-- **eq:em-fixedpoint, exactly as stated, from C² regularity alone.**  If the EM map
is twice continuously differentiable at the fixed point -- the standard regularity
hypothesis under which the expansion in the thesis is written -- then

    M θ - θ* - J (θ - θ*) = O(‖θ - θ*‖²),   J = ∂M/∂θ|_{θ*} = fderiv ℝ M θ*,

in arbitrary dimension `d`.  Nothing beyond `ContDiffAt ℝ 2 M θ*` and `M θ* = θ*` is
assumed: the Lipschitz bound on the derivative that `ch09_em_fixedpoint_remainder_sq`
consumes is *derived* here, by bounding the second derivative on a compact ball and
applying the mean value inequality to `fderiv ℝ M`. -/
theorem ch09_em_fixedpoint_bigO_sq_of_contDiffAt
    (M : EuclideanSpace ℝ (Fin d) → EuclideanSpace ℝ (Fin d))
    (θs : EuclideanSpace ℝ (Fin d)) (hfix : M θs = θs)
    (hM : ContDiffAt ℝ 2 M θs) :
    (fun θ => M θ - θs - (fderiv ℝ M θs) (θ - θs)) =O[nhds θs]
      fun θ => ‖θ - θs‖ ^ 2 := by
  have hev : ∀ᶠ y in nhds θs, ContDiffAt ℝ 2 M y := hM.eventually (by simp)
  rw [Metric.eventually_nhds_iff] at hev
  obtain ⟨r₀, hr₀, hball⟩ := hev
  refine ?_
  set r : ℝ := r₀ / 2 with hrdef
  have hr : 0 < r := by positivity
  have hmem : ∀ x ∈ Metric.closedBall θs r, ContDiffAt ℝ 2 M x := by
    intro x hx
    refine hball ?_
    have hx' : dist x θs ≤ r := hx
    have : r < r₀ := by rw [hrdef]; linarith
    exact lt_of_le_of_lt hx' this
  have hdiff : ∀ x ∈ Metric.closedBall θs r, HasFDerivAt M (fderiv ℝ M x) x := by
    intro x hx
    exact ((hmem x hx).differentiableAt (by simp)).hasFDerivAt
  have hD1 : ∀ x ∈ Metric.closedBall θs r, ContDiffAt ℝ 1 (fderiv ℝ M) x := by
    intro x hx
    exact (hmem x hx).fderiv_right (m := 1) (by norm_num)
  have hDdiff : ∀ x ∈ Metric.closedBall θs r,
      HasFDerivAt (fderiv ℝ M) (fderiv ℝ (fderiv ℝ M) x) x := by
    intro x hx
    exact ((hD1 x hx).differentiableAt (by simp)).hasFDerivAt
  have hNcont : ContinuousOn (fun x => ‖fderiv ℝ (fderiv ℝ M) x‖)
      (Metric.closedBall θs r) := by
    intro x hx
    exact (((hD1 x hx).continuousAt_fderiv (by simp)).norm).continuousWithinAt
  have hcomp : IsCompact (Metric.closedBall θs r) := isCompact_closedBall θs r
  obtain ⟨x₀, hx₀, hmax⟩ := hcomp.exists_isMaxOn
    ⟨θs, Metric.mem_closedBall_self hr.le⟩ hNcont
  have hC : ∀ x ∈ Metric.closedBall θs r,
      ‖fderiv ℝ (fderiv ℝ M) x‖ ≤ ‖fderiv ℝ (fderiv ℝ M) x₀‖ :=
    fun x hx => isMaxOn_iff.mp hmax x hx
  have hK : (0 : ℝ) ≤ ‖fderiv ℝ (fderiv ℝ M) x₀‖ := ContinuousLinearMap.opNorm_nonneg _
  have hlip : ∀ x ∈ Metric.closedBall θs r,
      ‖fderiv ℝ M x - fderiv ℝ M θs‖ ≤ ‖fderiv ℝ (fderiv ℝ M) x₀‖ * ‖x - θs‖ := by
    intro x hx
    exact (convex_closedBall θs r).norm_image_sub_le_of_norm_hasFDerivWithin_le
      (fun y hy => (hDdiff y hy).hasFDerivWithinAt) (fun y hy => hC y hy)
      (Metric.mem_closedBall_self hr.le) hx
  exact ch09_em_fixedpoint_bigO_sq M (fderiv ℝ M) θs (fderiv ℝ M θs) r
    ‖fderiv ℝ (fderiv ℝ M) x₀‖ hr hK hfix hdiff hlip

end EMFixedPoint

/-! ## eq:em-cumulant-damping (ch09-em-kernel.tex lines 139-144) with genuine random
variables

The display is

    κ₂(X_t) = κ₂(A) for all t,     κ₄(X_t) = e^{-4t} κ₄(A),

for the variance-preserving OU channel `X_t = e^{-t} A + √(1 - e^{-2t}) ε` with `ε`
standard Gaussian independent of `A`, under the chapter's normalisation `κ₂(A) = 1`.

Everything here is about actual random variables on a probability space.  `κ₂` and `κ₄`
are *defined* from the raw moments, the transformation laws that the earlier
formalisation assumed -- additivity over independent summands, the homogeneity
`κ_n(cY) = cⁿ κ_n(Y)`, and the vanishing of the Gaussian fourth cumulant -- are all
*derived*: additivity and homogeneity from `IndepFun` (mixed moments factor), and
`κ₄(ε) = 0` from the moment-generating function of `gaussianReal 0 1`, whose fourth
derivative at the origin is computed here.  The only inputs are measurability,
independence, four finite moments for `A`, and the law of the noise. -/

section EMCumulants

open MeasureTheory ProbabilityTheory

variable {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω}

/-- The second cumulant of a real random variable, from its raw moments. -/
noncomputable def ch09_kappa2 (P : Measure Ω) (X : Ω → ℝ) : ℝ :=
  (∫ ω, X ω ^ 2 ∂P) - (∫ ω, X ω ∂P) ^ 2

/-- The fourth cumulant of a real random variable, from its raw moments:
`κ₄ = m₄ - 4 m₃ m₁ - 3 m₂² + 12 m₂ m₁² - 6 m₁⁴`. -/
noncomputable def ch09_kappa4 (P : Measure Ω) (X : Ω → ℝ) : ℝ :=
  (∫ ω, X ω ^ 4 ∂P) - 4 * (∫ ω, X ω ^ 3 ∂P) * (∫ ω, X ω ∂P)
    - 3 * (∫ ω, X ω ^ 2 ∂P) ^ 2 + 12 * (∫ ω, X ω ^ 2 ∂P) * (∫ ω, X ω ∂P) ^ 2
    - 6 * (∫ ω, X ω ∂P) ^ 4

/-- For a centred variable the fourth cumulant is the familiar `E X⁴ - 3 (E X²)²`. -/
theorem ch09_kappa4_centered (X : Ω → ℝ) (h : (∫ ω, X ω ∂P) = 0) :
    ch09_kappa4 P X = (∫ ω, X ω ^ 4 ∂P) - 3 * (∫ ω, X ω ^ 2 ∂P) ^ 2 := by
  simp [ch09_kappa4, h]

/-- `κ₂` is the variance. -/
theorem ch09_kappa2_eq_variance [IsProbabilityMeasure P] (X : Ω → ℝ) (hX : MemLp X 2 P) :
    ch09_kappa2 P X = Var[X; P] := by
  rw [variance_eq_sub hX]
  simp [ch09_kappa2, Pi.pow_apply]

/-- Mixed moments of independent variables factor. -/
theorem ch09_integral_pow_mul_pow (A Z : Ω → ℝ) (hA : Measurable A) (hZ : Measurable Z)
    (hAZ : IndepFun A Z P) (i j : ℕ) :
    ∫ ω, A ω ^ i * Z ω ^ j ∂P = (∫ ω, A ω ^ i ∂P) * (∫ ω, Z ω ^ j ∂P) := by
  have h : IndepFun (fun ω => A ω ^ i) (fun ω => Z ω ^ j) P :=
    hAZ.comp (measurable_id.pow_const i) (measurable_id.pow_const j)
  exact h.integral_mul_eq_mul_integral (hA.pow_const i).aestronglyMeasurable
    (hZ.pow_const j).aestronglyMeasurable

theorem ch09_integrable_pow_mul_pow (A Z : Ω → ℝ)
    (hAZ : IndepFun A Z P) (i j : ℕ)
    (hAi : Integrable (fun ω => A ω ^ i) P) (hZj : Integrable (fun ω => Z ω ^ j) P) :
    Integrable (fun ω => A ω ^ i * Z ω ^ j) P := by
  have h : IndepFun (fun ω => A ω ^ i) (fun ω => Z ω ^ j) P :=
    hAZ.comp (measurable_id.pow_const i) (measurable_id.pow_const j)
  simpa [Pi.mul_def] using h.integrable_mul hAi hZj

private theorem ch09_integral_add₃ (f₁ f₂ f₃ : Ω → ℝ)
    (h₁ : Integrable f₁ P) (h₂ : Integrable f₂ P) (h₃ : Integrable f₃ P) :
    ∫ ω, (f₁ ω + f₂ ω + f₃ ω) ∂P
      = (∫ ω, f₁ ω ∂P) + (∫ ω, f₂ ω ∂P) + (∫ ω, f₃ ω ∂P) := by
  have h₁₂ : Integrable (fun ω => f₁ ω + f₂ ω) P := h₁.add h₂
  rw [integral_add h₁₂ h₃, integral_add h₁ h₂]

private theorem ch09_integral_add₄ (f₁ f₂ f₃ f₄ : Ω → ℝ)
    (h₁ : Integrable f₁ P) (h₂ : Integrable f₂ P) (h₃ : Integrable f₃ P)
    (h₄ : Integrable f₄ P) :
    ∫ ω, (f₁ ω + f₂ ω + f₃ ω + f₄ ω) ∂P
      = (∫ ω, f₁ ω ∂P) + (∫ ω, f₂ ω ∂P) + (∫ ω, f₃ ω ∂P) + (∫ ω, f₄ ω ∂P) := by
  have h₁₂ : Integrable (fun ω => f₁ ω + f₂ ω) P := h₁.add h₂
  have h₁₂₃ : Integrable (fun ω => f₁ ω + f₂ ω + f₃ ω) P := h₁₂.add h₃
  rw [integral_add h₁₂₃ h₄, integral_add h₁₂ h₃, integral_add h₁ h₂]

private theorem ch09_integral_add₅ (f₁ f₂ f₃ f₄ f₅ : Ω → ℝ)
    (h₁ : Integrable f₁ P) (h₂ : Integrable f₂ P) (h₃ : Integrable f₃ P)
    (h₄ : Integrable f₄ P) (h₅ : Integrable f₅ P) :
    ∫ ω, (f₁ ω + f₂ ω + f₃ ω + f₄ ω + f₅ ω) ∂P
      = (∫ ω, f₁ ω ∂P) + (∫ ω, f₂ ω ∂P) + (∫ ω, f₃ ω ∂P) + (∫ ω, f₄ ω ∂P)
        + (∫ ω, f₅ ω ∂P) := by
  have h₁₂ : Integrable (fun ω => f₁ ω + f₂ ω) P := h₁.add h₂
  have h₁₂₃ : Integrable (fun ω => f₁ ω + f₂ ω + f₃ ω) P := h₁₂.add h₃
  have h₁₂₃₄ : Integrable (fun ω => f₁ ω + f₂ ω + f₃ ω + f₄ ω) P := h₁₂₃.add h₄
  rw [integral_add h₁₂₃₄ h₅, integral_add h₁₂₃ h₄, integral_add h₁₂ h₃, integral_add h₁ h₂]

/-- **Cumulant additivity and homogeneity, derived.**  For independent `A` and `Z` with
four finite moments and any scalars `c, s`, the second and fourth cumulants of
`X = c A + s Z` are

    κ₂(X) = c² κ₂(A) + s² κ₂(Z),      κ₄(X) = c⁴ κ₄(A) + s⁴ κ₄(Z).

Nothing about cumulants is assumed: the raw moments of `X` are expanded binomially,
every mixed moment is factored by independence, and the two identities then follow.
The `6 c² s²` cross terms in `m₄` cancel against those in `-3 m₂²` -- which is exactly
why the fourth cumulant, and not the fourth moment, is the quantity that damps. -/
theorem ch09_cumulants_of_indep_sum [IsProbabilityMeasure P] (A Z : Ω → ℝ) (c s : ℝ)
    (hA : Measurable A) (hZ : Measurable Z) (hAZ : IndepFun A Z P)
    (hA1 : Integrable A P) (hA2 : Integrable (fun ω => A ω ^ 2) P)
    (hA3 : Integrable (fun ω => A ω ^ 3) P) (hA4 : Integrable (fun ω => A ω ^ 4) P)
    (hZ1 : Integrable Z P) (hZ2 : Integrable (fun ω => Z ω ^ 2) P)
    (hZ3 : Integrable (fun ω => Z ω ^ 3) P) (hZ4 : Integrable (fun ω => Z ω ^ 4) P) :
    ch09_kappa2 P (fun ω => c * A ω + s * Z ω)
        = c ^ 2 * ch09_kappa2 P A + s ^ 2 * ch09_kappa2 P Z ∧
      ch09_kappa4 P (fun ω => c * A ω + s * Z ω)
        = c ^ 4 * ch09_kappa4 P A + s ^ 4 * ch09_kappa4 P Z := by
  have hA1' : Integrable (fun ω => A ω ^ 1) P := by simpa using hA1
  have hZ1' : Integrable (fun ω => Z ω ^ 1) P := by simpa using hZ1
  -- first moment
  have hm1 : ∫ ω, (c * A ω + s * Z ω) ∂P
      = c * (∫ ω, A ω ∂P) + s * (∫ ω, Z ω ∂P) := by
    rw [integral_add (hA1.const_mul c) (hZ1.const_mul s), integral_const_mul,
      integral_const_mul]
  -- second moment
  have hexp2 : ∀ ω, (c * A ω + s * Z ω) ^ 2
      = c ^ 2 * A ω ^ 2 + (2 * c * s) * (A ω ^ 1 * Z ω ^ 1) + s ^ 2 * Z ω ^ 2 :=
    fun ω => by ring
  have hm2 : ∫ ω, (c * A ω + s * Z ω) ^ 2 ∂P
      = c ^ 2 * (∫ ω, A ω ^ 2 ∂P)
        + (2 * c * s) * ((∫ ω, A ω ∂P) * (∫ ω, Z ω ∂P))
        + s ^ 2 * (∫ ω, Z ω ^ 2 ∂P) := by
    simp only [hexp2]
    rw [ch09_integral_add₃ _ _ _ (hA2.const_mul _)
        ((ch09_integrable_pow_mul_pow A Z hAZ 1 1 hA1' hZ1').const_mul _)
        (hZ2.const_mul _)]
    simp only [integral_const_mul]
    rw [ch09_integral_pow_mul_pow A Z hA hZ hAZ 1 1]
    simp only [pow_one]
  -- third moment
  have hexp3 : ∀ ω, (c * A ω + s * Z ω) ^ 3
      = c ^ 3 * A ω ^ 3 + (3 * c ^ 2 * s) * (A ω ^ 2 * Z ω ^ 1)
        + (3 * c * s ^ 2) * (A ω ^ 1 * Z ω ^ 2) + s ^ 3 * Z ω ^ 3 :=
    fun ω => by ring
  have hm3 : ∫ ω, (c * A ω + s * Z ω) ^ 3 ∂P
      = c ^ 3 * (∫ ω, A ω ^ 3 ∂P)
        + (3 * c ^ 2 * s) * ((∫ ω, A ω ^ 2 ∂P) * (∫ ω, Z ω ∂P))
        + (3 * c * s ^ 2) * ((∫ ω, A ω ∂P) * (∫ ω, Z ω ^ 2 ∂P))
        + s ^ 3 * (∫ ω, Z ω ^ 3 ∂P) := by
    simp only [hexp3]
    rw [ch09_integral_add₄ _ _ _ _ (hA3.const_mul _)
        ((ch09_integrable_pow_mul_pow A Z hAZ 2 1 hA2 hZ1').const_mul _)
        ((ch09_integrable_pow_mul_pow A Z hAZ 1 2 hA1' hZ2).const_mul _)
        (hZ3.const_mul _)]
    simp only [integral_const_mul]
    rw [ch09_integral_pow_mul_pow A Z hA hZ hAZ 2 1,
      ch09_integral_pow_mul_pow A Z hA hZ hAZ 1 2]
    simp only [pow_one]
  -- fourth moment
  have hexp4 : ∀ ω, (c * A ω + s * Z ω) ^ 4
      = c ^ 4 * A ω ^ 4 + (4 * c ^ 3 * s) * (A ω ^ 3 * Z ω ^ 1)
        + (6 * c ^ 2 * s ^ 2) * (A ω ^ 2 * Z ω ^ 2)
        + (4 * c * s ^ 3) * (A ω ^ 1 * Z ω ^ 3) + s ^ 4 * Z ω ^ 4 :=
    fun ω => by ring
  have hm4 : ∫ ω, (c * A ω + s * Z ω) ^ 4 ∂P
      = c ^ 4 * (∫ ω, A ω ^ 4 ∂P)
        + (4 * c ^ 3 * s) * ((∫ ω, A ω ^ 3 ∂P) * (∫ ω, Z ω ∂P))
        + (6 * c ^ 2 * s ^ 2) * ((∫ ω, A ω ^ 2 ∂P) * (∫ ω, Z ω ^ 2 ∂P))
        + (4 * c * s ^ 3) * ((∫ ω, A ω ∂P) * (∫ ω, Z ω ^ 3 ∂P))
        + s ^ 4 * (∫ ω, Z ω ^ 4 ∂P) := by
    simp only [hexp4]
    rw [ch09_integral_add₅ _ _ _ _ _ (hA4.const_mul _)
        ((ch09_integrable_pow_mul_pow A Z hAZ 3 1 hA3 hZ1').const_mul _)
        ((ch09_integrable_pow_mul_pow A Z hAZ 2 2 hA2 hZ2).const_mul _)
        ((ch09_integrable_pow_mul_pow A Z hAZ 1 3 hA1' hZ3).const_mul _)
        (hZ4.const_mul _)]
    simp only [integral_const_mul]
    rw [ch09_integral_pow_mul_pow A Z hA hZ hAZ 3 1,
      ch09_integral_pow_mul_pow A Z hA hZ hAZ 2 2,
      ch09_integral_pow_mul_pow A Z hA hZ hAZ 1 3]
    simp only [pow_one]
  constructor
  · simp only [ch09_kappa2]
    rw [hm1, hm2]
    ring
  · simp only [ch09_kappa4]
    rw [hm1, hm2, hm3, hm4]
    ring

/-! ### The noise is a genuine standard Gaussian, and its fourth cumulant is *derived* -/

private theorem ch09_gauss_deriv :
    (deriv (fun t : ℝ => Real.exp (t ^ 2 / 2)) = fun t : ℝ => t * Real.exp (t ^ 2 / 2)) ∧
    (deriv (fun t : ℝ => t * Real.exp (t ^ 2 / 2))
        = fun t : ℝ => (1 + t ^ 2) * Real.exp (t ^ 2 / 2)) ∧
    (deriv (fun t : ℝ => (1 + t ^ 2) * Real.exp (t ^ 2 / 2))
        = fun t : ℝ => (3 * t + t ^ 3) * Real.exp (t ^ 2 / 2)) ∧
    (deriv (fun t : ℝ => (3 * t + t ^ 3) * Real.exp (t ^ 2 / 2))
        = fun t : ℝ => (3 + 6 * t ^ 2 + t ^ 4) * Real.exp (t ^ 2 / 2)) := by
  have hsq : ∀ t : ℝ, HasDerivAt (fun s : ℝ => s ^ 2 / 2) t t := by
    intro t
    have h : HasDerivAt (fun s : ℝ => s ^ 2) (2 * t) t := by simpa using hasDerivAt_pow 2 t
    have h' := h.div_const 2
    have hx : 2 * t / 2 = t := by ring
    rwa [hx] at h'
  have hE : ∀ t : ℝ, HasDerivAt (fun s : ℝ => Real.exp (s ^ 2 / 2))
      (t * Real.exp (t ^ 2 / 2)) t := by
    intro t
    have h := (hsq t).exp
    rwa [mul_comm] at h
  have step : ∀ Pp Qp Rp : ℝ → ℝ, (∀ t, HasDerivAt Pp (Qp t) t) →
      (∀ t, Qp t + Pp t * t = Rp t) →
      ∀ t : ℝ, HasDerivAt (fun s => Pp s * Real.exp (s ^ 2 / 2))
        (Rp t * Real.exp (t ^ 2 / 2)) t := by
    intro Pp Qp Rp hPQ hR t
    have h := (hPQ t).mul (hE t)
    have hrw : Qp t * Real.exp (t ^ 2 / 2) + Pp t * (t * Real.exp (t ^ 2 / 2))
        = Rp t * Real.exp (t ^ 2 / 2) := by rw [← hR t]; ring
    rwa [hrw] at h
  have hid : ∀ t : ℝ, HasDerivAt (fun s : ℝ => s) 1 t := fun t => hasDerivAt_id t
  have hq2 : ∀ t : ℝ, HasDerivAt (fun s : ℝ => 1 + s ^ 2) (2 * t) t := by
    intro t
    have h : HasDerivAt (fun s : ℝ => s ^ 2) (2 * t) t := by simpa using hasDerivAt_pow 2 t
    simpa using h.const_add 1
  have hq3 : ∀ t : ℝ, HasDerivAt (fun s : ℝ => 3 * s + s ^ 3) (3 + 3 * t ^ 2) t := by
    intro t
    have ha : HasDerivAt (fun s : ℝ => 3 * s) 3 t := by simpa using (hid t).const_mul (3 : ℝ)
    have hb : HasDerivAt (fun s : ℝ => s ^ 3) (3 * t ^ 2) t := by
      simpa using hasDerivAt_pow 3 t
    exact ha.add hb
  refine ⟨funext fun t => (hE t).deriv, funext fun t => ?_, funext fun t => ?_,
    funext fun t => ?_⟩
  · exact (step (fun s => s) (fun _ => 1) (fun s => 1 + s ^ 2) hid
      (fun t => by ring) t).deriv
  · exact (step (fun s => 1 + s ^ 2) (fun t => 2 * t) (fun s => 3 * s + s ^ 3) hq2
      (fun t => by ring) t).deriv
  · exact (step (fun s => 3 * s + s ^ 3) (fun t => 3 + 3 * t ^ 2)
      (fun s => 3 + 6 * s ^ 2 + s ^ 4) hq3 (fun t => by ring) t).deriv

private theorem ch09_gauss_iteratedDeriv :
    iteratedDeriv 1 (fun t : ℝ => Real.exp (t ^ 2 / 2)) 0 = 0 ∧
    iteratedDeriv 2 (fun t : ℝ => Real.exp (t ^ 2 / 2)) 0 = 1 ∧
    iteratedDeriv 3 (fun t : ℝ => Real.exp (t ^ 2 / 2)) 0 = 0 ∧
    iteratedDeriv 4 (fun t : ℝ => Real.exp (t ^ 2 / 2)) 0 = 3 := by
  obtain ⟨D1, D2, D3, D4⟩ := ch09_gauss_deriv
  have I1 : iteratedDeriv 1 (fun t : ℝ => Real.exp (t ^ 2 / 2))
      = fun t : ℝ => t * Real.exp (t ^ 2 / 2) := by
    rw [iteratedDeriv_one]; exact D1
  have I2 : iteratedDeriv 2 (fun t : ℝ => Real.exp (t ^ 2 / 2))
      = fun t : ℝ => (1 + t ^ 2) * Real.exp (t ^ 2 / 2) := by
    show iteratedDeriv (1 + 1) (fun t : ℝ => Real.exp (t ^ 2 / 2)) = _
    rw [iteratedDeriv_succ, I1]; exact D2
  have I3 : iteratedDeriv 3 (fun t : ℝ => Real.exp (t ^ 2 / 2))
      = fun t : ℝ => (3 * t + t ^ 3) * Real.exp (t ^ 2 / 2) := by
    show iteratedDeriv (2 + 1) (fun t : ℝ => Real.exp (t ^ 2 / 2)) = _
    rw [iteratedDeriv_succ, I2]; exact D3
  have I4 : iteratedDeriv 4 (fun t : ℝ => Real.exp (t ^ 2 / 2))
      = fun t : ℝ => (3 + 6 * t ^ 2 + t ^ 4) * Real.exp (t ^ 2 / 2) := by
    show iteratedDeriv (3 + 1) (fun t : ℝ => Real.exp (t ^ 2 / 2)) = _
    rw [iteratedDeriv_succ, I3]; exact D4
  refine ⟨by rw [I1]; simp, by rw [I2]; simp, by rw [I3]; simp, by rw [I4]; simp⟩

/-- **The first four moments of a standard Gaussian**, read off the derivatives of its
moment-generating function `t ↦ e^{t²/2}` at the origin: `0, 1, 0, 3`.  All powers are
integrable. -/
theorem ch09_standard_gaussian_moments [IsProbabilityMeasure P] (Z : Ω → ℝ)
    (hZlaw : P.map Z = gaussianReal 0 1) :
    (∀ i : ℕ, Integrable (fun ω => Z ω ^ i) P) ∧ (∫ ω, Z ω ∂P) = 0 ∧
      (∫ ω, Z ω ^ 2 ∂P) = 1 ∧ (∫ ω, Z ω ^ 3 ∂P) = 0 ∧ (∫ ω, Z ω ^ 4 ∂P) = 3 := by
  have hZm : AEMeasurable Z P := aemeasurable_of_map_neZero (by rw [hZlaw]; infer_instance)
  have hexp : ∀ t : ℝ, Integrable (fun ω => Real.exp (t * Z ω)) P := by
    intro t
    have hc : Continuous fun x : ℝ => Real.exp (t * x) :=
      Real.continuous_exp.comp (continuous_const.mul continuous_id)
    have h1 : Integrable (fun x : ℝ => Real.exp (t * x)) (P.map Z) := by
      rw [hZlaw]; exact integrable_exp_mul_gaussianReal t
    rw [integrable_map_measure hc.aestronglyMeasurable hZm] at h1
    exact h1
  have huniv : integrableExpSet Z P = Set.univ := Set.eq_univ_of_forall fun t => hexp t
  have hset : (0 : ℝ) ∈ interior (integrableExpSet Z P) := by
    rw [huniv, interior_univ]; trivial
  have hint : ∀ i : ℕ, Integrable (fun ω => Z ω ^ i) P := fun i =>
    integrable_pow_of_mem_interior_integrableExpSet hset i
  have hmgf : mgf Z P = fun t : ℝ => Real.exp (t ^ 2 / 2) := by
    funext t
    rw [mgf_gaussianReal hZlaw]
    norm_num
  have hmom : ∀ i : ℕ, ∫ ω, Z ω ^ i ∂P = iteratedDeriv i (mgf Z P) 0 := by
    intro i
    rw [iteratedDeriv_mgf_zero hset i]
    simp [Pi.pow_apply]
  obtain ⟨G1, G2, G3, G4⟩ := ch09_gauss_iteratedDeriv
  refine ⟨hint, ?_, ?_, ?_, ?_⟩
  · have h := hmom 1
    rw [hmgf, G1] at h
    simpa using h
  · have h := hmom 2
    rw [hmgf, G2] at h
    exact h
  · have h := hmom 3
    rw [hmgf, G3] at h
    exact h
  · have h := hmom 4
    rw [hmgf, G4] at h
    exact h

/-- **The Gaussian second and fourth cumulants**, derived: `κ₂ = 1`, `κ₄ = 0`. -/
theorem ch09_standard_gaussian_cumulants [IsProbabilityMeasure P] (Z : Ω → ℝ)
    (hZlaw : P.map Z = gaussianReal 0 1) :
    ch09_kappa2 P Z = 1 ∧ ch09_kappa4 P Z = 0 := by
  obtain ⟨-, h1, h2, h3, h4⟩ := ch09_standard_gaussian_moments Z hZlaw
  constructor
  · rw [ch09_kappa2, h1, h2]; ring
  · rw [ch09_kappa4, h1, h2, h3, h4]; ring

/-! ### eq:em-cumulant-damping -/

/-- **eq:em-cumulant-damping.**  For the variance-preserving OU channel

    X = e^{-t} A + √(1 - e^{-2t}) Z,   Z standard Gaussian, independent of A,

with `A` measurable with four finite moments and normalised to `κ₂(A) = 1` (the
chapter's normalisation `ν = 1 - α²`), the second cumulant is preserved exactly at
every `t ≥ 0` and the fourth is damped by `e^{-4t}`.

Both cumulant transformation laws are derived, not assumed: additivity/homogeneity
come from `ch09_cumulants_of_indep_sum` (independence, mixed moments factoring) and
`κ₄(Z) = 0` from `ch09_standard_gaussian_cumulants` (the Gaussian mgf).  Note that the
`κ₂` half genuinely needs the variance matching, while the `e^{-4t}` damping of `κ₄`
needs only that the noise is Gaussian. -/
theorem ch09_em_cumulant_damping_rv [IsProbabilityMeasure P] (A Z X : Ω → ℝ) (t : ℝ)
    (ht : 0 ≤ t) (hA : Measurable A) (hZ : Measurable Z) (hAZ : IndepFun A Z P)
    (hA1 : Integrable A P) (hA2 : Integrable (fun ω => A ω ^ 2) P)
    (hA3 : Integrable (fun ω => A ω ^ 3) P) (hA4 : Integrable (fun ω => A ω ^ 4) P)
    (hZlaw : P.map Z = gaussianReal 0 1)
    (hnorm : ch09_kappa2 P A = 1)
    (hX : ∀ ω, X ω
      = Real.exp (-t) * A ω + Real.sqrt (1 - Real.exp (-(2 * t))) * Z ω) :
    ch09_kappa2 P X = ch09_kappa2 P A ∧
      ch09_kappa4 P X = Real.exp (-(4 * t)) * ch09_kappa4 P A := by
  obtain ⟨hZint, -, -, -, -⟩ := ch09_standard_gaussian_moments Z hZlaw
  obtain ⟨hZk2, hZk4⟩ := ch09_standard_gaussian_cumulants Z hZlaw
  have hZ1 : Integrable Z P := by simpa using hZint 1
  have hXfun : X = fun ω => Real.exp (-t) * A ω
      + Real.sqrt (1 - Real.exp (-(2 * t))) * Z ω := funext hX
  obtain ⟨hk2, hk4⟩ := ch09_cumulants_of_indep_sum A Z (Real.exp (-t))
    (Real.sqrt (1 - Real.exp (-(2 * t)))) hA hZ hAZ hA1 hA2 hA3 hA4 hZ1 (hZint 2)
    (hZint 3) (hZint 4)
  rw [hXfun]
  have hnn : (0 : ℝ) ≤ 1 - Real.exp (-(2 * t)) := by
    have h1 : Real.exp (-(2 * t)) ≤ 1 := by
      rw [Real.exp_le_one_iff]; linarith
    linarith
  have hs2 : Real.sqrt (1 - Real.exp (-(2 * t))) ^ 2 = 1 - Real.exp (-(2 * t)) :=
    Real.sq_sqrt hnn
  have hs4 : Real.sqrt (1 - Real.exp (-(2 * t))) ^ 4
      = (1 - Real.exp (-(2 * t))) ^ 2 := by
    rw [show (4 : ℕ) = 2 * 2 from rfl, pow_mul, hs2]
  have hc2 : Real.exp (-t) ^ 2 = Real.exp (-(2 * t)) := by
    rw [← Real.exp_nat_mul]
    congr 1
    push_cast
    ring
  have hc4 : Real.exp (-t) ^ 4 = Real.exp (-(4 * t)) := by
    rw [← Real.exp_nat_mul]
    congr 1
    push_cast
    ring
  constructor
  · rw [hk2, hZk2, hnorm, hc2, hs2]; ring
  · rw [hk4, hZk4, hc4, hs4]; ring

end EMCumulants

end ThesisAudit
