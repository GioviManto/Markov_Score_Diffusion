import Mathlib

/-!
Audit of the display-math formulas in
`thesis/chapters/ch07-laplace.tex`, lines 1-746
("Beyond Gaussianity: Laplace Innovations and the Gaussian-Message
Approximation").

Conventions used throughout this file:

* `Δ_t = 1 - e^{-2t}` and `μ = e^{-t}` are the variance-preserving OU channel
  parameters fixed in Chapters 3 and 5 (`ch07_Delta`, `ch07_mu`).
* Cumulants are not available in mathlib, so the fourth-cumulant claims are
  checked as the *arithmetic* they reduce to once the two standard cumulant
  rules (additivity over independent summands, homogeneity of degree four) are
  granted; the rules themselves are stated in the comment above each claim.
* Statements about conditional expectations of Gaussian vectors are checked
  through the completed square of the joint density, which is where the
  conditional mean actually comes from.
* Structural sum-product statements are checked on a discrete alphabet, where
  the Fubini step is a finite `Finset.sum_comm`; sizes are recorded per lemma.
* Grid/quadrature formulas are checked as the exact finite identities they are
  (the trapezoid weights, the matrix-vector form of the two sweeps, the scale
  invariance of the reported ratio).
-/

namespace ThesisAudit

open Real MeasureTheory Matrix

noncomputable section

/-! ### Shared definitions -/

/-- `Δ_t = 1 - e^{-2t}`. -/
def ch07_Delta (t : ℝ) : ℝ := 1 - Real.exp (-2 * t)

/-- `μ = e^{-t}`. -/
def ch07_mu (t : ℝ) : ℝ := Real.exp (-t)

/-- The normalised scalar Gaussian density `N(x; m, v)`. -/
def ch07_gauss (m v x : ℝ) : ℝ :=
  (1 / Real.sqrt (2 * π * v)) * Real.exp (-(x - m) ^ 2 / (2 * v))

/-- The centred Laplace density with scale `b`. -/
def ch07_laplacePdf (b r : ℝ) : ℝ := (1 / (2 * b)) * Real.exp (-|r| / b)

/-- The standard Gaussian cdf `Φ`. -/
def ch07_Phi (z : ℝ) : ℝ := ∫ u in Set.Iic z, (1 / Real.sqrt (2 * π)) * Real.exp (-u ^ 2 / 2)

/-! ### The Gaussian integral with a linear term (workhorse) -/

theorem ch07_gauss_integral {A : ℝ} (hA : 0 < A) (B : ℝ) :
    (∫ u : ℝ, Real.exp (-(A / 2) * u ^ 2 + B * u))
      = Real.exp (B ^ 2 / (2 * A)) * Real.sqrt (π / (A / 2)) := by
  have hA' : A ≠ 0 := ne_of_gt hA
  have hsq : ∀ u : ℝ, Real.exp (-(A / 2) * u ^ 2 + B * u)
      = Real.exp (B ^ 2 / (2 * A)) * Real.exp (-(A / 2) * (u - B / A) ^ 2) := by
    intro u
    rw [← Real.exp_add]
    congr 1
    field_simp
    ring
  simp_rw [hsq]
  rw [MeasureTheory.integral_const_mul]
  have htrans : (∫ u : ℝ, Real.exp (-(A / 2) * (u - B / A) ^ 2))
      = ∫ u : ℝ, Real.exp (-(A / 2) * u ^ 2) :=
    integral_sub_right_eq_self (fun y : ℝ => Real.exp (-(A / 2) * y ^ 2)) (B / A)
  rw [htrans, integral_gaussian (A / 2)]

/-- Gaussian convolution: `∫ exp(-(a-αu)²/2q) exp(-(u-m)²/2v) du` is a Gaussian
kernel in `a` with mean `αm` and variance `α²v+q`.  This is the analytic engine
behind both moment maps of Section 7.4.1. -/
theorem ch07_gauss_conv {q v : ℝ} (hq : 0 < q) (hv : 0 < v) (α a m : ℝ) :
    (∫ u : ℝ, Real.exp (-(a - α * u) ^ 2 / (2 * q)) * Real.exp (-(u - m) ^ 2 / (2 * v)))
      = Real.sqrt (π / ((α ^ 2 / q + 1 / v) / 2))
        * Real.exp (-(a - α * m) ^ 2 / (2 * (α ^ 2 * v + q))) := by
  have hq' : q ≠ 0 := ne_of_gt hq
  have hv' : v ≠ 0 := ne_of_gt hv
  have hApos : 0 < α ^ 2 / q + 1 / v := by
    have h1 : 0 ≤ α ^ 2 / q := div_nonneg (sq_nonneg α) hq.le
    have h2 : 0 < 1 / v := one_div_pos.mpr hv
    linarith
  have hA' : α ^ 2 / q + 1 / v ≠ 0 := ne_of_gt hApos
  have hqv : (0:ℝ) < α ^ 2 * v + q := by nlinarith [sq_nonneg α]
  have hqv' : α ^ 2 * v + q ≠ 0 := ne_of_gt hqv
  have hpt : ∀ u : ℝ, Real.exp (-(a - α * u) ^ 2 / (2 * q)) * Real.exp (-(u - m) ^ 2 / (2 * v))
      = Real.exp (-(a ^ 2 / (2 * q)) - m ^ 2 / (2 * v))
        * Real.exp (-((α ^ 2 / q + 1 / v) / 2) * u ^ 2 + (α * a / q + m / v) * u) := by
    intro u
    rw [← Real.exp_add, ← Real.exp_add]
    congr 1
    field_simp
    ring
  simp_rw [hpt]
  rw [MeasureTheory.integral_const_mul, ch07_gauss_integral hApos]
  have key : Real.exp (-(a ^ 2 / (2 * q)) - m ^ 2 / (2 * v))
      * Real.exp ((α * a / q + m / v) ^ 2 / (2 * (α ^ 2 / q + 1 / v)))
      = Real.exp (-(a - α * m) ^ 2 / (2 * (α ^ 2 * v + q))) := by
    rw [← Real.exp_add]
    congr 1
    field_simp
    ring
  rw [← mul_assoc, key]
  ring

/-! ## Section 7.1 -- the Laplace chain

### eq:l-chain-prior (ch07-laplace.tex lines 20-26)

`a_0 ~ N(0,1)`, `a_k = α a_{k-1} + ε_k` with `ε_k ~ Laplace(0,b)` i.i.d.,
`b = √((1-α²)/2)`.  This is a *definition*; the claims made about it in the
surrounding prose (unit innovation variance `1-α²`, unit marginal variance,
`Cov(a_j,a_k) = α^{|j-k|}`) are the lemmas below. -/

/-- The innovation scale `b = √((1-α²)/2)` of eq:l-chain-prior. -/
def ch07_l_chain_prior_b (α : ℝ) : ℝ := Real.sqrt ((1 - α ^ 2) / 2)

/-- A `Laplace(0,b)` variable has variance `2b²`; with `b = √((1-α²)/2)` this is
`1-α²`, as the text requires. -/
theorem ch07_l_chain_prior_ivar {α : ℝ} (h : α ^ 2 ≤ 1) :
    2 * (ch07_l_chain_prior_b α) ^ 2 = 1 - α ^ 2 := by
  unfold ch07_l_chain_prior_b
  rw [Real.sq_sqrt (by linarith)]
  ring

/-- The variance recursion `v_0 = 1`, `v_k = α² v_{k-1} + (1-α²)`. -/
def ch07_chain_var (α : ℝ) : ℕ → ℝ
  | 0 => 1
  | (k + 1) => α ^ 2 * ch07_chain_var α k + (1 - α ^ 2)

/-- Unit marginal variance at every site. -/
theorem ch07_l_chain_prior_unit_variance (α : ℝ) (k : ℕ) : ch07_chain_var α k = 1 := by
  induction k with
  | zero => rfl
  | succ n ih => simp only [ch07_chain_var, ih]; ring

/-- The covariance recursion `c_0 = Var(a_j) = 1`, `c_{m+1} = α c_m`. -/
def ch07_chain_cov (α : ℝ) : ℕ → ℝ
  | 0 => 1
  | (m + 1) => α * ch07_chain_cov α m

/-- `Cov(a_j, a_{j+m}) = α^m`, exactly as in the Gaussian case. -/
theorem ch07_l_chain_prior_cov (α : ℝ) (m : ℕ) : ch07_chain_cov α m = α ^ m := by
  induction m with
  | zero => rfl
  | succ n ih => simp only [ch07_chain_cov, ih]; ring

/-! ## Section 7.2 -- the score

### eq:l-score-identity (lines 49-52)

`S_k(x,t) = (μ E[a_k|x] - x_k)/Δ_t` with `μ = e^{-t}`.  Tweedie's identity in
general needs differentiation under the integral sign; we check the scalar
Gaussian instance, where both sides are available in closed form: for a prior
`a ~ N(0,σ²)` the posterior mean is `μσ²x/(μ²σ²+Δ)` and the marginal of `x` is
`N(0, μ²σ²+Δ)`, whose score is `-x/(μ²σ²+Δ)`. -/

/-- The Gaussian posterior mean `E[a|x] = μσ²/(μ²σ²+Δ) · x`. -/
def ch07_post_mean_gauss (μ σ2 Δ x : ℝ) : ℝ := (μ * σ2 / (μ ^ 2 * σ2 + Δ)) * x

/-- The score of the marginal `N(0,V)` is `-x/V`. -/
theorem ch07_l_score_identity_marginal {V : ℝ} (hV : V ≠ 0) (c x : ℝ) :
    HasDerivAt (fun y : ℝ => -(y ^ 2) / (2 * V) + c) (-x / V) x := by
  have hp : HasDerivAt (fun y : ℝ => y ^ 2) (2 * x) x := by
    simpa using hasDerivAt_pow 2 x
  have hfun : (fun y : ℝ => (-1 / (2 * V)) * y ^ 2 + c)
      = fun y : ℝ => -(y ^ 2) / (2 * V) + c := by
    funext y; ring
  have hd : (-1 / (2 * V)) * (2 * x) = -x / V := by
    first
      | (field_simp; ring)
      | field_simp
      | ring
  have h := (hp.const_mul (-1 / (2 * V))).add_const c
  first
    | (rw [hfun, hd] at h; exact h)
    | (rw [hfun, add_zero, hd] at h; exact h)

/-- eq:l-score-identity, scalar Gaussian instance: the Tweedie combination of
the posterior mean *is* the score of the marginal. -/
theorem ch07_l_score_identity_gauss {μ σ2 Δ : ℝ} (hΔ : Δ ≠ 0) (hV : μ ^ 2 * σ2 + Δ ≠ 0) (x : ℝ) :
    (μ * ch07_post_mean_gauss μ σ2 Δ x - x) / Δ = -x / (μ ^ 2 * σ2 + Δ) := by
  unfold ch07_post_mean_gauss
  field_simp
  ring

/-- Consistency check with the chapter's own normalisation: for the unit-variance
marginal (`σ² = 1`) and the VP channel (`Δ = Δ_t`, `μ = e^{-t}`), the posterior
mean is `e^{-t} x` and the score is exactly `-x`. -/
theorem ch07_l_score_identity_vp (t x : ℝ) :
    ch07_post_mean_gauss (ch07_mu t) 1 (ch07_Delta t) x = Real.exp (-t) * x := by
  unfold ch07_post_mean_gauss ch07_mu ch07_Delta
  have h : Real.exp (-t) ^ 2 * 1 + (1 - Real.exp (-2 * t)) = 1 := by
    have : Real.exp (-t) ^ 2 = Real.exp (-2 * t) := by
      rw [pow_two, ← Real.exp_add]; congr 1; ring
    rw [this]; ring
  rw [h]
  ring

/-! A second, genuinely *non-Gaussian* check of eq:l-score-identity.  Take any
finitely supported prior `p_0 = Σ_k w_k δ_{v_k}` (weights and atoms arbitrary
reals) and push it through the same OU channel `x = μ a + √Δ z`.  The marginal
density of `x` is then proportional to `p(x) = Σ_k w_k exp(-(x-μ v_k)²/(2Δ))`,
and the posterior mean is `E[a|x] = Σ_k w_k v_k e_k / Σ_k w_k e_k`.  The two
lemmas below say exactly that `d/dx log p(x) = (μ E[a|x] - x)/Δ`, i.e.
eq:l-score-identity, with no Gaussianity assumed anywhere: the score is a ratio
of two *different* weighted sums, hence nonlinear in `x` for a general prior,
which is the point of Section 7.2.  (Dropping the normalising constant of `p_0`
is harmless: it shifts `log p` by a constant.) -/

theorem ch07_l_score_identity_discrete {K : ℕ} (w v : Fin K → ℝ) {Δ : ℝ} (hΔ : Δ ≠ 0)
    (μ x : ℝ) :
    HasDerivAt (fun y : ℝ => ∑ k : Fin K, w k * Real.exp (-(y - μ * v k) ^ 2 / (2 * Δ)))
      ((μ * (∑ k : Fin K, w k * v k * Real.exp (-(x - μ * v k) ^ 2 / (2 * Δ)))
        - x * ∑ k : Fin K, w k * Real.exp (-(x - μ * v k) ^ 2 / (2 * Δ))) / Δ) x := by
  have hterm : ∀ k ∈ (Finset.univ : Finset (Fin K)),
      HasDerivAt (fun y : ℝ => w k * Real.exp (-(y - μ * v k) ^ 2 / (2 * Δ)))
        (w k * (Real.exp (-(x - μ * v k) ^ 2 / (2 * Δ)) * (-(x - μ * v k) / Δ))) x := by
    intro k _
    have h0 : HasDerivAt (fun y : ℝ => y - μ * v k) 1 x := (hasDerivAt_id x).sub_const _
    have hbase : HasDerivAt (fun y : ℝ => (y - μ * v k) ^ 2) (2 * (x - μ * v k)) x := by
      first
        | simpa using h0.fun_pow 2
        | simpa using h0.pow 2
    have h2 := hbase.const_mul (-1 / (2 * Δ))
    have hfun : (fun y : ℝ => (-1 / (2 * Δ)) * (y - μ * v k) ^ 2)
        = fun y : ℝ => -(y - μ * v k) ^ 2 / (2 * Δ) := by
      funext y; ring
    have hder : (-1 / (2 * Δ)) * (2 * (x - μ * v k)) = -(x - μ * v k) / Δ := by
      first
        | (field_simp; ring)
        | field_simp
        | ring
    rw [hfun, hder] at h2
    exact (h2.exp).const_mul (w k)
  have hsum := HasDerivAt.fun_sum hterm
  have hval : (∑ k : Fin K, w k * (Real.exp (-(x - μ * v k) ^ 2 / (2 * Δ))
        * (-(x - μ * v k) / Δ)))
      = (μ * (∑ k : Fin K, w k * v k * Real.exp (-(x - μ * v k) ^ 2 / (2 * Δ)))
        - x * ∑ k : Fin K, w k * Real.exp (-(x - μ * v k) ^ 2 / (2 * Δ))) / Δ := by
    first
      | (simp only [Finset.mul_sum, Finset.sum_div, ← Finset.sum_sub_distrib];
         refine Finset.sum_congr rfl fun k _ => ?_; field_simp; ring)
      | (simp only [Finset.mul_sum, Finset.sum_div, ← Finset.sum_sub_distrib];
         refine Finset.sum_congr rfl fun k _ => ?_; field_simp)
      | (rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_sub_distrib, Finset.sum_div];
         refine Finset.sum_congr rfl fun k _ => ?_; field_simp; ring)
  rw [hval] at hsum
  exact hsum

/-- The other half of the discrete Tweedie check: the log-derivative really is
the combination eq:l-score-identity writes.  With `N = Σ_k w_k v_k e_k`,
`D = Σ_k w_k e_k` and `E[a|x] = N/D`, the Tweedie expression `(μ E[a|x] - x)/Δ`
equals `p'(x)/p(x) = ((μ N - x D)/Δ)/D`, the derivative supplied by the previous
lemma divided by `p(x)`. -/
theorem ch07_l_score_identity_discrete_log {N D Δ : ℝ} (hΔ : Δ ≠ 0) (hD : D ≠ 0) (μ x : ℝ) :
    (μ * (N / D) - x) / Δ = ((μ * N - x * D) / Δ) / D := by
  first
    | (field_simp; ring)
    | field_simp

/-! ### eq:l-state-kurtosis (lines 71-76)

`κ₄(a_k) = 3(1-α²)² Σ_{m=0}^{k-1} α^{4m} = 3 (1-α²)/(1+α²) (1-α^{4k})`.

Granted the two cumulant rules (`κ₄` adds over independent summands and is
homogeneous of degree 4), `κ₄(a_k) = Σ_{m<k} (α^m)^4 κ₄(ε)` with
`κ₄(ε) = 3(1-α²)²`; both steps are the lemmas below. -/

/-- A centred Laplace variable of variance `v = 2b²` has `κ₄ = E X⁴ - 3(E X²)² =
24b⁴ - 3(2b²)² = 3v²`, i.e. standardised fourth cumulant 3 (line 61). -/
theorem ch07_laplace_kurtosis (b : ℝ) : 24 * b ^ 4 - 3 * (2 * b ^ 2) ^ 2 = 3 * (2 * b ^ 2) ^ 2 := by
  ring

/-- Cumulant additivity/homogeneity step: `Σ_{m<k} (α^m)^4 c = c Σ_{m<k} (α^4)^m`
(the reindexing `m = k-j` of the text). -/
theorem ch07_l_state_kurtosis_sum (α c : ℝ) (k : ℕ) :
    ∑ m ∈ Finset.range k, (α ^ m) ^ 4 * c = c * ∑ m ∈ Finset.range k, (α ^ 4) ^ m := by
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl ?_
  intro m _
  rw [← pow_mul, ← pow_mul, Nat.mul_comm]
  ring

/-- eq:l-state-kurtosis: the closed form of the geometric sum. -/
theorem ch07_l_state_kurtosis (α : ℝ) (k : ℕ) (h : α ^ 2 ≠ 1) :
    3 * (1 - α ^ 2) ^ 2 * ∑ m ∈ Finset.range k, (α ^ 4) ^ m
      = 3 * ((1 - α ^ 2) / (1 + α ^ 2)) * (1 - α ^ (4 * k)) := by
  have h1 : (1 : ℝ) + α ^ 2 ≠ 0 := by positivity
  have h2 : (1 : ℝ) - α ^ 2 ≠ 0 := sub_ne_zero.mpr (Ne.symm h)
  have h4 : (α ^ 4 : ℝ) ≠ 1 := by
    intro hc
    apply h2
    have : (α ^ 2 - 1) * (α ^ 2 + 1) = 0 := by nlinarith [hc]
    rcases mul_eq_zero.mp this with h' | h'
    · linarith
    · nlinarith [sq_nonneg α]
  rw [geom_sum_eq h4]
  have hpow : (α ^ 4 : ℝ) ^ k = α ^ (4 * k) := by rw [← pow_mul]
  rw [hpow]
  have hd : (α ^ 4 : ℝ) - 1 = (α ^ 2 - 1) * (α ^ 2 + 1) := by ring
  rw [hd]
  field_simp
  ring

/-- The numerical value quoted on line 80: at `α = 0.85` the bulk excess kurtosis
is `3·0.2775/1.7225 ≈ 0.48`. -/
theorem ch07_l_state_kurtosis_num :
    |3 * ((1 - (0.85 : ℝ) ^ 2) / (1 + (0.85 : ℝ) ^ 2)) - 0.48| < 0.005 := by
  rw [abs_lt]
  constructor <;> norm_num

/-! ### eq:l-noised-kurtosis (lines 90-93)

`κ₄(x_{t,k}) = e^{-4t} κ₄(a_k)`, from `x_{t,k} = e^{-t}a_k + √Δ_t z_k` with
`κ₄(z) = 0` and degree-4 homogeneity. -/

theorem ch07_l_noised_kurtosis (t κa : ℝ) :
    (Real.exp (-t)) ^ 4 * κa + (Real.sqrt (ch07_Delta t)) ^ 4 * 0 = Real.exp (-4 * t) * κa := by
  have h : (Real.exp (-t)) ^ 4 = Real.exp (-4 * t) := by
    have hx : (-4 : ℝ) * t = -t + (-t + (-t + -t)) := by ring
    rw [hx, Real.exp_add, Real.exp_add, Real.exp_add]
    ring
  rw [h]
  ring

/-! ## Section 7.3 -- chain messages

### eq:l-markov-score (lines 128-135)

`E[a_k|x] = ∫ a_k α_k ψ_ob β_k / ∫ α_k ψ_ob β_k`.  Checked on a discrete chain
of `L = 3` sites over a 2-letter alphabet, where the forward/backward messages
are finite sums: both the numerator and the denominator of the message formula
equal the corresponding sum against the full joint. -/

theorem ch07_l_markov_score_num (val p0 ob0 ob1 ob2 : Fin 2 → ℝ) (M : Fin 2 → Fin 2 → ℝ) :
    (∑ a1 : Fin 2, val a1 * ((∑ a0 : Fin 2, p0 a0 * ob0 a0 * M a1 a0) * ob1 a1
        * ∑ a2 : Fin 2, M a2 a1 * ob2 a2))
      = ∑ a0 : Fin 2, ∑ a1 : Fin 2, ∑ a2 : Fin 2,
          val a1 * (p0 a0 * ob0 a0 * M a1 a0 * ob1 a1 * M a2 a1 * ob2 a2) := by
  simp only [Fin.sum_univ_two]
  ring

theorem ch07_l_markov_score_den (p0 ob0 ob1 ob2 : Fin 2 → ℝ) (M : Fin 2 → Fin 2 → ℝ) :
    (∑ a1 : Fin 2, ((∑ a0 : Fin 2, p0 a0 * ob0 a0 * M a1 a0) * ob1 a1
        * ∑ a2 : Fin 2, M a2 a1 * ob2 a2))
      = ∑ a0 : Fin 2, ∑ a1 : Fin 2, ∑ a2 : Fin 2,
          (p0 a0 * ob0 a0 * M a1 a0 * ob1 a1 * M a2 a1 * ob2 a2) := by
  simp only [Fin.sum_univ_two]
  ring

/-- The same numerator identity on a chain of three sites over an *arbitrary*
finite alphabet, not just two letters: the message decomposition of
eq:l-markov-score is a Fubini rearrangement and nothing else. -/
theorem ch07_l_markov_score_num_gen {A : Type} [Fintype A] (val p0 ob0 ob1 ob2 : A → ℝ)
    (M : A → A → ℝ) :
    (∑ a1 : A, val a1 * ((∑ a0 : A, p0 a0 * ob0 a0 * M a1 a0) * ob1 a1
        * ∑ a2 : A, M a2 a1 * ob2 a2))
      = ∑ a0 : A, ∑ a1 : A, ∑ a2 : A,
          val a1 * (p0 a0 * ob0 a0 * M a1 a0 * ob1 a1 * M a2 a1 * ob2 a2) := by
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun a1 _ => ?_
  rw [show val a1 * ((∑ a0 : A, p0 a0 * ob0 a0 * M a1 a0) * ob1 a1
        * ∑ a2 : A, M a2 a1 * ob2 a2)
      = ((∑ a0 : A, p0 a0 * ob0 a0 * M a1 a0) * ∑ a2 : A, M a2 a1 * ob2 a2)
        * (val a1 * ob1 a1) by ring]
  rw [Fintype.sum_mul_sum, Finset.sum_mul]
  refine Finset.sum_congr rfl fun a0 _ => ?_
  rw [Finset.sum_mul]
  refine Finset.sum_congr rfl fun a2 _ => ?_
  ring

/-- The denominator of eq:l-markov-score over the same arbitrary finite
alphabet, obtained from the numerator lemma at `val ≡ 1`. -/
theorem ch07_l_markov_score_den_gen {A : Type} [Fintype A] (p0 ob0 ob1 ob2 : A → ℝ)
    (M : A → A → ℝ) :
    (∑ a1 : A, ((∑ a0 : A, p0 a0 * ob0 a0 * M a1 a0) * ob1 a1
        * ∑ a2 : A, M a2 a1 * ob2 a2))
      = ∑ a0 : A, ∑ a1 : A, ∑ a2 : A,
          p0 a0 * ob0 a0 * M a1 a0 * ob1 a1 * M a2 a1 * ob2 a2 := by
  have h := ch07_l_markov_score_num_gen (fun _ => (1 : ℝ)) p0 ob0 ob1 ob2 M
  simpa using h

/-- The message ratio is invariant under rescaling either message by a nonzero
constant (sum-product messages are defined only up to a positive constant). -/
theorem ch07_l_markov_score_scale_inv {c N : ℝ} (hc : c ≠ 0) (D : ℝ) :
    (c * N) / (c * D) = N / D := mul_div_mul_left N D hc

/-! ### Remark 7.3 (rmk:l-breakdown)

noname-1 (lines 143-152): the definitions `M(a'|a) = (1/2b)e^{-|a'-αa|/b}` and
`p(x'|a') = N(x'; μa', Δ_t)`. -/

/-- The Laplace transition kernel of the remark. -/
def ch07_M (α b : ℝ) (a a' : ℝ) : ℝ := ch07_laplacePdf b (a' - α * a)

/-- The OU observation likelihood of the remark. -/
def ch07_obs (μ Δ : ℝ) (a' x' : ℝ) : ℝ := ch07_gauss (μ * a') Δ x'

/-- Both factors are nonnegative densities. -/
theorem ch07_M_nonneg {b : ℝ} (hb : 0 < b) (α a a' : ℝ) : 0 ≤ ch07_M α b a a' := by
  unfold ch07_M ch07_laplacePdf
  positivity

/-! noname-2 (lines 154-166): the change of variables `r = a'-αa`,
`y = x'-μαa`, which turns the transition integral into a convolution. -/

theorem ch07_remark_change_of_var (b Δ μ α a x' : ℝ) :
    (∫ a' : ℝ, ch07_M α b a a' * ch07_obs μ Δ a' x')
      = (1 / (2 * b) * (1 / Real.sqrt (2 * π * Δ)))
        * ∫ r : ℝ, Real.exp (-|r| / b + -((x' - μ * (α * a)) - μ * r) ^ 2 / (2 * Δ)) := by
  have hfun : (fun a' : ℝ => ch07_M α b a a' * ch07_obs μ Δ a' x')
      = fun a' : ℝ => (fun r : ℝ => (1 / (2 * b) * (1 / Real.sqrt (2 * π * Δ)))
          * Real.exp (-|r| / b + -((x' - μ * (α * a)) - μ * r) ^ 2 / (2 * Δ))) (a' - α * a) := by
    funext a'
    simp only [ch07_M, ch07_obs, ch07_laplacePdf, ch07_gauss]
    rw [mul_mul_mul_comm, ← Real.exp_add]
    congr 2
    ring
  rw [hfun]
  rw [integral_sub_right_eq_self (fun r : ℝ => (1 / (2 * b) * (1 / Real.sqrt (2 * π * Δ)))
        * Real.exp (-|r| / b + -((x' - μ * (α * a)) - μ * r) ^ 2 / (2 * Δ))) (α * a)]
  rw [MeasureTheory.integral_const_mul]

/-! noname-3 (lines 169-188): the boxed closed form

`I(y) = e^{Δ/(2μ²b²)}/(2μb) [ e^{-y/(μb)} Φ(y/√Δ - √Δ/(μb))
                            + e^{y/(μb)} Φ(-y/√Δ - √Δ/(μb)) ]`,

obtained by splitting at the kink `r = 0` and completing the square on each
half-line.  The two completions are exact identities (below); together with the
normalisation and argument identities they give the boxed formula from the
standard Gaussian half-line mass. -/

/-- Completion of the square on `r > 0` (where `|r| = r`). -/
theorem ch07_remark_square_pos {b μ Δ : ℝ} (hb : b ≠ 0) (hμ : μ ≠ 0) (hΔ : Δ ≠ 0) (y r : ℝ) :
    -r / b + -(y - μ * r) ^ 2 / (2 * Δ)
      = (Δ / (2 * (μ * b) ^ 2) - y / (μ * b))
        + -(r - (y - Δ / (μ * b)) / μ) ^ 2 / (2 * (Δ / μ ^ 2)) := by
  field_simp
  ring

/-- Completion of the square on `r < 0` (where `|r| = -r`). -/
theorem ch07_remark_square_neg {b μ Δ : ℝ} (hb : b ≠ 0) (hμ : μ ≠ 0) (hΔ : Δ ≠ 0) (y r : ℝ) :
    r / b + -(y - μ * r) ^ 2 / (2 * Δ)
      = (Δ / (2 * (μ * b) ^ 2) + y / (μ * b))
        + -(r - (y + Δ / (μ * b)) / μ) ^ 2 / (2 * (Δ / μ ^ 2)) := by
  field_simp
  ring

/-- The shifted mean of the `r > 0` branch, divided by the branch standard
deviation `√(Δ/μ²)`, is exactly the first `Φ` argument of the boxed formula. -/
theorem ch07_remark_Phi_arg_pos {μ Δ : ℝ} (hμ : 0 < μ) (hΔ : 0 < Δ) (b y : ℝ) :
    ((y - Δ / (μ * b)) / μ) / Real.sqrt (Δ / μ ^ 2)
      = y / Real.sqrt Δ - Real.sqrt Δ / (μ * b) := by
  have hs : Real.sqrt (Δ / μ ^ 2) = Real.sqrt Δ / μ := by
    rw [show Δ / μ ^ 2 = Δ * (1 / μ) ^ 2 by field_simp]
    rw [Real.sqrt_mul hΔ.le, Real.sqrt_sq (by positivity)]
    field_simp
  have hsΔ : Real.sqrt Δ ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hΔ)
  have hsq : Real.sqrt Δ * Real.sqrt Δ = Δ := Real.mul_self_sqrt hΔ.le
  have hsq2 : Real.sqrt Δ ^ 2 = Δ := Real.sq_sqrt hΔ.le
  rw [hs]
  field_simp
  first
    | (rw [Real.sq_sqrt hΔ.le]; ring)
    | rw [Real.sq_sqrt hΔ.le]
    | nlinarith [hsq, hsq2]

/-- The shifted mean of the `r < 0` branch: minus (mean / branch standard
deviation) is exactly the second `Φ` argument of the boxed formula. -/
theorem ch07_remark_Phi_arg_neg {μ Δ : ℝ} (hμ : 0 < μ) (hΔ : 0 < Δ) (b y : ℝ) :
    (-((y + Δ / (μ * b)) / μ)) / Real.sqrt (Δ / μ ^ 2)
      = -(y / Real.sqrt Δ) - Real.sqrt Δ / (μ * b) := by
  have hs : Real.sqrt (Δ / μ ^ 2) = Real.sqrt Δ / μ := by
    rw [show Δ / μ ^ 2 = Δ * (1 / μ) ^ 2 by field_simp]
    rw [Real.sqrt_mul hΔ.le, Real.sqrt_sq (by positivity)]
    field_simp
  have hsΔ : Real.sqrt Δ ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hΔ)
  have hsq : Real.sqrt Δ * Real.sqrt Δ = Δ := Real.mul_self_sqrt hΔ.le
  have hsq2 : Real.sqrt Δ ^ 2 = Δ := Real.sq_sqrt hΔ.le
  rw [hs]
  field_simp
  first
    | (rw [Real.sq_sqrt hΔ.le]; ring)
    | rw [Real.sq_sqrt hΔ.le]
    | nlinarith [hsq, hsq2]

/-- The prefactor identity: the `1/(2b√(2πΔ))` in front of the convolution times
the Gaussian mass `√(2π(Δ/μ²))` of each branch is the boxed `1/(2μb)`. -/
theorem ch07_remark_prefactor {b μ Δ : ℝ} (hb : b ≠ 0) (hμ : 0 < μ) (hΔ : 0 < Δ) :
    (1 / (2 * b * Real.sqrt (2 * π * Δ))) * Real.sqrt (2 * π * (Δ / μ ^ 2))
      = 1 / (2 * μ * b) := by
  have h : Real.sqrt (2 * π * (Δ / μ ^ 2)) = Real.sqrt (2 * π * Δ) / μ := by
    rw [show 2 * π * (Δ / μ ^ 2) = (2 * π * Δ) * (1 / μ) ^ 2 by field_simp]
    rw [Real.sqrt_mul (by positivity), Real.sqrt_sq (by positivity)]
    field_simp
  have hpos : 0 < Real.sqrt (2 * π * Δ) := Real.sqrt_pos.mpr (by positivity)
  rw [h]
  field_simp

/-! noname-4 (lines 195-197) and noname-5 (lines 200-206): the *shape* of the
message after one update, `e^{ca}Φ(da+e)`, and the two truncated integrals that
the next update requires.  These are notations for the obstruction, not claims;
they are recorded as definitions. -/

/-- The post-update message shape `e^{ca} Φ(da+e)`. -/
def ch07_msg_shape (c d e : ℝ) (a : ℝ) : ℝ := Real.exp (c * a) * ch07_Phi (d * a + e)

/-- The two truncated integrals of line 202-205. -/
def ch07_trunc_lower (q l d e α a' : ℝ) : ℝ :=
  ∫ a in Set.Iic (α * a'), Real.exp (-q * a ^ 2 + l * a) * ch07_Phi (d * a + e)

def ch07_trunc_upper (q l d e α a' : ℝ) : ℝ :=
  ∫ a in Set.Ioi (α * a'), Real.exp (-q * a ^ 2 + l * a) * ch07_Phi (d * a + e)

/-! noname-6 (lines 215-219): the innovation coordinates `r_0 = a_0`,
`r_k = a_k - α a_{k-1}`; noname-7 (lines 221-226): `a_k = Σ_{j≤k} α^{k-j} r_j`,
i.e. `a = B r` with `B_{kj} = α^{k-j} 1_{k≥j}`. -/

/-- The chain reconstructed from its innovations: `A 0 = r 0`,
`A (k+1) = α A k + r (k+1)`. -/
def ch07_A (α : ℝ) (r : ℕ → ℝ) : ℕ → ℝ
  | 0 => r 0
  | (k + 1) => α * ch07_A α r k + r (k + 1)

/-- noname-7: `a_k = Σ_{j=0}^{k} α^{k-j} r_j`. -/
theorem ch07_innov_sum (α : ℝ) (r : ℕ → ℝ) :
    ∀ k : ℕ, ch07_A α r k = ∑ j ∈ Finset.range (k + 1), α ^ (k - j) * r j := by
  intro k
  induction k with
  | zero => simp [ch07_A]
  | succ n ih =>
      rw [ch07_A, ih, Finset.sum_range_succ (fun j => α ^ (n + 1 - j) * r j)]
      rw [Finset.mul_sum]
      have hterm : ∀ j ∈ Finset.range (n + 1), α * (α ^ (n - j) * r j) = α ^ (n + 1 - j) * r j := by
        intro j hj
        have hj' : j ≤ n := Nat.lt_succ_iff.mp (Finset.mem_range.mp hj)
        have : n + 1 - j = (n - j) + 1 := by omega
        rw [this, pow_succ]
        ring
      rw [Finset.sum_congr rfl hterm]
      simp

/-- The matrix `B_{kj} = α^{k-j} 1_{k≥j}` of noname-7. -/
def ch07_B (α : ℝ) (L : ℕ) : Matrix (Fin L) (Fin L) ℝ :=
  fun k j => if (j : ℕ) ≤ (k : ℕ) then α ^ ((k : ℕ) - (j : ℕ)) else 0

/-- noname-8 (lines 228-230): the quadratic form `rᵀ BᵀB r` that the likelihood
contributes in innovation coordinates. -/
def ch07_quadform (α : ℝ) (L : ℕ) (r : Fin L → ℝ) : ℝ :=
  r ⬝ᵥ (Matrix.mulVec ((ch07_B α L)ᵀ * ch07_B α L) r)

/-! noname-9 (lines 232-238): `(BᵀB)_{ij} = α^{|i-j|}(1-α^{2(L-max(i,j))})/(1-α²)`.
Checked for `L = 2, 3` at every index pair with symbolic `α`. -/

/-- The right-hand side of noname-9. -/
def ch07_BtB_rhs (α : ℝ) (L i j : ℕ) : ℝ :=
  α ^ (max i j - min i j) * (1 - α ^ (2 * (L - max i j))) / (1 - α ^ 2)

theorem ch07_BtB_two (α : ℝ) (h : α ^ 2 ≠ 1) (i j : Fin 2) :
    ((ch07_B α 2)ᵀ * ch07_B α 2) i j = ch07_BtB_rhs α 2 (i : ℕ) (j : ℕ) := by
  have h2 : (1 : ℝ) - α ^ 2 ≠ 0 := sub_ne_zero.mpr (Ne.symm h)
  fin_cases i <;> fin_cases j <;>
    simp [Matrix.mul_apply, ch07_B, ch07_BtB_rhs, Fin.sum_univ_two, Matrix.transpose_apply] <;>
    field_simp <;> ring

theorem ch07_BtB_three (α : ℝ) (h : α ^ 2 ≠ 1) (i j : Fin 3) :
    ((ch07_B α 3)ᵀ * ch07_B α 3) i j = ch07_BtB_rhs α 3 (i : ℕ) (j : ℕ) := by
  have h2 : (1 : ℝ) - α ^ 2 ≠ 0 := sub_ne_zero.mpr (Ne.symm h)
  fin_cases i <;> fin_cases j <;>
    simp [Matrix.mul_apply, ch07_B, ch07_BtB_rhs, Fin.sum_univ_three, Matrix.transpose_apply] <;>
    field_simp <;> ring

theorem ch07_BtB_four (α : ℝ) (h : α ^ 2 ≠ 1) (i j : Fin 4) :
    ((ch07_B α 4)ᵀ * ch07_B α 4) i j = ch07_BtB_rhs α 4 (i : ℕ) (j : ℕ) := by
  have h2 : (1 : ℝ) - α ^ 2 ≠ 0 := sub_ne_zero.mpr (Ne.symm h)
  fin_cases i <;> fin_cases j <;>
    simp [Matrix.mul_apply, ch07_B, ch07_BtB_rhs, Fin.sum_univ_four, Matrix.transpose_apply] <;>
    field_simp <;> ring

/-! ## Section 7.4 -- grid belief propagation

### eq:l-grid (lines 252-256): `g_i = -A + i h`, `h = 2A/(N-1)`. -/

def ch07_h (A : ℝ) (N : ℕ) : ℝ := 2 * A / ((N : ℝ) - 1)

def ch07_g (A : ℝ) (N : ℕ) (i : ℕ) : ℝ := -A + (i : ℝ) * ch07_h A N

/-- The grid covers `[-A, A]`: the first node is `-A` and the last is `A`. -/
theorem ch07_l_grid_endpoints (A : ℝ) {N : ℕ} (hN : 2 ≤ N) :
    ch07_g A N 0 = -A ∧ ch07_g A N (N - 1) = A := by
  have hcast : ((N - 1 : ℕ) : ℝ) = (N : ℝ) - 1 := by
    have : (1 : ℕ) ≤ N := by omega
    push_cast [Nat.cast_sub this]
    ring
  have hne : (N : ℝ) - 1 ≠ 0 := by
    have : (2 : ℝ) ≤ (N : ℝ) := by exact_mod_cast hN
    linarith
  constructor
  · simp [ch07_g]
  · unfold ch07_g ch07_h
    rw [hcast]
    field_simp
    ring

/-! ### eq:l-trap-weights (lines 260-267): `w_i = h/2` at the endpoints, `h`
otherwise. -/

def ch07_w (A : ℝ) (N : ℕ) (i : ℕ) : ℝ :=
  if i = 0 ∨ i = N - 1 then ch07_h A N / 2 else ch07_h A N

/-- The weights sum to the length of the interval, `2A` (checked at `N = 3,4,5`). -/
theorem ch07_l_trap_weights_sum3 (A : ℝ) : ∑ i ∈ Finset.range 3, ch07_w A 3 i = 2 * A := by
  simp [Finset.sum_range_succ, ch07_w, ch07_h]
  ring

theorem ch07_l_trap_weights_sum4 (A : ℝ) : ∑ i ∈ Finset.range 4, ch07_w A 4 i = 2 * A := by
  simp [Finset.sum_range_succ, ch07_w, ch07_h]
  ring

theorem ch07_l_trap_weights_sum5 (A : ℝ) : ∑ i ∈ Finset.range 5, ch07_w A 5 i = 2 * A := by
  simp [Finset.sum_range_succ, ch07_w, ch07_h]
  ring

/-! ### eq:l-grid-operators (lines 273-280): the transition matrix and the
likelihood diagonal. -/

def ch07_K (α b : ℝ) (g : ℕ → ℝ) (i j : ℕ) : ℝ :=
  (1 / (2 * b)) * Real.exp (-|g i - α * g j| / b)

def ch07_ell (μ Δ xk : ℝ) (g : ℕ → ℝ) (i : ℕ) : ℝ := ch07_gauss (μ * g i) Δ xk

/-- `K_{ij}` is the Laplace transition density evaluated at the grid pair. -/
theorem ch07_K_eq_M (α b : ℝ) (g : ℕ → ℝ) (i j : ℕ) :
    ch07_K α b g i j = ch07_M α b (g j) (g i) := by
  unfold ch07_K ch07_M ch07_laplacePdf
  ring_nf

theorem ch07_K_nonneg {b : ℝ} (hb : 0 < b) (α : ℝ) (g : ℕ → ℝ) (i j : ℕ) :
    0 ≤ ch07_K α b g i j := by
  unfold ch07_K
  positivity

/-! ### eq:l-grid-fwd (lines 284-288) and eq:l-grid-bwd (lines 290-292)

The two sweeps as matrix-vector products.  The content is that the forward
sweep contracts `K` over its *second* index and the backward sweep over its
*first* one -- i.e. that the transpose in eq:l-grid-bwd is the right one. -/

theorem ch07_l_grid_fwd {N : ℕ} (K : Matrix (Fin N) (Fin N) ℝ) (w f ell : Fin N → ℝ) (i : Fin N) :
    ell i * (Matrix.mulVec K (fun j => w j * f j)) i
      = ell i * ∑ j : Fin N, w j * K i j * f j := by
  simp only [Matrix.mulVec, dotProduct]
  congr 1
  refine Finset.sum_congr rfl ?_
  intro j _
  ring

theorem ch07_l_grid_bwd {N : ℕ} (K : Matrix (Fin N) (Fin N) ℝ) (w ell b : Fin N → ℝ) (i : Fin N) :
    (Matrix.mulVec Kᵀ (fun j => w j * (ell j * b j))) i
      = ∑ j : Fin N, w j * K j i * (ell j * b j) := by
  simp only [Matrix.mulVec, dotProduct, Matrix.transpose_apply]
  refine Finset.sum_congr rfl ?_
  intro j _
  ring

/-! ### eq:l-grid-score (lines 296-304): belief, posterior mean, score. -/

def ch07_bel {N : ℕ} (f b : Fin N → ℝ) : Fin N → ℝ := fun i => f i * b i

def ch07_mhat {N : ℕ} (w g bel : Fin N → ℝ) : ℝ :=
  (∑ i : Fin N, w i * g i * bel i) / ∑ i : Fin N, w i * bel i

def ch07_shat (μ Δ mhat xk : ℝ) : ℝ := (μ * mhat - xk) / Δ

/-- The reported mean (hence the score) is invariant under renormalisation of the
belief, which is what makes the renormalisation of Section 7.4.2 free. -/
theorem ch07_l_grid_score_scale_inv {N : ℕ} (w g bel : Fin N → ℝ) {c : ℝ} (hc : c ≠ 0) :
    ch07_mhat w g (fun i => c * bel i) = ch07_mhat w g bel := by
  unfold ch07_mhat
  have h1 : ∑ i : Fin N, w i * g i * (c * bel i) = c * ∑ i : Fin N, w i * g i * bel i := by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl ?_
    intro i _
    ring
  have h2 : ∑ i : Fin N, w i * (c * bel i) = c * ∑ i : Fin N, w i * bel i := by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl ?_
    intro i _
    ring
  rw [h1, h2, mul_div_mul_left _ _ hc]

/-! ### eq:l-resolution (lines 334-337): `√(2t) ≳ 3h = 6A/(N-1)`. -/

theorem ch07_l_resolution (A : ℝ) (N : ℕ) : 3 * ch07_h A N = 6 * A / ((N : ℝ) - 1) := by
  unfold ch07_h
  ring

/-- The working configuration `N_g = 401, A = 8` has `3h = 0.12`. -/
theorem ch07_l_resolution_401 : 3 * ch07_h 8 401 = 0.12 := by
  unfold ch07_h
  norm_num

/-- At `N_g = 101, A = 8` the gate demands `t ≥ 0.1152`, the "t ≳ 0.115" of line 360. -/
theorem ch07_l_resolution_101 : (3 * ch07_h 8 101) ^ 2 / 2 = 0.1152 := by
  unfold ch07_h
  norm_num

/-- `√(2·0.05) ≈ 0.32`, the margin quoted on line 342. -/
theorem ch07_l_resolution_margin : 0.316 < Real.sqrt (2 * 0.05) ∧ Real.sqrt (2 * 0.05) < 0.317 := by
  constructor
  · rw [show (2 : ℝ) * 0.05 = 0.1 by norm_num, Real.lt_sqrt (by norm_num)]
    norm_num
  · rw [show (2 : ℝ) * 0.05 = 0.1 by norm_num, Real.sqrt_lt' (by norm_num)]
    norm_num

/-- Bookkeeping on the same sentence.  Table `tab:l-closure` reports `t = 0.02`
as its smallest diffusion time, where `√(2t) = 0.2`; the `0.32` the text quotes
"at the smallest reported `t`" is `√(2·0.05)`.  The gate itself still clears at
`t = 0.02` (`3h = 0.12 < 0.2`), so only the quoted number, not the conclusion,
is affected. -/
theorem ch07_l_resolution_smallest_reported_t :
    3 * ch07_h 8 401 < Real.sqrt (2 * 0.02)
      ∧ Real.sqrt (2 * 0.02) < 0.201
      ∧ Real.sqrt (2 * 0.02) < 0.316 := by
  have h1 : Real.sqrt (2 * 0.02) < 0.201 := by
    rw [show (2 : ℝ) * 0.02 = 0.04 by norm_num, Real.sqrt_lt' (by norm_num)]
    norm_num
  have h2 : (0.199 : ℝ) < Real.sqrt (2 * 0.02) := by
    rw [show (2 : ℝ) * 0.02 = 0.04 by norm_num, Real.lt_sqrt (by norm_num)]
    norm_num
  refine ⟨?_, h1, by linarith⟩
  rw [ch07_l_resolution_401]
  linarith

/-! ### eq:bp-master-error (lines 387-393)

`ŝ - s_ref = (e^{-t}/Δ_t)(m̂ - m_ref)` and `e^{-t}/Δ_t ~ 1/(2t)` as `t ↓ 0`. -/

theorem ch07_bp_master_error (t Δ mh mr xk : ℝ) (hΔ : Δ ≠ 0) :
    ch07_shat (ch07_mu t) Δ mh xk - ch07_shat (ch07_mu t) Δ mr xk
      = (Real.exp (-t) / Δ) * (mh - mr) := by
  unfold ch07_shat ch07_mu
  field_simp
  ring

/-- `e^{-t}/Δ_t = 1/(e^t - e^{-t}) = 1/(2 sinh t)`, whose `t ↓ 0` equivalent is
`1/(2t)`. -/
theorem ch07_bp_master_error_prefactor {t : ℝ} (ht : t ≠ 0) :
    Real.exp (-t) / ch07_Delta t = 1 / (2 * Real.sinh t) := by
  unfold ch07_Delta
  have hexp : Real.exp (-2 * t) = Real.exp (-t) * Real.exp (-t) := by
    rw [← Real.exp_add]; ring_nf
  have hone : Real.exp (-t) * Real.exp t = 1 := by
    rw [← Real.exp_add]; simp
  have hpos : Real.exp (-t) ≠ 0 := ne_of_gt (Real.exp_pos _)
  have hne : Real.exp t - Real.exp (-t) ≠ 0 := by
    intro hcon
    have h1 : Real.exp t = Real.exp (-t) := by linarith
    rw [Real.exp_eq_exp] at h1
    exact ht (by linarith)
  have hkey : (1 : ℝ) - Real.exp (-2 * t)
      = Real.exp (-t) * (Real.exp t - Real.exp (-t)) := by
    rw [hexp, mul_sub, hone]
  have hsinh : 2 * Real.sinh t = Real.exp t - Real.exp (-t) := by
    rw [Real.sinh_eq]; ring
  rw [hkey, hsinh]
  field_simp

/-! ## Section 7.5 -- the Gaussian-message approximation

### eq:l-lmmse (lines 414-422) -/

theorem ch07_l_lmmse {n : ℕ} (t : ℝ) (S0 Cax Cxx : Matrix (Fin n) (Fin n) ℝ)
    (hCax : Cax = (Real.exp (-t)) • S0)
    (hCxx : Cxx = (Real.exp (-2 * t)) • S0 + (ch07_Delta t) • (1 : Matrix (Fin n) (Fin n) ℝ))
    (x : Fin n → ℝ) :
    Matrix.mulVec (Cax * Cxx⁻¹) x
      = Matrix.mulVec (((Real.exp (-t)) • S0)
          * ((Real.exp (-2 * t)) • S0 + (ch07_Delta t) • (1 : Matrix (Fin n) (Fin n) ℝ))⁻¹) x := by
  rw [hCax, hCxx]

/-- The normal equations behind eq:l-lmmse: the gain `G = Cov(a,x)Cov(x)^{-1}`
is exactly the matrix solving `G Cov(x) = Cov(a,x)`, which is what makes it the
linear minimum-mean-square-error gain. -/
theorem ch07_l_lmmse_normal_eq {n : ℕ} (Cax Cxx : Matrix (Fin n) (Fin n) ℝ)
    (h : IsUnit Cxx.det) :
    (Cax * Cxx⁻¹) * Cxx = Cax := by
  rw [Matrix.mul_assoc, Matrix.nonsing_inv_mul _ h, Matrix.mul_one]

/-- eq:l-lmmse at the chapter's own normalisation `Σ₀ = I` (unit marginal
variance): the gain matrix collapses to `e^{-t} • I`, because
`e^{-2t} + Δ_t = 1`. -/
theorem ch07_l_lmmse_identity {n : ℕ} (t : ℝ) :
    ((Real.exp (-t)) • (1 : Matrix (Fin n) (Fin n) ℝ))
        * ((Real.exp (-2 * t)) • (1 : Matrix (Fin n) (Fin n) ℝ)
            + (ch07_Delta t) • (1 : Matrix (Fin n) (Fin n) ℝ))⁻¹
      = (Real.exp (-t)) • (1 : Matrix (Fin n) (Fin n) ℝ) := by
  have h : (Real.exp (-2 * t)) • (1 : Matrix (Fin n) (Fin n) ℝ)
      + (ch07_Delta t) • (1 : Matrix (Fin n) (Fin n) ℝ) = 1 := by
    rw [← add_smul]
    unfold ch07_Delta
    rw [show Real.exp (-2 * t) + (1 - Real.exp (-2 * t)) = 1 by ring, one_smul]
  rw [h]
  simp

/-- Scalar instance of eq:l-lmmse with the chapter's unit marginal variance: the
LMMSE gain is exactly `e^{-t}`, which combined with eq:l-score-identity returns
the score `-x`. -/
theorem ch07_l_lmmse_scalar (t x : ℝ) :
    Real.exp (-t) * 1 / (Real.exp (-2 * t) * 1 + ch07_Delta t) * x = Real.exp (-t) * x := by
  unfold ch07_Delta
  have h : Real.exp (-2 * t) * 1 + (1 - Real.exp (-2 * t)) = 1 := by ring
  rw [h]
  ring

/-! ### noname-10 (lines 444-446) and noname-11 (lines 449-455): the Gaussian
incoming message `N(a; m, v)` and the propagation law `a_i = α a_{i-1} + ε_i`. -/

def ch07_fwd_msg (m v : ℝ) : ℝ → ℝ := fun a => ch07_gauss m v a

def ch07_prop (α : ℝ) (a e : ℝ) : ℝ := α * a + e

/-! ### eq:lmmse-fwd-map (lines 457-462): `(m,v) ↦ (αm, α²v + q)`. -/

/-- The forward moment map, proved as the Gaussian convolution it abbreviates:
propagating `N(·; m, v)` through `a_i = α a_{i-1} + ε_i` (innovation variance
`q`) gives a Gaussian in `a_i` with mean `αm` and variance `α²v + q`. -/
theorem ch07_lmmse_fwd_map {q v : ℝ} (hq : 0 < q) (hv : 0 < v) (α m a : ℝ) :
    (∫ u : ℝ, Real.exp (-(a - α * u) ^ 2 / (2 * q)) * Real.exp (-(u - m) ^ 2 / (2 * v)))
      = Real.sqrt (π / ((α ^ 2 / q + 1 / v) / 2))
        * Real.exp (-(a - α * m) ^ 2 / (2 * (α ^ 2 * v + q))) :=
  ch07_gauss_conv hq hv α a m

/-! ### eq:lmmse-bwd-map (lines 472-478): `(m,v) ↦ (m/α, (v+q)/α²)`. -/

/-- The backward transition integral, as a function of the *preceding* state `u`:
`∫ N(a; αu, q) N(a; m, v) da ∝ exp(-(αu-m)²/(2(q+v)))`. -/
theorem ch07_lmmse_bwd_integral {q v : ℝ} (hq : 0 < q) (hv : 0 < v) (α m u : ℝ) :
    (∫ a : ℝ, Real.exp (-(m - a) ^ 2 / (2 * v)) * Real.exp (-(a - α * u) ^ 2 / (2 * q)))
      = Real.sqrt (π / ((1 / v + 1 / q) / 2))
        * Real.exp (-(m - α * u) ^ 2 / (2 * (v + q))) := by
  have h := ch07_gauss_conv hv hq 1 m (α * u)
  simp only [one_pow, one_mul] at h
  rw [h, add_comm q v]

/-- eq:lmmse-bwd-map: that kernel, read as a density in the preceding state, has
mean `m/α` and variance `(v+q)/α²`. -/
theorem ch07_lmmse_bwd_map {α : ℝ} (hα : α ≠ 0) {q v : ℝ} (hqv : v + q ≠ 0) (m u : ℝ) :
    -(m - α * u) ^ 2 / (2 * (v + q)) = -(u - m / α) ^ 2 / (2 * ((v + q) / α ^ 2)) := by
  field_simp
  ring

/-! ### noname-12 (lines 489-493): `E[a|x] = Cov(a,x)Cov(x)^{-1}x` for jointly
Gaussian zero-mean `(a,x)`.  Checked in the scalar case by completing the square
in the joint density: with `Cov = [[s, c],[c, V]]`, the exponent of the joint is
`-(V a² - 2cax + s x²)/(2D)` and its `a`-part completes to a Gaussian with mean
`(c/V)x = Cov(a,x)Cov(x)^{-1}x`. -/

theorem ch07_gauss_cond_mean {s c V : ℝ} (hV : V ≠ 0) (a x : ℝ) :
    V * a ^ 2 - 2 * c * a * x + s * x ^ 2
      = V * (a - (c / V) * x) ^ 2 + (s - c ^ 2 / V) * x ^ 2 := by
  field_simp
  ring

/-! ### noname-13 (lines 495-498) and eq:lmmse-covs (lines 500-505):
`x = e^{-t}a + √Δ_t z` with `z ⊥ a`, hence `Cov(a,x) = e^{-t}Σ₀` and
`Cov(x) = e^{-2t}Σ₀ + Δ_t I`. -/

theorem ch07_lmmse_covs {n : ℕ} {t : ℝ} (ht : 0 ≤ t) (S0 : Matrix (Fin n) (Fin n) ℝ) :
    ((Real.exp (-t)) • S0 + (Real.sqrt (ch07_Delta t)) • (0 : Matrix (Fin n) (Fin n) ℝ)
        = (Real.exp (-t)) • S0)
    ∧ ((Real.exp (-t)) ^ 2 • S0 + (Real.sqrt (ch07_Delta t)) ^ 2 • (1 : Matrix (Fin n) (Fin n) ℝ)
        = (Real.exp (-2 * t)) • S0 + (ch07_Delta t) • (1 : Matrix (Fin n) (Fin n) ℝ)) := by
  have hΔ : 0 ≤ ch07_Delta t := by
    unfold ch07_Delta
    have : Real.exp (-2 * t) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith)
    linarith
  constructor
  · simp
  · have h1 : (Real.exp (-t)) ^ 2 = Real.exp (-2 * t) := by
      rw [sq, ← Real.exp_add]; ring_nf
    have h2 : (Real.sqrt (ch07_Delta t)) ^ 2 = ch07_Delta t := Real.sq_sqrt hΔ
    rw [h1, h2]

/-! ### eq:l-excess (lines 542-549): the excess-risk decomposition. -/

/-- Pythagoras in `ℝ^n`: if the cross term vanishes the risk splits exactly. -/
theorem ch07_l_excess {n : ℕ} (a m ah : Fin n → ℝ)
    (h : ∑ i : Fin n, (a i - m i) * (m i - ah i) = 0) :
    (∑ i : Fin n, (a i - ah i) ^ 2)
      = (∑ i : Fin n, (a i - m i) ^ 2) + ∑ i : Fin n, (m i - ah i) ^ 2 := by
  have hpt : ∀ i : Fin n, (a i - ah i) ^ 2
      = (a i - m i) ^ 2 + 2 * ((a i - m i) * (m i - ah i)) + (m i - ah i) ^ 2 := by
    intro i; ring
  simp_rw [hpt]
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, h]
  ring

/-- The same with an expectation over a finite sample space: the cross term is
killed by the tower property (`E[a - m(x) | x] = 0`), which is exactly the
hypothesis `h`. -/
theorem ch07_l_excess_expect {n : ℕ} {Ω : Type} [Fintype Ω] (p : Ω → ℝ)
    (a m ah : Ω → Fin n → ℝ)
    (h : ∑ w : Ω, p w * ∑ i : Fin n, (a w i - m w i) * (m w i - ah w i) = 0) :
    (∑ w : Ω, p w * ∑ i : Fin n, (a w i - ah w i) ^ 2)
      = (∑ w : Ω, p w * ∑ i : Fin n, (a w i - m w i) ^ 2)
        + ∑ w : Ω, p w * ∑ i : Fin n, (m w i - ah w i) ^ 2 := by
  have hpt : ∀ w : Ω, p w * ∑ i : Fin n, (a w i - ah w i) ^ 2
      = p w * (∑ i : Fin n, (a w i - m w i) ^ 2)
        + 2 * (p w * ∑ i : Fin n, (a w i - m w i) * (m w i - ah w i))
        + p w * ∑ i : Fin n, (m w i - ah w i) ^ 2 := by
    intro w
    have := ch07_l_excess (n := n) (a w) (m w) (ah w)
    have hexp : ∀ i : Fin n, (a w i - ah w i) ^ 2
        = (a w i - m w i) ^ 2 + 2 * ((a w i - m w i) * (m w i - ah w i)) + (m w i - ah w i) ^ 2 := by
      intro i; ring
    simp_rw [hexp]
    rw [Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum]
    ring
  simp_rw [hpt]
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, h]
  ring

/-- eq:l-excess with the cross term *derived* rather than assumed.  On a finite
sample space, writing `p x y` for the joint mass of (observation `x`, latent
`y`), the hypothesis `hm` is exactly `E[a - m(x) | x] = 0` componentwise -- the
tower property the thesis proof invokes -- and `ah` is any estimator depending
on `x` alone.  The conclusion is eq:l-excess. -/
theorem ch07_l_excess_derived {n : ℕ} {X Y : Type} [Fintype X] [Fintype Y]
    (p : X → Y → ℝ) (a : Y → Fin n → ℝ) (ah m : X → Fin n → ℝ)
    (hm : ∀ x : X, ∀ i : Fin n, ∑ y : Y, p x y * (a y i - m x i) = 0) :
    (∑ x : X, ∑ y : Y, p x y * ∑ i : Fin n, (a y i - ah x i) ^ 2)
      = (∑ x : X, ∑ y : Y, p x y * ∑ i : Fin n, (a y i - m x i) ^ 2)
        + ∑ x : X, ∑ y : Y, p x y * ∑ i : Fin n, (m x i - ah x i) ^ 2 := by
  have key : ∀ x : X,
      ∑ y : Y, p x y * ∑ i : Fin n, (a y i - m x i) * (m x i - ah x i) = 0 := by
    intro x
    calc ∑ y : Y, p x y * ∑ i : Fin n, (a y i - m x i) * (m x i - ah x i)
        = ∑ y : Y, ∑ i : Fin n, p x y * ((a y i - m x i) * (m x i - ah x i)) := by
          refine Finset.sum_congr rfl fun y _ => ?_
          rw [Finset.mul_sum]
      _ = ∑ i : Fin n, ∑ y : Y, p x y * ((a y i - m x i) * (m x i - ah x i)) :=
          Finset.sum_comm
      _ = ∑ i : Fin n, (m x i - ah x i) * ∑ y : Y, p x y * (a y i - m x i) := by
          refine Finset.sum_congr rfl fun i _ => ?_
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl fun y _ => ?_
          ring
      _ = 0 := by
          refine Finset.sum_eq_zero fun i _ => ?_
          rw [hm x i, mul_zero]
  have step : ∀ x : X,
      (∑ y : Y, p x y * ∑ i : Fin n, (a y i - ah x i) ^ 2)
        = (∑ y : Y, p x y * ∑ i : Fin n, (a y i - m x i) ^ 2)
          + ∑ y : Y, p x y * ∑ i : Fin n, (m x i - ah x i) ^ 2 := by
    intro x
    have expand : ∀ y : Y, p x y * ∑ i : Fin n, (a y i - ah x i) ^ 2
        = p x y * (∑ i : Fin n, (a y i - m x i) ^ 2)
          + 2 * (p x y * ∑ i : Fin n, (a y i - m x i) * (m x i - ah x i))
          + p x y * ∑ i : Fin n, (m x i - ah x i) ^ 2 := by
      intro y
      have hpt : ∀ i : Fin n, (a y i - ah x i) ^ 2
          = (a y i - m x i) ^ 2 + 2 * ((a y i - m x i) * (m x i - ah x i))
            + (m x i - ah x i) ^ 2 := by
        intro i; ring
      simp_rw [hpt]
      rw [Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum]
      ring
    simp_rw [expand]
    rw [Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, key x]
    ring
  simp_rw [step]
  rw [Finset.sum_add_distrib]

/-! ### eq:l-epspm (lines 567-573): the normalised posterior-mean error. -/

def ch07_l_epspm {L : ℕ} (d : Fin L → ℝ) : ℝ :=
  Real.sqrt ((1 / (L : ℝ)) * ∑ i : Fin L, (d i) ^ 2)

theorem ch07_l_epspm_nonneg {L : ℕ} (d : Fin L → ℝ) : 0 ≤ ch07_l_epspm d :=
  Real.sqrt_nonneg _

/-- The metric vanishes exactly when the two estimators agree. -/
theorem ch07_l_epspm_eq_zero {L : ℕ} (hL : 0 < L) (d : Fin L → ℝ) :
    ch07_l_epspm d = 0 ↔ ∀ i, d i = 0 := by
  have hLpos : (0 : ℝ) < (L : ℝ) := by exact_mod_cast hL
  unfold ch07_l_epspm
  rw [Real.sqrt_eq_zero']
  have hnn : ∀ j ∈ (Finset.univ : Finset (Fin L)), (0 : ℝ) ≤ (d j) ^ 2 := by
    intro j _; positivity
  have hinv : (0 : ℝ) < 1 / (L : ℝ) := by positivity
  constructor
  · intro hc
    have hsum : ∑ i : Fin L, (d i) ^ 2 ≤ 0 := by
      by_contra hcon
      push_neg at hcon
      nlinarith [mul_pos hinv hcon]
    intro i
    have hz := (Finset.sum_eq_zero_iff_of_nonneg hnn).mp
      (le_antisymm hsum (Finset.sum_nonneg hnn)) i (Finset.mem_univ i)
    exact pow_eq_zero_iff (n := 2) (by norm_num) |>.mp hz
  · intro hc
    have hz : ∑ i : Fin L, (d i) ^ 2 = 0 := by
      refine Finset.sum_eq_zero ?_
      intro i _
      rw [hc i]
      ring
    rw [hz]
    simp

/-! ## Promotions: statements general in the dimension / the parameters

The declarations below replace the fixed-size instance checks recorded above by
statements proved at arbitrary size.  Nothing here weakens a chapter claim; each
theorem is the same assertion with `n = 2,3,4` (or `L = 3`, or a Gaussian prior)
replaced by an arbitrary parameter. -/

/-! ### eq:l-score-identity (lines 49-52), arbitrary prior

The chapter asserts that the score-posterior identity "is indifferent to the
prior".  Here that is proved: `ν` is an ARBITRARY prior on `ℝ` -- any finite
measure with a finite first moment, which is exactly the condition under which
`E[a | x]` exists at all -- and the channel is the chapter's, `x = μ a + √Δ z`
with `μ = e^{-t}`, `Δ = Δ_t`.  Differentiation under the integral sign is
supplied by `hasDerivAt_integral_of_dominated_loc_of_deriv_le`, dominating
`|∂_x K(x,a)| ≤ (|μ a| + |x₀| + 1)/Δ` on the unit ball around `x₀`; that bound
is integrable precisely because `ν` has a first moment. -/

theorem ch07_l_score_identity_prior {Δ : ℝ} (hΔ : 0 < Δ) (μ : ℝ)
    (ν : Measure ℝ) [IsFiniteMeasure ν] (hm : Integrable (fun a : ℝ => a) ν) (x : ℝ) :
    HasDerivAt (fun y => ∫ a, Real.exp (-(y - μ * a) ^ 2 / (2 * Δ)) ∂ν)
      ((∫ a, (μ * a - x) * Real.exp (-(x - μ * a) ^ 2 / (2 * Δ)) ∂ν) / Δ) x := by
  have hΔ0 : Δ ≠ 0 := ne_of_gt hΔ
  have hb1 : ∀ y a : ℝ, Real.exp (-(y - μ * a) ^ 2 / (2 * Δ)) ≤ 1 := by
    intro y a
    have h2 : (0:ℝ) ≤ (y - μ * a) ^ 2 / (2 * Δ) := by positivity
    rw [Real.exp_le_one_iff,
      show -(y - μ * a) ^ 2 / (2 * Δ) = -((y - μ * a) ^ 2 / (2 * Δ)) from by ring]
    linarith
  have hcont : ∀ y : ℝ, Continuous (fun a : ℝ => Real.exp (-(y - μ * a) ^ 2 / (2 * Δ))) := by
    intro y; fun_prop
  have hcont' : ∀ y : ℝ,
      Continuous (fun a : ℝ => Real.exp (-(y - μ * a) ^ 2 / (2 * Δ)) * ((μ * a - y) / Δ)) := by
    intro y; fun_prop
  have hFint : Integrable (fun a : ℝ => Real.exp (-(x - μ * a) ^ 2 / (2 * Δ))) ν := by
    refine Integrable.mono' (integrable_const (1:ℝ)) (hcont x).aestronglyMeasurable ?_
    filter_upwards with a
    rw [Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
    exact hb1 x a
  have hbd : Integrable (fun a : ℝ => (|μ * a| + (|x| + 1)) / Δ) ν :=
    (((hm.const_mul μ).abs).add (integrable_const _)).div_const Δ
  have main := hasDerivAt_integral_of_dominated_loc_of_deriv_le
      (F := fun y a => Real.exp (-(y - μ * a) ^ 2 / (2 * Δ)))
      (F' := fun y a => Real.exp (-(y - μ * a) ^ 2 / (2 * Δ)) * ((μ * a - y) / Δ))
      (bound := fun a => (|μ * a| + (|x| + 1)) / Δ)
      (s := Metric.ball x 1) (x₀ := x)
      (Metric.ball_mem_nhds x one_pos)
      (Filter.Eventually.of_forall (fun y => (hcont y).aestronglyMeasurable))
      hFint ((hcont' x).aestronglyMeasurable) ?_ hbd ?_
  · have hrw : (∫ a, Real.exp (-(x - μ * a) ^ 2 / (2 * Δ)) * ((μ * a - x) / Δ) ∂ν)
        = (∫ a, (μ * a - x) * Real.exp (-(x - μ * a) ^ 2 / (2 * Δ)) ∂ν) / Δ := by
      rw [← integral_div]
      exact integral_congr_ae (Filter.Eventually.of_forall (fun a => by ring))
    rw [← hrw]
    exact main.2
  · filter_upwards with a
    intro y hy
    have hy' : |y - x| < 1 := by
      rw [Metric.mem_ball, Real.dist_eq] at hy; exact hy
    have hyb : |y| ≤ |x| + 1 := by
      have h := abs_sub_abs_le_abs_sub y x
      linarith
    have hstep : |μ * a - y| ≤ |μ * a| + (|x| + 1) := by
      have h := abs_sub (μ * a) y
      linarith
    have hexppos : (0:ℝ) < Real.exp (-(y - μ * a) ^ 2 / (2 * Δ)) := Real.exp_pos _
    have hinv : (0:ℝ) < Δ⁻¹ := by positivity
    rw [Real.norm_eq_abs, abs_mul, abs_of_pos hexppos, abs_div, abs_of_pos hΔ]
    simp only [div_eq_mul_inv]
    calc Real.exp (-(y - μ * a) ^ 2 / (2 * Δ)) * (|μ * a - y| * Δ⁻¹)
        ≤ 1 * (|μ * a - y| * Δ⁻¹) :=
          mul_le_mul_of_nonneg_right (hb1 y a) (mul_nonneg (abs_nonneg _) hinv.le)
      _ = |μ * a - y| * Δ⁻¹ := one_mul _
      _ ≤ (|μ * a| + (|x| + 1)) * Δ⁻¹ := mul_le_mul_of_nonneg_right hstep hinv.le
  · filter_upwards with a
    intro y _
    have hb : HasDerivAt (fun z : ℝ => z - μ * a) 1 y := (hasDerivAt_id y).sub_const _
    have hp : HasDerivAt (fun z : ℝ => (z - μ * a) ^ 2) ((2:ℝ) * (y - μ * a) ^ 1 * 1) y :=
      hb.pow 2
    have hn := (hp.neg).div_const (2 * Δ)
    have hval : (-((2:ℝ) * (y - μ * a) ^ 1 * 1)) / (2 * Δ) = (μ * a - y) / Δ := by
      field_simp
      ring
    rw [hval] at hn
    exact hn.exp

/-- The algebraic step behind the `log` form: `(μN - xD)/Δ/D = (μ(N/D) - x)/Δ`. -/
theorem ch07_score_ratio_algebra {N D Δ : ℝ} (hD : D ≠ 0) (hΔ : Δ ≠ 0) (μ x : ℝ) :
    ((μ * N - x * D) / Δ) / D = (μ * (N / D) - x) / Δ := by
  field_simp
  all_goals ring

/-- The `log` form of the arbitrary-prior identity: with
`p(x) = ∫ K(x,a) dν(a)` and `E[a|x] = ∫ a K(x,a) dν / ∫ K(x,a) dν`,
`d/dx log p(x) = (μ E[a|x] - x)/Δ`.  This is eq:l-score-identity verbatim, for
every prior with a first moment -- no Gaussianity, no finite support. -/
theorem ch07_l_score_identity_prior_log {Δ : ℝ} (hΔ : 0 < Δ) (μ : ℝ)
    (ν : Measure ℝ) [IsFiniteMeasure ν] (hm : Integrable (fun a : ℝ => a) ν) (x : ℝ)
    (hD : (∫ a, Real.exp (-(x - μ * a) ^ 2 / (2 * Δ)) ∂ν) ≠ 0) :
    HasDerivAt (fun y => Real.log (∫ a, Real.exp (-(y - μ * a) ^ 2 / (2 * Δ)) ∂ν))
      ((μ * ((∫ a, a * Real.exp (-(x - μ * a) ^ 2 / (2 * Δ)) ∂ν)
          / (∫ a, Real.exp (-(x - μ * a) ^ 2 / (2 * Δ)) ∂ν)) - x) / Δ) x := by
  have hΔ0 : Δ ≠ 0 := ne_of_gt hΔ
  have hb1 : ∀ a : ℝ, Real.exp (-(x - μ * a) ^ 2 / (2 * Δ)) ≤ 1 := by
    intro a
    have h2 : (0:ℝ) ≤ (x - μ * a) ^ 2 / (2 * Δ) := by positivity
    rw [Real.exp_le_one_iff,
      show -(x - μ * a) ^ 2 / (2 * Δ) = -((x - μ * a) ^ 2 / (2 * Δ)) from by ring]
    linarith
  have hcont : Continuous (fun a : ℝ => Real.exp (-(x - μ * a) ^ 2 / (2 * Δ))) := by fun_prop
  have hcont2 : Continuous (fun a : ℝ => a * Real.exp (-(x - μ * a) ^ 2 / (2 * Δ))) := by fun_prop
  have hFint : Integrable (fun a : ℝ => Real.exp (-(x - μ * a) ^ 2 / (2 * Δ))) ν := by
    refine Integrable.mono' (integrable_const (1:ℝ)) hcont.aestronglyMeasurable ?_
    filter_upwards with a
    rw [Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
    exact hb1 a
  have hNint : Integrable (fun a : ℝ => a * Real.exp (-(x - μ * a) ^ 2 / (2 * Δ))) ν := by
    refine Integrable.mono' hm.abs hcont2.aestronglyMeasurable ?_
    filter_upwards with a
    rw [Real.norm_eq_abs, abs_mul, abs_of_pos (Real.exp_pos _)]
    have h0 : (0:ℝ) ≤ |a| := abs_nonneg a
    nlinarith [hb1 a, Real.exp_pos (-(x - μ * a) ^ 2 / (2 * Δ))]
  have hsplit : (∫ a, (μ * a - x) * Real.exp (-(x - μ * a) ^ 2 / (2 * Δ)) ∂ν)
      = μ * (∫ a, a * Real.exp (-(x - μ * a) ^ 2 / (2 * Δ)) ∂ν)
        - x * (∫ a, Real.exp (-(x - μ * a) ^ 2 / (2 * Δ)) ∂ν) := by
    rw [integral_congr_ae (Filter.Eventually.of_forall (fun a : ℝ =>
        show (μ * a - x) * Real.exp (-(x - μ * a) ^ 2 / (2 * Δ))
          = μ * (a * Real.exp (-(x - μ * a) ^ 2 / (2 * Δ)))
            - x * Real.exp (-(x - μ * a) ^ 2 / (2 * Δ)) from by ring)),
      integral_sub (hNint.const_mul μ) (hFint.const_mul x), integral_const_mul,
      integral_const_mul]
  have hd := (ch07_l_score_identity_prior hΔ μ ν hm x).log hD
  have heq : ((∫ a, (μ * a - x) * Real.exp (-(x - μ * a) ^ 2 / (2 * Δ)) ∂ν) / Δ)
      / (∫ a, Real.exp (-(x - μ * a) ^ 2 / (2 * Δ)) ∂ν)
      = (μ * ((∫ a, a * Real.exp (-(x - μ * a) ^ 2 / (2 * Δ)) ∂ν)
          / (∫ a, Real.exp (-(x - μ * a) ^ 2 / (2 * Δ)) ∂ν)) - x) / Δ := by
    rw [hsplit]
    exact ch07_score_ratio_algebra hD hΔ0 μ x
  rw [← heq]
  exact hd

/-! ### noname-9 (lines 232-238) at arbitrary chain length

`(BᵀB)_{ij} = α^{|i-j|}(1-α^{2(L-max(i,j))})/(1-α²)` for EVERY `L` and every
index pair, not just `L = 2,3,4`.  The proof is the entrywise one: the summand
`B_{ki}B_{kj}` vanishes unless `k ≥ max(i,j)`, reindexing `k = max + p` turns the
exponent `(k-i)+(k-j)` into `|i-j| + 2p`, and the remaining geometric series is
summed by `geom_sum_eq`. -/
theorem ch07_BtB_gen {L : ℕ} (α : ℝ) (h : α ^ 2 ≠ 1) (i j : Fin L) :
    ((ch07_B α L)ᵀ * ch07_B α L) i j = ch07_BtB_rhs α L (i : ℕ) (j : ℕ) := by
  have hα : (1 : ℝ) - α ^ 2 ≠ 0 := sub_ne_zero.mpr (Ne.symm h)
  have hα' : α ^ 2 - 1 ≠ 0 := sub_ne_zero.mpr h
  have hiL : (i : ℕ) < L := i.isLt
  have hjL : (j : ℕ) < L := j.isLt
  have step0 : ((ch07_B α L)ᵀ * ch07_B α L) i j
      = ∑ k : Fin L, (fun (p : ℕ) =>
          (if (i : ℕ) ≤ p then α ^ (p - (i : ℕ)) else 0)
            * (if (j : ℕ) ≤ p then α ^ (p - (j : ℕ)) else 0)) (k : ℕ) := by
    rw [Matrix.mul_apply]
    exact Finset.sum_congr rfl (fun k _ => by simp [ch07_B, Matrix.transpose_apply])
  rw [step0]
  rw [Fin.sum_univ_eq_sum_range (fun p : ℕ =>
      (if (i : ℕ) ≤ p then α ^ (p - (i : ℕ)) else 0)
        * (if (j : ℕ) ≤ p then α ^ (p - (j : ℕ)) else 0)) L]
  have hterm : ∀ k ∈ Finset.range L,
      (if (i : ℕ) ≤ k then α ^ (k - (i : ℕ)) else 0)
          * (if (j : ℕ) ≤ k then α ^ (k - (j : ℕ)) else 0)
        = if max (i : ℕ) (j : ℕ) ≤ k then α ^ ((k - (i : ℕ)) + (k - (j : ℕ))) else 0 := by
    intro k _
    by_cases hik : (i : ℕ) ≤ k
    · by_cases hjk : (j : ℕ) ≤ k
      · rw [if_pos hik, if_pos hjk, if_pos (by omega : max (i : ℕ) (j : ℕ) ≤ k), ← pow_add]
      · rw [if_neg hjk, if_neg (by omega : ¬ (max (i : ℕ) (j : ℕ) ≤ k)), mul_zero]
    · rw [if_neg hik, if_neg (by omega : ¬ (max (i : ℕ) (j : ℕ) ≤ k)), zero_mul]
  rw [Finset.sum_congr rfl hterm, ← Finset.sum_filter]
  have hset : (Finset.range L).filter (fun k => max (i : ℕ) (j : ℕ) ≤ k)
      = Finset.Ico (max (i : ℕ) (j : ℕ)) L := by
    ext k
    simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_Ico]
    omega
  rw [hset, Finset.sum_Ico_eq_sum_range]
  have hterm2 : ∀ p ∈ Finset.range (L - max (i : ℕ) (j : ℕ)),
      α ^ ((max (i : ℕ) (j : ℕ) + p - (i : ℕ)) + (max (i : ℕ) (j : ℕ) + p - (j : ℕ)))
        = α ^ (max (i : ℕ) (j : ℕ) - min (i : ℕ) (j : ℕ)) * (α ^ 2) ^ p := by
    intro p _
    rw [show (max (i : ℕ) (j : ℕ) + p - (i : ℕ)) + (max (i : ℕ) (j : ℕ) + p - (j : ℕ))
        = (max (i : ℕ) (j : ℕ) - min (i : ℕ) (j : ℕ)) + 2 * p from by omega, pow_add, pow_mul]
  rw [Finset.sum_congr rfl hterm2, ← Finset.mul_sum, geom_sum_eq h]
  unfold ch07_BtB_rhs
  rw [pow_mul]
  field_simp
  all_goals ring

/-! ### eq:l-trap-weights (lines 260-267) at arbitrary grid size

`∑_i w_i = 2A` for EVERY `N ≥ 2`, not just `N = 3,4,5`: the two halved endpoints
remove exactly one full `h` from `N h`, and `(N-1)h = 2A` by eq:l-grid. -/
theorem ch07_l_trap_weights_sum (A : ℝ) {N : ℕ} (hN : 2 ≤ N) :
    ∑ i ∈ Finset.range N, ch07_w A N i = 2 * A := by
  have hne : (0 : ℕ) ≠ N - 1 := by omega
  have hNR : ((N : ℝ) - 1) ≠ 0 := by
    have h2 : (2 : ℝ) ≤ (N : ℝ) := by exact_mod_cast hN
    linarith
  have hrw : ∀ i : ℕ, ch07_w A N i
      = ch07_h A N - (if i = 0 ∨ i = N - 1 then ch07_h A N / 2 else 0) := by
    intro i
    unfold ch07_w
    by_cases hc : i = 0 ∨ i = N - 1
    · rw [if_pos hc, if_pos hc]; ring
    · rw [if_neg hc, if_neg hc]; ring
  simp only [hrw]
  rw [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  have hfil : ∑ i ∈ Finset.range N, (if i = 0 ∨ i = N - 1 then ch07_h A N / 2 else 0)
      = ch07_h A N := by
    rw [← Finset.sum_filter]
    have hset : (Finset.range N).filter (fun i => i = 0 ∨ i = N - 1) = {0, N - 1} := by
      ext i
      simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_insert, Finset.mem_singleton]
      omega
    rw [hset, Finset.sum_const, Finset.card_pair hne, nsmul_eq_mul]
    push_cast
    ring
  rw [hfil]
  unfold ch07_h
  field_simp

/-! ### eq:l-resolution (lines 334-337): the `√(2t) + O(t^{3/2})` claim, explicitly

The text says the unary likelihood has standard deviation `√Δ_t/μ`, "which for
small `t` is `√(2t) + O(t^{3/2})`".  Below: the closed form `√(e^{2t}-1)`, the
two-sided bound `√(2t) ≤ √Δ_t/μ ≤ √(2t) e^t` for all `t ≥ 0`, and an explicit
`O(t^{3/2})` constant on `[0,1]`.  The first inequality also shows the gate
`√(2t) ≳ 3h` is the CONSERVATIVE form of the test: the true standard deviation
is never smaller than `√(2t)`. -/
theorem ch07_l_resolution_sigma {t : ℝ} (ht : 0 ≤ t) :
    Real.sqrt (ch07_Delta t) / ch07_mu t = Real.sqrt (Real.exp (2 * t) - 1) := by
  have hmu : (0 : ℝ) < ch07_mu t := Real.exp_pos _
  have hsq : (Real.exp (-t)) ^ 2 = Real.exp (-2 * t) := by
    rw [sq, ← Real.exp_add]; congr 1; ring
  have hee : Real.exp (2 * t) * Real.exp (-2 * t) = 1 := by
    rw [← Real.exp_add, show 2 * t + -2 * t = 0 from by ring, Real.exp_zero]
  have hX : (0 : ℝ) ≤ Real.exp (2 * t) - 1 := by
    have h := Real.add_one_le_exp (2 * t)
    linarith
  have hkey : ch07_Delta t = (Real.exp (2 * t) - 1) * (ch07_mu t) ^ 2 := by
    unfold ch07_Delta ch07_mu
    rw [hsq]
    linear_combination -hee
  rw [hkey, Real.sqrt_mul hX, Real.sqrt_sq hmu.le]
  field_simp

theorem ch07_l_resolution_sigma_bounds {t : ℝ} (ht : 0 ≤ t) :
    Real.sqrt (2 * t) ≤ Real.sqrt (ch07_Delta t) / ch07_mu t
      ∧ Real.sqrt (ch07_Delta t) / ch07_mu t ≤ Real.sqrt (2 * t) * Real.exp t := by
  rw [ch07_l_resolution_sigma ht]
  have h1 : 2 * t ≤ Real.exp (2 * t) - 1 := by
    have h := Real.add_one_le_exp (2 * t); linarith
  have h2 : Real.exp (2 * t) - 1 ≤ 2 * t * Real.exp (2 * t) := by
    have hneg := Real.add_one_le_exp (-(2 * t))
    have hpos : (0 : ℝ) < Real.exp (2 * t) := Real.exp_pos _
    have h3 : (-(2 * t) + 1) * Real.exp (2 * t) ≤ Real.exp (-(2 * t)) * Real.exp (2 * t) :=
      mul_le_mul_of_nonneg_right hneg hpos.le
    rw [← Real.exp_add, show -(2 * t) + 2 * t = 0 from by ring, Real.exp_zero] at h3
    nlinarith [h3]
  refine ⟨Real.sqrt_le_sqrt h1, ?_⟩
  calc Real.sqrt (Real.exp (2 * t) - 1) ≤ Real.sqrt (2 * t * Real.exp (2 * t)) :=
        Real.sqrt_le_sqrt h2
    _ = Real.sqrt (2 * t) * Real.exp t := by
        rw [Real.sqrt_mul (by linarith),
          show Real.exp (2 * t) = (Real.exp t) ^ 2 from by rw [sq, ← Real.exp_add]; congr 1; ring,
          Real.sqrt_sq (Real.exp_pos t).le]

theorem ch07_l_resolution_sigma_taylor {t : ℝ} (ht : 0 ≤ t) (ht1 : t ≤ 1) :
    0 ≤ Real.sqrt (ch07_Delta t) / ch07_mu t - Real.sqrt (2 * t)
      ∧ Real.sqrt (ch07_Delta t) / ch07_mu t - Real.sqrt (2 * t) ≤ 5 * (t * Real.sqrt t) := by
  obtain ⟨hlo, hhi⟩ := ch07_l_resolution_sigma_bounds ht
  refine ⟨by linarith, ?_⟩
  have hst : (0 : ℝ) ≤ Real.sqrt t := Real.sqrt_nonneg t
  have hs2 : Real.sqrt (2 * t) = Real.sqrt 2 * Real.sqrt t := Real.sqrt_mul (by norm_num) t
  have hsqrt2 : Real.sqrt 2 ≤ 1.5 := by
    rw [show (1.5 : ℝ) = Real.sqrt (1.5 ^ 2) from (Real.sqrt_sq (by norm_num)).symm]
    exact Real.sqrt_le_sqrt (by norm_num)
  have hE0 : (0 : ℝ) ≤ Real.exp t - 1 := by
    have h := Real.add_one_le_exp t; linarith
  have he1 : Real.exp t ≤ 3 := by
    have h := Real.exp_le_exp.mpr ht1
    have h9 := Real.exp_one_lt_d9
    linarith
  have he2 : Real.exp t - 1 ≤ t * Real.exp t := by
    have hneg := Real.add_one_le_exp (-t)
    have hpos : (0 : ℝ) < Real.exp t := Real.exp_pos t
    have h3 : (-t + 1) * Real.exp t ≤ Real.exp (-t) * Real.exp t :=
      mul_le_mul_of_nonneg_right hneg hpos.le
    rw [← Real.exp_add, show -t + t = 0 from by ring, Real.exp_zero] at h3
    nlinarith [h3]
  have he3 : Real.exp t - 1 ≤ 3 * t := by nlinarith [he2, he1, ht]
  have step1 : Real.sqrt (ch07_Delta t) / ch07_mu t - Real.sqrt (2 * t)
      ≤ Real.sqrt (2 * t) * (Real.exp t - 1) := by nlinarith [hhi]
  have a1 : Real.sqrt 2 * (Real.sqrt t * (Real.exp t - 1))
      ≤ 1.5 * (Real.sqrt t * (Real.exp t - 1)) :=
    mul_le_mul_of_nonneg_right hsqrt2 (mul_nonneg hst hE0)
  have a2 : Real.sqrt t * (Real.exp t - 1) ≤ Real.sqrt t * (3 * t) :=
    mul_le_mul_of_nonneg_left he3 hst
  have step2 : Real.sqrt (2 * t) * (Real.exp t - 1) ≤ 5 * (t * Real.sqrt t) := by
    rw [hs2, mul_assoc]
    nlinarith [a1, a2, hst, ht, mul_nonneg ht hst]
  linarith

/-! ### eq:l-lmmse (lines 414-422): optimality of the stated gain, in every dimension

The chapter calls `Cov(a,x)Cov(x)^{-1}x` the LMMSE estimator.  Below is the
property that justifies the name, proved for arbitrary dimensions `n, m` and
arbitrary second moments: among ALL linear maps `G`, the risk `E‖a - Gx‖²` is
minimised exactly at any `G⋆` solving the normal equations `G⋆ Cov(x) = Cov(a,x)`
-- in particular at `G⋆ = Cov(a,x)Cov(x)^{-1}` (`ch07_l_lmmse_normal_eq`) -- and
the excess risk is `tr[(G - G⋆)Cov(x)(G - G⋆)ᵀ] ≥ 0`.  As elsewhere in this file
the risk is written in the second moments it depends on, which is bilinearity of
expectation and nothing more (same convention as eq:lmmse-covs). -/
def ch07_lmmse_risk {n m : ℕ} (Sa : Matrix (Fin n) (Fin n) ℝ) (Cax : Matrix (Fin n) (Fin m) ℝ)
    (Cxx : Matrix (Fin m) (Fin m) ℝ) (G : Matrix (Fin n) (Fin m) ℝ) : ℝ :=
  Sa.trace - (G * Caxᵀ).trace - (Cax * Gᵀ).trace + (G * Cxx * Gᵀ).trace

theorem ch07_l_lmmse_excess {n m : ℕ} (Sa : Matrix (Fin n) (Fin n) ℝ)
    (Cax : Matrix (Fin n) (Fin m) ℝ) (Cxx : Matrix (Fin m) (Fin m) ℝ)
    (G Gstar : Matrix (Fin n) (Fin m) ℝ) (hsym : Cxxᵀ = Cxx) (hnormal : Gstar * Cxx = Cax) :
    ch07_lmmse_risk Sa Cax Cxx G
      = ch07_lmmse_risk Sa Cax Cxx Gstar + ((G - Gstar) * Cxx * (G - Gstar)ᵀ).trace := by
  have hC : Cxx * Gstarᵀ = Caxᵀ := by
    rw [← hnormal, Matrix.transpose_mul, hsym]
  have e1 : (G - Gstar) * Cxx * (G - Gstar)ᵀ
      = G * (Cxx * Gᵀ) - G * (Cxx * Gstarᵀ) - Gstar * (Cxx * Gᵀ) + Gstar * (Cxx * Gstarᵀ) := by
    simp only [Matrix.sub_mul, Matrix.mul_sub, Matrix.transpose_sub, Matrix.mul_assoc]
    abel
  have hstar : (Gstar * Caxᵀ).trace = (Cax * Gstarᵀ).trace := by
    rw [← hC, ← Matrix.mul_assoc, hnormal]
  unfold ch07_lmmse_risk
  rw [e1]
  simp only [Matrix.trace_add, Matrix.trace_sub, ← Matrix.mul_assoc, hC, hnormal]
  linarith [hstar]

theorem ch07_l_lmmse_optimal {n m : ℕ} (Sa : Matrix (Fin n) (Fin n) ℝ)
    (Cax : Matrix (Fin n) (Fin m) ℝ) (Cxx : Matrix (Fin m) (Fin m) ℝ)
    (G Gstar : Matrix (Fin n) (Fin m) ℝ) (hpsd : Cxx.PosSemidef)
    (hnormal : Gstar * Cxx = Cax) :
    ch07_lmmse_risk Sa Cax Cxx Gstar ≤ ch07_lmmse_risk Sa Cax Cxx G := by
  have hct : ∀ {p q : ℕ} (Mx : Matrix (Fin p) (Fin q) ℝ), Mxᴴ = Mxᵀ := by
    intro p q Mx
    ext a b
    simp [Matrix.conjTranspose_apply, Matrix.transpose_apply]
  have hsym : Cxxᵀ = Cxx := by rw [← hct Cxx]; exact hpsd.1
  rw [ch07_l_lmmse_excess Sa Cax Cxx G Gstar hsym hnormal]
  have hnn : (0 : ℝ) ≤ ((G - Gstar) * Cxx * (G - Gstar)ᵀ).trace := by
    have h := (hpsd.mul_mul_conjTranspose_same (G - Gstar)).trace_nonneg
    rwa [hct (G - Gstar)] at h
  linarith

/-! ### noname-12 (lines 489-493) in every dimension

The scalar completion of the square above is replaced by the general one.  Two
statements, both for arbitrary `n` (the state block) and `m` (the observation
block):

* `ch07_gauss_cond_mean_gen`: for a symmetric precision block `P`, the joint
  exponent `aᵀPa - 2aᵀ(PG)x + xᵀSx` equals `(a - Gx)ᵀP(a - Gx) + xᵀ(S - GᵀPG)x`,
  so as a density in `a` it is Gaussian with mean `Gx` and covariance `P⁻¹`.
* `ch07_gauss_cond_mean_schur`: for a joint covariance `Σ = [[S,C],[Cᵀ,V]]` the
  blocks of the precision `Σ⁻¹` satisfy `K₁₁ (C V⁻¹) = -K₁₂`, i.e. the `G` of the
  first statement is exactly `C V⁻¹ = Cov(a,x)Cov(x)^{-1}`.

Together: `E[a|x] = Cov(a,x)Cov(x)^{-1}x` for jointly Gaussian zero-mean `(a,x)`
in any dimension, granted only the standard reading that the mean of a Gaussian
exponent is the point where the completed square vanishes. -/
theorem ch07_gauss_cond_mean_gen {n m : ℕ} (P : Matrix (Fin n) (Fin n) ℝ)
    (G : Matrix (Fin n) (Fin m) ℝ) (S : Matrix (Fin m) (Fin m) ℝ) (hP : Pᵀ = P)
    (a : Fin n → ℝ) (x : Fin m → ℝ) :
    a ⬝ᵥ (P *ᵥ a) - 2 * (a ⬝ᵥ ((P * G) *ᵥ x)) + x ⬝ᵥ (S *ᵥ x)
      = (a - G *ᵥ x) ⬝ᵥ (P *ᵥ (a - G *ᵥ x)) + x ⬝ᵥ ((S - Gᵀ * P * G) *ᵥ x) := by
  have hadj : ∀ {p q : ℕ} (Mx : Matrix (Fin p) (Fin q) ℝ) (u : Fin p → ℝ) (v : Fin q → ℝ),
      u ⬝ᵥ (Mx *ᵥ v) = (Mxᵀ *ᵥ u) ⬝ᵥ v := by
    intro p q Mx u v
    rw [Matrix.dotProduct_mulVec, Matrix.mulVec_transpose]
  have hsymm : (G *ᵥ x) ⬝ᵥ (P *ᵥ a) = a ⬝ᵥ ((P * G) *ᵥ x) := by
    rw [hadj P (G *ᵥ x) a, hP, dotProduct_comm, Matrix.mulVec_mulVec]
  have hGG : (G *ᵥ x) ⬝ᵥ ((P * G) *ᵥ x) = x ⬝ᵥ ((Gᵀ * P * G) *ᵥ x) := by
    rw [hadj (P * G) (G *ᵥ x) x, Matrix.transpose_mul, hP, Matrix.mulVec_mulVec,
      dotProduct_comm]
  have e1 : (a - G *ᵥ x) ⬝ᵥ (P *ᵥ (a - G *ᵥ x))
      = a ⬝ᵥ (P *ᵥ a) - a ⬝ᵥ ((P * G) *ᵥ x) - (G *ᵥ x) ⬝ᵥ (P *ᵥ a)
        + (G *ᵥ x) ⬝ᵥ ((P * G) *ᵥ x) := by
    rw [Matrix.mulVec_sub, sub_dotProduct, dotProduct_sub, dotProduct_sub,
      Matrix.mulVec_mulVec]
    ring
  have e2 : x ⬝ᵥ ((S - Gᵀ * P * G) *ᵥ x) = x ⬝ᵥ (S *ᵥ x) - x ⬝ᵥ ((Gᵀ * P * G) *ᵥ x) := by
    rw [Matrix.sub_mulVec, dotProduct_sub]
  rw [e1, e2, hGG, hsymm]
  ring

theorem ch07_gauss_cond_mean_schur {n m : ℕ} (S : Matrix (Fin n) (Fin n) ℝ)
    (C : Matrix (Fin n) (Fin m) ℝ) (V : Matrix (Fin m) (Fin m) ℝ) [Invertible V]
    [Invertible (S - C * ⅟V * Cᵀ)] [Invertible (Matrix.fromBlocks S C Cᵀ V)] :
    (⅟(Matrix.fromBlocks S C Cᵀ V)).toBlocks₁₁ * (C * ⅟V)
      = -(⅟(Matrix.fromBlocks S C Cᵀ V)).toBlocks₁₂ := by
  rw [Matrix.invOf_fromBlocks₂₂_eq, Matrix.toBlocks_fromBlocks₁₁, Matrix.toBlocks_fromBlocks₁₂,
    neg_neg, ← Matrix.mul_assoc]

/-! ### eq:l-markov-score (lines 128-135) at arbitrary chain length and site

The checks recorded above fix the chain length at `L = 3`.  Here the theorem is
proved for a chain of `m + 1 + n` sites over an arbitrary finite alphabet with
the marked site at position `m`; every (length, marked site) pair arises this
way, so nothing is special-cased any more.

Bookkeeping convention.  The states of the left block are indexed *outwards*
from the marked site (`u i` is the state `i+1` steps to the left of it, and
`obL i` is that site's observation potential); likewise `v i` and `obR i` on the
right.  This is a relabelling of the sites only -- the potentials are arbitrary
per-site functions either way -- and it removes all index arithmetic from the
statement.  The prior `p₀` sits at the far-left end of the chain, which is where
the recursion bottoms out.  `M a' a` is `M(a' | a)`, as in the `L = 3` lemmas. -/

section MarkovScoreGen

variable {A : Type} [Fintype A]

/-- Splitting a sum over configurations of `n+1` sites at the first site. -/
theorem ch07_sum_pi_succ {n : ℕ} (g : (Fin (n + 1) → A) → ℝ) :
    ∑ f : Fin (n + 1) → A, g f = ∑ c : A, ∑ v : Fin n → A, g (Fin.cons c v) := by
  rw [← Fintype.sum_equiv
      (Equiv.mk (fun p : A × (Fin n → A) => Fin.cons p.1 p.2) (fun f => (f 0, Fin.tail f))
        (fun p => by simp) (fun f => by simp))
      (fun p => g (Fin.cons p.1 p.2)) g (fun _ => rfl),
    Fintype.sum_prod_type]

/-- The joint weight of the left block: `p₀` at the far end, one transition
`M(·|·)` per bond and one potential `obL` per site, ending in the state `c` of
the marked site. -/
def ch07_preW (p0 : A → ℝ) (M : A → A → ℝ) : (ℕ → A → ℝ) → (m : ℕ) → A → (Fin m → A) → ℝ
  | _, 0, c, _ => p0 c
  | obL, (m + 1), c, u =>
      M c (u 0) * obL 0 (u 0) * ch07_preW p0 M (fun i => obL (i + 1)) m (u 0) (Fin.tail u)

/-- The joint weight of the right block hanging off the marked state `c`; the
terminal message is `1`, as in eq:l-grid-bwd. -/
def ch07_sufW (M : A → A → ℝ) : (ℕ → A → ℝ) → (n : ℕ) → A → (Fin n → A) → ℝ
  | _, 0, _, _ => 1
  | obR, (n + 1), c, v =>
      M (v 0) c * obR 0 (v 0) * ch07_sufW M (fun i => obR (i + 1)) n (v 0) (Fin.tail v)

/-- The forward message `α_k` of Proposition `prop:bp-convA`, as a recursion:
`α_0 = p₀` and `α_{k}(c) = Σ_a M(c|a) ψ_ob(a) α_{k-1}(a)`.  It does *not*
include the potential at its own site, matching eq:l-markov-score. -/
def ch07_alphaMsg (p0 : A → ℝ) (M : A → A → ℝ) : (ℕ → A → ℝ) → ℕ → A → ℝ
  | _, 0, c => p0 c
  | obL, (m + 1), c =>
      ∑ a : A, M c a * obL 0 a * ch07_alphaMsg p0 M (fun i => obL (i + 1)) m a

/-- The backward message `β_k`: `β = 1` at the terminal site and
`β_k(c) = Σ_a M(a|c) ψ_ob(a) β_{k+1}(a)`. -/
def ch07_betaMsg (M : A → A → ℝ) : (ℕ → A → ℝ) → ℕ → A → ℝ
  | _, 0, _ => 1
  | obR, (n + 1), c =>
      ∑ a : A, M a c * obR 0 a * ch07_betaMsg M (fun i => obR (i + 1)) n a

/-- The forward message is the sum of the left-block joint weight over all
configurations of that block: the first half of the Fubini rearrangement. -/
theorem ch07_preW_sum (p0 : A → ℝ) (M : A → A → ℝ) :
    ∀ (m : ℕ) (obL : ℕ → A → ℝ) (c : A),
      ∑ u : Fin m → A, ch07_preW p0 M obL m c u = ch07_alphaMsg p0 M obL m c := by
  intro m
  induction m with
  | zero =>
      intro obL c
      simp [ch07_preW, ch07_alphaMsg]
  | succ m ih =>
      intro obL c
      rw [ch07_sum_pi_succ]
      simp only [ch07_preW, ch07_alphaMsg, Fin.cons_zero, Fin.tail_cons]
      refine Finset.sum_congr rfl fun a _ => ?_
      rw [← Finset.mul_sum, ih (fun i => obL (i + 1)) a]

/-- The backward message is the sum of the right-block joint weight over all
configurations of that block: the second half of the rearrangement. -/
theorem ch07_sufW_sum (M : A → A → ℝ) :
    ∀ (n : ℕ) (obR : ℕ → A → ℝ) (c : A),
      ∑ v : Fin n → A, ch07_sufW M obR n c v = ch07_betaMsg M obR n c := by
  intro n
  induction n with
  | zero =>
      intro obR c
      simp [ch07_sufW, ch07_betaMsg]
  | succ n ih =>
      intro obR c
      rw [ch07_sum_pi_succ]
      simp only [ch07_sufW, ch07_betaMsg, Fin.cons_zero, Fin.tail_cons]
      refine Finset.sum_congr rfl fun a _ => ?_
      rw [← Finset.mul_sum, ih (fun i => obR (i + 1)) a]

/-- eq:l-markov-score, numerator, at arbitrary chain length `m + 1 + n` and
arbitrary marked site `m`: summing any site function `val` against the *full
joint* of the chain equals summing it against the product of the two messages
and the potential at the marked site.  `L = 3` is the case `m = 1`, `n = 1`. -/
theorem ch07_l_markov_score_num_chain (val p0 obC : A → ℝ) (obL obR : ℕ → A → ℝ)
    (M : A → A → ℝ) (m n : ℕ) :
    (∑ u : Fin m → A, ∑ c : A, ∑ v : Fin n → A,
        val c * (ch07_preW p0 M obL m c u * obC c * ch07_sufW M obR n c v))
      = ∑ c : A, val c * (ch07_alphaMsg p0 M obL m c * obC c * ch07_betaMsg M obR n c) := by
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun c _ => ?_
  have key : ∀ (P : (Fin m → A) → ℝ) (S : (Fin n → A) → ℝ) (κ : ℝ),
      (∑ u : Fin m → A, ∑ v : Fin n → A, κ * (P u * S v))
        = κ * ((∑ u : Fin m → A, P u) * ∑ v : Fin n → A, S v) := by
    intro P S κ
    rw [Fintype.sum_mul_sum, Finset.mul_sum]
    refine Finset.sum_congr rfl fun u _ => ?_
    rw [Finset.mul_sum]
  have hre : ∀ (u : Fin m → A) (v : Fin n → A),
      val c * (ch07_preW p0 M obL m c u * obC c * ch07_sufW M obR n c v)
        = (val c * obC c) * (ch07_preW p0 M obL m c u * ch07_sufW M obR n c v) := by
    intro u v; ring
  simp_rw [hre]
  rw [key (fun u => ch07_preW p0 M obL m c u) (fun v => ch07_sufW M obR n c v) (val c * obC c),
    ch07_preW_sum p0 M m obL c, ch07_sufW_sum M n obR c]
  ring

/-- eq:l-markov-score, denominator: the same identity at `val ≡ 1`. -/
theorem ch07_l_markov_score_den_chain (p0 obC : A → ℝ) (obL obR : ℕ → A → ℝ)
    (M : A → A → ℝ) (m n : ℕ) :
    (∑ u : Fin m → A, ∑ c : A, ∑ v : Fin n → A,
        ch07_preW p0 M obL m c u * obC c * ch07_sufW M obR n c v)
      = ∑ c : A, ch07_alphaMsg p0 M obL m c * obC c * ch07_betaMsg M obR n c := by
  have h := ch07_l_markov_score_num_chain (fun _ => (1 : ℝ)) p0 obC obL obR M m n
  simpa using h

/-- eq:l-markov-score itself, at arbitrary chain length and marked site: the
posterior mean of the marked site under the full chain posterior is the ratio of
the two message integrals.  Both sides are the same ratio, so the equation holds
whatever the (common) value of the denominator. -/
theorem ch07_l_markov_score_chain (val p0 obC : A → ℝ) (obL obR : ℕ → A → ℝ)
    (M : A → A → ℝ) (m n : ℕ) :
    (∑ u : Fin m → A, ∑ c : A, ∑ v : Fin n → A,
        val c * (ch07_preW p0 M obL m c u * obC c * ch07_sufW M obR n c v))
      / (∑ u : Fin m → A, ∑ c : A, ∑ v : Fin n → A,
        ch07_preW p0 M obL m c u * obC c * ch07_sufW M obR n c v)
      = (∑ c : A, val c * (ch07_alphaMsg p0 M obL m c * obC c * ch07_betaMsg M obR n c))
      / (∑ c : A, ch07_alphaMsg p0 M obL m c * obC c * ch07_betaMsg M obR n c) := by
  rw [ch07_l_markov_score_num_chain, ch07_l_markov_score_den_chain]

end MarkovScoreGen

/-! ### noname-13 (lines 495-498) and eq:lmmse-covs (lines 500-505), derived
from the channel

`ch07_lmmse_covs` above encodes the cross-covariance claim as
`X + √Δ_t • 0 = X`, with the literal zero matrix standing in for `Cov(a,z)`, so
that conjunct could not fail; and the channel `x = e^{-t}a + √Δ_t z` itself was
never written down.  Both gaps are closed here.  The channel is a Lean
definition on a *product* sample space `X × Z` whose weight factorises -- that
product is precisely the independence `z ⊥ a` of the display -- and both
conjuncts of eq:lmmse-covs are then derived from bilinearity of the second
moment.  In particular `Cov(a,z) = 0` is *proved* from independence together
with `E[z] = 0`, instead of being asserted. -/

/-- The cross second-moment matrix `E[u vᵀ]` of two random vectors on a finite
weighted sample space; for centred variables this is `Cov(u,v)`. -/
def ch07_cov {n m : ℕ} {Ω : Type} [Fintype Ω] (p : Ω → ℝ)
    (u : Ω → Fin n → ℝ) (v : Ω → Fin m → ℝ) : Matrix (Fin n) (Fin m) ℝ :=
  Matrix.of fun i j => ∑ ω : Ω, p ω * (u ω i * v ω j)

/-- noname-13: the OU channel `x = e^{-t} a + √Δ_t z` in vector form.  The state
`a` is carried by the first factor of the sample space and the noise `z` by the
second, so a product weight on `X × Z` makes them independent. -/
def ch07_ou_obs {n : ℕ} {X Z : Type} (t : ℝ) (a : X → Fin n → ℝ) (z : Z → Fin n → ℝ) :
    X × Z → Fin n → ℝ :=
  fun ω i => Real.exp (-t) * a ω.1 i + Real.sqrt (ch07_Delta t) * z ω.2 i

theorem ch07_cov_transpose {n m : ℕ} {Ω : Type} [Fintype Ω] (p : Ω → ℝ)
    (u : Ω → Fin n → ℝ) (v : Ω → Fin m → ℝ) :
    (ch07_cov p u v)ᵀ = ch07_cov p v u := by
  ext i j
  simp only [Matrix.transpose_apply, ch07_cov, Matrix.of_apply]
  exact Finset.sum_congr rfl fun ω _ => by ring

/-- Under a product weight, a second moment of two functions of the *state*
reduces to the state's own second moment. -/
theorem ch07_cov_prod_fst {n m : ℕ} {X Z : Type} [Fintype X] [Fintype Z]
    (pa : X → ℝ) (pz : Z → ℝ) (hpz : ∑ w : Z, pz w = 1)
    (u : X → Fin n → ℝ) (v : X → Fin m → ℝ) :
    ch07_cov (fun ω : X × Z => pa ω.1 * pz ω.2) (fun ω => u ω.1) (fun ω => v ω.1)
      = ch07_cov pa u v := by
  ext i j
  simp only [ch07_cov, Matrix.of_apply]
  rw [Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun x _ => ?_
  have hx : ∀ w : Z, (pa x * pz w) * (u x i * v x j)
      = (pa x * (u x i * v x j)) * pz w := fun w => by ring
  simp_rw [hx]
  rw [← Finset.mul_sum, hpz, mul_one]

/-- The mirror statement for two functions of the noise. -/
theorem ch07_cov_prod_snd {n m : ℕ} {X Z : Type} [Fintype X] [Fintype Z]
    (pa : X → ℝ) (pz : Z → ℝ) (hpa : ∑ x : X, pa x = 1)
    (u : Z → Fin n → ℝ) (v : Z → Fin m → ℝ) :
    ch07_cov (fun ω : X × Z => pa ω.1 * pz ω.2) (fun ω => u ω.2) (fun ω => v ω.2)
      = ch07_cov pz u v := by
  ext i j
  simp only [ch07_cov, Matrix.of_apply]
  rw [Fintype.sum_prod_type]
  have hx : ∀ x : X, ∀ w : Z, (pa x * pz w) * (u w i * v w j)
      = pa x * (pz w * (u w i * v w j)) := fun x w => by ring
  simp_rw [hx, ← Finset.mul_sum, ← Finset.sum_mul, hpa, one_mul]

/-- Independence kills the cross-covariance: with a product weight and centred
noise, `Cov(a,z) = 0`.  This is the conjunct that `ch07_lmmse_covs` could only
assert. -/
theorem ch07_cov_prod_cross {n m : ℕ} {X Z : Type} [Fintype X] [Fintype Z]
    (pa : X → ℝ) (pz : Z → ℝ) (u : X → Fin n → ℝ) (z : Z → Fin m → ℝ)
    (hz : ∀ j, ∑ w : Z, pz w * z w j = 0) :
    ch07_cov (fun ω : X × Z => pa ω.1 * pz ω.2) (fun ω => u ω.1) (fun ω => z ω.2) = 0 := by
  ext i j
  simp only [ch07_cov, Matrix.of_apply, Matrix.zero_apply]
  rw [Fintype.sum_prod_type]
  have hx : ∀ x : X, ∑ w : Z, (pa x * pz w) * (u x i * z w j)
      = (pa x * u x i) * ∑ w : Z, pz w * z w j := by
    intro x
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun w _ => by ring
  simp_rw [hx, hz, mul_zero]
  exact Finset.sum_const_zero

/-- Bilinearity of the second moment in the channel's right argument. -/
theorem ch07_cov_channel_right {n k : ℕ} {X Z : Type} [Fintype X] [Fintype Z]
    (t : ℝ) (P : X × Z → ℝ) (u : X × Z → Fin k → ℝ)
    (a : X → Fin n → ℝ) (z : Z → Fin n → ℝ) :
    ch07_cov P u (ch07_ou_obs t a z)
      = Real.exp (-t) • ch07_cov P u (fun ω => a ω.1)
        + Real.sqrt (ch07_Delta t) • ch07_cov P u (fun ω => z ω.2) := by
  ext i j
  simp only [ch07_cov, ch07_ou_obs, Matrix.of_apply, Matrix.add_apply, Matrix.smul_apply,
    smul_eq_mul]
  rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun ω _ => by ring

/-- Bilinearity of the second moment in both channel arguments. -/
theorem ch07_cov_channel_both {n : ℕ} {X Z : Type} [Fintype X] [Fintype Z]
    (t : ℝ) (P : X × Z → ℝ) (a : X → Fin n → ℝ) (z : Z → Fin n → ℝ) :
    ch07_cov P (ch07_ou_obs t a z) (ch07_ou_obs t a z)
      = (Real.exp (-t)) ^ 2 • ch07_cov P (fun ω => a ω.1) (fun ω => a ω.1)
        + (Real.exp (-t) * Real.sqrt (ch07_Delta t))
            • ch07_cov P (fun ω => a ω.1) (fun ω => z ω.2)
        + (Real.exp (-t) * Real.sqrt (ch07_Delta t))
            • ch07_cov P (fun ω => z ω.2) (fun ω => a ω.1)
        + (Real.sqrt (ch07_Delta t)) ^ 2 • ch07_cov P (fun ω => z ω.2) (fun ω => z ω.2) := by
  ext i j
  simp only [ch07_cov, ch07_ou_obs, Matrix.of_apply, Matrix.add_apply, Matrix.smul_apply,
    smul_eq_mul]
  rw [Finset.mul_sum, Finset.mul_sum, Finset.mul_sum, Finset.mul_sum,
    ← Finset.sum_add_distrib, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun ω _ => by ring

/-- eq:lmmse-covs, derived from the channel of noname-13 rather than asserted:
for the OU channel `x = e^{-t}a + √Δ_t z` on a product (hence independent)
sample space with centred unit-covariance noise,
`Cov(a,x) = e^{-t}Σ₀` and `Cov(x) = e^{-2t}Σ₀ + Δ_t I`, in every dimension. -/
theorem ch07_lmmse_covs_channel {n : ℕ} {X Z : Type} [Fintype X] [Fintype Z]
    {t : ℝ} (ht : 0 ≤ t) (pa : X → ℝ) (pz : Z → ℝ)
    (hpa : ∑ x : X, pa x = 1) (hpz : ∑ w : Z, pz w = 1)
    (a : X → Fin n → ℝ) (z : Z → Fin n → ℝ)
    (hz0 : ∀ j, ∑ w : Z, pz w * z w j = 0)
    (hzI : ch07_cov pz z z = 1) :
    ch07_cov (fun ω : X × Z => pa ω.1 * pz ω.2) (fun ω => a ω.1) (ch07_ou_obs t a z)
        = Real.exp (-t) • ch07_cov pa a a
    ∧ ch07_cov (fun ω : X × Z => pa ω.1 * pz ω.2) (ch07_ou_obs t a z) (ch07_ou_obs t a z)
        = Real.exp (-2 * t) • ch07_cov pa a a
          + ch07_Delta t • (1 : Matrix (Fin n) (Fin n) ℝ) := by
  have hΔ : 0 ≤ ch07_Delta t := by
    unfold ch07_Delta
    have : Real.exp (-2 * t) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith)
    linarith
  have haa : ch07_cov (fun ω : X × Z => pa ω.1 * pz ω.2) (fun ω => a ω.1) (fun ω => a ω.1)
      = ch07_cov pa a a := ch07_cov_prod_fst pa pz hpz a a
  have hzz : ch07_cov (fun ω : X × Z => pa ω.1 * pz ω.2) (fun ω => z ω.2) (fun ω => z ω.2)
      = (1 : Matrix (Fin n) (Fin n) ℝ) := by
    rw [ch07_cov_prod_snd pa pz hpa z z, hzI]
  have haz : ch07_cov (fun ω : X × Z => pa ω.1 * pz ω.2) (fun ω => a ω.1) (fun ω => z ω.2)
      = 0 := ch07_cov_prod_cross pa pz a z hz0
  have hza : ch07_cov (fun ω : X × Z => pa ω.1 * pz ω.2) (fun ω => z ω.2) (fun ω => a ω.1)
      = 0 := by
    rw [← ch07_cov_transpose, haz, Matrix.transpose_zero]
  have hexp : (Real.exp (-t)) ^ 2 = Real.exp (-2 * t) := by
    rw [sq, ← Real.exp_add]; ring_nf
  have hsq : (Real.sqrt (ch07_Delta t)) ^ 2 = ch07_Delta t := Real.sq_sqrt hΔ
  constructor
  · rw [ch07_cov_channel_right, haa, haz, smul_zero, add_zero]
  · rw [ch07_cov_channel_both, haa, hzz, haz, hza, hexp, hsq]
    simp

/-! ### noname-3 (lines 169-188): the boxed closed form of `I(y)`, assembled

The record above proves five algebraic *ingredients* of the boxed formula (both
completions of the square, both `Φ` arguments, the prefactor) but never states
the formula itself: the half-line Gaussian mass and the summation of the two
branches were done on paper.  They are done in Lean here.  The missing lemma is
`ch07_gauss_halfline`, `∫_0^∞ N(r; c, σ²) dr = Φ(c/σ)`; with it the boxed
equation becomes one integral identity about the transition-times-likelihood
integral of noname-1, at arbitrary `b, μ > 0, Δ > 0` and arbitrary `α, a, x'`. -/

/-- Translation invariance of a half-line integral. -/
theorem ch07_integral_Ioi_sub (f : ℝ → ℝ) (c d : ℝ) :
    (∫ x in Set.Ioi c, f (x - d)) = ∫ x in Set.Ioi (c - d), f x := by
  rw [← MeasureTheory.integral_indicator measurableSet_Ioi,
    ← MeasureTheory.integral_indicator measurableSet_Ioi]
  have h : ∀ x : ℝ, Set.indicator (Set.Ioi c) (fun x => f (x - d)) x
      = Set.indicator (Set.Ioi (c - d)) f (x - d) := by
    intro x
    by_cases hx : x ∈ Set.Ioi c
    · have hx' : x - d ∈ Set.Ioi (c - d) := by
        simp only [Set.mem_Ioi] at hx ⊢; linarith
      simp [Set.indicator_apply, hx, hx']
    · have hx' : x - d ∉ Set.Ioi (c - d) := by
        simp only [Set.mem_Ioi] at hx ⊢
        intro hc; exact hx (by linarith)
      simp [Set.indicator_apply, hx, hx']
  simp_rw [h]
  exact integral_sub_right_eq_self (Set.indicator (Set.Ioi (c - d)) f) d

/-- `√(Δ/μ²) = √Δ/μ` for `μ > 0`. -/
theorem ch07_sqrt_div_sq {Δ μ : ℝ} (hΔ : 0 ≤ Δ) (hμ : 0 < μ) :
    Real.sqrt (Δ / μ ^ 2) = Real.sqrt Δ / μ := by
  rw [show Δ / μ ^ 2 = Δ * (1 / μ) ^ 2 by field_simp]
  rw [Real.sqrt_mul hΔ, Real.sqrt_sq (by positivity)]
  field_simp

/-- **The half-line Gaussian mass**, the lemma the audit was missing:
`∫_0^∞ exp(-(r-c)²/(2σ²)) dr = σ√(2π) Φ(c/σ)`. -/
theorem ch07_gauss_halfline {σ : ℝ} (hσ : 0 < σ) (c : ℝ) :
    (∫ r in Set.Ioi (0:ℝ), Real.exp (-(r - c) ^ 2 / (2 * σ ^ 2)))
      = σ * Real.sqrt (2 * π) * ch07_Phi (c / σ) := by
  have hσ' : σ ≠ 0 := ne_of_gt hσ
  have h2pi : (0:ℝ) < Real.sqrt (2 * π) := Real.sqrt_pos.mpr (by positivity)
  -- (i) rescale `r = σ x`
  have hscale : (∫ x in Set.Ioi (0:ℝ), Real.exp (-(σ * x - c) ^ 2 / (2 * σ ^ 2)))
      = σ⁻¹ * ∫ r in Set.Ioi (0:ℝ), Real.exp (-(r - c) ^ 2 / (2 * σ ^ 2)) := by
    have h := integral_comp_mul_left_Ioi
      (fun r : ℝ => Real.exp (-(r - c) ^ 2 / (2 * σ ^ 2))) 0 hσ
    simpa using h
  -- (ii) the rescaled integrand is the standard Gaussian shifted by `c/σ`
  have hform : ∀ x : ℝ,
      Real.exp (-(σ * x - c) ^ 2 / (2 * σ ^ 2)) = Real.exp (-(x - c / σ) ^ 2 / 2) := by
    intro x
    congr 1
    field_simp
  -- (iii) translate
  have htrans : (∫ x in Set.Ioi (0:ℝ), Real.exp (-(x - c / σ) ^ 2 / 2))
      = ∫ u in Set.Ioi (-(c / σ)), Real.exp (-u ^ 2 / 2) := by
    have h := ch07_integral_Ioi_sub (fun u : ℝ => Real.exp (-u ^ 2 / 2)) 0 (c / σ)
    simpa using h
  -- (iv) `Φ` as an upper-tail integral (the standard density is even)
  have hPhi : ch07_Phi (c / σ)
      = (1 / Real.sqrt (2 * π)) * ∫ u in Set.Ioi (-(c / σ)), Real.exp (-u ^ 2 / 2) := by
    unfold ch07_Phi
    rw [MeasureTheory.integral_const_mul]
    congr 1
    have h := integral_comp_neg_Iic (c / σ) (fun u : ℝ => Real.exp (-u ^ 2 / 2))
    simpa using h
  rw [hPhi, ← htrans]
  simp_rw [← hform]
  rw [hscale]
  field_simp

/-- The lower half-line mass, by reflection. -/
theorem ch07_gauss_halfline_Iic {σ : ℝ} (hσ : 0 < σ) (c : ℝ) :
    (∫ r in Set.Iic (0:ℝ), Real.exp (-(r - c) ^ 2 / (2 * σ ^ 2)))
      = σ * Real.sqrt (2 * π) * ch07_Phi (-c / σ) := by
  have key := integral_comp_neg_Iic (0:ℝ)
    (fun s : ℝ => Real.exp (-(s - -c) ^ 2 / (2 * σ ^ 2)))
  rw [neg_zero, ch07_gauss_halfline hσ (-c)] at key
  rw [← key]
  refine setIntegral_congr_fun measurableSet_Iic (fun r _ => ?_)
  congr 1
  ring

/-- The `r > 0` branch of the kinked integral, in closed form. -/
theorem ch07_remark_branch_pos {b μ Δ : ℝ} (hb : 0 < b) (hμ : 0 < μ) (hΔ : 0 < Δ) (y : ℝ) :
    (∫ r in Set.Ioi (0:ℝ), Real.exp (-r / b + -(y - μ * r) ^ 2 / (2 * Δ)))
      = Real.exp (Δ / (2 * (μ * b) ^ 2) - y / (μ * b))
        * (Real.sqrt Δ / μ * Real.sqrt (2 * π))
        * ch07_Phi (y / Real.sqrt Δ - Real.sqrt Δ / (μ * b)) := by
  have hσ : (0:ℝ) < Real.sqrt Δ / μ := by positivity
  have hpt : ∀ r : ℝ, Real.exp (-r / b + -(y - μ * r) ^ 2 / (2 * Δ))
      = Real.exp (Δ / (2 * (μ * b) ^ 2) - y / (μ * b))
        * Real.exp (-(r - (y - Δ / (μ * b)) / μ) ^ 2 / (2 * (Δ / μ ^ 2))) := by
    intro r
    rw [ch07_remark_square_pos (ne_of_gt hb) (ne_of_gt hμ) (ne_of_gt hΔ) y r, Real.exp_add]
  simp_rw [hpt]
  rw [MeasureTheory.integral_const_mul]
  rw [show (2 * (Δ / μ ^ 2)) = 2 * (Real.sqrt Δ / μ) ^ 2 by
    rw [div_pow, Real.sq_sqrt hΔ.le]]
  rw [ch07_gauss_halfline hσ ((y - Δ / (μ * b)) / μ)]
  have harg : ((y - Δ / (μ * b)) / μ) / (Real.sqrt Δ / μ)
      = y / Real.sqrt Δ - Real.sqrt Δ / (μ * b) := by
    rw [← ch07_sqrt_div_sq hΔ.le hμ]
    exact ch07_remark_Phi_arg_pos hμ hΔ b y
  rw [harg]
  ring

/-- The `r < 0` branch of the kinked integral, in closed form. -/
theorem ch07_remark_branch_neg {b μ Δ : ℝ} (hb : 0 < b) (hμ : 0 < μ) (hΔ : 0 < Δ) (y : ℝ) :
    (∫ r in Set.Iic (0:ℝ), Real.exp (r / b + -(y - μ * r) ^ 2 / (2 * Δ)))
      = Real.exp (Δ / (2 * (μ * b) ^ 2) + y / (μ * b))
        * (Real.sqrt Δ / μ * Real.sqrt (2 * π))
        * ch07_Phi (-(y / Real.sqrt Δ) - Real.sqrt Δ / (μ * b)) := by
  have hσ : (0:ℝ) < Real.sqrt Δ / μ := by positivity
  have hpt : ∀ r : ℝ, Real.exp (r / b + -(y - μ * r) ^ 2 / (2 * Δ))
      = Real.exp (Δ / (2 * (μ * b) ^ 2) + y / (μ * b))
        * Real.exp (-(r - (y + Δ / (μ * b)) / μ) ^ 2 / (2 * (Δ / μ ^ 2))) := by
    intro r
    rw [ch07_remark_square_neg (ne_of_gt hb) (ne_of_gt hμ) (ne_of_gt hΔ) y r, Real.exp_add]
  simp_rw [hpt]
  rw [MeasureTheory.integral_const_mul]
  rw [show (2 * (Δ / μ ^ 2)) = 2 * (Real.sqrt Δ / μ) ^ 2 by
    rw [div_pow, Real.sq_sqrt hΔ.le]]
  rw [ch07_gauss_halfline_Iic hσ ((y + Δ / (μ * b)) / μ)]
  have harg : (-((y + Δ / (μ * b)) / μ)) / (Real.sqrt Δ / μ)
      = -(y / Real.sqrt Δ) - Real.sqrt Δ / (μ * b) := by
    rw [← ch07_sqrt_div_sq hΔ.le hμ]
    exact ch07_remark_Phi_arg_neg hμ hΔ b y
  rw [harg]
  ring

/-- The kinked integrand is integrable: `-|r|/b ≤ 0`, so it is dominated by its
own Gaussian factor. -/
theorem ch07_remark_integrable {b μ Δ : ℝ} (hb : 0 < b) (hμ : 0 < μ) (hΔ : 0 < Δ) (y : ℝ) :
    Integrable (fun r : ℝ => Real.exp (-|r| / b + -(y - μ * r) ^ 2 / (2 * Δ))) := by
  have hg : Integrable (fun r : ℝ => Real.exp (-(y - μ * r) ^ 2 / (2 * Δ))) := by
    have h0 : Integrable (fun x : ℝ => Real.exp (-(μ ^ 2 / (2 * Δ)) * x ^ 2)) :=
      integrable_exp_neg_mul_sq (by positivity)
    have h1 := h0.comp_sub_right (y / μ)
    have hfun : (fun r : ℝ => Real.exp (-(μ ^ 2 / (2 * Δ)) * (r - y / μ) ^ 2))
        = fun r : ℝ => Real.exp (-(y - μ * r) ^ 2 / (2 * Δ)) := by
      funext r
      congr 1
      field_simp
      ring
    rwa [hfun] at h1
  refine hg.mono ?_ ?_
  · fun_prop
  · filter_upwards with r
    rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_of_pos (Real.exp_pos _),
      abs_of_pos (Real.exp_pos _), Real.exp_le_exp]
    have hr : -|r| / b ≤ 0 :=
      div_nonpos_of_nonpos_of_nonneg (neg_nonpos.mpr (abs_nonneg r)) hb.le
    linarith

/-- noname-3, in the convolution variable: the boxed closed form of `I(y)`. -/
theorem ch07_remark_I_closed_form_y {b μ Δ : ℝ} (hb : 0 < b) (hμ : 0 < μ) (hΔ : 0 < Δ)
    (y : ℝ) :
    (1 / (2 * b) * (1 / Real.sqrt (2 * π * Δ)))
        * ∫ r : ℝ, Real.exp (-|r| / b + -(y - μ * r) ^ 2 / (2 * Δ))
      = Real.exp (Δ / (2 * μ ^ 2 * b ^ 2)) / (2 * μ * b)
        * (Real.exp (-(y / (μ * b))) * ch07_Phi (y / Real.sqrt Δ - Real.sqrt Δ / (μ * b))
          + Real.exp (y / (μ * b))
              * ch07_Phi (-(y / Real.sqrt Δ) - Real.sqrt Δ / (μ * b))) := by
  have hint := ch07_remark_integrable hb hμ hΔ y
  rw [← intervalIntegral.integral_Iic_add_Ioi (b := (0:ℝ)) hint.integrableOn hint.integrableOn]
  have hIic : (∫ r in Set.Iic (0:ℝ), Real.exp (-|r| / b + -(y - μ * r) ^ 2 / (2 * Δ)))
      = ∫ r in Set.Iic (0:ℝ), Real.exp (r / b + -(y - μ * r) ^ 2 / (2 * Δ)) := by
    refine setIntegral_congr_fun measurableSet_Iic (fun r hr => ?_)
    rw [abs_of_nonpos (by exact hr)]
    congr 1
    ring
  have hIoi : (∫ r in Set.Ioi (0:ℝ), Real.exp (-|r| / b + -(y - μ * r) ^ 2 / (2 * Δ)))
      = ∫ r in Set.Ioi (0:ℝ), Real.exp (-r / b + -(y - μ * r) ^ 2 / (2 * Δ)) := by
    refine setIntegral_congr_fun measurableSet_Ioi (fun r hr => ?_)
    rw [abs_of_nonneg (le_of_lt (by exact hr))]
  rw [hIic, hIoi, ch07_remark_branch_neg hb hμ hΔ y, ch07_remark_branch_pos hb hμ hΔ y]
  have hsplit : Real.sqrt (2 * π * Δ) = Real.sqrt (2 * π) * Real.sqrt Δ :=
    Real.sqrt_mul (by positivity) Δ
  have h2pi : (0:ℝ) < Real.sqrt (2 * π) := Real.sqrt_pos.mpr (by positivity)
  have hsΔ : (0:ℝ) < Real.sqrt Δ := Real.sqrt_pos.mpr hΔ
  have hea : Real.exp (Δ / (2 * (μ * b) ^ 2) - y / (μ * b))
      = Real.exp (Δ / (2 * μ ^ 2 * b ^ 2)) * Real.exp (-(y / (μ * b))) := by
    rw [← Real.exp_add]
    congr 1
    ring
  have heb : Real.exp (Δ / (2 * (μ * b) ^ 2) + y / (μ * b))
      = Real.exp (Δ / (2 * μ ^ 2 * b ^ 2)) * Real.exp (y / (μ * b)) := by
    rw [← Real.exp_add]
    congr 1
    ring
  rw [hea, heb, hsplit]
  field_simp
  ring

/-- noname-3, as the chapter states it: the boxed closed form of
`I(y) = ∫ M(a'|a) N(x'; μa', Δ_t) da'` with `y = x' - μαa`, for `μ > 0`.  The
five ingredient lemmas above are what this proof consumes, and the half-line
Gaussian mass supplies the step that used to be done by hand. -/
theorem ch07_remark_I_closed_form {b μ Δ : ℝ} (hb : 0 < b) (hμ : 0 < μ) (hΔ : 0 < Δ)
    (α a x' : ℝ) :
    (∫ a' : ℝ, ch07_M α b a a' * ch07_obs μ Δ a' x')
      = Real.exp (Δ / (2 * μ ^ 2 * b ^ 2)) / (2 * μ * b)
        * (Real.exp (-((x' - μ * (α * a)) / (μ * b)))
              * ch07_Phi ((x' - μ * (α * a)) / Real.sqrt Δ - Real.sqrt Δ / (μ * b))
          + Real.exp ((x' - μ * (α * a)) / (μ * b))
              * ch07_Phi (-((x' - μ * (α * a)) / Real.sqrt Δ) - Real.sqrt Δ / (μ * b))) := by
  rw [ch07_remark_change_of_var]
  exact ch07_remark_I_closed_form_y hb hμ hΔ (x' - μ * (α * a))

/-! ### eq:l-grid-fwd / eq:l-grid-bwd (lines 284-293) with the chapter's own
kernel

`ch07_l_grid_fwd` / `ch07_l_grid_bwd` above are stated for an ARBITRARY matrix
`K`, so the mirrored statement with the transpose dropped is equally provable by
the same one-line proof, and the check has no discriminating power.  The lemmas
below fix exactly that.  `K` is instantiated at the chapter's own grid kernel
(`K_{ij} = M(g_i | g_j)`, `ch07_K`), `w` at the trapezoid weights and `g` at the
grid, so each sweep is identified with an explicit weighted sum against
`ch07_M` with its source and destination arguments in a definite order: the
forward sweep sums over the SOURCE, the backward one over the DESTINATION, which
is what the transpose in eq:l-grid-bwd does.  `ch07_M_not_symm` then exhibits a
parameter point where `M(a'|a) ≠ M(a|a')`, so the orientation is not a vacuous
choice and dropping the transpose really does change the sweep.

What is NOT claimed here: these are exact finite identities about the
discretised sweep.  No quadrature error bound against the continuous recursion
of `prop:bp-convA` is proved anywhere in this file. -/

/-- The grid transition matrix of eq:l-grid-operators, as a matrix over `Fin N`. -/
def ch07_Kmat (α b A : ℝ) (N : ℕ) : Matrix (Fin N) (Fin N) ℝ :=
  Matrix.of fun i j => ch07_K α b (ch07_g A N) i j

/-- eq:l-grid-fwd with the chapter's own kernel, weights and grid: the forward
sweep is the trapezoid sum against `M(g_i | g_j)`, the summation index `j` being
the SOURCE state. -/
theorem ch07_l_grid_fwd_kernel {N : ℕ} (α b μ Δ xk A : ℝ) (f : Fin N → ℝ) (i : Fin N) :
    ch07_ell μ Δ xk (ch07_g A N) i
        * ((ch07_Kmat α b A N).mulVec (fun j => ch07_w A N j * f j)) i
      = ch07_ell μ Δ xk (ch07_g A N) i
        * ∑ j : Fin N, ch07_w A N j * ch07_M α b (ch07_g A N j) (ch07_g A N i) * f j := by
  simp only [ch07_Kmat, Matrix.mulVec, dotProduct, Matrix.of_apply, ch07_K_eq_M]
  congr 1
  exact Finset.sum_congr rfl fun j _ => by ring

/-- eq:l-grid-bwd with the same instantiation: the backward sweep sums over the
DESTINATION state `g_j`, the free index `i` being the source.  This is what the
transpose buys. -/
theorem ch07_l_grid_bwd_kernel {N : ℕ} (α b μ Δ xk A : ℝ) (beta : Fin N → ℝ) (i : Fin N) :
    ((ch07_Kmat α b A N)ᵀ.mulVec
        (fun j => ch07_w A N j * (ch07_ell μ Δ xk (ch07_g A N) j * beta j))) i
      = ∑ j : Fin N, ch07_w A N j * ch07_M α b (ch07_g A N i) (ch07_g A N j)
          * (ch07_ell μ Δ xk (ch07_g A N) j * beta j) := by
  simp only [ch07_Kmat, Matrix.mulVec, dotProduct, Matrix.transpose_apply, Matrix.of_apply,
    ch07_K_eq_M]
  exact Finset.sum_congr rfl fun j _ => by ring

/-- The Laplace kernel is genuinely non-symmetric: at `α = 1/2`, `b = 1`,
`M(1 | 0) = e^{-1}/2` while `M(0 | 1) = e^{-1/2}/2`.  Hence the orientation
fixed by the two lemmas above is a real constraint, and the version of
eq:l-grid-bwd with the transpose dropped is a different recursion. -/
theorem ch07_M_not_symm : ch07_M (1/2) 1 0 1 ≠ ch07_M (1/2) 1 1 0 := by
  have h1 : ch07_M (1/2) 1 0 1 = (1/2) * Real.exp (-1) := by
    unfold ch07_M ch07_laplacePdf
    norm_num
  have h2 : ch07_M (1/2) 1 1 0 = (1/2) * Real.exp (-(1/2)) := by
    unfold ch07_M ch07_laplacePdf
    rw [show (0:ℝ) - 1/2 * 1 = -(1/2) by norm_num, abs_neg,
      abs_of_nonneg (by norm_num : (0:ℝ) ≤ 1/2)]
    norm_num
  rw [h1, h2]
  intro hc
  have hlt : Real.exp (-1) < Real.exp (-(1/2)) := Real.exp_lt_exp.mpr (by norm_num)
  linarith

/-! ### Audit checks on the two generalisations above

A general statement is only worth its definitions, so the block below ties the
new objects back to the concrete ones they replace: the `m = n = 1` case of the
chain joint and of the two messages is *literally* the `L = 3` summand and the
two message sums of `ch07_l_markov_score_num_gen`, and the hypotheses of
`ch07_lmmse_covs_channel` are exhibited as satisfiable, so that theorem is not
vacuously true. -/

section MarkovScoreCheck

variable {A : Type} [Fintype A]

/-- At `m = n = 1` the general chain joint is exactly the `L = 3` summand
`p₀(a₀) ob₀(a₀) M(a₁|a₀) ob₁(a₁) M(a₂|a₁) ob₂(a₂)`. -/
theorem ch07_chain_joint_three (p0 obC : A → ℝ) (obL obR : ℕ → A → ℝ) (M : A → A → ℝ)
    (u : Fin 1 → A) (c : A) (v : Fin 1 → A) :
    ch07_preW p0 M obL 1 c u * obC c * ch07_sufW M obR 1 c v
      = p0 (u 0) * obL 0 (u 0) * M c (u 0) * obC c * M (v 0) c * obR 0 (v 0) := by
  simp only [ch07_preW, ch07_sufW]
  ring

/-- At `m = 1` the forward recursion is the `L = 3` forward message. -/
theorem ch07_alphaMsg_one (p0 : A → ℝ) (M : A → A → ℝ) (obL : ℕ → A → ℝ) (c : A) :
    ch07_alphaMsg p0 M obL 1 c = ∑ a : A, M c a * obL 0 a * p0 a := by
  simp only [ch07_alphaMsg]

/-- At `n = 1` the backward recursion is the `L = 3` backward message. -/
theorem ch07_betaMsg_one (M : A → A → ℝ) (obR : ℕ → A → ℝ) (c : A) :
    ch07_betaMsg M obR 1 c = ∑ a : A, M a c * obR 0 a := by
  simp only [ch07_betaMsg, mul_one]

end MarkovScoreCheck

/-- A fair sign flip: the witness noise for the channel hypotheses. -/
def ch07_fair : Bool → ℝ := fun _ => 1 / 2

/-- The same, as a one-dimensional random vector. -/
def ch07_radem : Bool → Fin 1 → ℝ := fun w _ => if w then 1 else -1

/-- The hypotheses of `ch07_lmmse_covs_channel` are satisfiable -- a fair sign
flip is a centred, unit-covariance noise on a two-point space -- so that theorem
is not vacuously true. -/
theorem ch07_ou_noise_witness :
    (∑ w : Bool, ch07_fair w) = 1
    ∧ (∀ j : Fin 1, ∑ w : Bool, ch07_fair w * ch07_radem w j = 0)
    ∧ ch07_cov ch07_fair ch07_radem ch07_radem = 1 := by
  refine ⟨?_, ?_, ?_⟩
  · norm_num [ch07_fair, Fintype.sum_bool]
  · intro j
    norm_num [ch07_fair, ch07_radem, Fintype.sum_bool]
  · ext i j
    fin_cases i
    fin_cases j
    norm_num [ch07_cov, ch07_fair, ch07_radem, Fintype.sum_bool, Matrix.one_apply]

end

end ThesisAudit
