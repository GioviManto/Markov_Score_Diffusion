import Mathlib

/-!
# Audit of ch08-em-parameters.tex, lines 1-600

Formal check of every display-math formula in Sections 8.1-8.2 of the thesis
chapter "Learning the Parameters: EM on the Chain".

Modelling convention used throughout the EM part: the latent variable
`z = (a,c)` ranges over a *finite* type `Z`, so that the marginalisation
`\sum_c \int da` of the chapter becomes `\sum_{z : Z}`.  This is exactly the
discretisation the chapter itself introduces numerically (the grid), and it is
the setting in which the EM decomposition, Gibbs' inequality and Fisher's
identity are genuine theorems rather than statements needing dominated
convergence.  Where the chapter's claim is an interchange of limits
(differentiation under the integral sign) the finite version is proved and the
item is flagged accordingly.
-/

namespace ThesisAudit

open Finset Real

noncomputable section

/-! ## 8.1 Model: an estimated mixture innovation -/

-- eq:em-chain (ch08-em-parameters.tex lines 23-27):
-- a_1 ~ N(0,1), a_i = α a_{i-1} + ε_i with Var(ε) = 1 - α².
/-- One step of the clean AR(1) chain. -/
def ch08a_em_chain_next (α aprev ε : ℝ) : ℝ := α * aprev + ε

/-- Variance matching: with `Var(a_{i-1}) = 1` and `Var(ε_i) = 1 - α²` the chain is
stationary with unit variance, which is the normalisation the chapter asserts. -/
theorem ch08a_em_chain (α : ℝ) : α ^ 2 * 1 + (1 - α ^ 2) = 1 := by ring

/-- The chain step is affine in the previous state and the innovation. -/
theorem ch08a_em_chain_affine (α aprev ε : ℝ) :
    ch08a_em_chain_next α aprev ε = α * aprev + ε := rfl

-- eq:em-mixture-kernel (lines 39-44):
-- K_θ(a' | a) = Σ_c π_c N(a' - α a ; ν_c, s_c²).
/-- Gaussian density `N(u ; ν, s²)`. -/
def ch08a_gauss (ν s u : ℝ) : ℝ :=
  Real.exp (-((u - ν) ^ 2) / (2 * s ^ 2)) / Real.sqrt (2 * Real.pi * s ^ 2)

/-- The `C`-component Gaussian-mixture transition kernel. -/
def ch08a_em_mixture_kernel {C : ℕ} (α : ℝ) (p ν s : Fin C → ℝ) (a' a : ℝ) : ℝ :=
  ∑ c : Fin C, p c * ch08a_gauss (ν c) (s c) (a' - α * a)

theorem ch08a_gauss_nonneg (ν s u : ℝ) : 0 ≤ ch08a_gauss ν s u := by
  unfold ch08a_gauss; positivity

theorem ch08a_gauss_pos (ν u : ℝ) {s : ℝ} (hs : 0 < s) : 0 < ch08a_gauss ν s u := by
  have hsq : (0 : ℝ) < s ^ 2 := pow_pos hs 2
  have hprod : (0 : ℝ) < 2 * Real.pi * s ^ 2 :=
    mul_pos (by positivity) hsq
  unfold ch08a_gauss
  exact div_pos (Real.exp_pos _) (Real.sqrt_pos.mpr hprod)

theorem ch08a_em_mixture_kernel_nonneg {C : ℕ} (α : ℝ) (p ν s : Fin C → ℝ)
    (hp : ∀ c, 0 ≤ p c) (a' a : ℝ) : 0 ≤ ch08a_em_mixture_kernel α p ν s a' a :=
  Finset.sum_nonneg fun c _ => mul_nonneg (hp c) (ch08a_gauss_nonneg _ _ _)

/-- "At `C = 1` this collapses to a single Gaussian" (chapter line 69). -/
theorem ch08a_em_mixture_kernel_one (α : ℝ) (p ν s : Fin 1 → ℝ) (hp : p 0 = 1) (a' a : ℝ) :
    ch08a_em_mixture_kernel α p ν s a' a = ch08a_gauss (ν 0) (s 0) (a' - α * a) := by
  simp [ch08a_em_mixture_kernel, hp]

-- eq:em-parameter-space (lines 48-53):
-- Θ = { π_c ≥ 0, Σ π_c = 1, s_c² ≥ s_min², ν_c ∈ ℝ, α ∈ (-1,1) }.
/-- The admissible parameter set, as a subset of `(π, ν, s, α)`-space. -/
def ch08a_em_parameter_space (C : ℕ) (smin : ℝ) :
    Set ((Fin C → ℝ) × (Fin C → ℝ) × (Fin C → ℝ) × ℝ) :=
  {θ | (∀ c, 0 ≤ θ.1 c) ∧ (∑ c, θ.1 c) = 1 ∧
       (∀ c, smin ^ 2 ≤ (θ.2.2.1 c) ^ 2) ∧ θ.2.2.2 ∈ Set.Ioo (-1 : ℝ) 1}

/-- Sanity: a single centred unit-variance component with `α = 0` is admissible. -/
theorem ch08a_em_parameter_space_nonempty :
    (fun _ => (1 : ℝ), fun _ => (0 : ℝ), fun _ => (1 : ℝ), (0 : ℝ))
      ∈ ch08a_em_parameter_space 1 1 := by
  refine ⟨fun c => by norm_num, by simp, fun c => by norm_num, by norm_num⟩

/-- Stationarity of the variance recursion `v = α² v + Var(ε)`: the fixed point is
`Var(ε)/(1-α²)`, i.e. the *innovation variance* divided by `1-α²`. -/
theorem ch08a_stationary_variance (α ve : ℝ) (hα : (1 : ℝ) - α ^ 2 ≠ 0) :
    ve / (1 - α ^ 2) = α ^ 2 * (ve / (1 - α ^ 2)) + ve := by
  field_simp; ring

/-- For a mixture innovation, `Var(ε) = Σ π_c (ν_c² + s_c²) - (Σ π_c ν_c)²`, which is
strictly smaller than the second moment `Σ π_c (ν_c² + s_c²)` as soon as the innovation
mean `Σ π_c ν_c` is nonzero.  Concretely, with `C = 1`, `π = 1`, `ν = 1`, `s = 1`,
`α = 0` the stationary variance is `1`, while `Σ π_c(ν_c²+s_c²)/(1-α²) = 2`. -/
theorem ch08a_stationary_variance_not_second_moment :
    ((1 : ℝ) * (1 ^ 2 + 1 ^ 2)) / (1 - (0 : ℝ) ^ 2)
      ≠ ((1 : ℝ) * (1 ^ 2 + 1 ^ 2) - (1 * 1) ^ 2) / (1 - (0 : ℝ) ^ 2) := by
  norm_num

/-- Free-parameter count of the family (chapter line 84, inline):
`(C-1) + C + C + 1 = 3C`. -/
theorem ch08a_free_parameter_count (C : ℕ) (hC : 1 ≤ C) : (C - 1) + C + C + 1 = 3 * C := by
  omega

/-! ## 8.2 The EM algorithm, derived

Finite latent space `Z` standing for `(a,c)`; `P z` is `p_θ(z, x)` at a fixed
observation `x`. -/

/-- `p_θ(x) = Σ_z p_θ(z,x)`. -/
def ch08a_marg {Z : Type*} [Fintype Z] (P : Z → ℝ) : ℝ := ∑ z, P z

/-- `p_θ(z | x) = p_θ(z,x) / p_θ(x)`. -/
def ch08a_post {Z : Type*} [Fintype Z] (P : Z → ℝ) (z : Z) : ℝ := P z / ch08a_marg P

/-- `L(θ) = log p_θ(x)`. -/
def ch08a_L {Z : Type*} [Fintype Z] (P : Z → ℝ) : ℝ := Real.log (ch08a_marg P)

-- noname-1 (lines 91-93): x = (x_1, ..., x_N), the observed noised sequence.
/-- The observed sequence, as a tuple of length `N`. -/
def ch08a_obs_seq (N : ℕ) : Type := Fin N → ℝ

-- noname-2 (lines 95-97): a = (a_1, ..., a_N), the latent clean trajectory.
/-- The latent clean trajectory, as a tuple of length `N`. -/
def ch08a_clean_seq (N : ℕ) : Type := Fin N → ℝ

-- noname-3 (lines 103-105): c = (c_2, ..., c_N), one label per transition,
-- so N-1 of them.
/-- There are exactly `N - 1` transition labels `c_2, ..., c_N`. -/
theorem ch08a_labels_card (N : ℕ) : (Finset.Icc 2 N).card = N - 1 := by
  rw [Nat.card_Icc]; omega

-- eq:em-objective (lines 110-119):
-- θ̂ ∈ argmax_θ L(θ), L(θ) := log p_θ(x).
/-- The maximum-likelihood objective is `L(θ) = log p_θ(x)`; a maximiser is a point of
the `argmax`. -/
theorem ch08a_em_objective {Θ Z : Type*} [Fintype Z] (P : Θ → Z → ℝ) (θhat : Θ)
    (h : ∀ θ, ch08a_L (P θ) ≤ ch08a_L (P θhat)) :
    θhat ∈ {θ | ∀ θ', ch08a_L (P θ') ≤ ch08a_L (P θ)} := h

theorem ch08a_em_objective_def {Z : Type*} [Fintype Z] (P : Z → ℝ) :
    ch08a_L P = Real.log (ch08a_marg P) := rfl

-- eq:em-marginalisation (lines 122-129):
-- p_θ(x) = Σ_c ∫_{ℝ^N} p_θ(a,c,x) da.
theorem ch08a_em_marginalisation {Z : Type*} [Fintype Z] (P : Z → ℝ) :
    ch08a_marg P = ∑ z, P z := rfl

/-- The latent variable really is a pair `(a,c)`, and the marginalisation splits as a
sum over labels of a sum (the discrete analogue of the integral) over trajectories. -/
theorem ch08a_em_marginalisation_pair {A C : Type*} [Fintype A] [Fintype C]
    (P : A × C → ℝ) : ch08a_marg P = ∑ c : C, ∑ a : A, P (a, c) := by
  rw [ch08a_marg, Fintype.sum_prod_type]
  exact Finset.sum_comm

-- eq:em-marginal-likelihood (lines 131-141):
-- L(θ) = log [ Σ_c ∫ p_θ(a,c,x) da ].
theorem ch08a_em_marginal_likelihood {A C : Type*} [Fintype A] [Fintype C]
    (P : A × C → ℝ) : ch08a_L P = Real.log (∑ c : C, ∑ a : A, P (a, c)) := by
  rw [ch08a_L, ch08a_em_marginalisation_pair]

-- noname-4 (lines 163-165): z = (a,c).
/-- Collecting the latent variables. -/
def ch08a_z {A C : Type*} (a : A) (c : C) : A × C := (a, c)

-- eq:em-q-def (lines 167-172): q^(k)(a,c) := p_{θ^(k)}(a,c | x).
/-- The E-step distribution is the posterior at the current iterate. -/
theorem ch08a_em_q_def {Z : Type*} [Fintype Z] (P0 : Z → ℝ) (z : Z) :
    ch08a_post P0 z = P0 z / ch08a_marg P0 := rfl

/-- The posterior is a probability vector whenever the marginal is nonzero. -/
theorem ch08a_post_sum_one {Z : Type*} [Fintype Z] (P : Z → ℝ)
    (hm : ch08a_marg P ≠ 0) : ∑ z, ch08a_post P z = 1 := by
  unfold ch08a_post
  rw [← Finset.sum_div]
  exact div_self hm

theorem ch08a_post_pos {Z : Type*} [Fintype Z] (P : Z → ℝ) (hP : ∀ z, 0 < P z)
    (hm : 0 < ch08a_marg P) (z : Z) : 0 < ch08a_post P z := div_pos (hP z) hm

-- eq:em-Q-def (lines 177-185): Q(θ | θ^(k)) := E_{q^(k)}[ log p_θ(a,c,x) ].
/-- The EM surrogate. -/
def ch08a_Q {Z : Type*} [Fintype Z] (q P : Z → ℝ) : ℝ := ∑ z, q z * Real.log (P z)

-- noname-5 (lines 194-198): Bayes' rule p_θ(a,c|x) = p_θ(a,c,x)/p_θ(x).
theorem ch08a_bayes {Z : Type*} [Fintype Z] (P : Z → ℝ) (z : Z) :
    ch08a_post P z = P z / ch08a_marg P := rfl

-- eq:em-bayes-log (lines 200-207):
-- log p_θ(x) = log p_θ(a,c,x) - log p_θ(a,c|x).
theorem ch08a_em_bayes_log {Z : Type*} [Fintype Z] (P : Z → ℝ) (hP : ∀ z, 0 < P z)
    (hm : 0 < ch08a_marg P) (z : Z) :
    Real.log (ch08a_marg P) = Real.log (P z) - Real.log (ch08a_post P z) := by
  rw [ch08a_post, Real.log_div (hP z).ne' hm.ne']
  ring

-- eq:em-first-decomposition (lines 210-231):
-- L(θ) = E_q[log p_θ(a,c,x)] - E_q[log p_θ(a,c|x)] = Q(θ|θ^(k)) - E_q[log p_θ(a,c|x)].
theorem ch08a_em_first_decomposition {Z : Type*} [Fintype Z] (P q : Z → ℝ)
    (hq : ∑ z, q z = 1) (hP : ∀ z, 0 < P z) (hm : 0 < ch08a_marg P) :
    ch08a_L P = ch08a_Q q P - ∑ z, q z * Real.log (ch08a_post P z) := by
  have h1 : ∑ z, q z * Real.log (ch08a_marg P) = ch08a_L P := by
    rw [← Finset.sum_mul, hq, one_mul, ch08a_L]
  have h2 : ∀ z ∈ (Finset.univ : Finset Z),
      q z * Real.log (ch08a_marg P)
        = q z * Real.log (P z) - q z * Real.log (ch08a_post P z) := by
    intro z _
    rw [ch08a_em_bayes_log P hP hm z]; ring
  calc ch08a_L P = ∑ z, q z * Real.log (ch08a_marg P) := h1.symm
    _ = ∑ z, (q z * Real.log (P z) - q z * Real.log (ch08a_post P z)) :=
        Finset.sum_congr rfl h2
    _ = ch08a_Q q P - ∑ z, q z * Real.log (ch08a_post P z) := by
        rw [Finset.sum_sub_distrib, ch08a_Q]

-- noname-6 (lines 235-239): H(q) := -E_q[log q].
/-- Shannon entropy of a (finite) probability vector. -/
def ch08a_H {Z : Type*} [Fintype Z] (q : Z → ℝ) : ℝ := -∑ z, q z * Real.log (q z)

-- noname-7 (lines 241-247): KL(q || p) := E_q[ log (q/p) ].
/-- Kullback-Leibler divergence. -/
def ch08a_KL {Z : Type*} [Fintype Z] (q p : Z → ℝ) : ℝ := ∑ z, q z * Real.log (q z / p z)

-- eq:em-decomposition (lines 249-264):
-- L(θ) = Q(θ|θ^(k)) + H(q^(k)) + KL(q^(k) || p_θ(a,c|x)).
theorem ch08a_em_decomposition {Z : Type*} [Fintype Z] (P q : Z → ℝ)
    (hq1 : ∑ z, q z = 1) (hqpos : ∀ z, 0 < q z) (hP : ∀ z, 0 < P z)
    (hm : 0 < ch08a_marg P) :
    ch08a_L P = ch08a_Q q P + ch08a_H q + ch08a_KL q (ch08a_post P) := by
  have hpost : ∀ z, 0 < ch08a_post P z := ch08a_post_pos P hP hm
  have hKL : ch08a_KL q (ch08a_post P)
      = (∑ z, q z * Real.log (q z)) - ∑ z, q z * Real.log (ch08a_post P z) := by
    rw [ch08a_KL, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun z _ => ?_
    rw [Real.log_div (hqpos z).ne' (hpost z).ne']
    ring
  rw [ch08a_em_first_decomposition P q hq1 hP hm, hKL, ch08a_H]
  ring

/-- Gibbs' inequality (finite form): the KL divergence between two strictly positive
probability vectors is nonnegative. -/
theorem ch08a_kl_nonneg {Z : Type*} [Fintype Z] (q p : Z → ℝ)
    (hq : ∀ z, 0 < q z) (hp : ∀ z, 0 < p z)
    (hq1 : ∑ z, q z = 1) (hp1 : ∑ z, p z = 1) : 0 ≤ ch08a_KL q p := by
  have key : ∀ z ∈ (Finset.univ : Finset Z),
      q z * Real.log (p z / q z) ≤ p z - q z := by
    intro z _
    have h : Real.log (p z / q z) ≤ p z / q z - 1 :=
      Real.log_le_sub_one_of_pos (div_pos (hp z) (hq z))
    have h2 := mul_le_mul_of_nonneg_left h (hq z).le
    have hqz : q z ≠ 0 := (hq z).ne'
    have h3 : q z * (p z / q z - 1) = p z - q z := by
      field_simp
    linarith [h2, h3.le, h3.ge]
  have hsum := Finset.sum_le_sum key
  have hrw : ∑ z, q z * Real.log (p z / q z) = -ch08a_KL q p := by
    rw [ch08a_KL, ← Finset.sum_neg_distrib]
    refine Finset.sum_congr rfl fun z _ => ?_
    have hlog : Real.log (p z / q z) = -Real.log (q z / p z) := by
      rw [← Real.log_inv, inv_div]
    rw [hlog]; ring
  have h2 : ∑ z : Z, (p z - q z) = 0 := by
    rw [Finset.sum_sub_distrib, hp1, hq1, sub_self]
  rw [hrw, h2] at hsum
  linarith

-- eq:em-lower-bound (lines 266-283):
-- F(θ;q^(k)) := Q + H = L(θ) - KL(q^(k) || p_θ(a,c|x)) ≤ L(θ).
/-- The evidence lower bound `F(θ; q) = Q + H`. -/
def ch08a_F {Z : Type*} [Fintype Z] (q P : Z → ℝ) : ℝ := ch08a_Q q P + ch08a_H q

theorem ch08a_em_lower_bound_eq {Z : Type*} [Fintype Z] (P q : Z → ℝ)
    (hq1 : ∑ z, q z = 1) (hqpos : ∀ z, 0 < q z) (hP : ∀ z, 0 < P z)
    (hm : 0 < ch08a_marg P) :
    ch08a_F q P = ch08a_L P - ch08a_KL q (ch08a_post P) := by
  rw [ch08a_F, ch08a_em_decomposition P q hq1 hqpos hP hm]
  ring

theorem ch08a_em_lower_bound {Z : Type*} [Fintype Z] (P q : Z → ℝ)
    (hq1 : ∑ z, q z = 1) (hqpos : ∀ z, 0 < q z) (hP : ∀ z, 0 < P z)
    (hm : 0 < ch08a_marg P) :
    ch08a_F q P ≤ ch08a_L P := by
  have hpost : ∀ z, 0 < ch08a_post P z := ch08a_post_pos P hP hm
  have hp1 : ∑ z, ch08a_post P z = 1 := ch08a_post_sum_one P hm.ne'
  have := ch08a_kl_nonneg q (ch08a_post P) hqpos hpost hq1 hp1
  rw [ch08a_em_lower_bound_eq P q hq1 hqpos hP hm]
  linarith

/-- During the M-step the entropy `H(q)` is a constant, so maximising `F` is the same
as maximising `Q`. -/
theorem ch08a_em_F_argmax_eq_Q_argmax {Z : Type*} [Fintype Z] (q P₁ P₂ : Z → ℝ) :
    ch08a_F q P₁ ≤ ch08a_F q P₂ ↔ ch08a_Q q P₁ ≤ ch08a_Q q P₂ := by
  rw [ch08a_F, ch08a_F]
  constructor <;> intro h <;> linarith

-- noname-8 (lines 296-300): q^(k)(a,c) = p_{θ^(k)}(a,c|x) at the current iterate.
theorem ch08a_q_at_current {Z : Type*} [Fintype Z] (P0 : Z → ℝ) :
    ch08a_post P0 = fun z => P0 z / ch08a_marg P0 := rfl

-- noname-9 (lines 302-309): KL(q^(k) || p_{θ^(k)}(a,c|x)) = 0.
theorem ch08a_kl_self_zero {Z : Type*} [Fintype Z] (q : Z → ℝ) (hq : ∀ z, 0 < q z) :
    ch08a_KL q q = 0 := by
  rw [ch08a_KL]
  refine Finset.sum_eq_zero fun z _ => ?_
  rw [div_self (hq z).ne', Real.log_one, mul_zero]

-- eq:em-touching (lines 312-317): F(θ^(k); q^(k)) = L(θ^(k)).
theorem ch08a_em_touching {Z : Type*} [Fintype Z] (P0 : Z → ℝ) (hP : ∀ z, 0 < P0 z)
    (hm : 0 < ch08a_marg P0) :
    ch08a_F (ch08a_post P0) P0 = ch08a_L P0 := by
  have hqpos : ∀ z, 0 < ch08a_post P0 z := ch08a_post_pos P0 hP hm
  have hq1 : ∑ z, ch08a_post P0 z = 1 := ch08a_post_sum_one P0 hm.ne'
  rw [ch08a_em_lower_bound_eq P0 (ch08a_post P0) hq1 hqpos hP hm,
      ch08a_kl_self_zero (ch08a_post P0) hqpos, sub_zero]

/-! ### noname-10 (lines 319-327): the EM iteration picture

    θ^(k)  ⟶  q^(k)(a,c)  ⟶  Q(θ | θ^(k))  ⟶  θ^(k+1)

The display is a picture, not an identity.  What it asserts is that one EM sweep
is the *composite* of three maps, and in particular that the surrogate sees the
current iterate only through the posterior `q^(k)`: there is no direct arrow
`θ^(k) → Q`.  Each arrow is given below as an explicit function, that
factorisation is proved, and the ascent property which the surrounding
"therefore" appeals to -- going once round the picture cannot decrease
`L` -- is proved for an arbitrary finite latent space `Z` and an arbitrary
parameter set `Θ`, under a generalised-EM hypothesis (the M-step need only not
lose ground on `Q`).  Existence of an M-step output is assumed, not proved. -/

/-- `p_θ(x) > 0` as soon as every complete-data weight is positive. -/
theorem ch08a_marg_pos {Z : Type*} [Fintype Z] [Nonempty Z] (P : Z → ℝ)
    (hP : ∀ z, 0 < P z) : 0 < ch08a_marg P := by
  rw [ch08a_marg]
  exact Finset.sum_pos (fun z _ => hP z) Finset.univ_nonempty

/-- First arrow (E-step): `θ^(k) ↦ q^(k) = p_{θ^(k)}(a,c | x)`. -/
def ch08a_em_estep {Θ Z : Type*} [Fintype Z] (P : Θ → Z → ℝ) (θk : Θ) : Z → ℝ :=
  ch08a_post (P θk)

/-- Second arrow: the posterior becomes the weighted complete-data log-likelihood
`Q(θ | θ^(k)) = E_{q^(k)}[ log p_θ(a,c,x) ]`. -/
def ch08a_em_surrogate {Θ Z : Type*} [Fintype Z] (P : Θ → Z → ℝ) (θk θ : Θ) : ℝ :=
  ch08a_Q (ch08a_em_estep P θk) (P θ)

/-- Third arrow (M-step): `θ^(k+1)` maximises the surrogate. -/
def ch08a_em_mstep {Θ Z : Type*} [Fintype Z] (P : Θ → Z → ℝ) (θk θnext : Θ) : Prop :=
  ∀ θ, ch08a_em_surrogate P θk θ ≤ ch08a_em_surrogate P θk θnext

/-- The picture read literally: one sweep is the composite of the three arrows. -/
theorem ch08a_em_diagram {Θ Z : Type*} [Fintype Z] (P : Θ → Z → ℝ) (θk θ : Θ) :
    ch08a_em_surrogate P θk θ = ch08a_Q (ch08a_post (P θk)) (P θ) := rfl

/-- The first arrow really lands in the simplex: `q^(k)` is a strictly positive
probability vector on the latent space. -/
theorem ch08a_em_estep_is_distribution {Θ Z : Type*} [Fintype Z] [Nonempty Z]
    (P : Θ → Z → ℝ) (θk : Θ) (hk : ∀ z, 0 < P θk z) :
    (∀ z, 0 < ch08a_em_estep P θk z) ∧ ∑ z, ch08a_em_estep P θk z = 1 := by
  unfold ch08a_em_estep
  exact ⟨ch08a_post_pos (P θk) hk (ch08a_marg_pos _ hk),
         ch08a_post_sum_one (P θk) (ch08a_marg_pos _ hk).ne'⟩

/-- There is no direct `θ^(k) → Q` arrow: two parameter values with the same
E-step output give the same surrogate. -/
theorem ch08a_em_surrogate_factors_through_estep {Θ Z : Type*} [Fintype Z]
    (P : Θ → Z → ℝ) (θ₁ θ₂ : Θ) (h : ch08a_em_estep P θ₁ = ch08a_em_estep P θ₂)
    (θ : Θ) : ch08a_em_surrogate P θ₁ θ = ch08a_em_surrogate P θ₂ θ := by
  unfold ch08a_em_surrogate
  rw [h]

/-- Going once round the picture cannot decrease the marginal log-likelihood:
for an arbitrary finite latent space and an arbitrary parameter set, if `θ^(k+1)`
does not lose ground on the surrogate then `L(θ^(k)) ≤ L(θ^(k+1))`. -/
theorem ch08a_em_step_ascent {Θ Z : Type*} [Fintype Z] [Nonempty Z] (P : Θ → Z → ℝ)
    (θk θnext : Θ) (hk : ∀ z, 0 < P θk z) (hn : ∀ z, 0 < P θnext z)
    (hQ : ch08a_em_surrogate P θk θk ≤ ch08a_em_surrogate P θk θnext) :
    ch08a_L (P θk) ≤ ch08a_L (P θnext) := by
  have hmk : 0 < ch08a_marg (P θk) := ch08a_marg_pos _ hk
  have hmn : 0 < ch08a_marg (P θnext) := ch08a_marg_pos _ hn
  have hqpos : ∀ z, 0 < ch08a_post (P θk) z := ch08a_post_pos (P θk) hk hmk
  have hq1 : ∑ z, ch08a_post (P θk) z = 1 := ch08a_post_sum_one (P θk) hmk.ne'
  have hQ' : ch08a_Q (ch08a_post (P θk)) (P θk)
      ≤ ch08a_Q (ch08a_post (P θk)) (P θnext) := hQ
  have hF : ch08a_F (ch08a_post (P θk)) (P θk)
      ≤ ch08a_F (ch08a_post (P θk)) (P θnext) :=
    (ch08a_em_F_argmax_eq_Q_argmax _ _ _).mpr hQ'
  have htouch : ch08a_F (ch08a_post (P θk)) (P θk) = ch08a_L (P θk) :=
    ch08a_em_touching (P θk) hk hmk
  have hlb : ch08a_F (ch08a_post (P θk)) (P θnext) ≤ ch08a_L (P θnext) :=
    ch08a_em_lower_bound (P θnext) (ch08a_post (P θk)) hq1 hqpos hn hmn
  linarith

/-- Same statement with a genuine M-step in the third arrow. -/
theorem ch08a_em_step_ascent_of_mstep {Θ Z : Type*} [Fintype Z] [Nonempty Z]
    (P : Θ → Z → ℝ) (θk θnext : Θ) (hk : ∀ z, 0 < P θk z) (hn : ∀ z, 0 < P θnext z)
    (hM : ch08a_em_mstep P θk θnext) :
    ch08a_L (P θk) ≤ ch08a_L (P θnext) :=
  ch08a_em_step_ascent P θk θnext hk hn (hM θk)

/-- Iterating the picture: for any number of sweeps the marginal log-likelihood
along the EM sequence is nondecreasing. -/
theorem ch08a_em_ascent_iterated {Θ Z : Type*} [Fintype Z] [Nonempty Z]
    (P : Θ → Z → ℝ) (θ : ℕ → Θ) (hpos : ∀ k z, 0 < P (θ k) z)
    (hstep : ∀ k, ch08a_em_mstep P (θ k) (θ (k + 1))) :
    Monotone fun k => ch08a_L (P (θ k)) :=
  monotone_nat_of_le_succ fun k =>
    ch08a_em_step_ascent_of_mstep P (θ k) (θ (k + 1)) (hpos k) (hpos (k + 1)) (hstep k)

/-! ### Complete data and the surrogate -/

-- eq:em-complete-kernel (lines 347-361):
-- K̃_θ(a',c|a) = π_c N(a'-αa ; ν_c, s_c²),  K_θ(a'|a) = Σ_c K̃_θ(a',c|a).
/-- The label-augmented (complete-data) transition kernel. -/
def ch08a_ktilde {C : ℕ} (α : ℝ) (p ν s : Fin C → ℝ) (a' a : ℝ) (c : Fin C) : ℝ :=
  p c * ch08a_gauss (ν c) (s c) (a' - α * a)

theorem ch08a_em_complete_kernel {C : ℕ} (α : ℝ) (p ν s : Fin C → ℝ) (a' a : ℝ) :
    ch08a_em_mixture_kernel α p ν s a' a = ∑ c, ch08a_ktilde α p ν s a' a c := rfl

theorem ch08a_ktilde_pos {C : ℕ} (α : ℝ) (p ν s : Fin C → ℝ) (a' a : ℝ) (c : Fin C)
    (hp : 0 < p c) (hs : 0 < s c) : 0 < ch08a_ktilde α p ν s a' a c :=
  mul_pos hp (ch08a_gauss_pos _ _ hs)

-- eq:em-log-mixture (lines 365-378):
-- log K_θ(a_i|a_{i-1}) = log [ Σ_c π_c N(a_i - α a_{i-1} ; ν_c, s_c²) ].
theorem ch08a_em_log_mixture {C : ℕ} (α : ℝ) (p ν s : Fin C → ℝ) (ai aim1 : ℝ) :
    Real.log (ch08a_em_mixture_kernel α p ν s ai aim1)
      = Real.log (∑ c, p c * ch08a_gauss (ν c) (s c) (ai - α * aim1)) := rfl

-- eq:em-complete-factorisation (lines 385-394):
-- p_θ(a,c,x) = p(a_1) ∏_{i=2}^N K̃_θ(a_i,c_i|a_{i-1}) ∏_{i=1}^N φ_i(a_i).
/-- The complete-data joint density for one sequence. -/
def ch08a_joint {C : ℕ} (N : ℕ) (α : ℝ) (p ν s : Fin C → ℝ)
    (p1 : ℝ → ℝ) (φ : ℕ → ℝ → ℝ) (a : ℕ → ℝ) (lab : ℕ → Fin C) : ℝ :=
  p1 (a 1) * (∏ i ∈ Finset.Icc 2 N, ch08a_ktilde α p ν s (a i) (a (i - 1)) (lab i))
    * ∏ i ∈ Finset.Icc 1 N, φ i (a i)

theorem ch08a_joint_pos {C : ℕ} (N : ℕ) (α : ℝ) (p ν s : Fin C → ℝ)
    (p1 : ℝ → ℝ) (φ : ℕ → ℝ → ℝ) (a : ℕ → ℝ) (lab : ℕ → Fin C)
    (hp1 : 0 < p1 (a 1)) (hp : ∀ c, 0 < p c) (hs : ∀ c, 0 < s c)
    (hφ : ∀ i ∈ Finset.Icc 1 N, 0 < φ i (a i)) :
    0 < ch08a_joint N α p ν s p1 φ a lab := by
  refine mul_pos (mul_pos hp1 ?_) (Finset.prod_pos hφ)
  exact Finset.prod_pos fun i _ =>
    ch08a_ktilde_pos α p ν s (a i) (a (i - 1)) (lab i) (hp _) (hs _)

-- noname-11 (lines 396-398): φ_i(a_i) = p(x_i | a_i), the fixed OU observation factor.
/-- The OU observation factor, `φ_i(a_i) = p(x_i | a_i)`; under the channel
`x = e^{-t} a + √Δ_t ξ` it is a Gaussian in `x_i - e^{-t} a_i`. -/
def ch08a_phi (t Δ xi ai : ℝ) : ℝ := ch08a_gauss 0 (Real.sqrt Δ) (xi - Real.exp (-t) * ai)

-- eq:em-complete-ll (lines 400-413):
-- log p_θ(a,c,x) = log p(a_1) + Σ_{i=2}^N log K̃_θ(a_i,c_i|a_{i-1}) + Σ_{i=1}^N log φ_i(a_i).
theorem ch08a_em_complete_ll {C : ℕ} (N : ℕ) (α : ℝ) (p ν s : Fin C → ℝ)
    (p1 : ℝ → ℝ) (φ : ℕ → ℝ → ℝ) (a : ℕ → ℝ) (lab : ℕ → Fin C)
    (hp1 : p1 (a 1) ≠ 0)
    (hk : ∀ i ∈ Finset.Icc 2 N, ch08a_ktilde α p ν s (a i) (a (i - 1)) (lab i) ≠ 0)
    (hφ : ∀ i ∈ Finset.Icc 1 N, φ i (a i) ≠ 0) :
    Real.log (ch08a_joint N α p ν s p1 φ a lab)
      = Real.log (p1 (a 1))
        + (∑ i ∈ Finset.Icc 2 N, Real.log (ch08a_ktilde α p ν s (a i) (a (i - 1)) (lab i)))
        + ∑ i ∈ Finset.Icc 1 N, Real.log (φ i (a i)) := by
  have hprodk : (∏ i ∈ Finset.Icc 2 N, ch08a_ktilde α p ν s (a i) (a (i - 1)) (lab i)) ≠ 0 :=
    Finset.prod_ne_zero_iff.mpr hk
  have hprodφ : (∏ i ∈ Finset.Icc 1 N, φ i (a i)) ≠ 0 :=
    Finset.prod_ne_zero_iff.mpr hφ
  rw [ch08a_joint, Real.log_mul (mul_ne_zero hp1 hprodk) hprodφ,
      Real.log_mul hp1 hprodk, Real.log_prod hk, Real.log_prod hφ]

-- eq:em-single-transition-log (lines 418-439):
-- log K̃_θ(a_i,c_i|a_{i-1}) = log π_{c_i} + log N(a_i - α a_{i-1} ; ν_{c_i}, s_{c_i}²)
--   = log π_{c_i} - ½log(2π) - ½log s_{c_i}² - (a_i - α a_{i-1} - ν_{c_i})²/(2 s_{c_i}²).
theorem ch08a_em_single_transition_log {C : ℕ} (α : ℝ) (p ν s : Fin C → ℝ)
    (ai aim1 : ℝ) (c : Fin C) (hp : 0 < p c) (hs : 0 < s c) :
    Real.log (ch08a_ktilde α p ν s ai aim1 c)
      = Real.log (p c) - (1 / 2) * Real.log (2 * Real.pi)
        - (1 / 2) * Real.log (s c ^ 2)
        - (ai - α * aim1 - ν c) ^ 2 / (2 * (s c) ^ 2) := by
  have hsq : (0 : ℝ) < (s c) ^ 2 := pow_pos hs 2
  have h2pi : (0 : ℝ) < 2 * Real.pi := by positivity
  have hprod : (0 : ℝ) < 2 * Real.pi * (s c) ^ 2 := mul_pos h2pi hsq
  have hsqrt : (0 : ℝ) < Real.sqrt (2 * Real.pi * (s c) ^ 2) := Real.sqrt_pos.mpr hprod
  rw [ch08a_ktilde, ch08a_gauss,
      Real.log_mul hp.ne' (div_ne_zero (Real.exp_ne_zero _) hsqrt.ne'),
      Real.log_div (Real.exp_ne_zero _) hsqrt.ne', Real.log_exp,
      Real.log_sqrt hprod.le, Real.log_mul h2pi.ne' hsq.ne']
  ring

-- eq:em-indicator-transition (lines 441-457):
-- log K̃_θ(a_i,c_i|a_{i-1}) = Σ_c 1{c_i=c} [ log π_c - ½log(2π) - ½log s_c² - (…)²/(2s_c²) ].
theorem ch08a_em_indicator_transition {C : ℕ} (α : ℝ) (p ν s : Fin C → ℝ)
    (ai aim1 : ℝ) (ci : Fin C) (hp : 0 < p ci) (hs : 0 < s ci) :
    Real.log (ch08a_ktilde α p ν s ai aim1 ci)
      = ∑ c, (if ci = c then (1 : ℝ) else 0) *
          (Real.log (p c) - (1 / 2) * Real.log (2 * Real.pi)
            - (1 / 2) * Real.log (s c ^ 2)
            - (ai - α * aim1 - ν c) ^ 2 / (2 * (s c) ^ 2)) := by
  rw [ch08a_em_single_transition_log α p ν s ai aim1 ci hp hs]
  simp

-- eq:em-Q (lines 460-484):
-- Q(θ|θ^(k)) =^c Σ_{i=2}^N Σ_c E[ 1{c_i=c}( log π_c - ½log s_c² - (a_i-αa_{i-1}-ν_c)²/(2s_c²) ) ],
-- where "=^c" is equality up to an additive constant independent of θ; in particular the
-- dropped -½log(2π) per transition.
/-- The per-(transition, component) summand appearing inside the double sum of
eq:em-Q: `1{c_i = c} ( log π_c - ½ log s_c² - (a_i - α a_{i-1} - ν_c)²/(2 s_c²) )`. -/
def ch08a_Qsummand {C : ℕ} (α : ℝ) (p ν s : Fin C → ℝ) (ai aim1 : ℝ) (ci c : Fin C) : ℝ :=
  (if ci = c then (1 : ℝ) else 0) *
    (Real.log (p c) - (1 / 2) * Real.log (s c ^ 2)
      - (ai - α * aim1 - ν c) ^ 2 / (2 * (s c) ^ 2))

theorem ch08a_Qsummand_sum {C : ℕ} (α : ℝ) (p ν s : Fin C → ℝ) (ai aim1 : ℝ) (ci : Fin C) :
    (∑ c, ch08a_Qsummand α p ν s ai aim1 ci c)
      = Real.log (p ci) - (1 / 2) * Real.log (s ci ^ 2)
        - (ai - α * aim1 - ν ci) ^ 2 / (2 * (s ci) ^ 2) := by
  simp [ch08a_Qsummand]

/-- Per transition: the `θ`-dependent part of `log K̃` is exactly the bracket of
eq:em-Q, and the term dropped by `=^c` is exactly the constant `-½ log(2π)`. -/
theorem ch08a_em_Q_term {C : ℕ} (α : ℝ) (p ν s : Fin C → ℝ) (ai aim1 : ℝ) (ci : Fin C)
    (hp : 0 < p ci) (hs : 0 < s ci) :
    Real.log (ch08a_ktilde α p ν s ai aim1 ci)
      = (∑ c, ch08a_Qsummand α p ν s ai aim1 ci c)
        - (1 / 2) * Real.log (2 * Real.pi) := by
  rw [ch08a_em_single_transition_log α p ν s ai aim1 ci hp hs, ch08a_Qsummand_sum]
  ring

/-- Exchange of the posterior expectation with the finite sums over sites and mixture
components: the linearity step behind eq:em-Q. -/
theorem ch08a_sum_swap {Z ι κ : Type*} [Fintype Z] [Fintype κ] (u : Finset ι)
    (w : Z → ℝ) (G : Z → ι → κ → ℝ) :
    ∑ z, w z * (∑ i ∈ u, ∑ c : κ, G z i c)
      = ∑ i ∈ u, ∑ c : κ, ∑ z, w z * G z i c := by
  have h : ∀ z : Z, w z * (∑ i ∈ u, ∑ c : κ, G z i c)
      = ∑ i ∈ u, ∑ c : κ, w z * G z i c := by
    intro z
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun i _ => by rw [Finset.mul_sum]
  calc ∑ z, w z * (∑ i ∈ u, ∑ c : κ, G z i c)
      = ∑ z, ∑ i ∈ u, ∑ c : κ, w z * G z i c := Finset.sum_congr rfl fun z _ => h z
    _ = ∑ i ∈ u, ∑ z, ∑ c : κ, w z * G z i c := Finset.sum_comm
    _ = ∑ i ∈ u, ∑ c : κ, ∑ z, w z * G z i c :=
        Finset.sum_congr rfl fun i _ => Finset.sum_comm

/-- eq:em-Q in full: writing the complete-data log-likelihood as a `θ`-independent part
`K` plus the `N-1` transition terms (eq:em-complete-ll), the EM surrogate
`Q = E_q[log p_θ(a,c,x)]` equals a constant independent of `θ` plus exactly the double
sum `Σ_{i=2}^{N} Σ_c E[ 1{c_i=c}( log π_c - ½log s_c² - (a_i-αa_{i-1}-ν_c)²/(2s_c²) ) ]`
displayed in the chapter.  The dropped constant is `E_q[K] - (N-1)·½log(2π)`. -/
theorem ch08a_em_Q_full {C : ℕ} {Z : Type*} [Fintype Z] (N : ℕ)
    (w K : Z → ℝ) (A : Z → ℕ → ℝ) (lab : Z → ℕ → Fin C)
    (α : ℝ) (p ν s : Fin C → ℝ) (hp : ∀ c, 0 < p c) (hs : ∀ c, 0 < s c)
    (hw : ∑ z, w z = 1) :
    ∑ z, w z * (K z + ∑ i ∈ Finset.Icc 2 N,
        Real.log (ch08a_ktilde α p ν s (A z i) (A z (i - 1)) (lab z i)))
      = ((∑ z, w z * K z)
          - ((Finset.Icc 2 N).card : ℝ) * ((1 / 2) * Real.log (2 * Real.pi)))
        + ∑ i ∈ Finset.Icc 2 N, ∑ c : Fin C, ∑ z, w z *
            ch08a_Qsummand α p ν s (A z i) (A z (i - 1)) (lab z i) c := by
  have hi : ∀ z : Z,
      (∑ i ∈ Finset.Icc 2 N,
        Real.log (ch08a_ktilde α p ν s (A z i) (A z (i - 1)) (lab z i)))
      = (∑ i ∈ Finset.Icc 2 N, ∑ c : Fin C,
            ch08a_Qsummand α p ν s (A z i) (A z (i - 1)) (lab z i) c)
        - ((Finset.Icc 2 N).card : ℝ) * ((1 / 2) * Real.log (2 * Real.pi)) := by
    intro z
    rw [Finset.sum_congr rfl (fun i (_ : i ∈ Finset.Icc 2 N) =>
        ch08a_em_Q_term α p ν s (A z i) (A z (i - 1)) (lab z i) (hp _) (hs _)),
      Finset.sum_sub_distrib, Finset.sum_const, nsmul_eq_mul]
  have hmain : ∀ z ∈ (Finset.univ : Finset Z),
      w z * (K z + ∑ i ∈ Finset.Icc 2 N,
        Real.log (ch08a_ktilde α p ν s (A z i) (A z (i - 1)) (lab z i)))
      = (w z * K z
          - w z * (((Finset.Icc 2 N).card : ℝ) * ((1 / 2) * Real.log (2 * Real.pi))))
        + w z * (∑ i ∈ Finset.Icc 2 N, ∑ c : Fin C,
            ch08a_Qsummand α p ν s (A z i) (A z (i - 1)) (lab z i) c) := by
    intro z _; rw [hi z]; ring
  rw [Finset.sum_congr rfl hmain, Finset.sum_add_distrib, Finset.sum_sub_distrib,
    ← Finset.sum_mul, hw, one_mul,
    ch08a_sum_swap (Finset.Icc 2 N) w
      (fun z i c => ch08a_Qsummand α p ν s (A z i) (A z (i - 1)) (lab z i) c)]

/-- Linearity of the posterior expectation: dropping a `θ`-independent constant from
the integrand shifts `Q` by exactly that constant, so the two objectives have the same
maximisers.  Combined with `ch08a_em_Q_term` this is eq:em-Q. -/
theorem ch08a_em_Q_expectation {Z : Type*} [Fintype Z] (w f g : Z → ℝ) (κ : ℝ)
    (h : ∀ z, f z = g z - κ) (hw : ∑ z, w z = 1) :
    ∑ z, w z * f z = (∑ z, w z * g z) - κ := by
  have : ∀ z ∈ (Finset.univ : Finset Z), w z * f z = w z * g z - w z * κ := by
    intro z _; rw [h z]; ring
  rw [Finset.sum_congr rfl this, Finset.sum_sub_distrib, ← Finset.sum_mul, hw, one_mul]

/-! ### Fisher's identity -/

-- noname-12 (lines 499-503): L(θ) = log p_θ(x).
theorem ch08a_L_def {Z : Type*} [Fintype Z] (P : Z → ℝ) :
    ch08a_L P = Real.log (∑ z, P z) := rfl

-- eq:em-fisher-step1 (lines 513-525):
-- ∇L = (1/p_θ(x)) ∇p_θ(x) = (1/p_θ(x)) Σ_c ∫ ∇_θ p_θ(a,c,x) da.
theorem ch08a_em_fisher_step1 {Z : Type*} [Fintype Z] (P : ℝ → Z → ℝ) (P' : Z → ℝ)
    (θ : ℝ) (hd : ∀ z, HasDerivAt (fun t => P t z) (P' z) θ)
    (hm : (∑ z, P θ z) ≠ 0) :
    HasDerivAt (fun t => Real.log (∑ z, P t z))
      ((1 / (∑ z, P θ z)) * ∑ z, P' z) θ := by
  have hsum : HasDerivAt (fun t => ∑ z, P t z) (∑ z, P' z) θ :=
    HasDerivAt.fun_sum fun z _ => hd z
  have h := hsum.log hm
  convert h using 1
  field_simp

-- noname-13 (lines 527-532): ∇_θ p_θ(a,c,x) = p_θ(a,c,x) ∇_θ log p_θ(a,c,x).
theorem ch08a_score_identity (f : ℝ → ℝ) (f' θ : ℝ) (hf : HasDerivAt f f' θ)
    (h : f θ ≠ 0) :
    HasDerivAt (fun t => Real.log (f t)) (f' / f θ) θ ∧ f' = f θ * (f' / f θ) :=
  ⟨hf.log h, by field_simp⟩

-- eq:em-fisher-general (lines 534-557):
-- ∇L = Σ_c ∫ p_θ(a,c|x) ∇_θ log p_θ(a,c,x) da = E_{a,c|x,θ}[ ∇_θ log p_θ(a,c,x) ].
theorem ch08a_em_fisher_general {Z : Type*} [Fintype Z] (P : ℝ → Z → ℝ) (P' : Z → ℝ)
    (θ : ℝ) (hd : ∀ z, HasDerivAt (fun t => P t z) (P' z) θ)
    (hpos : ∀ z, P θ z ≠ 0) (hm : (∑ z, P θ z) ≠ 0) :
    HasDerivAt (fun t => Real.log (∑ z, P t z))
      (∑ z, (P θ z / (∑ w, P θ w)) * (P' z / P θ z)) θ := by
  have h := ch08a_em_fisher_step1 P P' θ hd hm
  convert h using 1
  have hterm : ∀ z ∈ (Finset.univ : Finset Z),
      (P θ z / (∑ w, P θ w)) * (P' z / P θ z) = P' z / (∑ w, P θ w) := by
    intro z _
    have hz := hpos z
    field_simp
  rw [Finset.sum_congr rfl hterm, ← Finset.sum_div]
  field_simp

-- eq:em-fisher (lines 562-574):
-- ∇L = E[ Σ_{i=2}^N ∇_θ log K̃_θ(a_i,c_i|a_{i-1}) ]: only the transition factors depend on θ.
theorem ch08a_em_fisher {ι : Type*} (s : Finset ι) (K : ℝ) (T : ι → ℝ → ℝ)
    (T' : ι → ℝ) (θ : ℝ) (hd : ∀ i ∈ s, HasDerivAt (T i) (T' i) θ) :
    HasDerivAt (fun t => K + ∑ i ∈ s, T i t) (∑ i ∈ s, T' i) θ :=
  HasDerivAt.const_add K (HasDerivAt.fun_sum hd)

-- eq:em-fisher-Q (lines 576-584):
-- ∇_θ L(θ)|_{θ=θ^(k)} = ∇_θ Q(θ|θ^(k))|_{θ=θ^(k)}.
theorem ch08a_em_fisher_Q {Z : Type*} [Fintype Z] (P : ℝ → Z → ℝ) (P' : Z → ℝ)
    (θ : ℝ) (hd : ∀ z, HasDerivAt (fun t => P t z) (P' z) θ)
    (hpos : ∀ z, P θ z ≠ 0) (hm : (∑ z, P θ z) ≠ 0) :
    HasDerivAt (fun t => ch08a_Q (ch08a_post (P θ)) (P t))
        (∑ z, (P θ z / (∑ w, P θ w)) * (P' z / P θ z)) θ
      ∧ HasDerivAt (fun t => ch08a_L (P t))
        (∑ z, (P θ z / (∑ w, P θ w)) * (P' z / P θ z)) θ := by
  constructor
  · have hQ : ∀ z : Z, HasDerivAt
        (fun t => (P θ z / ∑ w, P θ w) * Real.log (P t z))
        ((P θ z / ∑ w, P θ w) * (P' z / P θ z)) θ :=
      fun z => HasDerivAt.const_mul _ ((hd z).log (hpos z))
    have h := HasDerivAt.fun_sum (u := (Finset.univ : Finset Z)) fun z _ => hQ z
    simpa only [ch08a_Q, ch08a_post, ch08a_marg] using h
  · have h := ch08a_em_fisher_general P P' θ hd hpos hm
    simpa only [ch08a_L, ch08a_marg] using h

-- noname-14 (lines 588-592): q^(k)(a,c) = p_{θ^(k)}(a,c|x); BP supplies this posterior
-- and it is then held fixed, so nothing is differentiated through the recursion.
theorem ch08a_q_fixed {Z : Type*} [Fintype Z] (P0 : Z → ℝ) (z : Z) :
    ch08a_post P0 z = P0 z / ch08a_marg P0 := rfl

/-! ### prop:em-estep (lines 335-342): "The E-step is exact"

The chapter's Proposition: *since the posterior factor graph is a tree, one forward and
one backward sum-product sweep return the exact single-site and pairwise posterior
marginals, so the E-step needs neither a variational approximation nor Monte Carlo
sampling.*

A chain is a tree, and the chain is the case the chapter uses, so what is proved here is
exactness of forward-backward on the chain, for an **arbitrary** number of sites and an
arbitrary finite alphabet `A` at each site (the grid of the chapter; taking
`A = grid × Fin C` puts the mixture label inside the alphabet, which is how the
`(a_{i-1}, a_i, c_i)` expectations of eq:em-Q are covered -- see
`ch08a_estep_exact_mixture`).

* `ch08a_chainW` is the honest joint weight of a whole configuration,
  `φ_0(a_0) ∏_{j=1}^{n} ψ_j(a_{j-1},a_j) φ_j(a_j)`; no message occurs in it;
* `ch08a_fwd` is the left-to-right sweep and `ch08a_bwd` the right-to-left sweep, each
  given by its own one-pass recursion (`ch08a_fwd_succ`, `ch08a_bwd_succ`);
* `ch08a_estep_exact` / `ch08a_estep_exact_pair` prove that *summing the joint weight
  over every configuration compatible with a site value* (resp. with a pair of adjacent
  site values) -- which is the exact marginal, by definition -- returns exactly the
  product of the two messages.

Nothing here is approximate, and no fixed length, fixed alphabet or fixed site is used. -/

section Estep

variable {A : Type*}

/-- Joint weight of one complete chain configuration on the sites `0, …, n`:
`φ_0(a_0) ∏_{j=1}^{n} ψ_j(a_{j-1}, a_j) φ_j(a_j)`. -/
def ch08a_chainW (φ : ℕ → A → ℝ) (ψ : ℕ → A → A → ℝ) : (n : ℕ) → (Fin (n + 1) → A) → ℝ
  | 0, a => φ 0 (a (Fin.last 0))
  | n + 1, a =>
      ch08a_chainW φ ψ n (Fin.init a) *
        (ψ (n + 1) (Fin.init a (Fin.last n)) (a (Fin.last (n + 1))) *
          φ (n + 1) (a (Fin.last (n + 1))))

theorem ch08a_chainW_zero (φ : ℕ → A → ℝ) (ψ : ℕ → A → A → ℝ) (a : Fin 1 → A) :
    ch08a_chainW φ ψ 0 a = φ 0 (a (Fin.last 0)) := by
  simp only [ch08a_chainW]

theorem ch08a_chainW_succ (φ : ℕ → A → ℝ) (ψ : ℕ → A → A → ℝ) (n : ℕ) (a : Fin (n + 2) → A) :
    ch08a_chainW φ ψ (n + 1) a
      = ch08a_chainW φ ψ n (Fin.init a) *
        (ψ (n + 1) (Fin.init a (Fin.last n)) (a (Fin.last (n + 1))) *
          φ (n + 1) (a (Fin.last (n + 1)))) := by
  simp only [ch08a_chainW]

/-- Adding one site at the right end multiplies the weight by exactly one transition
factor and one site factor. -/
theorem ch08a_chainW_snoc (φ : ℕ → A → ℝ) (ψ : ℕ → A → A → ℝ) (n : ℕ)
    (b : Fin (n + 1) → A) (x : A) :
    ch08a_chainW φ ψ (n + 1) (Fin.snoc b x)
      = ch08a_chainW φ ψ n b * (ψ (n + 1) (b (Fin.last n)) x * φ (n + 1) x) := by
  rw [ch08a_chainW_succ, Fin.init_snoc, Fin.snoc_last]

/-- The recursion above is exactly the factorisation displayed in
eq:em-complete-factorisation: `φ_0(a_0) · ∏_{j=1}^{n} ψ_j(a_{j-1}, a_j) φ_j(a_j)`, with
`n` transition factors and `n+1` site factors.  (Recorded so that the object the
exactness theorems below marginalise is auditable as *the* chain joint, not as a
message-shaped surrogate.) -/
theorem ch08a_chainW_eq_prod (φ : ℕ → A → ℝ) (ψ : ℕ → A → A → ℝ) :
    ∀ (n : ℕ) (a : Fin (n + 1) → A),
      ch08a_chainW φ ψ n a
        = φ 0 (a 0) *
            ∏ j : Fin n, (ψ ((j : ℕ) + 1) (a j.castSucc) (a j.succ) *
              φ ((j : ℕ) + 1) (a j.succ)) := by
  intro n
  induction n with
  | zero =>
      intro a
      rw [ch08a_chainW_zero, Fin.last_zero]
      simp
  | succ n ih =>
      intro a
      have hinit : ∀ x : Fin (n + 1), Fin.init a x = a x.castSucc := fun _ => rfl
      rw [ch08a_chainW_succ, ih (Fin.init a), Fin.prod_univ_castSucc]
      simp only [hinit, Fin.val_castSucc, Fin.val_last, Fin.succ_castSucc, Fin.castSucc_zero,
        Fin.succ_last]
      ring

/-- The forward sum-product sweep `F_i` (one left-to-right pass). -/
def ch08a_fwd [Fintype A] (φ : ℕ → A → ℝ) (ψ : ℕ → A → A → ℝ) : ℕ → A → ℝ
  | 0, v => φ 0 v
  | i + 1, v => (∑ u : A, ch08a_fwd φ ψ i u * ψ (i + 1) u v) * φ (i + 1) v

theorem ch08a_fwd_zero [Fintype A] (φ : ℕ → A → ℝ) (ψ : ℕ → A → A → ℝ) (v : A) :
    ch08a_fwd φ ψ 0 v = φ 0 v := by simp only [ch08a_fwd]

/-- The forward recursion: the message after site `i+1` is one transfer step applied to
the message after site `i`. -/
theorem ch08a_fwd_succ [Fintype A] (φ : ℕ → A → ℝ) (ψ : ℕ → A → A → ℝ) (i : ℕ) (v : A) :
    ch08a_fwd φ ψ (i + 1) v = (∑ u : A, ch08a_fwd φ ψ i u * ψ (i + 1) u v) * φ (i + 1) v := by
  simp only [ch08a_fwd]

/-- The backward sum-product sweep, carrying a terminal function `g` at the far end:
`ch08a_bwdg φ ψ g m i v` is the message arriving at site `i` from the `m` sites to its
right. -/
def ch08a_bwdg [Fintype A] (φ : ℕ → A → ℝ) (ψ : ℕ → A → A → ℝ) (g : A → ℝ) : ℕ → ℕ → A → ℝ
  | 0, _, v => g v
  | m + 1, i, v => ∑ w : A, ψ (i + 1) v w * φ (i + 1) w * ch08a_bwdg φ ψ g m (i + 1) w

theorem ch08a_bwdg_zero [Fintype A] (φ : ℕ → A → ℝ) (ψ : ℕ → A → A → ℝ) (g : A → ℝ)
    (i : ℕ) (v : A) : ch08a_bwdg φ ψ g 0 i v = g v := by simp only [ch08a_bwdg]

theorem ch08a_bwdg_succ [Fintype A] (φ : ℕ → A → ℝ) (ψ : ℕ → A → A → ℝ) (g : A → ℝ)
    (m i : ℕ) (v : A) :
    ch08a_bwdg φ ψ g (m + 1) i v
      = ∑ w : A, ψ (i + 1) v w * φ (i + 1) w * ch08a_bwdg φ ψ g m (i + 1) w := by
  simp only [ch08a_bwdg]

/-- The backward message proper `B_i` (terminal function `1`). -/
def ch08a_bwd [Fintype A] (φ : ℕ → A → ℝ) (ψ : ℕ → A → A → ℝ) (m i : ℕ) (v : A) : ℝ :=
  ch08a_bwdg φ ψ (fun _ => 1) m i v

theorem ch08a_bwd_zero [Fintype A] (φ : ℕ → A → ℝ) (ψ : ℕ → A → A → ℝ) (i : ℕ) (v : A) :
    ch08a_bwd φ ψ 0 i v = 1 := by simp only [ch08a_bwd, ch08a_bwdg]

/-- The backward recursion: the message at site `i` with `m+1` sites to its right is one
transfer step applied to the message at site `i+1`. -/
theorem ch08a_bwd_succ [Fintype A] (φ : ℕ → A → ℝ) (ψ : ℕ → A → A → ℝ) (m i : ℕ) (v : A) :
    ch08a_bwd φ ψ (m + 1) i v
      = ∑ w : A, ψ (i + 1) v w * φ (i + 1) w * ch08a_bwd φ ψ m (i + 1) w := by
  simp only [ch08a_bwd, ch08a_bwdg_succ]

/-- The backward recursion may equally be peeled at its far end: absorbing the last
transition into the terminal function shortens the message by one step.  This is the
associativity of transfer products that makes the two sweeps, which run in opposite
directions, fit together. -/
theorem ch08a_bwdg_succ_last [Fintype A] (φ : ℕ → A → ℝ) (ψ : ℕ → A → A → ℝ) :
    ∀ (m i : ℕ) (v : A) (g : A → ℝ),
      ch08a_bwdg φ ψ g (m + 1) i v
        = ch08a_bwdg φ ψ (fun w => ∑ u : A, ψ (i + m + 1) w u * φ (i + m + 1) u * g u) m i v := by
  intro m
  induction m with
  | zero =>
      intro i v g
      rw [ch08a_bwdg_succ, ch08a_bwdg_zero]
      exact Finset.sum_congr rfl fun w _ => by rw [ch08a_bwdg_zero]
  | succ m ih =>
      intro i v g
      have hidx : i + 1 + m + 1 = i + (m + 1) + 1 := by omega
      rw [ch08a_bwdg_succ, ch08a_bwdg_succ]
      refine Finset.sum_congr rfl fun w _ => ?_
      rw [ih (i + 1) w g, hidx]

/-- Summing over configurations of `n+1` sites = summing over the value at the last site
and over configurations of the first `n` sites. -/
theorem ch08a_sum_snoc [Fintype A] {n : ℕ} (F : (Fin (n + 1) → A) → ℝ) :
    ∑ a : Fin (n + 1) → A, F a = ∑ x : A, ∑ b : Fin n → A, F (Fin.snoc b x) := by
  have h1 : ∑ q : A × (Fin n → A), F (Fin.snoc q.2 q.1) = ∑ a : Fin (n + 1) → A, F a :=
    Fintype.sum_equiv (Fin.snocEquiv fun _ : Fin (n + 1) => A)
      (fun q => F (Fin.snoc q.2 q.1)) F (fun _ => rfl)
  have h2 : ∑ q : A × (Fin n → A), F (Fin.snoc q.2 q.1)
      = ∑ x : A, ∑ b : Fin n → A, F (Fin.snoc b x) := Fintype.sum_prod_type _
  rw [← h1, h2]

/-- **The forward sweep is exact at its own end**: summing the joint weight over every
configuration whose last site carries the value `v` gives the forward message `F_i(v)`. -/
theorem ch08a_fwd_exact [Fintype A] [DecidableEq A] (φ : ℕ → A → ℝ) (ψ : ℕ → A → A → ℝ) :
    ∀ (i : ℕ) (v : A),
      (∑ a : Fin (i + 1) → A,
          (if a (Fin.last i) = v then (1 : ℝ) else 0) * ch08a_chainW φ ψ i a)
        = ch08a_fwd φ ψ i v := by
  intro i
  induction i with
  | zero =>
      intro v
      calc (∑ a : Fin 1 → A,
              (if a (Fin.last 0) = v then (1 : ℝ) else 0) * ch08a_chainW φ ψ 0 a)
          = ∑ x : A, ∑ b : Fin 0 → A,
              (if (Fin.snoc b x : Fin 1 → A) (Fin.last 0) = v then (1 : ℝ) else 0) *
                ch08a_chainW φ ψ 0 (Fin.snoc b x) := ch08a_sum_snoc _
        _ = ∑ x : A, (if x = v then (1 : ℝ) else 0) * φ 0 x := by
              refine Finset.sum_congr rfl fun x _ => ?_
              rw [Finset.univ_unique, Finset.sum_singleton, ch08a_chainW_zero, Fin.snoc_last]
        _ = ch08a_fwd φ ψ 0 v := by rw [ch08a_fwd_zero]; simp
  | succ i ih =>
      intro v
      have hpeel : ∀ (x : A) (b : Fin (i + 1) → A),
          (if (Fin.snoc b x : Fin (i + 1 + 1) → A) (Fin.last (i + 1)) = v then (1 : ℝ) else 0) *
              ch08a_chainW φ ψ (i + 1) (Fin.snoc b x)
            = (if x = v then (1 : ℝ) else 0) *
                (ch08a_chainW φ ψ i b * (ψ (i + 1) (b (Fin.last i)) x * φ (i + 1) x)) := by
        intro x b
        rw [Fin.snoc_last, ch08a_chainW_snoc]
      have hcollapse : ∀ u : A,
          ch08a_fwd φ ψ i u * ψ (i + 1) u v * φ (i + 1) v
            = ∑ b : Fin (i + 1) → A,
                (if b (Fin.last i) = u then (1 : ℝ) else 0) *
                  (ch08a_chainW φ ψ i b * (ψ (i + 1) u v * φ (i + 1) v)) := by
        intro u
        rw [← ih u, Finset.sum_mul, Finset.sum_mul]
        exact Finset.sum_congr rfl fun b _ => by ring
      calc (∑ a : Fin (i + 1 + 1) → A,
              (if a (Fin.last (i + 1)) = v then (1 : ℝ) else 0) * ch08a_chainW φ ψ (i + 1) a)
          = ∑ x : A, ∑ b : Fin (i + 1) → A,
              (if (Fin.snoc b x : Fin (i + 1 + 1) → A) (Fin.last (i + 1)) = v
                  then (1 : ℝ) else 0) *
                ch08a_chainW φ ψ (i + 1) (Fin.snoc b x) := ch08a_sum_snoc _
        _ = ∑ x : A, ∑ b : Fin (i + 1) → A,
              (if x = v then (1 : ℝ) else 0) *
                (ch08a_chainW φ ψ i b * (ψ (i + 1) (b (Fin.last i)) x * φ (i + 1) x)) :=
              Finset.sum_congr rfl fun x _ => Finset.sum_congr rfl fun b _ => hpeel x b
        _ = ∑ b : Fin (i + 1) → A, ∑ x : A,
              (if x = v then (1 : ℝ) else 0) *
                (ch08a_chainW φ ψ i b * (ψ (i + 1) (b (Fin.last i)) x * φ (i + 1) x)) :=
              Finset.sum_comm
        _ = ∑ b : Fin (i + 1) → A,
              ch08a_chainW φ ψ i b * (ψ (i + 1) (b (Fin.last i)) v * φ (i + 1) v) := by
              refine Finset.sum_congr rfl fun b _ => ?_
              simp
        _ = ch08a_fwd φ ψ (i + 1) v := by
              rw [ch08a_fwd_succ, Finset.sum_mul,
                Finset.sum_congr rfl (fun u (_ : u ∈ Finset.univ) => hcollapse u),
                Finset.sum_comm]
              refine Finset.sum_congr rfl fun b _ => ?_
              simp

/-- **The E-step is exact (single site).**  For a chain of arbitrary length `i + m` and
any site `i`, summing the joint weight over *every* configuration whose site `i` carries
the value `v` -- the exact marginal -- equals the forward message times the backward
message, `F_i(v) · B_i(v)`.  The statement carries a general terminal function `g`, which
is what makes the induction close (and which also delivers the joint law of site `i` and
the last site). -/
theorem ch08a_estep_exact [Fintype A] [DecidableEq A] (φ : ℕ → A → ℝ) (ψ : ℕ → A → A → ℝ) :
    ∀ (m i : ℕ) (v : A) (g : A → ℝ),
      (∑ a : Fin (i + m + 1) → A,
          (if a ⟨i, by omega⟩ = v then (1 : ℝ) else 0) *
            (ch08a_chainW φ ψ (i + m) a * g (a (Fin.last (i + m)))))
        = ch08a_fwd φ ψ i v * ch08a_bwdg φ ψ g m i v := by
  intro m
  induction m with
  | zero =>
      intro i v g
      have key : ∀ a : Fin (i + 1) → A,
          (if a (Fin.last i) = v then (1 : ℝ) else 0) *
            (ch08a_chainW φ ψ i a * g (a (Fin.last i)))
            = ((if a (Fin.last i) = v then (1 : ℝ) else 0) * ch08a_chainW φ ψ i a) * g v := by
        intro a
        by_cases h : a (Fin.last i) = v <;> simp [h]
      have hbase : (∑ a : Fin (i + 1) → A,
          (if a (Fin.last i) = v then (1 : ℝ) else 0) *
            (ch08a_chainW φ ψ i a * g (a (Fin.last i))))
            = ch08a_fwd φ ψ i v * g v := by
        rw [Finset.sum_congr rfl (fun a (_ : a ∈ Finset.univ) => key a), ← Finset.sum_mul,
          ch08a_fwd_exact φ ψ i v]
      exact hbase
  | succ m ih =>
      intro i v g
      have hlt : i < i + m + 1 := by omega
      have hpeel : ∀ (x : A) (b : Fin (i + m + 1) → A),
          (if (Fin.snoc b x : Fin (i + m + 1 + 1) → A) (Fin.castSucc ⟨i, hlt⟩) = v
              then (1 : ℝ) else 0) *
              (ch08a_chainW φ ψ (i + m + 1) (Fin.snoc b x) *
                g ((Fin.snoc b x : Fin (i + m + 1 + 1) → A) (Fin.last (i + m + 1))))
            = (if b ⟨i, hlt⟩ = v then (1 : ℝ) else 0) *
                (ch08a_chainW φ ψ (i + m) b *
                  (ψ (i + m + 1) (b (Fin.last (i + m))) x * φ (i + m + 1) x * g x)) := by
        intro x b
        rw [Fin.snoc_castSucc, Fin.snoc_last, ch08a_chainW_snoc]
        ring
      have hmain : (∑ a : Fin (i + m + 1 + 1) → A,
          (if a (Fin.castSucc ⟨i, hlt⟩) = v then (1 : ℝ) else 0) *
            (ch08a_chainW φ ψ (i + m + 1) a * g (a (Fin.last (i + m + 1)))))
            = ch08a_fwd φ ψ i v * ch08a_bwdg φ ψ g (m + 1) i v := by
        calc (∑ a : Fin (i + m + 1 + 1) → A,
                (if a (Fin.castSucc ⟨i, hlt⟩) = v then (1 : ℝ) else 0) *
                  (ch08a_chainW φ ψ (i + m + 1) a * g (a (Fin.last (i + m + 1)))))
            = ∑ x : A, ∑ b : Fin (i + m + 1) → A,
                (if (Fin.snoc b x : Fin (i + m + 1 + 1) → A) (Fin.castSucc ⟨i, hlt⟩) = v
                    then (1 : ℝ) else 0) *
                  (ch08a_chainW φ ψ (i + m + 1) (Fin.snoc b x) *
                    g ((Fin.snoc b x : Fin (i + m + 1 + 1) → A) (Fin.last (i + m + 1)))) :=
              ch08a_sum_snoc _
          _ = ∑ x : A, ∑ b : Fin (i + m + 1) → A,
                (if b ⟨i, hlt⟩ = v then (1 : ℝ) else 0) *
                  (ch08a_chainW φ ψ (i + m) b *
                    (ψ (i + m + 1) (b (Fin.last (i + m))) x * φ (i + m + 1) x * g x)) :=
              Finset.sum_congr rfl fun x _ => Finset.sum_congr rfl fun b _ => hpeel x b
          _ = ∑ b : Fin (i + m + 1) → A, ∑ x : A,
                (if b ⟨i, hlt⟩ = v then (1 : ℝ) else 0) *
                  (ch08a_chainW φ ψ (i + m) b *
                    (ψ (i + m + 1) (b (Fin.last (i + m))) x * φ (i + m + 1) x * g x)) :=
              Finset.sum_comm
          _ = ∑ b : Fin (i + m + 1) → A,
                (if b ⟨i, hlt⟩ = v then (1 : ℝ) else 0) *
                  (ch08a_chainW φ ψ (i + m) b *
                    ∑ x : A, ψ (i + m + 1) (b (Fin.last (i + m))) x * φ (i + m + 1) x * g x) := by
              refine Finset.sum_congr rfl fun b _ => ?_
              simp only [Finset.mul_sum]
          _ = ch08a_fwd φ ψ i v *
                ch08a_bwdg φ ψ
                  (fun w => ∑ x : A, ψ (i + m + 1) w x * φ (i + m + 1) x * g x) m i v :=
              ih i v (fun w => ∑ x : A, ψ (i + m + 1) w x * φ (i + m + 1) x * g x)
          _ = ch08a_fwd φ ψ i v * ch08a_bwdg φ ψ g (m + 1) i v := by
              rw [ch08a_bwdg_succ_last]
      exact hmain

/-- **The E-step is exact (adjacent pair).**  Summing the joint weight over every
configuration whose sites `i` and `i+1` carry the values `v` and `w` -- the exact pairwise
marginal -- equals `F_i(v) · ψ_{i+1}(v,w) φ_{i+1}(w) · B_{i+1}(w)`. -/
theorem ch08a_estep_exact_pair [Fintype A] [DecidableEq A]
    (φ : ℕ → A → ℝ) (ψ : ℕ → A → A → ℝ) :
    ∀ (m i : ℕ) (v w : A) (g : A → ℝ),
      (∑ a : Fin (i + 1 + m + 1) → A,
          (if a ⟨i, by omega⟩ = v then (1 : ℝ) else 0) *
            (if a ⟨i + 1, by omega⟩ = w then (1 : ℝ) else 0) *
              (ch08a_chainW φ ψ (i + 1 + m) a * g (a (Fin.last (i + 1 + m)))))
        = ch08a_fwd φ ψ i v * (ψ (i + 1) v w * φ (i + 1) w) * ch08a_bwdg φ ψ g m (i + 1) w := by
  intro m
  induction m with
  | zero =>
      intro i v w g
      have hlt : i < i + 1 + 1 := by omega
      have h1 := ch08a_estep_exact φ ψ 1 i v (fun y => (if y = w then (1 : ℝ) else 0) * g y)
      have hterm : ∀ y : A,
          ψ (i + 1) v y * φ (i + 1) y *
              ch08a_bwdg φ ψ (fun y => (if y = w then (1 : ℝ) else 0) * g y) 0 (i + 1) y
            = if y = w then ψ (i + 1) v y * φ (i + 1) y * g y else 0 := by
        intro y
        rw [ch08a_bwdg_zero]
        by_cases hy : y = w <;> simp [hy]
      have h2 : ch08a_bwdg φ ψ (fun y => (if y = w then (1 : ℝ) else 0) * g y) 1 i v
          = ψ (i + 1) v w * φ (i + 1) w * g w := by
        rw [ch08a_bwdg_succ, Finset.sum_congr rfl (fun y (_ : y ∈ Finset.univ) => hterm y)]
        simp
      have hbase : (∑ a : Fin (i + 1 + 1) → A,
          (if a ⟨i, hlt⟩ = v then (1 : ℝ) else 0) *
            (if a (Fin.last (i + 1)) = w then (1 : ℝ) else 0) *
              (ch08a_chainW φ ψ (i + 1) a * g (a (Fin.last (i + 1)))))
            = ch08a_fwd φ ψ i v * (ψ (i + 1) v w * φ (i + 1) w) * g w := by
        calc (∑ a : Fin (i + 1 + 1) → A,
                (if a ⟨i, hlt⟩ = v then (1 : ℝ) else 0) *
                  (if a (Fin.last (i + 1)) = w then (1 : ℝ) else 0) *
                    (ch08a_chainW φ ψ (i + 1) a * g (a (Fin.last (i + 1)))))
            = ∑ a : Fin (i + 1 + 1) → A,
                (if a ⟨i, hlt⟩ = v then (1 : ℝ) else 0) *
                  (ch08a_chainW φ ψ (i + 1) a *
                    ((if a (Fin.last (i + 1)) = w then (1 : ℝ) else 0) *
                      g (a (Fin.last (i + 1))))) :=
              Finset.sum_congr rfl fun a _ => by ring
          _ = ch08a_fwd φ ψ i v *
                ch08a_bwdg φ ψ (fun y => (if y = w then (1 : ℝ) else 0) * g y) 1 i v := h1
          _ = ch08a_fwd φ ψ i v * (ψ (i + 1) v w * φ (i + 1) w) * g w := by rw [h2]; ring
      exact hbase
  | succ m ih =>
      intro i v w g
      have hlt0 : i < i + 1 + m + 1 := by omega
      have hlt1 : i + 1 < i + 1 + m + 1 := by omega
      have hpeel : ∀ (x : A) (b : Fin (i + 1 + m + 1) → A),
          (if (Fin.snoc b x : Fin (i + 1 + m + 1 + 1) → A) (Fin.castSucc ⟨i, hlt0⟩) = v
              then (1 : ℝ) else 0) *
            (if (Fin.snoc b x : Fin (i + 1 + m + 1 + 1) → A) (Fin.castSucc ⟨i + 1, hlt1⟩) = w
              then (1 : ℝ) else 0) *
              (ch08a_chainW φ ψ (i + 1 + m + 1) (Fin.snoc b x) *
                g ((Fin.snoc b x : Fin (i + 1 + m + 1 + 1) → A) (Fin.last (i + 1 + m + 1))))
            = (if b ⟨i, hlt0⟩ = v then (1 : ℝ) else 0) *
                (if b ⟨i + 1, hlt1⟩ = w then (1 : ℝ) else 0) *
                  (ch08a_chainW φ ψ (i + 1 + m) b *
                    (ψ (i + 1 + m + 1) (b (Fin.last (i + 1 + m))) x *
                      φ (i + 1 + m + 1) x * g x)) := by
        intro x b
        rw [Fin.snoc_castSucc, Fin.snoc_castSucc, Fin.snoc_last, ch08a_chainW_snoc]
        ring
      have hmain : (∑ a : Fin (i + 1 + m + 1 + 1) → A,
          (if a (Fin.castSucc ⟨i, hlt0⟩) = v then (1 : ℝ) else 0) *
            (if a (Fin.castSucc ⟨i + 1, hlt1⟩) = w then (1 : ℝ) else 0) *
              (ch08a_chainW φ ψ (i + 1 + m + 1) a * g (a (Fin.last (i + 1 + m + 1)))))
            = ch08a_fwd φ ψ i v * (ψ (i + 1) v w * φ (i + 1) w) *
                ch08a_bwdg φ ψ g (m + 1) (i + 1) w := by
        calc (∑ a : Fin (i + 1 + m + 1 + 1) → A,
                (if a (Fin.castSucc ⟨i, hlt0⟩) = v then (1 : ℝ) else 0) *
                  (if a (Fin.castSucc ⟨i + 1, hlt1⟩) = w then (1 : ℝ) else 0) *
                    (ch08a_chainW φ ψ (i + 1 + m + 1) a * g (a (Fin.last (i + 1 + m + 1)))))
            = ∑ x : A, ∑ b : Fin (i + 1 + m + 1) → A,
                (if (Fin.snoc b x : Fin (i + 1 + m + 1 + 1) → A) (Fin.castSucc ⟨i, hlt0⟩) = v
                    then (1 : ℝ) else 0) *
                  (if (Fin.snoc b x : Fin (i + 1 + m + 1 + 1) → A)
                      (Fin.castSucc ⟨i + 1, hlt1⟩) = w then (1 : ℝ) else 0) *
                    (ch08a_chainW φ ψ (i + 1 + m + 1) (Fin.snoc b x) *
                      g ((Fin.snoc b x : Fin (i + 1 + m + 1 + 1) → A)
                        (Fin.last (i + 1 + m + 1)))) := ch08a_sum_snoc _
          _ = ∑ x : A, ∑ b : Fin (i + 1 + m + 1) → A,
                (if b ⟨i, hlt0⟩ = v then (1 : ℝ) else 0) *
                  (if b ⟨i + 1, hlt1⟩ = w then (1 : ℝ) else 0) *
                    (ch08a_chainW φ ψ (i + 1 + m) b *
                      (ψ (i + 1 + m + 1) (b (Fin.last (i + 1 + m))) x *
                        φ (i + 1 + m + 1) x * g x)) :=
              Finset.sum_congr rfl fun x _ => Finset.sum_congr rfl fun b _ => hpeel x b
          _ = ∑ b : Fin (i + 1 + m + 1) → A, ∑ x : A,
                (if b ⟨i, hlt0⟩ = v then (1 : ℝ) else 0) *
                  (if b ⟨i + 1, hlt1⟩ = w then (1 : ℝ) else 0) *
                    (ch08a_chainW φ ψ (i + 1 + m) b *
                      (ψ (i + 1 + m + 1) (b (Fin.last (i + 1 + m))) x *
                        φ (i + 1 + m + 1) x * g x)) := Finset.sum_comm
          _ = ∑ b : Fin (i + 1 + m + 1) → A,
                (if b ⟨i, hlt0⟩ = v then (1 : ℝ) else 0) *
                  (if b ⟨i + 1, hlt1⟩ = w then (1 : ℝ) else 0) *
                    (ch08a_chainW φ ψ (i + 1 + m) b *
                      ∑ x : A, ψ (i + 1 + m + 1) (b (Fin.last (i + 1 + m))) x *
                        φ (i + 1 + m + 1) x * g x) := by
              refine Finset.sum_congr rfl fun b _ => ?_
              simp only [Finset.mul_sum]
          _ = ch08a_fwd φ ψ i v * (ψ (i + 1) v w * φ (i + 1) w) *
                ch08a_bwdg φ ψ
                  (fun y => ∑ x : A, ψ (i + 1 + m + 1) y x * φ (i + 1 + m + 1) x * g x)
                  m (i + 1) w :=
              ih i v w (fun y => ∑ x : A, ψ (i + 1 + m + 1) y x * φ (i + 1 + m + 1) x * g x)
          _ = ch08a_fwd φ ψ i v * (ψ (i + 1) v w * φ (i + 1) w) *
                ch08a_bwdg φ ψ g (m + 1) (i + 1) w := by
              rw [ch08a_bwdg_succ_last]
      exact hmain

/-- The exact single-site marginal of the chain is the product of the two messages. -/
theorem ch08a_estep_single_site [Fintype A] [DecidableEq A]
    (φ : ℕ → A → ℝ) (ψ : ℕ → A → A → ℝ) (m i : ℕ) (v : A) :
    (∑ a : Fin (i + m + 1) → A,
        (if a ⟨i, by omega⟩ = v then (1 : ℝ) else 0) * ch08a_chainW φ ψ (i + m) a)
      = ch08a_fwd φ ψ i v * ch08a_bwd φ ψ m i v := by
  calc (∑ a : Fin (i + m + 1) → A,
          (if a ⟨i, by omega⟩ = v then (1 : ℝ) else 0) * ch08a_chainW φ ψ (i + m) a)
      = ∑ a : Fin (i + m + 1) → A,
          (if a ⟨i, by omega⟩ = v then (1 : ℝ) else 0) * (ch08a_chainW φ ψ (i + m) a * 1) :=
        Finset.sum_congr rfl fun a _ => by ring
    _ = ch08a_fwd φ ψ i v * ch08a_bwdg φ ψ (fun _ => 1) m i v :=
        ch08a_estep_exact φ ψ m i v (fun _ => 1)
    _ = ch08a_fwd φ ψ i v * ch08a_bwd φ ψ m i v := rfl

/-- The exact pairwise marginal of two adjacent sites. -/
theorem ch08a_estep_pairwise [Fintype A] [DecidableEq A]
    (φ : ℕ → A → ℝ) (ψ : ℕ → A → A → ℝ) (m i : ℕ) (v w : A) :
    (∑ a : Fin (i + 1 + m + 1) → A,
        (if a ⟨i, by omega⟩ = v then (1 : ℝ) else 0) *
          (if a ⟨i + 1, by omega⟩ = w then (1 : ℝ) else 0) * ch08a_chainW φ ψ (i + 1 + m) a)
      = ch08a_fwd φ ψ i v * (ψ (i + 1) v w * φ (i + 1) w) * ch08a_bwd φ ψ m (i + 1) w := by
  calc (∑ a : Fin (i + 1 + m + 1) → A,
          (if a ⟨i, by omega⟩ = v then (1 : ℝ) else 0) *
            (if a ⟨i + 1, by omega⟩ = w then (1 : ℝ) else 0) * ch08a_chainW φ ψ (i + 1 + m) a)
      = ∑ a : Fin (i + 1 + m + 1) → A,
          (if a ⟨i, by omega⟩ = v then (1 : ℝ) else 0) *
            (if a ⟨i + 1, by omega⟩ = w then (1 : ℝ) else 0) *
              (ch08a_chainW φ ψ (i + 1 + m) a * 1) :=
        Finset.sum_congr rfl fun a _ => by ring
    _ = ch08a_fwd φ ψ i v * (ψ (i + 1) v w * φ (i + 1) w) *
          ch08a_bwdg φ ψ (fun _ => 1) m (i + 1) w :=
        ch08a_estep_exact_pair φ ψ m i v w (fun _ => 1)
    _ = ch08a_fwd φ ψ i v * (ψ (i + 1) v w * φ (i + 1) w) * ch08a_bwd φ ψ m (i + 1) w := rfl

/-- **Consistency of the two sweeps**: the normalising constant read off at *any* cut is
the same number, the total weight `p_θ(x)` of the chain.  This is what lets a single pair
of sweeps serve every site at once. -/
theorem ch08a_estep_cut_consistency [Fintype A] [DecidableEq A]
    (φ : ℕ → A → ℝ) (ψ : ℕ → A → A → ℝ) (m i : ℕ) :
    ch08a_marg (ch08a_chainW φ ψ (i + m))
      = ∑ v : A, ch08a_fwd φ ψ i v * ch08a_bwd φ ψ m i v := by
  have key : ∀ a : Fin (i + m + 1) → A,
      ch08a_chainW φ ψ (i + m) a
        = ∑ v : A, (if a ⟨i, by omega⟩ = v then (1 : ℝ) else 0) *
            ch08a_chainW φ ψ (i + m) a := by
    intro a; simp
  calc ch08a_marg (ch08a_chainW φ ψ (i + m))
      = ∑ a : Fin (i + m + 1) → A, ch08a_chainW φ ψ (i + m) a := rfl
    _ = ∑ a : Fin (i + m + 1) → A, ∑ v : A,
          (if a ⟨i, by omega⟩ = v then (1 : ℝ) else 0) * ch08a_chainW φ ψ (i + m) a :=
        Finset.sum_congr rfl fun a _ => key a
    _ = ∑ v : A, ∑ a : Fin (i + m + 1) → A,
          (if a ⟨i, by omega⟩ = v then (1 : ℝ) else 0) * ch08a_chainW φ ψ (i + m) a :=
        Finset.sum_comm
    _ = ∑ v : A, ch08a_fwd φ ψ i v * ch08a_bwd φ ψ m i v :=
        Finset.sum_congr rfl fun v _ => ch08a_estep_single_site φ ψ m i v

/-- The exact single-site **posterior** marginal: the two messages, normalised by the
constant the same two sweeps supply. -/
theorem ch08a_estep_single_site_post [Fintype A] [DecidableEq A]
    (φ : ℕ → A → ℝ) (ψ : ℕ → A → A → ℝ) (m i : ℕ) (v : A) :
    (∑ a : Fin (i + m + 1) → A,
        (if a ⟨i, by omega⟩ = v then (1 : ℝ) else 0) *
          ch08a_post (ch08a_chainW φ ψ (i + m)) a)
      = ch08a_fwd φ ψ i v * ch08a_bwd φ ψ m i v /
          (∑ u : A, ch08a_fwd φ ψ i u * ch08a_bwd φ ψ m i u) := by
  have h1 : ∀ a : Fin (i + m + 1) → A,
      (if a ⟨i, by omega⟩ = v then (1 : ℝ) else 0) * ch08a_post (ch08a_chainW φ ψ (i + m)) a
        = ((if a ⟨i, by omega⟩ = v then (1 : ℝ) else 0) * ch08a_chainW φ ψ (i + m) a) /
            ch08a_marg (ch08a_chainW φ ψ (i + m)) := by
    intro a; rw [ch08a_bayes]; ring
  rw [Finset.sum_congr rfl (fun a (_ : a ∈ Finset.univ) => h1 a), ← Finset.sum_div,
    ch08a_estep_single_site, ch08a_estep_cut_consistency]

/-- **Every conditional expectation EM needs is exact.**  For any function `h` of two
adjacent site values, its exact posterior expectation is read off from the messages at
that one transition; no variational family and no Monte Carlo sample enters anywhere.
Applied to the summand of eq:em-Q (see `ch08a_estep_exact_mixture`) this is the
chapter's proposition. -/
theorem ch08a_estep_pair_expectation [Fintype A] [DecidableEq A]
    (φ : ℕ → A → ℝ) (ψ : ℕ → A → A → ℝ) (m i : ℕ) (h : A → A → ℝ) :
    (∑ a : Fin (i + 1 + m + 1) → A,
        ch08a_post (ch08a_chainW φ ψ (i + 1 + m)) a *
          h (a ⟨i, by omega⟩) (a ⟨i + 1, by omega⟩))
      = (∑ v : A, ∑ w : A, h v w *
            (ch08a_fwd φ ψ i v * (ψ (i + 1) v w * φ (i + 1) w) * ch08a_bwd φ ψ m (i + 1) w)) /
          ch08a_marg (ch08a_chainW φ ψ (i + 1 + m)) := by
  have hins : ∀ a : Fin (i + 1 + m + 1) → A,
      ch08a_chainW φ ψ (i + 1 + m) a * h (a ⟨i, by omega⟩) (a ⟨i + 1, by omega⟩)
        = ∑ v : A, ∑ w : A,
            ((if a ⟨i, by omega⟩ = v then (1 : ℝ) else 0) *
              (if a ⟨i + 1, by omega⟩ = w then (1 : ℝ) else 0) *
                ch08a_chainW φ ψ (i + 1 + m) a) * h v w := by
    intro a
    have inner : ∀ v : A,
        (∑ w : A, ((if a ⟨i, by omega⟩ = v then (1 : ℝ) else 0) *
            (if a ⟨i + 1, by omega⟩ = w then (1 : ℝ) else 0) *
              ch08a_chainW φ ψ (i + 1 + m) a) * h v w)
          = (if a ⟨i, by omega⟩ = v then (1 : ℝ) else 0) *
              (ch08a_chainW φ ψ (i + 1 + m) a * h v (a ⟨i + 1, by omega⟩)) := by
      intro v
      have hw : ∀ w : A,
          ((if a ⟨i, by omega⟩ = v then (1 : ℝ) else 0) *
            (if a ⟨i + 1, by omega⟩ = w then (1 : ℝ) else 0) *
              ch08a_chainW φ ψ (i + 1 + m) a) * h v w
            = if a ⟨i + 1, by omega⟩ = w then
                ((if a ⟨i, by omega⟩ = v then (1 : ℝ) else 0) *
                  (ch08a_chainW φ ψ (i + 1 + m) a * h v w)) else 0 := by
        intro w
        by_cases hcase : a ⟨i + 1, by omega⟩ = w <;> simp [hcase]
      rw [Finset.sum_congr rfl (fun w (_ : w ∈ Finset.univ) => hw w)]
      simp
    rw [Finset.sum_congr rfl (fun v (_ : v ∈ Finset.univ) => inner v)]
    simp
  have hdiv : ∀ a : Fin (i + 1 + m + 1) → A,
      ch08a_post (ch08a_chainW φ ψ (i + 1 + m)) a *
          h (a ⟨i, by omega⟩) (a ⟨i + 1, by omega⟩)
        = (ch08a_chainW φ ψ (i + 1 + m) a * h (a ⟨i, by omega⟩) (a ⟨i + 1, by omega⟩)) /
            ch08a_marg (ch08a_chainW φ ψ (i + 1 + m)) := by
    intro a; rw [ch08a_bayes]; ring
  rw [Finset.sum_congr rfl (fun a (_ : a ∈ Finset.univ) => hdiv a), ← Finset.sum_div]
  congr 1
  calc (∑ a : Fin (i + 1 + m + 1) → A,
          ch08a_chainW φ ψ (i + 1 + m) a * h (a ⟨i, by omega⟩) (a ⟨i + 1, by omega⟩))
      = ∑ a : Fin (i + 1 + m + 1) → A, ∑ v : A, ∑ w : A,
          ((if a ⟨i, by omega⟩ = v then (1 : ℝ) else 0) *
            (if a ⟨i + 1, by omega⟩ = w then (1 : ℝ) else 0) *
              ch08a_chainW φ ψ (i + 1 + m) a) * h v w :=
        Finset.sum_congr rfl fun a _ => hins a
    _ = ∑ v : A, ∑ a : Fin (i + 1 + m + 1) → A, ∑ w : A,
          ((if a ⟨i, by omega⟩ = v then (1 : ℝ) else 0) *
            (if a ⟨i + 1, by omega⟩ = w then (1 : ℝ) else 0) *
              ch08a_chainW φ ψ (i + 1 + m) a) * h v w := Finset.sum_comm
    _ = ∑ v : A, ∑ w : A, ∑ a : Fin (i + 1 + m + 1) → A,
          ((if a ⟨i, by omega⟩ = v then (1 : ℝ) else 0) *
            (if a ⟨i + 1, by omega⟩ = w then (1 : ℝ) else 0) *
              ch08a_chainW φ ψ (i + 1 + m) a) * h v w :=
        Finset.sum_congr rfl fun v _ => Finset.sum_comm
    _ = ∑ v : A, ∑ w : A, h v w *
          (ch08a_fwd φ ψ i v * (ψ (i + 1) v w * φ (i + 1) w) * ch08a_bwd φ ψ m (i + 1) w) := by
        refine Finset.sum_congr rfl fun v _ => Finset.sum_congr rfl fun w _ => ?_
        rw [← Finset.sum_mul, ch08a_estep_pairwise]
        ring

end Estep

/-- prop:em-estep for the chapter's own model.  Putting the mixture label inside the
alphabet, `A = (grid) × (Fin C)`, the label-augmented chain of eq:em-complete-kernel *is*
a chain factor graph: its transition factor is `K̃_θ(a_i, c_i | a_{i-1})` (which does not
depend on `c_{i-1}`) and its site factor is the fixed observation factor `φ_i`.  So the
exact posterior expectation of the eq:em-Q summand at a transition -- the only quantity
the M-step consumes -- is the normalised forward × transition × backward product,
produced by one forward and one backward sweep. -/
theorem ch08a_estep_exact_mixture {C : ℕ} {A₀ : Type*} [Fintype A₀] [DecidableEq A₀]
    (grid : A₀ → ℝ) (α : ℝ) (p ν s : Fin C → ℝ) (obs : ℕ → ℝ → ℝ) (m i : ℕ) (c : Fin C) :
    (∑ a : Fin (i + 1 + m + 1) → A₀ × Fin C,
        ch08a_post (ch08a_chainW (fun (j : ℕ) (r : A₀ × Fin C) => obs j (grid r.1))
            (fun (_ : ℕ) (q r : A₀ × Fin C) =>
              ch08a_ktilde α p ν s (grid r.1) (grid q.1) r.2) (i + 1 + m)) a *
          ch08a_Qsummand α p ν s (grid (a ⟨i + 1, by omega⟩).1) (grid (a ⟨i, by omega⟩).1)
            (a ⟨i + 1, by omega⟩).2 c)
      = (∑ u : A₀ × Fin C, ∑ z : A₀ × Fin C,
            ch08a_Qsummand α p ν s (grid z.1) (grid u.1) z.2 c *
              (ch08a_fwd (fun (j : ℕ) (r : A₀ × Fin C) => obs j (grid r.1))
                  (fun (_ : ℕ) (q r : A₀ × Fin C) =>
                    ch08a_ktilde α p ν s (grid r.1) (grid q.1) r.2) i u *
                (ch08a_ktilde α p ν s (grid z.1) (grid u.1) z.2 * obs (i + 1) (grid z.1)) *
                ch08a_bwd (fun (j : ℕ) (r : A₀ × Fin C) => obs j (grid r.1))
                  (fun (_ : ℕ) (q r : A₀ × Fin C) =>
                    ch08a_ktilde α p ν s (grid r.1) (grid q.1) r.2) m (i + 1) z)) /
          ch08a_marg (ch08a_chainW (fun (j : ℕ) (r : A₀ × Fin C) => obs j (grid r.1))
            (fun (_ : ℕ) (q r : A₀ × Fin C) =>
              ch08a_ktilde α p ν s (grid r.1) (grid q.1) r.2) (i + 1 + m)) :=
  ch08a_estep_pair_expectation (fun (j : ℕ) (r : A₀ × Fin C) => obs j (grid r.1))
    (fun (_ : ℕ) (q r : A₀ × Fin C) => ch08a_ktilde α p ν s (grid r.1) (grid q.1) r.2) m i
    (fun (q r : A₀ × Fin C) => ch08a_Qsummand α p ν s (grid r.1) (grid q.1) r.2 c)

/-! ### eq:em-fisher (lines 562-574), upgraded from `instance_checked`

`∇L = E_{a,c|x,θ}[ Σ_{i=2}^N ∇_θ log K̃_θ(a_i,c_i|a_{i-1}) ]`.  The earlier entry for this
display proved only the generic calculus step "the derivative of a constant plus a finite
sum is the sum of the derivatives".  Here the display itself is proved: `L`, the
posterior expectation and the transition kernels all occur, and the chapter's reason that
only the transition factors depend on `θ` enters as the log-factorisation
eq:em-complete-ll rather than being assumed by the shape of the integrand. -/

/-- Fisher's identity in the form eq:em-fisher: if the complete-data log-likelihood is a
`θ`-independent part `K` plus transition terms `log T_i` (which is eq:em-complete-ll),
then the gradient of `L(θ) = log p_θ(x)` is the posterior expectation of the sum of the
transition scores `∇_θ log T_i = T_i'/T_i`. -/
theorem ch08a_em_fisher_transitions {Z ι : Type*} [Fintype Z] [Nonempty Z] (u : Finset ι)
    (P : ℝ → Z → ℝ) (K : Z → ℝ) (T : ι → Z → ℝ → ℝ) (T' : ι → Z → ℝ) (θ : ℝ)
    (hPpos : ∀ t z, 0 < P t z)
    (hTpos : ∀ i ∈ u, ∀ (z : Z) (t : ℝ), 0 < T i z t)
    (hlog : ∀ t z, Real.log (P t z) = K z + ∑ i ∈ u, Real.log (T i z t))
    (hd : ∀ i ∈ u, ∀ z, HasDerivAt (T i z) (T' i z) θ) :
    HasDerivAt (fun t => ch08a_L (P t))
      (∑ z, ch08a_post (P θ) z * ∑ i ∈ u, T' i z / T i z θ) θ := by
  have hscore : ∀ z, HasDerivAt (fun t => P t z) (P θ z * ∑ i ∈ u, T' i z / T i z θ) θ := by
    intro z
    have hsum : HasDerivAt (fun t => K z + ∑ i ∈ u, Real.log (T i z t))
        (∑ i ∈ u, T' i z / T i z θ) θ :=
      HasDerivAt.const_add _ (HasDerivAt.fun_sum fun i hi => (hd i hi z).log (hTpos i hi z θ).ne')
    have hexp := hsum.exp
    have hfun : (fun t => Real.exp (K z + ∑ i ∈ u, Real.log (T i z t))) = fun t => P t z := by
      funext t
      rw [← hlog t z, Real.exp_log (hPpos t z)]
    have hval : Real.exp (K z + ∑ i ∈ u, Real.log (T i z θ)) = P θ z := by
      rw [← hlog θ z, Real.exp_log (hPpos θ z)]
    rw [hfun, hval] at hexp
    exact hexp
  have hmpos : 0 < ∑ z, P θ z := Finset.sum_pos (fun z _ => hPpos θ z) Finset.univ_nonempty
  have hgen := ch08a_em_fisher_general P (fun z => P θ z * ∑ i ∈ u, T' i z / T i z θ) θ hscore
    (fun z => (hPpos θ z).ne') hmpos.ne'
  have hval : (∑ z, (P θ z / ∑ y, P θ y) * ((P θ z * ∑ i ∈ u, T' i z / T i z θ) / P θ z))
      = ∑ z, ch08a_post (P θ) z * ∑ i ∈ u, T' i z / T i z θ := by
    refine Finset.sum_congr rfl fun z _ => ?_
    have hz : P θ z ≠ 0 := (hPpos θ z).ne'
    have hcancel : (P θ z * ∑ i ∈ u, T' i z / T i z θ) / P θ z
        = ∑ i ∈ u, T' i z / T i z θ := by field_simp
    rw [hcancel, ch08a_bayes, ch08a_em_marginalisation]
  rw [hval] at hgen
  exact hgen

/-- eq:em-fisher for the chapter's chain, with `α` as the parameter: the gradient of the
marginal log-likelihood `L` is the posterior expectation of the sum, over the `N-1`
transitions `i ∈ [2,N]`, of the transition scores `∇_α log K̃_θ(a_i, c_i | a_{i-1})`.  The
initial factor `p(a_1)` and the observation factors `φ_i` drop out because they do not
depend on `α`. -/
theorem ch08a_em_fisher_chain {C : ℕ} {Z : Type*} [Fintype Z] [Nonempty Z] (N : ℕ)
    (Acfg : Z → ℕ → ℝ) (lab : Z → ℕ → Fin C) (p ν s : Fin C → ℝ) (p1 : ℝ → ℝ)
    (φ : ℕ → ℝ → ℝ) (D : ℕ → Z → ℝ) (α : ℝ)
    (hp : ∀ c, 0 < p c) (hs : ∀ c, 0 < s c)
    (hp1 : ∀ z, 0 < p1 (Acfg z 1))
    (hφ : ∀ z, ∀ i ∈ Finset.Icc 1 N, 0 < φ i (Acfg z i))
    (hd : ∀ i ∈ Finset.Icc 2 N, ∀ z,
      HasDerivAt (fun t => ch08a_ktilde t p ν s (Acfg z i) (Acfg z (i - 1)) (lab z i))
        (D i z) α) :
    HasDerivAt (fun t => ch08a_L (fun z => ch08a_joint N t p ν s p1 φ (Acfg z) (lab z)))
      (∑ z, ch08a_post (fun z => ch08a_joint N α p ν s p1 φ (Acfg z) (lab z)) z *
        ∑ i ∈ Finset.Icc 2 N,
          D i z / ch08a_ktilde α p ν s (Acfg z i) (Acfg z (i - 1)) (lab z i)) α := by
  refine ch08a_em_fisher_transitions (Finset.Icc 2 N)
    (fun t z => ch08a_joint N t p ν s p1 φ (Acfg z) (lab z))
    (fun z => Real.log (p1 (Acfg z 1)) + ∑ i ∈ Finset.Icc 1 N, Real.log (φ i (Acfg z i)))
    (fun i z t => ch08a_ktilde t p ν s (Acfg z i) (Acfg z (i - 1)) (lab z i)) D α ?_ ?_ ?_ hd
  · intro t z
    exact ch08a_joint_pos N t p ν s p1 φ (Acfg z) (lab z) (hp1 z) hp hs (hφ z)
  · intro i _ z t
    exact ch08a_ktilde_pos t p ν s (Acfg z i) (Acfg z (i - 1)) (lab z i) (hp _) (hs _)
  · intro t z
    rw [ch08a_em_complete_ll N t p ν s p1 φ (Acfg z) (lab z) (hp1 z).ne'
      (fun i _ => (ch08a_ktilde_pos t p ν s (Acfg z i) (Acfg z (i - 1)) (lab z i)
        (hp _) (hs _)).ne')
      (fun i hi => (hφ z i hi).ne')]
    ring


end

end ThesisAudit
