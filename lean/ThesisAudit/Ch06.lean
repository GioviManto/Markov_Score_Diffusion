import Mathlib

/-!
Audit of the display-math formulas in
`thesis/chapters/ch06-gaussian-bp.tex`, lines 1-779
("The Gaussian Chain by Belief Propagation").

Conventions used throughout this file:

* Gaussian *kernels* (unnormalised, `exp (-(x-m)^2/(2v))`) are used wherever the
  text writes `∝`, since every message in the chapter is only defined up to a
  positive constant.  `ch06_gauss` is the normalised density when a
  normalisation is actually needed.
* The natural (canonical) parametrisation of \eqref{eq:bp-natural-message},
  `exp (-(1/2) λ a² + r a)`, is `ch06_nat λ r`.  The two message recursions are
  proved as genuine Lebesgue integrals over `ℝ` (`ch06_bp_forward_closure`,
  `ch06_bp_backward_closure`) via `ch06_gauss_integral`.
* The *structural* sum-product statements (splitting the posterior at a site,
  forward/backward partial marginals, the combination rule) are checked in the
  discrete alphabet setting, where the Fubini step is a finite `Finset.sum_comm`
  and no integrability side conditions arise.  The chapter itself states the
  discrete form in the "Computational cost" paragraph, so this is the same
  algebra.  Sizes of the checked instances are always recorded in the comment.
* Asymptotic complexity claims are recorded as explicit operation-count models
  with a proved `O(·)` bound; the model is stated in the comment.
-/

namespace ThesisAudit

open Real MeasureTheory

noncomputable section

/-! ### Shared definitions -/

/-- `Δ_t = 1 - e^{-2t}` (Chapter 5 convention, used throughout Chapter 6). -/
def ch06_Delta (t : ℝ) : ℝ := 1 - Real.exp (-2 * t)

/-- The normalised scalar Gaussian density `N(x; m, v)`. -/
def ch06_gauss (m v x : ℝ) : ℝ :=
  Real.exp (-(x - m) ^ 2 / (2 * v)) / Real.sqrt (2 * Real.pi * v)

theorem ch06_gauss_pos {m v x : ℝ} (hv : 0 < v) : 0 < ch06_gauss m v x := by
  have h : 0 < Real.sqrt (2 * Real.pi * v) := Real.sqrt_pos.mpr (by positivity)
  exact div_pos (Real.exp_pos _) h

/-- eq:bp-natural-message (ch06 lines 337-344): an unnormalised scalar Gaussian in
natural form, `μ(a) ∝ exp(-½ λ a² + r a)`, with `λ` the precision and `r` the field. -/
def ch06_nat (lam r a : ℝ) : ℝ := Real.exp (-(1 / 2) * lam * a ^ 2 + r * a)

theorem ch06_nat_pos (lam r a : ℝ) : 0 < ch06_nat lam r a := Real.exp_pos _

/-- Multiplying two messages adds their natural parameters (the algebraic core of
eq:bp-product-update and eq:bp-belief-precision/field). -/
theorem ch06_nat_mul (l₁ r₁ l₂ r₂ a : ℝ) :
    ch06_nat l₁ r₁ a * ch06_nat l₂ r₂ a = ch06_nat (l₁ + l₂) (r₁ + r₂) a := by
  simp only [ch06_nat, ← Real.exp_add]
  congr 1
  ring

/-! ### The Gaussian integral with a linear term

The workhorse behind both closure computations: completing the square and
translating the Lebesgue integral. -/

theorem ch06_gauss_integral {A : ℝ} (hA : 0 < A) (B : ℝ) :
    (∫ u : ℝ, Real.exp (-(A / 2) * u ^ 2 + B * u))
      = Real.exp (B ^ 2 / (2 * A)) * Real.sqrt (Real.pi / (A / 2)) := by
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

/-! ### Section 6.1 — the posterior and its factor graph -/

/-- The prior factor `ψ_pr(a₀) = N(a₀; 0, 1)` of eq:bp-posterior. -/
def ch06_psiPr (a₀ : ℝ) : ℝ := ch06_gauss 0 1 a₀

/-- The transition factor `ψ_tr^{(k)}(a_k, a_{k+1}) = N(a_{k+1}; α a_k, σ_η²)`. -/
def ch06_psiTr (α σ2 : ℝ) (u w : ℝ) : ℝ := ch06_gauss (α * u) σ2 w

/-- The observation factor `ψ_ob^{(k)}(a_k) = N(x_k; e^{-t} a_k, Δ_t)`. -/
def ch06_psiOb (t : ℝ) (xk ak : ℝ) : ℝ :=
  ch06_gauss (Real.exp (-t) * ak) (ch06_Delta t) xk

-- eq:bp-posterior (ch06 lines 36-46): the posterior of the hidden Markov model is
-- proportional to prior x transitions x observations.
def ch06_bp_posterior (L : ℕ) (α σ2 t : ℝ) (x a : ℕ → ℝ) : ℝ :=
  ch06_psiPr (a 0)
    * (∏ k ∈ Finset.range (L - 1), ch06_psiTr α σ2 (a k) (a (k + 1)))
    * (∏ k ∈ Finset.range L, ch06_psiOb t (x k) (a k))

/-- eq:bp-posterior, sanity: the stated product *is* (chain prior) x (per-frame
likelihoods) — Bayes' rule with a likelihood that never couples two frames. -/
theorem ch06_bp_posterior_eq_prior_mul_lik (L : ℕ) (α σ2 t : ℝ) (x a : ℕ → ℝ) :
    ch06_bp_posterior L α σ2 t x a
      = (ch06_psiPr (a 0) * ∏ k ∈ Finset.range (L - 1), ch06_psiTr α σ2 (a k) (a (k + 1)))
        * (∏ k ∈ Finset.range L, ch06_psiOb t (x k) (a k)) := rfl

/-- eq:bp-posterior, sanity: the unnormalised posterior is strictly positive. -/
theorem ch06_bp_posterior_pos {L : ℕ} {α σ2 t : ℝ} (x a : ℕ → ℝ)
    (hσ : 0 < σ2) (hD : 0 < ch06_Delta t) :
    0 < ch06_bp_posterior L α σ2 t x a := by
  unfold ch06_bp_posterior
  have h1 : 0 < ch06_psiPr (a 0) := ch06_gauss_pos (by norm_num)
  have h2 : 0 < ∏ k ∈ Finset.range (L - 1), ch06_psiTr α σ2 (a k) (a (k + 1)) :=
    Finset.prod_pos fun k _ => ch06_gauss_pos hσ
  have h3 : 0 < ∏ k ∈ Finset.range L, ch06_psiOb t (x k) (a k) :=
    Finset.prod_pos fun k _ => ch06_gauss_pos hD
  positivity

/-! ### Section 6.2 — the two sweeps (continuous form) -/

/-- eq:bp-fwd (ch06 lines 88-94): the forward message recursion, transcribed as a
Lebesgue integral.  `ch06_msgFwd ψpr ψtr ψob k` is `μ_{→k}`. -/
def ch06_msgFwd (ψpr : ℝ → ℝ) (ψtr : ℕ → ℝ → ℝ → ℝ) (ψob : ℕ → ℝ → ℝ) : ℕ → ℝ → ℝ
  | 0 => ψpr
  | (k + 1) => fun w => ∫ u : ℝ, ψtr k u w * ψob k u * ch06_msgFwd ψpr ψtr ψob k u

/-- noname-1 (ch06 lines 193-195): `μ_{→0} = ψ_pr`. -/
theorem ch06_msgFwd_zero (ψpr : ℝ → ℝ) (ψtr : ℕ → ℝ → ℝ → ℝ) (ψob : ℕ → ℝ → ℝ) :
    ch06_msgFwd ψpr ψtr ψob 0 = ψpr := rfl

/-- noname-2 (ch06 lines 198-206): the forward recursion, one step. -/
theorem ch06_msgFwd_succ (ψpr : ℝ → ℝ) (ψtr : ℕ → ℝ → ℝ → ℝ) (ψob : ℕ → ℝ → ℝ)
    (k : ℕ) (w : ℝ) :
    ch06_msgFwd ψpr ψtr ψob (k + 1) w
      = ∫ u : ℝ, ψtr k u w * ψob k u * ch06_msgFwd ψpr ψtr ψob k u := rfl

/-- eq:bp-bwd (ch06 lines 95-101): the backward message recursion.
`ch06_msgBwd ψtr ψob L j` is `μ_{←(L-1-j)}`, i.e. the message `j` hops in from the
right end of a length-`L` chain. -/
def ch06_msgBwd (ψtr : ℕ → ℝ → ℝ → ℝ) (ψob : ℕ → ℝ → ℝ) (L : ℕ) : ℕ → ℝ → ℝ
  | 0 => fun _ => 1
  | (j + 1) => fun b =>
      ∫ u : ℝ, ψtr (L - 2 - j) b u * ψob (L - 1 - j) u * ch06_msgBwd ψtr ψob L j u

/-- noname-3 (ch06 lines 222-224): `μ_{←(L-1)} = 1`. -/
theorem ch06_msgBwd_zero (ψtr : ℕ → ℝ → ℝ → ℝ) (ψob : ℕ → ℝ → ℝ) (L : ℕ) :
    ch06_msgBwd ψtr ψob L 0 = fun _ => 1 := rfl

/-- noname-4 (ch06 lines 226-234): the backward recursion, one step. -/
theorem ch06_msgBwd_succ (ψtr : ℕ → ℝ → ℝ → ℝ) (ψob : ℕ → ℝ → ℝ) (L j : ℕ) (b : ℝ) :
    ch06_msgBwd ψtr ψob L (j + 1) b
      = ∫ u : ℝ, ψtr (L - 2 - j) b u * ψob (L - 1 - j) u * ch06_msgBwd ψtr ψob L j u := rfl

/-! ### Section 6.2 — the two sweeps (discrete form, where Fubini is finite)

This is the setting of the "Computational cost" paragraph (ch06 lines 266-295):
each `a_k` ranges over a finite alphabet `S`, and every integral is a finite sum. -/

section Discrete

variable {S : Type*} [Fintype S]

/-- noname-6 (ch06 lines 269-276): the discrete forward update.
`ch06_fwdD ψpr ψtr ψob k` is `μ_{→k}`. -/
def ch06_fwdD (ψpr : S → ℝ) (ψtr : ℕ → S → S → ℝ) (ψob : ℕ → S → ℝ) : ℕ → S → ℝ
  | 0 => ψpr
  | (k + 1) => fun w => ∑ u : S, ψtr k u w * ψob k u * ch06_fwdD ψpr ψtr ψob k u

/-- Discrete backward message: `ch06_bwdD ψtr ψob L j` is `μ_{←(L-1-j)}`. -/
def ch06_bwdD (ψtr : ℕ → S → S → ℝ) (ψob : ℕ → S → ℝ) (L : ℕ) : ℕ → S → ℝ
  | 0 => fun _ => 1
  | (j + 1) => fun b =>
      ∑ u : S, ψtr (L - 2 - j) b u * ψob (L - 1 - j) u * ch06_bwdD ψtr ψob L j u

/-- eq:bp-fwd-partial (ch06 lines 180-191) at `k = 2`: the recursively defined
forward message equals the explicit partial marginalisation over `a₀, a₁`. -/
theorem ch06_bp_fwd_partial_two (ψpr : S → ℝ) (ψtr : ℕ → S → S → ℝ) (ψob : ℕ → S → ℝ)
    (c : S) :
    ch06_fwdD ψpr ψtr ψob 2 c
      = ∑ a₁ : S, ∑ a₀ : S,
          ψpr a₀ * (ψtr 0 a₀ a₁ * ψtr 1 a₁ c) * (ψob 0 a₀ * ψob 1 a₁) := by
  simp only [ch06_fwdD]
  refine Finset.sum_congr rfl fun a₁ _ => ?_
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun a₀ _ => by ring

/-- eq:bp-fwd-partial at `k = 3`: partial marginalisation over `a₀, a₁, a₂`. -/
theorem ch06_bp_fwd_partial_three (ψpr : S → ℝ) (ψtr : ℕ → S → S → ℝ) (ψob : ℕ → S → ℝ)
    (c : S) :
    ch06_fwdD ψpr ψtr ψob 3 c
      = ∑ a₂ : S, ∑ a₁ : S, ∑ a₀ : S,
          ψpr a₀ * (ψtr 0 a₀ a₁ * ψtr 1 a₁ a₂ * ψtr 2 a₂ c)
            * (ψob 0 a₀ * ψob 1 a₁ * ψob 2 a₂) := by
  simp only [ch06_fwdD, Finset.mul_sum]
  exact Finset.sum_congr rfl fun a₂ _ =>
    Finset.sum_congr rfl fun a₁ _ => Finset.sum_congr rfl fun a₀ _ => by ring

/-- eq:bp-bwd-partial (ch06 lines 210-220) at `L = 3`, `k = 1`: the recursively
defined backward message equals the explicit partial marginalisation over `a₂`. -/
theorem ch06_bp_bwd_partial_three (ψtr : ℕ → S → S → ℝ) (ψob : ℕ → S → ℝ) (a₁ : S) :
    ch06_bwdD ψtr ψob 3 1 a₁ = ∑ a₂ : S, ψtr 1 a₁ a₂ * ψob 2 a₂ := by
  simp only [ch06_bwdD]
  exact Finset.sum_congr rfl fun a₂ _ => by norm_num

/-- eq:bp-bwd-partial at `L = 4`, `k = 1`: marginalisation over `a₂, a₃`. -/
theorem ch06_bp_bwd_partial_four (ψtr : ℕ → S → S → ℝ) (ψob : ℕ → S → ℝ) (a₁ : S) :
    ch06_bwdD ψtr ψob 4 2 a₁
      = ∑ a₂ : S, ∑ a₃ : S,
          (ψtr 1 a₁ a₂ * ψtr 2 a₂ a₃) * (ψob 2 a₂ * ψob 3 a₃) := by
  simp only [ch06_bwdD, Finset.mul_sum]
  refine Finset.sum_congr rfl fun a₂ _ => Finset.sum_congr rfl fun a₃ _ => ?_
  norm_num
  all_goals ring

/-- eq:bp-combine / eq:bp-combine-proof / eq:bp-split-integrals / noname-16
(ch06 lines 104-110, 150-176, 239-248, 491-497) at `L = 3`, `k = 1`: summing the
posterior over every variable but `a₁` factorises as
`μ_{→1}(a₁) · ψ_ob^{(1)}(a₁) · μ_{←1}(a₁)`. -/
theorem ch06_bp_combine_three (ψpr : S → ℝ) (ψtr : ℕ → S → S → ℝ) (ψob : ℕ → S → ℝ)
    (a₁ : S) :
    (∑ a₂ : S, ∑ a₀ : S,
        ψpr a₀ * ψtr 0 a₀ a₁ * ψtr 1 a₁ a₂ * ψob 0 a₀ * ψob 1 a₁ * ψob 2 a₂)
      = ch06_fwdD ψpr ψtr ψob 1 a₁ * ψob 1 a₁ * ch06_bwdD ψtr ψob 3 1 a₁ := by
  rw [ch06_bp_bwd_partial_three]
  simp only [ch06_fwdD, Finset.sum_mul, Finset.mul_sum]
  exact Finset.sum_congr rfl fun a₂ _ => Finset.sum_congr rfl fun a₀ _ => by ring

/-- eq:bp-split-integrals (ch06 lines 150-176), general form.  Once `a_k` is held
fixed, the variables to its left and the variables to its right occur in disjoint
factors (eq:bp-split-at-k), so the marginalisation over all of them separates into a
left bracket, the local observation, and a right bracket.  `Lft` and `Rgt` are the
configuration types of the two variable blocks, so this covers every `L` and every
site `k`; in the discrete setting the Fubini step is this finite identity. -/
theorem ch06_bp_sum_factorises {Lft Rgt : Type*} [Fintype Lft] [Fintype Rgt]
    (F : Lft → ℝ) (g : ℝ) (H : Rgt → ℝ) :
    (∑ l : Lft, ∑ r : Rgt, F l * g * H r)
      = (∑ l : Lft, F l) * g * (∑ r : Rgt, H r) := by
  rw [Finset.sum_mul, Finset.sum_mul]
  refine Finset.sum_congr rfl fun l _ => ?_
  rw [Finset.mul_sum]

/-- eq:bp-combine at `L = 4`, `k = 2`: a site with two variables on its left and one
on its right.  Same conclusion as `ch06_bp_combine_three`. -/
theorem ch06_bp_combine_four (ψpr : S → ℝ) (ψtr : ℕ → S → S → ℝ) (ψob : ℕ → S → ℝ)
    (a₂ : S) :
    (∑ a₃ : S, ∑ a₁ : S, ∑ a₀ : S,
        ψpr a₀ * ψtr 0 a₀ a₁ * ψtr 1 a₁ a₂ * ψtr 2 a₂ a₃
          * ψob 0 a₀ * ψob 1 a₁ * ψob 2 a₂ * ψob 3 a₃)
      = ch06_fwdD ψpr ψtr ψob 2 a₂ * ψob 2 a₂ * ch06_bwdD ψtr ψob 4 1 a₂ := by
  have hb : ch06_bwdD ψtr ψob 4 1 a₂ = ∑ a₃ : S, ψtr 2 a₂ a₃ * ψob 3 a₃ := by
    simp only [ch06_bwdD]
    exact Finset.sum_congr rfl fun a₃ _ => by norm_num
  rw [hb]
  simp only [ch06_fwdD, Finset.sum_mul, Finset.mul_sum]
  exact Finset.sum_congr rfl fun a₃ _ =>
    Finset.sum_congr rfl fun a₁ _ => Finset.sum_congr rfl fun a₀ _ => by ring

/-- noname-5 (ch06 lines 250-260): the forward message into site `k` depends only on
the observations `x₀, …, x_{k-1}`.  Proved in full generality by induction. -/
theorem ch06_bp_fwd_evidence (ψpr : S → ℝ) (ψtr : ℕ → S → S → ℝ)
    (ψob ψob' : ℕ → S → ℝ) (k : ℕ) (h : ∀ j, j < k → ψob j = ψob' j) :
    ch06_fwdD ψpr ψtr ψob k = ch06_fwdD ψpr ψtr ψob' k := by
  induction k with
  | zero => rfl
  | succ k ih =>
      have hk : ∀ j, j < k → ψob j = ψob' j := fun j hj => h j (Nat.lt_succ_of_lt hj)
      funext w
      simp only [ch06_fwdD, ih hk, h k (Nat.lt_succ_self k)]

/-- noname-5, right half: the backward message `j` hops in from the right depends
only on the observations at the `j` right-most sites it has crossed.  Proved in
full generality by induction. -/
theorem ch06_bp_bwd_evidence (ψtr : ℕ → S → S → ℝ) (ψob ψob' : ℕ → S → ℝ)
    (L j : ℕ) (h : ∀ i, i < j → ψob (L - 1 - i) = ψob' (L - 1 - i)) :
    ch06_bwdD ψtr ψob L j = ch06_bwdD ψtr ψob' L j := by
  induction j with
  | zero => rfl
  | succ j ih =>
      have hj : ∀ i, i < j → ψob (L - 1 - i) = ψob' (L - 1 - i) :=
        fun i hi => h i (Nat.lt_succ_of_lt hi)
      funext b
      simp only [ch06_bwdD, ih hj, h j (Nat.lt_succ_self j)]

end Discrete

/-! ### eq:bp-fwd-partial, eq:bp-bwd-partial and eq:bp-combine, general in `L` and `k`

The partial marginalisations of eq:bp-fwd-partial / eq:bp-bwd-partial integrate over
*all* configurations of the integrated variables, i.e. (discretely) over `Fin k → S`.
`ch06_ext` and `ch06_pathB` turn such a tuple into the path `a₀, a₁, …` that the
factors of eq:bp-posterior are evaluated on, so the sums below are literal
transcriptions of the two displays. -/

section DiscreteGeneral

variable {S : Type*} [Fintype S]

/-- The path `a₀, …, a_{k-1}, w`: the `k` integrated variables of eq:bp-fwd-partial
followed by the free endpoint `a_k = w` (and `w` beyond, which no factor reads). -/
def ch06_ext (k : ℕ) (a : Fin k → S) (w : S) : ℕ → S :=
  fun i => if h : i < k then a ⟨i, h⟩ else w

theorem ch06_ext_of_lt {k : ℕ} (a : Fin k → S) (w : S) {i : ℕ} (h : i < k) :
    ch06_ext k a w i = a ⟨i, h⟩ := dif_pos h

theorem ch06_ext_of_ge {k : ℕ} (a : Fin k → S) (w : S) {i : ℕ} (h : k ≤ i) :
    ch06_ext k a w i = w := dif_neg (Nat.not_lt.mpr h)

/-- The path `a_k = b, a_{k+1}, …, a_{k+j}`: the free endpoint `b` followed by the `j`
integrated variables of eq:bp-bwd-partial. -/
def ch06_pathB (j : ℕ) (a : Fin j → S) (b : S) : ℕ → S
  | 0 => b
  | (i + 1) => if h : i < j then a ⟨i, h⟩ else b

theorem ch06_pathB_zero {j : ℕ} (a : Fin j → S) (b : S) : ch06_pathB j a b 0 = b := rfl

theorem ch06_pathB_succ {j : ℕ} (a : Fin j → S) (b : S) {i : ℕ} (h : i < j) :
    ch06_pathB j a b (i + 1) = a ⟨i, h⟩ := dif_pos h

/-- Peeling the *last* integrated variable off a `(k+1)`-tuple: this is the step that
separates `a_k` in the derivation of eq:bp-fwd. -/
def ch06_joinLast (k : ℕ) : S × (Fin k → S) ≃ (Fin (k + 1) → S) where
  toFun p := fun i => ch06_ext k p.2 p.1 (i : ℕ)
  invFun a := (a ⟨k, Nat.lt_succ_self k⟩, fun i : Fin k => a ⟨(i : ℕ), Nat.lt_succ_of_lt i.isLt⟩)
  left_inv := by
    rintro ⟨u, l⟩
    simp only [Prod.mk.injEq]
    exact ⟨ch06_ext_of_ge _ _ (le_refl k), funext fun i => ch06_ext_of_lt _ _ i.isLt⟩
  right_inv := by
    intro a
    funext i
    rcases i with ⟨v, hv⟩
    by_cases h : v < k
    · exact ch06_ext_of_lt _ _ h
    · have hvk : v = k := by omega
      subst hvk
      exact ch06_ext_of_ge _ _ (le_refl _)

/-- Peeling the *first* integrated variable off a `(j+1)`-tuple: the step that
separates `a_{k+1}` in the derivation of eq:bp-bwd. -/
def ch06_joinFirst (j : ℕ) : S × (Fin j → S) ≃ (Fin (j + 1) → S) where
  toFun p := fun i => ch06_pathB j p.2 p.1 (i : ℕ)
  invFun a := (a ⟨0, Nat.succ_pos j⟩, fun i : Fin j => a ⟨(i : ℕ) + 1, by omega⟩)
  left_inv := by
    rintro ⟨u, c⟩
    simp only [Prod.mk.injEq]
    exact ⟨rfl, funext fun i => ch06_pathB_succ _ _ i.isLt⟩
  right_inv := by
    intro a
    funext i
    rcases i with ⟨v, hv⟩
    cases v with
    | zero => rfl
    | succ t => exact ch06_pathB_succ _ _ (by omega)

theorem ch06_sum_split_last (k : ℕ) (F : (Fin (k + 1) → S) → ℝ) :
    (∑ a : Fin (k + 1) → S, F a)
      = ∑ u : S, ∑ l : Fin k → S, F (fun i => ch06_ext k l u (i : ℕ)) := by
  rw [← Equiv.sum_comp (ch06_joinLast k) F, Fintype.sum_prod_type]
  rfl

theorem ch06_sum_split_first (j : ℕ) (F : (Fin (j + 1) → S) → ℝ) :
    (∑ a : Fin (j + 1) → S, F a)
      = ∑ u : S, ∑ c : Fin j → S, F (fun i => ch06_pathB j c u (i : ℕ)) := by
  rw [← Equiv.sum_comp (ch06_joinFirst j) F, Fintype.sum_prod_type]
  rfl

/-- eq:bp-fwd-partial (ch06 lines 180-191), verbatim and for every `k`: the forward
message as `∫ ψ_pr ∏_{j<k} ψ_tr^{(j)} ∏_{j<k} ψ_ob^{(j)} da₀ ⋯ da_{k-1}`, the
integral being a sum over all configurations of `a₀,…,a_{k-1}`. -/
def ch06_fwdPartial (ψpr : S → ℝ) (ψtr : ℕ → S → S → ℝ) (ψob : ℕ → S → ℝ)
    (k : ℕ) (w : S) : ℝ :=
  ∑ a : Fin k → S,
    ψpr (ch06_ext k a w 0)
      * (∏ j ∈ Finset.range k, ψtr j (ch06_ext k a w j) (ch06_ext k a w (j + 1)))
      * (∏ j ∈ Finset.range k, ψob j (ch06_ext k a w j))

/-- noname-2 (ch06 lines 198-206), general in `k`: separating the last integration
variable `a_k` out of the partial marginal eq:bp-fwd-partial produces exactly one step
of the forward recursion eq:bp-fwd. -/
theorem ch06_fwdPartial_succ (ψpr : S → ℝ) (ψtr : ℕ → S → S → ℝ) (ψob : ℕ → S → ℝ)
    (k : ℕ) (w : S) :
    ch06_fwdPartial ψpr ψtr ψob (k + 1) w
      = ∑ u : S, ψtr k u w * ψob k u * ch06_fwdPartial ψpr ψtr ψob k u := by
  unfold ch06_fwdPartial
  rw [ch06_sum_split_last]
  refine Finset.sum_congr rfl fun u _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun l _ => ?_
  have hE : ∀ i : ℕ, i < k + 1 →
      ch06_ext (k + 1) (fun i : Fin (k + 1) => ch06_ext k l u (i : ℕ)) w i
        = ch06_ext k l u i := fun i hi => ch06_ext_of_lt _ _ hi
  have hEtop : ch06_ext (k + 1) (fun i : Fin (k + 1) => ch06_ext k l u (i : ℕ)) w (k + 1) = w :=
    ch06_ext_of_ge _ _ (le_refl (k + 1))
  have hp1 : (∏ j ∈ Finset.range k,
        ψtr j (ch06_ext (k + 1) (fun i : Fin (k + 1) => ch06_ext k l u (i : ℕ)) w j)
              (ch06_ext (k + 1) (fun i : Fin (k + 1) => ch06_ext k l u (i : ℕ)) w (j + 1)))
      = ∏ j ∈ Finset.range k, ψtr j (ch06_ext k l u j) (ch06_ext k l u (j + 1)) :=
    Finset.prod_congr rfl fun j hj => by
      simp only [Finset.mem_range] at hj
      rw [hE j (by omega), hE (j + 1) (by omega)]
  have hp2 : (∏ j ∈ Finset.range k,
        ψob j (ch06_ext (k + 1) (fun i : Fin (k + 1) => ch06_ext k l u (i : ℕ)) w j))
      = ∏ j ∈ Finset.range k, ψob j (ch06_ext k l u j) :=
    Finset.prod_congr rfl fun j hj => by
      simp only [Finset.mem_range] at hj
      rw [hE j (by omega)]
  have hek : ch06_ext k l u k = u := ch06_ext_of_ge _ _ (le_refl k)
  rw [Finset.prod_range_succ, Finset.prod_range_succ, hp1, hp2, hE 0 (by omega), hEtop,
    hE k (by omega), hek]
  ring

/-- eq:bp-fwd-partial + noname-2 (ch06 lines 180-206), general in `k` and in the
alphabet: the recursively defined forward message of eq:bp-fwd *is* the explicit
partial marginalisation of eq:bp-fwd-partial. -/
theorem ch06_bp_fwd_partial (ψpr : S → ℝ) (ψtr : ℕ → S → S → ℝ) (ψob : ℕ → S → ℝ)
    (k : ℕ) (w : S) :
    ch06_fwdD ψpr ψtr ψob k w = ch06_fwdPartial ψpr ψtr ψob k w := by
  induction k generalizing w with
  | zero =>
      unfold ch06_fwdPartial
      rw [Finset.univ_unique, Finset.sum_singleton]
      simp only [Finset.range_zero, Finset.prod_empty, mul_one]
      rw [ch06_ext_of_ge _ _ (le_refl 0)]
      rfl
  | succ k ih =>
      rw [ch06_fwdPartial_succ]
      simp only [ch06_fwdD]
      exact Finset.sum_congr rfl fun u _ => by rw [ih u]

/-- eq:bp-bwd-partial (ch06 lines 210-220), verbatim and for every `k` and `j`: the
site-`k` backward message of a chain whose last site is `k + j`, as the integral of
`∏_{i=k}^{k+j-1} ψ_tr^{(i)} ∏_{i=k+1}^{k+j} ψ_ob^{(i)}` over `a_{k+1},…,a_{k+j}`. -/
def ch06_bwdPartial (ψtr : ℕ → S → S → ℝ) (ψob : ℕ → S → ℝ) (k j : ℕ) (b : S) : ℝ :=
  ∑ a : Fin j → S,
    (∏ i ∈ Finset.range j, ψtr (k + i) (ch06_pathB j a b i) (ch06_pathB j a b (i + 1)))
      * (∏ i ∈ Finset.range j, ψob (k + 1 + i) (ch06_pathB j a b (i + 1)))

/-- noname-4 (ch06 lines 226-234), general in `k` and `j`: peeling the first variable
on the right off eq:bp-bwd-partial produces one step of the backward recursion
eq:bp-bwd. -/
theorem ch06_bwdPartial_succ (ψtr : ℕ → S → S → ℝ) (ψob : ℕ → S → ℝ) (k j : ℕ) (b : S) :
    ch06_bwdPartial ψtr ψob k (j + 1) b
      = ∑ u : S, ψtr k b u * ψob (k + 1) u * ch06_bwdPartial ψtr ψob (k + 1) j u := by
  unfold ch06_bwdPartial
  rw [ch06_sum_split_first]
  refine Finset.sum_congr rfl fun u _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun c _ => ?_
  have hP : ∀ i : ℕ, i < j + 1 →
      ch06_pathB (j + 1) (fun i : Fin (j + 1) => ch06_pathB j c u (i : ℕ)) b (i + 1)
        = ch06_pathB j c u i := fun i hi => ch06_pathB_succ _ _ hi
  have h1 : (∏ i ∈ Finset.range (j + 1),
        ψtr (k + i) (ch06_pathB (j + 1) (fun i : Fin (j + 1) => ch06_pathB j c u (i : ℕ)) b i)
          (ch06_pathB (j + 1) (fun i : Fin (j + 1) => ch06_pathB j c u (i : ℕ)) b (i + 1)))
      = (∏ i ∈ Finset.range j,
            ψtr (k + 1 + i) (ch06_pathB j c u i) (ch06_pathB j c u (i + 1))) * ψtr k b u := by
    rw [Finset.prod_range_succ']
    congr 1
    refine Finset.prod_congr rfl fun i hi => ?_
    simp only [Finset.mem_range] at hi
    rw [hP i (by omega), hP (i + 1) (by omega), show k + (i + 1) = k + 1 + i by omega]
  have h2 : (∏ i ∈ Finset.range (j + 1),
        ψob (k + 1 + i)
          (ch06_pathB (j + 1) (fun i : Fin (j + 1) => ch06_pathB j c u (i : ℕ)) b (i + 1)))
      = (∏ i ∈ Finset.range j, ψob (k + 1 + 1 + i) (ch06_pathB j c u (i + 1)))
          * ψob (k + 1) u := by
    rw [Finset.prod_range_succ']
    congr 1
    refine Finset.prod_congr rfl fun i hi => ?_
    simp only [Finset.mem_range] at hi
    rw [hP (i + 1) (by omega), show k + 1 + (i + 1) = k + 1 + 1 + i by omega]
  rw [h1, h2]
  ring

/-- eq:bp-bwd-partial + noname-4 (ch06 lines 210-234), general in the chain length and
the site: the recursively defined backward message of eq:bp-bwd *is* the explicit
partial marginalisation of eq:bp-bwd-partial.  Here the chain has length `k + j + 1`
and the message sits at site `k`, i.e. `j` hops in from the right end. -/
theorem ch06_bp_bwd_partial (ψtr : ℕ → S → S → ℝ) (ψob : ℕ → S → ℝ) (k j : ℕ) (b : S) :
    ch06_bwdD ψtr ψob (k + j + 1) j b = ch06_bwdPartial ψtr ψob k j b := by
  induction j generalizing k b with
  | zero =>
      unfold ch06_bwdPartial
      rw [Finset.univ_unique, Finset.sum_singleton]
      simp only [Finset.range_zero, Finset.prod_empty, mul_one]
      rfl
  | succ j ih =>
      have h1 : k + (j + 1) + 1 - 2 - j = k := by omega
      have h2 : k + (j + 1) + 1 - 1 - j = k + 1 := by omega
      have h3 : k + (j + 1) + 1 = k + 1 + j + 1 := by omega
      rw [ch06_bwdPartial_succ]
      simp only [ch06_bwdD, h1, h2]
      refine Finset.sum_congr rfl fun u _ => ?_
      rw [show ch06_bwdD ψtr ψob (k + (j + 1) + 1) j u
            = ch06_bwdD ψtr ψob (k + 1 + j + 1) j u by rw [h3], ih (k + 1) u]

/-- The full path `(a₀,…,a_{k-1}, c, a_{k+1},…,a_{k+m})` assembled from the left block
`l`, the value `c` held fixed at site `k`, and the right block `r`. -/
def ch06_join (k m : ℕ) (l : Fin k → S) (c : S) (r : Fin m → S) : ℕ → S :=
  fun i => if i ≤ k then ch06_ext k l c i else ch06_pathB m r c (i - k)

theorem ch06_join_le {k m : ℕ} (l : Fin k → S) (c : S) (r : Fin m → S) {i : ℕ} (h : i ≤ k) :
    ch06_join k m l c r i = ch06_ext k l c i := if_pos h

theorem ch06_join_add {k m : ℕ} (l : Fin k → S) (c : S) (r : Fin m → S) (i : ℕ) :
    ch06_join k m l c r (k + i) = ch06_pathB m r c i := by
  cases i with
  | zero =>
      simp only [Nat.add_zero]
      rw [ch06_join_le _ _ _ (le_refl k), ch06_ext_of_ge _ _ (le_refl k)]
      rfl
  | succ i =>
      have h : ¬ (k + (i + 1) ≤ k) := by omega
      unfold ch06_join
      rw [if_neg h, show k + (i + 1) - k = i + 1 by omega]

/-- eq:bp-posterior (ch06 lines 36-46), discrete form: the weight of one path. -/
def ch06_postW (ψpr : S → ℝ) (ψtr : ℕ → S → S → ℝ) (ψob : ℕ → S → ℝ)
    (L : ℕ) (p : ℕ → S) : ℝ :=
  ψpr (p 0) * (∏ j ∈ Finset.range (L - 1), ψtr j (p j) (p (j + 1)))
    * (∏ j ∈ Finset.range L, ψob j (p j))

/-- eq:bp-split-at-k (ch06 lines 123-145) on an assembled path, general in `k` and in
the number `m` of sites to the right of `k`: the posterior weight is
(left block) × ψ_ob^{(k)}(a_k) × (right block), the two blocks being exactly the
integrands of eq:bp-fwd-partial and eq:bp-bwd-partial. -/
theorem ch06_postW_join (ψpr : S → ℝ) (ψtr : ℕ → S → S → ℝ) (ψob : ℕ → S → ℝ)
    (k m : ℕ) (l : Fin k → S) (c : S) (r : Fin m → S) :
    ch06_postW ψpr ψtr ψob (k + 1 + m) (ch06_join k m l c r)
      = (ψpr (ch06_ext k l c 0)
          * (∏ j ∈ Finset.range k, ψtr j (ch06_ext k l c j) (ch06_ext k l c (j + 1)))
          * (∏ j ∈ Finset.range k, ψob j (ch06_ext k l c j)))
        * ψob k c
        * ((∏ i ∈ Finset.range m,
              ψtr (k + i) (ch06_pathB m r c i) (ch06_pathB m r c (i + 1)))
            * (∏ i ∈ Finset.range m, ψob (k + 1 + i) (ch06_pathB m r c (i + 1)))) := by
  unfold ch06_postW
  rw [show k + 1 + m - 1 = k + m by omega]
  have htr : (∏ j ∈ Finset.range (k + m),
        ψtr j (ch06_join k m l c r j) (ch06_join k m l c r (j + 1)))
      = (∏ j ∈ Finset.range k, ψtr j (ch06_ext k l c j) (ch06_ext k l c (j + 1)))
        * (∏ i ∈ Finset.range m,
            ψtr (k + i) (ch06_pathB m r c i) (ch06_pathB m r c (i + 1))) := by
    rw [← Finset.prod_range_mul_prod_Ico _ (Nat.le_add_right k m)]
    congr 1
    · refine Finset.prod_congr rfl fun j hj => ?_
      simp only [Finset.mem_range] at hj
      rw [ch06_join_le _ _ _ (by omega), ch06_join_le _ _ _ (by omega)]
    · rw [Finset.prod_Ico_eq_prod_range]
      simp only [Nat.add_sub_cancel_left]
      refine Finset.prod_congr rfl fun i _ => ?_
      rw [ch06_join_add, show k + i + 1 = k + (i + 1) by omega, ch06_join_add]
  have hob : (∏ j ∈ Finset.range (k + 1 + m), ψob j (ch06_join k m l c r j))
      = (∏ j ∈ Finset.range k, ψob j (ch06_ext k l c j)) * ψob k c
        * (∏ i ∈ Finset.range m, ψob (k + 1 + i) (ch06_pathB m r c (i + 1))) := by
    rw [← Finset.prod_range_mul_prod_Ico _ (show k ≤ k + 1 + m by omega),
      Finset.prod_eq_prod_Ico_succ_bot (show k < k + 1 + m by omega),
      Finset.prod_Ico_eq_prod_range, show k + 1 + m - (k + 1) = m by omega]
    have e1 : (∏ j ∈ Finset.range k, ψob j (ch06_join k m l c r j))
        = ∏ j ∈ Finset.range k, ψob j (ch06_ext k l c j) := by
      refine Finset.prod_congr rfl fun j hj => ?_
      simp only [Finset.mem_range] at hj
      rw [ch06_join_le _ _ _ (by omega)]
    have e2 : ch06_join k m l c r k = c := by
      rw [ch06_join_le _ _ _ (le_refl k), ch06_ext_of_ge _ _ (le_refl k)]
    have e3 : (∏ i ∈ Finset.range m, ψob (k + 1 + i) (ch06_join k m l c r (k + 1 + i)))
        = ∏ i ∈ Finset.range m, ψob (k + 1 + i) (ch06_pathB m r c (i + 1)) := by
      refine Finset.prod_congr rfl fun i _ => ?_
      rw [show k + 1 + i = k + (i + 1) by omega, ch06_join_add]
    rw [e1, e2, e3]
    ring
  rw [htr, hob, ch06_join_le _ _ _ (Nat.zero_le k)]
  ring

/-- eq:bp-combine / eq:bp-split-integrals / eq:bp-combine-proof / noname-16
(ch06 lines 104-110, 150-176, 239-248, 491-497), **general in the chain length and in
the site**: summing the posterior over every variable except `a_k = c` gives
`μ_{→k}(c) · ψ_ob^{(k)}(c) · μ_{←k}(c)`, with both messages the recursively computed
ones of eq:bp-fwd / eq:bp-bwd.  Here the chain has `k` sites to the left of `k` and
`m` to its right, i.e. length `k + 1 + m`. -/
theorem ch06_bp_combine (ψpr : S → ℝ) (ψtr : ℕ → S → S → ℝ) (ψob : ℕ → S → ℝ)
    (k m : ℕ) (c : S) :
    (∑ l : Fin k → S, ∑ r : Fin m → S,
        ch06_postW ψpr ψtr ψob (k + 1 + m) (ch06_join k m l c r))
      = ch06_fwdD ψpr ψtr ψob k c * ψob k c * ch06_bwdD ψtr ψob (k + 1 + m) m c := by
  have h1 : (∑ l : Fin k → S, ∑ r : Fin m → S,
        ch06_postW ψpr ψtr ψob (k + 1 + m) (ch06_join k m l c r))
      = ch06_fwdPartial ψpr ψtr ψob k c * ψob k c * ch06_bwdPartial ψtr ψob k m c := by
    unfold ch06_fwdPartial ch06_bwdPartial
    simp_rw [ch06_postW_join]
    exact ch06_bp_sum_factorises _ _ _
  rw [h1, ← ch06_bp_fwd_partial]
  congr 1
  rw [show k + 1 + m = k + m + 1 by omega]
  exact (ch06_bp_bwd_partial ψtr ψob k m c).symm

/-- eq:bp-combine in the chapter's own indexing: for every chain length `L` and every
site `k < L`, marginalising the posterior down to `a_k` gives the product of the
forward message into `k`, the local observation factor, and the backward message
`L-1-k` hops in from the right end. -/
theorem ch06_bp_combine_general (ψpr : S → ℝ) (ψtr : ℕ → S → S → ℝ) (ψob : ℕ → S → ℝ)
    {L k : ℕ} (hk : k < L) (c : S) :
    (∑ l : Fin k → S, ∑ r : Fin (L - 1 - k) → S,
        ch06_postW ψpr ψtr ψob L (ch06_join k (L - 1 - k) l c r))
      = ch06_fwdD ψpr ψtr ψob k c * ψob k c * ch06_bwdD ψtr ψob L (L - 1 - k) c := by
  obtain ⟨m, rfl⟩ : ∃ m, L = k + 1 + m := ⟨L - 1 - k, by omega⟩
  rw [show k + 1 + m - 1 - k = m by omega]
  exact ch06_bp_combine ψpr ψtr ψob k m c

end DiscreteGeneral

/-- eq:bp-split-at-k (ch06 lines 123-145): the posterior factors split into
"left of `a_k`", the local observation, and "right of `a_k`", with every factor
used exactly once.  Proved for every `L` and every site `k`. -/
theorem ch06_bp_split_at_k {L k : ℕ} (hk1 : k ≤ L - 1) (hk2 : k < L)
    (f g : ℕ → ℝ) (P : ℝ) :
    P * (∏ j ∈ Finset.range (L - 1), f j) * (∏ j ∈ Finset.range L, g j)
      = (P * (∏ j ∈ Finset.range k, f j) * (∏ j ∈ Finset.range k, g j))
        * g k
        * ((∏ j ∈ Finset.Ico k (L - 1), f j) * (∏ j ∈ Finset.Ico (k + 1) L, g j)) := by
  rw [← Finset.prod_range_mul_prod_Ico f hk1,
      ← Finset.prod_range_mul_prod_Ico g (le_of_lt hk2),
      Finset.prod_eq_prod_Ico_succ_bot hk2 g]
  ring

/-! ### Computational cost (ch06 lines 266-295, 316-332)

Cost model: scalar arithmetic is unit cost — one multiplication, one addition or one
division counts as one operation — and a stored real number counts as one unit of
memory.  Nothing else is assumed: the operation *counts* below are read off the
sum-product recursion itself by `ch06_fwdCounted` / `ch06_bwdCounted`, instrumented
copies of `ch06_fwdD` / `ch06_bwdD` that carry a counter, and proved by induction to
compute the very same messages. -/

/-- Scalar operations in one entry of one discrete transition message: evaluating
`∑_{a_k} ψ_tr^{(k)}(a_k,a_{k+1}) ψ_ob^{(k)}(a_k) μ_{→k}(a_k)` at a single value of
`a_{k+1}` costs two multiplications for each of the `K` summands and `K-1` additions
to combine them. -/
def ch06_msgEntryOps (K : ℕ) : ℕ := 2 * K + (K - 1)

/-- Scalar operations in one whole transition message: one entry per value of
`a_{k+1}`, so `K` of them.  This is the chapter's "one transition message costs
`O(K²)`". -/
def ch06_msgOps (K : ℕ) : ℕ := K * ch06_msgEntryOps K

/-- Time of one forward-backward sweep on a length-`L` chain over a `K`-state
alphabet: `L-1` transition messages in each direction. -/
def ch06_bp_ops (L K : ℕ) : ℕ := 2 * (L - 1) * ch06_msgOps K

/-- Memory: one forward and one backward message, `K` numbers each, at each of the `L`
sites. -/
def ch06_bp_mem (L K : ℕ) : ℕ := 2 * L * K

/-- eq:bp-time (ch06 lines 280-285): `time = O(LK²)`, as an explicit bound with an
explicit constant, valid for every `L` and every `K`. -/
theorem ch06_bp_time : ∃ C : ℕ, 0 < C ∧ ∀ L K : ℕ, ch06_bp_ops L K ≤ C * (L * K ^ 2) := by
  refine ⟨6, by norm_num, fun L K => ?_⟩
  unfold ch06_bp_ops ch06_msgOps ch06_msgEntryOps
  have h1 : 2 * (L - 1) ≤ 2 * L := by omega
  have h2 : K * (2 * K + (K - 1)) ≤ 3 * K ^ 2 := by
    calc K * (2 * K + (K - 1)) ≤ K * (3 * K) := Nat.mul_le_mul le_rfl (by omega)
      _ = 3 * K ^ 2 := by ring
  calc 2 * (L - 1) * (K * (2 * K + (K - 1))) ≤ 2 * L * (3 * K ^ 2) := Nat.mul_le_mul h1 h2
    _ = 6 * (L * K ^ 2) := by ring

/-- eq:bp-time, tightness: the sweep really does cost `Ω(LK²)` in the same model, so
the bound above is not vacuous — `LK²` is the true order, not just an upper bound. -/
theorem ch06_bp_time_tight (L K : ℕ) : 2 * (L - 1) * K ^ 2 ≤ ch06_bp_ops L K := by
  unfold ch06_bp_ops ch06_msgOps ch06_msgEntryOps
  have h : K ^ 2 ≤ K * (2 * K + (K - 1)) := by
    calc K ^ 2 = K * K := by ring
      _ ≤ K * (2 * K + (K - 1)) := Nat.mul_le_mul le_rfl (by omega)
  exact Nat.mul_le_mul le_rfl h

/-- eq:bp-space (ch06 lines 288-293): `space = O(LK)`, explicit bound and constant. -/
theorem ch06_bp_space : ∃ C : ℕ, 0 < C ∧ ∀ L K : ℕ, ch06_bp_mem L K ≤ C * (L * K) := by
  refine ⟨2, by norm_num, fun L K => ?_⟩
  unfold ch06_bp_mem
  exact le_of_eq (by ring)

/-- Operations in one natural-parameter (Gaussian) message update.  It is left as a
parameter `c` on purpose: all that matters for the `O(L)` claim of
Theorem thm:bp-closure is that `ch06_fwdMap` / `ch06_bwdMap` are *fixed* closed-form
expressions, so `c` does not grow with `L` or with the site. -/
def ch06_gaussSweepOps (c L : ℕ) : ℕ := 2 * (L - 1) * c

/-- Memory for a Gaussian sweep: two scalars per message, `2L` messages.  That a
message really is two scalars is `ch06_bp_forward_closure_iterated` /
`ch06_bp_backward_closure_iterated`, not an assumption. -/
def ch06_gaussSweepMem (L : ℕ) : ℕ := 4 * L

/-- noname-8 (ch06 lines 324-330, Theorem "Gaussian closure of BP on the chain"):
whatever the constant per-update cost `c`, a Gaussian forward-backward sweep costs
`O(L)` time and `O(L)` space.  The premise that a message is a constant-size object —
two scalars, evolving under `ch06_fwdMap` / `ch06_bwdMap` — is proved in general in
`ch06_bp_forward_closure_iterated` / `ch06_bp_backward_closure_iterated`. -/
theorem ch06_bp_gaussian_linear (c : ℕ) :
    ∃ C : ℕ, 0 < C ∧ ∀ L : ℕ,
      ch06_gaussSweepOps c L ≤ C * L ∧ ch06_gaussSweepMem L ≤ C * L := by
  refine ⟨2 * c + 4, by omega, fun L => ⟨?_, ?_⟩⟩
  · unfold ch06_gaussSweepOps
    calc 2 * (L - 1) * c ≤ 2 * L * c := Nat.mul_le_mul (by omega) le_rfl
      _ = 2 * c * L := by ring
      _ ≤ (2 * c + 4) * L := Nat.mul_le_mul (by omega) le_rfl
  · unfold ch06_gaussSweepMem
    exact Nat.mul_le_mul (by omega) le_rfl

/-! #### Grounding the per-message operation count

`ch06_msgEntryOps` is *not* a stipulation about how many operations a message entry
takes: the entry is exhibited below as an explicit arithmetic expression — the `K`
summands `ψ_tr^{(k)}(a_k,a_{k+1}) · ψ_ob^{(k)}(a_k) · μ_{→k}(a_k)`, two multiplications
each, combined by `K-1` additions — which is *proved* to evaluate to the entry and to
have exactly `ch06_msgEntryOps K` arithmetic nodes (`ch06_msgEntry_ops`).  The only
modelling assumption remaining in eq:bp-time is that one real multiplication or
addition costs one unit. -/

/-- Syntax of a closed arithmetic expression over `ℝ`. -/
inductive ch06_Arith : Type where
  | lit : ℝ → ch06_Arith
  | add : ch06_Arith → ch06_Arith → ch06_Arith
  | mul : ch06_Arith → ch06_Arith → ch06_Arith

/-- Value of an arithmetic expression. -/
def ch06_eval : ch06_Arith → ℝ
  | .lit x => x
  | .add e f => ch06_eval e + ch06_eval f
  | .mul e f => ch06_eval e * ch06_eval f

/-- Cost of an arithmetic expression: one unit per `add` / `mul` node. -/
def ch06_cost : ch06_Arith → ℕ
  | .lit _ => 0
  | .add e f => ch06_cost e + ch06_cost f + 1
  | .mul e f => ch06_cost e + ch06_cost f + 1

/-- One summand of a transition message: the triple product `ψ_tr · ψ_ob · μ`, i.e.
two multiplications. -/
def ch06_termExpr (p : ℝ × ℝ × ℝ) : ch06_Arith :=
  .mul (.mul (.lit p.1) (.lit p.2.1)) (.lit p.2.2)

/-- Accumulate a list of summands onto an accumulator, one `add` node each. -/
def ch06_sumExpr : ch06_Arith → List ch06_Arith → ch06_Arith
  | e, [] => e
  | e, f :: fs => ch06_sumExpr (.add e f) fs

theorem ch06_sumExpr_eval (l : List ch06_Arith) (e : ch06_Arith) :
    ch06_eval (ch06_sumExpr e l) = ch06_eval e + (l.map ch06_eval).sum := by
  induction l generalizing e with
  | nil => simp [ch06_sumExpr]
  | cons f fs ih =>
      rw [show ch06_sumExpr e (f :: fs) = ch06_sumExpr (ch06_Arith.add e f) fs from rfl, ih]
      simp only [ch06_eval, List.map_cons, List.sum_cons]
      ring

theorem ch06_sumExpr_cost (l : List ch06_Arith) (e : ch06_Arith) :
    ch06_cost (ch06_sumExpr e l) = ch06_cost e + (l.map ch06_cost).sum + l.length := by
  induction l generalizing e with
  | nil => simp [ch06_sumExpr]
  | cons f fs ih =>
      rw [show ch06_sumExpr e (f :: fs) = ch06_sumExpr (ch06_Arith.add e f) fs from rfl, ih]
      simp only [ch06_cost, List.map_cons, List.sum_cons, List.length_cons]
      omega

/-- A list of summands added up: `K` terms need `K-1` additions. -/
def ch06_sumList : List ch06_Arith → ch06_Arith
  | [] => .lit 0
  | e :: es => ch06_sumExpr e es

theorem ch06_sumList_eval (l : List ch06_Arith) :
    ch06_eval (ch06_sumList l) = (l.map ch06_eval).sum := by
  cases l with
  | nil => simp [ch06_sumList, ch06_eval]
  | cons e es =>
      rw [show ch06_sumList (e :: es) = ch06_sumExpr e es from rfl, ch06_sumExpr_eval]
      simp only [List.map_cons, List.sum_cons]

theorem ch06_sumList_cost (l : List ch06_Arith) :
    ch06_cost (ch06_sumList l) = (l.map ch06_cost).sum + (l.length - 1) := by
  cases l with
  | nil => simp [ch06_sumList, ch06_cost]
  | cons e es =>
      rw [show ch06_sumList (e :: es) = ch06_sumExpr e es from rfl, ch06_sumExpr_cost]
      simp only [List.map_cons, List.sum_cons, List.length_cons]
      omega


section DiscreteCost

variable {S : Type*} [Fintype S]

/-- The discrete forward sweep of eq:bp-fwd, instrumented with a scalar-operation
counter. -/
def ch06_fwdCounted (ψpr : S → ℝ) (ψtr : ℕ → S → S → ℝ) (ψob : ℕ → S → ℝ) :
    ℕ → (S → ℝ) × ℕ
  | 0 => (ψpr, 0)
  | (k + 1) =>
      ((fun w => ∑ u : S, ψtr k u w * ψob k u * (ch06_fwdCounted ψpr ψtr ψob k).1 u),
        (ch06_fwdCounted ψpr ψtr ψob k).2 + ch06_msgOps (Fintype.card S))

/-- The instrumented sweep computes the same messages as `ch06_fwdD`: the operations
counted are the operations of the real recursion. -/
theorem ch06_fwdCounted_fst (ψpr : S → ℝ) (ψtr : ℕ → S → S → ℝ) (ψob : ℕ → S → ℝ)
    (k : ℕ) : (ch06_fwdCounted ψpr ψtr ψob k).1 = ch06_fwdD ψpr ψtr ψob k := by
  induction k with
  | zero => rfl
  | succ k ih => simp only [ch06_fwdCounted, ch06_fwdD, ih]

/-- `k` forward messages cost `k · (operations per message)`. -/
theorem ch06_fwdCounted_snd (ψpr : S → ℝ) (ψtr : ℕ → S → S → ℝ) (ψob : ℕ → S → ℝ)
    (k : ℕ) : (ch06_fwdCounted ψpr ψtr ψob k).2 = k * ch06_msgOps (Fintype.card S) := by
  induction k with
  | zero => simp [ch06_fwdCounted]
  | succ k ih => simp only [ch06_fwdCounted, ih]; ring

/-- The discrete backward sweep of eq:bp-bwd, instrumented with a counter. -/
def ch06_bwdCounted (ψtr : ℕ → S → S → ℝ) (ψob : ℕ → S → ℝ) (L : ℕ) :
    ℕ → (S → ℝ) × ℕ
  | 0 => ((fun _ => 1), 0)
  | (j + 1) =>
      ((fun b => ∑ u : S, ψtr (L - 2 - j) b u * ψob (L - 1 - j) u
            * (ch06_bwdCounted ψtr ψob L j).1 u),
        (ch06_bwdCounted ψtr ψob L j).2 + ch06_msgOps (Fintype.card S))

theorem ch06_bwdCounted_fst (ψtr : ℕ → S → S → ℝ) (ψob : ℕ → S → ℝ) (L j : ℕ) :
    (ch06_bwdCounted ψtr ψob L j).1 = ch06_bwdD ψtr ψob L j := by
  induction j with
  | zero => rfl
  | succ j ih => simp only [ch06_bwdCounted, ch06_bwdD, ih]

theorem ch06_bwdCounted_snd (ψtr : ℕ → S → S → ℝ) (ψob : ℕ → S → ℝ) (L j : ℕ) :
    (ch06_bwdCounted ψtr ψob L j).2 = j * ch06_msgOps (Fintype.card S) := by
  induction j with
  | zero => simp [ch06_bwdCounted]
  | succ j ih => simp only [ch06_bwdCounted, ih]; ring

/-- eq:bp-time, the counting step: the operation count `ch06_bp_ops` bounded above is
exactly what a full forward-backward sweep performs — `L-1` messages each way. -/
theorem ch06_bp_sweep_ops (ψpr : S → ℝ) (ψtr : ℕ → S → S → ℝ) (ψob : ℕ → S → ℝ)
    (L : ℕ) :
    (ch06_fwdCounted ψpr ψtr ψob (L - 1)).2 + (ch06_bwdCounted ψtr ψob L (L - 1)).2
      = ch06_bp_ops L (Fintype.card S) := by
  rw [ch06_fwdCounted_snd, ch06_bwdCounted_snd]
  unfold ch06_bp_ops
  ring

/-- eq:bp-space, the counting step: `ch06_bp_mem L K = 2LK` is the number of stored
real numbers, `Fin (2L) × S` indexing them. -/
theorem ch06_bp_space_count (L : ℕ) :
    Fintype.card (Fin (2 * L) × S) = ch06_bp_mem L (Fintype.card S) := by
  unfold ch06_bp_mem
  simp [Fintype.card_prod]

/-- eq:bp-space, the substantive half: `2L` stored messages — a forward and a backward
one per site, `K` numbers each, `2LK` numbers in all — suffice to form *every* one of
the `L` smoothing marginals of eq:bp-combine.  (Which product is the marginal is
`ch06_bp_combine_general`.) -/
theorem ch06_bp_space_sufficient (ψpr : S → ℝ) (ψtr : ℕ → S → S → ℝ) (ψob : ℕ → S → ℝ)
    (L : ℕ) :
    ∃ store : Fin (2 * L) → S → ℝ,
      ∀ (k : ℕ) (hk : k < L) (c : S),
        ch06_fwdD ψpr ψtr ψob k c * ψob k c * ch06_bwdD ψtr ψob L (L - 1 - k) c
          = store ⟨k, by omega⟩ c * ψob k c * store ⟨L + k, by omega⟩ c := by
  refine ⟨fun i => if (i : ℕ) < L then ch06_fwdD ψpr ψtr ψob (i : ℕ)
            else ch06_bwdD ψtr ψob L (L - 1 - ((i : ℕ) - L)), ?_⟩
  intro k hk c
  have e1 : ((⟨k, by omega⟩ : Fin (2 * L)) : ℕ) = k := rfl
  have e2 : ((⟨L + k, by omega⟩ : Fin (2 * L)) : ℕ) = L + k := rfl
  simp only [e1, e2, if_pos hk, if_neg (show ¬ L + k < L by omega), Nat.add_sub_cancel_left]

/-- eq:bp-time, the per-entry counting step, general in the alphabet: one entry of a
discrete transition message *is* an arithmetic expression with exactly
`ch06_msgEntryOps K = 2K + (K-1)` operation nodes — `2K` multiplications and `K-1`
additions — whose value is the entry `∑_{a_k} ψ_tr ψ_ob μ`.  The chapter's `O(K²)`
per message is then `K` such entries (`ch06_msgOps`). -/
theorem ch06_msgEntry_ops (g h m : S → ℝ) :
    ∃ e : ch06_Arith,
      ch06_eval e = ∑ u : S, g u * h u * m u ∧
      ch06_cost e = ch06_msgEntryOps (Fintype.card S) := by
  have hev : (ch06_eval ∘ fun u : S => ch06_termExpr (g u, h u, m u))
      = fun u : S => g u * h u * m u := funext fun _ => rfl
  have hco : (ch06_cost ∘ fun u : S => ch06_termExpr (g u, h u, m u))
      = fun _ : S => 2 := funext fun _ => rfl
  refine ⟨ch06_sumList ((Finset.univ : Finset S).toList.map
      (fun u : S => ch06_termExpr (g u, h u, m u))), ?_, ?_⟩
  · rw [ch06_sumList_eval, List.map_map, hev, Finset.sum_map_toList]
  · rw [ch06_sumList_cost, List.map_map, hco, Finset.sum_map_toList, Finset.sum_const,
      Finset.card_univ, smul_eq_mul, List.length_map, Finset.length_toList, Finset.card_univ]
    unfold ch06_msgEntryOps
    omega


end DiscreteCost

/-! ### Section 6.3 — the Gaussian closure -/

-- noname-7 (ch06 lines 307-311): reading the observation factor as a function of the
-- clean variable turns it into a Gaussian at the deconvolved location `e^t x_k`
-- with variance `Δ_t e^{2t}` (precision `e^{-2t}/Δ_t`).
/-- Exponent form of noname-7, with `e·E = 1` abstracting `e^{-t}·e^{t} = 1`. -/
theorem ch06_ob_exponent {D x a e E : ℝ} (hD : D ≠ 0) (h : e * E = 1) :
    -(x - e * a) ^ 2 / (2 * D) = -(e * e / (2 * D)) * (a - E * x) ^ 2 := by
  have key : (e * e) * (a - E * x) ^ 2 = (x - e * a) ^ 2 := by
    calc (e * e) * (a - E * x) ^ 2 = (e * a - (e * E) * x) ^ 2 := by ring
      _ = (e * a - 1 * x) ^ 2 := by rw [h]
      _ = (x - e * a) ^ 2 := by ring
  rw [← key]
  field_simp

/-- noname-7 (ch06 lines 307-311), instantiated at `e = e^{-t}`, `E = e^{t}`:
`ψ_ob^{(k)}(a_k) ∝ exp(-(e^{-2t}/(2Δ_t))(a_k - e^{t}x_k)²)`, i.e. the Gaussian
`N(a_k; e^{t}x_k, Δ_t e^{2t})` up to normalisation. -/
theorem ch06_ob_as_gauss_in_a {t x a : ℝ} (hD : ch06_Delta t ≠ 0) :
    Real.exp (-(x - Real.exp (-t) * a) ^ 2 / (2 * ch06_Delta t))
      = Real.exp (-(Real.exp (-2 * t) / (2 * ch06_Delta t)) * (a - Real.exp t * x) ^ 2) := by
  have he : Real.exp (-t) * Real.exp t = 1 := by
    rw [← Real.exp_add]; norm_num
  have h2 : Real.exp (-2 * t) = Real.exp (-t) * Real.exp (-t) := by
    rw [← Real.exp_add]; ring_nf
  rw [h2, ch06_ob_exponent hD he]

/-- noname-7, variance form: `Δ_t e^{2t}` is indeed the variance matching the
precision `e^{-2t}/Δ_t`. -/
theorem ch06_ob_variance {t : ℝ} (hD : ch06_Delta t ≠ 0) :
    (Real.exp (-2 * t) / ch06_Delta t)⁻¹ = ch06_Delta t * Real.exp t ^ 2 := by
  have hsq : Real.exp t ^ 2 = Real.exp (2 * t) := by
    rw [pow_two, ← Real.exp_add]; congr 1; ring
  have hneg : Real.exp (-2 * t) = (Real.exp (2 * t))⁻¹ := by
    rw [← Real.exp_neg]; congr 1; ring
  have h2 : Real.exp (2 * t) ≠ 0 := ne_of_gt (Real.exp_pos _)
  rw [hsq, hneg]
  field_simp

-- eq:bp-observation-natural (ch06 lines 355-371) and
-- eq:bp-observation-parameters (ch06 lines 373-382).
/-- Natural parameters of the observation factor: precision `e^{-2t}/Δ_t`. -/
def ch06_lamOb (t : ℝ) : ℝ := Real.exp (-2 * t) / ch06_Delta t

/-- Natural parameters of the observation factor: field `e^{-t}x_k/Δ_t`. -/
def ch06_rOb (t xk : ℝ) : ℝ := Real.exp (-t) * xk / ch06_Delta t

/-- eq:bp-observation-natural, abstract form: `exp(-(x-ea)²/(2D))` is, up to a factor
not depending on `a`, the natural-form message with `λ = e²/D`, `r = ex/D`. -/
theorem ch06_ob_natural_abstract {D : ℝ} (hD : D ≠ 0) (x a e : ℝ) :
    Real.exp (-(x - e * a) ^ 2 / (2 * D))
      = Real.exp (-(x ^ 2) / (2 * D)) * ch06_nat (e * e / D) (e * x / D) a := by
  simp only [ch06_nat, ← Real.exp_add]
  congr 1
  field_simp
  ring

/-- eq:bp-observation-natural + eq:bp-observation-parameters (ch06 lines 355-382):
the observation factor at site `k`, as a function of `a_k`, is the natural-form
Gaussian with `λ_ob = e^{-2t}/Δ_t` and `r_{ob,k} = e^{-t}x_k/Δ_t`. -/
theorem ch06_bp_observation_parameters {t : ℝ} (hD : ch06_Delta t ≠ 0) (xk ak : ℝ) :
    Real.exp (-(xk - Real.exp (-t) * ak) ^ 2 / (2 * ch06_Delta t))
      = Real.exp (-(xk ^ 2) / (2 * ch06_Delta t)) * ch06_nat (ch06_lamOb t) (ch06_rOb t xk) ak := by
  have h2 : Real.exp (-t) * Real.exp (-t) = Real.exp (-2 * t) := by
    rw [← Real.exp_add]; ring_nf
  have := ch06_ob_natural_abstract hD xk ak (Real.exp (-t))
  rw [this, ch06_lamOb, ch06_rOb, h2]

/-- noname-10 (ch06 lines 388-397) and eq:bp-product-update (ch06 lines 399-408):
multiplying an incoming message by the local observation factor adds the natural
parameters, `λ̄ = λ + λ_ob`, `r̄ = r + r_{ob,k}`. -/
theorem ch06_bp_product_update (lam r : ℝ) (t xk ak : ℝ) :
    ch06_nat lam r ak * ch06_nat (ch06_lamOb t) (ch06_rOb t xk) ak
      = ch06_nat (lam + ch06_lamOb t) (r + ch06_rOb t xk) ak :=
  ch06_nat_mul _ _ _ _ _

/-- noname-9 (ch06 lines 346-350) and noname-12 (ch06 lines 424-431):
for `λ > 0` the natural-form message is, up to a constant, the Gaussian kernel with
variance `v = 1/λ` and mean `m = r/λ`. -/
theorem ch06_nat_as_gauss {lam : ℝ} (hlam : 0 < lam) (r a : ℝ) :
    ch06_nat lam r a
      = Real.exp (r ^ 2 / (2 * lam)) * Real.exp (-(a - r / lam) ^ 2 / (2 * (1 / lam))) := by
  have hlam' : lam ≠ 0 := ne_of_gt hlam
  simp only [ch06_nat, ← Real.exp_add]
  congr 1
  field_simp
  ring

/-- noname-11 (ch06 lines 415-421): the AR(1) transition `a_{k+1} = α a_k + η_k`. -/
def ch06_ar1_step (α a η : ℝ) : ℝ := α * a + η

/-! #### The forward closure -/

/-- eq:bp-forward-precision (ch06 lines 443-448). -/
def ch06_lamFwd (α s lamBar : ℝ) : ℝ := lamBar / (α ^ 2 + s * lamBar)

/-- eq:bp-forward-field (ch06 lines 449-453). -/
def ch06_rFwd (α s lamBar rBar : ℝ) : ℝ := α * rBar / (α ^ 2 + s * lamBar)

/-- noname-13 + eq:bp-forward-precision + eq:bp-forward-field
(ch06 lines 434-454).  The forward message

  `μ_{→k+1}(w) = ∫ N(w; α u, σ_η²) · exp(-½ λ̄ u² + r̄ u) du`

is again a natural-form Gaussian, with `λ_{→k+1} = λ̄/(α² + σ_η² λ̄)` and
`r_{→k+1} = α r̄/(α² + σ_η² λ̄)`.  Proved as a genuine Lebesgue integral. -/
theorem ch06_bp_forward_closure {s lam : ℝ} (hs : 0 < s) (hlam : 0 < lam) (α r w : ℝ) :
    (∫ u : ℝ, Real.exp (-(w - α * u) ^ 2 / (2 * s)) * ch06_nat lam r u)
      = Real.sqrt (Real.pi / ((α ^ 2 / s + lam) / 2))
        * (Real.exp (s * r ^ 2 / (2 * (α ^ 2 + s * lam)))
            * ch06_nat (ch06_lamFwd α s lam) (ch06_rFwd α s lam r) w) := by
  have hs' : s ≠ 0 := ne_of_gt hs
  have hden : (0:ℝ) < α ^ 2 + s * lam := by positivity
  have hden' : α ^ 2 + s * lam ≠ 0 := ne_of_gt hden
  have hA : (0:ℝ) < α ^ 2 / s + lam := by positivity
  have hA' : α ^ 2 / s + lam ≠ 0 := ne_of_gt hA
  have hstep : ∀ u : ℝ, Real.exp (-(w - α * u) ^ 2 / (2 * s)) * ch06_nat lam r u
      = Real.exp (-(w ^ 2) / (2 * s))
        * Real.exp (-((α ^ 2 / s + lam) / 2) * u ^ 2 + (α * w / s + r) * u) := by
    intro u
    simp only [ch06_nat, ← Real.exp_add]
    congr 1
    field_simp
    ring
  simp_rw [hstep]
  rw [MeasureTheory.integral_const_mul, ch06_gauss_integral hA (α * w / s + r)]
  simp only [ch06_nat, ch06_lamFwd, ch06_rFwd]
  rw [show ∀ p q c : ℝ, p * (q * c) = c * (p * q) from fun p q c => by ring]
  congr 1
  rw [← Real.exp_add, ← Real.exp_add]
  congr 1
  field_simp
  ring

/-- noname-13 (ch06 lines 434-441), moment form: the forward message has variance
`σ_η² + α²/λ̄` and mean `α r̄/λ̄`, i.e. exactly the law of `α a_k + η_k`. -/
theorem ch06_bp_transition_moments {s lam : ℝ} (hs : 0 < s) (hlam : 0 < lam) (α r : ℝ) :
    1 / ch06_lamFwd α s lam = s + α ^ 2 / lam
      ∧ ch06_rFwd α s lam r / ch06_lamFwd α s lam = α * (r / lam) := by
  have hlam' : lam ≠ 0 := ne_of_gt hlam
  have hden : (0:ℝ) < α ^ 2 + s * lam := by positivity
  have hden' : α ^ 2 + s * lam ≠ 0 := ne_of_gt hden
  refine ⟨?_, ?_⟩
  · simp only [ch06_lamFwd]
    field_simp
    ring
  · simp only [ch06_lamFwd, ch06_rFwd]
    field_simp
    all_goals ring

/-- noname-14 (ch06 lines 456-460): the forward map on natural parameters,
`(λ_{→k}, r_{→k}) ↦ (λ_{→k+1}, r_{→k+1})`, absorbing the local observation on the
way (eq:bp-product-update then eq:bp-forward-precision/field). -/
def ch06_fwdMap (α s t xk : ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  (ch06_lamFwd α s (p.1 + ch06_lamOb t), ch06_rFwd α s (p.1 + ch06_lamOb t) (p.2 + ch06_rOb t xk))

/-! #### The backward closure -/

/-- eq:bp-backward-precision (ch06 lines 475-479). -/
def ch06_lamBwd (α s lamBar : ℝ) : ℝ := α ^ 2 * lamBar / (1 + s * lamBar)

/-- eq:bp-backward-field (ch06 lines 480-485). -/
def ch06_rBwd (α s lamBar rBar : ℝ) : ℝ := α * rBar / (1 + s * lamBar)

/-- noname-15 + eq:bp-backward-precision + eq:bp-backward-field
(ch06 lines 466-486).  The backward message

  `μ_{←k-1}(b) = ∫ N(u; α b, σ_η²) · exp(-½ λ̄ u² + r̄ u) du`

is again a natural-form Gaussian, with `λ_{←k-1} = α² λ̄/(1 + σ_η² λ̄)` and
`r_{←k-1} = α r̄/(1 + σ_η² λ̄)`.  Proved as a genuine Lebesgue integral. -/
theorem ch06_bp_backward_closure {s lam : ℝ} (hs : 0 < s) (hlam : 0 < lam) (α r b : ℝ) :
    (∫ u : ℝ, Real.exp (-(u - α * b) ^ 2 / (2 * s)) * ch06_nat lam r u)
      = Real.sqrt (Real.pi / ((1 / s + lam) / 2))
        * (Real.exp (s * r ^ 2 / (2 * (1 + s * lam)))
            * ch06_nat (ch06_lamBwd α s lam) (ch06_rBwd α s lam r) b) := by
  have hs' : s ≠ 0 := ne_of_gt hs
  have hden : (0:ℝ) < 1 + s * lam := by positivity
  have hden' : (1:ℝ) + s * lam ≠ 0 := ne_of_gt hden
  have hA : (0:ℝ) < 1 / s + lam := by positivity
  have hA' : (1:ℝ) / s + lam ≠ 0 := ne_of_gt hA
  have hstep : ∀ u : ℝ, Real.exp (-(u - α * b) ^ 2 / (2 * s)) * ch06_nat lam r u
      = Real.exp (-(α ^ 2 * b ^ 2) / (2 * s))
        * Real.exp (-((1 / s + lam) / 2) * u ^ 2 + (α * b / s + r) * u) := by
    intro u
    simp only [ch06_nat, ← Real.exp_add]
    congr 1
    field_simp
    ring
  simp_rw [hstep]
  rw [MeasureTheory.integral_const_mul, ch06_gauss_integral hA (α * b / s + r)]
  simp only [ch06_nat, ch06_lamBwd, ch06_rBwd]
  rw [show ∀ p q c : ℝ, p * (q * c) = c * (p * q) from fun p q c => by ring]
  congr 1
  rw [← Real.exp_add, ← Real.exp_add]
  congr 1
  field_simp
  ring

/-! #### Combining the two sweeps -/

/-- eq:bp-belief-precision (ch06 lines 499-505) and eq:bp-belief-field
(ch06 lines 506-512): the three Gaussian factors of the belief multiply, so their
natural parameters add. -/
theorem ch06_bp_belief_parameters (lamF rF lamB rB : ℝ) (t xk ak : ℝ) :
    ch06_nat lamF rF ak * ch06_nat (ch06_lamOb t) (ch06_rOb t xk) ak * ch06_nat lamB rB ak
      = ch06_nat (lamF + ch06_lamOb t + lamB) (rF + ch06_rOb t xk + rB) ak := by
  rw [ch06_nat_mul, ch06_nat_mul]

/-- eq:bp-belief-mean (ch06 lines 514-522): the belief with natural parameters
`(λ^post, r^post)`, `λ^post > 0`, is the Gaussian kernel of mean `r^post/λ^post`,
so `E[a_k | x] = r^post/λ^post`. -/
theorem ch06_bp_belief_mean {lam : ℝ} (hlam : 0 < lam) (r a : ℝ) :
    ch06_nat lam r a
      = Real.exp (r ^ 2 / (2 * lam)) * Real.exp (-(a - r / lam) ^ 2 / (2 * (1 / lam))) :=
  ch06_nat_as_gauss hlam r a

/-- noname-17 (ch06 lines 524-532): substituting eq:bp-belief-mean into the
score-posterior identity eq:tweedie-joint,
`S_k(x,t) = (e^{-t} r^post/λ^post - x_k)/Δ_t`. -/
theorem ch06_bp_score (t xk lam r m : ℝ) (hm : m = r / lam) :
    (Real.exp (-t) * m - xk) / ch06_Delta t
      = (Real.exp (-t) * r / lam - xk) / ch06_Delta t := by
  rw [hm]; ring

/-! #### The whole sweep stays Gaussian

Theorem thm:bp-closure (ch06 lines 316-332) claims that *every* sum-product message
and every one-node belief is Gaussian, not merely that one update preserves
Gaussianity.  That is an induction along the chain, carried out here: the message
after `k` steps is a positive constant times the natural-form Gaussian whose two
parameters are the `k`-th iterate of the fixed map of noname-14.  This is what makes
the `O(1)`-per-update, `O(L)`-per-sweep accounting of `ch06_bp_gaussian_linear`
legitimate: the state carried along the chain never grows beyond two scalars. -/

/-- The two natural parameters `(λ_{→k}, r_{→k})` of the forward message, obtained by
iterating the update map `ch06_fwdMap` of noname-14 from the prior `N(0,1)`. -/
def ch06_fwdParams (α σ2 t : ℝ) (x : ℕ → ℝ) : ℕ → ℝ × ℝ
  | 0 => (1, 0)
  | (k + 1) => ch06_fwdMap α σ2 t (x k) (ch06_fwdParams α σ2 t x k)

/-- The natural-parameter map of the backward sweep: absorb the local observation
(eq:bp-product-update), then pass through the transition
(eq:bp-backward-precision/field). -/
def ch06_bwdMap (α s t xk : ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  (ch06_lamBwd α s (p.1 + ch06_lamOb t), ch06_rBwd α s (p.1 + ch06_lamOb t) (p.2 + ch06_rOb t xk))

/-- The two natural parameters of the backward message `j` hops in from the right end
of a length-`L` chain, from the flat message `μ_{←L-1} = 1` (`λ = r = 0`). -/
def ch06_bwdParams (α σ2 t : ℝ) (x : ℕ → ℝ) (L : ℕ) : ℕ → ℝ × ℝ
  | 0 => (0, 0)
  | (j + 1) => ch06_bwdMap α σ2 t (x (L - 1 - j)) (ch06_bwdParams α σ2 t x L j)

/-- One forward step for an arbitrary incoming natural-form Gaussian message: the
transition integral of eq:bp-fwd sends `C·exp(-½λu²+ru)` to a positive constant times
the natural-form Gaussian with the parameters of
eq:bp-forward-precision/eq:bp-forward-field, applied to `λ̄ = λ + λ_ob`,
`r̄ = r + r_{ob,k}` as eq:bp-product-update prescribes. -/
theorem ch06_fwd_step_gaussian {α σ2 t : ℝ} (hσ : 0 < σ2) (hD : 0 < ch06_Delta t)
    (xk : ℝ) {lam r C : ℝ} (hlam : 0 ≤ lam) (hC : 0 < C) (μ : ℝ → ℝ)
    (hμ : ∀ u, μ u = C * ch06_nat lam r u) :
    ∃ C' : ℝ, 0 < C' ∧ ∀ w : ℝ,
      (∫ u : ℝ, ch06_psiTr α σ2 u w * ch06_psiOb t xk u * μ u)
        = C' * ch06_nat (ch06_lamFwd α σ2 (lam + ch06_lamOb t))
            (ch06_rFwd α σ2 (lam + ch06_lamOb t) (r + ch06_rOb t xk)) w := by
  have hDne : ch06_Delta t ≠ 0 := ne_of_gt hD
  have hlamOb : 0 < ch06_lamOb t := div_pos (Real.exp_pos _) hD
  have hbar : 0 < lam + ch06_lamOb t := by linarith
  have h2pi : (0:ℝ) < 2 * Real.pi := by positivity
  have hs1 : (0:ℝ) < Real.sqrt (2 * Real.pi * σ2) := Real.sqrt_pos.mpr (mul_pos h2pi hσ)
  have hs2 : (0:ℝ) < Real.sqrt (2 * Real.pi * ch06_Delta t) := Real.sqrt_pos.mpr (mul_pos h2pi hD)
  have hs1' : Real.sqrt (2 * Real.pi * σ2) ≠ 0 := ne_of_gt hs1
  have hs2' : Real.sqrt (2 * Real.pi * ch06_Delta t) ≠ 0 := ne_of_gt hs2
  obtain ⟨Kc, hKc, hpt⟩ : ∃ Kc : ℝ, 0 < Kc ∧ ∀ w u : ℝ,
      ch06_psiTr α σ2 u w * ch06_psiOb t xk u * μ u
        = Kc * (Real.exp (-(w - α * u) ^ 2 / (2 * σ2))
            * ch06_nat (lam + ch06_lamOb t) (r + ch06_rOb t xk) u) := by
    refine ⟨C * Real.exp (-(xk ^ 2) / (2 * ch06_Delta t)) /
        (Real.sqrt (2 * Real.pi * σ2) * Real.sqrt (2 * Real.pi * ch06_Delta t)),
      div_pos (mul_pos hC (Real.exp_pos _)) (mul_pos hs1 hs2), fun w u => ?_⟩
    rw [hμ u]
    unfold ch06_psiTr ch06_psiOb ch06_gauss
    rw [ch06_bp_observation_parameters hDne xk u, ← ch06_nat_mul]
    field_simp
  have hA : (0:ℝ) < α ^ 2 / σ2 + (lam + ch06_lamOb t) := by
    have h0 : (0:ℝ) ≤ α ^ 2 / σ2 := div_nonneg (sq_nonneg α) (le_of_lt hσ)
    linarith
  refine ⟨Kc * (Real.sqrt (Real.pi / ((α ^ 2 / σ2 + (lam + ch06_lamOb t)) / 2))
      * Real.exp (σ2 * (r + ch06_rOb t xk) ^ 2 / (2 * (α ^ 2 + σ2 * (lam + ch06_lamOb t))))),
    mul_pos hKc (mul_pos (Real.sqrt_pos.mpr (div_pos Real.pi_pos (by linarith)))
      (Real.exp_pos _)), fun w => ?_⟩
  simp_rw [hpt w]
  rw [MeasureTheory.integral_const_mul, ch06_bp_forward_closure hσ hbar α (r + ch06_rOb t xk) w]
  ring

/-- One backward step for an arbitrary incoming natural-form Gaussian message.  Note
that `λ ≥ 0` suffices: the backward sweep starts from the flat message `μ_{←L-1} = 1`,
which is the `λ = 0` member of the family. -/
theorem ch06_bwd_step_gaussian {α σ2 t : ℝ} (hσ : 0 < σ2) (hD : 0 < ch06_Delta t)
    (xk : ℝ) {lam r C : ℝ} (hlam : 0 ≤ lam) (hC : 0 < C) (μ : ℝ → ℝ)
    (hμ : ∀ u, μ u = C * ch06_nat lam r u) :
    ∃ C' : ℝ, 0 < C' ∧ ∀ b : ℝ,
      (∫ u : ℝ, ch06_psiTr α σ2 b u * ch06_psiOb t xk u * μ u)
        = C' * ch06_nat (ch06_lamBwd α σ2 (lam + ch06_lamOb t))
            (ch06_rBwd α σ2 (lam + ch06_lamOb t) (r + ch06_rOb t xk)) b := by
  have hDne : ch06_Delta t ≠ 0 := ne_of_gt hD
  have hlamOb : 0 < ch06_lamOb t := div_pos (Real.exp_pos _) hD
  have hbar : 0 < lam + ch06_lamOb t := by linarith
  have h2pi : (0:ℝ) < 2 * Real.pi := by positivity
  have hs1 : (0:ℝ) < Real.sqrt (2 * Real.pi * σ2) := Real.sqrt_pos.mpr (mul_pos h2pi hσ)
  have hs2 : (0:ℝ) < Real.sqrt (2 * Real.pi * ch06_Delta t) := Real.sqrt_pos.mpr (mul_pos h2pi hD)
  have hs1' : Real.sqrt (2 * Real.pi * σ2) ≠ 0 := ne_of_gt hs1
  have hs2' : Real.sqrt (2 * Real.pi * ch06_Delta t) ≠ 0 := ne_of_gt hs2
  obtain ⟨Kc, hKc, hpt⟩ : ∃ Kc : ℝ, 0 < Kc ∧ ∀ b u : ℝ,
      ch06_psiTr α σ2 b u * ch06_psiOb t xk u * μ u
        = Kc * (Real.exp (-(u - α * b) ^ 2 / (2 * σ2))
            * ch06_nat (lam + ch06_lamOb t) (r + ch06_rOb t xk) u) := by
    refine ⟨C * Real.exp (-(xk ^ 2) / (2 * ch06_Delta t)) /
        (Real.sqrt (2 * Real.pi * σ2) * Real.sqrt (2 * Real.pi * ch06_Delta t)),
      div_pos (mul_pos hC (Real.exp_pos _)) (mul_pos hs1 hs2), fun b u => ?_⟩
    rw [hμ u]
    unfold ch06_psiTr ch06_psiOb ch06_gauss
    rw [ch06_bp_observation_parameters hDne xk u, ← ch06_nat_mul]
    field_simp
  have hA : (0:ℝ) < 1 / σ2 + (lam + ch06_lamOb t) := by
    have h0 : (0:ℝ) < 1 / σ2 := by positivity
    linarith
  refine ⟨Kc * (Real.sqrt (Real.pi / ((1 / σ2 + (lam + ch06_lamOb t)) / 2))
      * Real.exp (σ2 * (r + ch06_rOb t xk) ^ 2 / (2 * (1 + σ2 * (lam + ch06_lamOb t))))),
    mul_pos hKc (mul_pos (Real.sqrt_pos.mpr (div_pos Real.pi_pos (by linarith)))
      (Real.exp_pos _)), fun b => ?_⟩
  simp_rw [hpt b]
  rw [MeasureTheory.integral_const_mul, ch06_bp_backward_closure hσ hbar α (r + ch06_rOb t xk) b]
  ring

/-- Theorem thm:bp-closure, forward half, for **every** site: the forward message
`μ_{→k}` of eq:bp-fwd is a positive constant times the natural-form Gaussian with
parameters the `k`-th iterate of `ch06_fwdMap`, and its precision is strictly
positive.  Proved by induction along the chain. -/
theorem ch06_bp_forward_closure_iterated {α σ2 t : ℝ} (hσ : 0 < σ2) (hD : 0 < ch06_Delta t)
    (x : ℕ → ℝ) (k : ℕ) :
    0 < (ch06_fwdParams α σ2 t x k).1 ∧
      ∃ C : ℝ, 0 < C ∧ ∀ w : ℝ,
        ch06_msgFwd ch06_psiPr (fun _ => ch06_psiTr α σ2) (fun j => ch06_psiOb t (x j)) k w
          = C * ch06_nat (ch06_fwdParams α σ2 t x k).1 (ch06_fwdParams α σ2 t x k).2 w := by
  induction k with
  | zero =>
      refine ⟨?_, (Real.sqrt (2 * Real.pi))⁻¹,
        inv_pos.mpr (Real.sqrt_pos.mpr (by positivity)), fun w => ?_⟩
      · simp only [ch06_fwdParams]
        norm_num
      · show ch06_psiPr w = _
        unfold ch06_psiPr ch06_gauss
        simp only [ch06_fwdParams, ch06_nat]
        rw [show -(w - 0) ^ 2 / (2 * 1) = -(1 / 2) * (1:ℝ) * w ^ 2 + 0 * w from by ring,
          show (2:ℝ) * Real.pi * 1 = 2 * Real.pi from by ring, div_eq_inv_mul]
  | succ k ih =>
      obtain ⟨hlam, C, hC, hrec⟩ := ih
      have hlamOb : 0 < ch06_lamOb t := div_pos (Real.exp_pos _) hD
      have hbar : 0 < (ch06_fwdParams α σ2 t x k).1 + ch06_lamOb t := by linarith
      have hden : (0:ℝ) < α ^ 2 + σ2 * ((ch06_fwdParams α σ2 t x k).1 + ch06_lamOb t) := by
        have h1 : (0:ℝ) < σ2 * ((ch06_fwdParams α σ2 t x k).1 + ch06_lamOb t) := mul_pos hσ hbar
        have h2 : (0:ℝ) ≤ α ^ 2 := sq_nonneg α
        linarith
      obtain ⟨C', hC', hstep⟩ :=
        ch06_fwd_step_gaussian (α := α) hσ hD (x k) (le_of_lt hlam) hC
          (ch06_msgFwd ch06_psiPr (fun _ => ch06_psiTr α σ2) (fun j => ch06_psiOb t (x j)) k) hrec
      have hfst : (ch06_fwdParams α σ2 t x (k + 1)).1
          = ch06_lamFwd α σ2 ((ch06_fwdParams α σ2 t x k).1 + ch06_lamOb t) := rfl
      have hsnd : (ch06_fwdParams α σ2 t x (k + 1)).2
          = ch06_rFwd α σ2 ((ch06_fwdParams α σ2 t x k).1 + ch06_lamOb t)
              ((ch06_fwdParams α σ2 t x k).2 + ch06_rOb t (x k)) := rfl
      rw [hfst, hsnd]
      refine ⟨?_, C', hC', fun w => ?_⟩
      · unfold ch06_lamFwd
        exact div_pos hbar hden
      · rw [ch06_msgFwd_succ]
        exact hstep w

/-- Theorem thm:bp-closure, backward half, for **every** number of hops: the backward
message of eq:bp-bwd is a positive constant times the natural-form Gaussian with
parameters the iterate of `ch06_bwdMap`, with precision `≥ 0` (it is `= 0` exactly at
the right endpoint, where the message is the flat `μ_{←L-1} = 1`). -/
theorem ch06_bp_backward_closure_iterated {α σ2 t : ℝ} (hσ : 0 < σ2) (hD : 0 < ch06_Delta t)
    (x : ℕ → ℝ) (L j : ℕ) :
    0 ≤ (ch06_bwdParams α σ2 t x L j).1 ∧
      ∃ C : ℝ, 0 < C ∧ ∀ b : ℝ,
        ch06_msgBwd (fun _ => ch06_psiTr α σ2) (fun i => ch06_psiOb t (x i)) L j b
          = C * ch06_nat (ch06_bwdParams α σ2 t x L j).1 (ch06_bwdParams α σ2 t x L j).2 b := by
  induction j with
  | zero =>
      refine ⟨?_, 1, one_pos, fun b => ?_⟩
      · simp [ch06_bwdParams]
      · show (1:ℝ) = _
        simp only [ch06_bwdParams, ch06_nat]
        norm_num
  | succ j ih =>
      obtain ⟨hlam, C, hC, hrec⟩ := ih
      have hlamOb : 0 < ch06_lamOb t := div_pos (Real.exp_pos _) hD
      have hbar : 0 < (ch06_bwdParams α σ2 t x L j).1 + ch06_lamOb t := by linarith
      have hden : (0:ℝ) < 1 + σ2 * ((ch06_bwdParams α σ2 t x L j).1 + ch06_lamOb t) := by
        have h1 : (0:ℝ) < σ2 * ((ch06_bwdParams α σ2 t x L j).1 + ch06_lamOb t) := mul_pos hσ hbar
        linarith
      obtain ⟨C', hC', hstep⟩ :=
        ch06_bwd_step_gaussian (α := α) hσ hD (x (L - 1 - j)) hlam hC
          (ch06_msgBwd (fun _ => ch06_psiTr α σ2) (fun i => ch06_psiOb t (x i)) L j) hrec
      have hfst : (ch06_bwdParams α σ2 t x L (j + 1)).1
          = ch06_lamBwd α σ2 ((ch06_bwdParams α σ2 t x L j).1 + ch06_lamOb t) := rfl
      have hsnd : (ch06_bwdParams α σ2 t x L (j + 1)).2
          = ch06_rBwd α σ2 ((ch06_bwdParams α σ2 t x L j).1 + ch06_lamOb t)
              ((ch06_bwdParams α σ2 t x L j).2 + ch06_rOb t (x (L - 1 - j))) := rfl
      rw [hfst, hsnd]
      refine ⟨?_, C', hC', fun b => ?_⟩
      · unfold ch06_lamBwd
        exact div_nonneg (mul_nonneg (sq_nonneg α) (le_of_lt hbar)) (le_of_lt hden)
      · rw [ch06_msgBwd_succ]
        exact hstep b

/-- Theorem thm:bp-closure, belief half: the one-node belief of eq:bp-combine — the
product of the two messages with the local observation factor — is a positive constant
times a *proper* natural-form Gaussian (`λ^post > 0`), with the summed parameters of
eq:bp-belief-precision / eq:bp-belief-field.  With eq:bp-belief-mean this is what makes
`E[a_k | x] = r^post/λ^post` meaningful at every site. -/
theorem ch06_bp_belief_gaussian {α σ2 t : ℝ} (hσ : 0 < σ2) (hD : 0 < ch06_Delta t)
    (x : ℕ → ℝ) (L k : ℕ) :
    ∃ C : ℝ, 0 < C ∧
      0 < (ch06_fwdParams α σ2 t x k).1 + ch06_lamOb t
            + (ch06_bwdParams α σ2 t x L (L - 1 - k)).1 ∧
      ∀ a : ℝ,
        ch06_msgFwd ch06_psiPr (fun _ => ch06_psiTr α σ2) (fun j => ch06_psiOb t (x j)) k a
          * ch06_psiOb t (x k) a
          * ch06_msgBwd (fun _ => ch06_psiTr α σ2) (fun i => ch06_psiOb t (x i)) L (L - 1 - k) a
          = C * ch06_nat ((ch06_fwdParams α σ2 t x k).1 + ch06_lamOb t
                            + (ch06_bwdParams α σ2 t x L (L - 1 - k)).1)
                         ((ch06_fwdParams α σ2 t x k).2 + ch06_rOb t (x k)
                            + (ch06_bwdParams α σ2 t x L (L - 1 - k)).2) a := by
  obtain ⟨hlamF, CF, hCF, hF⟩ := ch06_bp_forward_closure_iterated hσ hD x k
  obtain ⟨hlamB, CB, hCB, hB⟩ := ch06_bp_backward_closure_iterated hσ hD x L (L - 1 - k)
  have hDne : ch06_Delta t ≠ 0 := ne_of_gt hD
  have hlamOb : 0 < ch06_lamOb t := div_pos (Real.exp_pos _) hD
  have h2pi : (0:ℝ) < 2 * Real.pi := by positivity
  have hs2 : (0:ℝ) < Real.sqrt (2 * Real.pi * ch06_Delta t) := Real.sqrt_pos.mpr (mul_pos h2pi hD)
  refine ⟨CF * CB * (Real.exp (-((x k) ^ 2) / (2 * ch06_Delta t))
      / Real.sqrt (2 * Real.pi * ch06_Delta t)),
    mul_pos (mul_pos hCF hCB) (div_pos (Real.exp_pos _) hs2), by linarith, fun a => ?_⟩
  rw [hF a, hB a]
  unfold ch06_psiOb ch06_gauss
  rw [ch06_bp_observation_parameters hDne (x k) a, ← ch06_bp_belief_parameters]
  field_simp

/-- Theorem thm:bp-closure, the "two scalars and a fixed update" half, stated rather
than merely visible in the definitions: the forward sweep carries a state in `ℝ × ℝ`
and advances it by one application of the *same* closed-form map `ch06_fwdMap` at every
site.  The map reads the local datum `x k`, but neither the map nor the size of the
state depends on `k` or on `L` — which is precisely why the per-update cost `c` of
`ch06_gaussSweepOps` is a constant. -/
theorem ch06_bp_gauss_update_fixed (α σ2 t : ℝ) (x : ℕ → ℝ) (k : ℕ) :
    ch06_fwdParams α σ2 t x (k + 1)
      = ch06_fwdMap α σ2 t (x k) (ch06_fwdParams α σ2 t x k) := rfl

/-- The same for the backward sweep: one application of the fixed map `ch06_bwdMap`
per hop, on a state of two scalars. -/
theorem ch06_bp_gauss_update_fixed_bwd (α σ2 t : ℝ) (x : ℕ → ℝ) (L j : ℕ) :
    ch06_bwdParams α σ2 t x L (j + 1)
      = ch06_bwdMap α σ2 t (x (L - 1 - j)) (ch06_bwdParams α σ2 t x L j) := rfl


/-- Theorem thm:bp-closure, space half: the `2L` stored messages of a Gaussian sweep
are `2L` *pairs* of scalars, i.e. `ch06_gaussSweepMem L = 4L` real numbers. -/
theorem ch06_bp_gaussian_mem_count (L : ℕ) :
    Fintype.card (Fin (2 * L) × Fin 2) = ch06_gaussSweepMem L := by
  simp only [Fintype.card_prod, Fintype.card_fin, ch06_gaussSweepMem]
  ring

/-- Theorem thm:bp-closure (noname-8, ch06 lines 324-330), the substantive half of
"all posterior marginals … are obtained exactly in space `O(L)`": a *single* store of
`2L` scalar pairs — `4L = ch06_gaussSweepMem L` numbers, independent of the site — is
enough to reproduce **every** one of the `L` exact smoothing beliefs of eq:bp-combine,
each as a positive constant times a proper (`λ^post > 0`) natural-form Gaussian whose
parameters are read off the store by the additions of
eq:bp-belief-precision / eq:bp-belief-field.  Via eq:bp-belief-mean and `ch06_bp_score`
this is also the complete joint score.  The beliefs here are the genuine
Lebesgue-integral messages of eq:bp-fwd / eq:bp-bwd, and the statement is general in
the chain length. -/
theorem ch06_bp_gaussian_store_sufficient {α σ2 t : ℝ} (hσ : 0 < σ2)
    (hD : 0 < ch06_Delta t) (x : ℕ → ℝ) (L : ℕ) :
    ∃ store : Fin (2 * L) → ℝ × ℝ,
      ∀ (k : ℕ) (hk : k < L),
        ∃ C : ℝ, 0 < C ∧
          0 < (store ⟨k, by omega⟩).1 + ch06_lamOb t + (store ⟨L + k, by omega⟩).1 ∧
          ∀ a : ℝ,
            ch06_msgFwd ch06_psiPr (fun _ => ch06_psiTr α σ2)
                (fun j => ch06_psiOb t (x j)) k a
              * ch06_psiOb t (x k) a
              * ch06_msgBwd (fun _ => ch06_psiTr α σ2) (fun i => ch06_psiOb t (x i))
                  L (L - 1 - k) a
              = C * ch06_nat
                  ((store ⟨k, by omega⟩).1 + ch06_lamOb t + (store ⟨L + k, by omega⟩).1)
                  ((store ⟨k, by omega⟩).2 + ch06_rOb t (x k)
                    + (store ⟨L + k, by omega⟩).2) a := by
  refine ⟨fun i => if (i : ℕ) < L then ch06_fwdParams α σ2 t x (i : ℕ)
            else ch06_bwdParams α σ2 t x L (L - 1 - ((i : ℕ) - L)), ?_⟩
  intro k hk
  have e1 : ((⟨k, by omega⟩ : Fin (2 * L)) : ℕ) = k := rfl
  have e2 : ((⟨L + k, by omega⟩ : Fin (2 * L)) : ℕ) = L + k := rfl
  simp only [e1, e2, if_pos hk, if_neg (show ¬ L + k < L by omega), Nat.add_sub_cancel_left]
  exact ch06_bp_belief_gaussian hσ hD x L k


/-! ### Section 6.4 — the price of locality -/

section Locality

open Matrix

variable {n : Type*} [Fintype n]

/-- eq:bp-banded (ch06 lines 593-598): keep only the entries of `Q_t` inside a band
of half-width `b`. -/
def ch06_banded {N : ℕ} (Q : Matrix (Fin N) (Fin N) ℝ) (b : ℕ) :
    Matrix (Fin N) (Fin N) ℝ :=
  fun i j => if Nat.dist i.val j.val ≤ b then Q i j else 0

/-- eq:bp-banded: the banded score estimator `Ŝ^{(b)}(x,t) = -Q_t^{(b)} x`. -/
def ch06_Shat_banded {N : ℕ} (Q : Matrix (Fin N) (Fin N) ℝ) (b : ℕ) (x : Fin N → ℝ) :
    Fin N → ℝ := -(ch06_banded Q b *ᵥ x)

/-- eq:bp-banded, sanity: at `b = 1` the truncation really is tridiagonal. -/
theorem ch06_banded_one_tridiagonal {N : ℕ} (Q : Matrix (Fin N) (Fin N) ℝ) (i j : Fin N)
    (h : 1 < Nat.dist i.val j.val) : ch06_banded Q 1 i j = 0 := by
  simp [ch06_banded, Nat.not_le.mpr h]

/-- eq:bp-windowed (ch06 lines 608-613): the windowed score estimator built from the
`r`-hop posterior mean `m^{(r)}_k`. -/
def ch06_Shat_windowed (t mr xk : ℝ) : ℝ := (Real.exp (-t) * mr - xk) / ch06_Delta t

/-- noname-18 (ch06 lines 617-619): any linear estimator, `Ŝ(x,t) = -Mx`. -/
def ch06_Shat_lin (M : Matrix n n ℝ) (x : n → ℝ) : n → ℝ := -(M *ᵥ x)

/-- noname-19 (ch06 lines 621-623): the exact Gaussian score, `S(x,t) = -Q_t x`. -/
def ch06_S_exact (Q : Matrix n n ℝ) (x : n → ℝ) : n → ℝ := -(Q *ᵥ x)

/-- noname-20 (ch06 lines 625-629): the error of a linear estimator is
`Ŝ(x,t) - S(x,t) = -(M - Q_t)x`. -/
theorem ch06_bp_error_vector (M Q : Matrix n n ℝ) (x : n → ℝ) :
    ch06_Shat_lin M x - ch06_S_exact Q x = -((M - Q) *ᵥ x) := by
  simp only [ch06_Shat_lin, ch06_S_exact, Matrix.sub_mulVec]
  abel

/-- `‖A x‖² = xᵀ(AᵀA)x`, with `‖·‖²` written as the dot product with itself. -/
theorem ch06_normSq_as_quad (A : Matrix n n ℝ) (x : n → ℝ) :
    (A *ᵥ x) ⬝ᵥ (A *ᵥ x) = x ⬝ᵥ ((Aᵀ * A) *ᵥ x) := by
  rw [← Matrix.mulVec_mulVec, Matrix.dotProduct_mulVec x Aᵀ (A *ᵥ x),
    Matrix.vecMul_transpose]

/-- The pointwise form of eq:bp-quadratic-trace: `xᵀAx = tr(A x xᵀ)`. -/
theorem ch06_quad_eq_trace (A : Matrix n n ℝ) (x : n → ℝ) :
    x ⬝ᵥ (A *ᵥ x) = Matrix.trace (A * Matrix.vecMulVec x x) := by
  simp only [Matrix.trace, Matrix.diag, Matrix.mul_apply, Matrix.vecMulVec_apply,
    dotProduct, Matrix.mulVec, Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
  first | rfl | ring

/-- eq:bp-quadratic-trace (ch06 lines 654-663): `E[xᵀAx] = tr(A Σ)`, with the
expectation modelled as a finite sum over samples and `Sg = Σᵢ xᵢ xᵢᵀ` the
corresponding (unnormalised) second-moment matrix.  Dividing both sides by the
sample count gives the average form; the continuous case is the same two steps,
`xᵀAx = tr(A xxᵀ)` followed by linearity. -/
theorem ch06_bp_quadratic_trace {ι : Type*} [Fintype ι] (A : Matrix n n ℝ) (x : ι → n → ℝ) :
    (∑ i : ι, (x i) ⬝ᵥ (A *ᵥ (x i)))
      = Matrix.trace (A * ∑ i : ι, Matrix.vecMulVec (x i) (x i)) := by
  rw [Finset.mul_sum, Matrix.trace_sum]
  exact Finset.sum_congr rfl fun i _ => ch06_quad_eq_trace A (x i)

/-- eq:bp-error-numerator (ch06 lines 666-677): `tr[(M-Q)ᵀ(M-Q)Σ] = tr[(M-Q)Σ(M-Q)ᵀ]`
by cyclicity of the trace. -/
theorem ch06_bp_error_numerator (M Q Sg : Matrix n n ℝ) :
    Matrix.trace ((M - Q)ᵀ * (M - Q) * Sg) = Matrix.trace ((M - Q) * Sg * (M - Q)ᵀ) := by
  rw [mul_assoc ((M - Q)ᵀ) (M - Q) Sg, Matrix.trace_mul_comm]

/-! #### The expectation step of eq:bp-error-numerator

`ch06_bp_error_numerator` above is only the *second* equality of eq:bp-error-numerator
(cyclicity of the trace).  The first equality is where the data law `x ∼ N(0, Σ_t)`
actually enters, and it is carried out below for a genuine Bochner integral:
`ch06_bp_quadratic_trace_integral` is eq:bp-quadratic-trace as an expectation over an
arbitrary law with second-moment matrix `Σ`, `ch06_bp_error_numerator_expectation`
composes it with `ch06_normSq_as_quad` to give the first display, and
`ch06_bp_error_numerator_gaussian` specialises the whole chain to the chapter's own law,
mathlib's `multivariateGaussian 0 Σ`.  Throughout, `‖v‖²` is written `v ⬝ᵥ v`;
`ch06_dotProduct_self_eq_normSq` records that this is the squared Euclidean norm. -/

/-- `v ⬝ᵥ v` really is the squared Euclidean norm of `v`. -/
theorem ch06_dotProduct_self_eq_normSq (v : n → ℝ) :
    v ⬝ᵥ v = ‖(WithLp.toLp 2 v : EuclideanSpace ℝ n)‖ ^ 2 := by
  rw [EuclideanSpace.real_norm_sq_eq]
  simp [dotProduct, pow_two]

/-- `xᵀAx = ∑ᵢ ∑ⱼ Aᵢⱼ xᵢ xⱼ`: the entrywise form in which the expectation is taken. -/
theorem ch06_quad_expand (A : Matrix n n ℝ) (x : n → ℝ) :
    x ⬝ᵥ (A *ᵥ x) = ∑ i, ∑ j, A i j * (x i * x j) := by
  simp only [dotProduct, Matrix.mulVec, Finset.mul_sum]
  exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by ring

/-- eq:bp-quadratic-trace (ch06 lines 654-663) as a genuine expectation: if the random
vector `X` on `(Ω, μ)` has second moments `E[xᵢxⱼ] = Σᵢⱼ` — so `Σ` is its covariance
matrix whenever `X` is centred — then `E[xᵀAx] = tr(AΣ)` for every matrix `A`.  Only the
second moments of the law enter, and Gaussianity is never used. -/
theorem ch06_bp_quadratic_trace_integral {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    (X : Ω → n → ℝ) (Sg A : Matrix n n ℝ)
    (hint : ∀ i j, Integrable (fun ω => X ω i * X ω j) μ)
    (hcov : ∀ i j, ∫ ω, X ω i * X ω j ∂μ = Sg i j) :
    ∫ ω, (X ω) ⬝ᵥ (A *ᵥ X ω) ∂μ = Matrix.trace (A * Sg) := by
  have hintA : ∀ i j : n, Integrable (fun ω => A i j * (X ω i * X ω j)) μ :=
    fun i j => (hint i j).const_mul (A i j)
  have hrow : ∀ i : n, ∫ ω, ∑ j, A i j * (X ω i * X ω j) ∂μ = ∑ j, A i j * Sg i j := by
    intro i
    rw [integral_finsetSum _ fun j _ => hintA i j]
    exact Finset.sum_congr rfl fun j _ => by rw [integral_const_mul, hcov i j]
  have hsymm : ∀ i j : n, Sg i j = Sg j i := by
    intro i j
    rw [← hcov i j, ← hcov j i]
    simp_rw [mul_comm]
  calc ∫ ω, (X ω) ⬝ᵥ (A *ᵥ X ω) ∂μ
      = ∫ ω, ∑ i, ∑ j, A i j * (X ω i * X ω j) ∂μ := by simp_rw [ch06_quad_expand]
    _ = ∑ i, ∑ j, A i j * Sg i j := by
        rw [integral_finsetSum _ fun i _ => integrable_finsetSum _ fun j _ => hintA i j]
        exact Finset.sum_congr rfl fun i _ => hrow i
    _ = Matrix.trace (A * Sg) := by
        simp only [Matrix.trace, Matrix.diag, Matrix.mul_apply]
        exact Finset.sum_congr rfl fun i _ =>
          Finset.sum_congr rfl fun j _ => by rw [hsymm i j]

/-- eq:bp-error-numerator, first equality (ch06 lines 666-672): for any data law with
second-moment matrix `Σ`, the expected squared error of the linear estimator
`Ŝ(x,t) = -Mx` against the exact score `S(x,t) = -Q_t x` is `tr[(M - Q_t)ᵀ(M - Q_t)Σ]`.
This is the step the chapter takes with `A = (M - Q_t)ᵀ(M - Q_t)`. -/
theorem ch06_bp_error_numerator_expectation {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    (X : Ω → n → ℝ) (M Q Sg : Matrix n n ℝ)
    (hint : ∀ i j, Integrable (fun ω => X ω i * X ω j) μ)
    (hcov : ∀ i j, ∫ ω, X ω i * X ω j ∂μ = Sg i j) :
    ∫ ω, (ch06_Shat_lin M (X ω) - ch06_S_exact Q (X ω)) ⬝ᵥ
        (ch06_Shat_lin M (X ω) - ch06_S_exact Q (X ω)) ∂μ
      = Matrix.trace ((M - Q)ᵀ * (M - Q) * Sg) := by
  have hpt : ∀ ω, (ch06_Shat_lin M (X ω) - ch06_S_exact Q (X ω)) ⬝ᵥ
      (ch06_Shat_lin M (X ω) - ch06_S_exact Q (X ω))
      = (X ω) ⬝ᵥ (((M - Q)ᵀ * (M - Q)) *ᵥ X ω) := by
    intro ω
    rw [ch06_bp_error_vector, neg_dotProduct_neg]
    exact ch06_normSq_as_quad (M - Q) (X ω)
  simp_rw [hpt]
  exact ch06_bp_quadratic_trace_integral μ X Sg _ hint hcov

/-- eq:bp-error-numerator (ch06 lines 666-677), both equalities at once, for any data law
with second-moment matrix `Σ`. -/
theorem ch06_bp_error_numerator_chain {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    (X : Ω → n → ℝ) (M Q Sg : Matrix n n ℝ)
    (hint : ∀ i j, Integrable (fun ω => X ω i * X ω j) μ)
    (hcov : ∀ i j, ∫ ω, X ω i * X ω j ∂μ = Sg i j) :
    ∫ ω, (ch06_Shat_lin M (X ω) - ch06_S_exact Q (X ω)) ⬝ᵥ
        (ch06_Shat_lin M (X ω) - ch06_S_exact Q (X ω)) ∂μ
      = Matrix.trace ((M - Q) * Sg * (M - Q)ᵀ) := by
  rw [ch06_bp_error_numerator_expectation μ X M Q Sg hint hcov, ch06_bp_error_numerator]

open ProbabilityTheory in
/-- eq:bp-error-numerator (ch06 lines 666-677) at the chapter's own data law: for
`x ∼ N(0, Σ_t)` — mathlib's `multivariateGaussian 0 Σ_t`, with `Σ_t` positive semidefinite
as a covariance matrix is — the expectation `E‖Ŝ(x,t) - S(x,t)‖²` of the squared score
error equals `tr[(M - Q_t)Σ_t(M - Q_t)ᵀ]`.  The expectation is a Bochner integral against
the Gaussian measure and `Σ_t` is its actual covariance matrix
(`covariance_eval_multivariateGaussian`), so no step of the display is assumed. -/
theorem ch06_bp_error_numerator_gaussian [DecidableEq n] (M Q Sg : Matrix n n ℝ)
    (hSg : Sg.PosSemidef) :
    ∫ x : EuclideanSpace ℝ n,
        (ch06_Shat_lin M x.ofLp - ch06_S_exact Q x.ofLp) ⬝ᵥ
          (ch06_Shat_lin M x.ofLp - ch06_S_exact Q x.ofLp)
          ∂(multivariateGaussian 0 Sg)
      = Matrix.trace ((M - Q) * Sg * (M - Q)ᵀ) := by
  have hmem : ∀ i : n, MemLp (fun x : EuclideanSpace ℝ n => x i) 2
      (multivariateGaussian 0 Sg) := by
    intro i
    have h : MemLp (id : EuclideanSpace ℝ n → EuclideanSpace ℝ n) 2
        (multivariateGaussian 0 Sg) := IsGaussian.memLp_two_id
    exact memLp_piLp_iff.mp h i
  have hmean : ∀ i : n, ∫ x : EuclideanSpace ℝ n, x i ∂(multivariateGaussian 0 Sg) = 0 := by
    intro i
    have hI : Integrable (id : EuclideanSpace ℝ n → EuclideanSpace ℝ n)
        (multivariateGaussian 0 Sg) := IsGaussian.integrable_id
    simpa using (EuclideanSpace.proj (𝕜 := ℝ) i).integral_comp_comm hI
  have hint : ∀ i j : n, Integrable (fun x : EuclideanSpace ℝ n => x i * x j)
      (multivariateGaussian 0 Sg) := by
    intro i j
    simpa [Pi.mul_def] using (hmem i).integrable_mul (hmem j)
  have hcov : ∀ i j : n, ∫ x : EuclideanSpace ℝ n, x i * x j ∂(multivariateGaussian 0 Sg)
      = Sg i j := by
    intro i j
    have h := covariance_eq_sub (hmem i) (hmem j)
    rw [covariance_eval_multivariateGaussian hSg i j] at h
    simp only [Pi.mul_apply, hmean i, hmean j, mul_zero, sub_zero] at h
    exact h.symm
  exact ch06_bp_error_numerator_chain (multivariateGaussian 0 Sg)
    (fun x : EuclideanSpace ℝ n => x.ofLp) M Q Sg hint hcov

/-- eq:bp-error-denominator (ch06 lines 682-689): with `Q = Σ⁻¹` and both matrices
symmetric, `E‖Qx‖² = tr(QΣQ) = tr(Q)`. -/
theorem ch06_bp_error_denominator [DecidableEq n] (Q Sg : Matrix n n ℝ)
    (hQsym : Qᵀ = Q) (hQS : Q * Sg = 1) :
    Matrix.trace (Qᵀ * Q * Sg) = Matrix.trace Q := by
  rw [hQsym, mul_assoc, hQS, mul_one]

/-- eq:bp-relerr-def (ch06 lines 638-646): the relative RMS score error. -/
def ch06_relerr (M Q Sg : Matrix n n ℝ) : ℝ :=
  Real.sqrt (Matrix.trace ((M - Q) * Sg * (M - Q)ᵀ) / Matrix.trace Q)

/-- eq:bp-relerr (ch06 lines 691-706): combining the two evaluations, the population
error is deterministic once `M`, `Σ_t` and `Q_t` are known. -/
theorem ch06_bp_relerr [DecidableEq n] (M Q Sg : Matrix n n ℝ)
    (hQsym : Qᵀ = Q) (hQS : Q * Sg = 1) :
    Real.sqrt (Matrix.trace ((M - Q)ᵀ * (M - Q) * Sg) / Matrix.trace (Qᵀ * Q * Sg))
      = ch06_relerr M Q Sg := by
  rw [ch06_relerr, ch06_bp_error_numerator, ch06_bp_error_denominator Q Sg hQsym hQS]

/-- eq:bp-relerr, sanity: an exact estimator has zero error. -/
theorem ch06_relerr_self (Q Sg : Matrix n n ℝ) : ch06_relerr Q Q Sg = 0 := by
  simp [ch06_relerr]

end Locality

end

end ThesisAudit
