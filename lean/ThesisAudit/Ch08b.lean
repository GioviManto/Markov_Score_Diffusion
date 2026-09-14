import Mathlib

/-!
# Audit of ch08-em-parameters.tex, lines 601-1383

Formal check of every display-math formula in the second half of the thesis
chapter "Learning the Parameters: EM on the Chain": the E-step (pairwise
beliefs, responsibilities, sufficient statistics), the M-step (all four
closed-form conditional maximisers), the ECM monotonicity argument, and the
small-time score-amplification estimate of the numerical section.

Modelling convention (same as `ThesisAudit.Ch08a`): the chapter's integrals
`\int\!\!\int f(a_{i-1},a_i) b_i(a_{i-1},a_i)\,da_{i-1}da_i` are realised as
finite weighted sums over a fintype `Ω` of quadrature nodes carrying the
pairwise belief weight `b`, the parent value `x ω = a_{i-1}` and the child
value `y ω = a_i`.  This is *exactly* the object the implementation
manipulates (the chapter itself discretises on a grid in
Section~\ref{sec:em-numerical}), and it makes the E-step/M-step algebra
genuine theorems rather than statements needing dominated convergence.
The sum over edges `i` is absorbed into `Ω` (the chapter says the statistics
are "simply summed over all edges of all sequences").
-/

namespace ThesisAudit

open Finset Real

noncomputable section

/-! ## Generic helpers -/

/-- Derivative of a generic quadratic `a + b t + c t²`. -/
theorem ch08b_quad_hasDerivAt (a b c x : ℝ) :
    HasDerivAt (fun t : ℝ => a + b * t + c * t ^ 2) (b + 2 * c * x) x := by
  have h1 : HasDerivAt (fun t : ℝ => t) 1 x := by simpa using hasDerivAt_pow 1 x
  have h2 : HasDerivAt (fun t : ℝ => t ^ 2) (2 * x) x := by simpa using hasDerivAt_pow 2 x
  have h3 : HasDerivAt (fun t : ℝ => a + b * t + c * t ^ 2) (0 + b * 1 + c * (2 * x)) x :=
    ((hasDerivAt_const x a).add (h1.const_mul b)).add (h2.const_mul c)
  have he : (0 : ℝ) + b * 1 + c * (2 * x) = b + 2 * c * x := by ring
  rwa [he] at h3

/-- A quadratic with positive leading coefficient is minimised exactly at the
stationary point, with the exact excess `c (t - t₀)²`. -/
theorem ch08b_quad_excess (a b c t t₀ : ℝ) (hc : c ≠ 0) (ht₀ : t₀ = -b / (2 * c)) :
    (a + b * t + c * t ^ 2) - (a + b * t₀ + c * t₀ ^ 2) = c * (t - t₀) ^ 2 := by
  subst ht₀
  field_simp
  ring

/-- Gibbs' inequality in the form needed for the mixture-weight M-step:
`Σ R_c log π_c` over the simplex is maximised at `π_c = R_c / Σ_d R_d`. -/
theorem ch08b_gibbs {C : ℕ} (R p : Fin C → ℝ) (hR : ∀ c, 0 < R c) (hp : ∀ c, 0 < p c)
    (hsum : ∑ c, p c = 1) :
    ∑ c, R c * Real.log (p c) ≤ ∑ c, R c * Real.log (R c / ∑ d, R d) := by
  rcases Finset.eq_empty_or_nonempty (Finset.univ : Finset (Fin C)) with h | h
  · simp [h]
  · have hT : 0 < ∑ d, R d := Finset.sum_pos (fun c _ => hR c) h
    have key : ∀ c : Fin C,
        R c * Real.log (p c) - R c * Real.log (R c / ∑ d, R d) ≤ p c * (∑ d, R d) - R c := by
      intro c
      have e1 : Real.log (R c / ∑ d, R d) = Real.log (R c) - Real.log (∑ d, R d) :=
        Real.log_div (ne_of_gt (hR c)) (ne_of_gt hT)
      have e2 : Real.log (p c * (∑ d, R d) / R c)
          = Real.log (p c) + Real.log (∑ d, R d) - Real.log (R c) := by
        rw [Real.log_div (ne_of_gt (mul_pos (hp c) hT)) (ne_of_gt (hR c)),
          Real.log_mul (ne_of_gt (hp c)) (ne_of_gt hT)]
      have hlog : Real.log (p c * (∑ d, R d) / R c) ≤ p c * (∑ d, R d) / R c - 1 :=
        Real.log_le_sub_one_of_pos (div_pos (mul_pos (hp c) hT) (hR c))
      have hmul := mul_le_mul_of_nonneg_left hlog (le_of_lt (hR c))
      rw [e2] at hmul
      have hRc : R c ≠ 0 := ne_of_gt (hR c)
      have hrew : R c * (p c * (∑ d, R d) / R c - 1) = p c * (∑ d, R d) - R c := by
        field_simp <;> ring
      rw [hrew] at hmul
      have hexp : R c * (Real.log (p c) + Real.log (∑ d, R d) - Real.log (R c))
          = R c * Real.log (p c) - R c * (Real.log (R c) - Real.log (∑ d, R d)) := by ring
      rw [hexp] at hmul
      rw [e1]
      linarith
    have hle : ∑ c, (R c * Real.log (p c) - R c * Real.log (R c / ∑ d, R d))
        ≤ ∑ c, (p c * (∑ d, R d) - R c) := Finset.sum_le_sum (fun c _ => key c)
    rw [Finset.sum_sub_distrib, Finset.sum_sub_distrib, ← Finset.sum_mul, hsum, one_mul,
      sub_self] at hle
    linarith

/-- Non-negativity of the discrete Kullback-Leibler divergence. -/
theorem ch08b_KL_nonneg {Z : Type*} [Fintype Z] (q p : Z → ℝ) (hq : ∀ z, 0 < q z)
    (hp : ∀ z, 0 < p z) (hq1 : ∑ z, q z = 1) (hp1 : ∑ z, p z = 1) :
    0 ≤ ∑ z, q z * Real.log (q z / p z) := by
  have key : ∀ z : Z, q z - p z ≤ q z * Real.log (q z / p z) := by
    intro z
    have h1 : Real.log (p z / q z) ≤ p z / q z - 1 :=
      Real.log_le_sub_one_of_pos (div_pos (hp z) (hq z))
    have h2 := mul_le_mul_of_nonneg_left h1 (le_of_lt (hq z))
    have hqz : q z ≠ 0 := ne_of_gt (hq z)
    have h3 : q z * (p z / q z - 1) = p z - q z := by field_simp <;> ring
    rw [h3] at h2
    have h4 : Real.log (p z / q z) = -Real.log (q z / p z) := by
      rw [Real.log_div (ne_of_gt (hp z)) (ne_of_gt (hq z)),
        Real.log_div (ne_of_gt (hq z)) (ne_of_gt (hp z))]
      ring
    rw [h4] at h2
    linarith
  have hle : ∑ z, (q z - p z) ≤ ∑ z, q z * Real.log (q z / p z) :=
    Finset.sum_le_sum (fun z _ => key z)
  rw [Finset.sum_sub_distrib, hq1, hp1, sub_self] at hle
  exact hle

/-! ## 8.3 The E-step -/

-- eq:em-pairwise (ch08-em-parameters.tex lines 614-623):
-- b_i(a_{i-1},a_i) ∝ λ_{i-1}(a_{i-1}) φ_{i-1}(a_{i-1}) K(a_i|a_{i-1}) φ_i(a_i) ρ_i(a_i).
/-- The message-product form of the pairwise marginal on a chain.  Summing the
unnormalised joint weight over *everything strictly to the left* (`u : U`) and
*everything strictly to the right* (`v : V`) of the edge collapses into the
forward message `λ_{i-1}(a) = Σ_u Lpart u a` and the backward message
`ρ_i(a') = Σ_v Rpart a' v`, leaving exactly the five-factor product of
`eq:em-pairwise`. -/
theorem ch08b_em_pairwise {S U V : Type*} [Fintype U] [Fintype V]
    (Lpart : U → S → ℝ) (Rpart : S → V → ℝ) (phiL phiR : S → ℝ) (K : S → S → ℝ)
    (a a' : S) :
    (∑ u : U, ∑ v : V, Lpart u a * phiL a * K a' a * phiR a' * Rpart a' v)
      = (∑ u : U, Lpart u a) * phiL a * K a' a * phiR a' * (∑ v : V, Rpart a' v) := by
  have h1 : ∀ u : U, (∑ v : V, Lpart u a * phiL a * K a' a * phiR a' * Rpart a' v)
      = (Lpart u a * phiL a * K a' a * phiR a') * ∑ v : V, Rpart a' v := by
    intro u; rw [Finset.mul_sum]
  rw [Finset.sum_congr rfl (fun u _ => h1 u), ← Finset.sum_mul]
  have h2 : (∑ u : U, Lpart u a * phiL a * K a' a * phiR a')
      = (∑ u : U, Lpart u a) * (phiL a * K a' a * phiR a') := by
    rw [Finset.sum_mul]
    exact Finset.sum_congr rfl (fun u _ => by ring)
  rw [h2]; ring

-- eq:em-pairwise-def (lines 606-611): b_i(a_{i-1},a_i) := p_{θ^(k)}(a_{i-1},a_i | x).
/-- The pairwise posterior: the unnormalised message product, normalised. -/
def ch08b_em_pairwise_def {S : Type*} [Fintype S] (lam phiL : S → ℝ) (K : S → S → ℝ)
    (phiR rho : S → ℝ) (a a' : S) : ℝ :=
  (lam a * phiL a * K a' a * phiR a' * rho a') /
    ∑ u : S, ∑ v : S, lam u * phiL u * K v u * phiR v * rho v

-- noname-1 (lines 625-630): ∫∫ b_i(a_{i-1},a_i) da_{i-1} da_i = 1.
/-- The pairwise posterior is normalised. -/
theorem ch08b_em_pairwise_normalised {S : Type*} [Fintype S] (lam phiL : S → ℝ)
    (K : S → S → ℝ) (phiR rho : S → ℝ)
    (hZ : (∑ u : S, ∑ v : S, lam u * phiL u * K v u * phiR v * rho v) ≠ 0) :
    (∑ u : S, ∑ v : S, ch08b_em_pairwise_def lam phiL K phiR rho u v) = 1 := by
  unfold ch08b_em_pairwise_def
  simp only [← Finset.sum_div]
  exact div_self hZ

/-- General statement: any nonnegative weight normalised by its own total mass
sums to one. -/
theorem ch08b_normalisation {S : Type*} [Fintype S] (b : S → S → ℝ)
    (hZ : (∑ u : S, ∑ v : S, b u v) ≠ 0) :
    (∑ u : S, ∑ v : S, b u v / ∑ p : S, ∑ q : S, b p q) = 1 := by
  simp only [← Finset.sum_div]
  exact div_self hZ

/-- Gaussian density `N(u ; ν, s²)` (same convention as `Ch08a`). -/
def ch08b_gauss (ν s u : ℝ) : ℝ :=
  Real.exp (-((u - ν) ^ 2) / (2 * s ^ 2)) / Real.sqrt (2 * Real.pi * s ^ 2)

/-- The `C`-component mixture transition kernel `K_θ(a' | a)` of eq:em-mixture-kernel. -/
def ch08b_kernel {C : ℕ} (p ν s : Fin C → ℝ) (α a' a : ℝ) : ℝ :=
  ∑ c : Fin C, p c * ch08b_gauss (ν c) (s c) (a' - α * a)

-- eq:em-responsibility (lines 635-673):
-- r_{i,c} := P(c_i = c | a_{i-1}, a_i) = π_c N(...) / Σ_d π_d N(...) = π_c N(...) / K(a_i|a_{i-1}).
/-- The mixture responsibility. -/
def ch08b_em_responsibility {C : ℕ} (p ν s : Fin C → ℝ) (α a' a : ℝ) (c : Fin C) : ℝ :=
  p c * ch08b_gauss (ν c) (s c) (a' - α * a) / ch08b_kernel p ν s α a' a

/-- Bayes' rule on a finite mixture: the posterior over the label is the
component weight divided by the mixture density, and the mixture density is
exactly the transition kernel.  This is the content of the last equality of
eq:em-responsibility. -/
theorem ch08b_em_responsibility_eq {C : ℕ} (p ν s : Fin C → ℝ) (α a' a : ℝ) (c : Fin C) :
    ch08b_em_responsibility p ν s α a' a c
      = p c * ch08b_gauss (ν c) (s c) (a' - α * a) /
          ∑ d : Fin C, p d * ch08b_gauss (ν d) (s d) (a' - α * a) := rfl

-- noname-2 (lines 675-679): Σ_c r_{i,c}(a_{i-1},a_i) = 1.
/-- The responsibilities sum to one. -/
theorem ch08b_em_responsibility_sum {C : ℕ} (p ν s : Fin C → ℝ) (α a' a : ℝ)
    (hK : ch08b_kernel p ν s α a' a ≠ 0) :
    (∑ c : Fin C, ch08b_em_responsibility p ν s α a' a c) = 1 := by
  unfold ch08b_em_responsibility ch08b_kernel
  rw [← Finset.sum_div]
  exact div_self (by simpa [ch08b_kernel] using hK)

-- eq:em-bracket (lines 682-690): ⟨f⟩_i := ∫∫ f(a_{i-1},a_i) b_i(a_{i-1},a_i) da_{i-1} da_i.
/-- Expectation against the pairwise posterior, as a finite weighted sum. -/
def ch08b_em_bracket {Ω : Type*} [Fintype Ω] (b x y : Ω → ℝ) (f : ℝ → ℝ → ℝ) : ℝ :=
  ∑ ω : Ω, f (x ω) (y ω) * b ω

-- eq:em-indicator-responsibility (lines 692-700):
-- E[1{c_i = c} | a_{i-1}, a_i, x, θ^(k)] = r_{i,c}^{(k)}(a_{i-1}, a_i).
/-- The conditional expectation of the label indicator is the responsibility. -/
theorem ch08b_em_indicator_responsibility {C : ℕ} (r : Fin C → ℝ) (c : Fin C) :
    (∑ d : Fin C, (if d = c then (1 : ℝ) else 0) * r d) = r c := by
  simp

-- eq:em-suffstats (lines 705-748): the six posterior sufficient statistics
-- R_c, U_c, V_c, P_c, W_c, G_c.
/-- `R_c = Σ_i ⟨r_{i,c}⟩_i`: posterior effective number of transitions. -/
def ch08b_R {Ω : Type*} [Fintype Ω] (b r : Ω → ℝ) : ℝ := ∑ ω : Ω, b ω * r ω

/-- `U_c = Σ_i ⟨r_{i,c} a_i⟩_i`: weighted first moment of the child state. -/
def ch08b_U {Ω : Type*} [Fintype Ω] (b r y : Ω → ℝ) : ℝ := ∑ ω : Ω, b ω * r ω * y ω

/-- `V_c = Σ_i ⟨r_{i,c} a_{i-1}⟩_i`: weighted first moment of the parent state. -/
def ch08b_V {Ω : Type*} [Fintype Ω] (b r x : Ω → ℝ) : ℝ := ∑ ω : Ω, b ω * r ω * x ω

/-- `P_c = Σ_i ⟨r_{i,c} a_{i-1} a_i⟩_i`: parent-child cross moment. -/
def ch08b_P {Ω : Type*} [Fintype Ω] (b r x y : Ω → ℝ) : ℝ :=
  ∑ ω : Ω, b ω * r ω * (x ω * y ω)

/-- `W_c = Σ_i ⟨r_{i,c} a_{i-1}²⟩_i`. -/
def ch08b_W {Ω : Type*} [Fintype Ω] (b r x : Ω → ℝ) : ℝ := ∑ ω : Ω, b ω * r ω * x ω ^ 2

/-- `G_c = Σ_i ⟨r_{i,c} a_i²⟩_i`. -/
def ch08b_G {Ω : Type*} [Fintype Ω] (b r y : Ω → ℝ) : ℝ := ∑ ω : Ω, b ω * r ω * y ω ^ 2

/-- "For several training sequences the same quantities are simply summed over
all edges of all sequences": the statistics are additive over a disjoint union
of node sets. -/
theorem ch08b_R_additive {Ω₁ Ω₂ : Type*} [Fintype Ω₁] [Fintype Ω₂]
    (b r : Ω₁ ⊕ Ω₂ → ℝ) :
    ch08b_R b r = ch08b_R (fun ω => b (Sum.inl ω)) (fun ω => r (Sum.inl ω))
      + ch08b_R (fun ω => b (Sum.inr ω)) (fun ω => r (Sum.inr ω)) := by
  unfold ch08b_R
  rw [Fintype.sum_sum_type]

/-! ## 8.4 The M-step -/

-- eq:em-residual-expand (lines 767-777):
-- (a_i - α a_{i-1} - ν_c)² = a_i² - 2α a_{i-1}a_i + α² a_{i-1}² - 2ν_c a_i + 2α ν_c a_{i-1} + ν_c².
theorem ch08b_em_residual_expand (ai aprev α ν : ℝ) :
    (ai - α * aprev - ν) ^ 2
      = ai ^ 2 - 2 * α * (aprev * ai) + α ^ 2 * aprev ^ 2
        - 2 * ν * ai + 2 * α * ν * aprev + ν ^ 2 := by ring

-- eq:em-residual-stat (lines 779-789):
-- S_c(α,ν_c) := G_c - 2α P_c + α² W_c - 2ν_c U_c + 2α ν_c V_c + ν_c² R_c.
/-- The posterior-weighted residual, as a function of the sufficient statistics. -/
def ch08b_S (G P W U V R α ν : ℝ) : ℝ :=
  G - 2 * α * P + α ^ 2 * W - 2 * ν * U + 2 * α * ν * V + ν ^ 2 * R

/-- `S_c` really is the posterior-weighted sum of squared residuals. -/
theorem ch08b_em_residual_stat {Ω : Type*} [Fintype Ω] (b r x y : Ω → ℝ) (α ν : ℝ) :
    (∑ ω : Ω, b ω * r ω * (y ω - α * x ω - ν) ^ 2)
      = ch08b_S (ch08b_G b r y) (ch08b_P b r x y) (ch08b_W b r x) (ch08b_U b r y)
          (ch08b_V b r x) (ch08b_R b r) α ν := by
  unfold ch08b_S ch08b_G ch08b_P ch08b_W ch08b_U ch08b_V ch08b_R
  have hsplit : ∀ ω : Ω, b ω * r ω * (y ω - α * x ω - ν) ^ 2
      = b ω * r ω * y ω ^ 2 - 2 * α * (b ω * r ω * (x ω * y ω))
        + α ^ 2 * (b ω * r ω * x ω ^ 2) - 2 * ν * (b ω * r ω * y ω)
        + 2 * α * ν * (b ω * r ω * x ω) + ν ^ 2 * (b ω * r ω) := by
    intro ω; ring
  rw [Finset.sum_congr rfl (fun ω _ => hsplit ω)]
  simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum]

-- eq:em-Q-suffstats (lines 791-807):
-- Q(θ|θ^(k)) =^c Σ_c [ R_c log π_c - (R_c/2) log s_c² - S_c(α,ν_c)/(2 s_c²) ].
/-- One component's contribution to the surrogate, in closed form. -/
theorem ch08b_Q_component {Ω : Type*} [Fintype Ω] (b r x y : Ω → ℝ) (α ν lp s2 : ℝ)
    (hs2 : s2 ≠ 0) :
    (∑ ω : Ω, b ω * r ω *
        (lp - (1 / 2) * Real.log s2 - (y ω - α * x ω - ν) ^ 2 / (2 * s2)))
      = ch08b_R b r * lp - (ch08b_R b r / 2) * Real.log s2
        - ch08b_S (ch08b_G b r y) (ch08b_P b r x y) (ch08b_W b r x) (ch08b_U b r y)
            (ch08b_V b r x) (ch08b_R b r) α ν / (2 * s2) := by
  have hres := ch08b_em_residual_stat b r x y α ν
  have hsplit : ∀ ω : Ω, b ω * r ω *
      (lp - (1 / 2) * Real.log s2 - (y ω - α * x ω - ν) ^ 2 / (2 * s2))
      = lp * (b ω * r ω) - (Real.log s2 / 2) * (b ω * r ω)
        - (1 / (2 * s2)) * (b ω * r ω * (y ω - α * x ω - ν) ^ 2) := by
    intro ω; field_simp <;> ring
  rw [Finset.sum_congr rfl (fun ω _ => hsplit ω), Finset.sum_sub_distrib, Finset.sum_sub_distrib,
    ← Finset.mul_sum, ← Finset.mul_sum, ← Finset.mul_sum, hres]
  unfold ch08b_R
  field_simp <;> ring

/-- The surrogate in terms of the sufficient statistics (eq:em-Q-suffstats). -/
theorem ch08b_em_Q_suffstats {Ω : Type*} [Fintype Ω] {C : ℕ}
    (b x y : Ω → ℝ) (r : Fin C → Ω → ℝ) (p ν s2 : Fin C → ℝ) (α : ℝ)
    (hs2 : ∀ c, s2 c ≠ 0) :
    (∑ ω : Ω, ∑ c : Fin C, b ω * r c ω *
        (Real.log (p c) - (1 / 2) * Real.log (s2 c)
          - (y ω - α * x ω - ν c) ^ 2 / (2 * s2 c)))
      = ∑ c : Fin C, (ch08b_R b (r c) * Real.log (p c)
          - (ch08b_R b (r c) / 2) * Real.log (s2 c)
          - ch08b_S (ch08b_G b (r c) y) (ch08b_P b (r c) x y) (ch08b_W b (r c) x)
              (ch08b_U b (r c) y) (ch08b_V b (r c) x) (ch08b_R b (r c)) α (ν c)
              / (2 * s2 c)) := by
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl
    (fun c _ => ch08b_Q_component b (r c) x y α (ν c) (Real.log (p c)) (s2 c) (hs2 c))

/-! ### Mixture weights -/

-- noname-3 (lines 815-820): Q_π = Σ_c R_c log π_c.
/-- The weight-dependent part of the surrogate. -/
def ch08b_Qpi {C : ℕ} (R p : Fin C → ℝ) : ℝ := ∑ c : Fin C, R c * Real.log (p c)

-- noname-4 (lines 822-824): Σ_c π_c = 1.
/-- The simplex constraint. -/
def ch08b_simplex {C : ℕ} (p : Fin C → ℝ) : Prop := (∀ c, 0 ≤ p c) ∧ ∑ c : Fin C, p c = 1

-- noname-5 (lines 826-835): J = Σ_c R_c log π_c + λ(Σ_c π_c - 1).
/-- The Lagrangian for the constrained weight maximisation. -/
def ch08b_lagrangian {C : ℕ} (R p : Fin C → ℝ) (lam : ℝ) : ℝ :=
  (∑ c : Fin C, R c * Real.log (p c)) + lam * ((∑ c : Fin C, p c) - 1)

-- noname-6 (lines 837-845): ∂J/∂π_c = R_c/π_c + λ = 0.
/-- Differentiating the Lagrangian in a single weight gives `R_c/π_c + λ`.
(The remaining components enter only through the constant `rest`.) -/
theorem ch08b_lagrangian_deriv (R lam rest pc : ℝ) (hpc : pc ≠ 0) :
    HasDerivAt (fun t : ℝ => R * Real.log t + lam * t + rest) (R / pc + lam) pc := by
  have hl : HasDerivAt Real.log pc⁻¹ pc := Real.hasDerivAt_log hpc
  have h1 : HasDerivAt (fun t : ℝ => t) 1 pc := by simpa using hasDerivAt_pow 1 pc
  have h3 : HasDerivAt (fun t : ℝ => R * Real.log t + lam * t + rest)
      (R * pc⁻¹ + lam * 1 + 0) pc :=
    ((hl.const_mul R).add (h1.const_mul lam)).add (hasDerivAt_const pc rest)
  have he : R * pc⁻¹ + lam * 1 + 0 = R / pc + lam := by field_simp <;> ring
  rwa [he] at h3

-- noname-7 (lines 847-851): π_c = -R_c/λ.
theorem ch08b_pi_from_lambda (R lam pc : ℝ) (hpc : pc ≠ 0) (hlam : lam ≠ 0)
    (h : R / pc + lam = 0) : pc = -R / lam := by
  field_simp at h ⊢
  linarith

-- noname-8 (lines 853-857): λ = -Σ_d R_d.
theorem ch08b_lambda_value {C : ℕ} (R p : Fin C → ℝ) (lam : ℝ) (hlam : lam ≠ 0)
    (hp : ∀ c, p c = -R c / lam) (hsum : ∑ c : Fin C, p c = 1) :
    lam = -∑ d : Fin C, R d := by
  rw [Finset.sum_congr rfl (fun c _ => hp c)] at hsum
  have : (∑ c : Fin C, -R c / lam) = (-∑ c : Fin C, R c) / lam := by
    rw [← Finset.sum_div, ← Finset.sum_neg_distrib]
  rw [this, div_eq_one_iff_eq hlam] at hsum
  linarith

-- eq:em-mstep-pi (lines 859-868): π_c⁺ = R_c / Σ_d R_d.
/-- The mixture-weight update. -/
def ch08b_mstep_pi {C : ℕ} (R : Fin C → ℝ) (c : Fin C) : ℝ := R c / ∑ d : Fin C, R d

/-- The update satisfies the constraint and the stationarity condition with
`λ = -Σ_d R_d`. -/
theorem ch08b_em_mstep_pi {C : ℕ} (R : Fin C → ℝ) (hR : ∀ c, 0 < R c)
    (hne : (Finset.univ : Finset (Fin C)).Nonempty) :
    (∑ c : Fin C, ch08b_mstep_pi R c) = 1 ∧
      ∀ c, R c / ch08b_mstep_pi R c + (-∑ d : Fin C, R d) = 0 := by
  have hT : 0 < ∑ d : Fin C, R d := Finset.sum_pos (fun c _ => hR c) hne
  constructor
  · unfold ch08b_mstep_pi
    rw [← Finset.sum_div]
    exact div_self (ne_of_gt hT)
  · intro c
    have hRc : R c ≠ 0 := ne_of_gt (hR c)
    have hTne : (∑ d : Fin C, R d) ≠ 0 := ne_of_gt hT
    unfold ch08b_mstep_pi
    have hkey : R c / (R c / ∑ d : Fin C, R d) = ∑ d : Fin C, R d := by
      field_simp
    rw [hkey]
    ring

/-- The update is the *global* maximiser of `Q_π` on the simplex. -/
theorem ch08b_em_mstep_pi_max {C : ℕ} (R p : Fin C → ℝ) (hR : ∀ c, 0 < R c)
    (hp : ∀ c, 0 < p c) (hsum : ∑ c : Fin C, p c = 1) :
    ch08b_Qpi R p ≤ ch08b_Qpi R (ch08b_mstep_pi R) :=
  ch08b_gibbs R p hR hp hsum

/-! ### Component means -/

-- noname-9 (lines 877-883): ∂S_c/∂ν_c = -2U_c + 2α V_c + 2ν_c R_c.
theorem ch08b_S_deriv_nu (G P W U V R α ν : ℝ) :
    HasDerivAt (fun t : ℝ => ch08b_S G P W U V R α t)
      (-2 * U + 2 * α * V + 2 * ν * R) ν := by
  have hfun : (fun t : ℝ => ch08b_S G P W U V R α t)
      = fun t : ℝ => (G - 2 * α * P + α ^ 2 * W) + (-2 * U + 2 * α * V) * t + R * t ^ 2 := by
    funext t; unfold ch08b_S; ring
  rw [hfun]
  have h := ch08b_quad_hasDerivAt (G - 2 * α * P + α ^ 2 * W) (-2 * U + 2 * α * V) R ν
  have he : (-2 * U + 2 * α * V) + 2 * R * ν = -2 * U + 2 * α * V + 2 * ν * R := by ring
  rwa [he] at h

-- noname-10 (lines 885-890): ∂Q/∂ν_c = -(1/(2 s_c²)) ∂S_c/∂ν_c.
theorem ch08b_Q_deriv_nu (G P W U V R α ν s2 lp : ℝ) (hs2 : s2 ≠ 0) :
    HasDerivAt (fun t : ℝ => R * lp - (R / 2) * Real.log s2 - ch08b_S G P W U V R α t / (2 * s2))
      (-(1 / (2 * s2)) * (-2 * U + 2 * α * V + 2 * ν * R)) ν := by
  have hS := ch08b_S_deriv_nu G P W U V R α ν
  have h1 : HasDerivAt
      (fun t : ℝ => R * lp - (R / 2) * Real.log s2 - ch08b_S G P W U V R α t / (2 * s2))
      (0 - (-2 * U + 2 * α * V + 2 * ν * R) / (2 * s2)) ν :=
    (hasDerivAt_const ν (R * lp - (R / 2) * Real.log s2)).sub (hS.div_const (2 * s2))
  have he : (0 : ℝ) - (-2 * U + 2 * α * V + 2 * ν * R) / (2 * s2)
      = -(1 / (2 * s2)) * (-2 * U + 2 * α * V + 2 * ν * R) := by field_simp <;> ring
  rwa [he] at h1

-- noname-11 (lines 892-898): the stationarity condition -2U_c + 2α V_c + 2ν_c R_c = 0.
-- eq:em-mstep-nu (lines 900-909): ν_c⁺ = (U_c - α V_c)/R_c.
/-- The component-mean update. -/
def ch08b_mstep_nu (U V R α : ℝ) : ℝ := (U - α * V) / R

theorem ch08b_em_mstep_nu (U V R α ν : ℝ) (hR : R ≠ 0) :
    (-2 * U + 2 * α * V + 2 * ν * R = 0) ↔ ν = ch08b_mstep_nu U V R α := by
  unfold ch08b_mstep_nu
  rw [eq_div_iff hR]
  constructor <;> intro h <;> linarith

/-- The update is the global minimiser of `S_c` in `ν_c` (hence the maximiser
of `Q`), with exact excess `R_c (ν - ν_c⁺)²`. -/
theorem ch08b_em_mstep_nu_min (G P W U V R α ν : ℝ) (hR : R ≠ 0) :
    ch08b_S G P W U V R α ν - ch08b_S G P W U V R α (ch08b_mstep_nu U V R α)
      = R * (ν - ch08b_mstep_nu U V R α) ^ 2 := by
  unfold ch08b_S ch08b_mstep_nu
  field_simp
  ring

/-! ### Component variances -/

-- noname-12 (lines 914-916): τ_c = s_c².
/-- The reparametrisation `τ_c = s_c²`. -/
def ch08b_tau (s : ℝ) : ℝ := s ^ 2

-- noname-13 (lines 918-924): Q_c(τ_c) = -(R_c/2) log τ_c - S_c(α,ν_c)/(2τ_c).
/-- The variance-dependent part of the surrogate. -/
def ch08b_Qtau (R S τ : ℝ) : ℝ := -(R / 2) * Real.log τ - S / (2 * τ)

-- noname-14 (lines 926-934): ∂Q_c/∂τ_c = -R_c/(2τ_c) + S_c/(2τ_c²).
theorem ch08b_Qtau_deriv (R S τ : ℝ) (hτ : τ ≠ 0) :
    HasDerivAt (fun t : ℝ => ch08b_Qtau R S t) (-(R / (2 * τ)) + S / (2 * τ ^ 2)) τ := by
  have hfun : (fun t : ℝ => ch08b_Qtau R S t)
      = fun t : ℝ => (-(R / 2)) * Real.log t + (-(S / 2)) * t⁻¹ := by
    funext t; unfold ch08b_Qtau; ring
  rw [hfun]
  have hl : HasDerivAt Real.log τ⁻¹ τ := Real.hasDerivAt_log hτ
  have hi : HasDerivAt (fun t : ℝ => t⁻¹) (-(τ ^ 2)⁻¹) τ := hasDerivAt_inv hτ
  have h3 : HasDerivAt (fun t : ℝ => (-(R / 2)) * Real.log t + (-(S / 2)) * t⁻¹)
      ((-(R / 2)) * τ⁻¹ + (-(S / 2)) * (-(τ ^ 2)⁻¹)) τ :=
    (hl.const_mul (-(R / 2))).add (hi.const_mul (-(S / 2)))
  have he : (-(R / 2)) * τ⁻¹ + (-(S / 2)) * (-(τ ^ 2)⁻¹) = -(R / (2 * τ)) + S / (2 * τ ^ 2) := by
    field_simp
  rwa [he] at h3

-- noname-15 (lines 936-940): R_c τ_c = S_c(α,ν_c).
theorem ch08b_var_stationary (R S τ : ℝ) (hτ : τ ≠ 0) :
    (-(R / (2 * τ)) + S / (2 * τ ^ 2) = 0) ↔ R * τ = S := by
  rw [← sub_eq_zero (a := R * τ)]
  constructor <;> intro h
  · field_simp at h; linarith
  · field_simp; linarith [h]

-- eq:em-mstep-var-general (lines 942-949): (s_c²)⁺ = S_c(α, ν_c⁺)/R_c.
theorem ch08b_em_mstep_var_general (R S τ : ℝ) (hR : R ≠ 0) (hτ : τ ≠ 0) :
    R * τ = S ↔ τ = S / R := by
  rw [eq_div_iff hR]
  constructor <;> intro h <;> linarith

/-- The stationary point really is the global maximiser of `Q_c(τ)` on `τ > 0`
(when `S > 0`): a genuine maximum, not just a critical point. -/
theorem ch08b_em_mstep_var_max (R S τ : ℝ) (hR : 0 < R) (hS : 0 < S) (hτ : 0 < τ) :
    ch08b_Qtau R S τ ≤ ch08b_Qtau R S (S / R) := by
  have hSR : 0 < S / R := div_pos hS hR
  set t := τ * R / S with ht
  have htpos : 0 < t := by positivity
  have hlog : Real.log t⁻¹ ≤ t⁻¹ - 1 := Real.log_le_sub_one_of_pos (by positivity)
  have hlt : Real.log t⁻¹ = -Real.log t := Real.log_inv t
  have hsplit : Real.log τ = Real.log t + Real.log (S / R) := by
    rw [ht, ← Real.log_mul (by positivity) (by positivity)]
    congr 1
    field_simp
  unfold ch08b_Qtau
  rw [hsplit]
  have hτS : S / (2 * τ) = R / (2 * t) := by
    rw [ht]; field_simp <;> ring
  rw [hτS]
  have hSS : S / (2 * (S / R)) = R / 2 := by field_simp
  rw [hSS]
  have hgoal : R / (2 * t) = (R / 2) * t⁻¹ := by field_simp
  rw [hgoal]
  have h2 : -Real.log t - t⁻¹ + 1 ≤ 0 := by rw [← hlt] at *; linarith
  nlinarith [hR.le, h2]

-- noname-16 (lines 952-960): S_c(α, ν_c⁺) = G_c - 2α P_c + α² W_c - R_c (ν_c⁺)².
theorem ch08b_S_at_nu_plus (G P W U V R α : ℝ) (hR : R ≠ 0) :
    ch08b_S G P W U V R α (ch08b_mstep_nu U V R α)
      = G - 2 * α * P + α ^ 2 * W - R * (ch08b_mstep_nu U V R α) ^ 2 := by
  unfold ch08b_S ch08b_mstep_nu
  field_simp
  ring

-- eq:em-mstep-var (lines 962-974):
-- (s_c²)⁺ = [G_c - 2α P_c + α² W_c - R_c (ν_c⁺)²]/R_c.
/-- The component-variance update. -/
def ch08b_mstep_var (G P W U V R α : ℝ) : ℝ :=
  (G - 2 * α * P + α ^ 2 * W - R * (ch08b_mstep_nu U V R α) ^ 2) / R

theorem ch08b_em_mstep_var (G P W U V R α : ℝ) (hR : R ≠ 0) :
    ch08b_S G P W U V R α (ch08b_mstep_nu U V R α) / R = ch08b_mstep_var G P W U V R α := by
  unfold ch08b_mstep_var
  rw [ch08b_S_at_nu_plus G P W U V R α hR]

-- eq:em-mstep-mixture (lines 977-995): the three mixture updates together.
theorem ch08b_em_mstep_mixture {C : ℕ} (R U V P W G : Fin C → ℝ) (α : ℝ)
    (hR : ∀ c, R c ≠ 0) (c : Fin C) :
    ch08b_mstep_pi R c = R c / ∑ d : Fin C, R d
      ∧ ch08b_mstep_nu (U c) (V c) (R c) α = (U c - α * V c) / R c
      ∧ ch08b_mstep_var (G c) (P c) (W c) (U c) (V c) (R c) α
          = (G c - 2 * α * P c + α ^ 2 * W c
              - R c * (ch08b_mstep_nu (U c) (V c) (R c) α) ^ 2) / R c :=
  ⟨rfl, rfl, rfl⟩

/-! ### Autoregressive coefficient -/

-- noname-17 (lines 1019-1025): ∂S_c/∂α = -2P_c + 2α W_c + 2ν_c V_c.
theorem ch08b_S_deriv_alpha (G P W U V R ν α : ℝ) :
    HasDerivAt (fun t : ℝ => ch08b_S G P W U V R t ν)
      (-2 * P + 2 * α * W + 2 * ν * V) α := by
  have hfun : (fun t : ℝ => ch08b_S G P W U V R t ν)
      = fun t : ℝ => (G - 2 * ν * U + ν ^ 2 * R) + (-2 * P + 2 * ν * V) * t + W * t ^ 2 := by
    funext t; unfold ch08b_S; ring
  rw [hfun]
  have h := ch08b_quad_hasDerivAt (G - 2 * ν * U + ν ^ 2 * R) (-2 * P + 2 * ν * V) W α
  have he : (-2 * P + 2 * ν * V) + 2 * W * α = -2 * P + 2 * α * W + 2 * ν * V := by ring
  rwa [he] at h

-- eq:em-alpha-derivative (lines 1027-1040):
-- ∂Q/∂α = -Σ_c (1/(2 s_c²)) ∂S_c/∂α = Σ_c (P_c - α W_c - ν_c V_c)/s_c².
theorem ch08b_em_alpha_derivative {C : ℕ} (P W V ν s2 : Fin C → ℝ) (α : ℝ)
    (hs2 : ∀ c, s2 c ≠ 0) :
    (-∑ c : Fin C, (1 / (2 * s2 c)) * (-2 * P c + 2 * α * W c + 2 * ν c * V c))
      = ∑ c : Fin C, (P c - α * W c - ν c * V c) / s2 c := by
  rw [← Finset.sum_neg_distrib]
  refine Finset.sum_congr rfl (fun c _ => ?_)
  have hc := hs2 c
  field_simp
  ring

/-- The full derivative statement, as a `HasDerivAt` in `α`. -/
theorem ch08b_em_alpha_hasDerivAt {C : ℕ} (G P W U V R ν s2 : Fin C → ℝ) (α : ℝ)
    (hs2 : ∀ c, s2 c ≠ 0) :
    HasDerivAt
      (fun t : ℝ => -∑ c : Fin C, ch08b_S (G c) (P c) (W c) (U c) (V c) (R c) t (ν c) / (2 * s2 c))
      (∑ c : Fin C, (P c - α * W c - ν c * V c) / s2 c) α := by
  have hc : ∀ c : Fin C, HasDerivAt
      (fun t : ℝ => ch08b_S (G c) (P c) (W c) (U c) (V c) (R c) t (ν c) / (2 * s2 c))
      ((-2 * P c + 2 * α * W c + 2 * ν c * V c) / (2 * s2 c)) α :=
    fun c => (ch08b_S_deriv_alpha (G c) (P c) (W c) (U c) (V c) (R c) (ν c) α).div_const _
  have hsum : HasDerivAt
      (fun t : ℝ => ∑ c : Fin C, ch08b_S (G c) (P c) (W c) (U c) (V c) (R c) t (ν c) / (2 * s2 c))
      (∑ c : Fin C, (-2 * P c + 2 * α * W c + 2 * ν c * V c) / (2 * s2 c)) α :=
    HasDerivAt.fun_sum (fun c _ => hc c)
  have hneg := hsum.neg
  have he : -∑ c : Fin C, (-2 * P c + 2 * α * W c + 2 * ν c * V c) / (2 * s2 c)
      = ∑ c : Fin C, (P c - α * W c - ν c * V c) / s2 c := by
    rw [← Finset.sum_neg_distrib]
    refine Finset.sum_congr rfl (fun c _ => ?_)
    have := hs2 c
    field_simp
    ring
  rwa [he] at hneg

-- noname-18 (lines 1042-1052): Σ_c s_c^{-2}(P_c - ν_c V_c) = α Σ_c s_c^{-2} W_c.
-- eq:em-mstep-alpha (lines 1054-1072):
-- α⁺ = Σ_c s_c^{-2}(P_c - ν_c V_c) / Σ_c s_c^{-2} W_c.
/-- The autoregressive-coefficient update. -/
def ch08b_mstep_alpha {C : ℕ} (P V W ν s2 : Fin C → ℝ) : ℝ :=
  (∑ c : Fin C, (s2 c)⁻¹ * (P c - ν c * V c)) / ∑ c : Fin C, (s2 c)⁻¹ * W c

theorem ch08b_em_mstep_alpha {C : ℕ} (P V W ν s2 : Fin C → ℝ) (α : ℝ)
    (hden : (∑ c : Fin C, (s2 c)⁻¹ * W c) ≠ 0) :
    ((∑ c : Fin C, (s2 c)⁻¹ * (P c - ν c * V c)) = α * ∑ c : Fin C, (s2 c)⁻¹ * W c)
      ↔ α = ch08b_mstep_alpha P V W ν s2 := by
  unfold ch08b_mstep_alpha
  rw [eq_div_iff hden]
  constructor <;> intro h <;> linarith

/-- The stationary point of the `α`-block really is its global maximiser: the
excess is exactly `-(B/2)(α - α⁺)²` with `B = Σ_c W_c/s_c² > 0`. -/
theorem ch08b_em_mstep_alpha_max (A B α αp : ℝ) (hB : B ≠ 0) (hαp : αp = A / B) :
    (A * α - (B / 2) * α ^ 2) - (A * αp - (B / 2) * αp ^ 2) = -(B / 2) * (α - αp) ^ 2 := by
  subst hαp
  field_simp
  ring

-- noname-19 (lines 1078-1090): at C = 1 and ν_1 = 0, α⁺ = P_1/W_1.
theorem ch08b_alpha_single_component (P V W ν s2 : Fin 1 → ℝ) (hν : ν 0 = 0)
    (hs2 : s2 0 ≠ 0) (hW : W 0 ≠ 0) :
    ch08b_mstep_alpha P V W ν s2 = P 0 / W 0 := by
  unfold ch08b_mstep_alpha
  rw [Fin.sum_univ_one, Fin.sum_univ_one, hν, zero_mul, sub_zero]
  field_simp <;> ring

/-- At `C = 1` the responsibility is identically one, so `P_1` and `W_1` are the
plain posterior moments `Σ_i E[a_{i-1}a_i | x]` and `Σ_i E[a_{i-1}²| x]`. -/
theorem ch08b_single_component_moments {Ω : Type*} [Fintype Ω] (b x y : Ω → ℝ) :
    ch08b_P b (fun _ => 1) x y = ∑ ω : Ω, b ω * (x ω * y ω)
      ∧ ch08b_W b (fun _ => 1) x = ∑ ω : Ω, b ω * x ω ^ 2 := by
  constructor
  · unfold ch08b_P
    exact Finset.sum_congr rfl (fun ω _ => by ring)
  · unfold ch08b_W
    exact Finset.sum_congr rfl (fun ω _ => by ring)

-- eq:em-summary (lines 1095-1116): E-step -> sufficient statistics -> M-step.
/-- One complete M-step: sufficient statistics in, new parameters out. -/
def ch08b_em_summary {C : ℕ} (R U V P W G : Fin C → ℝ) (α : ℝ) (s2 : Fin C → ℝ) :
    (Fin C → ℝ) × (Fin C → ℝ) × (Fin C → ℝ) × ℝ :=
  (fun c => ch08b_mstep_pi R c,
   fun c => ch08b_mstep_nu (U c) (V c) (R c) α,
   fun c => ch08b_mstep_var (G c) (P c) (W c) (U c) (V c) (R c) α,
   ch08b_mstep_alpha P V W (fun c => ch08b_mstep_nu (U c) (V c) (R c) α) s2)

/-! ## 8.5 Generalised EM and monotonicity -/

-- eq:em-monotone (lines 1132-1148):
-- L(θ) - L(θ^(k)) = Q(θ|θ^(k)) - Q(θ^(k)|θ^(k)) + KL(p_{θ^(k)}(a,c|x) ‖ p_θ(a,c|x)).
theorem ch08b_em_monotone {Z : Type*} [Fintype Z] (pj0 pj1 : Z → ℝ)
    (h0 : ∀ z, 0 < pj0 z) (h1 : ∀ z, 0 < pj1 z)
    (hZ0 : 0 < ∑ w, pj0 w) (hZ1 : 0 < ∑ w, pj1 w) :
    Real.log (∑ z, pj1 z) - Real.log (∑ z, pj0 z)
      = ((∑ z, (pj0 z / ∑ w, pj0 w) * Real.log (pj1 z))
          - (∑ z, (pj0 z / ∑ w, pj0 w) * Real.log (pj0 z)))
        + ∑ z, (pj0 z / ∑ w, pj0 w) *
            Real.log ((pj0 z / ∑ w, pj0 w) / (pj1 z / ∑ w, pj1 w)) := by
  have hq1 : ∑ z, pj0 z / ∑ w, pj0 w = 1 := by
    rw [← Finset.sum_div]
    exact div_self (ne_of_gt hZ0)
  have hlog : ∀ z : Z, Real.log ((pj0 z / ∑ w, pj0 w) / (pj1 z / ∑ w, pj1 w))
      = (Real.log (pj0 z) - Real.log (∑ w, pj0 w))
        - (Real.log (pj1 z) - Real.log (∑ w, pj1 w)) := by
    intro z
    have a1 : pj0 z / (∑ w, pj0 w) ≠ 0 := ne_of_gt (div_pos (h0 z) hZ0)
    have a2 : pj1 z / (∑ w, pj1 w) ≠ 0 := ne_of_gt (div_pos (h1 z) hZ1)
    rw [Real.log_div a1 a2, Real.log_div (ne_of_gt (h0 z)) (ne_of_gt hZ0),
      Real.log_div (ne_of_gt (h1 z)) (ne_of_gt hZ1)]
  simp only [hlog]
  rw [← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
  have hcollapse : ∀ z : Z,
      ((pj0 z / ∑ w, pj0 w) * Real.log (pj1 z) - (pj0 z / ∑ w, pj0 w) * Real.log (pj0 z))
        + (pj0 z / ∑ w, pj0 w) *
            ((Real.log (pj0 z) - Real.log (∑ w, pj0 w))
              - (Real.log (pj1 z) - Real.log (∑ w, pj1 w)))
      = (pj0 z / ∑ w, pj0 w) * (Real.log (∑ w, pj1 w) - Real.log (∑ w, pj0 w)) := by
    intro z; ring
  rw [Finset.sum_congr rfl (fun z _ => hcollapse z), ← Finset.sum_mul, hq1, one_mul]

-- noname-20 (lines 1150-1154): Q(θ^(k+1)|θ^(k)) ≥ Q(θ^(k)|θ^(k))
-- eq:em-likelihood-monotone (lines 1156-1163): L(θ^(k+1)) ≥ L(θ^(k)).
/-- Generalised-EM monotonicity: merely *increasing* the surrogate suffices,
because the KL term is non-negative. -/
theorem ch08b_em_likelihood_monotone (L0 L1 Q0 Q1 KL : ℝ)
    (hdec : L1 - L0 = (Q1 - Q0) + KL) (hKL : 0 ≤ KL) (hQ : Q0 ≤ Q1) : L0 ≤ L1 := by
  linarith

/-- The concrete version: the hypotheses of `ch08b_em_likelihood_monotone` are
met by the decomposition `ch08b_em_monotone` together with `ch08b_KL_nonneg`. -/
theorem ch08b_em_likelihood_monotone_concrete {Z : Type*} [Fintype Z] (pj0 pj1 : Z → ℝ)
    (h0 : ∀ z, 0 < pj0 z) (h1 : ∀ z, 0 < pj1 z)
    (hZ0 : 0 < ∑ w, pj0 w) (hZ1 : 0 < ∑ w, pj1 w)
    (hQ : (∑ z, (pj0 z / ∑ w, pj0 w) * Real.log (pj0 z))
        ≤ ∑ z, (pj0 z / ∑ w, pj0 w) * Real.log (pj1 z)) :
    Real.log (∑ z, pj0 z) ≤ Real.log (∑ z, pj1 z) := by
  have hdec := ch08b_em_monotone pj0 pj1 h0 h1 hZ0 hZ1
  have hq1 : ∑ z, pj0 z / ∑ w, pj0 w = 1 := by
    rw [← Finset.sum_div]; exact div_self (ne_of_gt hZ0)
  have hp1 : ∑ z, pj1 z / ∑ w, pj1 w = 1 := by
    rw [← Finset.sum_div]; exact div_self (ne_of_gt hZ1)
  have hKL : 0 ≤ ∑ z, (pj0 z / ∑ w, pj0 w) *
      Real.log ((pj0 z / ∑ w, pj0 w) / (pj1 z / ∑ w, pj1 w)) :=
    ch08b_KL_nonneg _ _ (fun z => div_pos (h0 z) hZ0) (fun z => div_pos (h1 z) hZ1) hq1 hp1
  linarith

/-! ## 8.6 Numerical representation: score amplification at small times -/

-- noname-21 (lines 1288-1290): e^{-t}/Δ_t ~ 1/(2t), with Δ_t = 1 - e^{-2t}.
/-- Exact closed form: `e^{-t}/(1 - e^{-2t}) = 1/(2 sinh t)`. -/
theorem ch08b_score_amplification_exact (t : ℝ) (ht : t ≠ 0) :
    Real.exp (-t) / (1 - Real.exp (-(2 * t))) = 1 / (2 * Real.sinh t) := by
  have hexp : Real.exp (-t) * Real.exp t = 1 := by
    rw [← Real.exp_add]; simp
  have hsq : Real.exp (-(2 * t)) = Real.exp (-t) * Real.exp (-t) := by
    rw [← Real.exp_add]; ring_nf
  have key : 1 - Real.exp (-(2 * t)) = Real.exp (-t) * (2 * Real.sinh t) := by
    rw [Real.sinh_eq, hsq]
    field_simp
    nlinarith [hexp]
  rw [key]
  have hs : Real.sinh t ≠ 0 := Real.sinh_ne_zero.mpr ht
  have he : Real.exp (-t) ≠ 0 := ne_of_gt (Real.exp_pos _)
  field_simp

/-- The asymptotic statement `e^{-t}/Δ_t ~ 1/(2t)` as `t → 0`: the ratio of the
two sides tends to one. -/
theorem ch08b_score_amplification_asymptotic :
    Filter.Tendsto (fun t : ℝ => (Real.exp (-t) / (1 - Real.exp (-(2 * t)))) * (2 * t))
      (nhdsWithin 0 {(0 : ℝ)}ᶜ) (nhds 1) := by
  have hslope : Filter.Tendsto (fun t : ℝ => Real.sinh t / t)
      (nhdsWithin 0 {(0 : ℝ)}ᶜ) (nhds 1) := by
    have h := (Real.hasDerivAt_sinh 0).tendsto_slope
    simp only [Real.cosh_zero] at h
    refine h.congr (fun t => ?_)
    simp [slope_def_field, div_eq_iff, Real.sinh_zero, sub_zero]
  have hinv : Filter.Tendsto (fun t : ℝ => t / Real.sinh t)
      (nhdsWithin 0 {(0 : ℝ)}ᶜ) (nhds 1) := by
    have := hslope.inv₀ one_ne_zero
    simpa [inv_div] using this
  refine hinv.congr' ?_
  filter_upwards [self_mem_nhdsWithin] with t ht
  have ht' : t ≠ 0 := ht
  have hs : Real.sinh t ≠ 0 := Real.sinh_ne_zero.mpr ht'
  rw [ch08b_score_amplification_exact t ht']
  field_simp

/-! ## Strengthenings (pass 2)

The chapter derives each M-step update by setting a derivative to zero.  A
stationary point is not by itself a maximiser, and the ECM monotonicity
argument of Section 8.5 needs the conditional maximisation to be genuine.  The
following upgrade the stationarity statements to honest ascent statements.
-/

-- eq:em-responsibility (ch08-em-parameters.tex lines 635-673):
-- the first equality r_{i,c} = π_c N(·)/Σ_d π_d N(·) is Bayes' rule.
/-- Bayes' rule on a finite mixture, as a *characterisation*: any label
distribution proportional to `prior c * lik c` that sums to one is exactly the
normalised joint weight.  This is the content of the first equality of
eq:em-responsibility, as opposed to merely restating the definition. -/
theorem ch08b_bayes_mixture {C : ℕ} (prior lik q : Fin C → ℝ) (κ : ℝ)
    (hq : ∀ c, q c = κ * (prior c * lik c))
    (hsum : ∑ c : Fin C, q c = 1)
    (hden : (∑ d : Fin C, prior d * lik d) ≠ 0) (c : Fin C) :
    q c = prior c * lik c / ∑ d : Fin C, prior d * lik d := by
  have hκ : κ * (∑ d : Fin C, prior d * lik d) = 1 := by
    rw [Finset.mul_sum]
    rw [Finset.sum_congr rfl (fun d _ => hq d)] at hsum
    exact hsum
  rw [hq c, eq_div_iff hden]
  calc κ * (prior c * lik c) * ∑ d : Fin C, prior d * lik d
      = (κ * ∑ d : Fin C, prior d * lik d) * (prior c * lik c) := by ring
    _ = prior c * lik c := by rw [hκ]; ring

/-- The denominator of eq:em-responsibility really is the mixture kernel of
eq:em-mixture-kernel: this is the last equality of the align block. -/
theorem ch08b_em_responsibility_kernel {C : ℕ} (p ν s : Fin C → ℝ) (α a' a : ℝ) :
    (∑ d : Fin C, p d * ch08b_gauss (ν d) (s d) (a' - α * a))
      = ch08b_kernel p ν s α a' a := rfl

/-- Specialised to the mixture kernel: the responsibility is the *unique* label
distribution proportional to `π_c N(a_i - α a_{i-1}; ν_c, s_c²)`. -/
theorem ch08b_em_responsibility_bayes {C : ℕ} (p ν s : Fin C → ℝ) (α a' a : ℝ)
    (q : Fin C → ℝ) (κ : ℝ)
    (hq : ∀ c, q c = κ * (p c * ch08b_gauss (ν c) (s c) (a' - α * a)))
    (hsum : ∑ c : Fin C, q c = 1)
    (hK : ch08b_kernel p ν s α a' a ≠ 0) (c : Fin C) :
    q c = ch08b_em_responsibility p ν s α a' a c := by
  have h := ch08b_bayes_mixture p (fun d => ch08b_gauss (ν d) (s d) (a' - α * a)) q κ hq hsum
    (by simpa [ch08b_kernel] using hK) c
  simpa [ch08b_em_responsibility, ch08b_kernel] using h

-- eq:em-suffstats (lines 705-748), trailing claim at lines 749-750:
-- "For several training sequences the same quantities are simply summed over
-- all edges of all sequences."
/-- Every statistic of the form `Σ_ω F ω` splits over a disjoint union of edge
sets, so all six of eq:em-suffstats are additive across sequences. -/
theorem ch08b_stat_additive {Ω₁ Ω₂ : Type*} [Fintype Ω₁] [Fintype Ω₂]
    (F : Ω₁ ⊕ Ω₂ → ℝ) :
    (∑ ω : Ω₁ ⊕ Ω₂, F ω) = (∑ ω : Ω₁, F (Sum.inl ω)) + ∑ ω : Ω₂, F (Sum.inr ω) := by
  rw [Fintype.sum_sum_type]

-- eq:em-mstep-mixture (lines 977-995): the (ν_c, s_c²) block update is a
-- genuine conditional *maximiser* of the surrogate, not just a critical point.
/-- ECM ascent for the mixture block: replacing `(ν_c, τ_c)` by
`(ν_c⁺, (s_c²)⁺)` never decreases the `c`-th summand of eq:em-Q-suffstats.
Combined with `ch08b_em_mstep_pi_max` for the weights, this is the conditional
maximisation the monotonicity argument of Section 8.5 assumes. -/
theorem ch08b_mixture_block_ascent (G P W U V R α ν τ : ℝ)
    (hR : 0 < R) (hτ : 0 < τ)
    (hSp : 0 < ch08b_S G P W U V R α (ch08b_mstep_nu U V R α)) :
    ch08b_Qtau R (ch08b_S G P W U V R α ν) τ
      ≤ ch08b_Qtau R (ch08b_S G P W U V R α (ch08b_mstep_nu U V R α))
          (ch08b_mstep_var G P W U V R α) := by
  have hex : ch08b_S G P W U V R α ν - ch08b_S G P W U V R α (ch08b_mstep_nu U V R α)
      = R * (ν - ch08b_mstep_nu U V R α) ^ 2 :=
    ch08b_em_mstep_nu_min G P W U V R α ν (ne_of_gt hR)
  have hnum : 0 ≤ ch08b_S G P W U V R α ν - ch08b_S G P W U V R α (ch08b_mstep_nu U V R α) := by
    rw [hex]
    exact mul_nonneg hR.le (sq_nonneg _)
  have hdiv : 0 ≤ (ch08b_S G P W U V R α ν - ch08b_S G P W U V R α (ch08b_mstep_nu U V R α))
      / (2 * τ) := div_nonneg hnum (by linarith)
  rw [sub_div] at hdiv
  have step1 : ch08b_Qtau R (ch08b_S G P W U V R α ν) τ
      ≤ ch08b_Qtau R (ch08b_S G P W U V R α (ch08b_mstep_nu U V R α)) τ := by
    unfold ch08b_Qtau
    linarith
  have step2 : ch08b_Qtau R (ch08b_S G P W U V R α (ch08b_mstep_nu U V R α)) τ
      ≤ ch08b_Qtau R (ch08b_S G P W U V R α (ch08b_mstep_nu U V R α))
          (ch08b_S G P W U V R α (ch08b_mstep_nu U V R α) / R) :=
    ch08b_em_mstep_var_max R _ τ hR hSp hτ
  have hvar : ch08b_mstep_var G P W U V R α
      = ch08b_S G P W U V R α (ch08b_mstep_nu U V R α) / R :=
    (ch08b_em_mstep_var G P W U V R α (ne_of_gt hR)).symm
  rw [hvar]
  linarith

-- eq:em-alpha-derivative / eq:em-mstep-alpha (lines 1027-1072):
-- the α-block of the surrogate is an exact concave quadratic in α.
/-- The α-dependent part of eq:em-Q-suffstats is the quadratic
`A α - (B/2) α² - K` with `A = Σ_c s_c^{-2}(P_c - ν_c V_c)` and
`B = Σ_c s_c^{-2} W_c`. -/
theorem ch08b_alpha_block_quadratic {C : ℕ} (G P W U V R ν s2 : Fin C → ℝ) (t : ℝ)
    (hs2 : ∀ c, s2 c ≠ 0) :
    (-∑ c : Fin C, ch08b_S (G c) (P c) (W c) (U c) (V c) (R c) t (ν c) / (2 * s2 c))
      = ((∑ c : Fin C, (s2 c)⁻¹ * (P c - ν c * V c)) * t
          - ((∑ c : Fin C, (s2 c)⁻¹ * W c) / 2) * t ^ 2)
        - ∑ c : Fin C, (G c - 2 * ν c * U c + ν c ^ 2 * R c) / (2 * s2 c) := by
  have hterm : ∀ c : Fin C,
      ch08b_S (G c) (P c) (W c) (U c) (V c) (R c) t (ν c) / (2 * s2 c)
        = (G c - 2 * ν c * U c + ν c ^ 2 * R c) / (2 * s2 c)
          - ((s2 c)⁻¹ * (P c - ν c * V c)) * t
          + ((s2 c)⁻¹ * W c / 2) * t ^ 2 := by
    intro c
    have h := hs2 c
    unfold ch08b_S
    field_simp
    ring
  have hsum : (∑ c : Fin C, ch08b_S (G c) (P c) (W c) (U c) (V c) (R c) t (ν c) / (2 * s2 c))
      = (∑ c : Fin C, (G c - 2 * ν c * U c + ν c ^ 2 * R c) / (2 * s2 c))
        - (∑ c : Fin C, (s2 c)⁻¹ * (P c - ν c * V c)) * t
        + ((∑ c : Fin C, (s2 c)⁻¹ * W c) / 2) * t ^ 2 := by
    rw [Finset.sum_congr rfl (fun c _ => hterm c), Finset.sum_add_distrib,
      Finset.sum_sub_distrib, ← Finset.sum_mul, ← Finset.sum_mul, ← Finset.sum_div]
  rw [hsum]
  ring

/-- ECM ascent for the `α` block: `α⁺` of eq:em-mstep-alpha is the global
conditional maximiser of the surrogate in `α`, not merely a stationary point. -/
theorem ch08b_em_alpha_block_ascent {C : ℕ} (G P W U V R ν s2 : Fin C → ℝ) (α : ℝ)
    (hs2 : ∀ c, s2 c ≠ 0) (hB : 0 < ∑ c : Fin C, (s2 c)⁻¹ * W c) :
    (-∑ c : Fin C, ch08b_S (G c) (P c) (W c) (U c) (V c) (R c) α (ν c) / (2 * s2 c))
      ≤ -∑ c : Fin C, ch08b_S (G c) (P c) (W c) (U c) (V c) (R c)
            (ch08b_mstep_alpha P V W ν s2) (ν c) / (2 * s2 c) := by
  rw [ch08b_alpha_block_quadratic G P W U V R ν s2 α hs2,
    ch08b_alpha_block_quadratic G P W U V R ν s2 (ch08b_mstep_alpha P V W ν s2) hs2]
  have hαp : ch08b_mstep_alpha P V W ν s2
      = (∑ c : Fin C, (s2 c)⁻¹ * (P c - ν c * V c)) / ∑ c : Fin C, (s2 c)⁻¹ * W c := rfl
  have hmax := ch08b_em_mstep_alpha_max (∑ c : Fin C, (s2 c)⁻¹ * (P c - ν c * V c))
    (∑ c : Fin C, (s2 c)⁻¹ * W c) α (ch08b_mstep_alpha P V W ν s2) (ne_of_gt hB) hαp
  have hneg : -((∑ c : Fin C, (s2 c)⁻¹ * W c) / 2)
      * (α - ch08b_mstep_alpha P V W ν s2) ^ 2 ≤ 0 := by
    have h0 : 0 ≤ ((∑ c : Fin C, (s2 c)⁻¹ * W c) / 2)
        * (α - ch08b_mstep_alpha P V W ν s2) ^ 2 :=
      mul_nonneg (by linarith) (sq_nonneg _)
    linarith
  linarith

/-! ## Generalisations (pass 3)

### The chain factor graph and its belief-propagation messages

`ch08b_em_pairwise` above records only the *distributivity* step of the
forward--backward argument: the left/right split of the summand is handed to it
as a hypothesis, so it says nothing about chains.  The material in this section
removes that hypothesis.  The chain weight is defined as a plain product over
the chain, with no edge singled out; the forward and backward messages are
*defined* by the BP recursions of Section 8.2; and `ch08b_em_pairwise_chain`
derives the five-factor form of eq:em-pairwise at an arbitrary edge of a chain
of arbitrary length, over an arbitrary finite state space, with arbitrary
site potentials and an arbitrary transition kernel.
-/

/-- The weight contributed by the sites strictly to the right of site `j` (whose
value is `a`) when they take the values listed in `v`, in order: each further
site contributes the transition factor `K(next | prev)` and its own site
potential. -/
def ch08b_run {S : Type*} (φ : ℕ → S → ℝ) (K : S → S → ℝ) : ℕ → S → List S → ℝ
  | _, _, [] => 1
  | j, a, b :: t => K b a * φ (j + 1) b * ch08b_run φ K (j + 1) b t

/-- The weight contributed by the sites strictly to the left of site `u.length`
(whose value is `a`) when they take the values listed in `u` **from the edge
outwards**, i.e. `u = (a_{m-1}, a_{m-2}, …, a_0)`.  The head of `u = b :: t`
therefore sits at site `t.length`, which keeps every index absolute and avoids
truncated subtraction. -/
def ch08b_left {S : Type*} (φ : ℕ → S → ℝ) (K : S → S → ℝ) : S → List S → ℝ
  | _, [] => 1
  | a, b :: t => K a b * φ t.length b * ch08b_left φ K b t

/-- The unnormalised weight of a whole chain configuration `(a_0, …, a_L)`,
`φ_0(a_0) ∏_{j} K(a_{j+1} | a_j) φ_{j+1}(a_{j+1})`.  Every site potential and
every transition factor occurs exactly once, and no edge is singled out: this
is the factor-graph model of Section 8.2, not a pre-factorised form. -/
def ch08b_chainW {S : Type*} (φ : ℕ → S → ℝ) (K : S → S → ℝ) : List S → ℝ
  | [] => 1
  | a :: t => φ 0 a * ch08b_run φ K 0 a t

/-- **On a chain the joint weight splits at every edge.**  Writing a
configuration as `u` (the sites left of the edge, listed outwards), the edge
pair `(a, a')`, and `v` (the sites right of the edge), the chain weight is the
product of a factor depending only on `(u, a)`, the two site potentials of the
edge, the transition factor, and a factor depending only on `(a', v)`.  This is
the step that `ch08b_em_pairwise` assumed. -/
theorem ch08b_chain_split {S : Type*} (φ : ℕ → S → ℝ) (K : S → S → ℝ) :
    ∀ (u : List S) (a a' : S) (v : List S),
      ch08b_chainW φ K (u.reverse ++ a :: a' :: v)
        = ch08b_left φ K a u * φ u.length a * K a' a * φ (u.length + 1) a'
            * ch08b_run φ K (u.length + 1) a' v := by
  intro u
  induction u with
  | nil =>
      intro a a' v
      simp only [List.reverse_nil, List.nil_append, List.length_nil, ch08b_chainW,
        ch08b_left, ch08b_run]
      ring
  | cons b t ih =>
      intro a a' v
      have hlist : (b :: t).reverse ++ a :: a' :: v = t.reverse ++ b :: a :: (a' :: v) := by
        simp
      rw [hlist, ih b a (a' :: v)]
      simp only [ch08b_left, ch08b_run, List.length_cons]
      ring

/-- The forward BP message `λ_m`, *defined* by the forward recursion of
Section 8.2. -/
def ch08b_lam {S : Type*} [Fintype S] (φ : ℕ → S → ℝ) (K : S → S → ℝ) : ℕ → S → ℝ
  | 0, _ => 1
  | m + 1, a => ∑ b : S, ch08b_lam φ K m b * φ m b * K a b

/-- The backward BP message at site `j` over the `n` sites to its right,
*defined* by the backward recursion of Section 8.2. -/
def ch08b_rho {S : Type*} [Fintype S] (φ : ℕ → S → ℝ) (K : S → S → ℝ) : ℕ → ℕ → S → ℝ
  | 0, _, _ => 1
  | n + 1, j, a => ∑ b : S, K b a * φ (j + 1) b * ch08b_rho φ K n (j + 1) b

/-- The forward message really is the exhaustive sum of the left-segment weight
over all configurations of the `m` sites to the left of the edge. -/
theorem ch08b_lam_eq_sum {S : Type*} [Fintype S] (φ : ℕ → S → ℝ) (K : S → S → ℝ) :
    ∀ (m : ℕ) (a : S),
      ch08b_lam φ K m a = ∑ u : Fin m → S, ch08b_left φ K a (List.ofFn u) := by
  intro m
  induction m with
  | zero => intro a; simp [ch08b_lam, ch08b_left]
  | succ m ih =>
      intro a
      have hsplit : (∑ u : Fin (m + 1) → S, ch08b_left φ K a (List.ofFn u))
          = ∑ b : S, ∑ w : Fin m → S, ch08b_left φ K a (List.ofFn (Fin.cons b w)) := by
        rw [← Equiv.sum_comp (Fin.consEquiv (fun _ : Fin (m + 1) => S))
          (fun u => ch08b_left φ K a (List.ofFn u)), Fintype.sum_prod_type]
        simp [Fin.consEquiv_apply]
      rw [ch08b_lam, hsplit]
      refine Finset.sum_congr rfl (fun b _ => ?_)
      have hterm : ∀ w : Fin m → S, ch08b_left φ K a (List.ofFn (Fin.cons b w))
          = (K a b * φ m b) * ch08b_left φ K b (List.ofFn w) := by
        intro w
        simp [ch08b_left, List.ofFn_succ]
      rw [Finset.sum_congr rfl (fun w _ => hterm w), ← Finset.mul_sum, ← ih b]
      ring

/-- The backward message really is the exhaustive sum of the right-segment
weight over all configurations of the `n` sites to the right of the edge. -/
theorem ch08b_rho_eq_sum {S : Type*} [Fintype S] (φ : ℕ → S → ℝ) (K : S → S → ℝ) :
    ∀ (n j : ℕ) (a : S),
      ch08b_rho φ K n j a = ∑ v : Fin n → S, ch08b_run φ K j a (List.ofFn v) := by
  intro n
  induction n with
  | zero => intro j a; simp [ch08b_rho, ch08b_run]
  | succ n ih =>
      intro j a
      have hsplit : (∑ v : Fin (n + 1) → S, ch08b_run φ K j a (List.ofFn v))
          = ∑ b : S, ∑ w : Fin n → S, ch08b_run φ K j a (List.ofFn (Fin.cons b w)) := by
        rw [← Equiv.sum_comp (Fin.consEquiv (fun _ : Fin (n + 1) => S))
          (fun v => ch08b_run φ K j a (List.ofFn v)), Fintype.sum_prod_type]
        simp [Fin.consEquiv_apply]
      rw [ch08b_rho, hsplit]
      refine Finset.sum_congr rfl (fun b _ => ?_)
      have hterm : ∀ w : Fin n → S, ch08b_run φ K j a (List.ofFn (Fin.cons b w))
          = (K b a * φ (j + 1) b) * ch08b_run φ K (j + 1) b (List.ofFn w) := by
        intro w
        simp [ch08b_run, List.ofFn_succ]
      rw [Finset.sum_congr rfl (fun w _ => hterm w), ← Finset.mul_sum, ← ih (j + 1) b]

-- eq:em-pairwise (ch08-em-parameters.tex lines 614-623), in general form.
/-- **eq:em-pairwise, for an arbitrary chain.**  Marginalising the chain weight
over every variable except the two endpoints of the edge `(m, m+1)` gives
exactly the five-factor product
`λ_{m}(a) φ_{m}(a) K(a' | a) φ_{m+1}(a') ρ_{m+1}(a')`,
with `λ` and `ρ` the messages produced by the forward and backward BP
recursions.  Nothing about the summand is assumed: the split is derived from
the chain structure by `ch08b_chain_split`, and the identification of the two
marginal sums with the BP messages is `ch08b_lam_eq_sum` / `ch08b_rho_eq_sum`. -/
theorem ch08b_em_pairwise_chain {S : Type*} [Fintype S] (φ : ℕ → S → ℝ) (K : S → S → ℝ)
    (m n : ℕ) (a a' : S) :
    (∑ u : Fin m → S, ∑ v : Fin n → S,
        ch08b_chainW φ K ((List.ofFn u).reverse ++ a :: a' :: List.ofFn v))
      = ch08b_lam φ K m a * φ m a * K a' a * φ (m + 1) a' * ch08b_rho φ K n (m + 1) a' := by
  have hsplit : ∀ (u : Fin m → S) (v : Fin n → S),
      ch08b_chainW φ K ((List.ofFn u).reverse ++ a :: a' :: List.ofFn v)
        = (ch08b_left φ K a (List.ofFn u) * φ m a * K a' a * φ (m + 1) a')
            * ch08b_run φ K (m + 1) a' (List.ofFn v) := by
    intro u v
    rw [ch08b_chain_split φ K (List.ofFn u) a a' (List.ofFn v), List.length_ofFn]
  have hinner : ∀ u : Fin m → S,
      (∑ v : Fin n → S, ch08b_chainW φ K ((List.ofFn u).reverse ++ a :: a' :: List.ofFn v))
        = (ch08b_left φ K a (List.ofFn u) * φ m a * K a' a * φ (m + 1) a')
            * ∑ v : Fin n → S, ch08b_run φ K (m + 1) a' (List.ofFn v) := by
    intro u
    rw [Finset.sum_congr rfl (fun v _ => hsplit u v), ← Finset.mul_sum]
  rw [Finset.sum_congr rfl (fun u _ => hinner u), ← Finset.sum_mul,
    ch08b_lam_eq_sum φ K m a, ch08b_rho_eq_sum φ K n (m + 1) a']
  simp only [Finset.sum_mul]

/-- eq:em-pairwise-def and eq:em-pairwise together: the *normalised* five-factor
message product of `ch08b_em_pairwise_def` is exactly the pairwise posterior
`p_{θ}(a_{m}, a_{m+1} | x)`, i.e. the chain weight summed over every other
variable and divided by the total mass.  This is the link the pass-1 file was
missing between the definition and the message product. -/
theorem ch08b_em_pairwise_posterior {S : Type*} [Fintype S] (φ : ℕ → S → ℝ)
    (K : S → S → ℝ) (m n : ℕ) (a a' : S) :
    (∑ u : Fin m → S, ∑ v : Fin n → S,
        ch08b_chainW φ K ((List.ofFn u).reverse ++ a :: a' :: List.ofFn v))
      / (∑ p : S, ∑ q : S, ∑ u : Fin m → S, ∑ v : Fin n → S,
          ch08b_chainW φ K ((List.ofFn u).reverse ++ p :: q :: List.ofFn v))
      = ch08b_em_pairwise_def (ch08b_lam φ K m) (φ m) K (φ (m + 1))
          (ch08b_rho φ K n (m + 1)) a a' := by
  unfold ch08b_em_pairwise_def
  rw [ch08b_em_pairwise_chain φ K m n a a']
  congr 1
  exact Finset.sum_congr rfl
    (fun p _ => Finset.sum_congr rfl (fun q _ => ch08b_em_pairwise_chain φ K m n p q))


/-! ### Assembling the ECM ascent

`ch08b_em_mstep_pi_max`, `ch08b_mixture_block_ascent` and
`ch08b_em_alpha_block_ascent` each bound a *different* expression, and nothing
above puts them back together, so the antecedent of the chapter's sufficiency
claim (lines 1150-1154) still had to be assumed.  What follows defines the
surrogate itself -- the right-hand side of `ch08b_em_Q_suffstats` -- splits it
into exactly those pieces, and composes the block inequalities in the order ECM
performs them: the mixture block at the current `α`, then `α` at the *updated*
means and variances.  The conclusion is
`Q(θ^{(k+1)} | θ^{(k)}) ≥ Q(θ^{(k)} | θ^{(k)})` for the chapter's own updates,
so `hQ` becomes a theorem rather than a hypothesis.
-/

/-- The surrogate of eq:em-Q-suffstats as an explicit function of the
parameters `(π, ν, s², α)`, at fixed E-step sufficient statistics.  This is
literally the right-hand side of `ch08b_em_Q_suffstats`. -/
def ch08b_Qstats {C : ℕ} (R U V P W G p ν s2 : Fin C → ℝ) (α : ℝ) : ℝ :=
  ∑ c : Fin C, (R c * Real.log (p c) - (R c / 2) * Real.log (s2 c)
    - ch08b_S (G c) (P c) (W c) (U c) (V c) (R c) α (ν c) / (2 * s2 c))

/-- Splitting the surrogate into the weight part `Q_π` and the per-component
variance parts `Q_c(τ_c)`: exactly the two objects the mixture-block lemmas
bound. -/
theorem ch08b_Qstats_split_mixture {C : ℕ} (R U V P W G p ν s2 : Fin C → ℝ) (α : ℝ) :
    ch08b_Qstats R U V P W G p ν s2 α
      = ch08b_Qpi R p
        + ∑ c : Fin C, ch08b_Qtau (R c)
            (ch08b_S (G c) (P c) (W c) (U c) (V c) (R c) α (ν c)) (s2 c) := by
  unfold ch08b_Qstats ch08b_Qpi ch08b_Qtau
  rw [← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl (fun c _ => by ring)

/-- Splitting the surrogate into its `α`-independent part and the `α`-block:
exactly the object `ch08b_em_alpha_block_ascent` bounds. -/
theorem ch08b_Qstats_split_alpha {C : ℕ} (R U V P W G p ν s2 : Fin C → ℝ) (α : ℝ) :
    ch08b_Qstats R U V P W G p ν s2 α
      = (∑ c : Fin C, (R c * Real.log (p c) - (R c / 2) * Real.log (s2 c)))
        - ∑ c : Fin C, ch08b_S (G c) (P c) (W c) (U c) (V c) (R c) α (ν c) / (2 * s2 c) := by
  unfold ch08b_Qstats
  rw [← Finset.sum_sub_distrib]

/-- **ECM step 1**: with `α` held fixed, replacing `(π, ν, s²)` by the closed
forms of eq:em-mstep-mixture never decreases the surrogate. -/
theorem ch08b_ecm_mixture_step {C : ℕ} (R U V P W G p ν s2 : Fin C → ℝ) (α : ℝ)
    (hR : ∀ c, 0 < R c) (hp : ∀ c, 0 < p c) (hpsum : ∑ c : Fin C, p c = 1)
    (hs2 : ∀ c, 0 < s2 c)
    (hS : ∀ c, 0 < ch08b_S (G c) (P c) (W c) (U c) (V c) (R c) α
            (ch08b_mstep_nu (U c) (V c) (R c) α)) :
    ch08b_Qstats R U V P W G p ν s2 α
      ≤ ch08b_Qstats R U V P W G (ch08b_mstep_pi R)
          (fun c => ch08b_mstep_nu (U c) (V c) (R c) α)
          (fun c => ch08b_mstep_var (G c) (P c) (W c) (U c) (V c) (R c) α) α := by
  rw [ch08b_Qstats_split_mixture, ch08b_Qstats_split_mixture]
  have h1 : ch08b_Qpi R p ≤ ch08b_Qpi R (ch08b_mstep_pi R) :=
    ch08b_em_mstep_pi_max R p hR hp hpsum
  have h2 : ∀ c : Fin C,
      ch08b_Qtau (R c) (ch08b_S (G c) (P c) (W c) (U c) (V c) (R c) α (ν c)) (s2 c)
        ≤ ch08b_Qtau (R c)
            (ch08b_S (G c) (P c) (W c) (U c) (V c) (R c) α
              (ch08b_mstep_nu (U c) (V c) (R c) α))
            (ch08b_mstep_var (G c) (P c) (W c) (U c) (V c) (R c) α) :=
    fun c => ch08b_mixture_block_ascent (G c) (P c) (W c) (U c) (V c) (R c) α (ν c) (s2 c)
      (hR c) (hs2 c) (hS c)
  have h3 : (∑ c : Fin C,
        ch08b_Qtau (R c) (ch08b_S (G c) (P c) (W c) (U c) (V c) (R c) α (ν c)) (s2 c))
      ≤ ∑ c : Fin C, ch08b_Qtau (R c)
          (ch08b_S (G c) (P c) (W c) (U c) (V c) (R c) α
            (ch08b_mstep_nu (U c) (V c) (R c) α))
          (ch08b_mstep_var (G c) (P c) (W c) (U c) (V c) (R c) α) :=
    Finset.sum_le_sum (fun c _ => h2 c)
  exact add_le_add h1 h3

/-- **ECM step 2**: with `(π, ν, s²)` held fixed, replacing `α` by the closed
form of eq:em-mstep-alpha never decreases the surrogate. -/
theorem ch08b_ecm_alpha_step {C : ℕ} (R U V P W G p ν s2 : Fin C → ℝ) (α : ℝ)
    (hs2 : ∀ c, s2 c ≠ 0) (hB : 0 < ∑ c : Fin C, (s2 c)⁻¹ * W c) :
    ch08b_Qstats R U V P W G p ν s2 α
      ≤ ch08b_Qstats R U V P W G p ν s2 (ch08b_mstep_alpha P V W ν s2) := by
  rw [ch08b_Qstats_split_alpha, ch08b_Qstats_split_alpha]
  have h := ch08b_em_alpha_block_ascent G P W U V R ν s2 α hs2 hB
  have h' : (∑ c : Fin C, ch08b_S (G c) (P c) (W c) (U c) (V c) (R c)
        (ch08b_mstep_alpha P V W ν s2) (ν c) / (2 * s2 c))
      ≤ ∑ c : Fin C, ch08b_S (G c) (P c) (W c) (U c) (V c) (R c) α (ν c) / (2 * s2 c) := by
    linarith
  exact sub_le_sub_left h' _

-- noname-20 (ch08-em-parameters.tex lines 1150-1154):
-- Q(θ^{(k+1)} | θ^{(k)}) ≥ Q(θ^{(k)} | θ^{(k)}).
/-- **The antecedent of eq:em-likelihood-monotone, discharged.**  One full ECM
sweep -- the mixture block of eq:em-mstep-mixture at the current `α`, then `α⁺`
of eq:em-mstep-alpha evaluated at the *updated* means and variances -- never
decreases the surrogate of eq:em-Q-suffstats.  The hypotheses are the
non-degeneracy conditions the chapter itself flags: positive posterior counts
`R_c` (rmk:em-degeneracy), a strictly positive current weight vector on the
simplex, positive current variances, a non-vanishing posterior residual, and a
non-vanishing parent second moment. -/
theorem ch08b_em_ecm_ascent {C : ℕ} (R U V P W G p ν s2 : Fin C → ℝ) (α : ℝ)
    (hR : ∀ c, 0 < R c) (hp : ∀ c, 0 < p c) (hpsum : ∑ c : Fin C, p c = 1)
    (hs2 : ∀ c, 0 < s2 c)
    (hS : ∀ c, 0 < ch08b_S (G c) (P c) (W c) (U c) (V c) (R c) α
            (ch08b_mstep_nu (U c) (V c) (R c) α))
    (hW0 : ∀ c, 0 ≤ W c) (hWpos : ∃ c, 0 < W c) :
    ch08b_Qstats R U V P W G p ν s2 α
      ≤ ch08b_Qstats R U V P W G (ch08b_mstep_pi R)
          (fun c => ch08b_mstep_nu (U c) (V c) (R c) α)
          (fun c => ch08b_mstep_var (G c) (P c) (W c) (U c) (V c) (R c) α)
          (ch08b_mstep_alpha P V W (fun c => ch08b_mstep_nu (U c) (V c) (R c) α)
            (fun c => ch08b_mstep_var (G c) (P c) (W c) (U c) (V c) (R c) α)) := by
  have hvar : ∀ c : Fin C,
      ch08b_mstep_var (G c) (P c) (W c) (U c) (V c) (R c) α
        = ch08b_S (G c) (P c) (W c) (U c) (V c) (R c) α
            (ch08b_mstep_nu (U c) (V c) (R c) α) / R c :=
    fun c => (ch08b_em_mstep_var (G c) (P c) (W c) (U c) (V c) (R c) α (ne_of_gt (hR c))).symm
  have hs2p : ∀ c : Fin C, 0 < ch08b_mstep_var (G c) (P c) (W c) (U c) (V c) (R c) α := by
    intro c
    rw [hvar c]
    exact div_pos (hS c) (hR c)
  have step1 := ch08b_ecm_mixture_step R U V P W G p ν s2 α hR hp hpsum hs2 hS
  have hB : 0 < ∑ c : Fin C,
      (ch08b_mstep_var (G c) (P c) (W c) (U c) (V c) (R c) α)⁻¹ * W c := by
    obtain ⟨c₀, hc₀⟩ := hWpos
    refine Finset.sum_pos' (fun c _ => ?_) ⟨c₀, Finset.mem_univ c₀, ?_⟩
    · exact mul_nonneg (le_of_lt (inv_pos.mpr (hs2p c))) (hW0 c)
    · exact mul_pos (inv_pos.mpr (hs2p c₀)) hc₀
  have step2 := ch08b_ecm_alpha_step R U V P W G (ch08b_mstep_pi R)
    (fun c => ch08b_mstep_nu (U c) (V c) (R c) α)
    (fun c => ch08b_mstep_var (G c) (P c) (W c) (U c) (V c) (R c) α) α
    (fun c => ne_of_gt (hs2p c)) hB
  exact le_trans step1 step2

/-- The same statement written on the chapter's own surrogate rather than on
its closed form: the posterior-weighted complete-data log-likelihood sum of
eq:em-Q -- up to the `θ`-independent `-½ log 2π` per node -- does not decrease
across one ECM sweep, with the sufficient statistics computed from the E-step
belief `b` and responsibilities `r` exactly as in eq:em-suffstats. -/
theorem ch08b_em_Q_increase {Ω : Type*} [Fintype Ω] {C : ℕ}
    (b x y : Ω → ℝ) (r : Fin C → Ω → ℝ) (p ν s2 p' ν' s2' : Fin C → ℝ) (α α' : ℝ)
    (hb : ∀ ω, 0 ≤ b ω) (hr : ∀ c ω, 0 ≤ r c ω)
    (hR : ∀ c, 0 < ch08b_R b (r c)) (hp : ∀ c, 0 < p c) (hpsum : ∑ c : Fin C, p c = 1)
    (hs2 : ∀ c, 0 < s2 c)
    (hS : ∀ c, 0 < ch08b_S (ch08b_G b (r c) y) (ch08b_P b (r c) x y) (ch08b_W b (r c) x)
            (ch08b_U b (r c) y) (ch08b_V b (r c) x) (ch08b_R b (r c)) α
            (ch08b_mstep_nu (ch08b_U b (r c) y) (ch08b_V b (r c) x) (ch08b_R b (r c)) α))
    (hWpos : ∃ c, 0 < ch08b_W b (r c) x)
    (hp' : p' = ch08b_mstep_pi (fun d => ch08b_R b (r d)))
    (hν' : ν' = fun c => ch08b_mstep_nu (ch08b_U b (r c) y) (ch08b_V b (r c) x)
      (ch08b_R b (r c)) α)
    (hs2' : s2' = fun c => ch08b_mstep_var (ch08b_G b (r c) y) (ch08b_P b (r c) x y)
      (ch08b_W b (r c) x) (ch08b_U b (r c) y) (ch08b_V b (r c) x) (ch08b_R b (r c)) α)
    (hα' : α' = ch08b_mstep_alpha (fun d => ch08b_P b (r d) x y) (fun d => ch08b_V b (r d) x)
      (fun d => ch08b_W b (r d) x)
      (fun d => ch08b_mstep_nu (ch08b_U b (r d) y) (ch08b_V b (r d) x) (ch08b_R b (r d)) α)
      (fun d => ch08b_mstep_var (ch08b_G b (r d) y) (ch08b_P b (r d) x y) (ch08b_W b (r d) x)
        (ch08b_U b (r d) y) (ch08b_V b (r d) x) (ch08b_R b (r d)) α)) :
    (∑ ω : Ω, ∑ c : Fin C, b ω * r c ω *
        (Real.log (p c) - (1 / 2) * Real.log (s2 c)
          - (y ω - α * x ω - ν c) ^ 2 / (2 * s2 c)))
      ≤ ∑ ω : Ω, ∑ c : Fin C, b ω * r c ω *
          (Real.log (p' c) - (1 / 2) * Real.log (s2' c)
            - (y ω - α' * x ω - ν' c) ^ 2 / (2 * s2' c)) := by
  have hvar : ∀ c : Fin C,
      ch08b_mstep_var (ch08b_G b (r c) y) (ch08b_P b (r c) x y) (ch08b_W b (r c) x)
          (ch08b_U b (r c) y) (ch08b_V b (r c) x) (ch08b_R b (r c)) α
        = ch08b_S (ch08b_G b (r c) y) (ch08b_P b (r c) x y) (ch08b_W b (r c) x)
            (ch08b_U b (r c) y) (ch08b_V b (r c) x) (ch08b_R b (r c)) α
            (ch08b_mstep_nu (ch08b_U b (r c) y) (ch08b_V b (r c) x) (ch08b_R b (r c)) α)
          / ch08b_R b (r c) :=
    fun c => (ch08b_em_mstep_var _ _ _ _ _ _ α (ne_of_gt (hR c))).symm
  have hs2'pos : ∀ c : Fin C, 0 < s2' c := by
    intro c
    have hc : s2' c = ch08b_mstep_var (ch08b_G b (r c) y) (ch08b_P b (r c) x y)
        (ch08b_W b (r c) x) (ch08b_U b (r c) y) (ch08b_V b (r c) x) (ch08b_R b (r c)) α :=
      congrFun hs2' c
    rw [hc, hvar c]
    exact div_pos (hS c) (hR c)
  have hW0 : ∀ c : Fin C, 0 ≤ ch08b_W b (r c) x := by
    intro c
    unfold ch08b_W
    exact Finset.sum_nonneg
      (fun ω _ => mul_nonneg (mul_nonneg (hb ω) (hr c ω)) (sq_nonneg (x ω)))
  rw [ch08b_em_Q_suffstats b x y r p ν s2 α (fun c => ne_of_gt (hs2 c)),
    ch08b_em_Q_suffstats b x y r p' ν' s2' α' (fun c => ne_of_gt (hs2'pos c))]
  subst hp' hν' hs2' hα'
  exact ch08b_em_ecm_ascent (fun c => ch08b_R b (r c)) (fun c => ch08b_U b (r c) y)
    (fun c => ch08b_V b (r c) x) (fun c => ch08b_P b (r c) x y) (fun c => ch08b_W b (r c) x)
    (fun c => ch08b_G b (r c) y) p ν s2 α hR hp hpsum hs2 hS hW0 hWpos

-- eq:em-likelihood-monotone (lines 1156-1163), with its antecedent supplied.
/-- Likelihood monotonicity for the chapter's own ECM sweep, with *no* surrogate
hypothesis: the remaining inputs are the decomposition eq:em-monotone (proved
abstractly as `ch08b_em_monotone`) and non-negativity of its KL term (proved as
`ch08b_KL_nonneg`).  The surrogate increase that `ch08b_em_likelihood_monotone`
takes as the hypothesis `hQ` is here supplied by `ch08b_em_ecm_ascent`. -/
theorem ch08b_em_likelihood_monotone_ecm {C : ℕ} (R U V P W G p ν s2 : Fin C → ℝ)
    (α L0 L1 KL : ℝ)
    (hR : ∀ c, 0 < R c) (hp : ∀ c, 0 < p c) (hpsum : ∑ c : Fin C, p c = 1)
    (hs2 : ∀ c, 0 < s2 c)
    (hS : ∀ c, 0 < ch08b_S (G c) (P c) (W c) (U c) (V c) (R c) α
            (ch08b_mstep_nu (U c) (V c) (R c) α))
    (hW0 : ∀ c, 0 ≤ W c) (hWpos : ∃ c, 0 < W c) (hKL : 0 ≤ KL)
    (hdec : L1 - L0
      = (ch08b_Qstats R U V P W G (ch08b_mstep_pi R)
            (fun c => ch08b_mstep_nu (U c) (V c) (R c) α)
            (fun c => ch08b_mstep_var (G c) (P c) (W c) (U c) (V c) (R c) α)
            (ch08b_mstep_alpha P V W (fun c => ch08b_mstep_nu (U c) (V c) (R c) α)
              (fun c => ch08b_mstep_var (G c) (P c) (W c) (U c) (V c) (R c) α))
          - ch08b_Qstats R U V P W G p ν s2 α) + KL) :
    L0 ≤ L1 := by
  have hQ := ch08b_em_ecm_ascent R U V P W G p ν s2 α hR hp hpsum hs2 hS hW0 hWpos
  linarith
end

end ThesisAudit
