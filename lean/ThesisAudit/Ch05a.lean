import Mathlib

/-!
Audit of the display-math formulas in
`thesis/chapters/ch05-gaussian-matrix.tex`, lines 1-784.

Conventions used throughout this file:

* Random variables are handled through their *algebraic* content.  The AR(1)
  chain is a genuine recursion `ch05a_ar1`; variances and covariances are
  computed from the unrolled representation \eqref{eq:g-unroll} as dot
  products of coefficient vectors over independent unit-variance atoms
  (`ch05a_cov`), which is exactly the computation the text performs.
* Conditional expectations of a centred Gaussian are encoded as the
  minimiser of the quadratic form (completing the square), which is the
  standard characterisation.
* Statements "for every `L`" about `L x L` matrices that are out of reach in
  full generality are checked at several concrete `L` with a *symbolic*
  parameter `α`; the size is always recorded in the comment.
-/

namespace ThesisAudit

open Finset

noncomputable section

/-! ### Shared definitions for Chapter 5 -/

/-- `σ_η² = 1 - α²` (tex line 44). -/
def ch05a_sigEta2 (α : ℝ) : ℝ := 1 - α ^ 2

/-- `Δ_t = 1 - e^{-2t}` (tex line 592). -/
def ch05a_Delta (t : ℝ) : ℝ := 1 - Real.exp (-2 * t)

theorem ch05a_Delta_nonneg {t : ℝ} (ht : 0 ≤ t) : 0 ≤ ch05a_Delta t := by
  simp only [ch05a_Delta, sub_nonneg]
  exact Real.exp_le_one_iff.mpr (by linarith)

/-- Stationary AR(1) covariance `(Σ₀)_{ij} = α^{|i-j|}`. -/
def ch05a_Sig0 (n : ℕ) (α : ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  fun i j => α ^ (Nat.dist i.val j.val)

/-- Clean tridiagonal precision `Q₀` of \eqref{eq:g-Q0}. -/
def ch05a_Q0 (n : ℕ) (α : ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  fun i j =>
    if i = j then
      (if i.val = 0 ∨ i.val = n - 1 then 1 else 1 + α ^ 2) / ch05a_sigEta2 α
    else if Nat.dist i.val j.val = 1 then -α / ch05a_sigEta2 α
    else 0

/-- Noisy covariance `Σ_t = e^{-2t} Σ₀ + Δ_t I_L`. -/
def ch05a_Sigt (n : ℕ) (α t : ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  Real.exp (-2 * t) • ch05a_Sig0 n α + ch05a_Delta t • (1 : Matrix (Fin n) (Fin n) ℝ)

/-- The one-dimensional Gaussian density `N(x; μ, v)`. -/
def ch05a_gauss (μ v x : ℝ) : ℝ :=
  Real.exp (-(x - μ) ^ 2 / (2 * v)) / Real.sqrt (2 * Real.pi * v)

theorem ch05a_gauss_pos {μ v x : ℝ} (hv : 0 < v) : 0 < ch05a_gauss μ v x := by
  have h : 0 < Real.sqrt (2 * Real.pi * v) := Real.sqrt_pos.mpr (by positivity)
  unfold ch05a_gauss
  positivity

theorem ch05a_neg2log_gauss {μ v x : ℝ} (hv : 0 < v) :
    -2 * Real.log (ch05a_gauss μ v x)
      = (x - μ) ^ 2 / v + Real.log (2 * Real.pi) + Real.log v := by
  have h2 : (0:ℝ) < 2 * Real.pi := by positivity
  have h3 : (0:ℝ) < 2 * Real.pi * v := by positivity
  unfold ch05a_gauss
  rw [Real.log_div (Real.exp_ne_zero _) (by positivity), Real.log_exp,
    Real.log_sqrt (le_of_lt h3), Real.log_mul (ne_of_gt h2) (ne_of_gt hv)]
  field_simp
  ring

/-! ### Section sec:g-model -/

-- eq:g-channel (ch05 lines 15-19): x_k = e^{-t} a_k + √Δ_t ξ_k.  Definition of the
-- variance-preserving OU channel acting on one frame.
def ch05a_g_channel (t a ξ : ℝ) : ℝ :=
  Real.exp (-t) * a + Real.sqrt (ch05a_Delta t) * ξ

-- sanity: at t = 0 the channel is the identity, since Δ₀ = 0.
theorem ch05a_g_channel_at_zero (a ξ : ℝ) : ch05a_g_channel 0 a ξ = a := by
  norm_num [ch05a_g_channel, ch05a_Delta]

-- eq:g-object (lines 21-27): s(x,t) = ∇_x log P_t(x).  Definition of the joint score
-- (one-dimensional encoding of the gradient).
def ch05a_g_score (P : ℝ → ℝ) (x : ℝ) : ℝ := deriv (fun y => Real.log (P y)) x

-- sanity: the score is the logarithmic derivative P'/P.
theorem ch05a_g_score_eq (P : ℝ → ℝ) (x d : ℝ) (h : HasDerivAt P d x) (hP : P x ≠ 0) :
    ch05a_g_score P x = d / P x := (h.log hP).deriv

/-! ### Section sec:g-clean: unrolling the recursion -/

-- eq:g-recursion (lines 38-46): a_{k+1} = α a_k + η_k.  Definition of the AR(1) chain.
def ch05a_ar1 (α a0 : ℝ) (η : ℕ → ℝ) : ℕ → ℝ
  | 0 => a0
  | k + 1 => α * ch05a_ar1 α a0 η k + η k

theorem ch05a_g_recursion (α a0 : ℝ) (η : ℕ → ℝ) (k : ℕ) :
    ch05a_ar1 α a0 η (k + 1) = α * ch05a_ar1 α a0 η k + η k := rfl

-- noname-2 (lines 60-62): a₁ = α a₀ + η₀.
theorem ch05a_noname2 (α a0 : ℝ) (η : ℕ → ℝ) : ch05a_ar1 α a0 η 1 = α * a0 + η 0 := rfl

-- noname-3 (lines 64-69): a₂ = α² a₀ + α η₀ + η₁.
theorem ch05a_noname3 (α a0 : ℝ) (η : ℕ → ℝ) :
    ch05a_ar1 α a0 η 2 = α ^ 2 * a0 + α * η 0 + η 1 := by
  simp only [ch05a_ar1]; ring

-- noname-4 (lines 71-75): a₃ = α³ a₀ + α² η₀ + α η₁ + η₂.
theorem ch05a_noname4 (α a0 : ℝ) (η : ℕ → ℝ) :
    ch05a_ar1 α a0 η 3 = α ^ 3 * a0 + α ^ 2 * η 0 + α * η 1 + η 2 := by
  simp only [ch05a_ar1]; ring

-- eq:g-unroll (lines 77-85): a_k = α^k a₀ + Σ_{j=0}^{k-1} α^{k-1-j} η_j.
theorem ch05a_g_unroll (α a0 : ℝ) (η : ℕ → ℝ) (k : ℕ) :
    ch05a_ar1 α a0 η k = α ^ k * a0 + ∑ j ∈ Finset.range k, α ^ (k - 1 - j) * η j := by
  induction k with
  | zero => simp [ch05a_ar1]
  | succ k ih =>
      rw [ch05a_g_recursion, ih, Finset.sum_range_succ]
      have hk : k + 1 - 1 - k = 0 := by omega
      rw [hk]
      have h : ∑ j ∈ Finset.range k, α ^ (k + 1 - 1 - j) * η j
             = α * ∑ j ∈ Finset.range k, α ^ (k - 1 - j) * η j := by
        rw [Finset.mul_sum]
        refine Finset.sum_congr rfl fun j hj => ?_
        rw [Finset.mem_range] at hj
        have hh : k + 1 - 1 - j = (k - 1 - j) + 1 := by omega
        rw [hh]; ring
      rw [h]; ring

-- noname-5 (lines 93-107): the induction step for eq:g-unroll, written out.
theorem ch05a_noname5 (α a0 : ℝ) (η : ℕ → ℝ) (k : ℕ) :
    α * (α ^ k * a0 + ∑ j ∈ Finset.range k, α ^ (k - 1 - j) * η j) + η k
      = α ^ (k + 1) * a0 + ∑ j ∈ Finset.range (k + 1), α ^ (k - j) * η j := by
  rw [Finset.sum_range_succ]
  have hk : k - k = 0 := by omega
  rw [hk]
  have h : ∑ j ∈ Finset.range k, α ^ (k - j) * η j
         = α * ∑ j ∈ Finset.range k, α ^ (k - 1 - j) * η j := by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun j hj => ?_
    rw [Finset.mem_range] at hj
    have hh : k - j = (k - 1 - j) + 1 := by omega
    rw [hh]; ring
  rw [h]; ring

-- noname-6 (lines 112-116): ε = (a₀, η₀, …, η_{L-2})ᵀ.  Definition.
def ch05a_eps (a0 : ℝ) (η : ℕ → ℝ) : ℕ → ℝ
  | 0 => a0
  | j + 1 => η j

-- eq:g-linear-map (lines 120-127): a = G_α ε with G_α lower triangular,
-- (G_α)_{ij} = α^{i-j} for j ≤ i and 0 otherwise.
def ch05a_G (n : ℕ) (α : ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  fun i j => if (j : ℕ) ≤ (i : ℕ) then α ^ ((i : ℕ) - (j : ℕ)) else 0

theorem ch05a_G_lower_triangular {n : ℕ} (α : ℝ) (i j : Fin n) (h : (i : ℕ) < (j : ℕ)) :
    ch05a_G n α i j = 0 := by
  simp only [ch05a_G, if_neg (by omega : ¬ ((j : ℕ) ≤ (i : ℕ)))]

-- eq:g-linear-map, checked at L = 4 with symbolic α.
theorem ch05a_g_linear_map_four (α a0 : ℝ) (η : ℕ → ℝ) (i : Fin 4) :
    ∑ j : Fin 4, ch05a_G 4 α i j * ch05a_eps a0 η (j : ℕ) = ch05a_ar1 α a0 η (i : ℕ) := by
  fin_cases i <;>
    simp [ch05a_G, ch05a_eps, ch05a_ar1, Fin.sum_univ_four] <;> ring

/-! ### Stationarity of the marginal variance -/

-- noname-7 (lines 134-140): Var(a_{k+1}) = α² Var(a_k) + σ_η² = 1 when Var(a_k) = 1.
theorem ch05a_noname7 (α v : ℝ) (hv : v = 1) :
    α ^ 2 * v + ch05a_sigEta2 α = 1 := by
  rw [hv, ch05a_sigEta2]; ring

/-- The variance recursion `v_{k+1} = α² v_k + σ_η²` with `v₀ = 1`. -/
def ch05a_var (α : ℝ) : ℕ → ℝ
  | 0 => 1
  | k + 1 => α ^ 2 * ch05a_var α k + ch05a_sigEta2 α

-- eq:g-stationary-marginal (lines 142-146): a_k ~ N(0,1) for every k, i.e. Var(a_k) = 1.
theorem ch05a_g_stationary_marginal (α : ℝ) (k : ℕ) : ch05a_var α k = 1 := by
  induction k with
  | zero => rfl
  | succ k ih => simp only [ch05a_var, ih, ch05a_sigEta2]; ring

-- eq:g-memory (lines 150-154): E[a_k | a₀] = α^k a₀.
def ch05a_condmean (α a0 : ℝ) : ℕ → ℝ
  | 0 => a0
  | k + 1 => α * ch05a_condmean α a0 k

theorem ch05a_g_memory (α a0 : ℝ) (k : ℕ) : ch05a_condmean α a0 k = α ^ k * a0 := by
  induction k with
  | zero => simp [ch05a_condmean]
  | succ k ih => simp only [ch05a_condmean, ih]; ring

-- noname-8 (lines 156-159): |α|^k → 0 as k → ∞ when |α| < 1.
theorem ch05a_noname8 (α : ℝ) (h : |α| < 1) :
    Filter.Tendsto (fun k : ℕ => |α| ^ k) Filter.atTop (nhds 0) :=
  tendsto_pow_atTop_nhds_zero_of_lt_one (abs_nonneg α) h

-- noname-9 (lines 162-166) and eq:g-split (lines 216-222):
-- a_{k+d} = α^d a_k + Σ_{j=k}^{k+d-1} α^{k+d-1-j} η_j.
theorem ch05a_g_split (α a0 : ℝ) (η : ℕ → ℝ) (k d : ℕ) :
    ch05a_ar1 α a0 η (k + d)
      = α ^ d * ch05a_ar1 α a0 η k
        + ∑ j ∈ Finset.Ico k (k + d), α ^ (k + d - 1 - j) * η j := by
  induction d with
  | zero => simp
  | succ d ih =>
      have he : k + (d + 1) = (k + d) + 1 := by omega
      rw [he, ch05a_g_recursion, ih, Finset.sum_Ico_succ_top (Nat.le_add_right k d)]
      have h2 : (k + d) + 1 - 1 - (k + d) = 0 := by omega
      rw [h2]
      have h1 : ∑ j ∈ Finset.Ico k (k + d), α ^ ((k + d) + 1 - 1 - j) * η j
              = α * ∑ j ∈ Finset.Ico k (k + d), α ^ (k + d - 1 - j) * η j := by
        rw [Finset.mul_sum]
        refine Finset.sum_congr rfl fun j hj => ?_
        rw [Finset.mem_Ico] at hj
        have hh : (k + d) + 1 - 1 - j = (k + d - 1 - j) + 1 := by omega
        rw [hh]; ring
      rw [h1]; ring

theorem ch05a_noname9 (α a0 : ℝ) (η : ℕ → ℝ) (j k : ℕ) (h : j ≤ k) :
    ch05a_ar1 α a0 η k
      = α ^ (k - j) * ch05a_ar1 α a0 η j
        + ∑ r ∈ Finset.Ico j k, α ^ (k - 1 - r) * η r := by
  have he : k = j + (k - j) := by omega
  calc ch05a_ar1 α a0 η k = ch05a_ar1 α a0 η (j + (k - j)) := by rw [← he]
    _ = α ^ (k - j) * ch05a_ar1 α a0 η j
        + ∑ r ∈ Finset.Ico j (j + (k - j)), α ^ (j + (k - j) - 1 - r) * η r :=
          ch05a_g_split α a0 η j (k - j)
    _ = α ^ (k - j) * ch05a_ar1 α a0 η j
        + ∑ r ∈ Finset.Ico j k, α ^ (k - 1 - r) * η r := by rw [← he]

/-! ### The stationary covariance -/

/-- Covariance of `a_k` and `a_{k+d}` computed from the unrolled representation
\eqref{eq:g-unroll}: the atom `a₀` has variance 1 and each `η_j` has variance `σ_η²`. -/
def ch05a_gcov (α : ℝ) (k d : ℕ) : ℝ :=
  α ^ k * α ^ (k + d)
    + ch05a_sigEta2 α * ∑ j ∈ Finset.range k, α ^ (k - 1 - j) * α ^ (k + d - 1 - j)

-- eq:g-cov-lag (lines 225-232): Cov(a_k, a_{k+d}) = α^d Var(a_k) = α^d, i.e. (Σ₀)_{ij}=α^{|i-j|}.
theorem ch05a_g_cov_lag (α : ℝ) (hα : α ^ 2 ≠ 1) (k d : ℕ) : ch05a_gcov α k d = α ^ d := by
  have hne : α ^ 2 - 1 ≠ 0 := sub_ne_zero_of_ne hα
  have key : ∑ j ∈ Finset.range k, α ^ (k - 1 - j) * α ^ (k + d - 1 - j)
           = ∑ j ∈ Finset.range k, α ^ j * α ^ (d + j) := by
    rw [← Finset.sum_range_reflect (fun i => α ^ i * α ^ (d + i)) k]
    refine Finset.sum_congr rfl fun j hj => ?_
    rw [Finset.mem_range] at hj
    have h : k + d - 1 - j = d + (k - 1 - j) := by omega
    rw [h]
  have key2 : ∑ j ∈ Finset.range k, α ^ j * α ^ (d + j)
            = α ^ d * ∑ j ∈ Finset.range k, (α ^ 2) ^ j := by
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun j _ => by ring
  rw [ch05a_gcov, key, key2, geom_sum_eq hα k, ch05a_sigEta2]
  field_simp
  ring

-- eq:g-ar1-covariance (lines 174-179): Cov(a_j,a_k) = α^{|k-j|}.
theorem ch05a_g_ar1_covariance (α : ℝ) (hα : α ^ 2 ≠ 1) (j k : ℕ) :
    ch05a_gcov α (min j k) (Nat.dist j k) = α ^ (Nat.dist j k) :=
  ch05a_g_cov_lag α hα _ _

-- noname-10 (lines 168-172): Cov(a_j,a_k) = α^{k-j} Var(a_j) = α^{k-j}.
theorem ch05a_noname10 (α : ℝ) (hα : α ^ 2 ≠ 1) (k d : ℕ) :
    ch05a_gcov α k d = α ^ d * ch05a_var α k ∧ ch05a_gcov α k d = α ^ d := by
  refine ⟨?_, ch05a_g_cov_lag α hα k d⟩
  rw [ch05a_g_cov_lag α hα k d, ch05a_g_stationary_marginal]; ring

-- eq:g-var-sum (lines 190-197): Var(a_k) = α^{2k}·Var(a₀) + σ_η² Σ_{j<k} α^{2(k-1-j)}
--                                        = α^{2k} + σ_η² Σ_{i<k} (α²)^i.
theorem ch05a_g_var_sum (α : ℝ) (k : ℕ) :
    α ^ (2 * k) * 1 + ch05a_sigEta2 α * ∑ j ∈ Finset.range k, α ^ (2 * (k - 1 - j))
      = α ^ (2 * k) + ch05a_sigEta2 α * ∑ i ∈ Finset.range k, (α ^ 2) ^ i := by
  have h : ∑ j ∈ Finset.range k, α ^ (2 * (k - 1 - j))
         = ∑ i ∈ Finset.range k, (α ^ 2) ^ i := by
    rw [← Finset.sum_range_reflect (fun i => (α ^ 2) ^ i) k]
    exact Finset.sum_congr rfl fun j _ => pow_mul α 2 (k - 1 - j)
  rw [h]; ring

-- eq:g-geom (lines 200-204): Σ_{i<k} (α²)^i = (1-α^{2k})/(1-α²).
theorem ch05a_g_geom (α : ℝ) (hα : α ^ 2 ≠ 1) (k : ℕ) :
    ∑ i ∈ Finset.range k, (α ^ 2) ^ i = (1 - α ^ (2 * k)) / (1 - α ^ 2) := by
  have h1 : α ^ 2 - 1 ≠ 0 := sub_ne_zero_of_ne hα
  have h2 : (1 : ℝ) - α ^ 2 ≠ 0 := sub_ne_zero_of_ne (Ne.symm hα)
  rw [geom_sum_eq hα k, pow_mul]
  field_simp
  ring

-- eq:g-var-one (lines 206-212): α^{2k} + (1-α²)(1-α^{2k})/(1-α²) = 1.
theorem ch05a_g_var_one (α : ℝ) (hα : α ^ 2 ≠ 1) (k : ℕ) :
    α ^ (2 * k) + (1 - α ^ 2) * ((1 - α ^ (2 * k)) / (1 - α ^ 2)) = 1 := by
  have h2 : (1 : ℝ) - α ^ 2 ≠ 0 := sub_ne_zero_of_ne (Ne.symm hα)
  field_simp
  ring

-- noname-24 (lines 661-663) and eq:g-cov-lag right half: (Σ₀)_{ij} = α^{|i-j|}.
theorem ch05a_noname24 (n : ℕ) (α : ℝ) (i j : Fin n) :
    ch05a_Sig0 n α i j = α ^ (Nat.dist i.val j.val) := rfl

-- eq:g-Sig0-matrix (lines 244-255): the explicit Toeplitz display, checked at L = 4.
theorem ch05a_g_Sig0_matrix (α : ℝ) :
    ch05a_Sig0 4 α = !![1, α, α ^ 2, α ^ 3;
                        α, 1, α, α ^ 2;
                        α ^ 2, α, 1, α;
                        α ^ 3, α ^ 2, α, 1] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [ch05a_Sig0, Nat.dist]

/-! ### Section sec:g-precision -/

-- eq:g-Q0-K2 (lines 286-293): inverse of the 2x2 stationary covariance.
theorem ch05a_g_Q0_K2 (α : ℝ) (hα : (1 : ℝ) - α ^ 2 ≠ 0) :
    (!![1, α; α, 1] : Matrix (Fin 2) (Fin 2) ℝ)⁻¹
      = (1 - α ^ 2)⁻¹ • !![1, -α; -α, 1] := by
  apply Matrix.inv_eq_right_inv
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.one_apply] <;> field_simp <;> ring

-- eq:g-Q0-K3 (lines 298-311): inverse of the 3x3 stationary covariance.
theorem ch05a_g_Q0_K3 (α : ℝ) (hα : (1 : ℝ) - α ^ 2 ≠ 0) :
    (!![1, α, α ^ 2; α, 1, α; α ^ 2, α, 1] : Matrix (Fin 3) (Fin 3) ℝ)⁻¹
      = (1 - α ^ 2)⁻¹ • !![1, -α, 0; -α, 1 + α ^ 2, -α; 0, -α, 1] := by
  apply Matrix.inv_eq_right_inv
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.one_apply] <;> field_simp <;> ring

-- eq:g-Q0 (lines 320-334): Q₀ Σ₀ = I for the tridiagonal Q₀, checked at L = 2,3,4,5.
theorem ch05a_g_Q0_two (α : ℝ) (hα : (1 : ℝ) - α ^ 2 ≠ 0) :
    ch05a_Q0 2 α * ch05a_Sig0 2 α = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [ch05a_Q0, ch05a_Sig0, ch05a_sigEta2, Matrix.mul_apply, Fin.sum_univ_succ,
      Matrix.one_apply, Nat.dist] <;> field_simp <;> ring

theorem ch05a_g_Q0_three (α : ℝ) (hα : (1 : ℝ) - α ^ 2 ≠ 0) :
    ch05a_Q0 3 α * ch05a_Sig0 3 α = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [ch05a_Q0, ch05a_Sig0, ch05a_sigEta2, Matrix.mul_apply, Fin.sum_univ_succ,
      Matrix.one_apply, Nat.dist] <;> field_simp <;> ring

theorem ch05a_g_Q0_four (α : ℝ) (hα : (1 : ℝ) - α ^ 2 ≠ 0) :
    ch05a_Q0 4 α * ch05a_Sig0 4 α = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [ch05a_Q0, ch05a_Sig0, ch05a_sigEta2, Matrix.mul_apply, Fin.sum_univ_succ,
      Matrix.one_apply, Nat.dist] <;> field_simp <;> ring

theorem ch05a_g_Q0_five (α : ℝ) (hα : (1 : ℝ) - α ^ 2 ≠ 0) :
    ch05a_Q0 5 α * ch05a_Sig0 5 α = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [ch05a_Q0, ch05a_Sig0, ch05a_sigEta2, Matrix.mul_apply, Fin.sum_univ_succ,
      Matrix.one_apply, Nat.dist] <;> field_simp <;> ring

-- noname-11 (lines 336-349): the entrywise description of Q₀ (this is `ch05a_Q0`);
-- eq:g-precision-offdiag (lines 437-444): (Q₀)_{k,k+1} = (Q₀)_{k+1,k} = -α/σ_η².
theorem ch05a_g_precision_offdiag {n : ℕ} (α : ℝ) (i j : Fin n) (h : Nat.dist i.val j.val = 1) :
    ch05a_Q0 n α i j = -α / ch05a_sigEta2 α := by
  have hij : i ≠ j := by
    intro he; rw [he, Nat.dist_self] at h; exact absurd h (by norm_num)
  simp [ch05a_Q0, hij, h]

-- eq:g-precision-distant-zero (lines 445-448): (Q₀)_{ij} = 0 for |i-j| ≥ 2.
theorem ch05a_g_precision_distant_zero {n : ℕ} (α : ℝ) (i j : Fin n)
    (h : 2 ≤ Nat.dist i.val j.val) : ch05a_Q0 n α i j = 0 := by
  have hij : i ≠ j := by
    intro he; rw [he, Nat.dist_self] at h; exact absurd h (by norm_num)
  have h1 : Nat.dist i.val j.val ≠ 1 := by omega
  simp [ch05a_Q0, hij, h1]

-- eq:g-precision-interior (lines 463-470): (Q₀)_{mm} = (1+α²)/σ_η² for interior m.
theorem ch05a_g_precision_interior {n : ℕ} (α : ℝ) (m : Fin n)
    (h0 : m.val ≠ 0) (h1 : m.val ≠ n - 1) :
    ch05a_Q0 n α m m = (1 + α ^ 2) / ch05a_sigEta2 α := by
  simp [ch05a_Q0, h0, h1]

-- noname-17 (lines 487-491): (Q₀)_{00} = 1/σ_η².
theorem ch05a_noname17 {n : ℕ} (α : ℝ) (h : 0 < n) :
    ch05a_Q0 n α ⟨0, h⟩ ⟨0, h⟩ = 1 / ch05a_sigEta2 α := by
  simp [ch05a_Q0]

-- noname-19 (lines 498-502): (Q₀)_{L-1,L-1} = 1/σ_η².
theorem ch05a_noname19 {n : ℕ} (α : ℝ) (h : 0 < n) :
    ch05a_Q0 n α ⟨n - 1, by omega⟩ ⟨n - 1, by omega⟩ = 1 / ch05a_sigEta2 α := by
  simp [ch05a_Q0]

/-! ### Reading Q₀ off -2 log P₀ -/

-- eq:g-gaussian-precision (lines 355-362): the centred Gaussian density in precision form.
def ch05a_mvgauss (L : ℕ) (Q : Matrix (Fin L) (Fin L) ℝ) (detS : ℝ) (a : Fin L → ℝ) : ℝ :=
  (2 * Real.pi) ^ (-(L : ℝ) / 2) * detS ^ (-(1 : ℝ) / 2) *
    Real.exp (-(1 / 2) * (a ⬝ᵥ (Matrix.mulVec Q a)))

theorem ch05a_g_gaussian_precision_pos (L : ℕ) (Q : Matrix (Fin L) (Fin L) ℝ)
    {detS : ℝ} (h : 0 < detS) (a : Fin L → ℝ) : 0 < ch05a_mvgauss L Q detS a := by
  have h2 : (0:ℝ) < 2 * Real.pi := by positivity
  unfold ch05a_mvgauss
  have h3 : 0 < (2 * Real.pi) ^ (-(L : ℝ) / 2) := Real.rpow_pos_of_pos h2 _
  have h4 : 0 < detS ^ (-(1 : ℝ) / 2) := Real.rpow_pos_of_pos h _
  exact mul_pos (mul_pos h3 h4) (Real.exp_pos _)

-- eq:g-read-precision (lines 365-373): -2 log P₀(a) = aᵀQ₀a + L log 2π + log|Σ₀|.
theorem ch05a_g_read_precision (L : ℕ) (Q : Matrix (Fin L) (Fin L) ℝ) {detS : ℝ}
    (h : 0 < detS) (a : Fin L → ℝ) :
    -2 * Real.log (ch05a_mvgauss L Q detS a)
      = (a ⬝ᵥ (Matrix.mulVec Q a)) + (L : ℝ) * Real.log (2 * Real.pi) + Real.log detS := by
  have h2 : (0:ℝ) < 2 * Real.pi := by positivity
  have h3 : 0 < (2 * Real.pi) ^ (-(L : ℝ) / 2) := Real.rpow_pos_of_pos h2 _
  have h4 : 0 < detS ^ (-(1 : ℝ) / 2) := Real.rpow_pos_of_pos h _
  unfold ch05a_mvgauss
  rw [Real.log_mul (by positivity) (Real.exp_ne_zero _),
    Real.log_mul (ne_of_gt h3) (ne_of_gt h4),
    Real.log_rpow h2, Real.log_rpow h, Real.log_exp]
  ring

-- noname-12 (lines 380-385): the Markov factorisation P₀(a) = p(a₀) ∏ p(a_{k+1}|a_k).
-- noname-13 (lines 387-392): a₀ ~ N(0,1) and a_{k+1}|a_k ~ N(α a_k, σ_η²).  Definitions.
def ch05a_P0_factorised (m : ℕ) (α : ℝ) (a : ℕ → ℝ) : ℝ :=
  ch05a_gauss 0 1 (a 0) *
    ∏ k ∈ Finset.range m, ch05a_gauss (α * a k) (ch05a_sigEta2 α) (a (k + 1))

theorem ch05a_P0_factorised_pos (m : ℕ) (α : ℝ) (hσ : 0 < ch05a_sigEta2 α) (a : ℕ → ℝ) :
    0 < ch05a_P0_factorised m α a :=
  mul_pos (ch05a_gauss_pos one_pos)
    (Finset.prod_pos fun k _ => ch05a_gauss_pos hσ)

-- eq:g-precision-expand-1 (lines 394-403):
-- -2 log P₀(a) = a₀² + (1/σ_η²) Σ_{k=0}^{L-2} (a_{k+1} - α a_k)² + const.
theorem ch05a_g_precision_expand_1 (m : ℕ) (α : ℝ) (hσ : 0 < ch05a_sigEta2 α) (a : ℕ → ℝ) :
    -2 * Real.log (ch05a_P0_factorised m α a)
      = (a 0) ^ 2
        + (1 / ch05a_sigEta2 α) * ∑ k ∈ Finset.range m, (a (k + 1) - α * a k) ^ 2
        + (((m : ℝ) + 1) * Real.log (2 * Real.pi) + (m : ℝ) * Real.log (ch05a_sigEta2 α)) := by
  have hA : -2 * Real.log (ch05a_gauss 0 1 (a 0)) = (a 0) ^ 2 + Real.log (2 * Real.pi) := by
    rw [ch05a_neg2log_gauss one_pos]; simp
  unfold ch05a_P0_factorised
  rw [Real.log_mul (ne_of_gt (ch05a_gauss_pos one_pos))
      (ne_of_gt (Finset.prod_pos fun k _ => ch05a_gauss_pos hσ)),
    Real.log_prod (fun k _ => ne_of_gt (ch05a_gauss_pos hσ)),
    mul_add, Finset.mul_sum, hA]
  have hB : ∀ k ∈ Finset.range m,
      -2 * Real.log (ch05a_gauss (α * a k) (ch05a_sigEta2 α) (a (k + 1)))
        = (a (k + 1) - α * a k) ^ 2 / ch05a_sigEta2 α
          + Real.log (2 * Real.pi) + Real.log (ch05a_sigEta2 α) :=
    fun k _ => ch05a_neg2log_gauss hσ
  rw [Finset.sum_congr rfl hB, Finset.sum_add_distrib, Finset.sum_add_distrib,
    Finset.sum_const, Finset.sum_const, Finset.card_range, nsmul_eq_mul, nsmul_eq_mul]
  have hsum : (1 / ch05a_sigEta2 α) * ∑ k ∈ Finset.range m, (a (k + 1) - α * a k) ^ 2
            = ∑ k ∈ Finset.range m, (a (k + 1) - α * a k) ^ 2 / ch05a_sigEta2 α := by
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun k _ => by ring
  rw [hsum]; ring

-- eq:g-precision-expand-2 (lines 404-415): expanding the square inside the sum.
theorem ch05a_g_precision_expand_2 (m : ℕ) (α s c : ℝ) (a : ℕ → ℝ) :
    (a 0) ^ 2 + (1 / s) * ∑ k ∈ Finset.range m, (a (k + 1) - α * a k) ^ 2 + c
      = (a 0) ^ 2 + (1 / s) *
          ∑ k ∈ Finset.range m,
            ((a (k + 1)) ^ 2 - 2 * α * (a k) * (a (k + 1)) + α ^ 2 * (a k) ^ 2) + c := by
  have h : ∀ k ∈ Finset.range m, (a (k + 1) - α * a k) ^ 2
      = (a (k + 1)) ^ 2 - 2 * α * (a k) * (a (k + 1)) + α ^ 2 * (a k) ^ 2 :=
    fun k _ => by ring
  rw [Finset.sum_congr rfl h]

-- eq:g-general-quadratic-form (lines 418-426): aᵀQa = Σ Q_mm a_m² + 2 Σ_{m<n} Q_mn a_m a_n,
-- checked at L = 3 and L = 4 for a symmetric Q.
theorem ch05a_g_general_quadratic_form_three (Q : Matrix (Fin 3) (Fin 3) ℝ) (a : Fin 3 → ℝ)
    (h01 : Q 1 0 = Q 0 1) (h02 : Q 2 0 = Q 0 2) (h12 : Q 2 1 = Q 1 2) :
    a ⬝ᵥ (Matrix.mulVec Q a)
      = (Q 0 0 * (a 0) ^ 2 + Q 1 1 * (a 1) ^ 2 + Q 2 2 * (a 2) ^ 2)
        + 2 * (Q 0 1 * a 0 * a 1 + Q 0 2 * a 0 * a 2 + Q 1 2 * a 1 * a 2) := by
  simp [dotProduct, Matrix.mulVec, Fin.sum_univ_three, h01, h02, h12]
  ring

theorem ch05a_g_general_quadratic_form_four (Q : Matrix (Fin 4) (Fin 4) ℝ) (a : Fin 4 → ℝ)
    (h01 : Q 1 0 = Q 0 1) (h02 : Q 2 0 = Q 0 2) (h03 : Q 3 0 = Q 0 3)
    (h12 : Q 2 1 = Q 1 2) (h13 : Q 3 1 = Q 1 3) (h23 : Q 3 2 = Q 2 3) :
    a ⬝ᵥ (Matrix.mulVec Q a)
      = (Q 0 0 * (a 0) ^ 2 + Q 1 1 * (a 1) ^ 2 + Q 2 2 * (a 2) ^ 2 + Q 3 3 * (a 3) ^ 2)
        + 2 * (Q 0 1 * a 0 * a 1 + Q 0 2 * a 0 * a 2 + Q 0 3 * a 0 * a 3
              + Q 1 2 * a 1 * a 2 + Q 1 3 * a 1 * a 3 + Q 2 3 * a 2 * a 3) := by
  simp [dotProduct, Matrix.mulVec, Fin.sum_univ_four, h01, h02, h03, h12, h13, h23]
  ring

-- noname-14 (lines 431-434): the only cross terms are -(2α/σ_η²) Σ a_k a_{k+1}.
theorem ch05a_noname14 (m : ℕ) (α s : ℝ) (a : ℕ → ℝ) :
    (1 / s) * ∑ k ∈ Finset.range m, (a (k + 1) - α * a k) ^ 2
      = (1 / s) * ∑ k ∈ Finset.range m, ((a (k + 1)) ^ 2 + α ^ 2 * (a k) ^ 2)
        - (2 * α / s) * ∑ k ∈ Finset.range m, a k * a (k + 1) := by
  rw [Finset.mul_sum, Finset.mul_sum, Finset.mul_sum, ← Finset.sum_sub_distrib]
  exact Finset.sum_congr rfl fun k _ => by ring

-- noname-15 (lines 456-461): the two transition factors containing a_m.
theorem ch05a_noname15 (α s x y z : ℝ) :
    (1 / s) * (y - α * x) ^ 2 = (1 / s) * y ^ 2 - (2 * α / s) * y * x + (α ^ 2 / s) * x ^ 2
    ∧ (1 / s) * (z - α * y) ^ 2
        = (α ^ 2 / s) * y ^ 2 - (2 * α / s) * z * y + (1 / s) * z ^ 2 := by
  constructor <;> ring

-- noname-16 (lines 475-484): a₀² + (α²/σ_η²) a₀² = ((σ_η²+α²)/σ_η²) a₀² = (1/σ_η²) a₀².
theorem ch05a_noname16 (α x : ℝ) (hσ : ch05a_sigEta2 α ≠ 0) :
    x ^ 2 + (α ^ 2 / ch05a_sigEta2 α) * x ^ 2
        = ((ch05a_sigEta2 α + α ^ 2) / ch05a_sigEta2 α) * x ^ 2
    ∧ ((ch05a_sigEta2 α + α ^ 2) / ch05a_sigEta2 α) * x ^ 2 = (1 / ch05a_sigEta2 α) * x ^ 2 := by
  constructor
  · field_simp
  · simp only [ch05a_sigEta2]; ring_nf

-- noname-18 (lines 493-496): the last transition factor, expanded.
theorem ch05a_noname18 (α s x y : ℝ) :
    (1 / s) * (x - α * y) ^ 2 = (1 / s) * x ^ 2 - (2 * α / s) * x * y + (α ^ 2 / s) * y ^ 2 := by
  ring

-- eq:g-precision-ci (lines 513-520): (Q)_{ij} = 0 ⟺ a_i ⊥ a_j given the rest.
-- Encoded as: the conditional density (a quadratic exponential in the pair (x,y) with all
-- other coordinates fixed) factorises as g(x)h(y) if and only if the off-diagonal q₁₂ vanishes.
theorem ch05a_g_precision_ci (q11 q12 q22 b1 b2 : ℝ) :
    (∀ x y : ℝ,
        Real.exp (-(1/2) * (q11 * x ^ 2 + 2 * q12 * x * y + q22 * y ^ 2 + b1 * x + b2 * y))
          * Real.exp (-(1/2) * (q11 * 0 ^ 2 + 2 * q12 * 0 * 0 + q22 * 0 ^ 2 + b1 * 0 + b2 * 0))
        = Real.exp (-(1/2) * (q11 * x ^ 2 + 2 * q12 * x * 0 + q22 * 0 ^ 2 + b1 * x + b2 * 0))
          * Real.exp (-(1/2) * (q11 * 0 ^ 2 + 2 * q12 * 0 * y + q22 * y ^ 2 + b1 * 0 + b2 * y)))
      ↔ q12 = 0 := by
  constructor
  · intro h
    have h1 := h 1 1
    rw [← Real.exp_add, ← Real.exp_add, Real.exp_eq_exp] at h1
    nlinarith [h1]
  · intro h x y
    rw [← Real.exp_add, ← Real.exp_add, Real.exp_eq_exp, h]
    ring

-- noname-20 (lines 527-534): -log P(a) = -log p(a₀) - Σ log p(a_{k+1}|a_k).
theorem ch05a_noname20 (m : ℕ) (p0 : ℝ) (p : ℕ → ℝ) (hp0 : p0 ≠ 0)
    (hp : ∀ k ∈ Finset.range m, p k ≠ 0) :
    -Real.log (p0 * ∏ k ∈ Finset.range m, p k)
      = -Real.log p0 - ∑ k ∈ Finset.range m, Real.log (p k) := by
  rw [Real.log_mul hp0 (Finset.prod_ne_zero_iff.mpr hp), Real.log_prod hp]
  ring

-- eq:g-markov-hessian-sparsity (lines 536-542): ∂²(-log P)/∂a_i∂a_j = 0 for |i-j| ≥ 2.
-- Instance: when the negative log-density splits as g(a_i) + h(a_j) in the two coordinates
-- (no factor of a first-order chain contains both), the mixed partial vanishes.
theorem ch05a_g_markov_hessian_sparsity (g h : ℝ → ℝ) (x0 z0 : ℝ) :
    deriv (fun z => deriv (fun x => g x + h z) x0) z0 = 0 := by
  have hc : (fun z => deriv (fun x => g x + h z) x0) = fun _ => deriv g x0 := by
    funext z
    simp [deriv_add_const]
  rw [hc, deriv_const]

-- eq:g-gaussian-cond-precision (lines 549-555): E[a_k | a_{\k}] = -(Q_kk)^{-1} Σ_{l≠k} Q_kl a_l,
-- encoded as: that value is the minimiser of the quadratic form in coordinate k.
theorem ch05a_g_gaussian_cond_precision {n : ℕ} (Q : Matrix (Fin n) (Fin n) ℝ) (a : Fin n → ℝ)
    (k : Fin n) (hq : 0 < Q k k) (x : ℝ) :
    Q k k * (-(Q k k)⁻¹ * ∑ l ∈ Finset.univ.erase k, Q k l * a l) ^ 2
      + 2 * (-(Q k k)⁻¹ * ∑ l ∈ Finset.univ.erase k, Q k l * a l)
        * (∑ l ∈ Finset.univ.erase k, Q k l * a l)
      ≤ Q k k * x ^ 2 + 2 * x * (∑ l ∈ Finset.univ.erase k, Q k l * a l) := by
  set q := Q k k with hqdef
  set S := ∑ l ∈ Finset.univ.erase k, Q k l * a l with hSdef
  have hq0 : q ≠ 0 := ne_of_gt hq
  have key : q * x ^ 2 + 2 * x * S - (q * (-q⁻¹ * S) ^ 2 + 2 * (-q⁻¹ * S) * S)
      = q * (x + q⁻¹ * S) ^ 2 := by
    field_simp
    ring
  have h2 : 0 ≤ q * (x + q⁻¹ * S) ^ 2 := mul_nonneg hq.le (sq_nonneg _)
  linarith [key, h2]

-- eq:g-cond-mean (lines 558-572): for an interior frame,
-- E[a_k | a_{\k}] = α/(1+α²) (a_{k-1} + a_{k+1}).
theorem ch05a_g_cond_mean (α s x y : ℝ) (hs : s ≠ 0) (hα : 1 + α ^ 2 ≠ 0) :
    -(s / (1 + α ^ 2)) * ((-α / s) * x + (-α / s) * y) = α / (1 + α ^ 2) * (x + y) := by
  field_simp
  ring

/-! ### Section sec:g-noisy -/

/-- Covariance of two random variables represented by their coefficient vectors over
a finite family of independent unit-variance atoms. -/
def ch05a_cov {m : ℕ} (u v : Fin m → ℝ) : ℝ := ∑ r : Fin m, u r * v r

-- eq:g-channel-vec (lines 594-600): x = e^{-t} a + √Δ_t ξ, ξ ~ N(0, I_L), ξ ⊥ a.  Definition.
def ch05a_g_channel_vec {n : ℕ} (t : ℝ) (a ξ : Fin n → ℝ) : Fin n → ℝ :=
  fun i => Real.exp (-t) * a i + Real.sqrt (ch05a_Delta t) * ξ i

-- noname-21 (lines 624-631): E[x] = e^{-t} E[a] + √Δ_t E[ξ] = 0.
theorem ch05a_noname21 (c s ma mξ : ℝ) (ha : ma = 0) (hξ : mξ = 0) :
    c * ma + s * mξ = 0 := by rw [ha, hξ]; ring

-- noname-22 (lines 633-648): Cov(c a + s ξ) = c² Cov(a) + s² Cov(ξ) + c s (Cov(a,ξ)+Cov(ξ,a)).
theorem ch05a_noname22 {n m : ℕ} (A X : Fin n → Fin m → ℝ) (c s : ℝ) (i j : Fin n) :
    ch05a_cov (fun r => c * A i r + s * X i r) (fun r => c * A j r + s * X j r)
      = c ^ 2 * ch05a_cov (A i) (A j) + s ^ 2 * ch05a_cov (X i) (X j)
        + c * s * (ch05a_cov (A i) (X j) + ch05a_cov (X i) (A j)) := by
  simp only [ch05a_cov, Finset.mul_sum, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun r _ => by ring

-- eq:g-Sigt (lines 607-618) and noname-23 (lines 652-657):
-- Σ_t = e^{-2t} Σ₀ + Δ_t I_L.
theorem ch05a_g_Sigt_cov {n m : ℕ} (A X : Fin n → Fin m → ℝ) (S : Matrix (Fin n) (Fin n) ℝ)
    (c s : ℝ)
    (hA : ∀ i j, ch05a_cov (A i) (A j) = S i j)
    (hX : ∀ i j, ch05a_cov (X i) (X j) = if i = j then 1 else 0)
    (hAX : ∀ i j, ch05a_cov (A i) (X j) = 0)
    (hXA : ∀ i j, ch05a_cov (X i) (A j) = 0)
    (i j : Fin n) :
    ch05a_cov (fun r => c * A i r + s * X i r) (fun r => c * A j r + s * X j r)
      = c ^ 2 * S i j + s ^ 2 * (if i = j then 1 else 0) := by
  rw [ch05a_noname22, hA, hX, hAX, hXA]
  ring

theorem ch05a_noname23 {n m : ℕ} (A X : Fin n → Fin m → ℝ) (S : Matrix (Fin n) (Fin n) ℝ)
    {t : ℝ} (ht : 0 ≤ t)
    (hA : ∀ i j, ch05a_cov (A i) (A j) = S i j)
    (hX : ∀ i j, ch05a_cov (X i) (X j) = if i = j then 1 else 0)
    (hAX : ∀ i j, ch05a_cov (A i) (X j) = 0)
    (hXA : ∀ i j, ch05a_cov (X i) (A j) = 0)
    (i j : Fin n) :
    ch05a_cov (fun r => Real.exp (-t) * A i r + Real.sqrt (ch05a_Delta t) * X i r)
        (fun r => Real.exp (-t) * A j r + Real.sqrt (ch05a_Delta t) * X j r)
      = Real.exp (-2 * t) * S i j + ch05a_Delta t * (if i = j then 1 else 0) := by
  rw [ch05a_g_Sigt_cov A X S _ _ hA hX hAX hXA i j,
    Real.sq_sqrt (ch05a_Delta_nonneg ht)]
  have hc : Real.exp (-t) ^ 2 = Real.exp (-2 * t) := by
    rw [pow_two, ← Real.exp_add]; congr 1; ring
  rw [hc]

-- eq:g-Sigt-entrywise (lines 665-672): (Σ_t)_{ij} = e^{-2t} α^{|i-j|} + Δ_t δ_{ij}.
theorem ch05a_g_Sigt_entrywise (n : ℕ) (α t : ℝ) (i j : Fin n) :
    ch05a_Sigt n α t i j
      = Real.exp (-2 * t) * α ^ (Nat.dist i.val j.val)
        + ch05a_Delta t * (if i = j then 1 else 0) := by
  simp [ch05a_Sigt, ch05a_Sig0, Matrix.one_apply]

-- noname-25 (lines 677-684): (Σ_t)_{ii} = e^{-2t} + Δ_t = 1: the channel is variance preserving.
theorem ch05a_noname25 (n : ℕ) (α t : ℝ) (i : Fin n) : ch05a_Sigt n α t i i = 1 := by
  rw [ch05a_g_Sigt_entrywise]
  simp [ch05a_Delta, Nat.dist_self]

-- eq:g-Sigt-offdiag (lines 690-696): (Σ_t)_{ij} = e^{-2t} α^{|i-j|} for i ≠ j.
theorem ch05a_g_Sigt_offdiag (n : ℕ) (α t : ℝ) (i j : Fin n) (h : i ≠ j) :
    ch05a_Sigt n α t i j = Real.exp (-2 * t) * α ^ (Nat.dist i.val j.val) := by
  rw [ch05a_g_Sigt_entrywise]; simp [h]

-- eq:g-noisy-correlation (lines 699-704): Corr(x_i,x_j) = e^{-2t} α^{|i-j|}  (i ≠ j).
theorem ch05a_g_noisy_correlation (n : ℕ) (α t : ℝ) (i j : Fin n) (h : i ≠ j) :
    ch05a_Sigt n α t i j / Real.sqrt (ch05a_Sigt n α t i i * ch05a_Sigt n α t j j)
      = Real.exp (-2 * t) * α ^ (Nat.dist i.val j.val) := by
  rw [ch05a_noname25, ch05a_noname25, ch05a_g_Sigt_offdiag n α t i j h]
  simp

-- eq:g-Sigt-matrix (lines 709-731): the explicit noisy Toeplitz display, checked at L = 4.
theorem ch05a_g_Sigt_matrix (α t : ℝ) :
    ch05a_Sigt 4 α t
      = !![1, Real.exp (-2*t) * α, Real.exp (-2*t) * α ^ 2, Real.exp (-2*t) * α ^ 3;
           Real.exp (-2*t) * α, 1, Real.exp (-2*t) * α, Real.exp (-2*t) * α ^ 2;
           Real.exp (-2*t) * α ^ 2, Real.exp (-2*t) * α, 1, Real.exp (-2*t) * α;
           Real.exp (-2*t) * α ^ 3, Real.exp (-2*t) * α ^ 2, Real.exp (-2*t) * α, 1] := by
  ext i j
  rw [ch05a_g_Sigt_entrywise]
  fin_cases i <;> fin_cases j <;> norm_num [Nat.dist, ch05a_Delta]

-- eq:g-Sigt-limits (lines 737-745): Σ_t|_{t=0} = Σ₀ and Σ_t → I_L as t → ∞.
theorem ch05a_g_Sigt_limit_zero (n : ℕ) (α : ℝ) : ch05a_Sigt n α 0 = ch05a_Sig0 n α := by
  ext i j
  rw [ch05a_g_Sigt_entrywise]
  simp [ch05a_Delta, ch05a_Sig0]

theorem ch05a_g_Sigt_limit_infty (n : ℕ) (α : ℝ) (i j : Fin n) :
    Filter.Tendsto (fun t => ch05a_Sigt n α t i j) Filter.atTop
      (nhds ((1 : Matrix (Fin n) (Fin n) ℝ) i j)) := by
  have h2 : Filter.Tendsto (fun t : ℝ => 2 * t) Filter.atTop Filter.atTop :=
    Filter.tendsto_atTop_atTop.mpr fun b => ⟨b / 2, fun a ha => by linarith⟩
  have hexp : Filter.Tendsto (fun t : ℝ => Real.exp (-2 * t)) Filter.atTop (nhds 0) := by
    have key : (fun t : ℝ => Real.exp (-2 * t))
        = (fun x : ℝ => Real.exp (-x)) ∘ (fun t : ℝ => 2 * t) := by
      funext t; simp [Function.comp_def, neg_mul]
    rw [key]
    exact Real.tendsto_exp_neg_atTop_nhds_zero.comp h2
  have hmain :
      Filter.Tendsto
        (fun t : ℝ => Real.exp (-2 * t) * α ^ (Nat.dist i.val j.val)
          + (1 - Real.exp (-2 * t)) * (if i = j then (1:ℝ) else 0))
        Filter.atTop (nhds (0 * α ^ (Nat.dist i.val j.val)
          + (1 - 0) * (if i = j then (1:ℝ) else 0))) :=
    (hexp.mul_const _).add (((tendsto_const_nhds.sub hexp)).mul_const _)
  simp only [ch05a_g_Sigt_entrywise, ch05a_Delta, Matrix.one_apply]
  simpa using hmain

-- eq:g-Qt-def (lines 752-755): Q_t := Σ_t^{-1}.  Definition.
def ch05a_Qt (n : ℕ) (α t : ℝ) : Matrix (Fin n) (Fin n) ℝ := (ch05a_Sigt n α t)⁻¹

-- noname-26 (lines 757-759): at t = 0, Q_t = Q₀.
theorem ch05a_noname26 (n : ℕ) (α : ℝ) : ch05a_Qt n α 0 = (ch05a_Sig0 n α)⁻¹ := by
  rw [ch05a_Qt, ch05a_g_Sigt_limit_zero]

theorem ch05a_noname26_three (α : ℝ) (hα : (1 : ℝ) - α ^ 2 ≠ 0) :
    ch05a_Qt 3 α 0 = ch05a_Q0 3 α := by
  rw [ch05a_noname26]
  exact Matrix.inv_eq_left_inv (ch05a_g_Q0_three α hα)

-- eq:g-locality-loss-summary (lines 768-780): Q₀ is tridiagonal but Q_t is dense for α ≠ 0.
-- Instance: L = 3, α = 1/2, e^{-2t} = 1/2.  Then (Q₀)_{02} = 0 but (Q_t)_{02} = -1/14 ≠ 0.
theorem ch05a_Sigt_instance {t : ℝ} (ht : Real.exp (-2 * t) = 1 / 2) :
    ch05a_Sigt 3 ((1:ℝ)/2) t = !![1, 1/4, 1/8; 1/4, 1, 1/4; 1/8, 1/4, 1] := by
  have ht' : Real.exp (-(2 * t)) = 1 / 2 := by rw [← neg_mul]; exact ht
  ext i j
  rw [ch05a_g_Sigt_entrywise]
  fin_cases i <;> fin_cases j <;> norm_num [ht, ht', ch05a_Delta, Nat.dist]

theorem ch05a_Qt_instance_inv :
    ((!![1, (1:ℝ)/4, 1/8; 1/4, 1, 1/4; 1/8, 1/4, 1] : Matrix (Fin 3) (Fin 3) ℝ))⁻¹
      = !![15/14, -1/4, -1/14; -1/4, 9/8, -1/4; -1/14, -1/4, 15/14] := by
  apply Matrix.inv_eq_right_inv
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.one_apply]

theorem ch05a_g_locality_loss_summary {t : ℝ} (ht : Real.exp (-2 * t) = 1 / 2) :
    ch05a_Q0 3 ((1:ℝ)/2) 0 2 = 0 ∧ ch05a_Qt 3 ((1:ℝ)/2) t 0 2 ≠ 0 := by
  constructor
  · exact ch05a_g_precision_distant_zero _ 0 2 (by norm_num [Nat.dist])
  · rw [ch05a_Qt, ch05a_Sigt_instance ht, ch05a_Qt_instance_inv]
    norm_num [Matrix.cons_val_two, Matrix.vecHead, Matrix.vecTail]

/-! ### The clean joint density (eq:g-P0) -/

-- eq:g-P0 (lines 235-241): N(a₀;0,1) ∏ N(a_{k+1}; α a_k, σ_η²) = N(a; 0, Σ₀),
-- checked in full (normalisation and exponent) at L = 2.
theorem ch05a_g_P0_two (α a0 a1 : ℝ) (hσ : 0 < ch05a_sigEta2 α) :
    ch05a_gauss 0 1 a0 * ch05a_gauss (α * a0) (ch05a_sigEta2 α) a1
      = Real.exp (-(1/2) * ((![a0, a1] : Fin 2 → ℝ) ⬝ᵥ (Matrix.mulVec (ch05a_Q0 2 α) ![a0, a1])))
        / ((2 * Real.pi) * Real.sqrt (ch05a_sigEta2 α)) := by
  have hσ' : ch05a_sigEta2 α ≠ 0 := ne_of_gt hσ
  have hs : (1:ℝ) - α ^ 2 ≠ 0 := by simpa [ch05a_sigEta2] using hσ'
  have hq : (![a0, a1] : Fin 2 → ℝ) ⬝ᵥ (Matrix.mulVec (ch05a_Q0 2 α) ![a0, a1])
      = (a0 ^ 2 - 2 * α * a0 * a1 + a1 ^ 2) / ch05a_sigEta2 α := by
    simp [dotProduct, Matrix.mulVec, ch05a_Q0, ch05a_sigEta2, Fin.sum_univ_two, Nat.dist]
    field_simp
    ring
  rw [ch05a_gauss, ch05a_gauss, hq, div_mul_div_comm, ← Real.exp_add]
  congr 1
  · simp only [ch05a_sigEta2] at hσ' ⊢
    field_simp
    ring
  · rw [← Real.sqrt_mul (by positivity),
      show (2 * Real.pi * 1) * (2 * Real.pi * ch05a_sigEta2 α)
        = (2 * Real.pi) ^ 2 * ch05a_sigEta2 α by ring,
      Real.sqrt_mul (by positivity), Real.sqrt_sq (by positivity)]

/-! ### Pass-2 cross-checks

These lemmas tie the entrywise encodings `ch05a_Q0` / `ch05a_Sig0` back to the literal
matrices displayed in the text, and check the quadratic-form bridge between
\eqref{eq:g-precision-expand-1} and \eqref{eq:g-Q0}. -/

-- eq:g-Q0-K2 (lines 286-293): the general entrywise Q₀ agrees with the L = 2 display.
theorem ch05a_g_Q0_K2_matches (α : ℝ) :
    ch05a_Q0 2 α = (ch05a_sigEta2 α)⁻¹ • !![1, -α; -α, 1] := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [ch05a_Q0, Nat.dist, div_eq_inv_mul]

-- eq:g-Q0-K3 (lines 298-311): the general entrywise Q₀ agrees with the L = 3 display,
-- including the exactly-zero corner and the extra α² on the interior diagonal.
theorem ch05a_g_Q0_K3_matches (α : ℝ) :
    ch05a_Q0 3 α = (ch05a_sigEta2 α)⁻¹ • !![1, -α, 0; -α, 1 + α ^ 2, -α; 0, -α, 1] := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [ch05a_Q0, Nat.dist, div_eq_inv_mul]

-- eq:g-Q0 (lines 320-334): the boxed tridiagonal display, checked at L = 4.
theorem ch05a_g_Q0_matrix_four (α : ℝ) :
    ch05a_Q0 4 α = (ch05a_sigEta2 α)⁻¹ •
      !![1, -α, 0, 0; -α, 1 + α ^ 2, -α, 0; 0, -α, 1 + α ^ 2, -α; 0, 0, -α, 1] := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [ch05a_Q0, Nat.dist, div_eq_inv_mul]

-- eq:g-Q0-K3 / eq:g-Sig0-matrix: the L = 3 covariance display.
theorem ch05a_g_Sig0_K3_matches (α : ℝ) :
    ch05a_Sig0 3 α = !![1, α, α ^ 2; α, 1, α; α ^ 2, α, 1] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [ch05a_Sig0, Nat.dist]

-- The bridge between eq:g-precision-expand-1 and eq:g-Q0:
-- a₀² + (1/σ_η²) Σ_{k=0}^{L-2} (a_{k+1} - α a_k)² = aᵀ Q₀ a.  Checked at L = 2, 3, 4.
theorem ch05a_quadform_two (α a0 a1 : ℝ) (hσ : (1:ℝ) - α ^ 2 ≠ 0) :
    a0 ^ 2 + (1 / ch05a_sigEta2 α) * (a1 - α * a0) ^ 2
      = (![a0, a1] : Fin 2 → ℝ) ⬝ᵥ (Matrix.mulVec (ch05a_Q0 2 α) ![a0, a1]) := by
  have hrhs : (![a0, a1] : Fin 2 → ℝ) ⬝ᵥ (Matrix.mulVec (ch05a_Q0 2 α) ![a0, a1])
      = (a0 ^ 2 - 2 * α * a0 * a1 + a1 ^ 2) / ch05a_sigEta2 α := by
    simp [dotProduct, Matrix.mulVec, ch05a_Q0, ch05a_sigEta2, Fin.sum_univ_two, Nat.dist]
    field_simp
    ring
  rw [hrhs, ch05a_sigEta2]
  field_simp
  ring

theorem ch05a_quadform_three (α a0 a1 a2 : ℝ) (hσ : (1:ℝ) - α ^ 2 ≠ 0) :
    a0 ^ 2 + (1 / ch05a_sigEta2 α) * ((a1 - α * a0) ^ 2 + (a2 - α * a1) ^ 2)
      = (![a0, a1, a2] : Fin 3 → ℝ) ⬝ᵥ (Matrix.mulVec (ch05a_Q0 3 α) ![a0, a1, a2]) := by
  have hrhs : (![a0, a1, a2] : Fin 3 → ℝ) ⬝ᵥ (Matrix.mulVec (ch05a_Q0 3 α) ![a0, a1, a2])
      = (a0 ^ 2 + (1 + α ^ 2) * a1 ^ 2 + a2 ^ 2 - 2 * α * a0 * a1 - 2 * α * a1 * a2)
          / ch05a_sigEta2 α := by
    simp [dotProduct, Matrix.mulVec, ch05a_Q0, ch05a_sigEta2, Fin.sum_univ_three, Nat.dist]
    field_simp
    ring
  rw [hrhs, ch05a_sigEta2]
  field_simp
  ring

theorem ch05a_quadform_four (α a0 a1 a2 a3 : ℝ) (hσ : (1:ℝ) - α ^ 2 ≠ 0) :
    a0 ^ 2 + (1 / ch05a_sigEta2 α)
        * ((a1 - α * a0) ^ 2 + (a2 - α * a1) ^ 2 + (a3 - α * a2) ^ 2)
      = (![a0, a1, a2, a3] : Fin 4 → ℝ)
          ⬝ᵥ (Matrix.mulVec (ch05a_Q0 4 α) ![a0, a1, a2, a3]) := by
  have hrhs : (![a0, a1, a2, a3] : Fin 4 → ℝ)
        ⬝ᵥ (Matrix.mulVec (ch05a_Q0 4 α) ![a0, a1, a2, a3])
      = (a0 ^ 2 + (1 + α ^ 2) * a1 ^ 2 + (1 + α ^ 2) * a2 ^ 2 + a3 ^ 2
          - 2 * α * a0 * a1 - 2 * α * a1 * a2 - 2 * α * a2 * a3) / ch05a_sigEta2 α := by
    simp [dotProduct, Matrix.mulVec, ch05a_Q0, ch05a_sigEta2, Fin.sum_univ_four, Nat.dist]
    field_simp
    ring
  rw [hrhs, ch05a_sigEta2]
  field_simp
  ring

-- eq:g-linear-map (lines 120-127), general L: a = G_α ε with G_α lower triangular.
theorem ch05a_g_linear_map {n : ℕ} (α a0 : ℝ) (η : ℕ → ℝ) (i : Fin n) :
    ∑ j : Fin n, ch05a_G n α i j * ch05a_eps a0 η (j : ℕ) = ch05a_ar1 α a0 η (i : ℕ) := by
  have step1 : ∑ j : Fin n, ch05a_G n α i j * ch05a_eps a0 η (j : ℕ)
      = ∑ j ∈ Finset.range n,
          (if j ≤ (i : ℕ) then α ^ ((i : ℕ) - j) else 0) * ch05a_eps a0 η j :=
    Fin.sum_univ_eq_sum_range
      (fun k : ℕ => (if k ≤ (i : ℕ) then α ^ ((i : ℕ) - k) else 0) * ch05a_eps a0 η k) n
  have step2 : ∑ j ∈ Finset.range n,
        (if j ≤ (i : ℕ) then α ^ ((i : ℕ) - j) else 0) * ch05a_eps a0 η j
      = ∑ j ∈ Finset.range ((i : ℕ) + 1),
        (if j ≤ (i : ℕ) then α ^ ((i : ℕ) - j) else 0) * ch05a_eps a0 η j := by
    refine (Finset.sum_subset ?_ ?_).symm
    · intro x hx
      rw [Finset.mem_range] at hx ⊢
      have := i.isLt
      omega
    · intro x hx hnx
      rw [Finset.mem_range] at hx hnx
      rw [if_neg (by omega), zero_mul]
  have step3 : ∑ j ∈ Finset.range ((i : ℕ) + 1),
        (if j ≤ (i : ℕ) then α ^ ((i : ℕ) - j) else 0) * ch05a_eps a0 η j
      = ∑ j ∈ Finset.range ((i : ℕ) + 1), α ^ ((i : ℕ) - j) * ch05a_eps a0 η j := by
    refine Finset.sum_congr rfl fun j hj => ?_
    rw [Finset.mem_range] at hj
    rw [if_pos (by omega)]
  rw [step1, step2, step3, Finset.sum_range_succ', ch05a_g_unroll]
  simp only [ch05a_eps, Nat.sub_zero]
  have h4 : ∑ j ∈ Finset.range (i : ℕ), α ^ ((i : ℕ) - (j + 1)) * η j
          = ∑ j ∈ Finset.range (i : ℕ), α ^ ((i : ℕ) - 1 - j) * η j := by
    refine Finset.sum_congr rfl fun j hj => ?_
    rw [Finset.mem_range] at hj
    have : (i : ℕ) - (j + 1) = (i : ℕ) - 1 - j := by omega
    rw [this]
  rw [h4]
  ring


-- eq:g-general-quadratic-form (lines 418-426), general L:
-- aᵀQa = Σ_m Q_mm a_m² + 2 Σ_{m<l} Q_ml a_m a_l for a symmetric Q.
theorem ch05a_g_general_quadratic_form {n : ℕ} (Q : Matrix (Fin n) (Fin n) ℝ)
    (hQ : ∀ i j, Q j i = Q i j) (a : Fin n → ℝ) :
    a ⬝ᵥ (Matrix.mulVec Q a)
      = ∑ m : Fin n, Q m m * a m ^ 2
        + 2 * ∑ m : Fin n, ∑ l ∈ Finset.univ.filter (fun l => m < l), Q m l * a m * a l := by
  classical
  have hexp : a ⬝ᵥ (Matrix.mulVec Q a) = ∑ i : Fin n, ∑ j : Fin n, Q i j * a i * a j := by
    simp only [dotProduct, Matrix.mulVec, Finset.mul_sum]
    exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by ring
  have hdiag : ∀ i : Fin n, ∑ j : Fin n, Q i j * a i * a j
      = Q i i * a i ^ 2 + ∑ j ∈ Finset.univ.erase i, Q i j * a i * a j := by
    intro i
    rw [← Finset.add_sum_erase _ (fun j => Q i j * a i * a j) (Finset.mem_univ i)]
    congr 1
    ring
  have herase : ∀ i : Fin n, ∑ j ∈ Finset.univ.erase i, Q i j * a i * a j
      = (∑ j ∈ Finset.univ.filter (fun j => i < j), Q i j * a i * a j)
        + ∑ j ∈ Finset.univ.filter (fun j => j < i), Q i j * a i * a j := by
    intro i
    have e1 : (Finset.univ.erase i).filter (fun j => i < j)
        = Finset.univ.filter (fun j => i < j) := by
      ext j
      simp only [Finset.mem_filter, Finset.mem_erase, Finset.mem_univ, and_true, true_and]
      exact ⟨fun h => h.2, fun h => ⟨ne_of_gt h, h⟩⟩
    have e2 : (Finset.univ.erase i).filter (fun j => ¬ i < j)
        = Finset.univ.filter (fun j => j < i) := by
      ext j
      simp only [Finset.mem_filter, Finset.mem_erase, Finset.mem_univ, and_true, true_and]
      constructor
      · rintro ⟨hne, hnlt⟩
        exact lt_of_le_of_ne (not_lt.mp hnlt) hne
      · intro h
        exact ⟨ne_of_lt h, not_lt.mpr (le_of_lt h)⟩
    rw [← Finset.sum_filter_add_sum_filter_not (Finset.univ.erase i) (fun j => i < j), e1, e2]
  have hswap : ∑ i : Fin n, ∑ j ∈ Finset.univ.filter (fun j => j < i), Q i j * a i * a j
      = ∑ m : Fin n, ∑ l ∈ Finset.univ.filter (fun l => m < l), Q m l * a m * a l := by
    rw [Finset.sum_comm' (s := (Finset.univ : Finset (Fin n)))
      (t := fun i => Finset.univ.filter (fun j => j < i))
      (t' := (Finset.univ : Finset (Fin n)))
      (s' := fun j => Finset.univ.filter (fun i => j < i))
      (by intro x y; simp)]
    refine Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun i _ => ?_
    rw [hQ]
    ring
  rw [hexp, Finset.sum_congr rfl (fun i _ => hdiag i),
    Finset.sum_add_distrib, Finset.sum_congr rfl (fun i _ => herase i),
    Finset.sum_add_distrib, hswap]
  ring

-- eq:g-markov-hessian-sparsity (lines 536-542), a more faithful instance: for a
-- first-order chain the negative log-density in three consecutive coordinates is
-- g(x,y) + h(y,z); with the middle coordinate fixed the mixed partial in x and z vanishes.
theorem ch05a_g_markov_hessian_sparsity_chain (g h : ℝ → ℝ → ℝ) (x0 y0 z0 : ℝ) :
    deriv (fun z => deriv (fun x => g x y0 + h y0 z) x0) z0 = 0 := by
  have hc : (fun z => deriv (fun x => g x y0 + h y0 z) x0)
      = fun _ => deriv (fun x => g x y0) x0 := by
    funext z
    simp [deriv_add_const]
  rw [hc, deriv_const]


/-! ### Pass-3: general-`L` theorems

The fixed-size instance checks above are superseded here by theorems stated for
an arbitrary chain length `n`.  The engine is the bidiagonal factorisation of
the AR(1) chain.  With

* `B` the differencing matrix, `(B a)_k = a_k - α a_{k-1}` (unit diagonal, `-α`
  on the subdiagonal) — one row per factor of the Markov factorisation
  \eqref{eq:g-P0};
* `G` the unrolling matrix `(G)_{ij} = α^{i-j}` (`j ≤ i`) of
  \eqref{eq:g-unroll}, which turns out to be exactly `B⁻¹`;
* `D = diag(1, σ_η², …, σ_η²)`, the covariance of `ε = (a₀, η₀, …, η_{L-2})`,

one gets `Σ₀ = G D Gᵀ` and `Q₀ = Bᵀ D⁻¹ B`, and from those two identities every
claim the chapter makes about `Σ₀` and `Q₀` follows at once for every `L`.

Size hypothesis: \eqref{eq:g-Q0} describes a chain with at least one
transition, so `2 ≤ L` is assumed wherever `Q₀` is claimed to be `Σ₀⁻¹`.  At
`L = 1` the displayed formula gives `(Q₀)_{00} = 1/σ_η²`, which is not the
inverse of `Σ₀ = (1)`; the text never uses that degenerate case. -/

/-- The differencing matrix `B = I - α S`: `B_{ii} = 1`, `B_{i,i-1} = -α`.
`(B a)_k = a_k - α a_{k-1}` is the `k`-th innovation of \eqref{eq:g-recursion}. -/
def ch05a_B (n : ℕ) (α : ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  fun i j => if (i : ℕ) = (j : ℕ) then 1 else if (j : ℕ) + 1 = (i : ℕ) then -α else 0

/-- `D = diag(1, σ_η², …, σ_η²)`: the covariance of `ε = (a₀, η₀, …, η_{L-2})`. -/
def ch05a_Dcov (n : ℕ) (α : ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  Matrix.diagonal fun i => if (i : ℕ) = 0 then 1 else ch05a_sigEta2 α

/-- `D⁻¹ = diag(1, 1/σ_η², …, 1/σ_η²)`. -/
def ch05a_Dprec (n : ℕ) (α : ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  Matrix.diagonal fun i => if (i : ℕ) = 0 then 1 else (ch05a_sigEta2 α)⁻¹

/-- Entrywise `Q₀` with every test written in `ℕ`-arithmetic (so `omega` can see it). -/
theorem ch05a_Q0_val {n : ℕ} (α : ℝ) (i k : Fin n) :
    ch05a_Q0 n α i k
      = if (i : ℕ) = (k : ℕ) then
          (if (i : ℕ) = 0 ∨ (i : ℕ) = n - 1 then 1 else 1 + α ^ 2) / ch05a_sigEta2 α
        else if (i : ℕ) - (k : ℕ) + ((k : ℕ) - (i : ℕ)) = 1 then -α / ch05a_sigEta2 α else 0 := by
  by_cases h : (i : ℕ) = (k : ℕ)
  · have hik : i = k := by ext; omega
    rw [if_pos h]
    simp only [ch05a_Q0]
    rw [if_pos hik]
  · have hik : i ≠ k := fun hh => h (by rw [hh])
    rw [if_neg h]
    simp only [ch05a_Q0, Nat.dist]
    rw [if_neg hik]
    rfl

/-! Row and column expansions of the bidiagonal `B`. -/

theorem ch05a_B_row_zero {n : ℕ} (α : ℝ) (g : Fin n → ℝ) (i : Fin n) (hi : (i : ℕ) = 0) :
    ∑ j : Fin n, ch05a_B n α i j * g j = g i := by
  classical
  refine (Finset.sum_eq_single i ?_ ?_).trans ?_
  · intro b _ hb
    have h1 : ¬ ((i : ℕ) = (b : ℕ)) := by
      intro hh; exact hb (by ext; omega)
    have h2 : ¬ ((b : ℕ) + 1 = (i : ℕ)) := by omega
    simp only [ch05a_B, if_neg h1, if_neg h2, zero_mul]
  · intro hh; exact absurd (Finset.mem_univ i) hh
  · simp [ch05a_B]

theorem ch05a_B_row_succ {n : ℕ} (α : ℝ) (g : Fin n → ℝ) (i p : Fin n)
    (hp : (p : ℕ) + 1 = (i : ℕ)) :
    ∑ j : Fin n, ch05a_B n α i j * g j = g i - α * g p := by
  classical
  have hpi : p ≠ i := by
    intro hh; rw [hh] at hp; omega
  have hmem : p ∈ Finset.univ.erase i := Finset.mem_erase.mpr ⟨hpi, Finset.mem_univ _⟩
  have hBii : ch05a_B n α i i = 1 := by simp [ch05a_B]
  have hBip : ch05a_B n α i p = -α := by
    simp only [ch05a_B]
    rw [if_neg (show ¬ ((i : ℕ) = (p : ℕ)) by omega), if_pos hp]
  have hrest : ∑ j ∈ (Finset.univ.erase i).erase p, ch05a_B n α i j * g j = 0 := by
    refine Finset.sum_eq_zero ?_
    intro b hb
    rw [Finset.mem_erase, Finset.mem_erase] at hb
    obtain ⟨hbp, hbi, -⟩ := hb
    have h1 : ¬ ((i : ℕ) = (b : ℕ)) := by
      intro hh; exact hbi (by ext; omega)
    have h2 : ¬ ((b : ℕ) + 1 = (i : ℕ)) := by
      intro hh; exact hbp (by ext; omega)
    simp only [ch05a_B, if_neg h1, if_neg h2, zero_mul]
  rw [← Finset.add_sum_erase _ (fun j => ch05a_B n α i j * g j) (Finset.mem_univ i),
    ← Finset.add_sum_erase _ (fun j => ch05a_B n α i j * g j) hmem]
  simp only [hrest, hBii, hBip, add_zero, one_mul]
  ring

theorem ch05a_Bt_col_last {n : ℕ} (α : ℝ) (g : Fin n → ℝ) (i : Fin n) (hi : (i : ℕ) + 1 = n) :
    ∑ j : Fin n, ch05a_B n α j i * g j = g i := by
  classical
  refine (Finset.sum_eq_single i ?_ ?_).trans ?_
  · intro b _ hb
    have h1 : ¬ ((b : ℕ) = (i : ℕ)) := by
      intro hh; exact hb (by ext; omega)
    have h2 : ¬ ((i : ℕ) + 1 = (b : ℕ)) := by
      have := b.isLt; omega
    simp only [ch05a_B, if_neg h1, if_neg h2, zero_mul]
  · intro hh; exact absurd (Finset.mem_univ i) hh
  · simp [ch05a_B]

theorem ch05a_Bt_col_succ {n : ℕ} (α : ℝ) (g : Fin n → ℝ) (i q : Fin n)
    (hq : (i : ℕ) + 1 = (q : ℕ)) :
    ∑ j : Fin n, ch05a_B n α j i * g j = g i - α * g q := by
  classical
  have hqi : q ≠ i := by
    intro hh; rw [hh] at hq; omega
  have hmem : q ∈ Finset.univ.erase i := Finset.mem_erase.mpr ⟨hqi, Finset.mem_univ _⟩
  have hBii : ch05a_B n α i i = 1 := by simp [ch05a_B]
  have hBqi : ch05a_B n α q i = -α := by
    simp only [ch05a_B]
    rw [if_neg (show ¬ ((q : ℕ) = (i : ℕ)) by omega), if_pos hq]
  have hrest : ∑ j ∈ (Finset.univ.erase i).erase q, ch05a_B n α j i * g j = 0 := by
    refine Finset.sum_eq_zero ?_
    intro b hb
    rw [Finset.mem_erase, Finset.mem_erase] at hb
    obtain ⟨hbq, hbi, -⟩ := hb
    have h1 : ¬ ((b : ℕ) = (i : ℕ)) := by
      intro hh; exact hbi (by ext; omega)
    have h2 : ¬ ((i : ℕ) + 1 = (b : ℕ)) := by
      intro hh; exact hbq (by ext; omega)
    simp only [ch05a_B, if_neg h1, if_neg h2, zero_mul]
  rw [← Finset.add_sum_erase _ (fun j => ch05a_B n α j i * g j) (Finset.mem_univ i),
    ← Finset.add_sum_erase _ (fun j => ch05a_B n α j i * g j) hmem]
  simp only [hrest, hBii, hBqi, add_zero, one_mul]
  ring

/-- `G = B⁻¹`: differencing undoes unrolling.  This is \eqref{eq:g-unroll} in
matrix form, for every `L`. -/
theorem ch05a_B_mul_G (n : ℕ) (α : ℝ) : ch05a_B n α * ch05a_G n α = 1 := by
  classical
  ext i k
  rw [Matrix.mul_apply, Matrix.one_apply]
  have hone : (if i = k then (1 : ℝ) else 0) = (if (i : ℕ) = (k : ℕ) then (1 : ℝ) else 0) := by
    by_cases h : (i : ℕ) = (k : ℕ)
    · rw [if_pos h, if_pos (show i = k by ext; omega)]
    · rw [if_neg h, if_neg (show ¬ (i = k) from fun hh => h (by rw [hh]))]
  rw [hone]
  rcases Nat.eq_zero_or_pos (i : ℕ) with h0 | h0
  · rw [ch05a_B_row_zero α (fun j => ch05a_G n α j k) i h0]
    simp only [ch05a_G]
    rcases Nat.eq_zero_or_pos (k : ℕ) with hk | hk
    · rw [if_pos (show (k : ℕ) ≤ (i : ℕ) by omega),
        if_pos (show (i : ℕ) = (k : ℕ) by omega),
        show (i : ℕ) - (k : ℕ) = 0 by omega, pow_zero]
    · rw [if_neg (show ¬ ((k : ℕ) ≤ (i : ℕ)) by omega),
        if_neg (show ¬ ((i : ℕ) = (k : ℕ)) by omega)]
  · obtain ⟨m, hm⟩ : ∃ m, (i : ℕ) = m + 1 := ⟨(i : ℕ) - 1, by omega⟩
    have hmn : m < n := by have := i.isLt; omega
    rw [ch05a_B_row_succ α (fun j => ch05a_G n α j k) i ⟨m, hmn⟩ hm.symm]
    have hpv : ((⟨m, hmn⟩ : Fin n) : ℕ) = m := rfl
    simp only [ch05a_G, hpv]
    rcases Nat.lt_trichotomy ((k : ℕ)) ((i : ℕ)) with hk | hk | hk
    · rw [if_pos (show (k : ℕ) ≤ (i : ℕ) by omega), if_pos (show (k : ℕ) ≤ m by omega),
        if_neg (show ¬ ((i : ℕ) = (k : ℕ)) by omega),
        show (i : ℕ) - (k : ℕ) = (m - (k : ℕ)) + 1 by omega, pow_succ]
      ring
    · rw [if_pos (show (k : ℕ) ≤ (i : ℕ) by omega), if_neg (show ¬ ((k : ℕ) ≤ m) by omega),
        if_pos (show (i : ℕ) = (k : ℕ) by omega),
        show (i : ℕ) - (k : ℕ) = 0 by omega, pow_zero]
      ring
    · rw [if_neg (show ¬ ((k : ℕ) ≤ (i : ℕ)) by omega),
        if_neg (show ¬ ((k : ℕ) ≤ m) by omega),
        if_neg (show ¬ ((i : ℕ) = (k : ℕ)) by omega)]
      ring

/-- `B` is lower triangular with unit diagonal, so `det B = 1`. -/
theorem ch05a_B_det (n : ℕ) (α : ℝ) : (ch05a_B n α).det = 1 := by
  classical
  have htri : (ch05a_B n α).IsLowerTriangular := by
    intro i j hij
    have hlt : (i : ℕ) < (j : ℕ) := hij
    simp only [ch05a_B]
    rw [if_neg (show ¬ ((i : ℕ) = (j : ℕ)) by omega),
      if_neg (show ¬ ((j : ℕ) + 1 = (i : ℕ)) by omega)]
  rw [Matrix.det_of_isLowerTriangular _ htri]
  simp [ch05a_B]

theorem ch05a_G_mul_B (n : ℕ) (α : ℝ) : ch05a_G n α * ch05a_B n α = 1 := by
  have hinv : (ch05a_B n α)⁻¹ = ch05a_G n α := Matrix.inv_eq_right_inv (ch05a_B_mul_G n α)
  have hu : IsUnit (ch05a_B n α).det := by rw [ch05a_B_det]; exact isUnit_one
  have hmul := Matrix.nonsing_inv_mul (ch05a_B n α) hu
  rwa [hinv] at hmul

/-- The geometric block sum behind `Σ₀ = G D Gᵀ`: the `(i,k)` entry with `i ≤ k`. -/
theorem ch05a_geom_block (α : ℝ) (I K : ℕ) (hIK : I ≤ K) :
    ∑ j ∈ Finset.range (I + 1),
        (if j ≤ I then α ^ (I - j) else 0) * (if j = 0 then 1 else ch05a_sigEta2 α)
          * (if j ≤ K then α ^ (K - j) else 0)
      = α ^ (K - I) := by
  have step1 :
      ∑ j ∈ Finset.range (I + 1),
          (if j ≤ I then α ^ (I - j) else 0) * (if j = 0 then 1 else ch05a_sigEta2 α)
            * (if j ≤ K then α ^ (K - j) else 0)
        = (∑ j ∈ Finset.range I,
            (if j + 1 ≤ I then α ^ (I - (j + 1)) else 0)
              * (if j + 1 = 0 then 1 else ch05a_sigEta2 α)
              * (if j + 1 ≤ K then α ^ (K - (j + 1)) else 0))
          + ((if 0 ≤ I then α ^ (I - 0) else 0)
              * (if (0 : ℕ) = 0 then 1 else ch05a_sigEta2 α)
              * (if 0 ≤ K then α ^ (K - 0) else 0)) :=
    Finset.sum_range_succ'
      (fun j => (if j ≤ I then α ^ (I - j) else 0) * (if j = 0 then 1 else ch05a_sigEta2 α)
        * (if j ≤ K then α ^ (K - j) else 0)) I
  have step2 :
      ∑ j ∈ Finset.range I,
          (if j + 1 ≤ I then α ^ (I - (j + 1)) else 0)
            * (if j + 1 = 0 then 1 else ch05a_sigEta2 α)
            * (if j + 1 ≤ K then α ^ (K - (j + 1)) else 0)
        = ∑ j ∈ Finset.range I, ch05a_sigEta2 α * (α ^ (K - I) * (α ^ 2) ^ (I - 1 - j)) := by
    refine Finset.sum_congr rfl ?_
    intro j hj
    rw [Finset.mem_range] at hj
    rw [if_pos (show j + 1 ≤ I by omega), if_neg (show ¬ (j + 1 = 0) by omega),
      if_pos (show j + 1 ≤ K by omega), ← pow_mul, ← pow_add]
    rw [show (K - I) + 2 * (I - 1 - j) = (I - (j + 1)) + (K - (j + 1)) by omega, pow_add]
    ring
  have step3 :
      ∑ j ∈ Finset.range I, ch05a_sigEta2 α * (α ^ (K - I) * (α ^ 2) ^ (I - 1 - j))
        = ch05a_sigEta2 α * (α ^ (K - I) * ∑ j ∈ Finset.range I, (α ^ 2) ^ (I - 1 - j)) := by
    rw [← Finset.mul_sum, ← Finset.mul_sum]
  have step4 : ∑ j ∈ Finset.range I, (α ^ 2) ^ (I - 1 - j) = ∑ j ∈ Finset.range I, (α ^ 2) ^ j :=
    Finset.sum_range_reflect (fun j => (α ^ 2) ^ j) I
  have hgeom : ch05a_sigEta2 α * ∑ j ∈ Finset.range I, (α ^ 2) ^ j = 1 - (α ^ 2) ^ I := by
    simp only [ch05a_sigEta2]
    exact mul_neg_geom_sum (α ^ 2) I
  have hfin : ch05a_sigEta2 α * (α ^ (K - I) * ∑ j ∈ Finset.range I, (α ^ 2) ^ j)
      = α ^ (K - I) * (1 - (α ^ 2) ^ I) := by
    rw [← hgeom]; ring
  have h2 : α ^ (K - I) * (α ^ 2) ^ I = α ^ I * α ^ K := by
    rw [← pow_mul, ← pow_add, ← pow_add]
    congr 1
    omega
  rw [step1, step2, step3, step4, hfin, if_pos (Nat.zero_le I), if_pos (rfl : (0 : ℕ) = 0),
    if_pos (Nat.zero_le K), Nat.sub_zero, Nat.sub_zero, mul_sub, mul_one, h2]
  ring

theorem ch05a_G_sum {n : ℕ} (α : ℝ) (i k : Fin n) (hik : (i : ℕ) ≤ (k : ℕ)) :
    ∑ j : Fin n,
        ch05a_G n α i j * (if (j : ℕ) = 0 then 1 else ch05a_sigEta2 α) * ch05a_G n α k j
      = α ^ ((k : ℕ) - (i : ℕ)) := by
  classical
  have hIn : (i : ℕ) + 1 ≤ n := i.isLt
  have e1 : ∑ j : Fin n,
        ch05a_G n α i j * (if (j : ℕ) = 0 then 1 else ch05a_sigEta2 α) * ch05a_G n α k j
      = ∑ j ∈ Finset.range n,
        (if j ≤ (i : ℕ) then α ^ ((i : ℕ) - j) else 0)
          * (if j = 0 then 1 else ch05a_sigEta2 α)
          * (if j ≤ (k : ℕ) then α ^ ((k : ℕ) - j) else 0) := by
    simp only [ch05a_G]
    exact Fin.sum_univ_eq_sum_range
      (fun j => (if j ≤ (i : ℕ) then α ^ ((i : ℕ) - j) else 0)
        * (if j = 0 then 1 else ch05a_sigEta2 α)
        * (if j ≤ (k : ℕ) then α ^ ((k : ℕ) - j) else 0)) n
  have e2 : ∑ j ∈ Finset.range n,
        (if j ≤ (i : ℕ) then α ^ ((i : ℕ) - j) else 0)
          * (if j = 0 then 1 else ch05a_sigEta2 α)
          * (if j ≤ (k : ℕ) then α ^ ((k : ℕ) - j) else 0)
      = ∑ j ∈ Finset.range ((i : ℕ) + 1),
        (if j ≤ (i : ℕ) then α ^ ((i : ℕ) - j) else 0)
          * (if j = 0 then 1 else ch05a_sigEta2 α)
          * (if j ≤ (k : ℕ) then α ^ ((k : ℕ) - j) else 0) := by
    refine (Finset.sum_subset (Finset.range_subset_range.mpr hIn) ?_).symm
    intro x hx hnx
    rw [Finset.mem_range] at hx hnx
    rw [if_neg (show ¬ (x ≤ (i : ℕ)) by omega), zero_mul, zero_mul]
  rw [e1, e2, ch05a_geom_block α _ _ hik]

/-- **`Σ₀ = G D Gᵀ`, every `L`.**  The stationary covariance is the unrolling map
applied to the diagonal covariance of `ε`; equivalently \eqref{eq:g-cov-lag}. -/
theorem ch05a_Sig0_factor (n : ℕ) (α : ℝ) :
    ch05a_Sig0 n α = ch05a_G n α * ch05a_Dcov n α * (ch05a_G n α).transpose := by
  classical
  ext i k
  rw [Matrix.mul_apply]
  simp only [ch05a_Dcov, Matrix.mul_diagonal, Matrix.transpose_apply]
  rcases le_total ((i : ℕ)) ((k : ℕ)) with hik | hik
  · rw [ch05a_G_sum α i k hik]
    simp only [ch05a_Sig0]
    rw [Nat.dist_eq_sub_of_le hik]
  · have hswap : ∑ j : Fin n,
          ch05a_G n α i j * (if (j : ℕ) = 0 then 1 else ch05a_sigEta2 α) * ch05a_G n α k j
        = ∑ j : Fin n,
          ch05a_G n α k j * (if (j : ℕ) = 0 then 1 else ch05a_sigEta2 α) * ch05a_G n α i j :=
      Finset.sum_congr rfl fun j _ => by ring
    rw [hswap, ch05a_G_sum α k i hik]
    simp only [ch05a_Sig0]
    rw [Nat.dist_eq_sub_of_le_right hik]

/-- **`Q₀ = Bᵀ D⁻¹ B`, every `L ≥ 2`.**  The matrix form of the "reading `Q₀` off
`-2 log P₀`" toolbox. -/
theorem ch05a_Q0_factor {n : ℕ} (hn : 2 ≤ n) (α : ℝ) (hs : ch05a_sigEta2 α ≠ 0) :
    ch05a_Q0 n α = (ch05a_B n α).transpose * ch05a_Dprec n α * ch05a_B n α := by
  classical
  ext i k
  rw [Matrix.mul_assoc, Matrix.mul_apply]
  simp only [Matrix.transpose_apply, ch05a_Dprec, Matrix.diagonal_mul]
  have hkn : (k : ℕ) < n := k.isLt
  have hin : (i : ℕ) < n := i.isLt
  by_cases hlast : (i : ℕ) + 1 = n
  · rw [ch05a_Bt_col_last α
      (fun j => (if (j : ℕ) = 0 then 1 else (ch05a_sigEta2 α)⁻¹) * ch05a_B n α j k) i hlast,
      ch05a_Q0_val]
    simp only [ch05a_B, ch05a_sigEta2] at hs ⊢
    split_ifs <;> (try (exfalso; omega)) <;> (try (field_simp <;> ring)) <;>
      (try field_simp) <;> (try ring)
  · obtain ⟨q, hq⟩ : ∃ q : Fin n, (i : ℕ) + 1 = (q : ℕ) :=
      ⟨⟨(i : ℕ) + 1, by omega⟩, rfl⟩
    have hqn : (q : ℕ) < n := q.isLt
    rw [ch05a_Bt_col_succ α
      (fun j => (if (j : ℕ) = 0 then 1 else (ch05a_sigEta2 α)⁻¹) * ch05a_B n α j k) i q hq,
      ch05a_Q0_val]
    simp only [ch05a_B, ch05a_sigEta2] at hs ⊢
    split_ifs <;> (try (exfalso; omega)) <;> (try (field_simp <;> ring)) <;>
      (try field_simp) <;> (try ring)

theorem ch05a_Dprec_mul_Dcov {n : ℕ} (α : ℝ) (hs : ch05a_sigEta2 α ≠ 0) :
    ch05a_Dprec n α * ch05a_Dcov n α = 1 := by
  classical
  ext i j
  simp only [ch05a_Dprec, ch05a_Dcov, Matrix.diagonal_mul_diagonal, Matrix.diagonal_apply,
    Matrix.one_apply]
  by_cases hij : i = j
  · rw [if_pos hij, if_pos hij]
    by_cases h : (i : ℕ) = 0
    · rw [if_pos h, if_pos h, one_mul]
    · rw [if_neg h, if_neg h, inv_mul_cancel₀ hs]
  · rw [if_neg hij, if_neg hij]

/-- **eq:g-Q0, general `L`.**  `Q₀ Σ₀ = I` for every `L ≥ 2` and every `α` with
`σ_η² = 1 - α² ≠ 0`. -/
theorem ch05a_Q0_mul_Sig0 {n : ℕ} (hn : 2 ≤ n) (α : ℝ) (hs : ch05a_sigEta2 α ≠ 0) :
    ch05a_Q0 n α * ch05a_Sig0 n α = 1 := by
  rw [ch05a_Q0_factor hn α hs, ch05a_Sig0_factor n α]
  have e1 : (ch05a_B n α).transpose * ch05a_Dprec n α * ch05a_B n α
        * (ch05a_G n α * ch05a_Dcov n α * (ch05a_G n α).transpose)
      = (ch05a_B n α).transpose * (ch05a_Dprec n α
          * ((ch05a_B n α * ch05a_G n α) * (ch05a_Dcov n α * (ch05a_G n α).transpose))) := by
    simp only [Matrix.mul_assoc]
  have e2 : ch05a_Dprec n α * (ch05a_Dcov n α * (ch05a_G n α).transpose) = (ch05a_G n α).transpose := by
    rw [← Matrix.mul_assoc, ch05a_Dprec_mul_Dcov α hs, Matrix.one_mul]
  rw [e1, ch05a_B_mul_G, Matrix.one_mul, e2, ← Matrix.transpose_mul, ch05a_G_mul_B,
    Matrix.transpose_one]

/-- **eq:g-Q0, general `L`.**  `Σ₀⁻¹` is exactly the tridiagonal matrix displayed
in \eqref{eq:g-Q0}: `1/σ_η²` at the two ends, `(1+α²)/σ_η²` inside, `-α/σ_η²` on
the two off-diagonals and exact zeros at distance `≥ 2`. -/
theorem ch05a_g_Q0_general {n : ℕ} (hn : 2 ≤ n) (α : ℝ) (hs : ch05a_sigEta2 α ≠ 0) :
    (ch05a_Sig0 n α)⁻¹ = ch05a_Q0 n α :=
  Matrix.inv_eq_left_inv (ch05a_Q0_mul_Sig0 hn α hs)

/-- `det Σ₀ = (σ_η²)^{L-1}` for every `L`. -/
theorem ch05a_Sig0_det (n : ℕ) (α : ℝ) :
    (ch05a_Sig0 n α).det = ch05a_sigEta2 α ^ (n - 1) := by
  classical
  have htri : (ch05a_G n α).IsLowerTriangular := by
    intro i j hij
    exact ch05a_G_lower_triangular α i j hij
  have hG : (ch05a_G n α).det = 1 := by
    rw [Matrix.det_of_isLowerTriangular _ htri]
    simp [ch05a_G]
  have hD : (ch05a_Dcov n α).det = ch05a_sigEta2 α ^ (n - 1) := by
    rw [ch05a_Dcov, Matrix.det_diagonal]
    cases n with
    | zero => simp
    | succ m =>
      rw [Fin.prod_univ_eq_prod_range
          (fun j => if j = 0 then (1 : ℝ) else ch05a_sigEta2 α) (m + 1),
        Finset.prod_range_succ' (fun j => if j = 0 then (1 : ℝ) else ch05a_sigEta2 α) m]
      simp
  rw [ch05a_Sig0_factor n α, Matrix.det_mul, Matrix.det_mul, Matrix.det_transpose, hG, hD]
  ring

/-- **eq:g-Sigt-matrix, general `L`.**  The displayed noisy Toeplitz matrix at
every size: unit diagonal, `e^{-2t} α^{|i-j|}` off the diagonal. -/
theorem ch05a_g_Sigt_matrix_general (n : ℕ) (α t : ℝ) (i j : Fin n) :
    ch05a_Sigt n α t i j
      = if i = j then 1 else Real.exp (-2 * t) * α ^ (Nat.dist i.val j.val) := by
  by_cases h : i = j
  · rw [if_pos h, h, ch05a_noname25]
  · rw [if_neg h, ch05a_g_Sigt_offdiag n α t i j h]

/-! #### The quadratic form, the full density, Markov sparsity and locality loss,
all at general `L`. -/

/-- `(B a)_0 = a_0` and `(B a)_k = a_k - α a_{k-1}` for `k ≥ 1`. -/
theorem ch05a_B_mulVec {n : ℕ} (α : ℝ) (a : ℕ → ℝ) (l : Fin n) :
    (ch05a_B n α).mulVec (fun i : Fin n => a (i : ℕ)) l
      = if (l : ℕ) = 0 then a 0 else a (l : ℕ) - α * a ((l : ℕ) - 1) := by
  classical
  simp only [Matrix.mulVec, dotProduct]
  rcases Nat.eq_zero_or_pos (l : ℕ) with h0 | h0
  · rw [if_pos h0, ch05a_B_row_zero α (fun j : Fin n => a (j : ℕ)) l h0, h0]
  · obtain ⟨m, hm⟩ : ∃ m, (l : ℕ) = m + 1 := ⟨(l : ℕ) - 1, by omega⟩
    have hmn : m < n := by have := l.isLt; omega
    rw [if_neg (show ¬ ((l : ℕ) = 0) by omega),
      ch05a_B_row_succ α (fun j : Fin n => a (j : ℕ)) l ⟨m, hmn⟩ hm.symm,
      show (l : ℕ) - 1 = m by omega]

/-- **The quadratic form of `Q₀`, general `L`.**
`aᵀ Q₀ a = a₀² + (1/σ_η²) Σ_{k<L-1} (a_{k+1} - α a_k)²`: this is exactly
\eqref{eq:g-precision-expand-1} read off \eqref{eq:g-general-quadratic-form}, now
for every `L ≥ 2` instead of `L = 2, 3, 4`. -/
theorem ch05a_g_quadform_general {n : ℕ} (hn : 2 ≤ n) (α : ℝ) (hs : ch05a_sigEta2 α ≠ 0)
    (a : ℕ → ℝ) :
    (fun i : Fin n => a (i : ℕ)) ⬝ᵥ ((ch05a_Q0 n α).mulVec fun i : Fin n => a (i : ℕ))
      = a 0 ^ 2 + (1 / ch05a_sigEta2 α)
          * ∑ k ∈ Finset.range (n - 1), (a (k + 1) - α * a k) ^ 2 := by
  classical
  have key : (fun i : Fin n => a (i : ℕ)) ⬝ᵥ ((ch05a_Q0 n α).mulVec fun i : Fin n => a (i : ℕ))
      = ∑ l : Fin n, (if (l : ℕ) = 0 then (1 : ℝ) else (ch05a_sigEta2 α)⁻¹)
          * ((ch05a_B n α).mulVec (fun i : Fin n => a (i : ℕ)) l) ^ 2 := by
    rw [ch05a_Q0_factor hn α hs, ← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec,
      Matrix.dotProduct_mulVec, Matrix.vecMul_transpose]
    simp only [dotProduct, ch05a_Dprec, Matrix.mulVec_diagonal]
    exact Finset.sum_congr rfl fun l _ => by ring
  have hentry : ∀ l : Fin n,
      (if (l : ℕ) = 0 then (1 : ℝ) else (ch05a_sigEta2 α)⁻¹)
          * ((ch05a_B n α).mulVec (fun i : Fin n => a (i : ℕ)) l) ^ 2
        = (fun j : ℕ => (if j = 0 then (1 : ℝ) else (ch05a_sigEta2 α)⁻¹)
            * (if j = 0 then a 0 else a j - α * a (j - 1)) ^ 2) (l : ℕ) := by
    intro l
    rw [ch05a_B_mulVec]
  rw [key, Finset.sum_congr rfl (fun l (_ : l ∈ Finset.univ) => hentry l),
    Fin.sum_univ_eq_sum_range
      (fun j : ℕ => (if j = 0 then (1 : ℝ) else (ch05a_sigEta2 α)⁻¹)
        * (if j = 0 then a 0 else a j - α * a (j - 1)) ^ 2) n]
  obtain ⟨m, hm⟩ : ∃ m, n = m + 1 := ⟨n - 1, by omega⟩
  subst hm
  rw [Finset.sum_range_succ'
    (fun j : ℕ => (if j = 0 then (1 : ℝ) else (ch05a_sigEta2 α)⁻¹)
      * (if j = 0 then a 0 else a j - α * a (j - 1)) ^ 2) m]
  have hF : ∀ j ∈ Finset.range m,
      (if j + 1 = 0 then (1 : ℝ) else (ch05a_sigEta2 α)⁻¹)
          * (if j + 1 = 0 then a 0 else a (j + 1) - α * a (j + 1 - 1)) ^ 2
        = (ch05a_sigEta2 α)⁻¹ * (a (j + 1) - α * a j) ^ 2 := by
    intro j _
    rw [if_neg (show ¬ (j + 1 = 0) by omega), if_neg (show ¬ (j + 1 = 0) by omega),
      show j + 1 - 1 = j by omega]
  rw [Finset.sum_congr rfl hF, if_pos (rfl : (0 : ℕ) = 0), if_pos (rfl : (0 : ℕ) = 0), one_mul,
    Nat.add_sub_cancel, ← Finset.mul_sum, one_div]
  ring

/-- `√(x^m) = (√x)^m` for `0 ≤ x`. -/
theorem ch05a_sqrt_pow {x : ℝ} (hx : 0 ≤ x) (m : ℕ) :
    Real.sqrt (x ^ m) = Real.sqrt x ^ m := by
  induction m with
  | zero => simp
  | succ k ih => rw [pow_succ, Real.sqrt_mul (by positivity), ih, pow_succ]

/-- **eq:g-P0, general `L`.**  The Markov factorisation of the clean chain equals
the centred Gaussian density with covariance `Σ₀`, normalisation included:
`N(a₀;0,1) ∏_k N(a_{k+1}; α a_k, σ_η²) = exp(-½ aᵀQ₀a) / ((2π)^{L/2} |Σ₀|^{1/2})`.
Previously checked only at `L = 2`. -/
theorem ch05a_g_P0_general {n : ℕ} (hn : 2 ≤ n) (α : ℝ) (hσ : 0 < ch05a_sigEta2 α)
    (a : ℕ → ℝ) :
    ch05a_gauss 0 1 (a 0)
        * ∏ k ∈ Finset.range (n - 1), ch05a_gauss (α * a k) (ch05a_sigEta2 α) (a (k + 1))
      = Real.exp (-(1 / 2) * ((fun i : Fin n => a (i : ℕ)) ⬝ᵥ
            ((ch05a_Q0 n α).mulVec fun i : Fin n => a (i : ℕ))))
          / (Real.sqrt (2 * Real.pi) ^ n * Real.sqrt ((ch05a_Sig0 n α).det)) := by
  have hs : ch05a_sigEta2 α ≠ 0 := ne_of_gt hσ
  have hpi : (0 : ℝ) < 2 * Real.pi := by positivity
  have hquad := ch05a_g_quadform_general hn α hs a
  have hprod : (∏ k ∈ Finset.range (n - 1), ch05a_gauss (α * a k) (ch05a_sigEta2 α) (a (k + 1)))
      = (∏ k ∈ Finset.range (n - 1),
            Real.exp (-(a (k + 1) - α * a k) ^ 2 / (2 * ch05a_sigEta2 α)))
          / ∏ _k ∈ Finset.range (n - 1), Real.sqrt (2 * Real.pi * ch05a_sigEta2 α) := by
    rw [← Finset.prod_div_distrib]
    rfl
  have hE : (-(a 0 - 0) ^ 2 / (2 * 1))
      + ∑ k ∈ Finset.range (n - 1), (-(a (k + 1) - α * a k) ^ 2 / (2 * ch05a_sigEta2 α))
      = -(1 / 2) * ((fun i : Fin n => a (i : ℕ)) ⬝ᵥ
          ((ch05a_Q0 n α).mulVec fun i : Fin n => a (i : ℕ))) := by
    rw [hquad]
    have hsplit : ∑ k ∈ Finset.range (n - 1),
          (-(a (k + 1) - α * a k) ^ 2 / (2 * ch05a_sigEta2 α))
        = (-(1 / (2 * ch05a_sigEta2 α)))
            * ∑ k ∈ Finset.range (n - 1), (a (k + 1) - α * a k) ^ 2 := by
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun k _ => ?_
      field_simp
    rw [hsplit]
    field_simp
    ring
  have hnorm : Real.sqrt (2 * Real.pi * 1) * Real.sqrt (2 * Real.pi * ch05a_sigEta2 α) ^ (n - 1)
      = Real.sqrt (2 * Real.pi) ^ n * Real.sqrt ((ch05a_Sig0 n α).det) := by
    rw [ch05a_Sig0_det n α, ch05a_sqrt_pow (le_of_lt hσ) (n - 1), mul_one,
      Real.sqrt_mul (le_of_lt hpi), mul_pow]
    obtain ⟨m, hm⟩ : ∃ m, n = m + 1 := ⟨n - 1, by omega⟩
    subst hm
    rw [Nat.add_sub_cancel, pow_succ]
    ring
  rw [hprod, ← Real.exp_sum, Finset.prod_const, Finset.card_range]
  simp only [ch05a_gauss]
  rw [div_mul_div_comm, ← Real.exp_add, hE, hnorm]

/-- The negative log-density of a first-order Markov chain: a boundary term in
`a₀` plus one pair term per edge (tex lines 519-533). -/
def ch05a_chainNegLog (m : ℕ) (f0 : ℝ → ℝ) (f : ℕ → ℝ → ℝ → ℝ) (a : ℕ → ℝ) : ℝ :=
  f0 (a 0) + ∑ k ∈ Finset.range m, f k (a k) (a (k + 1))

/-- The double update `a[i ↦ u][j ↦ v]`, written as a single `if`-cascade. -/
def ch05a_upd (a : ℕ → ℝ) (i j : ℕ) (u v : ℝ) : ℕ → ℝ :=
  fun k => if k = j then v else if k = i then u else a k

theorem ch05a_upd_eq (a : ℕ → ℝ) (i j : ℕ) (u v : ℝ) :
    ch05a_upd a i j u v = Function.update (Function.update a i u) j v := by
  funext k
  simp only [ch05a_upd, Function.update_apply]

theorem ch05a_upd_indep_u (a : ℕ → ℝ) (i j : ℕ) (u u' v : ℝ) (k : ℕ) (hk : k ≠ i) :
    ch05a_upd a i j u v k = ch05a_upd a i j u' v k := by
  simp only [ch05a_upd]
  by_cases h : k = j
  · rw [if_pos h, if_pos h]
  · rw [if_neg h, if_neg h, if_neg hk, if_neg hk]

theorem ch05a_upd_indep_v (a : ℕ → ℝ) (i j : ℕ) (u v v' : ℝ) (k : ℕ) (hk : k ≠ j) :
    ch05a_upd a i j u v k = ch05a_upd a i j u v' k := by
  simp only [ch05a_upd]
  rw [if_neg hk, if_neg hk]

/-- **The content of eq:g-markov-hessian-sparsity, general `L`.**  For a
first-order chain the negative log-density is *additively separable* in any two
coordinates at distance `≥ 2`: no factor of the Markov factorisation contains
both, so as a function of `(a_i, a_j)` it splits as `g(a_i) + h(a_j)`. -/
theorem ch05a_chain_separable (m : ℕ) (f0 : ℝ → ℝ) (f : ℕ → ℝ → ℝ → ℝ) (a : ℕ → ℝ)
    (i j : ℕ) (hij : i + 2 ≤ j) (u v : ℝ) :
    ch05a_chainNegLog m f0 f (ch05a_upd a i j u v)
      = ch05a_chainNegLog m f0 f (ch05a_upd a i j u (a j))
        + (ch05a_chainNegLog m f0 f (ch05a_upd a i j (a i) v)
           - ch05a_chainNegLog m f0 f (ch05a_upd a i j (a i) (a j))) := by
  classical
  simp only [ch05a_chainNegLog]
  have hsum : ∑ k ∈ Finset.range m,
        f k (ch05a_upd a i j u v k) (ch05a_upd a i j u v (k + 1))
      = ∑ k ∈ Finset.range m,
        (f k (ch05a_upd a i j u (a j) k) (ch05a_upd a i j u (a j) (k + 1))
          + f k (ch05a_upd a i j (a i) v k) (ch05a_upd a i j (a i) v (k + 1))
          - f k (ch05a_upd a i j (a i) (a j) k) (ch05a_upd a i j (a i) (a j) (k + 1))) := by
    refine Finset.sum_congr rfl ?_
    intro k _
    by_cases hkj : k ≠ j ∧ k + 1 ≠ j
    · obtain ⟨hkj1, hkj2⟩ := hkj
      rw [ch05a_upd_indep_v a i j u v (a j) k hkj1,
        ch05a_upd_indep_v a i j u v (a j) (k + 1) hkj2,
        ch05a_upd_indep_v a i j (a i) v (a j) k hkj1,
        ch05a_upd_indep_v a i j (a i) v (a j) (k + 1) hkj2]
      ring
    · have hor : k = j ∨ k + 1 = j := by
        by_contra hc
        exact hkj ⟨fun h => hc (Or.inl h), fun h => hc (Or.inr h)⟩
      have hki : k ≠ i := by omega
      have hki1 : k + 1 ≠ i := by omega
      rw [ch05a_upd_indep_u a i j u (a i) v k hki,
        ch05a_upd_indep_u a i j u (a i) v (k + 1) hki1,
        ch05a_upd_indep_u a i j u (a i) (a j) k hki,
        ch05a_upd_indep_u a i j u (a i) (a j) (k + 1) hki1]
      ring
  rw [ch05a_upd_indep_v a i j u v (a j) 0 (by omega),
    ch05a_upd_indep_v a i j (a i) v (a j) 0 (by omega), hsum,
    Finset.sum_sub_distrib, Finset.sum_add_distrib]
  ring

/-- **eq:g-markov-hessian-sparsity, general `L`.**  For *any* smooth first-order
Markov negative log-density `-log P(a) = f₀(a₀) + Σ_k f_k(a_k, a_{k+1})` and any
two indices with `|i - j| ≥ 2`, the mixed second partial derivative vanishes
identically — no smoothness assumption is needed, the separability does it. -/
theorem ch05a_g_markov_hessian_sparsity_general (m : ℕ) (f0 : ℝ → ℝ) (f : ℕ → ℝ → ℝ → ℝ)
    (a : ℕ → ℝ) (i j : ℕ) (hij : i + 2 ≤ j) (u0 v0 : ℝ) :
    deriv (fun v => deriv (fun u =>
        ch05a_chainNegLog m f0 f
          (Function.update (Function.update a i u) j v)) u0) v0 = 0 := by
  have hinner : ∀ v : ℝ, deriv (fun u => ch05a_chainNegLog m f0 f
        (Function.update (Function.update a i u) j v)) u0
      = deriv (fun u => ch05a_chainNegLog m f0 f (ch05a_upd a i j u (a j))) u0 := by
    intro v
    have hfun : (fun u => ch05a_chainNegLog m f0 f (Function.update (Function.update a i u) j v))
        = (fun u => ch05a_chainNegLog m f0 f (ch05a_upd a i j u (a j))
            + (ch05a_chainNegLog m f0 f (ch05a_upd a i j (a i) v)
               - ch05a_chainNegLog m f0 f (ch05a_upd a i j (a i) (a j)))) := by
      funext u
      rw [← ch05a_upd_eq]
      exact ch05a_chain_separable m f0 f a i j hij u v
    rw [hfun, deriv_add_const]
  simp only [hinner]
  exact deriv_const _ _

/-- If `S` has a **tridiagonal** right inverse `Q`, then the top-left `3 × 3`
block of `S` satisfies the "Green's function" relation `S_{10} S_{21} =
S_{11} S_{20}`.  Reason: column `0` of a tridiagonal `Q` has at most the two
entries `Q_{00}, Q_{10}`, and rows `1, 2` of `S Q = I` force the `2 × 2` system
`[[S_{10}, S_{11}], [S_{20}, S_{21}]] (Q_{00}, Q_{10})ᵀ = 0` to have the nonzero
solution `(Q_{00}, Q_{10})`. -/
theorem ch05a_tridiag_right_inv_relation {n : ℕ} (S Q : Matrix (Fin n) (Fin n) ℝ)
    (e0 e1 e2 : Fin n) (h0v : (e0 : ℕ) = 0) (h1v : (e1 : ℕ) = 1) (h2v : (e2 : ℕ) = 2)
    (hQ : ∀ x y : Fin n, 2 ≤ Nat.dist (x : ℕ) (y : ℕ) → Q x y = 0)
    (h : S * Q = 1) :
    S e1 e0 * S e2 e1 = S e1 e1 * S e2 e0 := by
  classical
  have hne10 : e1 ≠ e0 := fun hh => by rw [hh, h0v] at h1v; omega
  have hexp : ∀ x : Fin n, ∑ y : Fin n, S x y * Q y e0
      = S x e0 * Q e0 e0 + S x e1 * Q e1 e0 := by
    intro x
    have hmem : e1 ∈ Finset.univ.erase e0 := Finset.mem_erase.mpr ⟨hne10, Finset.mem_univ _⟩
    have hrest : ∑ y ∈ (Finset.univ.erase e0).erase e1, S x y * Q y e0 = 0 := by
      refine Finset.sum_eq_zero ?_
      intro y hy
      rw [Finset.mem_erase, Finset.mem_erase] at hy
      obtain ⟨hy1, hy0, -⟩ := hy
      have hv0 : (y : ℕ) ≠ 0 := fun hh => hy0 (by ext; omega)
      have hv1 : (y : ℕ) ≠ 1 := fun hh => hy1 (by ext; omega)
      have hd : 2 ≤ Nat.dist (y : ℕ) ((e0 : Fin n) : ℕ) := by
        simp only [Nat.dist, h0v]
        omega
      rw [hQ y e0 hd, mul_zero]
    rw [← Finset.add_sum_erase _ (fun y => S x y * Q y e0) (Finset.mem_univ e0),
      ← Finset.add_sum_erase _ (fun y => S x y * Q y e0) hmem]
    simp only [hrest, add_zero]
  have hcol : ∀ x : Fin n, S x e0 * Q e0 e0 + S x e1 * Q e1 e0
      = (1 : Matrix (Fin n) (Fin n) ℝ) x e0 := by
    intro x
    rw [← hexp x, ← Matrix.mul_apply, h]
  have hne20 : e2 ≠ e0 := fun hh => by rw [hh, h0v] at h2v; omega
  have h0 := hcol e0
  have h1 := hcol e1
  have h2 := hcol e2
  rw [Matrix.one_apply, if_pos rfl] at h0
  rw [Matrix.one_apply, if_neg hne10] at h1
  rw [Matrix.one_apply, if_neg hne20] at h2
  by_contra hne
  have hD : S e1 e0 * S e2 e1 - S e1 e1 * S e2 e0 ≠ 0 := sub_ne_zero.mpr hne
  have hq00 : Q e0 e0 = 0 := by
    have hz : (S e1 e0 * S e2 e1 - S e1 e1 * S e2 e0) * Q e0 e0 = 0 := by
      linear_combination S e2 e1 * h1 - S e1 e1 * h2
    exact (mul_eq_zero.mp hz).resolve_left hD
  have hq10 : Q e1 e0 = 0 := by
    have hz : (S e1 e0 * S e2 e1 - S e1 e1 * S e2 e0) * Q e1 e0 = 0 := by
      linear_combination S e1 e0 * h2 - S e2 e0 * h1
    exact (mul_eq_zero.mp hz).resolve_left hD
  rw [hq00, hq10, mul_zero, mul_zero, add_zero] at h0
  exact zero_ne_one h0

/-- **eq:g-locality-loss-summary — the locality half, general `L`.**
For every chain length `L ≥ 3`, every `α ≠ 0` and every diffusion time `t > 0`,
the noisy covariance `Σ_t` admits *no* tridiagonal right inverse.  So the exact
conditional-independence zeros of `Q₀` at distance `≥ 2` cannot all survive:
locality is destroyed at every chain length, not just at the `L = 3` instance
checked above. -/
theorem ch05a_g_locality_loss_general {n : ℕ} (α t : ℝ) (hα : α ≠ 0) (ht : 0 < t)
    (e0 e1 e2 : Fin n) (h0v : (e0 : ℕ) = 0) (h1v : (e1 : ℕ) = 1) (h2v : (e2 : ℕ) = 2)
    (Q : Matrix (Fin n) (Fin n) ℝ) (hmul : ch05a_Sigt n α t * Q = 1) :
    ¬ (∀ x y : Fin n, 2 ≤ Nat.dist (x : ℕ) (y : ℕ) → Q x y = 0) := by
  intro hQ
  have hrel := ch05a_tridiag_right_inv_relation (ch05a_Sigt n α t) Q e0 e1 e2 h0v h1v h2v hQ hmul
  have hne10 : e1 ≠ e0 := fun hh => by rw [hh, h0v] at h1v; omega
  have hne21 : e2 ≠ e1 := fun hh => by rw [hh, h1v] at h2v; omega
  have hne20 : e2 ≠ e0 := fun hh => by rw [hh, h0v] at h2v; omega
  have hS10 : ch05a_Sigt n α t e1 e0 = Real.exp (-2 * t) * α := by
    rw [ch05a_g_Sigt_offdiag n α t e1 e0 hne10, h1v, h0v]
    norm_num [Nat.dist]
  have hS21 : ch05a_Sigt n α t e2 e1 = Real.exp (-2 * t) * α := by
    rw [ch05a_g_Sigt_offdiag n α t e2 e1 hne21, h2v, h1v]
    norm_num [Nat.dist]
  have hS20 : ch05a_Sigt n α t e2 e0 = Real.exp (-2 * t) * α ^ 2 := by
    rw [ch05a_g_Sigt_offdiag n α t e2 e0 hne20, h2v, h0v]
    norm_num [Nat.dist]
  have hS11 : ch05a_Sigt n α t e1 e1 = 1 := ch05a_noname25 n α t e1
  rw [hS10, hS21, hS11, hS20] at hrel
  have hc : 0 < Real.exp (-2 * t) := Real.exp_pos _
  have hclt : Real.exp (-2 * t) < 1 := by
    have hneg : (-2 * t) < 0 := by linarith
    have hlt := Real.exp_lt_exp.mpr hneg
    rwa [Real.exp_zero] at hlt
  have hcontr : Real.exp (-2 * t) * α ^ 2 * (Real.exp (-2 * t) - 1) = 0 := by
    linear_combination hrel
  rcases mul_eq_zero.mp hcontr with hz | hz
  · rcases mul_eq_zero.mp hz with hz' | hz'
    · exact absurd hz' (ne_of_gt hc)
    · exact absurd hz' (pow_ne_zero 2 hα)
  · linarith

/-- Corollary: when `Σ_t` is invertible, the noisy precision `Q_t = Σ_t⁻¹`
is not tridiagonal. -/
theorem ch05a_g_locality_loss_Qt {n : ℕ} (α t : ℝ) (hα : α ≠ 0) (ht : 0 < t)
    (e0 e1 e2 : Fin n) (h0v : (e0 : ℕ) = 0) (h1v : (e1 : ℕ) = 1) (h2v : (e2 : ℕ) = 2)
    (hdet : IsUnit (ch05a_Sigt n α t).det) :
    ¬ (∀ x y : Fin n, 2 ≤ Nat.dist (x : ℕ) (y : ℕ) → ch05a_Qt n α t x y = 0) :=
  ch05a_g_locality_loss_general α t hα ht e0 e1 e2 h0v h1v h2v (ch05a_Qt n α t)
    (Matrix.mul_nonsing_inv _ hdet)

/-! #### The standing assumptions (tex lines 48-54)

`a₀ ~ N(0,1)`, `η_k` i.i.d. `N(0, σ_η²)`, `a₀ ⟂ (η₀, η₁, …)` and `|α| < 1` are
*hypotheses* of the chapter, not identities, so there is nothing to verify in
them.  What can be verified is that they are consistent and are realised by an
explicit model: the theorem below exhibits the product measure on
`ε = (a₀, η₀, …, η_{L-2}) ∈ ℝ^L` whose coordinates are independent with exactly
the prescribed laws, and records that `|α| < 1` forces `σ_η² = 1 - α² > 0`. -/

/-- The law of `ε = (a₀, η₀, …, η_{L-2})`: `N(0,1)` in coordinate `0`,
`N(0, σ_η²)` in every later coordinate. -/
def ch05a_epsLaw (L : ℕ) (α : ℝ) (i : Fin L) : MeasureTheory.Measure ℝ :=
  if (i : ℕ) = 0 then ProbabilityTheory.gaussianReal 0 1
  else ProbabilityTheory.gaussianReal 0 (Real.toNNReal (ch05a_sigEta2 α))

instance ch05a_epsLaw_isProbabilityMeasure (L : ℕ) (α : ℝ) (i : Fin L) :
    MeasureTheory.IsProbabilityMeasure (ch05a_epsLaw L α i) := by
  unfold ch05a_epsLaw
  split
  · infer_instance
  · infer_instance

/-- `|α| < 1 ⟹ σ_η² = 1 - α² > 0`. -/
theorem ch05a_sigEta2_pos {α : ℝ} (hα : |α| < 1) : 0 < ch05a_sigEta2 α := by
  have h1 : α ^ 2 < 1 := by
    have h := abs_lt.mp hα
    nlinarith [h.1, h.2]
  simp only [ch05a_sigEta2]
  linarith

/-- **The standing assumptions are realised.**  On `ℝ^L` with the product
measure `ch05a_epsLaw`, the coordinates of `ε` are independent, coordinate `0`
(i.e. `a₀`) is `N(0,1)`, and every later coordinate (i.e. `η_k`) is
`N(0, σ_η²)` with `σ_η² = 1 - α² > 0`. -/
theorem ch05a_standing_assumptions (L : ℕ) (α : ℝ) (hα : |α| < 1) :
    0 < ch05a_sigEta2 α ∧
    ProbabilityTheory.iIndepFun (fun (i : Fin L) (ω : Fin L → ℝ) => ω i)
        (MeasureTheory.Measure.pi (ch05a_epsLaw L α)) ∧
    (∀ i : Fin L, MeasureTheory.Measure.map (fun ω : Fin L → ℝ => ω i)
        (MeasureTheory.Measure.pi (ch05a_epsLaw L α)) = ch05a_epsLaw L α i) ∧
    (∀ i : Fin L, (i : ℕ) = 0 → ch05a_epsLaw L α i = ProbabilityTheory.gaussianReal 0 1) ∧
    (∀ i : Fin L, (i : ℕ) ≠ 0 → ch05a_epsLaw L α i
        = ProbabilityTheory.gaussianReal 0 (Real.toNNReal (ch05a_sigEta2 α))) := by
  refine ⟨ch05a_sigEta2_pos hα, ?_, ?_, ?_, ?_⟩
  · exact ProbabilityTheory.iIndepFun_pi (X := fun _ => id) fun _ => aemeasurable_id
  · intro i
    exact (MeasureTheory.measurePreserving_eval (μ := ch05a_epsLaw L α) i).map_eq
  · intro i hi
    simp only [ch05a_epsLaw, if_pos hi]
  · intro i hi
    simp only [ch05a_epsLaw, if_neg hi]

/-! #### `Σ_t` really is invertible, so the general nonlocality statement is not vacuous. -/

/-- `vᵀ Σ₀ v = Σ_j d_j (Gᵀv)_j² ≥ 0` whenever `σ_η² ≥ 0` (i.e. `|α| ≤ 1`). -/
theorem ch05a_Sig0_quad_nonneg {n : ℕ} (α : ℝ) (hs : 0 ≤ ch05a_sigEta2 α) (v : Fin n → ℝ) :
    0 ≤ v ⬝ᵥ ((ch05a_Sig0 n α).mulVec v) := by
  classical
  rw [ch05a_Sig0_factor n α, ← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec,
    Matrix.dotProduct_mulVec, ← Matrix.mulVec_transpose]
  simp only [dotProduct, ch05a_Dcov, Matrix.mulVec_diagonal]
  refine Finset.sum_nonneg ?_
  intro j _
  by_cases h : (j : ℕ) = 0
  · rw [if_pos h]
    nlinarith [sq_nonneg (((ch05a_G n α).transpose.mulVec v) j)]
  · rw [if_neg h]
    nlinarith [sq_nonneg (((ch05a_G n α).transpose.mulVec v) j), hs]

/-- For `|α| ≤ 1` and `t > 0`, `Σ_t = e^{-2t} Σ₀ + Δ_t I` is nonsingular:
its quadratic form is bounded below by `Δ_t ‖v‖² > 0`. -/
theorem ch05a_Sigt_det_ne_zero {n : ℕ} {α t : ℝ} (hs : 0 ≤ ch05a_sigEta2 α) (ht : 0 < t) :
    (ch05a_Sigt n α t).det ≠ 0 := by
  classical
  intro hdet
  obtain ⟨v, hv, hv0⟩ := Matrix.exists_mulVec_eq_zero_iff.mpr hdet
  have hexpand : v ⬝ᵥ ((ch05a_Sigt n α t).mulVec v)
      = Real.exp (-2 * t) * (v ⬝ᵥ ((ch05a_Sig0 n α).mulVec v))
        + ch05a_Delta t * (v ⬝ᵥ v) := by
    simp only [ch05a_Sigt, Matrix.add_mulVec, Matrix.smul_mulVec, Matrix.one_mulVec,
      dotProduct_add, dotProduct_smul, smul_eq_mul]
  have hzero : v ⬝ᵥ ((ch05a_Sigt n α t).mulVec v) = 0 := by
    rw [hv0, dotProduct_zero]
  have h1 : 0 ≤ v ⬝ᵥ ((ch05a_Sig0 n α).mulVec v) := ch05a_Sig0_quad_nonneg α hs v
  have hvv : 0 ≤ v ⬝ᵥ v := by
    simp only [dotProduct]
    exact Finset.sum_nonneg fun i _ => mul_self_nonneg _
  have h2 : 0 < v ⬝ᵥ v :=
    lt_of_le_of_ne hvv fun hh => hv (dotProduct_self_eq_zero.mp hh.symm)
  have hc : 0 < Real.exp (-2 * t) := Real.exp_pos _
  have hD : 0 < ch05a_Delta t := by
    have hneg : (-2 * t) < 0 := by linarith
    have hlt := Real.exp_lt_exp.mpr hneg
    rw [Real.exp_zero] at hlt
    simp only [ch05a_Delta]
    linarith
  rw [hexpand] at hzero
  nlinarith [mul_nonneg (le_of_lt hc) h1, mul_pos hD h2]

/-- **Unconditional form of the general locality loss.**  For every `L ≥ 3`
(through the three index witnesses), every `α` with `α ≠ 0` and `|α| ≤ 1`, and
every `t > 0`, the noisy precision `Q_t = Σ_t⁻¹` exists and is *not*
tridiagonal. -/
theorem ch05a_g_locality_loss_Qt_unconditional {n : ℕ} (α t : ℝ) (hα : α ≠ 0)
    (hs : 0 ≤ ch05a_sigEta2 α) (ht : 0 < t)
    (e0 e1 e2 : Fin n) (h0v : (e0 : ℕ) = 0) (h1v : (e1 : ℕ) = 1) (h2v : (e2 : ℕ) = 2) :
    ¬ (∀ x y : Fin n, 2 ≤ Nat.dist (x : ℕ) (y : ℕ) → ch05a_Qt n α t x y = 0) :=
  ch05a_g_locality_loss_Qt α t hα ht e0 e1 e2 h0v h1v h2v
    (isUnit_iff_ne_zero.mpr (ch05a_Sigt_det_ne_zero hs ht))


/-! ### Pass-4: closing the remaining general-`L` gaps

Three additions:

* \eqref{eq:g-P0} stated through `Σ₀⁻¹` itself rather than through the
  tridiagonal matrix `Q₀` that \eqref{eq:g-Q0} identifies it with, so that the
  right-hand side is literally `N(a; 0, Σ₀)`;
* \eqref{eq:g-markov-hessian-sparsity} for *both* orders of the two indices,
  i.e. under the symmetric hypothesis `|i - j| ≥ 2` rather than `j ≥ i + 2`;
* \eqref{eq:g-locality-loss-summary} strengthened from "`Q_t` is not
  tridiagonal" to "the first column of `Q_t` carries a nonzero entry at distance
  `≥ 2`", and the two halves of the boxed display assembled into one statement
  at general `L`. -/

/-- **eq:g-P0, general `L`, stated through `Σ₀⁻¹`.**  The same identity as
`ch05a_g_P0_general` with the exponent written through the inverse of the
covariance itself: the Markov factorisation of the clean chain *is* the centred
Gaussian density with covariance `Σ₀`, normalisation included. -/
theorem ch05a_g_P0_general_inv {n : ℕ} (hn : 2 ≤ n) (α : ℝ) (hσ : 0 < ch05a_sigEta2 α)
    (a : ℕ → ℝ) :
    ch05a_gauss 0 1 (a 0)
        * ∏ k ∈ Finset.range (n - 1), ch05a_gauss (α * a k) (ch05a_sigEta2 α) (a (k + 1))
      = Real.exp (-(1 / 2) * ((fun i : Fin n => a (i : ℕ)) ⬝ᵥ
            (((ch05a_Sig0 n α)⁻¹).mulVec fun i : Fin n => a (i : ℕ))))
          / (Real.sqrt (2 * Real.pi) ^ n * Real.sqrt ((ch05a_Sig0 n α).det)) := by
  rw [ch05a_g_Q0_general hn α (ne_of_gt hσ)]
  exact ch05a_g_P0_general hn α hσ a

/-- A function additively separable in its two arguments has vanishing mixed
second derivative. -/
theorem ch05a_mixed_deriv_of_separable (F : ℝ → ℝ → ℝ) (g h : ℝ → ℝ)
    (hF : ∀ u v, F u v = g u + h v) (u0 v0 : ℝ) :
    deriv (fun v => deriv (fun u => F u v) u0) v0 = 0 := by
  have hc : (fun v => deriv (fun u => F u v) u0) = fun _ => deriv g u0 := by
    funext v
    have hfun : (fun u => F u v) = fun u => g u + h v := by
      funext u; exact hF u v
    rw [hfun]
    simp [deriv_add_const]
  rw [hc, deriv_const]

/-- **eq:g-markov-hessian-sparsity, general `L`, symmetric in the two indices.**
For *any* first-order Markov negative log-density
`-log P(a) = f₀(a₀) + Σ_k f_k(a_k, a_{k+1})` and any two indices with
`|i - j| ≥ 2` — in either order — the mixed second partial derivative vanishes
identically.  No smoothness assumption is needed: separability does it. -/
theorem ch05a_g_markov_hessian_sparsity_dist (m : ℕ) (f0 : ℝ → ℝ) (f : ℕ → ℝ → ℝ → ℝ)
    (a : ℕ → ℝ) (i j : ℕ) (hij : 2 ≤ Nat.dist i j) (u0 v0 : ℝ) :
    deriv (fun v => deriv (fun u =>
        ch05a_chainNegLog m f0 f
          (Function.update (Function.update a i u) j v)) u0) v0 = 0 := by
  classical
  by_cases hle : i + 2 ≤ j
  · exact ch05a_g_markov_hessian_sparsity_general m f0 f a i j hle u0 v0
  · have hji : j + 2 ≤ i := by
      simp only [Nat.dist] at hij
      omega
    have hne : i ≠ j := by omega
    refine ch05a_mixed_deriv_of_separable
      (fun u v => ch05a_chainNegLog m f0 f (Function.update (Function.update a i u) j v))
      (fun u => ch05a_chainNegLog m f0 f (ch05a_upd a j i (a j) u))
      (fun v => ch05a_chainNegLog m f0 f (ch05a_upd a j i v (a i))
        - ch05a_chainNegLog m f0 f (ch05a_upd a j i (a j) (a i))) ?_ u0 v0
    intro u v
    show ch05a_chainNegLog m f0 f (Function.update (Function.update a i u) j v)
        = ch05a_chainNegLog m f0 f (ch05a_upd a j i (a j) u)
          + (ch05a_chainNegLog m f0 f (ch05a_upd a j i v (a i))
             - ch05a_chainNegLog m f0 f (ch05a_upd a j i (a j) (a i)))
    rw [Function.update_comm hne u v a, ← ch05a_upd_eq,
      ch05a_chain_separable m f0 f a j i hji v u]
    ring

/-- The same "Green's function" relation as `ch05a_tridiag_right_inv_relation`,
under the strictly weaker hypothesis that only the *first column* of the right
inverse `Q` is local, i.e. supported on the two rows `0, 1`.  The proof of the
tridiagonal version never uses any other column. -/
theorem ch05a_col0_right_inv_relation {n : ℕ} (S Q : Matrix (Fin n) (Fin n) ℝ)
    (e0 e1 e2 : Fin n) (h0v : (e0 : ℕ) = 0) (h1v : (e1 : ℕ) = 1) (h2v : (e2 : ℕ) = 2)
    (hQ : ∀ y : Fin n, 2 ≤ (y : ℕ) → Q y e0 = 0)
    (h : S * Q = 1) :
    S e1 e0 * S e2 e1 = S e1 e1 * S e2 e0 := by
  classical
  have hne10 : e1 ≠ e0 := fun hh => by rw [hh, h0v] at h1v; omega
  have hexp : ∀ x : Fin n, ∑ y : Fin n, S x y * Q y e0
      = S x e0 * Q e0 e0 + S x e1 * Q e1 e0 := by
    intro x
    have hmem : e1 ∈ Finset.univ.erase e0 := Finset.mem_erase.mpr ⟨hne10, Finset.mem_univ _⟩
    have hrest : ∑ y ∈ (Finset.univ.erase e0).erase e1, S x y * Q y e0 = 0 := by
      refine Finset.sum_eq_zero ?_
      intro y hy
      rw [Finset.mem_erase, Finset.mem_erase] at hy
      obtain ⟨hy1, hy0, -⟩ := hy
      have hv0 : (y : ℕ) ≠ 0 := fun hh => hy0 (by ext; omega)
      have hv1 : (y : ℕ) ≠ 1 := fun hh => hy1 (by ext; omega)
      rw [hQ y (by omega), mul_zero]
    rw [← Finset.add_sum_erase _ (fun y => S x y * Q y e0) (Finset.mem_univ e0),
      ← Finset.add_sum_erase _ (fun y => S x y * Q y e0) hmem]
    simp only [hrest, add_zero]
  have hcol : ∀ x : Fin n, S x e0 * Q e0 e0 + S x e1 * Q e1 e0
      = (1 : Matrix (Fin n) (Fin n) ℝ) x e0 := by
    intro x
    rw [← hexp x, ← Matrix.mul_apply, h]
  have hne20 : e2 ≠ e0 := fun hh => by rw [hh, h0v] at h2v; omega
  have h0 := hcol e0
  have h1 := hcol e1
  have h2 := hcol e2
  rw [Matrix.one_apply, if_pos rfl] at h0
  rw [Matrix.one_apply, if_neg hne10] at h1
  rw [Matrix.one_apply, if_neg hne20] at h2
  by_contra hne
  have hD : S e1 e0 * S e2 e1 - S e1 e1 * S e2 e0 ≠ 0 := sub_ne_zero.mpr hne
  have hq00 : Q e0 e0 = 0 := by
    have hz : (S e1 e0 * S e2 e1 - S e1 e1 * S e2 e0) * Q e0 e0 = 0 := by
      linear_combination S e2 e1 * h1 - S e1 e1 * h2
    exact (mul_eq_zero.mp hz).resolve_left hD
  have hq10 : Q e1 e0 = 0 := by
    have hz : (S e1 e0 * S e2 e1 - S e1 e1 * S e2 e0) * Q e1 e0 = 0 := by
      linear_combination S e1 e0 * h2 - S e2 e0 * h1
    exact (mul_eq_zero.mp hz).resolve_left hD
  rw [hq00, hq10, mul_zero, mul_zero, add_zero] at h0
  exact zero_ne_one h0

/-- **eq:g-locality-loss-summary, general `L`, strengthened.**  For every chain
length `L ≥ 3` (carried by the three index witnesses), every `α ≠ 0` and every
diffusion time `t > 0`, no right inverse of `Σ_t` has its first column supported
on the rows `0, 1`: there is a frame at distance `≥ 2` from frame `0` that is
coupled to it.  This locates the lost zero, where
`ch05a_g_locality_loss_general` only says that some zero is lost. -/
theorem ch05a_g_locality_loss_col0 {n : ℕ} (α t : ℝ) (hα : α ≠ 0) (ht : 0 < t)
    (e0 e1 e2 : Fin n) (h0v : (e0 : ℕ) = 0) (h1v : (e1 : ℕ) = 1) (h2v : (e2 : ℕ) = 2)
    (Q : Matrix (Fin n) (Fin n) ℝ) (hmul : ch05a_Sigt n α t * Q = 1) :
    ∃ y : Fin n, 2 ≤ (y : ℕ) ∧ Q y e0 ≠ 0 := by
  by_contra hc
  have hzero : ∀ y : Fin n, 2 ≤ (y : ℕ) → Q y e0 = 0 := by
    intro y hy
    by_contra hz
    exact hc ⟨y, hy, hz⟩
  have hrel :=
    ch05a_col0_right_inv_relation (ch05a_Sigt n α t) Q e0 e1 e2 h0v h1v h2v hzero hmul
  have hne10 : e1 ≠ e0 := fun hh => by rw [hh, h0v] at h1v; omega
  have hne21 : e2 ≠ e1 := fun hh => by rw [hh, h1v] at h2v; omega
  have hne20 : e2 ≠ e0 := fun hh => by rw [hh, h0v] at h2v; omega
  have hS10 : ch05a_Sigt n α t e1 e0 = Real.exp (-2 * t) * α := by
    rw [ch05a_g_Sigt_offdiag n α t e1 e0 hne10, h1v, h0v]
    norm_num [Nat.dist]
  have hS21 : ch05a_Sigt n α t e2 e1 = Real.exp (-2 * t) * α := by
    rw [ch05a_g_Sigt_offdiag n α t e2 e1 hne21, h2v, h1v]
    norm_num [Nat.dist]
  have hS20 : ch05a_Sigt n α t e2 e0 = Real.exp (-2 * t) * α ^ 2 := by
    rw [ch05a_g_Sigt_offdiag n α t e2 e0 hne20, h2v, h0v]
    norm_num [Nat.dist]
  have hS11 : ch05a_Sigt n α t e1 e1 = 1 := ch05a_noname25 n α t e1
  rw [hS10, hS21, hS11, hS20] at hrel
  have hc0 : 0 < Real.exp (-2 * t) := Real.exp_pos _
  have hclt : Real.exp (-2 * t) < 1 := by
    have hneg : (-2 * t) < 0 := by linarith
    have hlt := Real.exp_lt_exp.mpr hneg
    rwa [Real.exp_zero] at hlt
  have hcontr : Real.exp (-2 * t) * α ^ 2 * (Real.exp (-2 * t) - 1) = 0 := by
    linear_combination hrel
  rcases mul_eq_zero.mp hcontr with hz | hz
  · rcases mul_eq_zero.mp hz with hz' | hz'
    · exact absurd hz' (ne_of_gt hc0)
    · exact absurd hz' (pow_ne_zero 2 hα)
  · linarith

/-- Corollary with `Q_t = Σ_t⁻¹` itself, unconditionally: for `α ≠ 0`,
`|α| ≤ 1` and `t > 0`, `Σ_t` is invertible and its inverse has a nonzero entry
in the first column at distance `≥ 2`. -/
theorem ch05a_g_locality_loss_Qt_col0 {n : ℕ} (α t : ℝ) (hα : α ≠ 0)
    (hs : 0 ≤ ch05a_sigEta2 α) (ht : 0 < t)
    (e0 e1 e2 : Fin n) (h0v : (e0 : ℕ) = 0) (h1v : (e1 : ℕ) = 1) (h2v : (e2 : ℕ) = 2) :
    ∃ y : Fin n, 2 ≤ (y : ℕ) ∧ ch05a_Qt n α t y e0 ≠ 0 :=
  ch05a_g_locality_loss_col0 α t hα ht e0 e1 e2 h0v h1v h2v (ch05a_Qt n α t)
    (Matrix.mul_nonsing_inv _ (isUnit_iff_ne_zero.mpr (ch05a_Sigt_det_ne_zero hs ht)))

/-- **The boxed display eq:g-locality-loss-summary, both halves, general `L`.**
For every `L ≥ 3`, every `α` with `0 < |α| < 1` and every `t > 0`:

* *clean precision is local and tridiagonal*: `Σ₀⁻¹` equals the displayed
  tridiagonal `Q₀`, and every entry of it at distance `≥ 2` is exactly zero;
* *noisy precision is nonlocal*: `Q_t = Σ_t⁻¹` has a nonzero entry at distance
  `≥ 2` (located in the first column), so those zeros do not survive diffusion
  and marginalisation. -/
theorem ch05a_g_locality_loss_summary_general {n : ℕ} (hn : 3 ≤ n) (α t : ℝ)
    (hα : α ≠ 0) (hα1 : |α| < 1) (ht : 0 < t) :
    ((ch05a_Sig0 n α)⁻¹ = ch05a_Q0 n α
      ∧ ∀ i j : Fin n, 2 ≤ Nat.dist (i : ℕ) (j : ℕ) → ch05a_Q0 n α i j = 0)
    ∧ (∃ x y : Fin n, (x : ℕ) = 0 ∧ 2 ≤ (y : ℕ) ∧ ch05a_Qt n α t y x ≠ 0) := by
  have hs : 0 < ch05a_sigEta2 α := ch05a_sigEta2_pos hα1
  refine ⟨⟨ch05a_g_Q0_general (by omega) α (ne_of_gt hs),
    fun i j hij => ch05a_g_precision_distant_zero α i j hij⟩, ?_⟩
  obtain ⟨y, hy2, hy⟩ := ch05a_g_locality_loss_Qt_col0 (n := n) α t hα (le_of_lt hs) ht
    ⟨0, by omega⟩ ⟨1, by omega⟩ ⟨2, by omega⟩ rfl rfl rfl
  exact ⟨⟨0, by omega⟩, y, rfl, hy2, hy⟩


end

end ThesisAudit
