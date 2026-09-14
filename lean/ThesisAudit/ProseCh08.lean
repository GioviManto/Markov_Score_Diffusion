import Mathlib

/-!
# Prose / inline-math audit of ch08-em-parameters.tex

Witnesses for the numerical claims made in *prose* (not display math) in
Chapter 8.  Exact rational arithmetic throughout: every literal below is the
decimal printed in the chapter, read as an exact rational.
-/

namespace ThesisAudit
namespace ProseCh08

/-! ## 1. Remark "The inner budget of sixteen" (ch08 l.1183-1186)

The chapter writes: excess kurtosis moves "from $1.68$ to $2.14$, a $28\%$
change relative to the former value".  Relative to the former value the change
is `(2.14 - 1.68)/1.68 = 23/84`, which is `27.38...%`, i.e. `27%`, not `28%`. -/

theorem inner_budget_relative_change :
    (214/100 - 168/100 : ℚ) / (168/100) = 23/84 := by norm_num

/-- The printed 28% is strictly below the true ratio, and 27% is the correct
rounding: `27/100 < 23/84 < 275/1000 < 28/100`. -/
theorem inner_budget_rounds_to_27 :
    (27/100 : ℚ) < 23/84 ∧ (23/84 : ℚ) < 275/1000 ∧ (275/1000 : ℚ) < 28/100 := by
  refine ⟨by norm_num, by norm_num, by norm_num⟩

/-- A 28% change from 1.68 would land at 2.1504, not 2.14. -/
theorem inner_budget_endpoint_mismatch :
    (168/100 : ℚ) * (128/100) = 21504/10000 ∧ (21504/10000 : ℚ) ≠ 214/100 := by
  refine ⟨by norm_num, by norm_num⟩

/-! ## 2. Table `tab:em-recovery`, row `C = 8`, `N = 2048` (ch08 l.1377)

The row prints `\hat v = 0.2793` against the true `\ivar = 0.2775` declared in
the caption, and quotes `|Δv|/v = 0.7%`.  The actual ratio is `18/2775`, i.e.
`0.6486...%`, which rounds to `0.6%`. -/

theorem recovery_var_relative_error :
    ((2793/10000 : ℚ) - 2775/10000) / (2775/10000) = 18/2775 := by norm_num

/-- `0.6486...% < 0.65%`, so the value rounds to `0.6%`, strictly below the
printed `0.7%`. -/
theorem recovery_var_rounds_to_six_tenths :
    (6/1000 : ℚ) < 18/2775 ∧ (18/2775 : ℚ) < 65/10000 ∧ (65/10000 : ℚ) < 7/1000 := by
  refine ⟨by norm_num, by norm_num, by norm_num⟩

/-! ## 3. Controls: the neighbouring rows of the same column DO round as printed,
so the convention in use is ordinary round-to-nearest at one decimal. -/

theorem recovery_var_row_C8_N512 :
    ((2775/10000 : ℚ) - 2694/10000) / (2775/10000) < 295/10000 ∧
    (285/10000 : ℚ) < ((2775/10000 : ℚ) - 2694/10000) / (2775/10000) := by
  refine ⟨by norm_num, by norm_num⟩

theorem recovery_var_row_C4_N2048 :
    ((2818/10000 : ℚ) - 2775/10000) / (2775/10000) < 155/10000 ∧
    (145/10000 : ℚ) < ((2818/10000 : ℚ) - 2775/10000) / (2775/10000) := by
  refine ⟨by norm_num, by norm_num⟩

/-! ## 4. Stationary variance of the fitted mixture chain (ch08 l.57-62)

Two-component illustration.  With innovation mean `m = Σ π_c ν_c ≠ 0` the
stationary variance of `a_i = α a_{i-1} + ε_i` is
`(Σ π_c (ν_c² + s_c²) − m²)/(1 − α²)`, not `Σ π_c (ν_c² + s_c²)/(1 − α²)`.
Here `π = (1/2, 1/2)`, `ν = (1, 1)`, `s² = (1, 1)`, `α = 0`: the printed
formula gives `2`, the true stationary variance is `1`. -/

theorem stationary_variance_counterexample :
    let π₁ : ℚ := 1/2; let π₂ : ℚ := 1/2
    let ν₁ : ℚ := 1;   let ν₂ : ℚ := 1
    let s₁ : ℚ := 1;   let s₂ : ℚ := 1
    let α  : ℚ := 0
    let m := π₁ * ν₁ + π₂ * ν₂
    let second := π₁ * (ν₁^2 + s₁^2) + π₂ * (ν₂^2 + s₂^2)
    -- printed formula
    second / (1 - α^2) = 2 ∧
    -- true stationary variance (innovation variance / (1 - α²))
    (second - m^2) / (1 - α^2) = 1 ∧
    m ≠ 0 := by
  refine ⟨by norm_num, by norm_num, by norm_num⟩

end ProseCh08
end ThesisAudit
