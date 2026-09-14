import Mathlib

/-!
# Audit of `ch03-model.tex` ("The Model, and the Object of Study")

Every display-math environment of the chapter is treated here, in order of
appearance.  Naming convention: `ch03_<label>` for labelled equations,
`ch03_noname_<k>` for unlabelled displays (numbered in order of appearance).

Conventions read off the chapter's prose and the thesis preamble:
* `\Deltat` is `Δ_t = 1 - e^{-2t}`                      (preamble.tex line 187, ch03 line 195)
* `\ivar` is the innovation variance `q`, set to `1-α²`  (ch03 line 99)
* `\Nd` is the Gaussian law, `\kernel` the transition kernel `K(a'|a)`
* indices run `0, …, L-1`, so the transition product runs `k = 0, …, L-2`.
-/

namespace ThesisAudit

open MeasureTheory ProbabilityTheory Matrix

/-! ## §3.1 Markov sequences -/

/-- `Δ_t = 1 - e^{-2t}`, the OU channel's noise variance (ch03 lines 195, 205, 317). -/
noncomputable def ch03_Delta (t : ℝ) : ℝ := 1 - Real.exp (-2 * t)

-- eq:strict-stationary (ch03-model.tex lines 21-26): a process is strictly stationary
-- when all its finite-dimensional laws are invariant under a time shift.
def ch03_StrictlyStationary {Ω : Type*} [MeasurableSpace Ω]
    (X : ℤ → Ω → ℝ) (μ : Measure Ω) : Prop :=
  ∀ (n : ℕ) (k : Fin n → ℤ) (h : ℤ),
    μ.map (fun ω i => X (k i + h) ω) = μ.map (fun ω i => X (k i) ω)

-- eq:strict-stationary sanity check: a strictly stationary process stays strictly
-- stationary after any re-indexing by a constant time shift.
theorem ch03_strict_stationary_shift {Ω : Type*} [MeasurableSpace Ω]
    {X : ℤ → Ω → ℝ} {μ : Measure Ω} (H : ch03_StrictlyStationary X μ) (c : ℤ) :
    ch03_StrictlyStationary (fun k => X (k + c)) μ := by
  intro n k h
  have := H n (fun i => k i + c) h
  simpa [add_right_comm] using this

-- noname-1 (lines 29-31) and eq:weak-stationary (lines 33-36): weak (covariance)
-- stationarity, i.e. constant mean `μ` and covariance depending only on the lag.
def ch03_WeaklyStationary {Ω : Type*} [MeasurableSpace Ω]
    (X : ℤ → Ω → ℝ) (μ : Measure Ω) (m : ℝ) (γ : ℤ → ℝ) : Prop :=
  (∀ k, ∫ ω, X k ω ∂μ = m) ∧
  (∀ j k, ∫ ω, (X j ω - m) * (X k ω - m) ∂μ = γ (j - k))

-- eq:weak-stationary sanity check: the autocovariance of a real weakly stationary
-- process is an even function, which is the fact used at line 56 to fill in the
-- Toeplitz matrix of eq:toeplitz.
theorem ch03_weak_stationary_even {Ω : Type*} [MeasurableSpace Ω]
    {X : ℤ → Ω → ℝ} {μ : Measure Ω} {m : ℝ} {γ : ℤ → ℝ}
    (H : ch03_WeaklyStationary X μ m γ) (d : ℤ) : γ (-d) = γ d := by
  have h1 := H.2 0 d
  have h2 := H.2 d 0
  simp only [zero_sub, sub_zero] at h1 h2
  rw [← h1, ← h2]
  exact integral_congr_ae (Filter.Eventually.of_forall fun ω => mul_comm _ _)

-- eq:toeplitz (lines 44-55): the covariance matrix of a length-L segment of a
-- weakly stationary process, `Σ_ij = γ(i-j)`.
def ch03_toeplitz (L : ℕ) (γ : ℤ → ℝ) : Matrix (Fin L) (Fin L) ℝ :=
  Matrix.of fun i j => γ ((i : ℤ) - (j : ℤ))

-- eq:toeplitz: the matrix is constant along every diagonal (the defining property).
theorem ch03_toeplitz_const_diag (L : ℕ) (γ : ℤ → ℝ) (i j i' j' : Fin L)
    (h : (i : ℤ) - (j : ℤ) = (i' : ℤ) - (j' : ℤ)) :
    ch03_toeplitz L γ i j = ch03_toeplitz L γ i' j' := by
  simp [ch03_toeplitz, h]

-- eq:toeplitz: symmetry, using `γ(-d) = γ(d)` for a real process (line 56).
theorem ch03_toeplitz_symm (L : ℕ) (γ : ℤ → ℝ) (hev : ∀ d, γ (-d) = γ d) :
    (ch03_toeplitz L γ)ᵀ = ch03_toeplitz L γ := by
  ext i j
  simp only [Matrix.transpose_apply, ch03_toeplitz, Matrix.of_apply]
  rw [show ((j : ℤ) - (i : ℤ)) = -((i : ℤ) - (j : ℤ)) by ring, hev]

-- eq:toeplitz: the entries displayed in the chapter, checked on the 4×4 corner.
theorem ch03_toeplitz_display (γ : ℤ → ℝ) (hev : ∀ d, γ (-d) = γ d) :
    ch03_toeplitz 4 γ =
      !![γ 0, γ 1, γ 2, γ 3;
         γ 1, γ 0, γ 1, γ 2;
         γ 2, γ 1, γ 0, γ 1;
         γ 3, γ 2, γ 1, γ 0] := by
  have h1 : γ (-1 : ℤ) = γ 1 := hev 1
  have h2 : γ (-2 : ℤ) = γ 2 := hev 2
  have h3 : γ (-3 : ℤ) = γ 3 := hev 3
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [ch03_toeplitz, h1, h2, h3]

-- eq:markov-property (lines 61-65): in density form, the conditional law of `x_{n+1}`
-- given the whole past `x_0, …, x_n` collapses to a kernel of `x_n` alone.
-- `c n x` is the conditional density of `x_{n+1}` given `x_0, …, x_n`.
def ch03_MarkovProperty {α : Type*} (c : ℕ → (ℕ → α) → ℝ) (K : α → α → ℝ) : Prop :=
  ∀ (n : ℕ) (x : ℕ → α), c n x = K (x n) (x (n + 1))

/-- Chain-rule joint density of `(x_0, …, x_n)` built from the initial density `p₀`
and the conditional densities `c`. -/
noncomputable def ch03_joint {α : Type*} (p₀ : α → ℝ) (c : ℕ → (ℕ → α) → ℝ)
    (x : ℕ → α) : ℕ → ℝ
  | 0 => p₀ (x 0)
  | (n + 1) => ch03_joint p₀ c x n * c n x

-- eq:chain-factorisation (lines 73-77): the Markov property is exactly the
-- factorisation `p(x_0,…,x_{L-1}) = p₀(x_0) ∏_{k=0}^{L-2} K(x_{k+1} | x_k)`.
-- (Here `m = L-1`, so the product runs over `k = 0, …, L-2`.)
theorem ch03_chain_factorisation {α : Type*} (p₀ : α → ℝ) (c : ℕ → (ℕ → α) → ℝ)
    (K : α → α → ℝ) (hM : ch03_MarkovProperty c K) (x : ℕ → α) (m : ℕ) :
    ch03_joint p₀ c x m = p₀ (x 0) * ∏ k ∈ Finset.range m, K (x k) (x (k + 1)) := by
  induction m with
  | zero => simp [ch03_joint]
  | succ n ih =>
      rw [ch03_joint, ih, hM n x, Finset.prod_range_succ]
      ring

/-! ### Example: the Gaussian AR(1) chain -/

-- eq:ar1 (lines 91-97): `a_{k+1} = α a_k + η_k`.
noncomputable def ch03_ar1 (α a0 : ℝ) (η : ℕ → ℝ) : ℕ → ℝ
  | 0 => a0
  | (k + 1) => α * ch03_ar1 α a0 η k + η k

-- eq:ar1 sanity check: with the innovations switched off the recursion contracts
-- geometrically, `a_k = α^k a_0`.
theorem ch03_ar1_noiseless (α a0 : ℝ) (k : ℕ) :
    ch03_ar1 α a0 (fun _ => 0) k = α ^ k * a0 := by
  induction k with
  | zero => simp [ch03_ar1]
  | succ n ih => rw [ch03_ar1, ih]; ring

/-- `|i - j|` on ℕ, written without integer casts. -/
def ch03_absdiff (i j : ℕ) : ℕ := max i j - min i j

theorem ch03_absdiff_self (i : ℕ) : ch03_absdiff i i = 0 := by simp [ch03_absdiff]

theorem ch03_absdiff_comm (i j : ℕ) : ch03_absdiff i j = ch03_absdiff j i := by
  simp [ch03_absdiff, max_comm, min_comm]

theorem ch03_absdiff_eq_natAbs (i j : ℕ) : ch03_absdiff i j = ((i : ℤ) - (j : ℤ)).natAbs := by
  simp only [ch03_absdiff]
  omega

-- eq:ar1-toeplitz (lines 101-112): the stationary AR(1) autocovariance `α^{|j-k|}`.
noncomputable def ch03_ar1_cov (α : ℝ) (j k : ℕ) : ℝ := α ^ ch03_absdiff j k

theorem ch03_ar1_cov_symm (α : ℝ) (j k : ℕ) : ch03_ar1_cov α j k = ch03_ar1_cov α k j := by
  simp [ch03_ar1_cov, ch03_absdiff_comm]

-- eq:ar1-toeplitz: unit variance on the diagonal (the "a_k ~ N(0,1) for every k" of line 99).
theorem ch03_ar1_cov_diag (α : ℝ) (k : ℕ) : ch03_ar1_cov α k k = 1 := by
  simp [ch03_ar1_cov, ch03_absdiff_self]

-- eq:ar1-toeplitz: the cross-covariance recursion `Cov(a_j, a_{k+1}) = α Cov(a_j, a_k)`,
-- which is what `a_{k+1} = α a_k + η_k` with `η_k ⟂ a_j` (j ≤ k) gives.
theorem ch03_ar1_cov_step (α : ℝ) {j k : ℕ} (h : j ≤ k) :
    ch03_ar1_cov α j (k + 1) = α * ch03_ar1_cov α j k := by
  have h1 : ch03_absdiff j (k + 1) = ch03_absdiff j k + 1 := by
    simp only [ch03_absdiff]; omega
  rw [ch03_ar1_cov, ch03_ar1_cov, h1, pow_succ]
  ring

-- eq:ar1-toeplitz: the variance recursion `Var(a_{k+1}) = α² Var(a_k) + q` with the
-- innovation variance `q = 1 - α²` chosen at line 99.
theorem ch03_ar1_var_step (α : ℝ) (k : ℕ) :
    ch03_ar1_cov α (k + 1) (k + 1) = α ^ 2 * ch03_ar1_cov α k k + (1 - α ^ 2) := by
  rw [ch03_ar1_cov_diag, ch03_ar1_cov_diag]; ring

-- eq:ar1-toeplitz (lines 101-112), MAIN CLAIM: the two second-moment recursions of
-- the stationary AR(1) chain force `Cov(a_j, a_k) = α^{|j-k|}`; nothing else satisfies them.
theorem ch03_ar1_toeplitz (α : ℝ) (γ : ℕ → ℕ → ℝ)
    (hsymm : ∀ j k, γ j k = γ k j)
    (h0 : γ 0 0 = 1)
    (hvar : ∀ k, γ (k + 1) (k + 1) = α ^ 2 * γ k k + (1 - α ^ 2))
    (hstep : ∀ j k, j ≤ k → γ j (k + 1) = α * γ j k) :
    ∀ j k, γ j k = α ^ ch03_absdiff j k := by
  have hdiag : ∀ k, γ k k = 1 := by
    intro k
    induction k with
    | zero => exact h0
    | succ n ih => rw [hvar n, ih]; ring
  have hkey : ∀ (d j : ℕ), γ j (j + d) = α ^ d := by
    intro d
    induction d with
    | zero => intro j; simpa using hdiag j
    | succ n ih =>
        intro j
        rw [show j + (n + 1) = (j + n) + 1 by ring, hstep j (j + n) (Nat.le_add_right j n),
          ih j, pow_succ]
        ring
  intro j k
  rcases le_total j k with h | h
  · have hd : ch03_absdiff j k = k - j := by simp only [ch03_absdiff]; omega
    rw [hd]
    have h2 := hkey (k - j) j
    rwa [Nat.add_sub_cancel' h] at h2
  · have hd : ch03_absdiff j k = j - k := by simp only [ch03_absdiff]; omega
    rw [hd, hsymm]
    have h2 := hkey (j - k) k
    rwa [Nat.add_sub_cancel' h] at h2

-- eq:ar1-toeplitz: the AR(1) covariance matrix is the Toeplitz matrix of
-- eq:toeplitz for the lag function `γ(d) = α^{|d|}`.
theorem ch03_ar1_is_toeplitz (α : ℝ) (L : ℕ) (i j : Fin L) :
    ch03_toeplitz L (fun d => α ^ d.natAbs) i j = ch03_ar1_cov α (i : ℕ) (j : ℕ) := by
  simp [ch03_toeplitz, ch03_ar1_cov, ch03_absdiff_eq_natAbs]

-- eq:ar1-toeplitz: the explicit 5×5 instance of the displayed matrix.
theorem ch03_ar1_toeplitz_display (α : ℝ) :
    (Matrix.of fun i j : Fin 5 => ch03_ar1_cov α (i : ℕ) (j : ℕ)) =
      !![1,     α,     α ^ 2, α ^ 3, α ^ 4;
         α,     1,     α,     α ^ 2, α ^ 3;
         α ^ 2, α,     1,     α,     α ^ 2;
         α ^ 3, α ^ 2, α,     1,     α;
         α ^ 4, α ^ 3, α ^ 2, α,     1] := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [ch03_ar1_cov, ch03_absdiff]

/-! #### eq:ar1-toeplitz derived from the process, not from assumed moment recursions

`ch03_ar1_toeplitz` above is a *uniqueness* statement: it shows nothing but `α^{|j-k|}`
satisfies the two second-moment recursions.  What follows closes the remaining gap by
deriving those recursions from the model of eq:ar1 itself — the pathwise recursion
`a_{k+1} = α a_k + η_k`, independence of each innovation from the past states, and the
initialisation `Var(a₀) = 1`, `Var(η_k) = q = 1 - α²` of lines 98-100.
-/

section AR1Process

variable {Ω : Type*} [MeasurableSpace Ω]

/-- The Gaussian AR(1) chain of eq:ar1 as hypotheses on a genuine family of random
variables, with the stationary initialisation of ch03 lines 98-100.  Only second-order
information is used, so Gaussianity is not assumed; `hindep` is the "innovations are
independent of the past" clause carried by `η_k ~iid` together with `η ⟂ a₀`. -/
structure ch03_AR1 (α : ℝ) (a η : ℕ → Ω → ℝ) (μ : Measure Ω) : Prop where
  /-- each state is square-integrable -/
  memLp_state : ∀ k, MemLp (a k) 2 μ
  /-- each innovation is square-integrable -/
  memLp_noise : ∀ k, MemLp (η k) 2 μ
  /-- eq:ar1 itself, pathwise -/
  recursion : ∀ k ω, a (k + 1) ω = α * a k ω + η k ω
  /-- `η_k` is independent of every state it has not yet influenced -/
  indep : ∀ j k, j ≤ k → IndepFun (a j) (η k) μ
  /-- `a₀ ∼ N(0,1)`, at the level of second moments -/
  var_init : Var[a 0; μ] = 1
  /-- `η_k ∼ N(0, q)` with `q = 1 - α²` (ch03 line 99) -/
  var_noise : ∀ k, Var[η k; μ] = 1 - α ^ 2

variable {α : ℝ} {a η : ℕ → Ω → ℝ} {μ : Measure Ω}

/-- eq:ar1 rewritten as an equation between functions, ready for the bilinearity of `cov`. -/
theorem ch03_ar1_succ_fun (H : ch03_AR1 α a η μ) (k : ℕ) :
    a (k + 1) = (fun ω => α * a k ω) + η k :=
  funext fun ω => H.recursion k ω

/-- The cross-covariance recursion `Cov(a_j, a_{k+1}) = α Cov(a_j, a_k)` (`j ≤ k`),
*derived* from eq:ar1 and `η_k ⟂ a_j`. -/
theorem ch03_ar1_cov_step_proc [IsFiniteMeasure μ] (H : ch03_AR1 α a η μ) {j k : ℕ}
    (hjk : j ≤ k) :
    cov[a j, a (k + 1); μ] = α * cov[a j, a k; μ] := by
  rw [ch03_ar1_succ_fun H k,
    covariance_add_right (H.memLp_state j) ((H.memLp_state k).const_mul α) (H.memLp_noise k),
    covariance_const_mul_right,
    (H.indep j k hjk).covariance_eq_zero (H.memLp_state j) (H.memLp_noise k), add_zero]

/-- The variance recursion `Var(a_{k+1}) = α² Var(a_k) + (1 - α²)`, *derived* from eq:ar1,
`η_k ⟂ a_k` and the choice `q = 1 - α²` of line 99. -/
theorem ch03_ar1_var_step_proc [IsFiniteMeasure μ] (H : ch03_AR1 α a η μ) (k : ℕ) :
    cov[a (k + 1), a (k + 1); μ] = α ^ 2 * cov[a k, a k; μ] + (1 - α ^ 2) := by
  have hmA : MemLp (fun ω => α * a k ω) 2 μ := (H.memLp_state k).const_mul α
  have h0 : cov[a k, η k; μ] = 0 :=
    (H.indep k k le_rfl).covariance_eq_zero (H.memLp_state k) (H.memLp_noise k)
  have h0' : cov[η k, a k; μ] = 0 := by rw [covariance_comm]; exact h0
  have hηη : cov[η k, η k; μ] = 1 - α ^ 2 := by
    rw [covariance_self (H.memLp_noise k).aestronglyMeasurable.aemeasurable, H.var_noise k]
  rw [ch03_ar1_succ_fun H k,
    covariance_add_left hmA (H.memLp_noise k) (hmA.add (H.memLp_noise k)),
    covariance_add_right hmA hmA (H.memLp_noise k),
    covariance_add_right (H.memLp_noise k) hmA (H.memLp_noise k)]
  simp only [covariance_const_mul_left, covariance_const_mul_right, h0, h0', hηη]
  ring

/-- eq:ar1-toeplitz (lines 101-112), MAIN CLAIM, general form: for *any* process
satisfying the model of eq:ar1 with the stationary initialisation of lines 98-100,
`Cov(a_j, a_k) = α^{|j-k|}` at every pair of indices. -/
theorem ch03_ar1_toeplitz_proc [IsFiniteMeasure μ] (H : ch03_AR1 α a η μ) (j k : ℕ) :
    cov[a j, a k; μ] = α ^ ch03_absdiff j k :=
  ch03_ar1_toeplitz α (fun j k => cov[a j, a k; μ])
    (fun p q => covariance_comm (X := a p) (Y := a q))
    (by
      rw [covariance_self (H.memLp_state 0).aestronglyMeasurable.aemeasurable]
      exact H.var_init)
    (fun k => ch03_ar1_var_step_proc H k) (fun _ _ h => ch03_ar1_cov_step_proc H h) j k

/-- eq:ar1-toeplitz: the length-`L` covariance matrix of the process is *exactly* the
Toeplitz matrix `Σ` displayed in the chapter, at every `L`. -/
theorem ch03_ar1_toeplitz_matrix_proc [IsFiniteMeasure μ] (H : ch03_AR1 α a η μ) (L : ℕ) :
    (Matrix.of fun i j : Fin L => cov[a (i : ℕ), a (j : ℕ); μ])
      = ch03_toeplitz L (fun d => α ^ d.natAbs) := by
  ext i j
  rw [Matrix.of_apply, ch03_ar1_is_toeplitz]
  exact ch03_ar1_toeplitz_proc H (i : ℕ) (j : ℕ)

/-! ##### The independence clause is itself a consequence of `η ~iid`, `η ⟂ a₀`

`ch03_AR1.indep` above is not an extra modelling assumption: it follows from eq:ar1
together with joint independence of the driving vector `(a₀, η₀, η₁, …)`, because `a_j`
is a fixed affine function of `a₀, η₀, …, η_{j-1}` only.
-/

/-- The affine map that produces `a_m` from the driving vector `z = (a₀, η₀, η₁, …)`. -/
noncomputable def ch03_ar1_lin (α : ℝ) : ℕ → (ℕ → ℝ) → ℝ
  | 0 => fun z => z 0
  | (m + 1) => fun z => α * ch03_ar1_lin α m z + z (m + 1)

theorem ch03_ar1_lin_measurable (α : ℝ) (m : ℕ) : Measurable (ch03_ar1_lin α m) := by
  induction m with
  | zero => exact measurable_pi_apply 0
  | succ n ih =>
      exact (ih.const_mul α).add (measurable_pi_apply (n + 1))

/-- `ch03_ar1_lin α m` only reads the coordinates `0, …, m`. -/
theorem ch03_ar1_lin_congr (α : ℝ) (m : ℕ) {z z' : ℕ → ℝ} (h : ∀ i ≤ m, z i = z' i) :
    ch03_ar1_lin α m z = ch03_ar1_lin α m z' := by
  induction m with
  | zero => simp [ch03_ar1_lin, h 0 le_rfl]
  | succ n ih =>
      simp only [ch03_ar1_lin]
      rw [ih fun i hi => h i (hi.trans (Nat.le_succ n)), h (n + 1) le_rfl]

/-- The driving vector of the chain: `Z 0 = a₀` and `Z (i+1) = η i`. -/
def ch03_ar1_drive (a η : ℕ → Ω → ℝ) : ℕ → Ω → ℝ
  | 0 => a 0
  | (i + 1) => η i

/-- Solving eq:ar1: `a_m` is the fixed affine function `ch03_ar1_lin α m` of the driving
vector, so it depends only on `a₀, η₀, …, η_{m-1}`. -/
theorem ch03_ar1_state_eq_lin {α : ℝ} {a η : ℕ → Ω → ℝ}
    (hrec : ∀ k ω, a (k + 1) ω = α * a k ω + η k ω) (m : ℕ) (ω : Ω) :
    a m ω = ch03_ar1_lin α m fun i => ch03_ar1_drive a η i ω := by
  induction m with
  | zero => rfl
  | succ n ih => rw [hrec n ω, ih]; rfl

/-- The `indep` clause of `ch03_AR1`, derived: if the driving vector `(a₀, η₀, η₁, …)` is
jointly independent then every innovation `η_k` is independent of every earlier state
`a_j` (`j ≤ k`). -/
theorem ch03_ar1_indep_of_iIndepFun {α : ℝ} {a η : ℕ → Ω → ℝ} {μ : Measure Ω}
    (hrec : ∀ k ω, a (k + 1) ω = α * a k ω + η k ω)
    (hmeas : ∀ i, Measurable (ch03_ar1_drive a η i))
    (hdrive : iIndepFun (ch03_ar1_drive a η) μ) {j k : ℕ} (hjk : j ≤ k) :
    IndepFun (a j) (η k) μ := by
  classical
  have hdisj : Disjoint (Finset.range (j + 1)) ({k + 1} : Finset ℕ) := by
    simp only [Finset.disjoint_singleton_right, Finset.mem_range]
    omega
  have hbase := hdrive.indepFun_finset (Finset.range (j + 1)) ({k + 1} : Finset ℕ) hdisj hmeas
  set φ : ((Finset.range (j + 1) : Finset ℕ) → ℝ) → ℝ := fun v =>
    ch03_ar1_lin α j fun i => if h : i ∈ Finset.range (j + 1) then v ⟨i, h⟩ else 0 with hφ
  set ψ : (({k + 1} : Finset ℕ) → ℝ) → ℝ := fun v => v ⟨k + 1, Finset.mem_singleton_self _⟩ with hψ
  have hφm : Measurable φ := by
    refine (ch03_ar1_lin_measurable α j).comp (measurable_pi_lambda _ fun i => ?_)
    by_cases h : i ∈ Finset.range (j + 1)
    · simp only [dif_pos h]; exact measurable_pi_apply _
    · simp only [dif_neg h]; exact measurable_const
  have hψm : Measurable ψ := measurable_pi_apply _
  have hA : a j = φ ∘ fun ω (i : (Finset.range (j + 1) : Finset ℕ)) => ch03_ar1_drive a η i ω := by
    funext ω
    rw [ch03_ar1_state_eq_lin hrec j ω]
    refine ch03_ar1_lin_congr α j fun i hi => ?_
    have hmem : i ∈ Finset.range (j + 1) := Finset.mem_range.mpr (Nat.lt_succ_of_le hi)
    simp only [hφ, dif_pos hmem]
  have hB : η k = ψ ∘ fun ω (i : (({k + 1} : Finset ℕ) : Finset ℕ)) =>
      ch03_ar1_drive a η i ω := by
    funext ω; rfl
  rw [hA, hB]
  exact hbase.comp hφm hψm

/-- eq:ar1-toeplitz from the primitive hypotheses of eq:ar1 alone: the pathwise
recursion, joint independence of `(a₀, η₀, η₁, …)`, and the second moments of line 99. -/
theorem ch03_ar1_toeplitz_of_iIndepFun [IsFiniteMeasure μ]
    (hmemA : ∀ k, MemLp (a k) 2 μ) (hmemN : ∀ k, MemLp (η k) 2 μ)
    (hrec : ∀ k ω, a (k + 1) ω = α * a k ω + η k ω)
    (hmeas : ∀ i, Measurable (ch03_ar1_drive a η i))
    (hdrive : iIndepFun (ch03_ar1_drive a η) μ)
    (hvar0 : Var[a 0; μ] = 1) (hvarη : ∀ k, Var[η k; μ] = 1 - α ^ 2) (j k : ℕ) :
    cov[a j, a k; μ] = α ^ ch03_absdiff j k :=
  ch03_ar1_toeplitz_proc
    { memLp_state := hmemA
      memLp_noise := hmemN
      recursion := hrec
      indep := fun _ _ h => ch03_ar1_indep_of_iIndepFun hrec hmeas hdrive h
      var_init := hvar0
      var_noise := hvarη } j k

end AR1Process

/-! ##### `ch03_AR1` is inhabited, over exactly the chapter's parameter range

A hypothesis set that nothing satisfies would make `ch03_ar1_toeplitz_proc` vacuous.  It is
not: for every `α` with `α² ≤ 1` — precisely the range in which `q = 1 - α²` is a genuine
variance — a process satisfying every clause exists.
-/

section AR1Existence

/-- Rescaling of an iid standard-normal family: coordinate `0` keeps variance `1`
(the stationary initialisation `a₀ ∼ N(0,1)`), every later coordinate is scaled to
variance `q = 1 - α²` (the innovation law of line 99). -/
noncomputable def ch03_ar1_scale (r : ℝ) : ℕ → ℝ → ℝ
  | 0 => fun x => x
  | (_ + 1) => fun x => Real.sqrt (1 - r ^ 2) * x

theorem ch03_ar1_scale_measurable (r : ℝ) (i : ℕ) : Measurable (ch03_ar1_scale r i) := by
  cases i with
  | zero => exact measurable_id
  | succ n => exact measurable_const_mul _

/-- Any jointly independent driving vector with the right variances generates a process
satisfying the full model `ch03_AR1`: take `a_k = ch03_ar1_lin α k (Z · ω)`, the solution of
eq:ar1, and `η_k = Z_{k+1}`. -/
theorem ch03_AR1_of_drive {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} {r : ℝ}
    (Z : ℕ → Ω → ℝ) (hZmeas : ∀ i, Measurable (Z i)) (hZmem : ∀ i, MemLp (Z i) 2 P)
    (hZindep : iIndepFun Z P) (hZ0 : Var[Z 0; P] = 1)
    (hZs : ∀ i, Var[Z (i + 1); P] = 1 - r ^ 2) :
    ch03_AR1 r (fun k ω => ch03_ar1_lin r k fun i => Z i ω) (fun k => Z (k + 1)) P := by
  have hrec : ∀ k ω, ch03_ar1_lin r (k + 1) (fun i => Z i ω)
      = r * ch03_ar1_lin r k (fun i => Z i ω) + Z (k + 1) ω := fun _ _ => rfl
  have hdrive : ch03_ar1_drive (fun k ω => ch03_ar1_lin r k fun i => Z i ω)
      (fun k => Z (k + 1)) = Z := by
    funext i
    cases i with
    | zero => rfl
    | succ n => rfl
  have hmeas' : ∀ i, Measurable (ch03_ar1_drive
      (fun k ω => ch03_ar1_lin r k fun i => Z i ω) (fun k => Z (k + 1)) i) := by
    rw [hdrive]; exact hZmeas
  have hindep' : iIndepFun (ch03_ar1_drive
      (fun k ω => ch03_ar1_lin r k fun i => Z i ω) (fun k => Z (k + 1))) P := by
    rw [hdrive]; exact hZindep
  have hamem : ∀ k, MemLp (fun ω => ch03_ar1_lin r k fun i => Z i ω) 2 P := by
    intro k
    induction k with
    | zero => exact hZmem 0
    | succ n ih =>
        have h : (fun ω => ch03_ar1_lin r (n + 1) fun i => Z i ω)
            = (fun ω => r * ch03_ar1_lin r n fun i => Z i ω) + Z (n + 1) := rfl
        rw [h]
        exact (ih.const_mul r).add (hZmem (n + 1))
  exact
    { memLp_state := hamem
      memLp_noise := fun k => hZmem (k + 1)
      recursion := hrec
      indep := fun _ _ h => ch03_ar1_indep_of_iIndepFun hrec hmeas' hindep' h
      var_init := hZ0
      var_noise := hZs }

/-- ch03 lines 98-100, the distributional half: with `a₀ ∼ N(0,1)` and independent
innovations `η_k ∼ N(0, 1-α²)`, *every* state of the chain is again `N(0,1)` — the chain is
started in its own stationary law.  Proved by induction along eq:ar1. -/
theorem ch03_ar1_state_law {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} {r : ℝ}
    (hr : r ^ 2 ≤ 1) (Z : ℕ → Ω → ℝ) (hZmeas : ∀ i, Measurable (Z i))
    (hZindep : iIndepFun Z P)
    (hZ0 : P.map (Z 0) = gaussianReal 0 1)
    (hZs : ∀ i, P.map (Z (i + 1)) = gaussianReal 0 (1 - r ^ 2).toNNReal) (k : ℕ) :
    P.map (fun ω => ch03_ar1_lin r k fun i => Z i ω) = gaussianReal 0 1 := by
  have hq0 : (0 : ℝ) ≤ 1 - r ^ 2 := by linarith
  have hrec : ∀ m ω, ch03_ar1_lin r (m + 1) (fun i => Z i ω)
      = r * ch03_ar1_lin r m (fun i => Z i ω) + Z (m + 1) ω := fun _ _ => rfl
  have hdrive : ch03_ar1_drive (fun k ω => ch03_ar1_lin r k fun i => Z i ω)
      (fun k => Z (k + 1)) = Z := by
    funext i
    cases i with
    | zero => rfl
    | succ m => rfl
  have hmeas' : ∀ i, Measurable (ch03_ar1_drive
      (fun k ω => ch03_ar1_lin r k fun i => Z i ω) (fun k => Z (k + 1)) i) := by
    rw [hdrive]; exact hZmeas
  have hindep' : iIndepFun (ch03_ar1_drive
      (fun k ω => ch03_ar1_lin r k fun i => Z i ω) (fun k => Z (k + 1))) P := by
    rw [hdrive]; exact hZindep
  have hameas : ∀ m, Measurable fun ω => ch03_ar1_lin r m fun i => Z i ω := fun m =>
    (ch03_ar1_lin_measurable r m).comp (measurable_pi_lambda _ hZmeas)
  induction k with
  | zero => exact hZ0
  | succ n ih =>
      have hi0 : IndepFun (fun ω => ch03_ar1_lin r n fun i => Z i ω) (Z (n + 1)) P :=
        ch03_ar1_indep_of_iIndepFun hrec hmeas' hindep' (le_refl n)
      have hmapA := gaussianReal_map_const_mul (μ := 0) (v := 1) r
      rw [← ih, Measure.map_map (measurable_const_mul _) (hameas n)] at hmapA
      have hi : IndepFun ((fun x : ℝ => r * x) ∘ fun ω => ch03_ar1_lin r n fun i => Z i ω)
          (Z (n + 1)) P := hi0.comp (measurable_const_mul _) measurable_id
      have hsum := gaussianReal_add_gaussianReal_of_indepFun hi hmapA (hZs n)
      have hfun : (fun ω => ch03_ar1_lin r (n + 1) fun i => Z i ω)
          = ((fun x : ℝ => r * x) ∘ fun ω => ch03_ar1_lin r n fun i => Z i ω) + Z (n + 1) := rfl
      rw [hfun, hsum]
      congr 1
      · ring
      · apply NNReal.coe_injective
        simp only [NNReal.coe_add, NNReal.coe_mul, NNReal.coe_mk, NNReal.coe_one, mul_one]
        rw [Real.coe_toNNReal _ hq0]
        ring

/-- **`ch03_AR1` is not vacuous.**  For every `α` with `α² ≤ 1` there is a probability space
carrying a process that satisfies every clause of eq:ar1 with the initialisation of lines
98-100, and on it the displayed covariance really is `α^{|j-k|}`. -/
theorem ch03_AR1_exists (r : ℝ) (hr : r ^ 2 ≤ 1) :
    ∃ Ω : Type, ∃ _ : MeasurableSpace Ω, ∃ P : Measure Ω, ∃ a η : ℕ → Ω → ℝ,
      IsProbabilityMeasure P ∧ ch03_AR1 r a η P ∧
        (∀ k, P.map (a k) = gaussianReal 0 1) ∧
        ∀ j k, cov[a j, a k; P] = r ^ ch03_absdiff j k := by
  obtain ⟨Ω, mΩ, P, W, hWmeas, hWlaw, hWindep, hWprob⟩ := exists_iid ℕ (gaussianReal 0 1)
  haveI := hWprob
  have hq0 : (0 : ℝ) ≤ 1 - r ^ 2 := by linarith
  have hWmem : ∀ i, MemLp (W i) 2 P := fun i => by
    have h : MemLp id 2 (P.map (W i)) := by
      rw [(hWlaw i).map_eq]
      exact memLp_id_gaussianReal' 2 (by simp)
    exact h.comp_of_map (hWmeas i).aemeasurable
  have hWvar : ∀ i, Var[W i; P] = 1 := fun i => by
    rw [(hWlaw i).variance_eq, variance_id_gaussianReal]
    simp
  have hproc := ch03_AR1_of_drive (r := r) (fun i => ch03_ar1_scale r i ∘ W i)
    (fun i => (ch03_ar1_scale_measurable r i).comp (hWmeas i))
    (by
      intro i
      cases i with
      | zero => exact hWmem 0
      | succ n => exact (hWmem (n + 1)).const_mul _)
    (hWindep.comp _ (ch03_ar1_scale_measurable r))
    (hWvar 0)
    (by
      intro i
      show Var[fun ω => Real.sqrt (1 - r ^ 2) * W (i + 1) ω; P] = 1 - r ^ 2
      rw [variance_const_mul, hWvar (i + 1), Real.sq_sqrt hq0, mul_one])
  have hZlaw : ∀ i, P.map (ch03_ar1_scale r (i + 1) ∘ W (i + 1))
      = gaussianReal 0 (1 - r ^ 2).toNNReal := by
    intro i
    have h := gaussianReal_map_const_mul (μ := 0) (v := 1) (Real.sqrt (1 - r ^ 2))
    rw [← (hWlaw (i + 1)).map_eq, Measure.map_map (measurable_const_mul _) (hWmeas (i + 1))] at h
    rw [show (ch03_ar1_scale r (i + 1) ∘ W (i + 1))
        = ((fun x : ℝ => Real.sqrt (1 - r ^ 2) * x) ∘ W (i + 1)) from rfl, h]
    congr 1
    · ring
    · apply NNReal.coe_injective
      simp only [NNReal.coe_mul, NNReal.coe_mk, NNReal.coe_one, mul_one]
      rw [Real.sq_sqrt hq0, Real.coe_toNNReal _ hq0]
  exact ⟨Ω, mΩ, P, _, _, hWprob, hproc,
    ch03_ar1_state_law hr (fun i => ch03_ar1_scale r i ∘ W i)
      (fun i => (ch03_ar1_scale_measurable r i).comp (hWmeas i))
      (hWindep.comp _ (ch03_ar1_scale_measurable r)) (hWlaw 0).map_eq hZlaw,
    fun j k => ch03_ar1_toeplitz_proc hproc j k⟩

end AR1Existence


/-! ## §3.2 The Ornstein–Uhlenbeck corruption channel -/

-- eq:ou-sde (lines 138-142): `dX_t = -X_t dt + √2 dW_t`, `X_0 = a`.  The SDE itself is
-- a definition; its deterministic part (drift only, noise switched off) is the ODE
-- `x' = -x`, whose solution through `a` is `a e^{-t}` — the contraction factor of
-- eq:ou-channel.
theorem ch03_ou_sde_drift (a t : ℝ) :
    HasDerivAt (fun s : ℝ => a * Real.exp (-s)) (-(a * Real.exp (-t))) t := by
  have h2 : HasDerivAt (fun s : ℝ => Real.exp (-s)) (Real.exp (-t) * (-1)) t :=
    (Real.hasDerivAt_exp (-t)).comp t (hasDerivAt_neg t)
  have h3 := h2.const_mul a
  rw [show a * (Real.exp (-t) * (-1)) = -(a * Real.exp (-t)) by ring] at h3
  exact h3

theorem ch03_ou_sde_initial (a : ℝ) : a * Real.exp (-(0 : ℝ)) = a := by simp

-- noname-2 (lines 158-163): the integrating-factor product rule
-- `d(e^t X_t) = e^t X_t dt + e^t dX_t`, ordinary (not Itô) because `e^t` has no
-- stochastic part.
theorem ch03_noname_2_product_rule (X : ℝ → ℝ) (X' t : ℝ) (hX : HasDerivAt X X' t) :
    HasDerivAt (fun s => Real.exp s * X s) (Real.exp t * X t + Real.exp t * X') t :=
  (Real.hasDerivAt_exp t).mul hX

-- noname-2: the drift cancellation itself, as displayed —
-- `e^t X dt + e^t(-X dt + √2 dW) = √2 e^t dW`.
theorem ch03_noname_2_cancellation (X t dt dW : ℝ) :
    Real.exp t * X * dt + Real.exp t * (-(X * dt) + Real.sqrt 2 * dW)
      = Real.sqrt 2 * Real.exp t * dW := by ring

-- noname-2: consequence in the noiseless case — `e^t X_t` is constant, i.e. the
-- drift term cancels exactly.
theorem ch03_noname_2_drift_cancels (X : ℝ → ℝ) (t : ℝ) (hX : HasDerivAt X (-(X t)) t) :
    HasDerivAt (fun s => Real.exp s * X s) 0 t := by
  have h := (Real.hasDerivAt_exp t).mul hX
  rw [show Real.exp t * X t + Real.exp t * -(X t) = 0 by ring] at h
  exact h

-- noname-3 (lines 167-171): from `e^t X_t - a = I_t` to
-- `X_t = a e^{-t} + e^{-t} I_t`.
theorem ch03_noname_3 (a t I X : ℝ) (h : Real.exp t * X - a = I) :
    X = a * Real.exp (-t) + Real.exp (-t) * I := by
  have hX : Real.exp t * X = a + I := by linarith
  have key : Real.exp (-t) * (Real.exp t * X) = Real.exp (-t) * (a + I) := by rw [hX]
  rw [← mul_assoc, ← Real.exp_add, neg_add_cancel, Real.exp_zero, one_mul] at key
  rw [key]; ring

-- noname-4 (lines 180-189): the Itô-isometry variance computation
-- `Var(I_t) = 2e^{-2t} ∫_0^t e^{2s} ds = 2e^{-2t}(e^{2t}-1)/2 = 1-e^{-2t} = Δ_t`.
theorem ch03_noname_4_integral (t : ℝ) :
    (∫ s in (0 : ℝ)..t, Real.exp (2 * s)) = (Real.exp (2 * t) - 1) / 2 := by
  have hd : ∀ s ∈ Set.uIcc (0 : ℝ) t,
      HasDerivAt (fun u : ℝ => Real.exp (2 * u) / 2) (Real.exp (2 * s)) s := by
    intro s _
    have h1 : HasDerivAt (fun u : ℝ => 2 * u) 2 s := by
      simpa using (hasDerivAt_id s).const_mul (2 : ℝ)
    have h2 : HasDerivAt (fun u : ℝ => Real.exp (2 * u)) (Real.exp (2 * s) * 2) s :=
      (Real.hasDerivAt_exp (2 * s)).comp s h1
    have h3 := h2.div_const 2
    rw [show Real.exp (2 * s) * 2 / 2 = Real.exp (2 * s) by ring] at h3
    exact h3
  have hint : (∫ s in (0 : ℝ)..t, Real.exp (2 * s))
      = Real.exp (2 * t) / 2 - Real.exp (2 * 0) / 2 :=
    intervalIntegral.integral_eq_sub_of_hasDerivAt hd
      (by apply Continuous.intervalIntegrable; fun_prop)
  rw [hint, mul_zero, Real.exp_zero]
  ring

theorem ch03_noname_4 (t : ℝ) :
    2 * Real.exp (-2 * t) * (∫ s in (0 : ℝ)..t, Real.exp (2 * s)) = ch03_Delta t := by
  rw [ch03_noname_4_integral, ch03_Delta]
  have hcancel : Real.exp (-2 * t) * Real.exp (2 * t) = 1 := by
    rw [← Real.exp_add]; norm_num
  linear_combination hcancel

-- noname-5 (lines 193-196) and eq:ou-channel (lines 202-206):
-- `x = a e^{-t} + √Δ_t ε` with `ε ~ N(0,1)` has law `N(a e^{-t}, Δ_t)`.
theorem ch03_ou_channel_law {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω) (ε : Ω → ℝ)
    (hmeas : Measurable ε) (hε : P.map ε = gaussianReal 0 1) (a t : ℝ)
    (v : NNReal) (hv : (v : ℝ) = ch03_Delta t) :
    P.map (fun ω => a * Real.exp (-t) + Real.sqrt (ch03_Delta t) * ε ω)
      = gaussianReal (a * Real.exp (-t)) v := by
  have ht : 0 ≤ ch03_Delta t := hv ▸ v.2
  have h1 : (fun ω => a * Real.exp (-t) + Real.sqrt (ch03_Delta t) * ε ω)
      = (fun z : ℝ => a * Real.exp (-t) + z) ∘
          ((fun z : ℝ => Real.sqrt (ch03_Delta t) * z) ∘ ε) := rfl
  rw [h1, ← Measure.map_map (by fun_prop) (by fun_prop),
    ← Measure.map_map (by fun_prop) hmeas, hε,
    show (fun z : ℝ => Real.sqrt (ch03_Delta t) * z) = (Real.sqrt (ch03_Delta t) * ·) from rfl,
    gaussianReal_map_const_mul,
    show (fun z : ℝ => a * Real.exp (-t) + z) = (a * Real.exp (-t) + ·) from rfl,
    gaussianReal_map_const_add]
  congr 1
  · simp
  · ext
    rw [hv]
    simp [Real.sq_sqrt ht]

-- eq:ou-channel: at `t = 0` the channel is the identity (Δ₀ = 0, e^{-0} = 1).
theorem ch03_ou_channel_at_zero (a ε : ℝ) :
    a * Real.exp (-(0 : ℝ)) + Real.sqrt (ch03_Delta 0) * ε = a := by
  simp [ch03_Delta]

-- eq:ou-channel: as `t` grows the channel forgets `a` — Δ_t increases towards 1,
-- with the contraction factor `e^{-t}` shrinking.
theorem ch03_ou_channel_mono {s t : ℝ} (h : s ≤ t) : ch03_Delta s ≤ ch03_Delta t := by
  have : Real.exp (-2 * t) ≤ Real.exp (-2 * s) := Real.exp_le_exp.mpr (by linarith)
  simp only [ch03_Delta]; linarith

-- eq:ou-channel: `0 ≤ Δ_t < 1` for `t > 0`, so the channel really is a proper
-- interpolation between the data law and N(0,1).
theorem ch03_ou_channel_delta_mem (t : ℝ) (ht : 0 ≤ t) : 0 ≤ ch03_Delta t ∧ ch03_Delta t < 1 := by
  constructor
  · have : Real.exp (-2 * t) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith)
    simp only [ch03_Delta]; linarith
  · have : 0 < Real.exp (-2 * t) := Real.exp_pos _
    simp only [ch03_Delta]; linarith

/-! ## §3.3–3.4 The joint score -/

-- eq:dev-marginal-score (lines 245-248): the per-frame marginal score
-- `s(x_u, u, t) = ∂_{x_u} log p_t(x_u | u)`.
noncomputable def ch03_dev_marginal_score (p : ℝ → ℝ → ℝ → ℝ) (xu u t : ℝ) : ℝ :=
  deriv (fun y => Real.log (p y u t)) xu

-- eq:dev-joint-score (lines 274-279) and noname-6 (lines 281-285): the joint score
-- `s_t(x) = ∇_x log p_t(x_0,…,x_{L-1})`, k-th block; and the marginal score, which
-- by construction is a function of `x_k` alone.
noncomputable def ch03_dev_joint_score {n : ℕ} (p : (Fin n → ℝ) → ℝ)
    (x : Fin n → ℝ) (k : Fin n) : ℝ :=
  deriv (fun y => Real.log (p (Function.update x k y))) (x k)

noncomputable def ch03_dev_marg_score_block (q : ℝ → ℝ) (xk : ℝ) : ℝ :=
  deriv (fun y => Real.log (q y)) xk

/-- Matrix-vector product, entrywise. -/
theorem ch03_mulVec_apply {n : ℕ} (M : Matrix (Fin n) (Fin n) ℝ) (v : Fin n → ℝ) (i : Fin n) :
    (M *ᵥ v) i = ∑ j, M i j * v j := rfl

-- Derivative of a symmetric quadratic form in the k-th coordinate: the computational
-- core behind the score formulas eq:dev-L2-score and eq:dev-general-gaussian.
theorem ch03_quadform_hasDerivAt {n : ℕ} (S : Matrix (Fin n) (Fin n) ℝ)
    (hS : ∀ i j, S i j = S j i) (x : Fin n → ℝ) (k : Fin n) :
    HasDerivAt
      (fun y : ℝ => ∑ i, ∑ j, S i j * Function.update x k y i * Function.update x k y j)
      (2 * (S *ᵥ x) k) (x k) := by
  have hupd : Function.update x k (x k) = x := Function.update_eq_self k x
  have hu : ∀ i : Fin n,
      HasDerivAt (fun y : ℝ => Function.update x k y i) (if i = k then (1 : ℝ) else 0) (x k) := by
    intro i
    by_cases h : i = k
    · subst h
      simp only [Function.update_self, if_pos rfl]
      exact hasDerivAt_id _
    · simp only [Function.update_of_ne h, if_neg h]
      exact hasDerivAt_const _ _
  have hterm : ∀ i : Fin n, ∀ j : Fin n,
      HasDerivAt (fun y : ℝ => S i j * Function.update x k y i * Function.update x k y j)
        (S i j * (if i = k then (1 : ℝ) else 0) * x j +
          S i j * x i * (if j = k then (1 : ℝ) else 0)) (x k) := by
    intro i j
    have h2 := ((hu i).const_mul (S i j)).mul (hu j)
    rw [hupd] at h2
    exact h2
  have hsum : HasDerivAt
      (fun y : ℝ => ∑ i, ∑ j, S i j * Function.update x k y i * Function.update x k y j)
      (∑ i : Fin n, ∑ j : Fin n,
        (S i j * (if i = k then (1 : ℝ) else 0) * x j +
          S i j * x i * (if j = k then (1 : ℝ) else 0))) (x k) := by
    have key : (fun y : ℝ => ∑ i, ∑ j, S i j * Function.update x k y i * Function.update x k y j)
        = ∑ i : Fin n, ∑ j : Fin n,
            (fun y : ℝ => S i j * Function.update x k y i * Function.update x k y j) := by
      funext y
      simp only [Finset.sum_apply]
    rw [key]
    exact HasDerivAt.sum (fun i _ => HasDerivAt.sum (fun j _ => hterm i j))
  have e1 : (∑ i : Fin n, ∑ j : Fin n, S i j * (if i = k then (1 : ℝ) else 0) * x j)
      = (S *ᵥ x) k := by
    have step : ∀ i : Fin n, (∑ j : Fin n, S i j * (if i = k then (1 : ℝ) else 0) * x j)
        = if i = k then ∑ j : Fin n, S i j * x j else 0 := by
      intro i; by_cases h : i = k <;> simp [h]
    calc (∑ i : Fin n, ∑ j : Fin n, S i j * (if i = k then (1 : ℝ) else 0) * x j)
        = ∑ i : Fin n, (if i = k then ∑ j : Fin n, S i j * x j else 0) :=
          Finset.sum_congr rfl (fun i _ => step i)
      _ = ∑ j : Fin n, S k j * x j := by
          rw [Finset.sum_ite_eq' Finset.univ k (fun i => ∑ j : Fin n, S i j * x j)]
          simp
      _ = (S *ᵥ x) k := (ch03_mulVec_apply S x k).symm
  have e2 : (∑ i : Fin n, ∑ j : Fin n, S i j * x i * (if j = k then (1 : ℝ) else 0))
      = (S *ᵥ x) k := by
    have step : ∀ i : Fin n, (∑ j : Fin n, S i j * x i * (if j = k then (1 : ℝ) else 0))
        = S i k * x i := by
      intro i
      simp only [mul_ite, mul_one, mul_zero]
      rw [Finset.sum_ite_eq' Finset.univ k (fun j => S i j * x i)]
      simp
    calc (∑ i : Fin n, ∑ j : Fin n, S i j * x i * (if j = k then (1 : ℝ) else 0))
        = ∑ i : Fin n, S i k * x i := Finset.sum_congr rfl (fun i _ => step i)
      _ = ∑ i : Fin n, S k i * x i := Finset.sum_congr rfl (fun i _ => by rw [hS i k])
      _ = (S *ᵥ x) k := (ch03_mulVec_apply S x k).symm
  convert hsum using 1
  simp only [Finset.sum_add_distrib]
  rw [e1, e2]
  ring

/-- Log-density of a centred Gaussian with precision matrix `S = Σ⁻¹`, up to the
additive normalising constant `C`. -/
noncomputable def ch03_gauss_logdensity {n : ℕ} (S : Matrix (Fin n) (Fin n) ℝ) (C : ℝ)
    (z : Fin n → ℝ) : ℝ :=
  C - (1 / 2) * ∑ i, ∑ j, S i j * z i * z j

-- eq:dev-general-gaussian (lines 410-420), boxed, third identity:
-- `s_t(x) = -Σ_t^{-1} x` for a centred Gaussian, componentwise.
theorem ch03_dev_general_gaussian_score {n : ℕ} (S : Matrix (Fin n) (Fin n) ℝ)
    (hS : ∀ i j, S i j = S j i) (C : ℝ) (x : Fin n → ℝ) (k : Fin n) :
    ch03_dev_joint_score (fun z => Real.exp (ch03_gauss_logdensity S C z)) x k
      = -((S *ᵥ x) k) := by
  have hlog : (fun y : ℝ => Real.log (Real.exp (ch03_gauss_logdensity S C (Function.update x k y))))
      = fun y : ℝ => ch03_gauss_logdensity S C (Function.update x k y) := by
    funext y; rw [Real.log_exp]
  have hd : HasDerivAt (fun y : ℝ => ch03_gauss_logdensity S C (Function.update x k y))
      (-((S *ᵥ x) k)) (x k) := by
    have h := ((ch03_quadform_hasDerivAt S hS x k).const_mul (1 / 2 : ℝ)).const_sub C
    rw [show -((1 : ℝ) / 2 * (2 * (S *ᵥ x) k)) = -((S *ᵥ x) k) by ring] at h
    exact h
  rw [ch03_dev_joint_score, hlog]
  exact hd.deriv

/-- `Σ₀` for the stationary AR(1) chain: `(Σ₀)_{ij} = α^{|i-j|}` (eq:dev-general-gaussian). -/
noncomputable def ch03_Sigma0 (α : ℝ) (L : ℕ) : Matrix (Fin L) (Fin L) ℝ :=
  Matrix.of fun i j => α ^ ch03_absdiff (i : ℕ) (j : ℕ)

/-- `Σ_t = e^{-2t} Σ₀ + Δ_t I` (eq:dev-general-gaussian). -/
noncomputable def ch03_Sigmat (α t : ℝ) (L : ℕ) : Matrix (Fin L) (Fin L) ℝ :=
  Real.exp (-2 * t) • ch03_Sigma0 α L + ch03_Delta t • (1 : Matrix (Fin L) (Fin L) ℝ)

theorem ch03_Sigma0_symm (α : ℝ) (L : ℕ) (i j : Fin L) :
    ch03_Sigma0 α L i j = ch03_Sigma0 α L j i := by
  simp [ch03_Sigma0, ch03_absdiff_comm]

theorem ch03_Sigmat_symm (α t : ℝ) (L : ℕ) (i j : Fin L) :
    ch03_Sigmat α t L i j = ch03_Sigmat α t L j i := by
  simp only [ch03_Sigmat, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul]
  rw [ch03_Sigma0_symm]
  by_cases h : i = j
  · subst h; rfl
  · rw [Matrix.one_apply_ne h, Matrix.one_apply_ne (Ne.symm h)]

-- eq:dev-general-cov (lines 422-430): the entries of `Σ_t`, unit on the diagonal
-- (so every one-frame marginal is N(0,1), free of α) and `e^{-2t} α^{|i-j|}` off it.
theorem ch03_dev_general_cov (α t : ℝ) (L : ℕ) (i j : Fin L) :
    ch03_Sigmat α t L i j =
      if i = j then 1 else Real.exp (-2 * t) * α ^ ch03_absdiff (i : ℕ) (j : ℕ) := by
  by_cases h : i = j
  · subst h
    rw [if_pos rfl]
    simp only [ch03_Sigmat, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul, ch03_Sigma0,
      Matrix.of_apply, ch03_absdiff_self, pow_zero, Matrix.one_apply_eq, mul_one, ch03_Delta]
    ring
  · rw [if_neg h]
    simp only [ch03_Sigmat, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul, ch03_Sigma0,
      Matrix.of_apply, Matrix.one_apply_ne h, mul_zero, add_zero]

-- eq:dev-general-cov / noname-13 (lines 437-441): the diagonal is exactly 1, so the
-- one-frame marginal is the standard Gaussian for every α and every t.
theorem ch03_marginal_variance_one (α t : ℝ) (L : ℕ) (k : Fin L) :
    ch03_Sigmat α t L k k = 1 := by
  rw [ch03_dev_general_cov]; simp

theorem ch03_marginal_indep_of_alpha (α β t : ℝ) (L : ℕ) (k : Fin L) :
    ch03_Sigmat α t L k k = ch03_Sigmat β t L k k := by
  rw [ch03_marginal_variance_one, ch03_marginal_variance_one]

/-- The standard Gaussian density, the one-frame marginal of noname-13. -/
noncomputable def ch03_marginal_density (x : ℝ) : ℝ :=
  (1 / Real.sqrt (2 * Real.pi)) * Real.exp (-(x ^ 2 / 2))

-- noname-13 (lines 437-441): `p_{t,k}(x_k;α) = (2π)^{-1/2} e^{-x_k²/2}` carries no α.
theorem ch03_noname_13 (x α β : ℝ) :
    (fun _ : ℝ => ch03_marginal_density x) α = (fun _ : ℝ => ch03_marginal_density x) β := rfl

/-- The diagonal of `Σ₀`: `(Σ₀)_{kk} = α^{|k-k|} = α⁰ = 1`.  This is the *only* place `α`
enters a one-frame marginal, and it is where `α` cancels. -/
theorem ch03_Sigma0_diag (α : ℝ) (L : ℕ) (k : Fin L) : ch03_Sigma0 α L k k = 1 := by
  simp [ch03_Sigma0, ch03_absdiff_self]

/-- noname-13 (lines 437-441) at the level of laws, and the *premise* of
eq:dev-fisher-zero: if frame `k` of the clean chain carries the stationary marginal
`N(0, (Σ₀)_{kk})` — this is where `α` would enter — and the OU channel of eq:ou-channel
adds `√Δ_t` times an independent standard normal, then the corrupted frame
`X_k = e^{-t} a_k + √Δ_t Z` has law `N(0,1)` for every `α` and every `t ≥ 0`.
The `α`-dependence cancels because `(Σ₀)_{kk} = α⁰ = 1` and `e^{-2t} + Δ_t = 1`. -/
theorem ch03_ou_frame_law {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω)
    (A Z : Ω → ℝ) (α t : ℝ) (L : ℕ) (k : Fin L) (ht : 0 ≤ t)
    (hAm : Measurable A) (hZm : Measurable Z)
    (hA : P.map A = gaussianReal 0 (ch03_Sigma0 α L k k).toNNReal)
    (hZ : P.map Z = gaussianReal 0 1)
    (hindep : IndepFun A Z P) :
    P.map (fun ω => Real.exp (-t) * A ω + Real.sqrt (ch03_Delta t) * Z ω) = gaussianReal 0 1 := by
  have hΔ : 0 ≤ ch03_Delta t := by
    have h : Real.exp (-2 * t) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith)
    simp only [ch03_Delta]
    linarith
  have hexp : Real.exp (-t) ^ 2 = Real.exp (-2 * t) := by
    rw [sq, ← Real.exp_add]
    congr 1
    ring
  have hmapA := gaussianReal_map_const_mul (μ := 0)
    (v := (ch03_Sigma0 α L k k).toNNReal) (Real.exp (-t))
  rw [← hA, Measure.map_map (measurable_const_mul _) hAm] at hmapA
  have hmapZ := gaussianReal_map_const_mul (μ := 0) (v := 1) (Real.sqrt (ch03_Delta t))
  rw [← hZ, Measure.map_map (measurable_const_mul _) hZm] at hmapZ
  have hi : IndepFun ((fun x : ℝ => Real.exp (-t) * x) ∘ A)
      ((fun x : ℝ => Real.sqrt (ch03_Delta t) * x) ∘ Z) P :=
    hindep.comp (measurable_const_mul _) (measurable_const_mul _)
  have hsum := gaussianReal_add_gaussianReal_of_indepFun hi hmapA hmapZ
  have hfun : (fun ω => Real.exp (-t) * A ω + Real.sqrt (ch03_Delta t) * Z ω)
      = ((fun x : ℝ => Real.exp (-t) * x) ∘ A)
        + ((fun x : ℝ => Real.sqrt (ch03_Delta t) * x) ∘ Z) := rfl
  rw [hfun, hsum]
  congr 1
  · ring
  · rw [ch03_Sigma0_diag, Real.toNNReal_one, mul_one, mul_one]
    apply NNReal.coe_injective
    simp only [NNReal.coe_add, NNReal.coe_mk, NNReal.coe_one]
    rw [Real.sq_sqrt hΔ, hexp, ch03_Delta]
    ring

/-- The one-frame density of eq:dev-general-cov, *as a function of* `α`: the centred
Gaussian density whose variance is the `k`-th diagonal entry of `Σ_t = e^{-2t}Σ₀ + Δ_t I`.
Nothing here is assumed to be free of `α`; that is the content of the next lemma. -/
noncomputable def ch03_onemarginal_pdf (α t : ℝ) (L : ℕ) (k : Fin L) (x : ℝ) : ℝ :=
  gaussianPDFReal 0 (ch03_Sigmat α t L k k).toNNReal x

/-- noname-13 (lines 437-441): the one-frame density *derived from the model* really is
`(2π)^{-1/2} e^{-x²/2}`, at every `α`, `t`, `L` and frame `k`. -/
theorem ch03_onemarginal_pdf_eq (α t : ℝ) (L : ℕ) (k : Fin L) (x : ℝ) :
    ch03_onemarginal_pdf α t L k x = ch03_marginal_density x := by
  rw [ch03_onemarginal_pdf, ch03_marginal_variance_one, Real.toNNReal_one]
  simp only [gaussianPDFReal, ch03_marginal_density, NNReal.coe_one, mul_one, sub_zero,
    one_div]
  congr 1
  ring_nf

/-- noname-13: the `α`-slice of the one-frame density is a constant function of `α`.
Unlike the `α`-free definition it replaces, the left-hand side genuinely mentions `α`. -/
theorem ch03_onemarginal_pdf_const (t : ℝ) (L : ℕ) (k : Fin L) (x : ℝ) :
    (fun α : ℝ => ch03_onemarginal_pdf α t L k x) = fun _ : ℝ => ch03_marginal_density x :=
  funext fun α => ch03_onemarginal_pdf_eq α t L k x

-- eq:dev-fisher-zero (lines 443-454): `∂_α log p_{t,k} = 0`, hence `I_k(α) = 0`.
-- The differentiated function is the model's own one-frame density, a function of `α`
-- through `Σ_t`; its constancy is proved (`ch03_onemarginal_pdf_eq`), not assumed.
theorem ch03_dev_fisher_zero_deriv (t : ℝ) (L : ℕ) (k : Fin L) (x α : ℝ) :
    deriv (fun b : ℝ => Real.log (ch03_onemarginal_pdf b t L k x)) α = 0 := by
  have h : (fun b : ℝ => Real.log (ch03_onemarginal_pdf b t L k x))
      = fun _ : ℝ => Real.log (ch03_marginal_density x) :=
    funext fun b => by rw [ch03_onemarginal_pdf_eq]
  rw [h]
  exact deriv_const α _

/-- eq:dev-fisher-zero, boxed conclusion: `I_k(α) = E[(∂_α log p_{t,k}(X_k;α))²] = 0`,
for any observation `X` on any measure space. -/
theorem ch03_dev_fisher_zero {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω) (X : Ω → ℝ)
    (t α : ℝ) (L : ℕ) (k : Fin L) :
    (∫ ω, (deriv (fun b : ℝ => Real.log (ch03_onemarginal_pdf b t L k (X ω))) α) ^ 2 ∂P) = 0 := by
  simp [ch03_dev_fisher_zero_deriv]

/-- eq:dev-fisher-zero with the expectation taken under the frame's *own* law, the
`N(0,1)` of `ch03_ou_frame_law`. -/
theorem ch03_dev_fisher_zero_law (t α : ℝ) (L : ℕ) (k : Fin L) :
    (∫ x, (deriv (fun b : ℝ => Real.log (ch03_onemarginal_pdf b t L k x)) α) ^ 2
      ∂(gaussianReal 0 1)) = 0 :=
  ch03_dev_fisher_zero (gaussianReal 0 1) id t α L k

/-- `ch03_onemarginal_pdf α t L k` is a genuine density for the frame's own law: it is the
Radon–Nikodym derivative of `N(0,(Σ_t)_{kk})` against Lebesgue measure. -/
theorem ch03_onemarginal_pdf_rnDeriv (α t : ℝ) (L : ℕ) (k : Fin L) :
    (gaussianReal 0 (ch03_Sigmat α t L k k).toNNReal).rnDeriv volume
      =ᵐ[volume] fun x => ENNReal.ofReal (ch03_onemarginal_pdf α t L k x) :=
  rnDeriv_gaussianReal 0 _

/-- eq:dev-general-cov, law form: the frame-`k` law of the corrupted chain does not depend
on `α` at all. -/
theorem ch03_onemarginal_law_indep_of_alpha (α β t : ℝ) (L : ℕ) (k : Fin L) :
    gaussianReal 0 (ch03_Sigmat α t L k k).toNNReal
      = gaussianReal 0 (ch03_Sigmat β t L k k).toNNReal := by
  rw [ch03_marginal_variance_one, ch03_marginal_variance_one]

-- eq:dev-joint-block (lines 262-265) and noname-15 (lines 510-516): Tweedie's identity
-- `S_k(x,t) = (e^{-t} E[a_k | x] - x_k)/Δ_t`, verified against the Gaussian posterior
-- mean `E[a | x] = e^{-t} Σ₀ Σ_t^{-1} x`: the right-hand side collapses to `-(Σ_t^{-1}x)_k`,
-- which is the joint score of eq:dev-general-gaussian.  Here `y = Σ_t^{-1} x`.
theorem ch03_dev_joint_block {n : ℕ} (S0 St : Matrix (Fin n) (Fin n) ℝ) (t : ℝ)
    (hS : St = Real.exp (-2 * t) • S0 + ch03_Delta t • (1 : Matrix (Fin n) (Fin n) ℝ))
    (hΔ : ch03_Delta t ≠ 0) (x y : Fin n → ℝ) (hy : St *ᵥ y = x) (k : Fin n) :
    (Real.exp (-t) * (Real.exp (-t) * (S0 *ᵥ y) k) - x k) / ch03_Delta t = -(y k) := by
  have hee : Real.exp (-t) * Real.exp (-t) = Real.exp (-2 * t) := by
    rw [← Real.exp_add]; ring_nf
  have hcomp : Real.exp (-2 * t) * (S0 *ᵥ y) k + ch03_Delta t * y k = x k := by
    rw [← hy, hS]
    simp [Matrix.add_mulVec, Matrix.smul_mulVec, Matrix.one_mulVec]
  have hnum : Real.exp (-t) * (Real.exp (-t) * (S0 *ᵥ y) k) - x k = -(ch03_Delta t * y k) := by
    rw [← mul_assoc, hee]
    linarith
  rw [hnum]
  field_simp

/-! ### The two-frame example (§3.4) -/

-- noname-7 (lines 298-312): `Σ₀ = [[1, α], [α, 1]]` for two consecutive states.
theorem ch03_noname_7 (α : ℝ) : ch03_Sigma0 α 2 = !![1, α; α, 1] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [ch03_Sigma0, ch03_absdiff]

theorem ch03_noname_7_det (α : ℝ) : (ch03_Sigma0 α 2).det = 1 - α ^ 2 := by
  rw [ch03_noname_7, Matrix.det_fin_two_of]; ring

-- noname-8 (lines 314-320): `x_i = e^{-t} a_i + √Δ_t z_i`, the channel applied
-- coordinatewise with independent noise.
noncomputable def ch03_noname_8 (t a z : ℝ) : ℝ :=
  Real.exp (-t) * a + Real.sqrt (ch03_Delta t) * z

-- noname-9-a (lines 322-325): `Var(x_i) = e^{-2t} Var(a_i) + Δ_t = e^{-2t} + 1 - e^{-2t} = 1`.
theorem ch03_noname_9a (t : ℝ) : Real.exp (-2 * t) * 1 + ch03_Delta t = 1 := by
  simp [ch03_Delta]

-- noname-9-b (lines 326-329): `Cov(x_0, x_1) = e^{-2t} Cov(a_0, a_1) = α e^{-2t}`.
noncomputable def ch03_r (α t : ℝ) : ℝ := α * Real.exp (-2 * t)

theorem ch03_noname_9b (α t : ℝ) : Real.exp (-2 * t) * α = ch03_r α t := by
  rw [ch03_r]; ring

-- noname-10 (lines 331-333): `r_t := α e^{-2t}` — definition, with the range it lives in.
theorem ch03_noname_10_bound (α t : ℝ) (hα : |α| < 1) (ht : 0 ≤ t) : |ch03_r α t| < 1 := by
  have h1 : |Real.exp (-2 * t)| ≤ 1 := by
    rw [abs_of_pos (Real.exp_pos _)]
    exact Real.exp_le_one_iff.mpr (by linarith)
  calc |ch03_r α t| = |α| * |Real.exp (-2 * t)| := by rw [ch03_r, abs_mul]
    _ ≤ |α| * 1 := by
        apply mul_le_mul_of_nonneg_left h1 (abs_nonneg α)
    _ < 1 := by simpa using hα

-- eq:dev-L2-cov (lines 335-349): `Σ_t = [[1, r_t], [r_t, 1]]`.
theorem ch03_dev_L2_cov (α t : ℝ) : ch03_Sigmat α t 2 = !![1, ch03_r α t; ch03_r α t, 1] := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [ch03_Sigmat, ch03_Sigma0, ch03_absdiff, ch03_Delta, ch03_r, Matrix.one_apply] <;>
    ring

-- eq:dev-L2-marginal (lines 352-359): each marginal is N(0,1), so the marginal
-- scores are `-x_0` and `-x_1`, with no α anywhere.
theorem ch03_dev_L2_marginal (x : ℝ) :
    ch03_dev_marg_score_block ch03_marginal_density x = -x := by
  have hsqrt : Real.sqrt (2 * Real.pi) ≠ 0 := by
    have : (0 : ℝ) < 2 * Real.pi := by positivity
    exact ne_of_gt (Real.sqrt_pos.mpr this)
  have hfun : (fun y => Real.log (ch03_marginal_density y))
      = fun y => -(y ^ 2 / 2) - Real.log (Real.sqrt (2 * Real.pi)) := by
    funext y
    rw [ch03_marginal_density, Real.log_mul (by positivity) (Real.exp_ne_zero _), Real.log_exp,
      one_div, Real.log_inv]
    ring
  have h1 : HasDerivAt (fun y : ℝ => y ^ 2 / 2) x x := by
    have h : HasDerivAt (fun y : ℝ => y ^ 2) (2 * x) x := by simpa using hasDerivAt_pow 2 x
    have h' := h.div_const 2
    rw [show 2 * x / 2 = x by ring] at h'
    exact h'
  have h2 : HasDerivAt (fun y : ℝ => -(y ^ 2 / 2) - Real.log (Real.sqrt (2 * Real.pi))) (-x) x :=
    h1.neg.sub_const _
  rw [ch03_dev_marg_score_block, hfun]
  exact h2.deriv

-- noname-11 (lines 363-371): `Σ_t^{-1} = (1-r²)^{-1} [[1, -r], [-r, 1]]`.
theorem ch03_noname_11 (r : ℝ) (h : (1 : ℝ) - r ^ 2 ≠ 0) :
    (!![1, r; r, 1] : Matrix (Fin 2) (Fin 2) ℝ)⁻¹
      = (1 / (1 - r ^ 2)) • !![1, -r; -r, 1] := by
  apply Matrix.inv_eq_right_inv
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [Matrix.mul_apply, Fin.sum_univ_two, Matrix.one_apply] <;>
    field_simp <;> ring

-- eq:dev-L2-density (lines 373-382), part 1: `det Σ_t = 1 - r_t²`.
theorem ch03_dev_L2_det (r : ℝ) :
    (!![1, r; r, 1] : Matrix (Fin 2) (Fin 2) ℝ).det = 1 - r ^ 2 := by
  rw [Matrix.det_fin_two_of]; ring

-- eq:dev-L2-density, part 2: the quadratic form in the exponent,
-- `xᵀ Σ_t^{-1} x = (x_0² - 2 r x_0 x_1 + x_1²)/(1-r²)`.
theorem ch03_dev_L2_quadform (r x0 x1 : ℝ) (h : (1 : ℝ) - r ^ 2 ≠ 0) :
    ![x0, x1] ⬝ᵥ ((!![1, r; r, 1] : Matrix (Fin 2) (Fin 2) ℝ)⁻¹ *ᵥ ![x0, x1])
      = (x0 ^ 2 - 2 * r * x0 * x1 + x1 ^ 2) / (1 - r ^ 2) := by
  rw [ch03_noname_11 r h]
  simp [Matrix.mulVec, dotProduct, Fin.sum_univ_two]
  ring

-- eq:dev-L2-density, part 3: assembling the two gives exactly the displayed density,
-- `(2π√(1-r²))^{-1} exp[-(x_0²-2r x_0x_1+x_1²)/(2(1-r²))]`, from the general centred
-- bivariate normal `(2π)^{-1}(det Σ)^{-1/2} exp(-½ xᵀΣ^{-1}x)`.
theorem ch03_dev_L2_density (r x0 x1 : ℝ) (h : (1 : ℝ) - r ^ 2 ≠ 0) :
    (2 * Real.pi)⁻¹ * (Real.sqrt (1 - r ^ 2))⁻¹ *
        Real.exp (-(1 / 2) * ((x0 ^ 2 - 2 * r * x0 * x1 + x1 ^ 2) / (1 - r ^ 2)))
      = 1 / (2 * Real.pi * Real.sqrt (1 - r ^ 2)) *
          Real.exp (-(x0 ^ 2 - 2 * r * x0 * x1 + x1 ^ 2) / (2 * (1 - r ^ 2))) := by
  have hkey : -(1 / 2 : ℝ) * ((x0 ^ 2 - 2 * r * x0 * x1 + x1 ^ 2) / (1 - r ^ 2))
      = -(x0 ^ 2 - 2 * r * x0 * x1 + x1 ^ 2) / (2 * (1 - r ^ 2)) := by
    rw [← div_div]; ring
  rw [hkey] <;> ring

/-- The two-frame Gaussian log-density of eq:dev-L2-density, as a function on `Fin 2 → ℝ`. -/
noncomputable def ch03_dev_L2_logdensity (r : ℝ) (z : Fin 2 → ℝ) : ℝ :=
  Real.log (1 / (2 * Real.pi * Real.sqrt (1 - r ^ 2)))
    - (z 0 ^ 2 - 2 * r * (z 0) * (z 1) + z 1 ^ 2) / (2 * (1 - r ^ 2))

-- noname-12-a (lines 385-387): `s_{t,0}(x_0,x_1) = -(x_0 - r_t x_1)/(1-r_t²)`,
-- obtained by differentiating the log of eq:dev-L2-density in `x_0`.
theorem ch03_noname_12a (r : ℝ) (h : (1 : ℝ) - r ^ 2 ≠ 0) (x : Fin 2 → ℝ) :
    deriv (fun y => ch03_dev_L2_logdensity r (Function.update x 0 y)) (x 0)
      = -((x 0 - r * x 1) / (1 - r ^ 2)) := by
  have e0 : ∀ y : ℝ, Function.update x (0 : Fin 2) y 0 = y := fun y => Function.update_self _ _ _
  have e1 : ∀ y : ℝ, Function.update x (0 : Fin 2) y 1 = x 1 := fun y =>
    Function.update_of_ne (by decide) _ _
  have hfun : (fun y => ch03_dev_L2_logdensity r (Function.update x 0 y))
      = fun y => Real.log (1 / (2 * Real.pi * Real.sqrt (1 - r ^ 2)))
          - (y ^ 2 - 2 * r * y * (x 1) + (x 1) ^ 2) / (2 * (1 - r ^ 2)) := by
    funext y; rw [ch03_dev_L2_logdensity, e0 y, e1 y]
  have hq : HasDerivAt (fun y : ℝ => y ^ 2 - 2 * r * y * (x 1) + (x 1) ^ 2)
      (2 * x 0 - 2 * r * (x 1)) (x 0) := by
    have ha : HasDerivAt (fun y : ℝ => y ^ 2) (2 * x 0) (x 0) := by
      simpa using hasDerivAt_pow 2 (x 0)
    have h0 : HasDerivAt (fun y : ℝ => 2 * r * y) (2 * r) (x 0) := by
      have hh := (hasDerivAt_id (x 0)).const_mul (2 * r)
      rw [mul_one] at hh
      exact hh
    have hb : HasDerivAt (fun y : ℝ => 2 * r * y * (x 1)) (2 * r * (x 1)) (x 0) :=
      h0.mul_const (x 1)
    exact (ha.sub hb).add_const ((x 1) ^ 2)
  have hd := (hq.div_const (2 * (1 - r ^ 2))).const_sub
    (Real.log (1 / (2 * Real.pi * Real.sqrt (1 - r ^ 2))))
  rw [hfun]
  rw [hd.deriv]
  field_simp

-- noname-12-b (lines 388-390): `s_{t,1}(x_0,x_1) = -(x_1 - r_t x_0)/(1-r_t²)`.
theorem ch03_noname_12b (r : ℝ) (h : (1 : ℝ) - r ^ 2 ≠ 0) (x : Fin 2 → ℝ) :
    deriv (fun y => ch03_dev_L2_logdensity r (Function.update x 1 y)) (x 1)
      = -((x 1 - r * x 0) / (1 - r ^ 2)) := by
  have e0 : ∀ y : ℝ, Function.update x (1 : Fin 2) y 0 = x 0 := fun y =>
    Function.update_of_ne (by decide) _ _
  have e1 : ∀ y : ℝ, Function.update x (1 : Fin 2) y 1 = y := fun y => Function.update_self _ _ _
  have hfun : (fun y => ch03_dev_L2_logdensity r (Function.update x 1 y))
      = fun y => Real.log (1 / (2 * Real.pi * Real.sqrt (1 - r ^ 2)))
          - ((x 0) ^ 2 - 2 * r * (x 0) * y + y ^ 2) / (2 * (1 - r ^ 2)) := by
    funext y; rw [ch03_dev_L2_logdensity, e0 y, e1 y]
  have hq : HasDerivAt (fun y : ℝ => (x 0) ^ 2 - 2 * r * (x 0) * y + y ^ 2)
      (0 - 2 * r * (x 0) + 2 * x 1) (x 1) := by
    have ha : HasDerivAt (fun y : ℝ => y ^ 2) (2 * x 1) (x 1) := by
      simpa using hasDerivAt_pow 2 (x 1)
    have hb : HasDerivAt (fun y : ℝ => 2 * r * (x 0) * y) (2 * r * (x 0)) (x 1) := by
      have hh := (hasDerivAt_id (x 1)).const_mul (2 * r * (x 0))
      rw [mul_one] at hh
      exact hh
    exact (((hasDerivAt_const (x 1) ((x 0) ^ 2)).sub hb).add ha)
  have hd := (hq.div_const (2 * (1 - r ^ 2))).const_sub
    (Real.log (1 / (2 * Real.pi * Real.sqrt (1 - r ^ 2))))
  rw [hfun]
  rw [hd.deriv]
  field_simp
  ring

-- eq:dev-L2-score (lines 393-403), boxed: the vector form
-- `s_t(x) = -(1-r²)^{-1} (x_0 - r x_1, x_1 - r x_0)` coincides with `-Σ_t^{-1} x`.
theorem ch03_dev_L2_score (r x0 x1 : ℝ) (h : (1 : ℝ) - r ^ 2 ≠ 0) :
    -((!![1, r; r, 1] : Matrix (Fin 2) (Fin 2) ℝ)⁻¹ *ᵥ ![x0, x1])
      = (-(1 / (1 - r ^ 2))) • ![x0 - r * x1, x1 - r * x0] := by
  rw [ch03_noname_11 r h]
  ext i
  fin_cases i <;>
    simp [Matrix.mulVec, dotProduct, Fin.sum_univ_two] <;> ring

-- eq:dev-joint-vs-marginal (lines 287-294), boxed: the joint score of frame 0 really
-- does move when only `x_1` moves (so it "may depend on the other frames"), while the
-- marginal score of frame 0 is by construction a function of `x_0` alone.
theorem ch03_dev_joint_vs_marginal (r : ℝ) (hr0 : r ≠ 0) (h : (1 : ℝ) - r ^ 2 ≠ 0) :
    ∃ x0 x1 x1' : ℝ,
      -((x0 - r * x1) / (1 - r ^ 2)) ≠ -((x0 - r * x1') / (1 - r ^ 2)) := by
  refine ⟨0, 0, 1, ?_⟩
  have h1 : -((0 - r * 0) / (1 - r ^ 2)) = 0 := by simp
  have h2 : -((0 - r * 1) / (1 - r ^ 2)) = r / (1 - r ^ 2) := by
    rw [mul_one, zero_sub, neg_div, neg_neg]
  rw [h1, h2]
  intro hcon
  rcases div_eq_zero_iff.mp hcon.symm with h4 | h4
  · exact hr0 h4
  · exact h h4

/-! ### Remark 3.x: what changes outside the Gaussian case -/

-- eq:dev-cf (lines 461-469): the characteristic function through the channel,
-- `φ_{X_t}(u) = φ_A(e^{-t} u) exp(-Δ_t u²/2)`.
theorem ch03_dev_cf {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]
    (A Z : Ω → ℝ) (hA : AEMeasurable A P) (hZ : AEMeasurable Z P) (t : ℝ)
    (ht : 0 ≤ ch03_Delta t) (hZlaw : P.map Z = gaussianReal 0 1)
    (hind : IndepFun (fun ω => Real.exp (-t) * A ω)
      (fun ω => Real.sqrt (ch03_Delta t) * Z ω) P) (u : ℝ) :
    charFun (P.map (fun ω => Real.exp (-t) * A ω + Real.sqrt (ch03_Delta t) * Z ω)) u
      = charFun (P.map A) (Real.exp (-t) * u)
        * Complex.exp (-(ch03_Delta t * u ^ 2 / 2)) := by
  have hm1 : AEMeasurable (fun ω => Real.exp (-t) * A ω) P := hA.const_mul _
  have hm2 : AEMeasurable (fun ω => Real.sqrt (ch03_Delta t) * Z ω) P := hZ.const_mul _
  have hmul := hind.charFun_map_fun_add_eq_mul hm1 hm2
  have h1 : charFun (P.map (fun ω => Real.exp (-t) * A ω)) u
      = charFun (P.map A) (Real.exp (-t) * u) := charFun_map_mul_comp hA _ _
  have h2 : charFun (P.map (fun ω => Real.sqrt (ch03_Delta t) * Z ω)) u
      = charFun (gaussianReal 0 1) (Real.sqrt (ch03_Delta t) * u) := by
    rw [charFun_map_mul_comp hZ, hZlaw]
  have hsq : ((Real.sqrt (ch03_Delta t) * u : ℝ) : ℂ) ^ 2 = ((ch03_Delta t * u ^ 2 : ℝ) : ℂ) := by
    rw [← Complex.ofReal_pow, mul_pow, Real.sq_sqrt ht]
  rw [congrFun hmul u, Pi.mul_apply, h1, h2, charFun_gaussianReal]
  congr 1
  rw [hsq]
  push_cast
  ring

-- eq:dev-cf specialised to `A ~ N(0,1)` (line 470-471): the two Gaussian factors
-- combine and the marginal is exactly standard normal for every t.
theorem ch03_dev_cf_gaussian (t u : ℝ) :
    Real.exp (-(Real.exp (-t) * u) ^ 2 / 2) * Real.exp (-(ch03_Delta t * u ^ 2 / 2))
      = Real.exp (-(u ^ 2 / 2)) := by
  rw [← Real.exp_add]
  congr 1
  have hee : Real.exp (-t) * Real.exp (-t) = Real.exp (-2 * t) := by
    rw [← Real.exp_add]; ring_nf
  simp only [ch03_Delta]
  linear_combination (-(u ^ 2) / 2) * hee

-- noname-14 (lines 474-482): for a unit-variance Laplace input, `φ_A(v) = (1+v²/2)^{-1}`,
-- the channel output has `φ_{X_t}(u) = exp[-Δ_t u²/2] / (1 + e^{-2t} u²/2)`.
theorem ch03_noname_14 (t u : ℝ) (φ : ℝ → ℝ) (hφ : ∀ v, φ v = (1 + v ^ 2 / 2)⁻¹) :
    φ (Real.exp (-t) * u) * Real.exp (-(ch03_Delta t * u ^ 2 / 2))
      = Real.exp (-(ch03_Delta t * u ^ 2 / 2)) / (1 + Real.exp (-2 * t) * u ^ 2 / 2) := by
  rw [hφ]
  have hee : Real.exp (-t) * Real.exp (-t) = Real.exp (-2 * t) := by
    rw [← Real.exp_add]; ring_nf
  rw [show (Real.exp (-t) * u) ^ 2 = Real.exp (-2 * t) * u ^ 2 by
    rw [mul_pow, pow_two, hee]]
  ring

-- noname-14 supporting check (line 472): the scale of a unit-variance Laplace really is
-- `b = 1/√2`, since `Var = 2b²`; that is what turns `φ_A(u) = (1+b²u²)^{-1}` into
-- `(1+u²/2)^{-1}`.
theorem ch03_laplace_unit_variance : 2 * (1 / Real.sqrt 2) ^ 2 = 1 := by
  rw [div_pow, Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2)]
  norm_num

/-! ## §3.5 Markov random fields and factor graphs -/

-- eq:hammersley-clifford (lines 533-537): the clique factorisation and the additive
-- energy form are the same object, `∏_C ψ_C = exp[-∑_C E_C]` with `ψ_C = e^{-E_C}`.
theorem ch03_hammersley_clifford {ι : Type*} (s : Finset ι) (E : ι → ℝ) (Z : ℝ) :
    (1 / Z) * ∏ C ∈ s, Real.exp (-(E C)) = (1 / Z) * Real.exp (-∑ C ∈ s, E C) := by
  rw [show (-∑ C ∈ s, E C) = ∑ C ∈ s, -(E C) by simp, Real.exp_sum]

/-! ## §3.6 Belief propagation -/

-- eq:bp-rules (lines 605-614), first rule: the variable-to-factor message multiplies
-- the incoming factor messages from all *other* neighbours ("exclude the recipient").
noncomputable def ch03_bp_var_to_fac {α ι : Type*} [DecidableEq ι]
    (nbr : Finset ι) (a : ι) (nuhat : ι → α → ℝ) (xi : α) : ℝ :=
  ∏ b ∈ nbr.erase a, nuhat b xi

-- eq:bp-rules, second rule, for the degree-2 (pairwise) factors of a chain: the
-- factor-to-variable message integrates the factor against the incoming variable messages.
noncomputable def ch03_bp_fac_to_var {α β : Type*} [Fintype β]
    (ψ : α → β → ℝ) (nu : β → ℝ) (xi : α) : ℝ :=
  ∑ xj, ψ xi xj * nu xj

-- eq:bp-rules: the belief at node i is the product of all incoming factor messages.
noncomputable def ch03_bp_belief {α ι : Type*} (nbr : Finset ι) (nuhat : ι → α → ℝ)
    (xi : α) : ℝ :=
  ∏ a ∈ nbr, nuhat a xi

-- eq:bp-rules sanity: a leaf variable node (single factor neighbour) sends the
-- constant message 1 — the empty product of the "exclude the recipient" rule.
theorem ch03_bp_leaf_message {α ι : Type*} [DecidableEq ι] (a : ι) (nuhat : ι → α → ℝ)
    (xi : α) : ch03_bp_var_to_fac {a} a nuhat xi = 1 := by
  simp [ch03_bp_var_to_fac]

-- eq:bp-rules / thm:tree-exact instance, two-variable chain: the single sweep computes
-- the exact (unnormalised) marginal ∑_{x_1} ψ(x_0,x_1).
theorem ch03_bp_exact_two {α β : Type*} [Fintype β] (ψ : α → β → ℝ) (x0 : α) :
    ch03_bp_fac_to_var ψ (fun _ => 1) x0 = ∑ x1, ψ x0 x1 := by
  simp [ch03_bp_fac_to_var]

-- eq:bp-rules / thm:tree-exact instance, three-variable chain: the belief at the middle
-- node — the product of the forward and backward messages — is the exact marginal
-- ∑_{x_0} ∑_{x_2} ψ_1(x_0,x_1) ψ_2(x_1,x_2) of the chain factorisation.
theorem ch03_bp_exact_three {α β γ : Type*} [Fintype α] [Fintype γ]
    (ψ1 : α → β → ℝ) (ψ2 : β → γ → ℝ) (x1 : β) :
    (∑ x0, ψ1 x0 x1) * (∑ x2, ψ2 x1 x2)
      = ∑ x0, ∑ x2, ψ1 x0 x1 * ψ2 x1 x2 :=
  Fintype.sum_mul_sum _ _

/-! ### eq:dev-propagator (lines 225-228): the propagator hypothesis, made definite

The chapter writes eq:dev-propagator as a *conjecture* — with `≈`, and an unspecified
operator `P_t` — and §3.3 then reports that the programme built to test it failed.  The
display therefore has no content as an asserted identity.  What does have content is the
chapter's verdict on it (lines 254-258): the quantity the six toy models actually computed,
the per-frame *marginal* score, had the frames' dependence removed from it, so "no
propagator could have been found in it".

Both halves of that verdict are proved below, for every sequence length and with `P_t` an
*arbitrary* function — no linearity, continuity or measurability assumed — so the negative
results are the strongest form of the statement and are not artefacts of a restricted
search space.  Note that the second half (for the joint score) goes beyond what the chapter
asserts; it refutes only the *pointwise, frame-local* propagator that eq:dev-propagator
literally writes, namely one that reconstructs the block at frame `u` from the value of the
block at frame `u-1` alone. -/

/-- eq:dev-propagator made definite: `P` *propagates* a family `s` of per-frame score
blocks when `s u x = P (s (u-1) x)` holds exactly, for every pair of consecutive frames and
every trajectory `x`.  Frame blocks are scalars, as they are for the scalar chains of this
chapter. -/
def ch03_IsPropagator {n : ℕ} (s : Fin n → (Fin n → ℝ) → ℝ) (P : ℝ → ℝ) : Prop :=
  ∀ j k : Fin n, (j : ℕ) + 1 = (k : ℕ) → ∀ x : Fin n → ℝ, s k x = P (s j x)

/-- `ch03_IsPropagator` is satisfiable, so the non-existence theorems below are statements
about these particular score families and not artefacts of an unsatisfiable definition: a
family whose block does not depend on the frame index is propagated by the identity. -/
theorem ch03_IsPropagator_id {n : ℕ} (f : (Fin n → ℝ) → ℝ) :
    ch03_IsPropagator (fun _ => f) id := fun _ _ _ _ => rfl

/-- The per-frame **marginal** score family of §3.3-§3.4 — the object the six toy models
computed.  By noname-13 every one-frame marginal of the corrupted AR(1) chain is `N(0,1)`
for every `α` and every `t`, so the block at frame `u` is a function of `x_u` alone. -/
noncomputable def ch03_margScoreFamily (n : ℕ) : Fin n → (Fin n → ℝ) → ℝ :=
  fun u x => ch03_dev_marg_score_block ch03_marginal_density (x u)

/-- The marginal score block at frame `u` is `x ↦ -x_u`: the same function at every frame,
for every `α` and every `t` (neither appears).  This is the formal content of "the frames'
dependence had been removed before the study began". -/
theorem ch03_margScoreFamily_apply (n : ℕ) (u : Fin n) (x : Fin n → ℝ) :
    ch03_margScoreFamily n u x = -(x u) :=
  ch03_dev_L2_marginal (x u)

/-- eq:dev-propagator, the chapter's verdict (lines 254-258): the per-frame marginal score
admits **no** propagator, for any sequence length `L ≥ 2` and any function `P` whatsoever.
The obstruction is exactly the one the chapter names: the block at frame `u-1` sees only
`x_{u-1}`, and `x_u` is free of it, so two trajectories agreeing at frame `u-1` and
differing at frame `u` force `P` to take two values at the same argument. -/
theorem ch03_dev_propagator_marginal_none {n : ℕ} (hn : 2 ≤ n) (P : ℝ → ℝ) :
    ¬ ch03_IsPropagator (ch03_margScoreFamily n) P := by
  intro H
  have h0 : (0 : ℕ) < n := by omega
  have h1 : (1 : ℕ) < n := by omega
  have hjk : ((⟨0, h0⟩ : Fin n) : ℕ) + 1 = ((⟨1, h1⟩ : Fin n) : ℕ) := rfl
  have hne : (⟨0, h0⟩ : Fin n) ≠ (⟨1, h1⟩ : Fin n) := by
    intro h
    have h' := congrArg Fin.val h
    simp at h'
  -- the all-zero trajectory pins `P 0 = 0`
  have e0 := H ⟨0, h0⟩ ⟨1, h1⟩ hjk (fun _ => 0)
  rw [ch03_margScoreFamily_apply, ch03_margScoreFamily_apply] at e0
  simp only [neg_zero] at e0
  -- a trajectory that is `0` at frame `0` and `1` at frame `1` pins `P 0 = -1`
  have e1 := H ⟨0, h0⟩ ⟨1, h1⟩ hjk (Function.update (fun _ => (0 : ℝ)) ⟨1, h1⟩ 1)
  rw [ch03_margScoreFamily_apply, ch03_margScoreFamily_apply] at e1
  rw [Function.update_self, Function.update_of_ne hne] at e1
  simp only [neg_zero] at e1
  rw [← e0] at e1
  norm_num at e1

/-- The **joint** score family of a centred Gaussian with precision matrix `S = Σ_t⁻¹`
(eq:dev-joint-score, evaluated by eq:dev-general-gaussian). -/
noncomputable def ch03_jointScoreFamily {n : ℕ} (S : Matrix (Fin n) (Fin n) ℝ) (C : ℝ) :
    Fin n → (Fin n → ℝ) → ℝ :=
  fun u x => ch03_dev_joint_score (fun z => Real.exp (ch03_gauss_logdensity S C z)) x u

theorem ch03_jointScoreFamily_apply {n : ℕ} (S : Matrix (Fin n) (Fin n) ℝ)
    (hsymm : ∀ i j, S i j = S j i) (C : ℝ) (u : Fin n) (x : Fin n → ℝ) :
    ch03_jointScoreFamily S C u x = -((S *ᵥ x) u) :=
  ch03_dev_general_gaussian_score S hsymm C x u

/-- The linear-algebra core of the joint-score half: if `S` is invertible then, for any two
distinct frames `j ≠ k`, no function `P` sends `-(S x)_j` to `-(S x)_k`.  Witnesses: `x = 0`
and `x = S⁻¹ e_k`, which agree in coordinate `j` of `S x` and differ in coordinate `k`. -/
theorem ch03_dev_propagator_gaussian_none {n : ℕ} (S : Matrix (Fin n) (Fin n) ℝ)
    (hS : IsUnit S.det) (j k : Fin n) (hjk : j ≠ k) (P : ℝ → ℝ) :
    ¬ (∀ x : Fin n → ℝ, -((S *ᵥ x) k) = P (-((S *ᵥ x) j))) := by
  intro H
  have hv : S *ᵥ (S⁻¹ *ᵥ (Function.update (fun _ => (0 : ℝ)) k 1))
      = Function.update (fun _ => (0 : ℝ)) k 1 := by
    rw [Matrix.mulVec_mulVec, Matrix.mul_nonsing_inv S hS, Matrix.one_mulVec]
  have e0 := H 0
  rw [Matrix.mulVec_zero] at e0
  simp only [Pi.zero_apply, neg_zero] at e0
  have e1 := H (S⁻¹ *ᵥ (Function.update (fun _ => (0 : ℝ)) k 1))
  rw [hv, Function.update_self, Function.update_of_ne hjk] at e1
  simp only [neg_zero] at e1
  rw [← e0] at e1
  norm_num at e1

/-- eq:dev-propagator for the **corrected** object of §3.4: the joint score of a
nondegenerate centred Gaussian trajectory admits no pointwise propagator either. -/
theorem ch03_dev_propagator_joint_none {n : ℕ} (hn : 2 ≤ n)
    (S : Matrix (Fin n) (Fin n) ℝ) (hsymm : ∀ i j, S i j = S j i) (hS : IsUnit S.det)
    (C : ℝ) (P : ℝ → ℝ) :
    ¬ ch03_IsPropagator (ch03_jointScoreFamily S C) P := by
  intro H
  have h0 : (0 : ℕ) < n := by omega
  have h1 : (1 : ℕ) < n := by omega
  have hjk : ((⟨0, h0⟩ : Fin n) : ℕ) + 1 = ((⟨1, h1⟩ : Fin n) : ℕ) := rfl
  have hne : (⟨0, h0⟩ : Fin n) ≠ (⟨1, h1⟩ : Fin n) := by
    intro h
    have h' := congrArg Fin.val h
    simp at h'
  refine ch03_dev_propagator_gaussian_none S hS ⟨0, h0⟩ ⟨1, h1⟩ hne P (fun x => ?_)
  have h := H ⟨0, h0⟩ ⟨1, h1⟩ hjk x
  rwa [ch03_jointScoreFamily_apply S hsymm C, ch03_jointScoreFamily_apply S hsymm C] at h

/-- `Σ_t⁻¹` really is invertible in the regime the chapter works in, so the hypothesis of
`ch03_dev_propagator_joint_none` is not vacuous: the two-frame precision matrix
`(1-r²)⁻¹ [[1,-r],[-r,1]]` of noname-11 has determinant `(1-r²)⁻¹ ≠ 0`. -/
theorem ch03_dev_L2_precision_isUnit (r : ℝ) (h : (1 : ℝ) - r ^ 2 ≠ 0) :
    IsUnit (((1 / (1 - r ^ 2)) • !![1, -r; -r, 1] : Matrix (Fin 2) (Fin 2) ℝ)).det := by
  rw [isUnit_iff_ne_zero, Matrix.det_smul, Matrix.det_fin_two_of]
  rw [show (1 : ℝ) * 1 - -r * -r = 1 - r ^ 2 by ring]
  rw [show (1 / (1 - r ^ 2)) ^ (Fintype.card (Fin 2)) * (1 - r ^ 2) = 1 / (1 - r ^ 2) by
    simp only [Fintype.card_fin]
    field_simp]
  exact one_div_ne_zero h

/-! ### The Laplace characteristic function (line 472), computed rather than assumed -/

/-- The Laplace density with scale `b > 0`: `f(y) = (2b)⁻¹ e^{-|y|/b}`. -/
noncomputable def ch03_laplace_density (b y : ℝ) : ℝ := (2 * b)⁻¹ * Real.exp (-|y| / b)

/-- The Laplace density is a probability density: it integrates to `1`. -/
theorem ch03_laplace_density_integral_one (b : ℝ) (hb : 0 < b) :
    (∫ y : ℝ, ch03_laplace_density b y) = 1 := by
  have hbne : b ≠ 0 := ne_of_gt hb
  have hgoal : (∫ y : ℝ, ch03_laplace_density b y)
      = ∫ y : ℝ, (2 * b)⁻¹ * Real.exp (-|y| / b) := by
    refine integral_congr_ae (Filter.Eventually.of_forall fun y => ?_)
    rw [ch03_laplace_density]
  rw [hgoal, integral_comp_abs (f := fun s : ℝ => (2 * b)⁻¹ * Real.exp (-s / b))]
  have hc : (∫ s : ℝ in Set.Ioi (0 : ℝ), (2 * b)⁻¹ * Real.exp (-s / b))
      = (2 * b)⁻¹ * ∫ s : ℝ in Set.Ioi (0 : ℝ), Real.exp (-b⁻¹ * s) := by
    rw [← integral_const_mul]
    refine setIntegral_congr_fun measurableSet_Ioi (fun s _ => ?_)
    rw [show -s / b = -b⁻¹ * s by field_simp]
  have hneg : -b⁻¹ < 0 := by
    have := inv_pos.mpr hb
    linarith
  rw [hc, integral_exp_mul_Ioi hneg 0]
  simp only [mul_zero, Real.exp_zero]
  field_simp

/-- Line 472, the analytic input: the characteristic function of a Laplace variable with
scale `b` is `(1 + b²u²)⁻¹`.  Proved from the definition, by splitting `ℝ` at `0` and
evaluating the two one-sided complex exponential integrals. -/
theorem ch03_laplace_charFun (b : ℝ) (hb : 0 < b) (u : ℝ) :
    (∫ y : ℝ, Complex.exp ((y * u : ℝ) * Complex.I) * (ch03_laplace_density b y : ℂ))
      = ((1 + b ^ 2 * u ^ 2)⁻¹ : ℝ) := by
  have hbne : b ≠ 0 := ne_of_gt hb
  have hbC : (b : ℂ) ≠ 0 := by exact_mod_cast hbne
  obtain ⟨a₁, ha₁⟩ : ∃ z : ℂ, z = (u : ℂ) * Complex.I + ((b⁻¹ : ℝ) : ℂ) := ⟨_, rfl⟩
  obtain ⟨a₂, ha₂⟩ : ∃ z : ℂ, z = (u : ℂ) * Complex.I - ((b⁻¹ : ℝ) : ℂ) := ⟨_, rfl⟩
  have hre₁ : a₁.re = b⁻¹ := by
    rw [ha₁]; simp [Complex.add_re, Complex.mul_re, Complex.I_re, Complex.I_im]
  have hre₂ : a₂.re = -b⁻¹ := by
    rw [ha₂]; simp [Complex.sub_re, Complex.mul_re, Complex.I_re, Complex.I_im]
  have hpos₁ : 0 < a₁.re := by rw [hre₁]; positivity
  have hneg₂ : a₂.re < 0 := by rw [hre₂, neg_lt, neg_zero]; positivity
  -- on each half-line the integrand is a single complex exponential
  have key₁ : ∀ y : ℝ, y ∈ Set.Iic (0 : ℝ) →
      Complex.exp ((y * u : ℝ) * Complex.I) * (ch03_laplace_density b y : ℂ)
        = (2 * (b : ℂ))⁻¹ * Complex.exp (a₁ * y) := by
    intro y hy
    have hre : -|y| / b = y / b := by rw [abs_of_nonpos hy]; ring
    have hEA : Complex.exp ((y : ℂ) * (u : ℂ) * Complex.I) * Complex.exp ((y : ℂ) / (b : ℂ))
        = Complex.exp (a₁ * (y : ℂ)) := by
      rw [← Complex.exp_add, ha₁]
      congr 1
      push_cast ; ring
    rw [ch03_laplace_density, hre]
    push_cast
    linear_combination (2 * (b : ℂ))⁻¹ * hEA
  have key₂ : ∀ y : ℝ, y ∈ Set.Ioi (0 : ℝ) →
      Complex.exp ((y * u : ℝ) * Complex.I) * (ch03_laplace_density b y : ℂ)
        = (2 * (b : ℂ))⁻¹ * Complex.exp (a₂ * y) := by
    intro y hy
    have hre : -|y| / b = -y / b := by rw [abs_of_nonneg (le_of_lt hy)]
    have hEB : Complex.exp ((y : ℂ) * (u : ℂ) * Complex.I) * Complex.exp (-(y : ℂ) / (b : ℂ))
        = Complex.exp (a₂ * (y : ℂ)) := by
      rw [← Complex.exp_add, ha₂]
      congr 1
      push_cast ; ring
    rw [ch03_laplace_density, hre]
    push_cast
    linear_combination (2 * (b : ℂ))⁻¹ * hEB
  have hint₁ : IntegrableOn
      (fun y : ℝ => Complex.exp ((y * u : ℝ) * Complex.I) * (ch03_laplace_density b y : ℂ))
      (Set.Iic 0) :=
    IntegrableOn.congr_fun
      (Integrable.const_mul (integrableOn_exp_mul_complex_Iic hpos₁ 0) ((2 * (b : ℂ))⁻¹))
      (fun y hy => (key₁ y hy).symm) measurableSet_Iic
  have hint₂ : IntegrableOn
      (fun y : ℝ => Complex.exp ((y * u : ℝ) * Complex.I) * (ch03_laplace_density b y : ℂ))
      (Set.Ioi 0) :=
    IntegrableOn.congr_fun
      (Integrable.const_mul (integrableOn_exp_mul_complex_Ioi hneg₂ 0) ((2 * (b : ℂ))⁻¹))
      (fun y hy => (key₂ y hy).symm) measurableSet_Ioi
  rw [← intervalIntegral.integral_Iic_add_Ioi hint₁ hint₂,
    setIntegral_congr_fun measurableSet_Iic key₁,
    setIntegral_congr_fun measurableSet_Ioi key₂,
    integral_const_mul, integral_const_mul,
    integral_exp_mul_complex_Iic hpos₁, integral_exp_mul_complex_Ioi hneg₂]
  simp only [Complex.ofReal_zero, mul_zero, Complex.exp_zero]
  -- what is left is algebra in ℂ
  have ha₁0 : a₁ ≠ 0 := by intro h; rw [h] at hpos₁; simp at hpos₁
  have ha₂0 : a₂ ≠ 0 := by intro h; rw [h] at hneg₂; simp at hneg₂
  have hprod : a₁ * a₂ = -(((u ^ 2 + b⁻¹ ^ 2 : ℝ) : ℂ)) := by
    rw [ha₁, ha₂]
    push_cast
    linear_combination ((u : ℂ) ^ 2) * Complex.I_sq
  have hdiff : a₂ - a₁ = -(((2 * b⁻¹ : ℝ) : ℂ)) := by
    rw [ha₁, ha₂]; push_cast ; ring
  have hsplit : (2 * (b : ℂ))⁻¹ * (1 / a₁) + (2 * (b : ℂ))⁻¹ * (-1 / a₂)
      = (2 * (b : ℂ))⁻¹ * ((a₂ - a₁) / (a₁ * a₂)) := by
    field_simp ; ring
  have hdC : ((u : ℂ) ^ 2 + ((b : ℂ)⁻¹) ^ 2) ≠ 0 := by
    rw [show ((u : ℂ) ^ 2 + ((b : ℂ)⁻¹) ^ 2) = ((u ^ 2 + b⁻¹ ^ 2 : ℝ) : ℂ) by push_cast ; ring]
    have hh : (0 : ℝ) < u ^ 2 + b⁻¹ ^ 2 := by positivity
    exact_mod_cast ne_of_gt hh
  have hd'C : (1 + (b : ℂ) ^ 2 * (u : ℂ) ^ 2) ≠ 0 := by
    rw [show (1 + (b : ℂ) ^ 2 * (u : ℂ) ^ 2) = ((1 + b ^ 2 * u ^ 2 : ℝ) : ℂ) by
      push_cast ; ring]
    have hh : (0 : ℝ) < 1 + b ^ 2 * u ^ 2 := by positivity
    exact_mod_cast ne_of_gt hh
  have hratio : (a₂ - a₁) / (a₁ * a₂)
      = (2 * (b : ℂ)) / (1 + (b : ℂ) ^ 2 * (u : ℂ) ^ 2) := by
    rw [div_eq_div_iff (mul_ne_zero ha₁0 ha₂0) hd'C, hdiff, hprod]
    push_cast
    field_simp ; ring
  rw [hsplit, hratio,
    show (((1 + b ^ 2 * u ^ 2)⁻¹ : ℝ) : ℂ) = (1 + (b : ℂ) ^ 2 * (u : ℂ) ^ 2)⁻¹ by
      push_cast ; ring]
  field_simp

/-- Line 472 for the **unit-variance** Laplace, `b = 1/√2`: `φ_A(u) = (1 + u²/2)⁻¹`, which
is exactly the value noname-14 substitutes into eq:dev-cf. -/
theorem ch03_laplace_unit_charFun (u : ℝ) :
    (∫ y : ℝ, Complex.exp ((y * u : ℝ) * Complex.I)
        * (ch03_laplace_density (1 / Real.sqrt 2) y : ℂ))
      = ((1 + u ^ 2 / 2)⁻¹ : ℝ) := by
  have hs : (0 : ℝ) < Real.sqrt 2 := Real.sqrt_pos.mpr (by norm_num)
  have h2 : (0 : ℝ) < 1 / Real.sqrt 2 := by positivity
  rw [ch03_laplace_charFun _ h2 u,
    show (1 / Real.sqrt 2 : ℝ) ^ 2 * u ^ 2 = u ^ 2 / 2 by
      rw [div_pow, Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2)]; ring]

/-- Line 472: a Laplace variable with scale `b` has variance `2b²`; together with
`ch03_laplace_unit_variance` this is what forces `b = 1/√2` for unit variance.  (The law is
centred, so the second moment is the variance.) -/
theorem ch03_laplace_variance (b : ℝ) (hb : 0 < b) :
    (∫ y : ℝ, y ^ 2 * ch03_laplace_density b y) = 2 * b ^ 2 := by
  have hbne : b ≠ 0 := ne_of_gt hb
  have hgoal : (∫ y : ℝ, y ^ 2 * ch03_laplace_density b y)
      = ∫ y : ℝ, |y| ^ 2 * ((2 * b)⁻¹ * Real.exp (-|y| / b)) := by
    refine integral_congr_ae (Filter.Eventually.of_forall fun y => ?_)
    simp only [ch03_laplace_density, sq_abs]
  rw [hgoal, integral_comp_abs (f := fun s : ℝ => s ^ 2 * ((2 * b)⁻¹ * Real.exp (-s / b)))]
  have hG : Real.Gamma 3 = 2 := by simp
  have hgamma : (∫ s : ℝ in Set.Ioi (0 : ℝ), s ^ 2 * Real.exp (-(b⁻¹ * s))) = b ^ 3 * 2 := by
    have h := Real.integral_rpow_mul_exp_neg_mul_Ioi (a := 3) (r := b⁻¹)
      (by norm_num) (by positivity)
    simp only [show (3 : ℝ) - 1 = ((2 : ℕ) : ℝ) from by norm_num, Real.rpow_natCast] at h
    rw [hG, show (1 : ℝ) / b⁻¹ = b by field_simp,
      show (3 : ℝ) = ((3 : ℕ) : ℝ) from by norm_num, Real.rpow_natCast] at h
    exact h
  have hc : (∫ s : ℝ in Set.Ioi (0 : ℝ), s ^ 2 * ((2 * b)⁻¹ * Real.exp (-s / b)))
      = (2 * b)⁻¹ * ∫ s : ℝ in Set.Ioi (0 : ℝ), s ^ 2 * Real.exp (-(b⁻¹ * s)) := by
    rw [← integral_const_mul]
    refine setIntegral_congr_fun measurableSet_Ioi (fun s _ => ?_)
    rw [show -s / b = -(b⁻¹ * s) by field_simp]
    ring
  rw [hc, hgamma]
  field_simp

/-- noname-14 with the characteristic function of the input **computed** (by
`ch03_laplace_unit_charFun`) instead of assumed: the hypothesis `hφ` of `ch03_noname_14` is
discharged by the actual Laplace characteristic function. -/
theorem ch03_noname_14_laplace (t u : ℝ) :
    ((1 + (Real.exp (-t) * u) ^ 2 / 2)⁻¹ : ℝ) * Real.exp (-(ch03_Delta t * u ^ 2 / 2))
      = Real.exp (-(ch03_Delta t * u ^ 2 / 2)) / (1 + Real.exp (-2 * t) * u ^ 2 / 2) :=
  ch03_noname_14 t u (fun v => (1 + v ^ 2 / 2)⁻¹) (fun _ => rfl)

end ThesisAudit

