import Mathlib

/-!
Audit of display-math formulas in ch02-statmech-diffusion.tex.

Conventions used throughout this file:
* finite state spaces stand in for continuous ones where a formula's content is
  algebraic (sums replace integrals; this is stated in each comment);
* one dimension stands in for `ℝ^d` where the claim factorises coordinatewise;
* `HasDerivAt` statements encode gradients/scores in one dimension.
-/

namespace ThesisAudit

open MeasureTheory

/-! ### Section sm-what: entropy and the Boltzmann distribution -/

-- eq:sm-entropy (ch02-statmech-diffusion.tex lines 28-31): S[p] = -∑_x p(x) log p(x),
-- the Shannon/Gibbs entropy functional. A definition.
noncomputable def ch02_sm_entropy {ι : Type*} [Fintype ι] (p : ι → ℝ) : ℝ :=
  -∑ x, p x * Real.log (p x)

-- sanity: a point mass has zero entropy.
theorem ch02_sm_entropy_pointmass : ch02_sm_entropy (fun _ : Fin 1 => (1 : ℝ)) = 0 := by
  simp [ch02_sm_entropy]

-- eq:sm-boltzmann (lines 35-40): p(x) = e^{-βE(x)}/Z(β), Z(β) = ∑_x e^{-βE(x)}.
-- A definition; sanity lemmas below check Z > 0 and ∑ p = 1.
noncomputable def ch02_sm_boltzmann_Z {ι : Type*} [Fintype ι] (β : ℝ) (E : ι → ℝ) : ℝ :=
  ∑ x, Real.exp (-(β * E x))

noncomputable def ch02_sm_boltzmann_p {ι : Type*} [Fintype ι] (β : ℝ) (E : ι → ℝ) (x : ι) : ℝ :=
  Real.exp (-(β * E x)) / ch02_sm_boltzmann_Z β E

theorem ch02_sm_boltzmann_Z_pos {ι : Type*} [Fintype ι] [Nonempty ι] (β : ℝ) (E : ι → ℝ) :
    0 < ch02_sm_boltzmann_Z β E :=
  Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty

theorem ch02_sm_boltzmann_p_pos {ι : Type*} [Fintype ι] [Nonempty ι] (β : ℝ) (E : ι → ℝ)
    (x : ι) : 0 < ch02_sm_boltzmann_p β E x :=
  div_pos (Real.exp_pos _) (ch02_sm_boltzmann_Z_pos β E)

theorem ch02_sm_boltzmann_sum_one {ι : Type*} [Fintype ι] [Nonempty ι] (β : ℝ) (E : ι → ℝ) :
    ∑ x, ch02_sm_boltzmann_p β E x = 1 := by
  have hZ : (0:ℝ) < ∑ x, Real.exp (-(β * E x)) :=
    Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty
  unfold ch02_sm_boltzmann_p ch02_sm_boltzmann_Z
  rw [← Finset.sum_div]
  exact div_self (ne_of_gt hZ)

-- Kullback-Leibler divergence on a finite space; used by eq:sm-secondlaw,
-- eq:sm-elbo, eq:sm-elbo-gap and the maximum-entropy property of eq:sm-boltzmann.
noncomputable def ch02_KL {ι : Type*} [Fintype ι] (Q P : ι → ℝ) : ℝ :=
  ∑ x, Q x * Real.log (Q x / P x)

-- Gibbs' inequality: KL(Q ∥ P) ≥ 0 for probability vectors (finite version).
theorem ch02_KL_nonneg {ι : Type*} [Fintype ι] (Q P : ι → ℝ)
    (hQ : ∀ x, 0 < Q x) (hP : ∀ x, 0 < P x)
    (hQ1 : ∑ x, Q x = 1) (hP1 : ∑ x, P x = 1) : 0 ≤ ch02_KL Q P := by
  have key : ∑ x, Q x * Real.log (P x / Q x) ≤ 0 := by
    have hle : ∀ x : ι, Q x * Real.log (P x / Q x) ≤ P x - Q x := by
      intro x
      have h1 : Real.log (P x / Q x) ≤ P x / Q x - 1 :=
        Real.log_le_sub_one_of_pos (div_pos (hP x) (hQ x))
      have h2 := mul_le_mul_of_nonneg_left h1 (le_of_lt (hQ x))
      have h3 : Q x * (P x / Q x - 1) = P x - Q x := by
        have := ne_of_gt (hQ x); field_simp
      linarith
    calc ∑ x, Q x * Real.log (P x / Q x) ≤ ∑ x, (P x - Q x) :=
          Finset.sum_le_sum (fun x _ => hle x)
      _ = 0 := by rw [Finset.sum_sub_distrib, hQ1, hP1]; ring
  have flip : ∀ x : ι, Q x * Real.log (Q x / P x) = -(Q x * Real.log (P x / Q x)) := by
    intro x
    rw [show Q x / P x = (P x / Q x)⁻¹ by rw [inv_div], Real.log_inv]
    ring
  have : ch02_KL Q P = -∑ x, Q x * Real.log (P x / Q x) := by
    unfold ch02_KL
    rw [← Finset.sum_neg_distrib]
    exact Finset.sum_congr rfl (fun x _ => flip x)
  rw [this]
  linarith

-- Toolbox tb:sm-boltzmann / claim at lines 24-34: among all distributions with the
-- same mean energy, the Boltzmann distribution maximises the entropy (finite version).
theorem ch02_sm_boltzmann_maxent {ι : Type*} [Fintype ι] [Nonempty ι] (β : ℝ) (E : ι → ℝ)
    (q : ι → ℝ) (hq : ∀ x, 0 < q x) (hq1 : ∑ x, q x = 1)
    (hE : ∑ x, q x * E x = ∑ x, ch02_sm_boltzmann_p β E x * E x) :
    ch02_sm_entropy q ≤ ch02_sm_entropy (ch02_sm_boltzmann_p β E) := by
  have hZpos : 0 < ch02_sm_boltzmann_Z β E := ch02_sm_boltzmann_Z_pos β E
  have hp_pos : ∀ x, 0 < ch02_sm_boltzmann_p β E x := ch02_sm_boltzmann_p_pos β E
  have hsum1 : ∑ x, ch02_sm_boltzmann_p β E x = 1 := ch02_sm_boltzmann_sum_one β E
  have hlogp : ∀ x, Real.log (ch02_sm_boltzmann_p β E x)
      = -(β * E x) - Real.log (ch02_sm_boltzmann_Z β E) := by
    intro x
    unfold ch02_sm_boltzmann_p
    rw [Real.log_div (Real.exp_ne_zero _) (ne_of_gt hZpos), Real.log_exp]
  have hSp : ch02_sm_entropy (ch02_sm_boltzmann_p β E)
      = β * (∑ x, ch02_sm_boltzmann_p β E x * E x) + Real.log (ch02_sm_boltzmann_Z β E) := by
    unfold ch02_sm_entropy
    have hterm : ∀ x : ι, ch02_sm_boltzmann_p β E x * Real.log (ch02_sm_boltzmann_p β E x)
        = -(β * (ch02_sm_boltzmann_p β E x * E x))
          - Real.log (ch02_sm_boltzmann_Z β E) * ch02_sm_boltzmann_p β E x := by
      intro x; rw [hlogp x]; ring
    rw [Finset.sum_congr rfl (fun x _ => hterm x), Finset.sum_sub_distrib,
      Finset.sum_neg_distrib, ← Finset.mul_sum, ← Finset.mul_sum, hsum1]
    ring
  have hKLid : ch02_KL q (ch02_sm_boltzmann_p β E)
      = -ch02_sm_entropy q + β * (∑ x, q x * E x)
        + Real.log (ch02_sm_boltzmann_Z β E) := by
    unfold ch02_KL ch02_sm_entropy
    have hterm : ∀ x : ι, q x * Real.log (q x / ch02_sm_boltzmann_p β E x)
        = q x * Real.log (q x) + β * (q x * E x)
          + Real.log (ch02_sm_boltzmann_Z β E) * q x := by
      intro x
      rw [Real.log_div (ne_of_gt (hq x)) (ne_of_gt (hp_pos x)), hlogp x]
      ring
    rw [Finset.sum_congr rfl (fun x _ => hterm x), Finset.sum_add_distrib,
      Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum, hq1]
    ring
  have hKLnn : 0 ≤ ch02_KL q (ch02_sm_boltzmann_p β E) :=
    ch02_KL_nonneg q _ hq hp_pos hq1 hsum1
  rw [hKLid] at hKLnn
  rw [hSp]
  rw [hE] at hKLnn
  linarith

/-! ### Section sm-ebm: energy-based models -/

-- eq:sm-ebm (lines 103-108): p_θ(x) = e^{-E_θ(x)}/Z(θ), Z(θ) = ∫ e^{-E_θ(x)} dx.
-- A definition (encoded over ℝ; the audit's claims about it are in ch02_sm_ebm_grad).
noncomputable def ch02_sm_ebm_Z (Eθ : ℝ → ℝ) : ℝ := ∫ x : ℝ, Real.exp (-(Eθ x))

noncomputable def ch02_sm_ebm_p (Eθ : ℝ → ℝ) (x : ℝ) : ℝ :=
  Real.exp (-(Eθ x)) / ch02_sm_ebm_Z Eθ

-- eq:sm-ebm-grad (lines 112-119) together with toolbox tb:sm-ebm-grad (123-149):
-- maximum-likelihood gradient of an energy-based model.  Proved for a finite state
-- space and a scalar parameter θ (the continuum case only adds dominated convergence):
-- d/dθ E_{x∼data}[-log p_θ(x)] = E_{x∼data}[∂_θ E_θ(x)] - E_{x∼p_θ}[∂_θ E_θ(x)].
theorem ch02_sm_ebm_grad {ι : Type*} [Fintype ι] [Nonempty ι]
    (E : ℝ → ι → ℝ) (E' : ι → ℝ) (θ : ℝ) (w : ι → ℝ)
    (hE : ∀ x, HasDerivAt (fun s => E s x) (E' x) θ) (hw1 : ∑ x, w x = 1) :
    HasDerivAt
      (fun s => ∑ x, w x * (-Real.log (Real.exp (-(E s x)) / ∑ y, Real.exp (-(E s y)))))
      ((∑ x, w x * E' x)
        - ∑ x, (Real.exp (-(E θ x)) / ∑ y, Real.exp (-(E θ y))) * E' x) θ := by
  have hZpos : ∀ s : ℝ, 0 < ∑ y, Real.exp (-(E s y)) :=
    fun s => Finset.sum_pos (fun y _ => Real.exp_pos _) Finset.univ_nonempty
  have hfun : (fun s => ∑ x, w x * (-Real.log (Real.exp (-(E s x)) / ∑ y, Real.exp (-(E s y)))))
      = fun s => ∑ x, w x * (E s x + Real.log (∑ y, Real.exp (-(E s y)))) := by
    funext s
    apply Finset.sum_congr rfl
    intro x _
    rw [Real.log_div (Real.exp_ne_zero _) (ne_of_gt (hZpos s)), Real.log_exp]
    ring
  rw [hfun]
  have hZ : HasDerivAt (fun s => ∑ y, Real.exp (-(E s y)))
      (∑ y, Real.exp (-(E θ y)) * -E' y) θ := by
    apply HasDerivAt.fun_sum
    intro y _
    exact ((hE y).neg).exp
  have hlogZ : HasDerivAt (fun s => Real.log (∑ y, Real.exp (-(E s y))))
      ((∑ y, Real.exp (-(E θ y)) * -E' y) / ∑ y, Real.exp (-(E θ y))) θ :=
    hZ.log (ne_of_gt (hZpos θ))
  have hmain : HasDerivAt (fun s => ∑ x, w x * (E s x + Real.log (∑ y, Real.exp (-(E s y)))))
      (∑ x, w x * (E' x + (∑ y, Real.exp (-(E θ y)) * -E' y) / ∑ y, Real.exp (-(E θ y)))) θ := by
    apply HasDerivAt.fun_sum
    intro x _
    exact ((hE x).add hlogZ).const_mul (w x)
  convert hmain using 1
  have expand : ∑ x, w x * (E' x + (∑ y, Real.exp (-(E θ y)) * -E' y) / ∑ y, Real.exp (-(E θ y)))
      = (∑ x, w x * E' x)
        + (∑ y, Real.exp (-(E θ y)) * -E' y) / ∑ y, Real.exp (-(E θ y)) := by
    rw [Finset.sum_congr rfl (fun x _ => mul_add (w x) (E' x) _), Finset.sum_add_distrib,
      ← Finset.sum_mul, hw1, one_mul]
  rw [expand]
  have hflip : (∑ y, Real.exp (-(E θ y)) * -E' y) / ∑ y, Real.exp (-(E θ y))
      = -∑ x, Real.exp (-(E θ x)) / (∑ y, Real.exp (-(E θ y))) * E' x := by
    rw [Finset.sum_div, ← Finset.sum_neg_distrib]
    apply Finset.sum_congr rfl
    intro x _
    ring
  rw [hflip]
  ring

/-! ### Section sm-relaxation: Langevin dynamics and the OU channel -/

-- eq:sm-langevin (lines 224-227) states the Langevin SDE; the substantive claim
-- attached to it is that e^{-βE} is stationary.  Instance check in one dimension:
-- the probability flux  E'·q + β⁻¹·q'  of q = e^{-βE} vanishes identically, so the
-- 1D Fokker-Planck right-hand side  ∂_x[E'·q + β⁻¹·∂_x q]  is zero.
theorem ch02_sm_langevin_flux (E : ℝ → ℝ) (e' β x : ℝ) (hβ : β ≠ 0)
    (hE : HasDerivAt E e' x) :
    e' * Real.exp (-(β * E x)) + β⁻¹ * deriv (fun y => Real.exp (-(β * E y))) x = 0 := by
  have hq : HasDerivAt (fun y => Real.exp (-(β * E y)))
      (Real.exp (-(β * E x)) * -(β * e')) x := by
    have h1 : HasDerivAt (fun y => -(β * E y)) (-(β * e')) x := (hE.const_mul β).neg
    exact h1.exp
  rw [hq.deriv]
  field_simp
  ring

-- eq:sm-ou (lines 237-242): the OU transition x = e^{-t} a + √Δ_t ε, Δ_t = 1 - e^{-2t}.
-- Definitions, plus sanity lemmas checking the closed form against the OU moment ODEs
-- for f(x) = -x, g = √2: mean m(t) = e^{-t}a solves m' = -m with m(0)=a, and
-- variance Δ_t solves v' = 2 - 2v with v(0) = 0.
noncomputable def ch02_Delta (t : ℝ) : ℝ := 1 - Real.exp (-2 * t)

noncomputable def ch02_sm_ou_channel (t a ε : ℝ) : ℝ :=
  Real.exp (-t) * a + Real.sqrt (ch02_Delta t) * ε

theorem ch02_sm_ou_Delta_pos {t : ℝ} (ht : 0 < t) : 0 < ch02_Delta t := by
  unfold ch02_Delta
  have h : Real.exp (-2 * t) < Real.exp 0 := Real.exp_lt_exp.mpr (by linarith)
  rw [Real.exp_zero] at h
  linarith

theorem ch02_sm_ou_Delta_zero : ch02_Delta 0 = 0 := by
  simp [ch02_Delta]

theorem ch02_sm_ou_mean_ode (a t : ℝ) :
    HasDerivAt (fun s => Real.exp (-s) * a) (-(Real.exp (-t) * a)) t := by
  have h : HasDerivAt (fun s : ℝ => -s) (-1) t := (hasDerivAt_id t).neg
  have h2 : HasDerivAt (fun s : ℝ => Real.exp (-s) * a) (Real.exp (-t) * -1 * a) t :=
    h.exp.mul_const a
  have heq : Real.exp (-t) * -1 * a = -(Real.exp (-t) * a) := by ring
  rw [heq] at h2
  exact h2

theorem ch02_sm_ou_var_ode (t : ℝ) :
    HasDerivAt ch02_Delta (2 - 2 * ch02_Delta t) t := by
  have h : HasDerivAt (fun s : ℝ => -2 * s) (-2) t := by
    simpa using (hasDerivAt_id t).const_mul (-2)
  have h2 : HasDerivAt (fun s : ℝ => 1 - Real.exp (-2 * s)) (-(Real.exp (-2 * t) * -2)) t :=
    h.exp.const_sub 1
  have heq : -(Real.exp (-2 * t) * -2) = 2 - 2 * ch02_Delta t := by
    unfold ch02_Delta; ring
  rw [heq] at h2
  exact h2

/-! ### Section sm-nonequilibrium: path measures, Jarzynski, ELBO -/

-- eq:sm-forward-path (lines 288-298): Q(x_{0:T}) = p_0(x_0) ∏_{k=1}^T q(x_k | x_{k-1}).
-- A definition (paths encoded as x : ℕ → σ; q a b stands for q(a | b)).
noncomputable def ch02_sm_forward_path {σ : Type*} (T : ℕ) (p0 : σ → ℝ) (q : σ → σ → ℝ)
    (x : ℕ → σ) : ℝ :=
  p0 (x 0) * ∏ k ∈ Finset.range T, q (x (k + 1)) (x k)

-- sanity: for T = 1 over a finite state space the forward path law is normalised.
theorem ch02_sm_forward_path_norm_T1 {σ : Type*} [Fintype σ] (p0 : σ → ℝ) (q : σ → σ → ℝ)
    (hq : ∀ b, ∑ a, q a b = 1) (hp : ∑ b, p0 b = 1) :
    ∑ b, ∑ a, ch02_sm_forward_path 1 p0 q (fun n => if n = 0 then b else a) = 1 := by
  have hval : ∀ b a : σ,
      ch02_sm_forward_path 1 p0 q (fun n => if n = 0 then b else a) = p0 b * q a b := by
    intro b a
    simp [ch02_sm_forward_path]
  calc ∑ b, ∑ a, ch02_sm_forward_path 1 p0 q (fun n => if n = 0 then b else a)
      = ∑ b, ∑ a, p0 b * q a b := by
        exact Finset.sum_congr rfl fun b _ => Finset.sum_congr rfl fun a _ => hval b a
    _ = ∑ b, p0 b * ∑ a, q a b := by
        exact Finset.sum_congr rfl fun b _ => (Finset.mul_sum _ _ _).symm
    _ = ∑ b, p0 b := by
        exact Finset.sum_congr rfl fun b _ => by rw [hq b, mul_one]
    _ = 1 := hp

-- eq:sm-reverse-path (lines 302-309): P_θ(x_{0:T}) = π(x_T) ∏_{k=1}^T p_θ(x_{k-1} | x_k).
-- A definition (pθ a b stands for p_θ(a | b)).
noncomputable def ch02_sm_reverse_path {σ : Type*} (T : ℕ) (piT : σ → ℝ) (pθ : σ → σ → ℝ)
    (x : ℕ → σ) : ℝ :=
  piT (x T) * ∏ k ∈ Finset.range T, pθ (x k) (x (k + 1))

-- sanity: for T = 1 over a finite state space the reverse path law is normalised.
theorem ch02_sm_reverse_path_norm_T1 {σ : Type*} [Fintype σ] (piT : σ → ℝ) (pθ : σ → σ → ℝ)
    (hpθ : ∀ b, ∑ a, pθ a b = 1) (hpi : ∑ b, piT b = 1) :
    ∑ b, ∑ a, ch02_sm_reverse_path 1 piT pθ (fun n => if n = 0 then a else b) = 1 := by
  have hval : ∀ b a : σ,
      ch02_sm_reverse_path 1 piT pθ (fun n => if n = 0 then a else b) = piT b * pθ a b := by
    intro b a
    simp [ch02_sm_reverse_path]
  calc ∑ b, ∑ a, ch02_sm_reverse_path 1 piT pθ (fun n => if n = 0 then a else b)
      = ∑ b, ∑ a, piT b * pθ a b := by
        exact Finset.sum_congr rfl fun b _ => Finset.sum_congr rfl fun a _ => hval b a
    _ = ∑ b, piT b * ∑ a, pθ a b := by
        exact Finset.sum_congr rfl fun b _ => (Finset.mul_sum _ _ _).symm
    _ = ∑ b, piT b := by
        exact Finset.sum_congr rfl fun b _ => by rw [hpθ b, mul_one]
    _ = 1 := hpi

-- eq:sm-work (lines 319-325): W_θ(x_{0:T}) := log (Q(x_{0:T}) / P_θ(x_{0:T})).
-- A definition on an abstract (finite) path space Ω.
noncomputable def ch02_sm_work {Ω : Type*} (Q P : Ω → ℝ) (x : Ω) : ℝ :=
  Real.log (Q x / P x)

-- eq:sm-jarzynski (lines 333-347): E_Q[e^{-W_θ}] = ∑ Q·(P/Q) = ∑ P = 1.
-- Proved on a finite path space (the display's integral is the sum here).
theorem ch02_sm_jarzynski {Ω : Type*} [Fintype Ω] (Q P : Ω → ℝ)
    (hQ : ∀ x, 0 < Q x) (hP : ∀ x, 0 < P x) (hP1 : ∑ x, P x = 1) :
    ∑ x, Q x * Real.exp (-ch02_sm_work Q P x) = 1 := by
  have step : ∀ x : Ω, Q x * Real.exp (-ch02_sm_work Q P x) = P x := by
    intro x
    unfold ch02_sm_work
    rw [← Real.log_inv, inv_div, Real.exp_log (div_pos (hP x) (hQ x)), mul_comm,
      div_mul_cancel₀ _ (ne_of_gt (hQ x))]
  rw [Finset.sum_congr rfl (fun x _ => step x), hP1]

-- eq:sm-secondlaw (lines 350-359): E_Q[W_θ] = KL(Q ∥ P_θ) ≥ 0 (finite version).
theorem ch02_sm_secondlaw {Ω : Type*} [Fintype Ω] (Q P : Ω → ℝ)
    (hQ : ∀ x, 0 < Q x) (hP : ∀ x, 0 < P x)
    (hQ1 : ∑ x, Q x = 1) (hP1 : ∑ x, P x = 1) :
    (∑ x, Q x * ch02_sm_work Q P x) = ch02_KL Q P ∧ 0 ≤ ch02_KL Q P := by
  constructor
  · simp [ch02_sm_work, ch02_KL]
  · exact ch02_KL_nonneg Q P hQ hP hQ1 hP1

-- eq:sm-model-marginal (lines 368-375): p_θ(x_0) = ∫ P_θ(x_{0:T}) dx_{1:T}.
-- A definition: marginalisation over the (finite) space of intermediate paths.
noncomputable def ch02_sm_model_marginal {σ Ω : Type*} [Fintype Ω] (P : σ → Ω → ℝ)
    (x0 : σ) : ℝ :=
  ∑ ω, P x0 ω

-- eq:sm-importance-path (lines 379-401): p_θ(x_0) = ∑ q·(P/q) = E_q[P/q]
-- (multiply and divide by the forward conditional; finite version).
theorem ch02_sm_importance_path {σ Ω : Type*} [Fintype Ω] (P : σ → Ω → ℝ) (q : Ω → ℝ)
    (x0 : σ) (hq : ∀ ω, 0 < q ω) :
    ch02_sm_model_marginal P x0 = ∑ ω, q ω * (P x0 ω / q ω) := by
  unfold ch02_sm_model_marginal
  apply Finset.sum_congr rfl
  intro ω _
  rw [mul_comm, div_mul_cancel₀ _ (ne_of_gt (hq ω))]

-- eq:sm-elbo-gap (lines 439-451): log p_θ(x_0) - L_ELBO = KL(q(·|x_0) ∥ p_θ(·|x_0)) ≥ 0,
-- with p_θ(x_{1:T}|x_0) = P_θ(x_{0:T}) / p_θ(x_0).  Finite version; P ω is the joint
-- P_θ(x_0, ω) at fixed x_0, q the forward conditional.
theorem ch02_sm_elbo_gap {Ω : Type*} [Fintype Ω] [Nonempty Ω] (P q : Ω → ℝ)
    (hP : ∀ ω, 0 < P ω) (hq : ∀ ω, 0 < q ω) (hq1 : ∑ ω, q ω = 1) :
    Real.log (∑ ω, P ω) - (∑ ω, q ω * Real.log (P ω / q ω))
      = ch02_KL q (fun ω => P ω / ∑ ω', P ω') := by
  have hZ : 0 < ∑ ω, P ω := Finset.sum_pos (fun ω _ => hP ω) Finset.univ_nonempty
  unfold ch02_KL
  have key : ∀ ω : Ω, q ω * Real.log (q ω / (P ω / ∑ ω', P ω'))
      = q ω * Real.log (∑ ω', P ω') - q ω * Real.log (P ω / q ω) := by
    intro ω
    have hPne := ne_of_gt (hP ω)
    have hZne := ne_of_gt hZ
    have hqne := ne_of_gt (hq ω)
    have h1 : q ω / (P ω / ∑ ω', P ω') = q ω * (∑ ω', P ω') / P ω := by
      field_simp
    rw [h1, Real.log_div (mul_ne_zero hqne hZne) hPne, Real.log_mul hqne hZne,
      Real.log_div hPne hqne]
    ring
  rw [Finset.sum_congr rfl (fun ω _ => key ω), Finset.sum_sub_distrib, ← Finset.sum_mul,
    hq1, one_mul]

-- The nonnegativity half of eq:sm-elbo-gap.
theorem ch02_sm_elbo_gap_nonneg {Ω : Type*} [Fintype Ω] [Nonempty Ω] (P q : Ω → ℝ)
    (hP : ∀ ω, 0 < P ω) (hq : ∀ ω, 0 < q ω) (hq1 : ∑ ω, q ω = 1) :
    0 ≤ Real.log (∑ ω, P ω) - (∑ ω, q ω * Real.log (P ω / q ω)) := by
  have hZ : 0 < ∑ ω, P ω := Finset.sum_pos (fun ω _ => hP ω) Finset.univ_nonempty
  rw [ch02_sm_elbo_gap P q hP hq hq1]
  apply ch02_KL_nonneg q _ hq (fun ω => div_pos (hP ω) hZ) hq1
  rw [← Finset.sum_div]
  exact div_self (ne_of_gt hZ)

-- eq:sm-elbo (lines 403-418): the evidence lower bound
-- log p_θ(x_0) ≥ E_q[log (P_θ/q)] (Jensen; finite version, via the gap identity).
theorem ch02_sm_elbo {Ω : Type*} [Fintype Ω] [Nonempty Ω] (P q : Ω → ℝ)
    (hP : ∀ ω, 0 < P ω) (hq : ∀ ω, 0 < q ω) (hq1 : ∑ ω, q ω = 1) :
    (∑ ω, q ω * Real.log (P ω / q ω)) ≤ Real.log (∑ ω, P ω) := by
  have := ch02_sm_elbo_gap_nonneg P q hP hq hq1
  linarith

-- eq:sm-elbo-expanded (lines 421-435): pointwise expansion of the ELBO integrand,
-- log (P_θ(x_{0:T}) / q(x_{1:T}|x_0))
--   = log π(x_T) + ∑ log p_θ(x_{k-1}|x_k) - ∑ log q(x_k|x_{k-1});
-- taking E_q of both sides is then linearity of the (finite) expectation.
theorem ch02_sm_elbo_expanded {σ : Type*} (T : ℕ) (piT : σ → ℝ) (pθ q : σ → σ → ℝ)
    (x : ℕ → σ) (hpi : 0 < piT (x T))
    (hp : ∀ k ∈ Finset.range T, 0 < pθ (x k) (x (k + 1)))
    (hq : ∀ k ∈ Finset.range T, 0 < q (x (k + 1)) (x k)) :
    Real.log (ch02_sm_reverse_path T piT pθ x / ∏ k ∈ Finset.range T, q (x (k + 1)) (x k))
      = Real.log (piT (x T)) + (∑ k ∈ Finset.range T, Real.log (pθ (x k) (x (k + 1))))
        - ∑ k ∈ Finset.range T, Real.log (q (x (k + 1)) (x k)) := by
  unfold ch02_sm_reverse_path
  have hprodp : (0:ℝ) < ∏ k ∈ Finset.range T, pθ (x k) (x (k + 1)) := Finset.prod_pos hp
  have hprodq : (0:ℝ) < ∏ k ∈ Finset.range T, q (x (k + 1)) (x k) := Finset.prod_pos hq
  rw [Real.log_div (ne_of_gt (mul_pos hpi hprodp)) (ne_of_gt hprodq),
    Real.log_mul (ne_of_gt hpi) (ne_of_gt hprodp),
    Real.log_prod (fun k hk => ne_of_gt (hp k hk)),
    Real.log_prod (fun k hk => ne_of_gt (hq k hk))]

/-! ### Toolbox tb:sm-fokkerplanck and the probability-flow rewriting -/

-- eq:sm-fokkerplanck (lines 521-531): ∂_t p = -∇·(f p) + ½g²Δp.  The derivation needs
-- Itô's lemma and integration by parts; here the equation itself is instance-checked in
-- one dimension on the OU process (f(x) = -x, g = √2, so ½g² = 1) at its stationary
-- density p(x) = e^{-x²/2}: the right-hand side  ∂_x(x·p) + ∂_x²p  vanishes identically,
-- i.e. the standard Gaussian is a stationary solution of the OU Fokker-Planck equation.
theorem ch02_sm_fokkerplanck_ou_stationary (x : ℝ) :
    deriv (fun y => y * Real.exp (-(y ^ 2) / 2)) x
      + deriv (deriv (fun y => Real.exp (-(y ^ 2) / 2))) x = 0 := by
  have hgauss : ∀ y : ℝ, HasDerivAt (fun z => Real.exp (-(z ^ 2) / 2))
      (-y * Real.exp (-(y ^ 2) / 2)) y := by
    intro y
    have hpow : HasDerivAt (fun z : ℝ => z ^ 2) (2 * y) y := by
      have h := hasDerivAt_pow 2 y
      norm_num at h
      exact h
    have h1 : HasDerivAt (fun z : ℝ => -(z ^ 2) / 2) (-(2 * y) / 2) y := hpow.neg.div_const 2
    have h2 : HasDerivAt (fun z : ℝ => Real.exp (-(z ^ 2) / 2))
        (Real.exp (-(y ^ 2) / 2) * (-(2 * y) / 2)) y := h1.exp
    have heq : Real.exp (-(y ^ 2) / 2) * (-(2 * y) / 2) = -y * Real.exp (-(y ^ 2) / 2) := by
      ring
    rw [heq] at h2
    exact h2
  have hd1 : deriv (fun z => Real.exp (-(z ^ 2) / 2)) = fun y => -y * Real.exp (-(y ^ 2) / 2) := by
    funext y
    exact (hgauss y).deriv
  have hxp : HasDerivAt (fun y => y * Real.exp (-(y ^ 2) / 2))
      (1 * Real.exp (-(x ^ 2) / 2) + x * (-x * Real.exp (-(x ^ 2) / 2))) x :=
    (hasDerivAt_id x).mul (hgauss x)
  have hneg : HasDerivAt (fun y => -y * Real.exp (-(y ^ 2) / 2))
      (-1 * Real.exp (-(x ^ 2) / 2) + -x * (-x * Real.exp (-(x ^ 2) / 2))) x :=
    ((hasDerivAt_id x).neg).mul (hgauss x)
  rw [hd1, hxp.deriv, hneg.deriv]
  ring

-- noname-1 (unlabeled equation, lines 544-551): Δ_x p = ∇_x·(p ∇_x log p).
-- Proved in one dimension via the pointwise field identity p·(log p)' = p'
-- (so that ∇·(p∇log p) = ∇·(∇p) = Δp); positivity and differentiability assumed,
-- exactly as in the text.
theorem ch02_noname_1 (p : ℝ → ℝ) (hp : ∀ y, 0 < p y) (hd : Differentiable ℝ p) :
    (fun y => p y * deriv (fun z => Real.log (p z)) y) = deriv p := by
  funext y
  have h := ((hd y).hasDerivAt).log (ne_of_gt (hp y))
  rw [h.deriv]
  have := ne_of_gt (hp y)
  field_simp

-- consequence recorded at second-derivative level: Δp = ∇·(p ∇log p) in 1D.
theorem ch02_noname_1_laplacian (p : ℝ → ℝ) (hp : ∀ y, 0 < p y) (hd : Differentiable ℝ p)
    (x : ℝ) :
    deriv (deriv p) x = deriv (fun y => p y * deriv (fun z => Real.log (p z)) y) x := by
  rw [ch02_noname_1 p hp hd]

-- eq:sm-probflow-continuity (lines 553-565): the Fokker-Planck right-hand side equals
-- the continuity-equation form, -∇·[(f - ½g² ∇log p) p] = -∇·(f p) + ½g² Δp.
-- Proved in one dimension under the differentiability/positivity hypotheses of the text.
theorem ch02_sm_probflow_continuity (p f : ℝ → ℝ) (g2 x : ℝ)
    (hp : ∀ y, 0 < p y) (hd : Differentiable ℝ p)
    (hfp : DifferentiableAt ℝ (fun y => f y * p y) x)
    (hp' : DifferentiableAt ℝ (deriv p) x) :
    -deriv (fun y => (f y - g2 / 2 * deriv (fun z => Real.log (p z)) y) * p y) x
      = -deriv (fun y => f y * p y) x + g2 / 2 * deriv (deriv p) x := by
  have inner_eq : (fun y => (f y - g2 / 2 * deriv (fun z => Real.log (p z)) y) * p y)
      = fun y => f y * p y - g2 / 2 * deriv p y := by
    funext y
    have hlp : p y * deriv (fun z => Real.log (p z)) y = deriv p y :=
      congrFun (ch02_noname_1 p hp hd) y
    calc (f y - g2 / 2 * deriv (fun z => Real.log (p z)) y) * p y
        = f y * p y - g2 / 2 * (p y * deriv (fun z => Real.log (p z)) y) := by ring
      _ = f y * p y - g2 / 2 * deriv p y := by rw [hlp]
  rw [inner_eq, deriv_fun_sub hfp (hp'.const_mul (g2 / 2)), deriv_const_mul (g2 / 2) hp']
  ring

-- eq:sm-probability-flow (lines 566-575): the probability-flow ODE
-- Ẋ = f(X,t) - ½ g(t)² ∇log p_t(X).  A definition of the velocity field; the claim
-- that it preserves the marginals is exactly eq:sm-probflow-continuity above.
noncomputable def ch02_sm_probability_flow_velocity (f : ℝ → ℝ → ℝ) (g : ℝ → ℝ)
    (s : ℝ → ℝ → ℝ) (x t : ℝ) : ℝ :=
  f x t - g t ^ 2 / 2 * s x t

/-! ### DDPM parameterisation -/

-- eq:sm-ddpm-marginal (lines 648-657): x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε.
-- A definition, with the variance-preservation sanity check ᾱ + (1-ᾱ) = 1.
noncomputable def ch02_sm_ddpm_marginal (abar x0 ε : ℝ) : ℝ :=
  Real.sqrt abar * x0 + Real.sqrt (1 - abar) * ε

theorem ch02_sm_ddpm_marginal_vp (abar : ℝ) (h0 : 0 ≤ abar) (h1 : abar ≤ 1) :
    Real.sqrt abar ^ 2 + Real.sqrt (1 - abar) ^ 2 = 1 := by
  rw [Real.sq_sqrt h0, Real.sq_sqrt (by linarith)]
  ring

-- eq:sm-forward (lines 212-215): the generic forward SDE dX = f dt + g dW.  SKIPPED:
-- a stochastic differential equation, not an algebraic claim; stochastic integration
-- is not encoded in this audit.

-- eq:sm-ou, additional sanity (lines 237-242): the channel is variance-preserving,
-- (e^{-t})² + Δ_t = 1, so unit-variance data stays at unit variance.
theorem ch02_sm_ou_vp (t : ℝ) : Real.exp (-t) ^ 2 + ch02_Delta t = 1 := by
  unfold ch02_Delta
  have h : Real.exp (-t) ^ 2 = Real.exp (-2 * t) := by
    rw [sq, ← Real.exp_add]; congr 1; ring
  rw [h]; ring

-- eq:sm-jarzynski-physical (lines 271-274): E[e^{-β(W-ΔF)}] = 1.  The physical setup
-- (driven Hamiltonian dynamics) is out of reach; instance-checked on a finite path
-- space taking Crooks' pathwise relation β(W - ΔF) = log(Q/P) as the characterisation
-- of the dissipated work, under which the identity is exactly E_Q[P/Q] = 1.
theorem ch02_sm_jarzynski_physical {Ω : Type*} [Fintype Ω] (Q P W : Ω → ℝ) (β ΔF : ℝ)
    (hQ : ∀ x, 0 < Q x) (hP : ∀ x, 0 < P x) (hP1 : ∑ x, P x = 1)
    (hcrooks : ∀ x, β * (W x - ΔF) = Real.log (Q x / P x)) :
    ∑ x, Q x * Real.exp (-(β * (W x - ΔF))) = 1 := by
  have step : ∀ x : Ω, Q x * Real.exp (-(β * (W x - ΔF))) = P x := by
    intro x
    rw [hcrooks x, ← Real.log_inv, inv_div, Real.exp_log (div_pos (hP x) (hQ x)), mul_comm,
      div_mul_cancel₀ _ (ne_of_gt (hQ x))]
  rw [Finset.sum_congr rfl fun x _ => step x, hP1]

-- eq:sm-reverse (lines 603-616): Anderson's reverse-time SDE
-- dX = [f - g²∇log p]dt + g dW̄, run from T back to 0.  The pathwise time-reversal
-- theorem is out of reach here; what is checked (in one dimension) is the density-level
-- consistency behind the boxed drift: writing the reversed process in forward time, its
-- Fokker-Planck right-hand side with drift -(f - g²∂ₓlog p) at the same density p equals
-- MINUS the forward Fokker-Planck right-hand side, so the reverse SDE transports the
-- marginals backwards exactly when the forward SDE transports them forwards.
theorem ch02_sm_reverse (p f : ℝ → ℝ) (g2 x : ℝ)
    (hp : ∀ y, 0 < p y) (hd : Differentiable ℝ p)
    (hfp : DifferentiableAt ℝ (fun y => f y * p y) x)
    (hp' : DifferentiableAt ℝ (deriv p) x) :
    -deriv (fun y => (-(f y) + g2 * deriv (fun z => Real.log (p z)) y) * p y) x
      + g2 / 2 * deriv (deriv p) x
    = -(-deriv (fun y => f y * p y) x + g2 / 2 * deriv (deriv p) x) := by
  have inner_eq : (fun y => (-(f y) + g2 * deriv (fun z => Real.log (p z)) y) * p y)
      = fun y => g2 * deriv p y - f y * p y := by
    funext y
    have hlp : p y * deriv (fun z => Real.log (p z)) y = deriv p y :=
      congrFun (ch02_noname_1 p hp hd) y
    calc (-(f y) + g2 * deriv (fun z => Real.log (p z)) y) * p y
        = g2 * (p y * deriv (fun z => Real.log (p z)) y) - f y * p y := by ring
      _ = g2 * deriv p y - f y * p y := by rw [hlp]
  rw [inner_eq, deriv_fun_sub (hp'.const_mul g2) hfp, deriv_const_mul g2 hp']
  ring

/-! ### Toolbox tb:sm-tweedie and Section sm-three-scores: the score-posterior identity

Encoding: the prior P₀ is a finitely supported measure on ℝ — atoms `a i` with positive
weights `lam i` (not necessarily normalised; every displayed identity is a ratio in which
the overall normalisation cancels).  `c` stands for the contraction e^{-t}, `Δ` for the
channel variance Δ_t, and sums over `Fin N` replace ∫ P₀(a) … da.  One dimension stands
in for ℝ^d; the thesis's own toolbox (lines 801-802) notes that ∇ₓ and the kernel
factorise over components. -/

-- The 1D Gaussian kernel G(x; m, v) = (2πv)^{-1/2} exp(-(x-m)²/(2v)) of the channel.
noncomputable def ch02_gauss (v x m : ℝ) : ℝ :=
  Real.exp (-(x - m) ^ 2 / (2 * v)) / Real.sqrt (2 * Real.pi * v)

theorem ch02_gauss_pos {v : ℝ} (hv : 0 < v) (x m : ℝ) : 0 < ch02_gauss v x m := by
  have h2 : (0:ℝ) < 2 * Real.pi * v := by positivity
  exact div_pos (Real.exp_pos _) (Real.sqrt_pos.mpr h2)

-- eq:marg (lines 730-733): P_t(x) = ∫ P₀(a) G(x; e^{-t}a, Δ_t) da  (finite prior: sum).
noncomputable def ch02_marg {N : ℕ} (lam a : Fin N → ℝ) (Δ c x : ℝ) : ℝ :=
  ∑ i, lam i * ch02_gauss Δ x (c * a i)

-- the un-normalised posterior first moment ∫ a P₀(a) G(x; e^{-t}a, Δ_t) da.
noncomputable def ch02_marg_num {N : ℕ} (lam a : Fin N → ℝ) (Δ c x : ℝ) : ℝ :=
  ∑ i, lam i * ch02_gauss Δ x (c * a i) * a i

-- sanity for eq:marg: P_t > 0 everywhere (the text: "Since G > 0, we have P_t(x) > 0").
theorem ch02_marg_pos {N : ℕ} (hN : 0 < N) (lam a : Fin N → ℝ) {Δ : ℝ} (hΔ : 0 < Δ)
    (hlam : ∀ i, 0 < lam i) (c x : ℝ) : 0 < ch02_marg lam a Δ c x := by
  have : Nonempty (Fin N) := Fin.pos_iff_nonempty.mp hN
  unfold ch02_marg
  exact Finset.sum_pos (fun i _ => mul_pos (hlam i) (ch02_gauss_pos hΔ _ _))
    Finset.univ_nonempty

-- eq:loggrad (lines 739-742): S = ∇P_t/P_t for positive differentiable P_t (1D).
theorem ch02_loggrad (P : ℝ → ℝ) (P' x : ℝ) (hP : P x ≠ 0) (h : HasDerivAt P P' x) :
    HasDerivAt (fun y => Real.log (P y)) (P' / P x) x :=
  h.log hP

-- eq:kernelgrad (lines 756-760): ∇ₓG(x; m, Δ) = -((x - m)/Δ)·G(x; m, Δ)  (1D).
theorem ch02_kernelgrad {v : ℝ} (hv : 0 < v) (x m : ℝ) :
    HasDerivAt (fun y => ch02_gauss v y m) (-((x - m) / v) * ch02_gauss v x m) x := by
  have hy : HasDerivAt (fun y : ℝ => y - m) 1 x := (hasDerivAt_id x).sub_const m
  have hp2 : HasDerivAt (fun y : ℝ => (y - m) ^ 2) (2 * (x - m)) x := by
    simpa using hy.fun_pow 2
  have h1 : HasDerivAt (fun y : ℝ => -(y - m) ^ 2 / (2 * v)) (-(2 * (x - m)) / (2 * v)) x :=
    hp2.neg.div_const (2 * v)
  have h2 := (h1.exp).div_const (Real.sqrt (2 * Real.pi * v))
  have hsimp : -(2 * (x - m)) / (2 * v) = -((x - m) / v) := by
    rw [neg_div, mul_div_mul_left _ _ (by norm_num : (2:ℝ) ≠ 0)]
  have heq : Real.exp (-(x - m) ^ 2 / (2 * v)) * (-(2 * (x - m)) / (2 * v))
        / Real.sqrt (2 * Real.pi * v)
      = -((x - m) / v)
        * (Real.exp (-(x - m) ^ 2 / (2 * v)) / Real.sqrt (2 * Real.pi * v)) := by
    rw [hsimp]; ring
  rw [heq] at h2
  unfold ch02_gauss
  exact h2

-- eq:diffunder (lines 749-752): differentiation passes under the integral
-- (finite prior: the sum rule for derivatives stands in for dominated convergence).
theorem ch02_diffunder {N : ℕ} (lam a : Fin N → ℝ) {Δ : ℝ} (hΔ : 0 < Δ) (c x : ℝ) :
    HasDerivAt (fun y => ch02_marg lam a Δ c y)
      (∑ i, lam i * (-((x - c * a i) / Δ) * ch02_gauss Δ x (c * a i))) x := by
  unfold ch02_marg
  apply HasDerivAt.fun_sum
  intro i _
  exact (ch02_kernelgrad hΔ x (c * a i)).const_mul (lam i)

-- eq:split (lines 767-774): ∇P_t(x) = -(x/Δ)·P_t(x) + (e^{-t}/Δ)·∫ a P₀ G  (finite prior).
theorem ch02_split {N : ℕ} (lam a : Fin N → ℝ) {Δ : ℝ} (hΔ : 0 < Δ) (c x : ℝ) :
    HasDerivAt (fun y => ch02_marg lam a Δ c y)
      (-(x / Δ) * ch02_marg lam a Δ c x + c / Δ * ch02_marg_num lam a Δ c x) x := by
  have h := ch02_diffunder lam a hΔ c x
  convert h using 1
  unfold ch02_marg ch02_marg_num
  rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun i _ => ?_
  have h2 : Δ ≠ 0 := ne_of_gt hΔ
  field_simp
  ring

-- eq:posterior (lines 780-783): P_t(a|x) = P₀(a) G(x; e^{-t}a, Δ) / P_t(x)  (Bayes).
noncomputable def ch02_posterior {N : ℕ} (lam a : Fin N → ℝ) (Δ c x : ℝ) (i : Fin N) : ℝ :=
  lam i * ch02_gauss Δ x (c * a i) / ch02_marg lam a Δ c x

-- sanity for eq:posterior: it is a probability vector, "precisely because P_t(x) is its
-- normalising constant".
theorem ch02_posterior_sum_one {N : ℕ} (hN : 0 < N) (lam a : Fin N → ℝ) {Δ : ℝ}
    (hΔ : 0 < Δ) (hlam : ∀ i, 0 < lam i) (c x : ℝ) :
    ∑ i, ch02_posterior lam a Δ c x i = 1 := by
  have hpos := ch02_marg_pos hN lam a hΔ hlam c x
  unfold ch02_posterior
  rw [← Finset.sum_div]
  show ch02_marg lam a Δ c x / ch02_marg lam a Δ c x = 1
  exact div_self (ne_of_gt hpos)

-- the posterior mean ⟨a⟩_{x,t}.
noncomputable def ch02_postmean {N : ℕ} (lam a : Fin N → ℝ) (Δ c x : ℝ) : ℝ :=
  ch02_marg_num lam a Δ c x / ch02_marg lam a Δ c x

-- eq:condmean (lines 786-790): (1/P_t(x))·∫ a P₀ G = ∫ a P_t(a|x) da = ⟨a⟩_{x,t}.
theorem ch02_condmean {N : ℕ} (lam a : Fin N → ℝ) (Δ c x : ℝ) :
    ch02_postmean lam a Δ c x = ∑ i, ch02_posterior lam a Δ c x i * a i := by
  unfold ch02_postmean ch02_marg_num ch02_posterior
  rw [Finset.sum_div]
  exact Finset.sum_congr rfl fun i _ => by ring

-- eq:tweedie-tb (lines 795-800): S(x,t) = -x/Δ + (e^{-t}/Δ)·⟨a⟩_{x,t}
-- (finite prior, 1D; the main boxed identity of the toolbox).
theorem ch02_tweedie_tb {N : ℕ} (hN : 0 < N) (lam a : Fin N → ℝ) {Δ : ℝ} (hΔ : 0 < Δ)
    (hlam : ∀ i, 0 < lam i) (c x : ℝ) :
    HasDerivAt (fun y => Real.log (ch02_marg lam a Δ c y))
      (-(x / Δ) + c / Δ * ch02_postmean lam a Δ c x) x := by
  have hpos := ch02_marg_pos hN lam a hΔ hlam c x
  have h := (ch02_split lam a hΔ c x).log (ne_of_gt hpos)
  have h1 : ch02_marg lam a Δ c x ≠ 0 := ne_of_gt hpos
  have h2 : Δ ≠ 0 := ne_of_gt hΔ
  convert h using 1
  unfold ch02_postmean
  field_simp

-- eq:denoiser (lines 806-809): solving eq:tweedie-tb for the conditional mean,
-- ⟨a⟩ = e^t (x + Δ·S).  Pure algebra, fully general.
theorem ch02_denoiser (S A x t Δ : ℝ) (hΔ : Δ ≠ 0)
    (hS : S = (Real.exp (-t) * A - x) / Δ) :
    A = Real.exp t * (x + Δ * S) := by
  subst hS
  have h3 : Δ * ((Real.exp (-t) * A - x) / Δ) = Real.exp (-t) * A - x := by
    field_simp
  have key : Real.exp t * Real.exp (-t) = 1 := by
    rw [← Real.exp_add]; simp
  rw [h3]
  have h4 : Real.exp t * (x + (Real.exp (-t) * A - x)) = Real.exp t * Real.exp (-t) * A := by
    ring
  rw [h4, key, one_mul]

-- eq:tweedie (lines 705-714): the score-posterior identity and its inverse form,
-- ∇ₓ log p_t(x) = (e^{-t}·E[a|x] - x)/Δ_t  ⟺  E[a|x] = e^t (x + Δ_t ∇ₓ log p_t(x)).
theorem ch02_tweedie {N : ℕ} (hN : 0 < N) (lam a : Fin N → ℝ) {Δ : ℝ} (hΔ : 0 < Δ)
    (hlam : ∀ i, 0 < lam i) (t x : ℝ) :
    HasDerivAt (fun y => Real.log (ch02_marg lam a Δ (Real.exp (-t)) y))
      ((Real.exp (-t) * ch02_postmean lam a Δ (Real.exp (-t)) x - x) / Δ) x
    ∧ ch02_postmean lam a Δ (Real.exp (-t)) x
        = Real.exp t
          * (x + Δ * ((Real.exp (-t) * ch02_postmean lam a Δ (Real.exp (-t)) x - x) / Δ)) := by
  constructor
  · have h := ch02_tweedie_tb hN lam a hΔ hlam (Real.exp (-t)) x
    convert h using 1
    ring
  · exact ch02_denoiser _ _ x t Δ (ne_of_gt hΔ) rfl

-- posterior mean of the injected noise, by linearity: ⟨ξ⟩ = (x - e^{-t}⟨a⟩)/√Δ
-- ("with x passing through the conditional expectation unchanged", lines 810-813).
theorem ch02_noiseform_linearity {N : ℕ} (hN : 0 < N) (lam a : Fin N → ℝ) {Δ : ℝ}
    (hΔ : 0 < Δ) (hlam : ∀ i, 0 < lam i) (c x : ℝ) :
    ∑ i, ch02_posterior lam a Δ c x i * ((x - c * a i) / Real.sqrt Δ)
      = (x - c * ch02_postmean lam a Δ c x) / Real.sqrt Δ := by
  have hsum := ch02_posterior_sum_one hN lam a hΔ hlam c x
  have h1 : ∑ i, ch02_posterior lam a Δ c x i * ((x - c * a i) / Real.sqrt Δ)
      = ∑ i, (x / Real.sqrt Δ * ch02_posterior lam a Δ c x i
          - c / Real.sqrt Δ * (ch02_posterior lam a Δ c x i * a i)) :=
    Finset.sum_congr rfl fun i _ => by ring
  rw [h1, Finset.sum_sub_distrib, ← Finset.mul_sum, ← Finset.mul_sum, hsum,
    ← ch02_condmean lam a Δ c x]
  ring

-- eq:noiseform (lines 814-817): ⟨ξ⟩_{x,t} = -√Δ_t · S(x,t).
theorem ch02_noiseform (S A x c Δ : ℝ) (hΔ : 0 < Δ) (hS : S = (c * A - x) / Δ) :
    (x - c * A) / Real.sqrt Δ = -(Real.sqrt Δ * S) := by
  obtain ⟨s, hs0, rfl⟩ : ∃ s, 0 < s ∧ s * s = Δ :=
    ⟨Real.sqrt Δ, Real.sqrt_pos.mpr hΔ, Real.mul_self_sqrt hΔ.le⟩
  rw [Real.sqrt_mul_self hs0.le]
  subst hS
  have hsne : s ≠ 0 := ne_of_gt hs0
  field_simp
  ring

-- eq:sm-eps-score (lines 662-681): s_t(x_t) = -E[ε|x_t]/√(1-ᾱ_t), with ᾱ_t = e^{-2t}
-- so 1-ᾱ_t = Δ_t.  Given the score-posterior identity (hS) and the posterior noise
-- mean (hE, established by ch02_noiseform_linearity in the finite-prior model), the
-- displayed identity is this algebra; its second half s_θ = -ε_θ/√(1-ᾱ) is the
-- definition of the ε-parameterisation.
theorem ch02_sm_eps_score (S Eps A x c Δ : ℝ) (hΔ : 0 < Δ)
    (hS : S = (c * A - x) / Δ) (hE : Eps = (x - c * A) / Real.sqrt Δ) :
    S = -(Eps / Real.sqrt Δ) := by
  obtain ⟨s, hs0, rfl⟩ : ∃ s, 0 < s ∧ s * s = Δ :=
    ⟨Real.sqrt Δ, Real.sqrt_pos.mpr hΔ, Real.mul_self_sqrt hΔ.le⟩
  rw [Real.sqrt_mul_self hs0.le] at hE ⊢
  subst hS hE
  have hsne : s ≠ 0 := ne_of_gt hs0
  field_simp
  ring

-- eq:sm-empirical (lines 890-898): softmax weights of the empirical score.
noncomputable def ch02_softmax_w {N : ℕ} (a : Fin N → ℝ) (Δ c x : ℝ) (i : Fin N) : ℝ :=
  Real.exp (-(x - c * a i) ^ 2 / (2 * Δ)) / ∑ j, Real.exp (-(x - c * a j) ^ 2 / (2 * Δ))

-- With the uniform empirical prior (mass 1/N on each a i), the posterior mean is the
-- softmax-weighted average of training points: the (2πΔ)^{-1/2}/N factors cancel.
theorem ch02_postmean_softmax {N : ℕ} (hN : 0 < N) (a : Fin N → ℝ) {Δ : ℝ} (hΔ : 0 < Δ)
    (c x : ℝ) :
    ch02_postmean (fun _ => (N : ℝ)⁻¹) a Δ c x = ∑ i, ch02_softmax_w a Δ c x i * a i := by
  have hNc : (0:ℝ) < (N : ℝ) := Nat.cast_pos.mpr hN
  have hkpos : (0:ℝ) < Real.sqrt (2 * Real.pi * Δ) := Real.sqrt_pos.mpr (by positivity)
  have hCne : (N : ℝ)⁻¹ / Real.sqrt (2 * Real.pi * Δ) ≠ 0 :=
    ne_of_gt (div_pos (inv_pos.mpr hNc) hkpos)
  have hnum : ch02_marg_num (fun _ => (N : ℝ)⁻¹) a Δ c x
      = (N : ℝ)⁻¹ / Real.sqrt (2 * Real.pi * Δ)
        * ∑ i, Real.exp (-(x - c * a i) ^ 2 / (2 * Δ)) * a i := by
    simp only [ch02_marg_num, ch02_gauss]
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun i _ => by ring
  have hden : ch02_marg (fun _ => (N : ℝ)⁻¹) a Δ c x
      = (N : ℝ)⁻¹ / Real.sqrt (2 * Real.pi * Δ)
        * ∑ i, Real.exp (-(x - c * a i) ^ 2 / (2 * Δ)) := by
    simp only [ch02_marg, ch02_gauss]
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun i _ => by ring
  unfold ch02_postmean
  rw [hnum, hden, mul_div_mul_left _ _ hCne, Finset.sum_div]
  unfold ch02_softmax_w
  exact Finset.sum_congr rfl fun i _ => by ring

-- eq:sm-empirical (lines 890-898): ∇log p_t^emp(x) = (1/Δ)(e^{-t} ∑ᵢ wᵢ(x) aᵢ - x)
-- with wᵢ the Gaussian softmax over the N training points (1D, arbitrary N).
theorem ch02_sm_empirical {N : ℕ} (hN : 0 < N) (a : Fin N → ℝ) {Δ : ℝ} (hΔ : 0 < Δ)
    (t x : ℝ) :
    HasDerivAt
      (fun y => Real.log (ch02_marg (fun _ => (N : ℝ)⁻¹) a Δ (Real.exp (-t)) y))
      (1 / Δ * (Real.exp (-t) * ∑ i, ch02_softmax_w a Δ (Real.exp (-t)) x i * a i - x)) x := by
  have hNc : (0:ℝ) < (N : ℝ) := Nat.cast_pos.mpr hN
  have hlam : ∀ i : Fin N, 0 < (fun _ : Fin N => (N : ℝ)⁻¹) i := fun _ => inv_pos.mpr hNc
  have h : HasDerivAt
      (fun y => Real.log (ch02_marg (fun _ => (N : ℝ)⁻¹) a Δ (Real.exp (-t)) y))
      (-(x / Δ) + Real.exp (-t) / Δ
        * ∑ i, ch02_softmax_w a Δ (Real.exp (-t)) x i * a i) x := by
    have h0 := ch02_tweedie_tb hN (fun _ => (N : ℝ)⁻¹) a hΔ hlam (Real.exp (-t)) x
    rwa [ch02_postmean_softmax hN a hΔ (Real.exp (-t)) x] at h0
  convert h using 1
  ring

-- eq:tilted (lines 824-831): because (x - e^{-t}a)² = e^{-2t}(a - e^t x)², the posterior
-- mean is the prior tilted by a quadratic pinning to x̃ = e^t x with stiffness
-- κ_t = e^{-2t}/Δ_t (the (2πΔ)^{-1/2} prefactors cancel in the ratio).
theorem ch02_tilted {N : ℕ} (lam a : Fin N → ℝ) {Δ : ℝ} (hΔ : 0 < Δ) (t x : ℝ) :
    ch02_postmean lam a Δ (Real.exp (-t)) x
      = (∑ i, lam i * Real.exp (-(Real.exp (-2 * t) / Δ) * (a i - Real.exp t * x) ^ 2 / 2)
            * a i)
        / ∑ i, lam i * Real.exp (-(Real.exp (-2 * t) / Δ) * (a i - Real.exp t * x) ^ 2 / 2) := by
  have hkpos : (0:ℝ) < Real.sqrt (2 * Real.pi * Δ) := Real.sqrt_pos.mpr (by positivity)
  have hCne : (Real.sqrt (2 * Real.pi * Δ))⁻¹ ≠ 0 := inv_ne_zero (ne_of_gt hkpos)
  have hprod : Real.exp (-t) * Real.exp t = 1 := by rw [← Real.exp_add]; simp
  have h2 : Real.exp (-2 * t) = Real.exp (-t) * Real.exp (-t) := by
    rw [← Real.exp_add]; congr 1; ring
  have hexp : ∀ i : Fin N, -(x - Real.exp (-t) * a i) ^ 2 / (2 * Δ)
      = -(Real.exp (-2 * t) / Δ) * (a i - Real.exp t * x) ^ 2 / 2 := by
    intro i
    have key : Real.exp (-t) * (a i - Real.exp t * x) = Real.exp (-t) * a i - x := by
      rw [mul_sub, ← mul_assoc, hprod, one_mul]
    calc -(x - Real.exp (-t) * a i) ^ 2 / (2 * Δ)
        = -(Real.exp (-t) * a i - x) ^ 2 / (2 * Δ) := by ring
      _ = -(Real.exp (-t) * (a i - Real.exp t * x)) ^ 2 / (2 * Δ) := by rw [key]
      _ = -(Real.exp (-2 * t) / Δ) * (a i - Real.exp t * x) ^ 2 / 2 := by rw [h2]; ring
  have hnum : ch02_marg_num lam a Δ (Real.exp (-t)) x
      = (Real.sqrt (2 * Real.pi * Δ))⁻¹
        * ∑ i, lam i * Real.exp (-(Real.exp (-2 * t) / Δ) * (a i - Real.exp t * x) ^ 2 / 2)
            * a i := by
    simp only [ch02_marg_num, ch02_gauss]
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [hexp i]
    ring
  have hden : ch02_marg lam a Δ (Real.exp (-t)) x
      = (Real.sqrt (2 * Real.pi * Δ))⁻¹
        * ∑ i, lam i
            * Real.exp (-(Real.exp (-2 * t) / Δ) * (a i - Real.exp t * x) ^ 2 / 2) := by
    simp only [ch02_marg, ch02_gauss]
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [hexp i]
    ring
  unfold ch02_postmean
  rw [hnum, hden, mul_div_mul_left _ _ hCne]

-- eq:potentialform (lines 856-859): S(x,t) = e^t ⟨-∇V₀⟩_{x,t}.  The integration-by-
-- parts (Stein) step ⟨-V₀'⟩ = κ_t(⟨a⟩ - x̃) is taken as hypothesis (hstein); given it
-- and the score-posterior identity (hS), the boxed display follows algebraically, with
-- κ_t = e^{-2t}/Δ and x̃ = e^t x exactly as in the text.
theorem ch02_potentialform (S A F x t Δ : ℝ) (_hΔ : Δ ≠ 0)
    (hS : S = (Real.exp (-t) * A - x) / Δ)
    (hstein : F = Real.exp (-2 * t) / Δ * (A - Real.exp t * x)) :
    Real.exp t * F = S := by
  subst hS hstein
  have key1 : Real.exp t * Real.exp (-2 * t) = Real.exp (-t) := by
    rw [← Real.exp_add]; congr 1; ring
  have key2 : Real.exp (-t) * Real.exp t = 1 := by rw [← Real.exp_add]; simp
  calc Real.exp t * (Real.exp (-2 * t) / Δ * (A - Real.exp t * x))
      = Real.exp t * Real.exp (-2 * t) * (A - Real.exp t * x) / Δ := by ring
    _ = Real.exp (-t) * (A - Real.exp t * x) / Δ := by rw [key1]
    _ = (Real.exp (-t) * A - Real.exp (-t) * Real.exp t * x) / Δ := by ring
    _ = (Real.exp (-t) * A - x) / Δ := by rw [key2, one_mul]

-- eq:tweedie-joint (lines 874-878): the componentwise score with the FULL posterior.
-- Instance check with L = 2 frames: for a joint finitely-supported prior on pairs
-- (a i, b i) and independent channels on the two coordinates, the ∂ₓ-component of the
-- score of the joint mixture is (e^{-t}·E[a | x, y] - x)/Δ — the conditional mean of
-- the first frame given BOTH observations (the joint softmax weights).
noncomputable def ch02_marg2 {N : ℕ} (lam a b : Fin N → ℝ) (Δ c x y : ℝ) : ℝ :=
  ∑ i, lam i * ch02_gauss Δ y (c * b i) * ch02_gauss Δ x (c * a i)

noncomputable def ch02_postmean2 {N : ℕ} (lam a b : Fin N → ℝ) (Δ c x y : ℝ) : ℝ :=
  (∑ i, lam i * ch02_gauss Δ y (c * b i) * ch02_gauss Δ x (c * a i) * a i)
    / ch02_marg2 lam a b Δ c x y

theorem ch02_tweedie_joint {N : ℕ} (hN : 0 < N) (lam a b : Fin N → ℝ) {Δ : ℝ} (hΔ : 0 < Δ)
    (hlam : ∀ i, 0 < lam i) (c x y : ℝ) :
    HasDerivAt (fun x' => Real.log (ch02_marg2 lam a b Δ c x' y))
      ((c * ch02_postmean2 lam a b Δ c x y - x) / Δ) x := by
  have hμ : ∀ i, 0 < lam i * ch02_gauss Δ y (c * b i) :=
    fun i => mul_pos (hlam i) (ch02_gauss_pos hΔ _ _)
  have h : HasDerivAt (fun x' => Real.log (ch02_marg2 lam a b Δ c x' y))
      (-(x / Δ) + c / Δ * ch02_postmean2 lam a b Δ c x y) x :=
    ch02_tweedie_tb hN (fun i => lam i * ch02_gauss Δ y (c * b i)) a hΔ hμ c x
  convert h using 1
  ring

-- eq:sm-dsm (lines 909-916): the denoising score-matching objective.  A definition:
-- the pointwise integrand ‖s_φ(e^{-t}a + √Δ_t ε, t) + ε/√Δ_t‖² (1D); the displayed
-- objective is its expectation over t, a, ε.  Sanity: it is nonnegative.
noncomputable def ch02_sm_dsm (s : ℝ → ℝ → ℝ) (t a ε : ℝ) : ℝ :=
  (s (Real.exp (-t) * a + Real.sqrt (ch02_Delta t) * ε) t + ε / Real.sqrt (ch02_Delta t)) ^ 2

theorem ch02_sm_dsm_nonneg (s : ℝ → ℝ → ℝ) (t a ε : ℝ) : 0 ≤ ch02_sm_dsm s t a ε := by
  unfold ch02_sm_dsm
  positivity

/-! ### Pass 2: strengthening -/

-- Second x-derivative of the Gaussian kernel, for eq:sm-fokkerplanck below:
-- ∂ₓ[-((x-m)/v)·G] = (-(1/v) + ((x-m)/v)²)·G, i.e. ∂ₓ²G = ((x-m)²/v² - 1/v)·G.
theorem ch02_gauss_deriv2 {v : ℝ} (hv : 0 < v) (x m : ℝ) :
    HasDerivAt (fun y => -((y - m) / v) * ch02_gauss v y m)
      ((-(1 / v) + ((x - m) / v) ^ 2) * ch02_gauss v x m) x := by
  have h0 : HasDerivAt (fun y : ℝ => (y - m) / v) (1 / v) x :=
    ((hasDerivAt_id x).sub_const m).div_const v
  have hlin : HasDerivAt (fun y : ℝ => -((y - m) / v)) (-(1 / v)) x := h0.neg
  have h : HasDerivAt (fun y => -((y - m) / v) * ch02_gauss v y m)
      (-(1 / v) * ch02_gauss v x m
        + -((x - m) / v) * (-((x - m) / v) * ch02_gauss v x m)) x :=
    hlin.mul (ch02_kernelgrad hv x m)
  have heq : -(1 / v) * ch02_gauss v x m
      + -((x - m) / v) * (-((x - m) / v) * ch02_gauss v x m)
      = (-(1 / v) + ((x - m) / v) ^ 2) * ch02_gauss v x m := by ring
  rw [heq] at h
  exact h

-- The OU transition density of eq:sm-ou, written in log form so that its t-derivative
-- is available without differentiating a square root.
noncomputable def ch02_ouLogDensity (a t x : ℝ) : ℝ :=
  -(x - Real.exp (-t) * a) ^ 2 / (2 * ch02_Delta t)
    - Real.log (2 * Real.pi * ch02_Delta t) / 2

noncomputable def ch02_ouDensity (a t x : ℝ) : ℝ := Real.exp (ch02_ouLogDensity a t x)

-- For t > 0 it is exactly the Gaussian G(x; e^{-t}a, Δ_t) of eq:sm-ou.
theorem ch02_ouDensity_eq_gauss (a x : ℝ) {t : ℝ} (ht : 0 < t) :
    ch02_ouDensity a t x = ch02_gauss (ch02_Delta t) x (Real.exp (-t) * a) := by
  have hΔ : 0 < ch02_Delta t := ch02_sm_ou_Delta_pos ht
  have hz : (0:ℝ) < 2 * Real.pi * ch02_Delta t := by positivity
  have hsqrt : Real.sqrt (2 * Real.pi * ch02_Delta t)
      = Real.exp (Real.log (2 * Real.pi * ch02_Delta t) / 2) := by
    rw [← Real.log_sqrt hz.le, Real.exp_log (Real.sqrt_pos.mpr hz)]
  unfold ch02_ouDensity ch02_ouLogDensity ch02_gauss
  rw [hsqrt, Real.exp_sub]

-- t-derivative of the log density, using Δ_t' = 2 - 2Δ_t (ch02_sm_ou_var_ode) and
-- (d/dt) e^{-t}a = -(e^{-t}a).
theorem ch02_ouLogDensity_deriv (a x : ℝ) {t : ℝ} (ht : 0 < t) :
    HasDerivAt (fun s => ch02_ouLogDensity a s x)
      (-((x - Real.exp (-t) * a) * (Real.exp (-t) * a)) / ch02_Delta t
        + (x - Real.exp (-t) * a) ^ 2 * (2 - 2 * ch02_Delta t) / (2 * ch02_Delta t ^ 2)
        - (2 - 2 * ch02_Delta t) / (2 * ch02_Delta t)) t := by
  have hΔ : 0 < ch02_Delta t := ch02_sm_ou_Delta_pos ht
  have hΔne : ch02_Delta t ≠ 0 := ne_of_gt hΔ
  have hc : HasDerivAt (fun s : ℝ => Real.exp (-s) * a) (-(Real.exp (-t)) * a) t := by
    have h : HasDerivAt (fun s : ℝ => -s) (-1) t := (hasDerivAt_id t).neg
    have h2 : HasDerivAt (fun s : ℝ => Real.exp (-s)) (Real.exp (-t) * -1) t := h.exp
    have h3 := h2.mul_const a
    have heq : Real.exp (-t) * -1 * a = -(Real.exp (-t)) * a := by ring
    rwa [heq] at h3
  have hd : HasDerivAt (fun s : ℝ => x - Real.exp (-s) * a) (Real.exp (-t) * a) t := by
    have h := hc.const_sub x
    have heq : -(-(Real.exp (-t)) * a) = Real.exp (-t) * a := by ring
    rwa [heq] at h
  have hsq : HasDerivAt (fun s : ℝ => (x - Real.exp (-s) * a) ^ 2)
      (2 * (x - Real.exp (-t) * a) * (Real.exp (-t) * a)) t := by
    have h := hd.fun_pow 2
    simpa using h
  have hnum : HasDerivAt (fun s : ℝ => -(x - Real.exp (-s) * a) ^ 2)
      (-(2 * (x - Real.exp (-t) * a) * (Real.exp (-t) * a))) t := hsq.neg
  have hden : HasDerivAt (fun s : ℝ => 2 * ch02_Delta s) (2 * (2 - 2 * ch02_Delta t)) t :=
    (ch02_sm_ou_var_ode t).const_mul 2
  have hdenne : (2:ℝ) * ch02_Delta t ≠ 0 := by
    exact mul_ne_zero two_ne_zero hΔne
  have hlogarg : HasDerivAt (fun s : ℝ => 2 * Real.pi * ch02_Delta s)
      (2 * Real.pi * (2 - 2 * ch02_Delta t)) t := (ch02_sm_ou_var_ode t).const_mul (2 * Real.pi)
  have hlogargne : 2 * Real.pi * ch02_Delta t ≠ 0 := by positivity
  have hpine : Real.pi ≠ 0 := Real.pi_ne_zero
  have hfinal : HasDerivAt (fun s : ℝ => -(x - Real.exp (-s) * a) ^ 2 / (2 * ch02_Delta s)
        - Real.log (2 * Real.pi * ch02_Delta s) / 2)
      ((-(2 * (x - Real.exp (-t) * a) * (Real.exp (-t) * a)) * (2 * ch02_Delta t)
          - -(x - Real.exp (-t) * a) ^ 2 * (2 * (2 - 2 * ch02_Delta t)))
            / (2 * ch02_Delta t) ^ 2
        - 2 * Real.pi * (2 - 2 * ch02_Delta t) / (2 * Real.pi * ch02_Delta t) / 2) t :=
    (hnum.div hden hdenne).sub ((hlogarg.log hlogargne).div_const 2)
  have heq : (-(2 * (x - Real.exp (-t) * a) * (Real.exp (-t) * a)) * (2 * ch02_Delta t)
          - -(x - Real.exp (-t) * a) ^ 2 * (2 * (2 - 2 * ch02_Delta t)))
            / (2 * ch02_Delta t) ^ 2
        - 2 * Real.pi * (2 - 2 * ch02_Delta t) / (2 * Real.pi * ch02_Delta t) / 2
      = -((x - Real.exp (-t) * a) * (Real.exp (-t) * a)) / ch02_Delta t
        + (x - Real.exp (-t) * a) ^ 2 * (2 - 2 * ch02_Delta t) / (2 * ch02_Delta t ^ 2)
        - (2 - 2 * ch02_Delta t) / (2 * ch02_Delta t) := by
    field_simp
    ring
  rw [heq] at hfinal
  unfold ch02_ouLogDensity
  exact hfinal

-- eq:sm-ou / eq:sm-fokkerplanck (lines 240-245 and 524-534): the STRONG cross-check.
-- The closed-form OU transition density of eq:sm-ou, with Δ_t = 1 - e^{-2t} exactly as
-- displayed, satisfies the Fokker-Planck equation eq:sm-fokkerplanck for the OU choice
-- f(x,t) = -x, g = √2 (so ½g² = 1):   ∂_t p = ∂ₓ(x p) + ∂ₓ²p.
-- Any other exponent or constant in Δ_t would break this identity.
theorem ch02_sm_ou_fokkerplanck (a x : ℝ) {t : ℝ} (ht : 0 < t) :
    HasDerivAt (fun s => ch02_ouDensity a s x)
      (deriv (fun y => y * ch02_ouDensity a t y) x
        + deriv (deriv (fun y => ch02_ouDensity a t y)) x) t := by
  have hΔ : 0 < ch02_Delta t := ch02_sm_ou_Delta_pos ht
  have hΔne : ch02_Delta t ≠ 0 := ne_of_gt hΔ
  have hfun : (fun y => ch02_ouDensity a t y)
      = fun y => ch02_gauss (ch02_Delta t) y (Real.exp (-t) * a) := by
    funext y
    exact ch02_ouDensity_eq_gauss a y ht
  have hd1 : deriv (fun y => ch02_ouDensity a t y)
      = fun y => -((y - Real.exp (-t) * a) / ch02_Delta t)
          * ch02_gauss (ch02_Delta t) y (Real.exp (-t) * a) := by
    rw [hfun]
    funext y
    exact (ch02_kernelgrad hΔ y _).deriv
  have hd2 : deriv (deriv (fun y => ch02_ouDensity a t y)) x
      = (-(1 / ch02_Delta t) + ((x - Real.exp (-t) * a) / ch02_Delta t) ^ 2)
        * ch02_gauss (ch02_Delta t) x (Real.exp (-t) * a) := by
    rw [hd1]
    exact (ch02_gauss_deriv2 hΔ x _).deriv
  have hd3 : deriv (fun y => y * ch02_ouDensity a t y) x
      = 1 * ch02_gauss (ch02_Delta t) x (Real.exp (-t) * a)
        + x * (-((x - Real.exp (-t) * a) / ch02_Delta t)
              * ch02_gauss (ch02_Delta t) x (Real.exp (-t) * a)) := by
    have hxfun : (fun y => y * ch02_ouDensity a t y)
        = fun y => y * ch02_gauss (ch02_Delta t) y (Real.exp (-t) * a) := by
      funext y
      rw [ch02_ouDensity_eq_gauss a y ht]
    rw [hxfun]
    exact ((hasDerivAt_id x).mul (ch02_kernelgrad hΔ x _)).deriv
  rw [hd2, hd3]
  have hlogd := ch02_ouLogDensity_deriv a x ht
  have hexp : HasDerivAt (fun s => ch02_ouDensity a s x) _ t := hlogd.exp
  convert hexp using 1
  have hG : Real.exp (ch02_ouLogDensity a t x)
      = ch02_gauss (ch02_Delta t) x (Real.exp (-t) * a) := ch02_ouDensity_eq_gauss a x ht
  rw [hG]
  field_simp
  ring

-- eq:tweedie-joint (lines 877-881), general number of frames.  The distinguished
-- coordinate is x with prior values `a`; the remaining L-1 observations are `y` with
-- prior values `rest`.  The channel is independent across frames (the product of
-- kernels), but the prior couples them, so the posterior mean of frame k depends on
-- every observation.
noncomputable def ch02_margL {N M : ℕ} (lam : Fin N → ℝ) (a : Fin N → ℝ)
    (rest : Fin N → Fin M → ℝ) (Δ c x : ℝ) (y : Fin M → ℝ) : ℝ :=
  ∑ i, (lam i * ∏ j, ch02_gauss Δ (y j) (c * rest i j)) * ch02_gauss Δ x (c * a i)

noncomputable def ch02_postmeanL {N M : ℕ} (lam : Fin N → ℝ) (a : Fin N → ℝ)
    (rest : Fin N → Fin M → ℝ) (Δ c x : ℝ) (y : Fin M → ℝ) : ℝ :=
  (∑ i, (lam i * ∏ j, ch02_gauss Δ (y j) (c * rest i j)) * ch02_gauss Δ x (c * a i) * a i)
    / ch02_margL lam a rest Δ c x y

theorem ch02_tweedie_joint_L {N M : ℕ} (hN : 0 < N) (lam : Fin N → ℝ) (a : Fin N → ℝ)
    (rest : Fin N → Fin M → ℝ) {Δ : ℝ} (hΔ : 0 < Δ) (hlam : ∀ i, 0 < lam i)
    (c x : ℝ) (y : Fin M → ℝ) :
    HasDerivAt (fun x' => Real.log (ch02_margL lam a rest Δ c x' y))
      ((c * ch02_postmeanL lam a rest Δ c x y - x) / Δ) x := by
  have hμ : ∀ i, 0 < lam i * ∏ j, ch02_gauss Δ (y j) (c * rest i j) := fun i =>
    mul_pos (hlam i) (Finset.prod_pos (fun j _ => ch02_gauss_pos hΔ _ _))
  have h : HasDerivAt (fun x' => Real.log (ch02_margL lam a rest Δ c x' y))
      (-(x / Δ) + c / Δ * ch02_postmeanL lam a rest Δ c x y) x :=
    ch02_tweedie_tb hN (fun i => lam i * ∏ j, ch02_gauss Δ (y j) (c * rest i j)) a hΔ hμ c x
  convert h using 1
  ring

-- eq:sm-dsm (lines 915-922): the population optimum of the denoising score-matching
-- objective.  For any weighting w of the noise with ∑ w = 1, the quadratic
-- s ↦ ∑ᵢ wᵢ (s + zᵢ)² splits as (variance term) + (s + mean)², so it is minimised
-- exactly at s = -∑ᵢ wᵢ zᵢ.  With zᵢ = εᵢ/√Δ_t this says the minimiser of eq:sm-dsm
-- is -E[ε | x_t]/√Δ_t, which is the score by eq:sm-eps-score.
theorem ch02_sm_dsm_population_optimum {N : ℕ} (w z : Fin N → ℝ) (hw1 : ∑ i, w i = 1)
    (s : ℝ) :
    ∑ i, w i * (s + z i) ^ 2
      = (∑ i, w i * (z i - ∑ j, w j * z j) ^ 2) + (s + ∑ j, w j * z j) ^ 2 := by
  have key : ∀ i : Fin N, w i * (s + z i) ^ 2
      = w i * (z i - ∑ j, w j * z j) ^ 2
        + ((s + ∑ j, w j * z j) ^ 2 - 2 * (s + ∑ j, w j * z j) * ∑ j, w j * z j) * w i
        + 2 * (s + ∑ j, w j * z j) * (w i * z i) := by
    intro i; ring
  rw [Finset.sum_congr rfl (fun i _ => key i), Finset.sum_add_distrib, Finset.sum_add_distrib,
    ← Finset.mul_sum, ← Finset.mul_sum, hw1]
  ring

theorem ch02_sm_dsm_min {N : ℕ} (w z : Fin N → ℝ) (hw1 : ∑ i, w i = 1) (s : ℝ) :
    ∑ i, w i * ((-∑ j, w j * z j) + z i) ^ 2 ≤ ∑ i, w i * (s + z i) ^ 2 := by
  have h1 := ch02_sm_dsm_population_optimum w z hw1 s
  have h2 := ch02_sm_dsm_population_optimum w z hw1 (-∑ j, w j * z j)
  have h3 : (-∑ j, w j * z j + ∑ j, w j * z j) ^ 2 = 0 := by ring
  rw [h3] at h2
  have h4 : (0:ℝ) ≤ (s + ∑ j, w j * z j) ^ 2 := sq_nonneg _
  linarith

/-! ### Pass 3: from instance checks to general theorems

Three of the chapter's displays were previously recorded only at fixed sizes or
under an assumed identity.  This section replaces those instance checks by
theorems general in the relevant parameter: the number of frames L for
eq:tweedie-joint, the dimension d for eq:sm-langevin, and the whole protocol
(state space, number of steps, energies, kernels) for eq:sm-jarzynski-physical. -/

/-! #### eq:tweedie-joint (lines 877-881) for an arbitrary number of frames

The prior is a finitely supported measure on ℝ^L: atoms `A i : Fin L → ℝ` with
positive weights `lam i`.  The channel acts independently on the L frames (the
product of kernels), but the prior couples them.  `k` is the distinguished
coordinate, and differentiating in it means differentiating the marginal along
`Function.update x k ·`, i.e. taking the k-th partial derivative.  The result is
the display verbatim: ∂_k log p_t(x) = (e^{-t}·E[a_k | x] - x_k)/Δ_t with the
posterior mean taken under the FULL joint posterior given all L observations. -/

noncomputable def ch02_margFull {N L : ℕ} (lam : Fin N → ℝ) (A : Fin N → Fin L → ℝ)
    (Δ c : ℝ) (x : Fin L → ℝ) : ℝ :=
  ∑ i, lam i * ∏ j, ch02_gauss Δ (x j) (c * A i j)

noncomputable def ch02_postmeanFull {N L : ℕ} (lam : Fin N → ℝ) (A : Fin N → Fin L → ℝ)
    (Δ c : ℝ) (x : Fin L → ℝ) (k : Fin L) : ℝ :=
  (∑ i, lam i * (∏ j, ch02_gauss Δ (x j) (c * A i j)) * A i k)
    / ch02_margFull lam A Δ c x

-- the weights the other L-1 frames contribute to the k-th one
noncomputable def ch02_restWeight {N L : ℕ} (lam : Fin N → ℝ) (A : Fin N → Fin L → ℝ)
    (Δ c : ℝ) (x : Fin L → ℝ) (k : Fin L) (i : Fin N) : ℝ :=
  lam i * ∏ j ∈ Finset.univ.erase k, ch02_gauss Δ (x j) (c * A i j)

theorem ch02_prod_split {N L : ℕ} (lam : Fin N → ℝ) (A : Fin N → Fin L → ℝ)
    (Δ c : ℝ) (x : Fin L → ℝ) (k : Fin L) (i : Fin N) :
    lam i * ∏ j, ch02_gauss Δ (x j) (c * A i j)
      = ch02_restWeight lam A Δ c x k i * ch02_gauss Δ (x k) (c * A i k) := by
  unfold ch02_restWeight
  rw [← Finset.mul_prod_erase Finset.univ
    (fun j => ch02_gauss Δ (x j) (c * A i j)) (Finset.mem_univ k)]
  ring

theorem ch02_restWeight_update {N L : ℕ} (lam : Fin N → ℝ) (A : Fin N → Fin L → ℝ)
    (Δ c : ℝ) (x : Fin L → ℝ) (k : Fin L) (s : ℝ) (i : Fin N) :
    ch02_restWeight lam A Δ c (Function.update x k s) k i
      = ch02_restWeight lam A Δ c x k i := by
  unfold ch02_restWeight
  congr 1
  refine Finset.prod_congr rfl (fun j hj => ?_)
  have hjk : j ≠ k := Finset.ne_of_mem_erase hj
  simp [hjk]

theorem ch02_margFull_update {N L : ℕ} (lam : Fin N → ℝ) (A : Fin N → Fin L → ℝ)
    (Δ c : ℝ) (x : Fin L → ℝ) (k : Fin L) (s : ℝ) :
    ch02_margFull lam A Δ c (Function.update x k s)
      = ch02_marg (ch02_restWeight lam A Δ c x k) (fun i => A i k) Δ c s := by
  unfold ch02_margFull ch02_marg
  refine Finset.sum_congr rfl (fun i _ => ?_)
  rw [ch02_prod_split lam A Δ c (Function.update x k s) k i,
    ch02_restWeight_update lam A Δ c x k s i]
  simp

theorem ch02_postmeanFull_eq {N L : ℕ} (lam : Fin N → ℝ) (A : Fin N → Fin L → ℝ)
    (Δ c : ℝ) (x : Fin L → ℝ) (k : Fin L) :
    ch02_postmeanFull lam A Δ c x k
      = ch02_postmean (ch02_restWeight lam A Δ c x k) (fun i => A i k) Δ c (x k) := by
  unfold ch02_postmeanFull ch02_postmean ch02_marg_num ch02_marg ch02_margFull
  congr 1
  · exact Finset.sum_congr rfl (fun i _ => by
      rw [ch02_prod_split lam A Δ c x k i])
  · exact Finset.sum_congr rfl (fun i _ => ch02_prod_split lam A Δ c x k i)

-- eq:tweedie-joint, general L: the k-th component of the score of the joint
-- marginal is (e^{-t}⟨a_k⟩ - x_k)/Δ_t, the posterior mean being conditioned on
-- every frame.  Arbitrary N atoms, arbitrary L, arbitrary distinguished k.
theorem ch02_tweedie_joint_full {N L : ℕ} (hN : 0 < N) (lam : Fin N → ℝ)
    (A : Fin N → Fin L → ℝ) {Δ : ℝ} (hΔ : 0 < Δ) (hlam : ∀ i, 0 < lam i)
    (c : ℝ) (x : Fin L → ℝ) (k : Fin L) :
    HasDerivAt (fun s => Real.log (ch02_margFull lam A Δ c (Function.update x k s)))
      ((c * ch02_postmeanFull lam A Δ c x k - x k) / Δ) (x k) := by
  have hμ : ∀ i, 0 < ch02_restWeight lam A Δ c x k i := fun i =>
    mul_pos (hlam i) (Finset.prod_pos (fun j _ => ch02_gauss_pos hΔ _ _))
  have hfun : (fun s => Real.log (ch02_margFull lam A Δ c (Function.update x k s)))
      = fun s => Real.log (ch02_marg (ch02_restWeight lam A Δ c x k)
          (fun i => A i k) Δ c s) := by
    funext s
    rw [ch02_margFull_update]
  rw [hfun, ch02_postmeanFull_eq]
  have h := ch02_tweedie_tb hN (ch02_restWeight lam A Δ c x k) (fun i => A i k) hΔ hμ c (x k)
  convert h using 1
  ring

/-! #### eq:sm-langevin (lines 227-230) in arbitrary dimension

Toolbox tb:sm-relaxation argues that e^{-βE} is stationary for the overdamped
Langevin dynamics because the Fokker-Planck current
J = -∇E·p - β⁻¹∇p vanishes *pointwise* at p ∝ e^{-βE}, the chain rule giving
∇p = -β∇E·p.  That computation is reproduced here in full generality: any real
normed space (so any ℝ^d), any differentiable potential E, any β ≠ 0.  The
current is contracted against an arbitrary direction v, so the conclusion is
that the current vector field itself is identically zero — the "detailed
balance" statement the toolbox emphasises, not merely divergence-freeness.
(The Fokker-Planck equation itself is eq:sm-fokkerplanck, which is assumed here;
uniqueness of the stationary law is the ergodicity hypothesis the text states
and cites, and is not formalised.) -/

theorem ch02_sm_langevin_flux_nd {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]
    (E : F → ℝ) (β : ℝ) (hβ : β ≠ 0) (hE : Differentiable ℝ E) (v : F) :
    (fun y => fderiv ℝ E y v * Real.exp (-(β * E y))
        + β⁻¹ * fderiv ℝ (fun z => Real.exp (-(β * E z))) y v) = fun _ => (0:ℝ) := by
  funext y
  have h1 : HasFDerivAt E (fderiv ℝ E y) y := (hE y).hasFDerivAt
  have h2 : HasFDerivAt (fun z => -(β * E z)) (-(β • fderiv ℝ E y)) y :=
    (h1.const_mul β).neg
  have h3 : HasFDerivAt (fun z => Real.exp (-(β * E z)))
      (Real.exp (-(β * E y)) • (-(β • fderiv ℝ E y))) y := h2.exp
  rw [h3.fderiv]
  have happ : (Real.exp (-(β * E y)) • (-(β • fderiv ℝ E y))) v
      = Real.exp (-(β * E y)) * -(β * fderiv ℝ E y v) := by
    simp
  rw [happ]
  have hcancel : β⁻¹ * (Real.exp (-(β * E y)) * -(β * fderiv ℝ E y v))
      = -((β⁻¹ * β) * (Real.exp (-(β * E y)) * fderiv ℝ E y v)) := by ring
  rw [hcancel, inv_mul_cancel₀ hβ]
  ring

-- consequence in ℝ^d: the divergence ∑_k ∂_k J_k of the current vanishes, which
-- is ∂_t p = 0 for p = e^{-βE} in the Fokker-Planck equation of eq:sm-langevin.
theorem ch02_sm_langevin_fokkerplanck_nd {d : ℕ} (E : EuclideanSpace ℝ (Fin d) → ℝ) (β : ℝ)
    (hβ : β ≠ 0) (hE : Differentiable ℝ E) (y : EuclideanSpace ℝ (Fin d)) :
    ∑ k : Fin d, fderiv ℝ (fun z =>
        fderiv ℝ E z (EuclideanSpace.single k (1:ℝ)) * Real.exp (-(β * E z))
          + β⁻¹ * fderiv ℝ (fun w => Real.exp (-(β * E w))) z
              (EuclideanSpace.single k (1:ℝ))) y (EuclideanSpace.single k (1:ℝ)) = 0 := by
  refine Finset.sum_eq_zero (fun k _ => ?_)
  rw [ch02_sm_langevin_flux_nd E β hβ hE (EuclideanSpace.single k (1:ℝ))]
  simp

/-! #### eq:sm-jarzynski-physical (lines 336-339) from local detailed balance

The chapter states E[e^{-β(W-ΔF)}] = 1 for a driven system and cites Jarzynski
and Crooks for it.  Here it is proved, rather than assumed, for the standard
discrete driven model of that theory: a finite state space, a protocol of
energies E_0,…,E_T, and relaxation kernels K_k that satisfy local detailed
balance with respect to e^{-βE_k}.  A step of the protocol first changes the
energy at fixed configuration — that increment is the work — and then relaxes
with K_k.  W is the total work along the path, ΔF = F_T - F_0 is built from the
true partition functions, Q and P are the forward and reverse path laws.

Nothing here is assumed beyond detailed balance and row-stochasticity of the
kernels: the pathwise Crooks relation β(W - ΔF) = log(Q/P) is derived
(ch02_sm_crooks_pathwise), and the reverse path law is proved to be normalised
over the finite path space (ch02_reversePathLaw_sum_one), so the Jarzynski
equality follows with no free hypotheses.  The continuous-time Hamiltonian
derivation remains out of reach. -/

noncomputable def ch02_protocolZ {σ : Type*} [Fintype σ] (β : ℝ) (E : ℕ → σ → ℝ) (k : ℕ) : ℝ :=
  ∑ y, Real.exp (-(β * E k y))

theorem ch02_protocolZ_pos {σ : Type*} [Fintype σ] [Nonempty σ] (β : ℝ) (E : ℕ → σ → ℝ)
    (k : ℕ) : 0 < ch02_protocolZ β E k :=
  Finset.sum_pos (fun y _ => Real.exp_pos _) Finset.univ_nonempty

noncomputable def ch02_freeEnergy {σ : Type*} [Fintype σ] (β : ℝ) (E : ℕ → σ → ℝ) (k : ℕ) : ℝ :=
  -Real.log (ch02_protocolZ β E k) / β

-- work: at step k the protocol moves E_k → E_{k+1} at fixed configuration x_k.
noncomputable def ch02_pathWork {σ : Type*} (E : ℕ → σ → ℝ) (T : ℕ) (x : ℕ → σ) : ℝ :=
  ∑ k ∈ Finset.range T, (E (k + 1) (x k) - E k (x k))

noncomputable def ch02_forwardPathLaw {σ : Type*} [Fintype σ] (β : ℝ) (E : ℕ → σ → ℝ)
    (K : ℕ → σ → σ → ℝ) (T : ℕ) (x : ℕ → σ) : ℝ :=
  Real.exp (-(β * E 0 (x 0))) / ch02_protocolZ β E 0
    * ∏ k ∈ Finset.range T, K (k + 1) (x k) (x (k + 1))

noncomputable def ch02_reversePathLaw {σ : Type*} [Fintype σ] (β : ℝ) (E : ℕ → σ → ℝ)
    (K : ℕ → σ → σ → ℝ) (T : ℕ) (x : ℕ → σ) : ℝ :=
  Real.exp (-(β * E T (x T))) / ch02_protocolZ β E T
    * ∏ k ∈ Finset.range T, K (k + 1) (x (k + 1)) (x k)

theorem ch02_forwardPathLaw_pos {σ : Type*} [Fintype σ] [Nonempty σ] (β : ℝ) (E : ℕ → σ → ℝ)
    (K : ℕ → σ → σ → ℝ) (T : ℕ) (x : ℕ → σ) (hKpos : ∀ k y z, 0 < K k y z) :
    0 < ch02_forwardPathLaw β E K T x :=
  mul_pos (div_pos (Real.exp_pos _) (ch02_protocolZ_pos β E 0))
    (Finset.prod_pos (fun k _ => hKpos _ _ _))

theorem ch02_reversePathLaw_pos {σ : Type*} [Fintype σ] [Nonempty σ] (β : ℝ) (E : ℕ → σ → ℝ)
    (K : ℕ → σ → σ → ℝ) (T : ℕ) (x : ℕ → σ) (hKpos : ∀ k y z, 0 < K k y z) :
    0 < ch02_reversePathLaw β E K T x :=
  mul_pos (div_pos (Real.exp_pos _) (ch02_protocolZ_pos β E T))
    (Finset.prod_pos (fun k _ => hKpos _ _ _))

-- Crooks' pathwise relation, DERIVED from local detailed balance: the dissipated
-- work equals the log-ratio of the forward and reverse path laws, for every path.
theorem ch02_sm_crooks_pathwise {σ : Type*} [Fintype σ] [Nonempty σ] (β : ℝ) (hβ : β ≠ 0)
    (E : ℕ → σ → ℝ) (K : ℕ → σ → σ → ℝ) (T : ℕ) (x : ℕ → σ)
    (hKpos : ∀ k y z, 0 < K k y z)
    (hDB : ∀ k y z, Real.exp (-(β * E k y)) * K k y z
      = Real.exp (-(β * E k z)) * K k z y) :
    β * (ch02_pathWork E T x - (ch02_freeEnergy β E T - ch02_freeEnergy β E 0))
      = Real.log (ch02_forwardPathLaw β E K T x / ch02_reversePathLaw β E K T x) := by
  have hKne : ∀ k y z, K k y z ≠ 0 := fun k y z => ne_of_gt (hKpos k y z)
  have hZ : ∀ k, 0 < ch02_protocolZ β E k := fun k => ch02_protocolZ_pos β E k
  have hQpos := ch02_forwardPathLaw_pos β E K T x hKpos
  have hPpos := ch02_reversePathLaw_pos β E K T x hKpos
  have hprodQ : (0:ℝ) < ∏ k ∈ Finset.range T, K (k + 1) (x k) (x (k + 1)) :=
    Finset.prod_pos (fun k _ => hKpos _ _ _)
  have hprodP : (0:ℝ) < ∏ k ∈ Finset.range T, K (k + 1) (x (k + 1)) (x k) :=
    Finset.prod_pos (fun k _ => hKpos _ _ _)
  have hlogQ : Real.log (ch02_forwardPathLaw β E K T x)
      = -(β * E 0 (x 0)) - Real.log (ch02_protocolZ β E 0)
        + ∑ k ∈ Finset.range T, Real.log (K (k + 1) (x k) (x (k + 1))) := by
    unfold ch02_forwardPathLaw
    rw [Real.log_mul (ne_of_gt (div_pos (Real.exp_pos _) (hZ 0))) (ne_of_gt hprodQ),
      Real.log_div (Real.exp_ne_zero _) (ne_of_gt (hZ 0)), Real.log_exp,
      Real.log_prod (fun k _ => hKne _ _ _)]
  have hlogP : Real.log (ch02_reversePathLaw β E K T x)
      = -(β * E T (x T)) - Real.log (ch02_protocolZ β E T)
        + ∑ k ∈ Finset.range T, Real.log (K (k + 1) (x (k + 1)) (x k)) := by
    unfold ch02_reversePathLaw
    rw [Real.log_mul (ne_of_gt (div_pos (Real.exp_pos _) (hZ T))) (ne_of_gt hprodP),
      Real.log_div (Real.exp_ne_zero _) (ne_of_gt (hZ T)), Real.log_exp,
      Real.log_prod (fun k _ => hKne _ _ _)]
  have hstep : ∀ k ∈ Finset.range T,
      Real.log (K (k + 1) (x k) (x (k + 1))) - Real.log (K (k + 1) (x (k + 1)) (x k))
        = -(β * (E (k + 1) (x (k + 1)) - E (k + 1) (x k))) := by
    intro k _
    have h := hDB (k + 1) (x k) (x (k + 1))
    have h2 : Real.log (Real.exp (-(β * E (k + 1) (x k))) * K (k + 1) (x k) (x (k + 1)))
        = Real.log (Real.exp (-(β * E (k + 1) (x (k + 1)))) * K (k + 1) (x (k + 1)) (x k)) := by
      rw [h]
    rw [Real.log_mul (Real.exp_ne_zero _) (hKne _ _ _),
      Real.log_mul (Real.exp_ne_zero _) (hKne _ _ _), Real.log_exp, Real.log_exp] at h2
    linarith
  have hdiff : ∑ k ∈ Finset.range T, Real.log (K (k + 1) (x k) (x (k + 1)))
      - ∑ k ∈ Finset.range T, Real.log (K (k + 1) (x (k + 1)) (x k))
      = -(β * ∑ k ∈ Finset.range T, (E (k + 1) (x (k + 1)) - E (k + 1) (x k))) := by
    rw [← Finset.sum_sub_distrib, Finset.sum_congr rfl hstep, Finset.mul_sum,
      ← Finset.sum_neg_distrib]
  have htel : ∑ k ∈ Finset.range T, (E (k + 1) (x (k + 1)) - E k (x k))
      = E T (x T) - E 0 (x 0) := Finset.sum_range_sub (fun k => E k (x k)) T
  have hsum : ch02_pathWork E T x
      + ∑ k ∈ Finset.range T, (E (k + 1) (x (k + 1)) - E (k + 1) (x k))
      = E T (x T) - E 0 (x 0) := by
    unfold ch02_pathWork
    rw [← Finset.sum_add_distrib, ← htel]
    exact Finset.sum_congr rfl (fun k _ => by ring)
  have hβW : β * ch02_pathWork E T x
      = β * (E T (x T) - E 0 (x 0))
        - β * ∑ k ∈ Finset.range T, (E (k + 1) (x (k + 1)) - E (k + 1) (x k)) := by
    rw [← hsum]; ring
  rw [Real.log_div (ne_of_gt hQpos) (ne_of_gt hPpos), hlogQ, hlogP]
  unfold ch02_freeEnergy
  have hkey : β * (ch02_pathWork E T x
        - (-Real.log (ch02_protocolZ β E T) / β - -Real.log (ch02_protocolZ β E 0) / β))
      = β * ch02_pathWork E T x + Real.log (ch02_protocolZ β E T)
        - Real.log (ch02_protocolZ β E 0) := by
    simp only [div_eq_mul_inv]
    have hbb : β * β⁻¹ = 1 := mul_inv_cancel₀ hβ
    linear_combination
      (Real.log (ch02_protocolZ β E T) - Real.log (ch02_protocolZ β E 0)) * hbb
  rw [hkey]
  linear_combination hβW - hdiff

-- the path space is finite: a path is u : Fin (T+1) → σ, read as a function ℕ → σ
-- by clamping the index at T (only indices 0,…,T occur in the path laws).
def ch02_extend {σ : Type*} {T : ℕ} (u : Fin (T + 1) → σ) : ℕ → σ :=
  fun k => u ⟨min k T, Nat.lt_succ_of_le (min_le_right k T)⟩

theorem ch02_extend_zero {σ : Type*} {T : ℕ} (u : Fin (T + 1) → σ) :
    ch02_extend u 0 = u 0 := by
  unfold ch02_extend
  congr 1
  all_goals (apply Fin.ext; simp)

theorem ch02_extend_last {σ : Type*} {T : ℕ} (u : Fin (T + 1) → σ) :
    ch02_extend u T = u (Fin.last T) := by
  unfold ch02_extend
  congr 1
  apply Fin.ext
  simp

theorem ch02_extend_castSucc {σ : Type*} {T : ℕ} (u : Fin (T + 1) → σ) (k : Fin T) :
    ch02_extend u (k : ℕ) = u k.castSucc := by
  have hk : (k : ℕ) < T := k.isLt
  unfold ch02_extend
  congr 1
  apply Fin.ext
  simp only [Fin.val_castSucc]
  omega

theorem ch02_extend_succ {σ : Type*} {T : ℕ} (u : Fin (T + 1) → σ) (k : Fin T) :
    ch02_extend u ((k : ℕ) + 1) = u k.succ := by
  have hk : (k : ℕ) < T := k.isLt
  unfold ch02_extend
  congr 1
  apply Fin.ext
  simp only [Fin.val_succ]
  omega

theorem ch02_reversePathLaw_extend {σ : Type*} [Fintype σ] (β : ℝ) (E : ℕ → σ → ℝ)
    (K : ℕ → σ → σ → ℝ) (T : ℕ) (u : Fin (T + 1) → σ) :
    ch02_reversePathLaw β E K T (ch02_extend u)
      = Real.exp (-(β * E T (u (Fin.last T)))) / ch02_protocolZ β E T
        * ∏ k : Fin T, K ((k : ℕ) + 1) (u k.succ) (u k.castSucc) := by
  unfold ch02_reversePathLaw
  rw [ch02_extend_last]
  congr 1
  rw [← Fin.prod_univ_eq_prod_range
    (fun k => K (k + 1) (ch02_extend u (k + 1)) (ch02_extend u k)) T]
  exact Finset.prod_congr rfl (fun k _ => by
    rw [ch02_extend_succ, ch02_extend_castSucc])

-- a backward chain measure built from a normalised terminal law and row-stochastic
-- kernels is a probability measure on the path space.  Induction on the number of
-- steps, peeling the first state off with Fin.cons.
-- splitting a path into its first state and the rest
def ch02_consEquiv (σ : Type*) (n : ℕ) : (Fin (n + 2) → σ) ≃ σ × (Fin (n + 1) → σ) where
  toFun u := (u 0, Fin.tail u)
  invFun p := Fin.cons p.1 p.2
  left_inv u := Fin.cons_self_tail u
  right_inv p := by simp [Fin.tail_cons]

theorem ch02_chain_sum {σ : Type*} [Fintype σ] :
    ∀ (n : ℕ) (π : σ → ℝ) (K : ℕ → σ → σ → ℝ), (∑ y, π y = 1) →
      (∀ k y, ∑ z, K k y z = 1) →
      (∑ u : Fin (n + 1) → σ, π (u (Fin.last n))
        * ∏ k : Fin n, K ((k : ℕ) + 1) (u k.succ) (u k.castSucc)) = 1 := by
  intro n
  induction n with
  | zero =>
    intro π K hπ _
    rw [← Equiv.sum_comp (Equiv.funUnique (Fin 1) σ).symm]
    simpa using hπ
  | succ n ih =>
    intro π K hπ hK
    have hF : ∀ (a : σ) (y : Fin (n + 1) → σ),
        π ((Fin.cons a y : Fin (n + 2) → σ) (Fin.last (n + 1)))
          * ∏ k : Fin (n + 1), K ((k : ℕ) + 1)
              ((Fin.cons a y : Fin (n + 2) → σ) k.succ)
              ((Fin.cons a y : Fin (n + 2) → σ) k.castSucc)
        = (π (y (Fin.last n))
            * ∏ j : Fin n, K ((j : ℕ) + 1 + 1) (y j.succ) (y j.castSucc)) * K 1 (y 0) a := by
      intro a y
      have hlast : (Fin.cons a y : Fin (n + 2) → σ) (Fin.last (n + 1)) = y (Fin.last n) := by
        rw [← Fin.succ_last, Fin.cons_succ]
      rw [hlast, Fin.prod_univ_succ]
      simp only [Fin.val_zero, Fin.val_succ, Fin.castSucc_zero, Fin.cons_zero, Fin.cons_succ,
        ← Fin.succ_castSucc, zero_add]
      ring
    rw [← Equiv.sum_comp (ch02_consEquiv σ n).symm, Fintype.sum_prod_type, Finset.sum_comm]
    refine Eq.trans (Finset.sum_congr rfl (fun y _ => ?_))
      (ih π (fun k => K (k + 1)) hπ (fun k y => hK (k + 1) y))
    refine Eq.trans (Finset.sum_congr rfl (fun a _ => hF a y)) ?_
    rw [← Finset.mul_sum, hK 1 (y 0), mul_one]

theorem ch02_reversePathLaw_sum_one {σ : Type*} [Fintype σ] [Nonempty σ] (β : ℝ)
    (E : ℕ → σ → ℝ) (K : ℕ → σ → σ → ℝ) (T : ℕ)
    (hKrow : ∀ k y, ∑ z, K k y z = 1) :
    ∑ u : Fin (T + 1) → σ, ch02_reversePathLaw β E K T (ch02_extend u) = 1 := by
  have hZ : (0:ℝ) < ∑ y, Real.exp (-(β * E T y)) :=
    Finset.sum_pos (fun y _ => Real.exp_pos _) Finset.univ_nonempty
  have hπ : ∑ y, Real.exp (-(β * E T y)) / ch02_protocolZ β E T = 1 := by
    unfold ch02_protocolZ
    rw [← Finset.sum_div]
    exact div_self (ne_of_gt hZ)
  rw [Finset.sum_congr rfl (fun u _ => ch02_reversePathLaw_extend β E K T u)]
  exact ch02_chain_sum T (fun y => Real.exp (-(β * E T y)) / ch02_protocolZ β E T) K hπ hKrow

-- eq:sm-jarzynski-physical: E_Q[e^{-β(W-ΔF)}] = 1, proved for the discrete driven
-- model with no hypotheses beyond positivity, row-stochasticity and local detailed
-- balance of the relaxation kernels.
theorem ch02_sm_jarzynski_physical_db {σ : Type*} [Fintype σ] [Nonempty σ]
    (β : ℝ) (hβ : β ≠ 0) (E : ℕ → σ → ℝ) (K : ℕ → σ → σ → ℝ) (T : ℕ)
    (hKpos : ∀ k y z, 0 < K k y z) (hKrow : ∀ k y, ∑ z, K k y z = 1)
    (hDB : ∀ k y z, Real.exp (-(β * E k y)) * K k y z
      = Real.exp (-(β * E k z)) * K k z y) :
    ∑ u : Fin (T + 1) → σ, ch02_forwardPathLaw β E K T (ch02_extend u)
        * Real.exp (-(β * (ch02_pathWork E T (ch02_extend u)
            - (ch02_freeEnergy β E T - ch02_freeEnergy β E 0)))) = 1 := by
  have key : ∀ u : Fin (T + 1) → σ,
      ch02_forwardPathLaw β E K T (ch02_extend u)
          * Real.exp (-(β * (ch02_pathWork E T (ch02_extend u)
              - (ch02_freeEnergy β E T - ch02_freeEnergy β E 0))))
        = ch02_reversePathLaw β E K T (ch02_extend u) := by
    intro u
    have hQpos := ch02_forwardPathLaw_pos β E K T (ch02_extend u) hKpos
    have hPpos := ch02_reversePathLaw_pos β E K T (ch02_extend u) hKpos
    rw [ch02_sm_crooks_pathwise β hβ E K T (ch02_extend u) hKpos hDB, ← Real.log_inv,
      inv_div, Real.exp_log (div_pos hPpos hQpos), mul_comm,
      div_mul_cancel₀ _ (ne_of_gt hQpos)]
  rw [Finset.sum_congr rfl (fun u _ => key u)]
  exact ch02_reversePathLaw_sum_one β E K T hKrow

-- Non-vacuity: the hypotheses above are satisfiable for EVERY protocol, so the
-- theorem is not an empty statement about an unrealisable model.  The Gibbs
-- independence sampler K_k(y,·) = e^{-βE_k}/Z_k qualifies for any β and any
-- energies, and gives an unconditional instance of the Jarzynski equality.
noncomputable def ch02_gibbsKernel {σ : Type*} [Fintype σ] (β : ℝ) (E : ℕ → σ → ℝ)
    (k : ℕ) (_y z : σ) : ℝ :=
  Real.exp (-(β * E k z)) / ch02_protocolZ β E k

theorem ch02_gibbsKernel_pos {σ : Type*} [Fintype σ] [Nonempty σ] (β : ℝ) (E : ℕ → σ → ℝ)
    (k : ℕ) (y z : σ) : 0 < ch02_gibbsKernel β E k y z :=
  div_pos (Real.exp_pos _) (ch02_protocolZ_pos β E k)

theorem ch02_gibbsKernel_row {σ : Type*} [Fintype σ] [Nonempty σ] (β : ℝ) (E : ℕ → σ → ℝ)
    (k : ℕ) (y : σ) : ∑ z, ch02_gibbsKernel β E k y z = 1 := by
  have hZ : (0:ℝ) < ∑ z, Real.exp (-(β * E k z)) :=
    Finset.sum_pos (fun z _ => Real.exp_pos _) Finset.univ_nonempty
  unfold ch02_gibbsKernel ch02_protocolZ
  rw [← Finset.sum_div]
  exact div_self (ne_of_gt hZ)

theorem ch02_gibbsKernel_db {σ : Type*} [Fintype σ] (β : ℝ) (E : ℕ → σ → ℝ)
    (k : ℕ) (y z : σ) :
    Real.exp (-(β * E k y)) * ch02_gibbsKernel β E k y z
      = Real.exp (-(β * E k z)) * ch02_gibbsKernel β E k z y := by
  unfold ch02_gibbsKernel
  ring

theorem ch02_sm_jarzynski_gibbs {σ : Type*} [Fintype σ] [Nonempty σ] (β : ℝ) (hβ : β ≠ 0)
    (E : ℕ → σ → ℝ) (T : ℕ) :
    ∑ u : Fin (T + 1) → σ,
        ch02_forwardPathLaw β E (ch02_gibbsKernel β E) T (ch02_extend u)
        * Real.exp (-(β * (ch02_pathWork E T (ch02_extend u)
            - (ch02_freeEnergy β E T - ch02_freeEnergy β E 0)))) = 1 :=
  ch02_sm_jarzynski_physical_db β hβ E (ch02_gibbsKernel β E) T
    (fun k y z => ch02_gibbsKernel_pos β E k y z)
    (fun k y => ch02_gibbsKernel_row β E k y)
    (fun k y z => ch02_gibbsKernel_db β E k y z)


/-! #### eq:sm-forward (lines 215-218): the terminal law of the forward process

The display itself is a stochastic differential equation, and it is not
formalised: this Mathlib has Brownian motion but no stochastic integral, so
there is no solution concept in which to state it.  What the surrounding text
asserts *about* it -- that the process is "started at X_0 ~ p_0, the data law,
and producing marginals p_t that approach a simple terminal measure" -- is an
algebraic claim about the channel, and for the choice f(x,t) = -x, g = sqrt 2
used throughout this thesis it is proved here, in the generality of the audit's
encoding: for an ARBITRARY finitely supported data law p_0 (any number of atoms,
at arbitrary locations, with arbitrary weights summing to one) the time-t
marginal of eq:marg converges at every x, as t -> infinity, to the standard
Gaussian density.  Nothing is fixed at a numerical instance. -/

theorem ch02_exp_neg_two_tendsto :
    Filter.Tendsto (fun t : ℝ => Real.exp (-2 * t)) Filter.atTop (nhds 0) := by
  have he : Filter.Tendsto (fun t : ℝ => Real.exp (-t)) Filter.atTop (nhds 0) :=
    Real.tendsto_exp_neg_atTop_nhds_zero
  have h2 : Filter.Tendsto (fun t : ℝ => Real.exp (-t) * Real.exp (-t)) Filter.atTop
      (nhds 0) := by simpa using he.mul he
  have hfun : (fun t : ℝ => Real.exp (-t) * Real.exp (-t))
      = fun t : ℝ => Real.exp (-2 * t) := by
    funext t
    rw [← Real.exp_add]
    congr 1
    ring
  rwa [hfun] at h2

-- Δ_t = 1 - e^{-2t} → 1: the channel's variance saturates.
theorem ch02_Delta_tendsto_one : Filter.Tendsto ch02_Delta Filter.atTop (nhds 1) := by
  have hc : Filter.Tendsto (fun _ : ℝ => (1:ℝ)) Filter.atTop (nhds 1) := tendsto_const_nhds
  have h := hc.sub ch02_exp_neg_two_tendsto
  have hfun : ch02_Delta = fun t : ℝ => 1 - Real.exp (-2 * t) := rfl
  rw [hfun]
  simpa using h

-- the Gaussian kernel is jointly continuous in (variance, mean) wherever the
-- variance stays away from 0; stated as the limit needed below.
theorem ch02_gauss_tendsto {α : Type*} {l : Filter α} {v m : α → ℝ} (x : ℝ)
    (hv : Filter.Tendsto v l (nhds 1)) (hm : Filter.Tendsto m l (nhds 0)) :
    Filter.Tendsto (fun s => ch02_gauss (v s) x (m s)) l (nhds (ch02_gauss 1 x 0)) := by
  have hc2 : Filter.Tendsto (fun _ : α => (2:ℝ)) l (nhds 2) := tendsto_const_nhds
  have hcx : Filter.Tendsto (fun _ : α => x) l (nhds x) := tendsto_const_nhds
  have hcp : Filter.Tendsto (fun _ : α => 2 * Real.pi) l (nhds (2 * Real.pi)) :=
    tendsto_const_nhds
  have hden : Filter.Tendsto (fun s => 2 * v s) l (nhds (2 * 1)) := hc2.mul hv
  have hnum : Filter.Tendsto (fun s => -(x - m s) ^ 2) l (nhds (-(x - 0) ^ 2)) :=
    ((hcx.sub hm).pow 2).neg
  have hfrac : Filter.Tendsto (fun s => -(x - m s) ^ 2 / (2 * v s)) l
      (nhds (-(x - 0) ^ 2 / (2 * 1))) := hnum.div hden (by norm_num)
  have hexp : Filter.Tendsto (fun s => Real.exp (-(x - m s) ^ 2 / (2 * v s))) l
      (nhds (Real.exp (-(x - 0) ^ 2 / (2 * 1)))) :=
    (Real.continuous_exp.tendsto _).comp hfrac
  have hsq : Filter.Tendsto (fun s => Real.sqrt (2 * Real.pi * v s)) l
      (nhds (Real.sqrt (2 * Real.pi * 1))) := (hcp.mul hv).sqrt
  have hpos : (0:ℝ) < 2 * Real.pi * 1 := by positivity
  have hne : Real.sqrt (2 * Real.pi * 1) ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hpos)
  have hg1 : ch02_gauss 1 x 0
      = Real.exp (-(x - 0) ^ 2 / (2 * 1)) / Real.sqrt (2 * Real.pi * 1) := rfl
  have hgf : (fun s => ch02_gauss (v s) x (m s))
      = fun s => Real.exp (-(x - m s) ^ 2 / (2 * v s))
          / Real.sqrt (2 * Real.pi * v s) := rfl
  rw [hg1, hgf]
  exact hexp.div hsq hne

-- eq:sm-forward: the marginals of the forward process approach the standard
-- Gaussian, for an arbitrary finitely supported data law.
theorem ch02_sm_forward_terminal {N : ℕ} (lam a : Fin N → ℝ) (hlam : ∑ i, lam i = 1)
    (x : ℝ) :
    Filter.Tendsto (fun t : ℝ => ch02_marg lam a (ch02_Delta t) (Real.exp (-t)) x)
      Filter.atTop (nhds (ch02_gauss 1 x 0)) := by
  have hmean : ∀ i : Fin N,
      Filter.Tendsto (fun t : ℝ => Real.exp (-t) * a i) Filter.atTop (nhds 0) := by
    intro i
    simpa using Real.tendsto_exp_neg_atTop_nhds_zero.mul_const (a i)
  have hterm : ∀ i : Fin N, Filter.Tendsto
      (fun t : ℝ => lam i * ch02_gauss (ch02_Delta t) x (Real.exp (-t) * a i))
      Filter.atTop (nhds (lam i * ch02_gauss 1 x 0)) := fun i =>
    (ch02_gauss_tendsto x ch02_Delta_tendsto_one (hmean i)).const_mul (lam i)
  have hval : ∑ i : Fin N, lam i * ch02_gauss 1 x 0 = ch02_gauss 1 x 0 := by
    rw [← Finset.sum_mul, hlam, one_mul]
  have hfun : (fun t : ℝ => ch02_marg lam a (ch02_Delta t) (Real.exp (-t)) x)
      = fun t : ℝ => ∑ i, lam i * ch02_gauss (ch02_Delta t) x (Real.exp (-t) * a i) := rfl
  have hsum := tendsto_finsetSum (Finset.univ : Finset (Fin N)) (fun i _ => hterm i)
  rw [hval] at hsum
  rw [hfun]
  exact hsum

/-! #### eq:sm-reverse (lines 606-619): time reversal at the level of densities

The previous check, ch02_sm_reverse, compared the two Fokker-Planck right-hand
sides at a fixed density.  This is the statement it was standing in for: if the
forward marginal p_t solves the Fokker-Planck equation of eq:sm-fokkerplanck with
drift f(.,t) and diffusion g(t)^2, then the REVERSED density q_tau := p_{T-tau}
solves the Fokker-Planck equation of the reverse SDE of eq:sm-reverse, whose
drift in forward time is -(f - g^2 d_x log p).  So the boxed drift -- with the
full g^2, not the probability flow's g^2/2 -- is exactly the one that transports
the marginals backwards.  Arbitrary time-dependent f and g, arbitrary positive
p; one dimension, where the divergence acts coordinatewise.  What remains
out of reach is the pathwise statement (that the time-reversed PROCESS is a
diffusion with this drift), which needs a stochastic integral. -/

theorem ch02_sm_reverse_time_reversal (p : ℝ → ℝ → ℝ) (f : ℝ → ℝ → ℝ) (g2 : ℝ → ℝ)
    (T τ x dtp : ℝ)
    (hp : ∀ y, 0 < p (T - τ) y) (hd : Differentiable ℝ (p (T - τ)))
    (hfp : DifferentiableAt ℝ (fun y => f (T - τ) y * p (T - τ) y) x)
    (hp' : DifferentiableAt ℝ (deriv (p (T - τ))) x)
    (hdt : HasDerivAt (fun s => p s x) dtp (T - τ))
    (hfwd : dtp = -deriv (fun y => f (T - τ) y * p (T - τ) y) x
        + g2 (T - τ) / 2 * deriv (deriv (p (T - τ))) x) :
    HasDerivAt (fun s => p (T - s) x)
      (-deriv (fun y => (-(f (T - τ) y)
            + g2 (T - τ) * deriv (fun z => Real.log (p (T - τ) z)) y) * p (T - τ) y) x
        + g2 (T - τ) / 2 * deriv (deriv (p (T - τ))) x) τ := by
  have hlin : HasDerivAt (fun s : ℝ => T - s) (-1) τ := (hasDerivAt_id τ).const_sub T
  have hchain : HasDerivAt (fun s => p (T - s) x) (-dtp) τ := by
    have h := hdt.comp τ hlin
    simpa [Function.comp_def] using h
  have hrev := ch02_sm_reverse (p (T - τ)) (f (T - τ)) (g2 (T - τ)) x hp hd hfp hp'
  rw [hrev, ← hfwd]
  exact hchain

/-! ### Pass 4 (final gap-closing pass): the unlabelled displays, and the Itô boundary

Two things happen in this section.

(1) Every UNLABELLED display of `ch02-statmech-diffusion.tex` is itemised and given a
verdict.  The chapter has 56 top-level amsmath displays (39 `equation`, 13 `equation*`,
4 `align`; plus one nested `aligned`, which is how the orchestrator's count of 57 arises).
42 carry a `\label`; the remaining 14 are numbered `noname-1` ... `noname-14` here, in
order of appearance:

  noname-1   257-261  probability current of the Langevin dynamics, J = f p - β⁻¹∇p
  noname-2   265-271  that current vanishes pointwise at p ∝ e^{-βE}
  noname-3   286-292  boxed: the OU transition law (a restatement of eq:sm-ou)
  noname-4   303-307  Z = ∫ e^{-‖x‖²/2} = (2π)^{d/2}, F = -(d/2) log 2π
  noname-5   544-551  Itô's lemma, dφ(X) = ∇φ·dX + ½∑ ∂ᵢ∂ⱼφ d⟨Xⁱ,Xʲ⟩
  noname-6   557-571  Itô expansion with isotropic noise (drift/martingale split)
  noname-7   575-583  expectation of the integrated Itô expansion
  noname-8   610-616  d/dt E[φ(X_t)] = d/dt ∫ φ p_t = ∫ φ ∂_t p_t
  noname-9   634-640  product rule for a divergence, ∇·[φ f p] = ∇φ·(f p) + φ ∇·[f p]
  noname-10  645-649  integration by parts, drift term
  noname-11  654-658  integration by parts, ∫ Δφ p = -∫ ∇φ·∇p
  noname-12  661-665  integration by parts, -∫ ∇φ·∇p = ∫ φ Δp
  noname-13  672-682  a continuous function orthogonal to every test function vanishes
  noname-14  714-721  Δp = ∇·(p ∇log p)   (recorded by an earlier pass as `noname-1`;
                      its 1-D theorem `ch02_noname_1` keeps that name for stability)

The 9 remaining displays of the chapter are `\[ ... \]` blocks inside `toolbox`
environments; they are intermediate lines of derivations whose conclusions are the
labelled displays, and they are not amsmath environments, so they fall outside the
orchestrator's enumeration.  They are listed in the JSON note for completeness.

(2) The Itô boundary is stated precisely.  This Mathlib (toolchain
`leanprover/lean4:v4.33.0`) contains `Mathlib/Probability/BrownianMotion/Basic.lean`,
but a search of the whole library for `stochasticIntegral` / `itoIntegral` returns
nothing: there is no stochastic integral, hence no Itô formula, no "an Itô integral is
a martingale so its expectation vanishes" step, and no solution concept for
`dX = f dt + g dW`.  noname-5, noname-6, noname-7 and eq:sm-generator are exactly the
steps that need those, and they stay out of reach.  What does NOT need them --
Steps 3 to 6 of toolbox tb:sm-fokkerplanck, i.e. noname-8 and noname-9 to noname-13 --
is proved below, and the resulting implication eq:sm-weakform ⇒ eq:sm-fokkerplanck is
`ch02_sm_weakform_implies_fokkerplanck`. -/

open scoped ContDiff

/-! #### noname-1 (tex 257-261): the probability current of the Langevin dynamics

`J_t = f p_t - β⁻¹∇p_t = -∇E p_t - β⁻¹∇p_t`.  The display is the substitution
`f = -∇E` into the definition of the current; that is all it asserts. -/

theorem ch02_noname_1_current {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]
    (f gradE gradp : F → F) (p : F → ℝ) (β : ℝ) (x : F) (hf : f x = -gradE x) :
    p x • f x - β⁻¹ • gradp x = -(p x • gradE x) - β⁻¹ • gradp x := by
  rw [hf, smul_neg]

/-! #### noname-2 (tex 265-271): the current vanishes pointwise at the Boltzmann law

`J_∞ = -∇E p_∞ - β⁻¹(-β ∇E p_∞) = 0` at every `x`.  The chain-rule input
`∇p_∞ = -β ∇E p_∞` is `ch02_noname_2_chainrule`; the cancellation is
`ch02_noname_2_current_vanishes`.  (The same fact in the arrangement of the toolbox's
prose is `ch02_sm_langevin_flux_nd`.)  Arbitrary real normed space, arbitrary
differentiable `E`, arbitrary `β ≠ 0`. -/

theorem ch02_noname_2_chainrule {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]
    (E : F → ℝ) (β : ℝ) (hE : Differentiable ℝ E) (y v : F) :
    fderiv ℝ (fun z => Real.exp (-(β * E z))) y v
      = -(β * fderiv ℝ E y v) * Real.exp (-(β * E y)) := by
  have h1 : HasFDerivAt E (fderiv ℝ E y) y := (hE y).hasFDerivAt
  have h2 : HasFDerivAt (fun z => -(β * E z)) (-(β • fderiv ℝ E y)) y := (h1.const_mul β).neg
  have h3 : HasFDerivAt (fun z => Real.exp (-(β * E z)))
      (Real.exp (-(β * E y)) • (-(β • fderiv ℝ E y))) y := h2.exp
  rw [h3.fderiv]
  have happ : (Real.exp (-(β * E y)) • (-(β • fderiv ℝ E y))) v
      = Real.exp (-(β * E y)) * -(β * fderiv ℝ E y v) := by simp
  rw [happ]; ring

theorem ch02_noname_2_current_vanishes (β A G : ℝ) (hβ : β ≠ 0) :
    -(A * G) - β⁻¹ * (-(β * A) * G) = 0 := by
  have hb : β⁻¹ * β = 1 := inv_mul_cancel₀ hβ
  have hrw : β⁻¹ * (-(β * A) * G) = -((β⁻¹ * β) * (A * G)) := by ring
  rw [hrw, hb]; ring

/-! #### noname-3 (tex 286-292): the boxed OU transition law, and why it is coordinatewise

The boxed display repeats eq:sm-ou, so its status is eq:sm-ou's.  The one thing the
box adds is the reason it may be read coordinatewise -- "this energy separates across
coordinates and the drift is diagonal, so the coordinates evolve independently and the
scalar integrating-factor solution applies to each one".  Following the audit's rule
for independence, that is recorded as the density identity: the d-dimensional isotropic
Gaussian channel IS the product of the d scalar channels, at arbitrary `d`. -/

theorem ch02_noname_3_factorises {d : ℕ} (v : ℝ) (x m : Fin d → ℝ) :
    ∏ k, ch02_gauss v (x k) (m k)
      = Real.exp (-(∑ k, (x k - m k) ^ 2) / (2 * v)) / Real.sqrt (2 * Real.pi * v) ^ d := by
  have hsum : ∑ k, -(x k - m k) ^ 2 / (2 * v)
      = -(∑ k, (x k - m k) ^ 2) / (2 * v) := by
    rw [← Finset.sum_div, ← Finset.sum_neg_distrib]
  unfold ch02_gauss
  rw [Finset.prod_div_distrib, Finset.prod_const, Finset.card_univ, Fintype.card_fin,
    ← Real.exp_sum, hsum]

/-! #### noname-4 (tex 303-307): the partition function of the terminal equilibrium

`Z = ∫_{ℝ^d} e^{-‖x‖²/2} dx = (2π)^{d/2}` and `F = -β⁻¹ log Z = -(d/2) log 2π` at
`β = 1`.  Proved for every `d`, from `integral_gaussian` and Fubini on a product
measure; `ℝ^d` is `Fin d → ℝ` with the product Lebesgue measure and `‖x‖² = ∑ (x k)²`. -/

theorem ch02_noname_4_partition (d : ℕ) :
    ∫ x : Fin d → ℝ, Real.exp (-(∑ k, (x k) ^ 2) / 2) = Real.sqrt (2 * Real.pi) ^ d := by
  have hprod : (fun x : Fin d → ℝ => Real.exp (-(∑ k, (x k) ^ 2) / 2))
      = fun x : Fin d → ℝ => ∏ k, Real.exp (-(1 / 2 : ℝ) * (x k) ^ 2) := by
    funext x
    rw [← Real.exp_sum]
    congr 1
    rw [← Finset.mul_sum]
    ring
  have hpi : Real.pi / (1 / 2 : ℝ) = 2 * Real.pi := by ring
  have key : ∫ x : Fin d → ℝ, ∏ k, Real.exp (-(1 / 2 : ℝ) * (x k) ^ 2)
      = (∫ y : ℝ, Real.exp (-(1 / 2 : ℝ) * y ^ 2)) ^ (Fintype.card (Fin d)) :=
    MeasureTheory.integral_fintype_prod_volume_eq_pow (ι := Fin d)
      (fun y : ℝ => Real.exp (-(1 / 2 : ℝ) * y ^ 2))
  rw [hprod, key, Fintype.card_fin, integral_gaussian, hpi]

theorem ch02_noname_4_partition_rpow (d : ℕ) :
    Real.sqrt (2 * Real.pi) ^ d = (2 * Real.pi) ^ ((d : ℝ) / 2) := by
  have h2 : (0:ℝ) ≤ 2 * Real.pi := by positivity
  rw [show ((d : ℝ) / 2) = (1 / 2 : ℝ) * (d : ℝ) by ring, Real.rpow_mul h2,
    Real.rpow_natCast, ← Real.sqrt_eq_rpow]

theorem ch02_noname_4_freeEnergy (d : ℕ) :
    -Real.log (Real.sqrt (2 * Real.pi) ^ d) = -(d / 2 : ℝ) * Real.log (2 * Real.pi) := by
  have h2 : (0:ℝ) ≤ 2 * Real.pi := by positivity
  rw [Real.log_pow, Real.log_sqrt h2]
  ring

/-! #### noname-5 (tex 544-551), noname-6 (557-571), noname-7 (575-583): Itô

These three displays are Itô's lemma, its specialisation to isotropic noise, and the
expectation of its integrated form.  They are NOT formalised, and the obstruction is
not difficulty but absence: this Mathlib has no stochastic integral, so `dW_t`, the
quadratic variation `d⟨Xⁱ,Xʲ⟩_t` and "the Itô integral is a martingale started at zero"
have no statements to be proved.  See the section header.

One piece of noname-6 is pure algebra and is recorded: given
`d⟨Xⁱ,Xʲ⟩ = g² δᵢⱼ dt`, the double sum `½ ∑_{i,j} ∂ᵢ∂ⱼφ d⟨Xⁱ,Xʲ⟩` collapses onto the
Laplacian `½ g² ∑_i ∂ᵢ∂ᵢφ`.  That is the only step of the three which does not need a
stochastic integral. -/

theorem ch02_noname_6_isotropic_collapse {d : ℕ} (H : Fin d → Fin d → ℝ) (g2 : ℝ) :
    (1 / 2 : ℝ) * ∑ i, ∑ j, H i j * (g2 * (if i = j then (1:ℝ) else 0))
      = g2 / 2 * ∑ i, H i i := by
  have hinner : ∀ i : Fin d,
      ∑ j, H i j * (g2 * (if i = j then (1:ℝ) else 0)) = g2 * H i i := by
    intro i
    rw [Finset.sum_eq_single i (fun j _ hj => by rw [if_neg (fun h => hj h.symm)]; ring)
      (fun h => absurd (Finset.mem_univ i) h), if_pos rfl]
    ring
  rw [Finset.sum_congr rfl (fun i _ => hinner i), ← Finset.mul_sum]
  ring

/-! #### noname-8 (tex 610-616): the time derivative passes onto the density

`d/dt E[φ(X_t)] = d/dt ∫ φ p_t = ∫ φ ∂_t p_t`, justified in the text by dominated
convergence with `φ` bounded of compact support.  Proved as an instance of Mathlib's
dominated differentiation-under-the-integral theorem, with exactly those hypotheses
made explicit: `p` is the family of densities, `pt` its time derivative. -/

theorem ch02_noname_8_dt_under_integral
    (φ : ℝ → ℝ) (p pt : ℝ → ℝ → ℝ) (bd : ℝ → ℝ) {t₀ : ℝ} {S : Set ℝ}
    (hS : S ∈ nhds t₀)
    (hmeas : ∀ᶠ t in nhds t₀, AEStronglyMeasurable (fun x => φ x * p t x) volume)
    (hint : Integrable (fun x => φ x * p t₀ x) volume)
    (hmeas' : AEStronglyMeasurable (fun x => φ x * pt t₀ x) volume)
    (hbound : ∀ᵐ x ∂(volume : Measure ℝ), ∀ t ∈ S, ‖φ x * pt t x‖ ≤ bd x)
    (hbint : Integrable bd volume)
    (hdiff : ∀ᵐ x ∂(volume : Measure ℝ), ∀ t ∈ S,
      HasDerivAt (fun τ => φ x * p τ x) (φ x * pt t x) t) :
    HasDerivAt (fun τ => ∫ x, φ x * p τ x) (∫ x, φ x * pt t₀ x) t₀ :=
  (hasDerivAt_integral_of_dominated_loc_of_deriv_le (F := fun τ x => φ x * p τ x)
    (F' := fun τ x => φ x * pt τ x) (bound := bd) hS hmeas hint hmeas' hbound hbint hdiff).2

/-! #### noname-9 (tex 634-640) and noname-14 (tex 714-721) in ℝ^d

Partial derivatives, divergence and Laplacian on `Fin d → ℝ`, so that the two displays
that are genuinely `d`-dimensional vector calculus can be stated at arbitrary `d`
rather than in one dimension. -/

noncomputable def ch02_partial {d : ℕ} (F : (Fin d → ℝ) → ℝ) (k : Fin d) (x : Fin d → ℝ) : ℝ :=
  fderiv ℝ F x (Pi.single k 1)

noncomputable def ch02_div {d : ℕ} (V : (Fin d → ℝ) → Fin d → ℝ) (x : Fin d → ℝ) : ℝ :=
  ∑ k, ch02_partial (fun y => V y k) k x

noncomputable def ch02_laplacian {d : ℕ} (p : (Fin d → ℝ) → ℝ) (x : Fin d → ℝ) : ℝ :=
  ∑ k, ch02_partial (ch02_partial p k) k x

-- linearity and the Leibniz rule for a partial derivative, used below.
theorem ch02_partial_mul {d : ℕ} (F G : (Fin d → ℝ) → ℝ) (k : Fin d) (x : Fin d → ℝ)
    (hF : DifferentiableAt ℝ F x) (hG : DifferentiableAt ℝ G x) :
    ch02_partial (fun y => F y * G y) k x
      = ch02_partial F k x * G x + F x * ch02_partial G k x := by
  simp only [ch02_partial]
  rw [fderiv_fun_mul hF hG, ContinuousLinearMap.add_apply,
    ContinuousLinearMap.smul_apply, ContinuousLinearMap.smul_apply, smul_eq_mul, smul_eq_mul]
  ring

theorem ch02_partial_sub {d : ℕ} (F G : (Fin d → ℝ) → ℝ) (k : Fin d) (x : Fin d → ℝ)
    (hF : DifferentiableAt ℝ F x) (hG : DifferentiableAt ℝ G x) :
    ch02_partial (fun y => F y - G y) k x = ch02_partial F k x - ch02_partial G k x := by
  simp only [ch02_partial]
  rw [fderiv_fun_sub hF hG, ContinuousLinearMap.sub_apply]

theorem ch02_partial_const_mul {d : ℕ} (F : (Fin d → ℝ) → ℝ) (c : ℝ) (k : Fin d)
    (x : Fin d → ℝ) (hF : DifferentiableAt ℝ F x) :
    ch02_partial (fun y => c * F y) k x = c * ch02_partial F k x := by
  simp only [ch02_partial]
  rw [fderiv_const_mul hF c, ContinuousLinearMap.smul_apply, smul_eq_mul]

-- noname-9: ∇·[φ f p] = ∇φ·(f p) + φ ∇·[f p], in ℝ^d, for any differentiable φ and V.
theorem ch02_noname_9_div_product {d : ℕ} (φ : (Fin d → ℝ) → ℝ)
    (V : (Fin d → ℝ) → Fin d → ℝ) (x : Fin d → ℝ)
    (hφ : DifferentiableAt ℝ φ x) (hV : ∀ k, DifferentiableAt ℝ (fun y => V y k) x) :
    ch02_div (fun y k => φ y * V y k) x
      = (∑ k, ch02_partial φ k x * V x k) + φ x * ch02_div V x := by
  simp only [ch02_div]
  rw [Finset.mul_sum, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun k _ => ?_
  exact ch02_partial_mul φ (fun y => V y k) k x hφ (hV k)

-- noname-14, the field identity: p ∂ₖ log p = ∂ₖ p, in ℝ^d.
theorem ch02_noname_14_score_field {d : ℕ} (p : (Fin d → ℝ) → ℝ) (hp : ∀ y, 0 < p y)
    (hd : Differentiable ℝ p) (k : Fin d) :
    (fun y => p y * ch02_partial (fun z => Real.log (p z)) k y) = ch02_partial p k := by
  funext y
  have hne : p y ≠ 0 := ne_of_gt (hp y)
  have h1 : HasFDerivAt p (fderiv ℝ p y) y := (hd y).hasFDerivAt
  have h2 : HasFDerivAt (fun z => Real.log (p z)) ((p y)⁻¹ • fderiv ℝ p y) y := h1.log hne
  simp only [ch02_partial]
  rw [h2.fderiv, ContinuousLinearMap.smul_apply, smul_eq_mul]
  field_simp

-- noname-14 (tex 714-721): Δp = ∇·(p ∇log p), in ℝ^d at arbitrary d.
theorem ch02_noname_14 {d : ℕ} (p : (Fin d → ℝ) → ℝ) (hp : ∀ y, 0 < p y)
    (hd : Differentiable ℝ p) (x : Fin d → ℝ) :
    ch02_laplacian p x
      = ch02_div (fun y k => p y * ch02_partial (fun z => Real.log (p z)) k y) x := by
  simp only [ch02_laplacian, ch02_div]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [ch02_noname_14_score_field p hp hd k]

/-! #### noname-10, noname-11, noname-12 (tex 645-649, 654-658, 661-665)

The three integrations by parts of Steps 4 and 5 of toolbox tb:sm-fokkerplanck.  All
three are the same statement -- `∫ u' v = -∫ u v'` on the whole line, the boundary
terms vanishing -- which Mathlib proves from integrability of `u·v` alone; compact
support of `φ`, which is what the text spends here, is one way to supply that. -/

theorem ch02_ibp_real {u v u' v' : ℝ → ℝ}
    (hu : ∀ x ∈ tsupport v, HasDerivAt u (u' x) x)
    (hv : ∀ x ∈ tsupport u, HasDerivAt v (v' x) x)
    (huv' : Integrable (u * v')) (hu'v : Integrable (u' * v)) (huv : Integrable (u * v)) :
    ∫ x, u' x * v x = -∫ x, u x * v' x := by
  rw [MeasureTheory.integral_mul_deriv_eq_deriv_mul_of_integrable hu hv huv' hu'v huv, neg_neg]

-- noname-10: ∫ f·∇φ p = -∫ φ ∇·(f p).  Here `fp` is the field f·p and `fp'` its divergence.
theorem ch02_noname_10_ibp_drift {φ φ' fp fp' : ℝ → ℝ}
    (hφ : ∀ x ∈ tsupport fp, HasDerivAt φ (φ' x) x)
    (hfp : ∀ x ∈ tsupport φ, HasDerivAt fp (fp' x) x)
    (h1 : Integrable (φ * fp')) (h2 : Integrable (φ' * fp)) (h3 : Integrable (φ * fp)) :
    ∫ x, φ' x * fp x = -∫ x, φ x * fp' x :=
  ch02_ibp_real hφ hfp h1 h2 h3

-- noname-11: ∫ Δφ p = -∫ ∇φ·∇p.
theorem ch02_noname_11_ibp_lap1 {φ' φ'' p p' : ℝ → ℝ}
    (hφ : ∀ x ∈ tsupport p, HasDerivAt φ' (φ'' x) x)
    (hp : ∀ x ∈ tsupport φ', HasDerivAt p (p' x) x)
    (h1 : Integrable (φ' * p')) (h2 : Integrable (φ'' * p)) (h3 : Integrable (φ' * p)) :
    ∫ x, φ'' x * p x = -∫ x, φ' x * p' x :=
  ch02_ibp_real hφ hp h1 h2 h3

-- noname-12: -∫ ∇φ·∇p = ∫ φ Δp.
theorem ch02_noname_12_ibp_lap2 {φ φ' p' p'' : ℝ → ℝ}
    (hφ : ∀ x ∈ tsupport p', HasDerivAt φ (φ' x) x)
    (hp : ∀ x ∈ tsupport φ, HasDerivAt p' (p'' x) x)
    (h1 : Integrable (φ * p'')) (h2 : Integrable (φ' * p')) (h3 : Integrable (φ * p')) :
    -∫ x, φ' x * p' x = ∫ x, φ x * p'' x := by
  rw [ch02_ibp_real hφ hp h1 h2 h3, neg_neg]

/-! #### noname-13 (tex 672-682): a continuous function orthogonal to every test function

"A continuous function orthogonal to every test function is identically zero: if the
bracket were nonzero at some x₀ it would keep its sign on a ball around x₀, and a bump
function supported inside that ball would make the integral nonzero."  This is the
fundamental lemma of the calculus of variations; Mathlib proves the almost-everywhere
version (`ae_eq_zero_of_integral_contDiff_smul_eq_zero`, by exactly the bump-function
argument the text gives), and continuity upgrades it to everywhere. -/

theorem ch02_noname_13 (h : ℝ → ℝ) (hc : Continuous h)
    (hzero : ∀ φ : ℝ → ℝ, ContDiff ℝ ∞ φ → HasCompactSupport φ → ∫ x, φ x * h x = 0) :
    h = 0 := by
  have hae : ∀ᵐ x ∂(volume : Measure ℝ), h x = 0 :=
    ae_eq_zero_of_integral_contDiff_smul_eq_zero hc.locallyIntegrable
      (fun g hg hgc => by simpa using hzero g hg hgc)
  exact (hc.ae_eq_iff_eq volume continuous_const).mp hae

/-! #### eq:sm-weakform (tex 619-626): what can be proved without a stochastic integral

The display itself asserts the weak identity for the marginals of the SDE, and getting
there is Steps 1-3 of the toolbox: Itô's lemma, the vanishing of the Itô integral's
expectation, and the generator identity eq:sm-generator.  None of those can be stated
here (no stochastic integral).  What CAN be proved is the rest of the toolbox --
Steps 4, 5 and 6 -- namely that the weak form, together with the two integrations by
parts (noname-10 to noname-12) and the fundamental lemma (noname-13), gives the
Fokker-Planck equation eq:sm-fokkerplanck.  That is the theorem below.  It is general
in f, in g and in the density; it is one-dimensional, where the divergence and the
Laplacian act coordinatewise. -/

theorem ch02_sm_weakform_implies_fokkerplanck
    (p q fp dfp d2p : ℝ → ℝ) (g2 : ℝ)
    (hbr : Continuous (fun x => q x + dfp x - g2 / 2 * d2p x))
    (hweak : ∀ φ : ℝ → ℝ, ContDiff ℝ ∞ φ → HasCompactSupport φ →
        ∫ x, φ x * q x
          = (∫ x, deriv φ x * fp x) + g2 / 2 * ∫ x, deriv (deriv φ) x * p x)
    (hibp1 : ∀ φ : ℝ → ℝ, ContDiff ℝ ∞ φ → HasCompactSupport φ →
        ∫ x, deriv φ x * fp x = -∫ x, φ x * dfp x)
    (hibp2 : ∀ φ : ℝ → ℝ, ContDiff ℝ ∞ φ → HasCompactSupport φ →
        ∫ x, deriv (deriv φ) x * p x = ∫ x, φ x * d2p x)
    (hint1 : ∀ φ : ℝ → ℝ, ContDiff ℝ ∞ φ → HasCompactSupport φ →
        Integrable (fun x => φ x * q x))
    (hint2 : ∀ φ : ℝ → ℝ, ContDiff ℝ ∞ φ → HasCompactSupport φ →
        Integrable (fun x => φ x * dfp x))
    (hint3 : ∀ φ : ℝ → ℝ, ContDiff ℝ ∞ φ → HasCompactSupport φ →
        Integrable (fun x => φ x * d2p x)) :
    ∀ x, q x = -dfp x + g2 / 2 * d2p x := by
  have hzero : ∀ φ : ℝ → ℝ, ContDiff ℝ ∞ φ → HasCompactSupport φ →
      ∫ x, φ x * (q x + dfp x - g2 / 2 * d2p x) = 0 := by
    intro φ hφ hφc
    have e1 : (fun x => φ x * (q x + dfp x - g2 / 2 * d2p x))
        = fun x => (φ x * q x + φ x * dfp x) - g2 / 2 * (φ x * d2p x) := by
      funext x; ring
    have hsplit : ∫ x, φ x * (q x + dfp x - g2 / 2 * d2p x)
        = (∫ x, φ x * q x) + (∫ x, φ x * dfp x) - g2 / 2 * ∫ x, φ x * d2p x := by
      have hA : Integrable (fun x => φ x * q x + φ x * dfp x) volume :=
        (hint1 φ hφ hφc).add (hint2 φ hφ hφc)
      have hB : Integrable (fun x => g2 / 2 * (φ x * d2p x)) volume :=
        (hint3 φ hφ hφc).const_mul (g2 / 2)
      rw [e1, MeasureTheory.integral_sub hA hB,
        MeasureTheory.integral_add (hint1 φ hφ hφc) (hint2 φ hφ hφc),
        MeasureTheory.integral_const_mul]
    rw [hsplit, hweak φ hφ hφc, hibp1 φ hφ hφc, hibp2 φ hφ hφc]
    ring
  have hbrz := ch02_noname_13 _ hbr hzero
  intro x
  have hx := congrFun hbrz x
  simp only [Pi.zero_apply] at hx
  linarith

/-! #### eq:sm-fokkerplanck and eq:sm-ou: from one instance to the whole linear-Gaussian class

Pass 3 checked the Fokker-Planck equation on the single Ornstein-Uhlenbeck channel
`f(x,t) = -x`, `g = √2`.  Here the check is made general in the class the thesis
actually uses: an arbitrary time-dependent linear drift `f(x,t) = -θ(t)x` and an
arbitrary diffusion schedule `g(t)`.  The claim proved is that a Gaussian density with
mean `m` and variance `v` solves eq:sm-fokkerplanck exactly when `m` and `v` satisfy
the moment ODEs `m' = -θ m` and `v' = -2θ v + g²`.  Ornstein-Uhlenbeck (θ ≡ 1, g ≡ √2),
the variance-preserving schedule of eq:sm-ddpm-marginal (θ = β(t)/2, g² = β(t)) and the
variance-exploding schedule (θ ≡ 0) are all instances; the OU one is spelled out below
as `ch02_fokkerplanck_ou_of_general`, which recovers `ch02_sm_ou_fokkerplanck`. -/

noncomputable def ch02_lgLogDensity (m v : ℝ → ℝ) (x t : ℝ) : ℝ :=
  -(x - m t) ^ 2 / (2 * v t) - Real.log (2 * Real.pi * v t) / 2

noncomputable def ch02_lgDensity (m v : ℝ → ℝ) (x t : ℝ) : ℝ :=
  Real.exp (ch02_lgLogDensity m v x t)

theorem ch02_lgDensity_eq_gauss (m v : ℝ → ℝ) (x t : ℝ) (hv : 0 < v t) :
    ch02_lgDensity m v x t = ch02_gauss (v t) x (m t) := by
  have hz : (0:ℝ) < 2 * Real.pi * v t := by positivity
  have hsqrt : Real.sqrt (2 * Real.pi * v t)
      = Real.exp (Real.log (2 * Real.pi * v t) / 2) := by
    rw [← Real.log_sqrt hz.le, Real.exp_log (Real.sqrt_pos.mpr hz)]
  unfold ch02_lgDensity ch02_lgLogDensity ch02_gauss
  rw [hsqrt, Real.exp_sub]

theorem ch02_lgLogDensity_deriv (m v : ℝ → ℝ) (x : ℝ) {t m' v' : ℝ} (hv : 0 < v t)
    (hm : HasDerivAt m m' t) (hvd : HasDerivAt v v' t) :
    HasDerivAt (fun s => ch02_lgLogDensity m v x s)
      ((x - m t) * m' / v t + (x - m t) ^ 2 * v' / (2 * v t ^ 2) - v' / (2 * v t)) t := by
  have hvne : v t ≠ 0 := ne_of_gt hv
  have hd : HasDerivAt (fun s : ℝ => x - m s) (-m') t := hm.const_sub x
  have hsq : HasDerivAt (fun s : ℝ => (x - m s) ^ 2) (2 * (x - m t) * -m') t := by
    simpa using hd.fun_pow 2
  have hnum : HasDerivAt (fun s : ℝ => -(x - m s) ^ 2) (-(2 * (x - m t) * -m')) t := hsq.neg
  have hden : HasDerivAt (fun s : ℝ => 2 * v s) (2 * v') t := hvd.const_mul 2
  have hdenne : (2:ℝ) * v t ≠ 0 := mul_ne_zero two_ne_zero hvne
  have hlogarg : HasDerivAt (fun s : ℝ => 2 * Real.pi * v s) (2 * Real.pi * v') t :=
    hvd.const_mul (2 * Real.pi)
  have hlogargne : 2 * Real.pi * v t ≠ 0 := by positivity
  have hpine : Real.pi ≠ 0 := Real.pi_ne_zero
  have hfinal : HasDerivAt (fun s : ℝ => -(x - m s) ^ 2 / (2 * v s)
        - Real.log (2 * Real.pi * v s) / 2)
      ((-(2 * (x - m t) * -m') * (2 * v t) - -(x - m t) ^ 2 * (2 * v')) / (2 * v t) ^ 2
        - 2 * Real.pi * v' / (2 * Real.pi * v t) / 2) t :=
    (hnum.div hden hdenne).sub ((hlogarg.log hlogargne).div_const 2)
  have heq : ((-(2 * (x - m t) * -m') * (2 * v t) - -(x - m t) ^ 2 * (2 * v')) / (2 * v t) ^ 2
        - 2 * Real.pi * v' / (2 * Real.pi * v t) / 2)
      = (x - m t) * m' / v t + (x - m t) ^ 2 * v' / (2 * v t ^ 2) - v' / (2 * v t) := by
    field_simp
    ring
  rw [heq] at hfinal
  unfold ch02_lgLogDensity
  exact hfinal

-- eq:sm-fokkerplanck for the whole linear-Gaussian class: ∂_t p = -∂ₓ(f p) + ½g²∂ₓ²p
-- with f(x,t) = -θ(t)x, for any θ and g and any (m, v) solving the moment ODEs.
theorem ch02_fokkerplanck_linear_gaussian (m v θ g : ℝ → ℝ) (x : ℝ) {t : ℝ}
    (hv : 0 < v t)
    (hm : HasDerivAt m (-(θ t) * m t) t)
    (hvd : HasDerivAt v (-(2 * θ t * v t) + g t ^ 2) t) :
    HasDerivAt (fun s => ch02_lgDensity m v x s)
      (-deriv (fun y => (-(θ t) * y) * ch02_lgDensity m v y t) x
        + g t ^ 2 / 2 * deriv (deriv (fun y => ch02_lgDensity m v y t)) x) t := by
  have hvne : v t ≠ 0 := ne_of_gt hv
  have hfun : (fun y => ch02_lgDensity m v y t) = fun y => ch02_gauss (v t) y (m t) := by
    funext y; exact ch02_lgDensity_eq_gauss m v y t hv
  have hd1 : deriv (fun y => ch02_lgDensity m v y t)
      = fun y => -((y - m t) / v t) * ch02_gauss (v t) y (m t) := by
    rw [hfun]; funext y; exact (ch02_kernelgrad hv y _).deriv
  have hd2 : deriv (deriv (fun y => ch02_lgDensity m v y t)) x
      = (-(1 / v t) + ((x - m t) / v t) ^ 2) * ch02_gauss (v t) x (m t) := by
    rw [hd1]; exact (ch02_gauss_deriv2 hv x _).deriv
  have hd3 : deriv (fun y => (-(θ t) * y) * ch02_lgDensity m v y t) x
      = -(θ t) * 1 * ch02_gauss (v t) x (m t)
        + -(θ t) * x * (-((x - m t) / v t) * ch02_gauss (v t) x (m t)) := by
    have hxfun : (fun y => (-(θ t) * y) * ch02_lgDensity m v y t)
        = fun y => (-(θ t) * y) * ch02_gauss (v t) y (m t) := by
      funext y; rw [ch02_lgDensity_eq_gauss m v y t hv]
    rw [hxfun]
    exact (((hasDerivAt_id x).const_mul (-(θ t))).mul (ch02_kernelgrad hv x _)).deriv
  rw [hd2, hd3]
  have hlogd := ch02_lgLogDensity_deriv m v x hv hm hvd
  have hexp : HasDerivAt (fun s => ch02_lgDensity m v x s) _ t := hlogd.exp
  convert hexp using 1
  have hG : Real.exp (ch02_lgLogDensity m v x t) = ch02_gauss (v t) x (m t) :=
    ch02_lgDensity_eq_gauss m v x t hv
  rw [hG]
  field_simp
  ring

-- the Ornstein-Uhlenbeck channel of eq:sm-ou is the instance θ ≡ 1, g ≡ √2.
theorem ch02_fokkerplanck_ou_of_general (a x : ℝ) {t : ℝ} (ht : 0 < t) :
    HasDerivAt (fun s => ch02_lgDensity (fun r => Real.exp (-r) * a) ch02_Delta x s)
      (-deriv (fun y => (-(1:ℝ) * y)
            * ch02_lgDensity (fun r => Real.exp (-r) * a) ch02_Delta y t) x
        + Real.sqrt 2 ^ 2 / 2
          * deriv (deriv (fun y =>
              ch02_lgDensity (fun r => Real.exp (-r) * a) ch02_Delta y t)) x) t := by
  have hΔ : 0 < ch02_Delta t := ch02_sm_ou_Delta_pos ht
  have hs2 : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  have hm : HasDerivAt (fun r : ℝ => Real.exp (-r) * a) (-(1:ℝ) * (Real.exp (-t) * a)) t := by
    have h : HasDerivAt (fun r : ℝ => -r) (-1) t := (hasDerivAt_id t).neg
    have h2 : HasDerivAt (fun r : ℝ => Real.exp (-r)) (Real.exp (-t) * -1) t := h.exp
    have h3 := h2.mul_const a
    have heq : Real.exp (-t) * -1 * a = -(1:ℝ) * (Real.exp (-t) * a) := by ring
    rwa [heq] at h3
  have hvd : HasDerivAt ch02_Delta (-(2 * 1 * ch02_Delta t) + Real.sqrt 2 ^ 2) t := by
    have h := ch02_sm_ou_var_ode t
    have heq : (2 : ℝ) - 2 * ch02_Delta t = -(2 * 1 * ch02_Delta t) + Real.sqrt 2 ^ 2 := by
      rw [hs2]; ring
    rwa [heq] at h
  exact ch02_fokkerplanck_linear_gaussian _ _ (fun _ => (1:ℝ)) (fun _ => Real.sqrt 2) x hΔ hm hvd

/-! #### eq:sm-reverse (tex 773-786) in arbitrary dimension

Pass 3 proved the density-level reversal in one dimension.  With the divergence of
noname-9 available it is now proved in ℝ^d: the Fokker-Planck right-hand side of the
reverse drift `-(f - g²∇log p)` evaluated at the same density `p` is MINUS the forward
Fokker-Planck right-hand side, at arbitrary d, arbitrary vector field f, arbitrary
positive differentiable p.  What is still missing is unchanged and is not about
dimension: the pathwise statement (Anderson's theorem) needs a stochastic integral. -/

theorem ch02_sm_reverse_nd {d : ℕ} (p : (Fin d → ℝ) → ℝ) (f : (Fin d → ℝ) → Fin d → ℝ)
    (g2 : ℝ) (x : Fin d → ℝ)
    (hp : ∀ y, 0 < p y) (hd : Differentiable ℝ p)
    (hfld : ∀ k, DifferentiableAt ℝ (fun y => f y k * p y) x)
    (hpk : ∀ k, DifferentiableAt ℝ (ch02_partial p k) x) :
    -ch02_div (fun y k =>
        (-(f y k) + g2 * ch02_partial (fun z => Real.log (p z)) k y) * p y) x
      + g2 / 2 * ch02_laplacian p x
    = -(-ch02_div (fun y k => f y k * p y) x + g2 / 2 * ch02_laplacian p x) := by
  have hfield : ∀ k : Fin d,
      (fun y => (-(f y k) + g2 * ch02_partial (fun z => Real.log (p z)) k y) * p y)
        = fun y => g2 * ch02_partial p k y - f y k * p y := by
    intro k; funext y
    have hy := congrFun (ch02_noname_14_score_field p hp hd k) y
    calc (-(f y k) + g2 * ch02_partial (fun z => Real.log (p z)) k y) * p y
        = g2 * (p y * ch02_partial (fun z => Real.log (p z)) k y) - f y k * p y := by ring
      _ = g2 * ch02_partial p k y - f y k * p y := by rw [hy]
  have hdiv : ch02_div (fun y k =>
        (-(f y k) + g2 * ch02_partial (fun z => Real.log (p z)) k y) * p y) x
      = g2 * ch02_laplacian p x - ch02_div (fun y k => f y k * p y) x := by
    simp only [ch02_div, ch02_laplacian]
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [hfield k, ch02_partial_sub _ _ k x ((hpk k).const_mul g2) (hfld k),
      ch02_partial_const_mul _ g2 k x (hpk k)]
  rw [hdiv]
  ring

/-! #### eq:sm-dsm (tex 1082-1089): closing the two prose steps

The fidelity review recorded that two steps of the chapter's claim about eq:sm-dsm --
"the minimiser of eq:sm-dsm over all measurable functions is the exact score of the
noised population law" -- were prose only: (i) the reduction of minimisation over
functions to pointwise minimisation, and (ii) the identification of the pointwise
minimiser with the score of the module's own marginal `ch02_marg`.  Both are proved
here.  `ch02_sm_dsm_tower` is (i) in the finite setting.  `ch02_sm_dsm_optimum_is_score`
is (ii), and it names `ch02_sm_dsm`, `ch02_posterior`, `ch02_Delta` and `ch02_marg`
directly: at a fixed noisy observation `x`, the conditional DSM loss of the finite-prior
channel is minimised exactly at the value `sstar x t` which is, simultaneously, the
derivative of `log ch02_marg` at `x` -- the exact score. -/

theorem ch02_sm_dsm_tower {X : Type*} [Fintype X] (qw : X → ℝ) (hq : ∀ x, 0 ≤ qw x)
    (L : X → ℝ → ℝ) (sstar s : X → ℝ) (hmin : ∀ x, L x (sstar x) ≤ L x (s x)) :
    ∑ x, qw x * L x (sstar x) ≤ ∑ x, qw x * L x (s x) :=
  Finset.sum_le_sum fun x _ => mul_le_mul_of_nonneg_left (hmin x) (hq x)

theorem ch02_sm_dsm_optimum_is_score {N : ℕ} (hN : 0 < N) (lam a : Fin N → ℝ) {t : ℝ}
    (ht : 0 < t) (hlam : ∀ i, 0 < lam i) (x : ℝ) (sstar : ℝ → ℝ → ℝ)
    (hstar : sstar x t
      = (Real.exp (-t) * ch02_postmean lam a (ch02_Delta t) (Real.exp (-t)) x - x)
          / ch02_Delta t) :
    HasDerivAt (fun y => Real.log (ch02_marg lam a (ch02_Delta t) (Real.exp (-t)) y))
        (sstar x t) x
    ∧ ∀ s : ℝ → ℝ → ℝ,
        ∑ i, ch02_posterior lam a (ch02_Delta t) (Real.exp (-t)) x i
            * ch02_sm_dsm sstar t (a i)
                ((x - Real.exp (-t) * a i) / Real.sqrt (ch02_Delta t))
          ≤ ∑ i, ch02_posterior lam a (ch02_Delta t) (Real.exp (-t)) x i
            * ch02_sm_dsm s t (a i)
                ((x - Real.exp (-t) * a i) / Real.sqrt (ch02_Delta t)) := by
  have hΔ : 0 < ch02_Delta t := ch02_sm_ou_Delta_pos ht
  have hsq : 0 < Real.sqrt (ch02_Delta t) := Real.sqrt_pos.mpr hΔ
  have hsqne : Real.sqrt (ch02_Delta t) ≠ 0 := ne_of_gt hsq
  have hsqsq : Real.sqrt (ch02_Delta t) * Real.sqrt (ch02_Delta t) = ch02_Delta t :=
    Real.mul_self_sqrt hΔ.le
  -- the channel relation: every atom is carried to the SAME observation x
  have hchan : ∀ i : Fin N, Real.exp (-t) * a i
      + Real.sqrt (ch02_Delta t)
        * ((x - Real.exp (-t) * a i) / Real.sqrt (ch02_Delta t)) = x := by
    intro i
    field_simp
    ring
  have hrescale : ∀ i : Fin N,
      ((x - Real.exp (-t) * a i) / Real.sqrt (ch02_Delta t)) / Real.sqrt (ch02_Delta t)
        = (x - Real.exp (-t) * a i) / ch02_Delta t := by
    intro i; rw [div_div, hsqsq]
  have hloss : ∀ (u : ℝ → ℝ → ℝ) (i : Fin N),
      ch02_sm_dsm u t (a i) ((x - Real.exp (-t) * a i) / Real.sqrt (ch02_Delta t))
        = (u x t + (x - Real.exp (-t) * a i) / ch02_Delta t) ^ 2 := by
    intro u i
    unfold ch02_sm_dsm
    rw [hchan i, hrescale i]
  have hw1 : ∑ i, ch02_posterior lam a (ch02_Delta t) (Real.exp (-t)) x i = 1 :=
    ch02_posterior_sum_one hN lam a hΔ hlam _ x
  have hzbar : ∑ i, ch02_posterior lam a (ch02_Delta t) (Real.exp (-t)) x i
      * ((x - Real.exp (-t) * a i) / ch02_Delta t) = -(sstar x t) := by
    have hlin := ch02_noiseform_linearity hN lam a hΔ hlam (Real.exp (-t)) x
    have hstep : ∑ i, ch02_posterior lam a (ch02_Delta t) (Real.exp (-t)) x i
          * ((x - Real.exp (-t) * a i) / ch02_Delta t)
        = (∑ i, ch02_posterior lam a (ch02_Delta t) (Real.exp (-t)) x i
              * ((x - Real.exp (-t) * a i) / Real.sqrt (ch02_Delta t)))
            / Real.sqrt (ch02_Delta t) := by
      rw [Finset.sum_div]
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [← hrescale i]; ring
    rw [hstep, hlin, hstar, div_div, hsqsq]
    ring
  constructor
  · rw [hstar]
    exact (ch02_tweedie hN lam a hΔ hlam t x).1
  · intro s
    have hmin := ch02_sm_dsm_min
      (fun i => ch02_posterior lam a (ch02_Delta t) (Real.exp (-t)) x i)
      (fun i => (x - Real.exp (-t) * a i) / ch02_Delta t) hw1 (s x t)
    rw [hzbar, neg_neg] at hmin
    have e1 : ∀ u : ℝ → ℝ → ℝ,
        ∑ i, ch02_posterior lam a (ch02_Delta t) (Real.exp (-t)) x i
            * ch02_sm_dsm u t (a i)
                ((x - Real.exp (-t) * a i) / Real.sqrt (ch02_Delta t))
          = ∑ i, ch02_posterior lam a (ch02_Delta t) (Real.exp (-t)) x i
            * (u x t + (x - Real.exp (-t) * a i) / ch02_Delta t) ^ 2 :=
      fun u => Finset.sum_congr rfl fun i _ => by rw [hloss u i]
    rw [e1 sstar, e1 s]
    exact hmin


/-! ### Pass 5 (the dimension pass): toolbox tb:sm-fokkerplanck on ℝ^d

The chapter states this toolbox on `ℝ^d` -- tex 538-539, "Throughout, `X_t ∈ ℝ^d`",
and tex 641, "Integrate over `ℝ^d`" -- while Pass 4 proved Steps 4 to 6 on the line.
This pass lifts them to `ℝ^d`, realised as `Fin d → ℝ` with the product Lebesgue
measure: the same ambient space `ch02_partial`, `ch02_div`, `ch02_laplacian`,
`ch02_noname_3_factorises` and `ch02_noname_4_partition` already use.

The route is the text's own.  Each `∂ₖ` integration by parts is the scalar identity on
the line through `x` in direction `k`, and Fubini reassembles the slices; the
`d`-dimensional displays are then sums of that over the `d` coordinates.  The
`ℝ`-integration-by-parts input is `ch02_ibp_real` of Pass 4, i.e. Mathlib's
`integral_mul_deriv_eq_deriv_mul_of_integrable`. -/

/-- Inserting the scalar `t` into coordinate `k` of a fixed `(d-1)`-tuple is an affine map
of `t`, with derivative the `k`-th basis vector. -/
theorem ch02_insertNth_hasDerivAt {n : ℕ} (k : Fin (n + 1)) (y : Fin n → ℝ) (t : ℝ) :
    HasDerivAt (fun s : ℝ => (k.insertNth s y : Fin (n + 1) → ℝ))
      (Pi.single k (1 : ℝ)) t := by
  rw [hasDerivAt_pi]
  intro i
  rcases eq_or_ne i k with rfl | hik
  · simp only [Fin.insertNth_apply_same, Pi.single_eq_same]
    exact hasDerivAt_id' (𝕜 := ℝ) (x := t)
  · obtain ⟨j, rfl⟩ := Fin.exists_succAbove_eq hik
    simp only [Fin.insertNth_apply_succAbove, Pi.single_eq_of_ne (Fin.succAbove_ne k j)]
    exact hasDerivAt_const t (y j)

/-- The partial derivative `∂ₖF` is the ordinary derivative of `F` along the `k`-th
coordinate line. -/
theorem ch02_partial_slice {n : ℕ} (k : Fin (n + 1)) {F : (Fin (n + 1) → ℝ) → ℝ}
    (hF : Differentiable ℝ F) (y : Fin n → ℝ) (t : ℝ) :
    HasDerivAt (fun s : ℝ => F (k.insertNth s y))
      (ch02_partial F k (k.insertNth t y)) t :=
  ((hF (k.insertNth t y)).hasFDerivAt).comp_hasDerivAt t (ch02_insertNth_hasDerivAt k y t)

/-- A compactly supported function restricts to a compactly supported function on every
coordinate line: this is where the text's "φ vanishes outside a bounded set" is spent. -/
theorem ch02_slice_hasCompactSupport {n : ℕ} {u : (Fin (n + 1) → ℝ) → ℝ}
    (hcs : HasCompactSupport u) (k : Fin (n + 1)) (y : Fin n → ℝ) :
    HasCompactSupport (fun t : ℝ => u (k.insertNth t y)) := by
  have hcpt : IsCompact (tsupport u) := hcs
  obtain ⟨R, hR⟩ := hcpt.isBounded.subset_closedBall (0 : Fin (n + 1) → ℝ)
  refine HasCompactSupport.intro (K := Set.Icc (-R) R) isCompact_Icc ?_
  intro t ht
  by_contra hne
  have hsupp : (k.insertNth t y : Fin (n + 1) → ℝ) ∈ tsupport u :=
    subset_closure (by simpa [Function.mem_support] using hne)
  have hball : ‖(k.insertNth t y : Fin (n + 1) → ℝ)‖ ≤ R := by
    simpa using hR hsupp
  have hcoord : |t| ≤ ‖(k.insertNth t y : Fin (n + 1) → ℝ)‖ := by
    simpa using norm_le_pi_norm (k.insertNth t y : Fin (n + 1) → ℝ) k
  exact ht (Set.mem_Icc.mpr (abs_le.mp (le_trans hcoord hball)))

/-- Fubini along the `k`-th coordinate: an integral over `ℝ^d` is the integral over the
remaining `d-1` coordinates of the integral along the `k`-th coordinate line. -/
theorem ch02_fubini_slice {n : ℕ} (k : Fin (n + 1)) {F : (Fin (n + 1) → ℝ) → ℝ}
    (hF : Integrable F) :
    ∫ x : Fin (n + 1) → ℝ, F x = ∫ y : Fin n → ℝ, ∫ t : ℝ, F (k.insertNth t y) := by
  have hmp : MeasurePreserving
      (MeasurableEquiv.piFinSuccAbove (fun _ : Fin (n + 1) => ℝ) k).symm
      (volume : Measure (ℝ × (Fin n → ℝ))) (volume : Measure (Fin (n + 1) → ℝ)) :=
    (volume_preserving_piFinSuccAbove (fun _ : Fin (n + 1) => ℝ) k).symm
  have hcomp : Integrable
      (fun z : ℝ × (Fin n → ℝ) =>
        F ((MeasurableEquiv.piFinSuccAbove (fun _ : Fin (n + 1) => ℝ) k).symm z))
      (volume : Measure (ℝ × (Fin n → ℝ))) :=
    (hmp.integrable_comp_emb (MeasurableEquiv.measurableEmbedding _)).mpr hF
  have h1 : ∫ z : ℝ × (Fin n → ℝ),
      F ((MeasurableEquiv.piFinSuccAbove (fun _ : Fin (n + 1) => ℝ) k).symm z)
        = ∫ x : Fin (n + 1) → ℝ, F x := hmp.integral_comp' F
  rw [← h1]
  rw [MeasureTheory.Measure.volume_eq_prod] at hcomp ⊢
  rw [MeasureTheory.integral_prod_symm _ hcomp]
  rfl


/-- The coordinate line `t ↦ (…, t, …)` is continuous. -/
theorem ch02_insertNth_continuous {n : ℕ} (k : Fin (n + 1)) (y : Fin n → ℝ) :
    Continuous (fun s : ℝ => (k.insertNth s y : Fin (n + 1) → ℝ)) :=
  continuous_iff_continuousAt.mpr fun s => (ch02_insertNth_hasDerivAt k y s).continuousAt

/-- **Integration by parts on `ℝ^d` in one coordinate**: `∫ (∂ₖu) v = -∫ u (∂ₖv)`.

`u` plays the role of the toolbox's test function -- `C¹` with compact support, which is
what makes the boundary term vanish -- and `v` is an arbitrary `C¹` function.  The proof
is Fubini along the `k`-th coordinate (`ch02_fubini_slice`) plus the scalar integration
by parts `ch02_ibp_real` on each coordinate line. -/
theorem ch02_ibp_nd_coord {n : ℕ} (k : Fin (n + 1)) {u v : (Fin (n + 1) → ℝ) → ℝ}
    (hu : ContDiff ℝ 1 u) (hv : ContDiff ℝ 1 v) (hucs : HasCompactSupport u) :
    ∫ x : Fin (n + 1) → ℝ, ch02_partial u k x * v x
      = -∫ x : Fin (n + 1) → ℝ, u x * ch02_partial v k x := by
  have hud : Differentiable ℝ u := hu.differentiable (by norm_num)
  have hvd : Differentiable ℝ v := hv.differentiable (by norm_num)
  have hcu : Continuous (ch02_partial u k) := by
    show Continuous fun x => fderiv ℝ u x (Pi.single k (1 : ℝ))
    exact (hu.continuous_fderiv (by norm_num)).clm_apply continuous_const
  have hcv : Continuous (ch02_partial v k) := by
    show Continuous fun x => fderiv ℝ v x (Pi.single k (1 : ℝ))
    exact (hv.continuous_fderiv (by norm_num)).clm_apply continuous_const
  have hcsu : HasCompactSupport (ch02_partial u k) :=
    hucs.fderiv_apply ℝ (Pi.single k (1 : ℝ))
  have hA : Integrable (fun x : Fin (n + 1) → ℝ => ch02_partial u k x * v x) :=
    (hcu.mul hv.continuous).integrable_of_hasCompactSupport hcsu.mul_right
  have hB : Integrable (fun x : Fin (n + 1) → ℝ => u x * ch02_partial v k x) :=
    (hu.continuous.mul hcv).integrable_of_hasCompactSupport hucs.mul_right
  rw [ch02_fubini_slice k hA, ch02_fubini_slice k hB, ← MeasureTheory.integral_neg]
  refine MeasureTheory.integral_congr_ae (Filter.Eventually.of_forall fun y => ?_)
  have hins := ch02_insertNth_continuous k y
  have hsu : HasCompactSupport (fun t : ℝ => u (k.insertNth t y)) :=
    ch02_slice_hasCompactSupport hucs k y
  have hspu : HasCompactSupport (fun t : ℝ => ch02_partial u k (k.insertNth t y)) :=
    ch02_slice_hasCompactSupport hcsu k y
  have h1 : Integrable (fun t : ℝ =>
      u (k.insertNth t y) * ch02_partial v k (k.insertNth t y)) :=
    ((hu.continuous.comp hins).mul (hcv.comp hins)).integrable_of_hasCompactSupport
      hsu.mul_right
  have h2 : Integrable (fun t : ℝ =>
      ch02_partial u k (k.insertNth t y) * v (k.insertNth t y)) :=
    ((hcu.comp hins).mul (hv.continuous.comp hins)).integrable_of_hasCompactSupport
      hspu.mul_right
  have h3 : Integrable (fun t : ℝ => u (k.insertNth t y) * v (k.insertNth t y)) :=
    ((hu.continuous.comp hins).mul (hv.continuous.comp hins)).integrable_of_hasCompactSupport
      hsu.mul_right
  exact ch02_ibp_real (fun t _ => ch02_partial_slice k hud y t)
    (fun t _ => ch02_partial_slice k hvd y t) h1 h2 h3

/-- `ch02_ibp_nd_coord` at an arbitrary dimension (for `d = 0` both sides are `0`). -/
theorem ch02_ibp_nd : ∀ {d : ℕ} (k : Fin d) {u v : (Fin d → ℝ) → ℝ},
    ContDiff ℝ 1 u → ContDiff ℝ 1 v → HasCompactSupport u →
    ∫ x : Fin d → ℝ, ch02_partial u k x * v x
      = -∫ x : Fin d → ℝ, u x * ch02_partial v k x := by
  rintro (_ | n) k u v hu hv hucs
  · exact k.elim0
  · exact ch02_ibp_nd_coord k hu hv hucs

/-! #### noname-13 (tex 672-682) on ℝ^d

The fundamental lemma of the calculus of variations is already available in Mathlib at
this generality: `ae_eq_zero_of_integral_contDiff_smul_eq_zero` is stated for an
arbitrary finite-dimensional real normed space, so the `ℝ^d` statement costs no more
than the one-dimensional one.  The test-function class is the text's own -- smooth
(`ContDiff ℝ ∞`) with compact support -- and continuity upgrades the almost-everywhere
conclusion to "at every point", which is what the text asserts. -/

theorem ch02_noname_13_nd {d : ℕ} (h : (Fin d → ℝ) → ℝ) (hc : Continuous h)
    (hzero : ∀ φ : (Fin d → ℝ) → ℝ, ContDiff ℝ ∞ φ → HasCompactSupport φ →
      ∫ x : Fin d → ℝ, φ x * h x = 0) :
    h = 0 := by
  have hae : ∀ᵐ x ∂(volume : Measure (Fin d → ℝ)), h x = 0 :=
    ae_eq_zero_of_integral_contDiff_smul_eq_zero hc.locallyIntegrable
      (fun g hg hgc => by simpa using hzero g hg hgc)
  exact (hc.ae_eq_iff_eq volume continuous_const).mp hae

/-! ### Pass 6 (the dimension pass, completed): Steps 4-6 of tb:sm-fokkerplanck on ℝ^d

The chapter states this toolbox on `ℝ^d` (tex 538-539, "Throughout, `X_t ∈ ℝ^d`";
tex 641, "Integrate over `ℝ^d`").  Pass 5 built the two `ℝ^d` ingredients --
coordinatewise integration by parts `ch02_ibp_nd` and the fundamental lemma
`ch02_noname_13_nd` -- but the four displays they serve (noname-10, noname-11,
noname-12 and the implication eq:sm-weakform ⇒ eq:sm-fokkerplanck) were still stated
on the line.  They are stated and proved here on `ℝ^d`, at arbitrary `d`, with the
divergence and Laplacian of `ch02_div` / `ch02_laplacian`.

Two things improve besides the dimension.  The `ℝ^d` integration-by-parts statements
carry the text's OWN hypotheses -- `φ` smooth with compact support, the density and
the field continuously differentiable -- rather than assuming the integrability side
conditions, and the `ℝ^d` weak-form theorem consequently DISCHARGES the three
integrations by parts and the three integrability hypotheses that the one-dimensional
`ch02_sm_weakform_implies_fokkerplanck` had to take as assumptions. -/

/-- `∂ₖF` is continuous when `F` is `C¹`. -/
theorem ch02_partial_continuous {d : ℕ} {F : (Fin d → ℝ) → ℝ} (hF : ContDiff ℝ 1 F)
    (k : Fin d) : Continuous (ch02_partial F k) := by
  show Continuous fun x => fderiv ℝ F x (Pi.single k (1 : ℝ))
  exact (hF.continuous_fderiv (by norm_num)).clm_apply continuous_const

/-- `∂ₖF` is `C¹` when `F` is `C²`. -/
theorem ch02_partial_contDiff {d : ℕ} {F : (Fin d → ℝ) → ℝ} (hF : ContDiff ℝ 2 F)
    (k : Fin d) : ContDiff ℝ 1 (ch02_partial F k) := by
  show ContDiff ℝ 1 fun x => fderiv ℝ F x (Pi.single k (1 : ℝ))
  exact (hF.fderiv_right (m := 1) (by norm_num)).clm_apply contDiff_const

/-- `∂ₖφ` inherits the compact support of `φ`: the text's "`φ` vanishes outside a
bounded set", which is what makes every boundary term below vanish. -/
theorem ch02_partial_hasCompactSupport {d : ℕ} {F : (Fin d → ℝ) → ℝ}
    (hcs : HasCompactSupport F) (k : Fin d) : HasCompactSupport (ch02_partial F k) :=
  hcs.fderiv_apply ℝ (Pi.single k (1 : ℝ))

theorem ch02_div_continuous {d : ℕ} {V : (Fin d → ℝ) → Fin d → ℝ}
    (hV : ∀ k, ContDiff ℝ 1 (fun y => V y k)) : Continuous (ch02_div V) := by
  show Continuous fun x => ∑ k, ch02_partial (fun y => V y k) k x
  exact continuous_finset_sum _ fun k _ => ch02_partial_continuous (hV k) k

theorem ch02_laplacian_continuous {d : ℕ} {p : (Fin d → ℝ) → ℝ} (hp : ContDiff ℝ 2 p) :
    Continuous (ch02_laplacian p) := by
  show Continuous fun x => ∑ k, ch02_partial (ch02_partial p k) k x
  exact continuous_finset_sum _ fun k _ =>
    ch02_partial_continuous (ch02_partial_contDiff hp k) k

/-- A continuous compactly supported factor times a continuous factor is integrable:
this is how every integral below is known to exist. -/
theorem ch02_integrable_testMul {d : ℕ} {u v : (Fin d → ℝ) → ℝ}
    (hu : Continuous u) (hcs : HasCompactSupport u) (hv : Continuous v) :
    Integrable (fun x => u x * v x) (volume : Measure (Fin d → ℝ)) :=
  (hu.mul hv).integrable_of_hasCompactSupport hcs.mul_right

/-- **noname-10 (tex 645-649) on `ℝ^d`**: `∫ f·∇φ p = -∫ φ ∇·[f p]`.

`fp` is the vector field `f p_t`.  Proved at arbitrary `d` from coordinatewise
integration by parts `ch02_ibp_nd`, summed over the `d` coordinates. -/
theorem ch02_noname_10_ibp_drift_nd {d : ℕ} {φ : (Fin d → ℝ) → ℝ}
    {fp : (Fin d → ℝ) → Fin d → ℝ}
    (hφ : ContDiff ℝ 1 φ) (hcs : HasCompactSupport φ)
    (hfp : ∀ k, ContDiff ℝ 1 (fun y => fp y k)) :
    ∫ x : Fin d → ℝ, ∑ k, ch02_partial φ k x * fp x k
      = -∫ x : Fin d → ℝ, φ x * ch02_div fp x := by
  have hA : ∀ k : Fin d,
      Integrable (fun x : Fin d → ℝ => ch02_partial φ k x * fp x k) :=
    fun k => ch02_integrable_testMul (ch02_partial_continuous hφ k)
      (ch02_partial_hasCompactSupport hcs k) (hfp k).continuous
  have hB : ∀ k : Fin d,
      Integrable (fun x : Fin d → ℝ => φ x * ch02_partial (fun y => fp y k) k x) :=
    fun k => ch02_integrable_testMul hφ.continuous hcs (ch02_partial_continuous (hfp k) k)
  have hrhs : (fun x : Fin d → ℝ => φ x * ch02_div fp x)
      = fun x => ∑ k, φ x * ch02_partial (fun y => fp y k) k x := by
    funext x; simp only [ch02_div, Finset.mul_sum]
  rw [integral_finsetSum _ (fun k _ => hA k), hrhs,
    integral_finsetSum _ (fun k _ => hB k), ← Finset.sum_neg_distrib]
  exact Finset.sum_congr rfl fun k _ => ch02_ibp_nd k hφ (hfp k) hcs

/-- **noname-11 (tex 654-658) on `ℝ^d`**: `∫ Δφ p = -∫ ∇φ·∇p`. -/
theorem ch02_noname_11_ibp_lap1_nd {d : ℕ} {φ p : (Fin d → ℝ) → ℝ}
    (hφ : ContDiff ℝ 2 φ) (hcs : HasCompactSupport φ) (hp : ContDiff ℝ 1 p) :
    ∫ x : Fin d → ℝ, ch02_laplacian φ x * p x
      = -∫ x : Fin d → ℝ, ∑ k, ch02_partial φ k x * ch02_partial p k x := by
  have hφ1 : ContDiff ℝ 1 φ := hφ.of_le (by norm_num)
  have hA : ∀ k : Fin d,
      Integrable (fun x : Fin d → ℝ => ch02_partial (ch02_partial φ k) k x * p x) :=
    fun k => ch02_integrable_testMul
      (ch02_partial_continuous (ch02_partial_contDiff hφ k) k)
      (ch02_partial_hasCompactSupport (ch02_partial_hasCompactSupport hcs k) k)
      hp.continuous
  have hB : ∀ k : Fin d,
      Integrable (fun x : Fin d → ℝ => ch02_partial φ k x * ch02_partial p k x) :=
    fun k => ch02_integrable_testMul (ch02_partial_continuous hφ1 k)
      (ch02_partial_hasCompactSupport hcs k) (ch02_partial_continuous hp k)
  have hlhs : (fun x : Fin d → ℝ => ch02_laplacian φ x * p x)
      = fun x => ∑ k, ch02_partial (ch02_partial φ k) k x * p x := by
    funext x; simp only [ch02_laplacian, Finset.sum_mul]
  rw [hlhs, integral_finsetSum _ (fun k _ => hA k),
    integral_finsetSum _ (fun k _ => hB k), ← Finset.sum_neg_distrib]
  exact Finset.sum_congr rfl fun k _ =>
    ch02_ibp_nd k (ch02_partial_contDiff hφ k) hp (ch02_partial_hasCompactSupport hcs k)

/-- **noname-12 (tex 661-665) on `ℝ^d`**: `-∫ ∇φ·∇p = ∫ φ Δp`. -/
theorem ch02_noname_12_ibp_lap2_nd {d : ℕ} {φ p : (Fin d → ℝ) → ℝ}
    (hφ : ContDiff ℝ 1 φ) (hcs : HasCompactSupport φ) (hp : ContDiff ℝ 2 p) :
    -∫ x : Fin d → ℝ, ∑ k, ch02_partial φ k x * ch02_partial p k x
      = ∫ x : Fin d → ℝ, φ x * ch02_laplacian p x := by
  have hp1 : ContDiff ℝ 1 p := hp.of_le (by norm_num)
  have hA : ∀ k : Fin d,
      Integrable (fun x : Fin d → ℝ => ch02_partial φ k x * ch02_partial p k x) :=
    fun k => ch02_integrable_testMul (ch02_partial_continuous hφ k)
      (ch02_partial_hasCompactSupport hcs k) (ch02_partial_continuous hp1 k)
  have hB : ∀ k : Fin d,
      Integrable (fun x : Fin d → ℝ => φ x * ch02_partial (ch02_partial p k) k x) :=
    fun k => ch02_integrable_testMul hφ.continuous hcs
      (ch02_partial_continuous (ch02_partial_contDiff hp k) k)
  have hrhs : (fun x : Fin d → ℝ => φ x * ch02_laplacian p x)
      = fun x => ∑ k, φ x * ch02_partial (ch02_partial p k) k x := by
    funext x; simp only [ch02_laplacian, Finset.mul_sum]
  rw [hrhs, integral_finsetSum _ (fun k _ => hA k),
    integral_finsetSum _ (fun k _ => hB k), ← Finset.sum_neg_distrib]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [ch02_ibp_nd k hφ (ch02_partial_contDiff hp k) hcs, neg_neg]

/-- Step 5's conclusion, on `ℝ^d`: "the Laplacian moves across unchanged, the two sign
flips cancelling, which is the statement that it is formally self-adjoint". -/
theorem ch02_laplacian_selfAdjoint_nd {d : ℕ} {φ p : (Fin d → ℝ) → ℝ}
    (hφ : ContDiff ℝ 2 φ) (hcs : HasCompactSupport φ) (hp : ContDiff ℝ 2 p) :
    ∫ x : Fin d → ℝ, ch02_laplacian φ x * p x
      = ∫ x : Fin d → ℝ, φ x * ch02_laplacian p x := by
  rw [ch02_noname_11_ibp_lap1_nd hφ hcs (hp.of_le (by norm_num)),
    ch02_noname_12_ibp_lap2_nd (hφ.of_le (by norm_num)) hcs hp]

/-- **eq:sm-weakform ⇒ eq:sm-fokkerplanck on `ℝ^d`** (Steps 4-6 of tb:sm-fokkerplanck).

If the weak form holds against every test function -- `∫ φ ∂ₜp = ∫ f·∇φ p + ½g² ∫ Δφ p`
-- then the boxed strong form `∂ₜp = -∇·[f p] + ½g² Δp` holds at EVERY point of `ℝ^d`.
`q` stands for `∂ₜp_t` and `fp` for the field `f p_t`.  Unlike the one-dimensional
`ch02_sm_weakform_implies_fokkerplanck`, the integrations by parts and the
integrability of the three integrals are not assumed: they are derived from the
smoothness and compact support the text itself assumes.

What is NOT proved -- here or anywhere in this module -- is the weak form itself, which
is Steps 1-3 (Itô's lemma, the vanishing of the Itô integral's expectation, and the
generator identity eq:sm-generator): this Mathlib has no stochastic integral. -/
theorem ch02_sm_weakform_implies_fokkerplanck_nd {d : ℕ}
    {p q : (Fin d → ℝ) → ℝ} {fp : (Fin d → ℝ) → Fin d → ℝ} (g2 : ℝ)
    (hq : Continuous q) (hp : ContDiff ℝ 2 p)
    (hfp : ∀ k, ContDiff ℝ 1 (fun y => fp y k))
    (hweak : ∀ φ : (Fin d → ℝ) → ℝ, ContDiff ℝ ∞ φ → HasCompactSupport φ →
        ∫ x : Fin d → ℝ, φ x * q x
          = (∫ x : Fin d → ℝ, ∑ k, ch02_partial φ k x * fp x k)
            + g2 / 2 * ∫ x : Fin d → ℝ, ch02_laplacian φ x * p x) :
    ∀ x, q x = -ch02_div fp x + g2 / 2 * ch02_laplacian p x := by
  have hdiv : Continuous (ch02_div fp) := ch02_div_continuous hfp
  have hlap : Continuous (ch02_laplacian p) := ch02_laplacian_continuous hp
  have hbr : Continuous (fun x => q x + ch02_div fp x - g2 / 2 * ch02_laplacian p x) :=
    (hq.add hdiv).sub (continuous_const.mul hlap)
  have hzero : ∀ φ : (Fin d → ℝ) → ℝ, ContDiff ℝ ∞ φ → HasCompactSupport φ →
      ∫ x : Fin d → ℝ, φ x * (q x + ch02_div fp x - g2 / 2 * ch02_laplacian p x) = 0 := by
    intro φ hφ hφc
    have hφ1 : ContDiff ℝ 1 φ := contDiff_infty.1 hφ 1
    have hφ2 : ContDiff ℝ 2 φ := contDiff_infty.1 hφ 2
    have h1 : Integrable (fun x : Fin d → ℝ => φ x * q x) :=
      ch02_integrable_testMul hφ.continuous hφc hq
    have h2 : Integrable (fun x : Fin d → ℝ => φ x * ch02_div fp x) :=
      ch02_integrable_testMul hφ.continuous hφc hdiv
    have h3 : Integrable (fun x : Fin d → ℝ => φ x * ch02_laplacian p x) :=
      ch02_integrable_testMul hφ.continuous hφc hlap
    have e1 : (fun x : Fin d → ℝ =>
          φ x * (q x + ch02_div fp x - g2 / 2 * ch02_laplacian p x))
        = fun x => (φ x * q x + φ x * ch02_div fp x)
            - g2 / 2 * (φ x * ch02_laplacian p x) := by
      funext x; ring
    have hA : Integrable
        (fun x : Fin d → ℝ => φ x * q x + φ x * ch02_div fp x) volume := h1.add h2
    have hB : Integrable
        (fun x : Fin d → ℝ => g2 / 2 * (φ x * ch02_laplacian p x)) volume :=
      h3.const_mul (g2 / 2)
    rw [e1, integral_sub hA hB, integral_add h1 h2, integral_const_mul,
      hweak φ hφ hφc, ch02_noname_10_ibp_drift_nd hφ1 hφc hfp,
      ch02_laplacian_selfAdjoint_nd hφ2 hφc hp]
    ring
  have hbrz := ch02_noname_13_nd _ hbr hzero
  intro x
  have hx := congrFun hbrz x
  simp only [Pi.zero_apply] at hx
  linarith

/-! #### eq:sm-fokkerplanck (tex 688-698), eq:sm-ou (240-245) and noname-3 (286-292) on ℝ^d

Pass 4 proved the Fokker-Planck equation for the whole linear-Gaussian class in ONE
dimension (`ch02_fokkerplanck_linear_gaussian`).  The chapter states the equation on
`ℝ^d`, and the boxed display noname-3 states the channel on `ℝ^d` with a vector mean
`e^{-t}a` and a scalar variance `Δ_t`.  That is the family treated here: the Gaussian
density with an arbitrary per-coordinate mean `m k` and one shared variance `v`, which
by `ch02_lgDensityNd_eq_gauss` IS the isotropic Gaussian of the boxed display, and by
`ch02_lgDensityNd_eq_prod` IS the product of the `d` scalar channels of
`ch02_noname_3_factorises`.  For it the `d`-dimensional equation
`∂ₜp = -∇·[f p] + ½g²Δp`, with `∇·` and `Δ` the `ch02_div` and `ch02_laplacian` used
throughout, is proved at arbitrary `d` for an arbitrary time-dependent linear drift
`f(x,t) = -θ(t)x` and an arbitrary diffusion schedule `g(t)`. -/

/-- The `ℝ^d` log-density of the linear-Gaussian family: a sum over coordinates, which is
the formal content of the boxed display's "this energy separates across coordinates". -/
noncomputable def ch02_lgLogDensityNd {d : ℕ} (m : Fin d → ℝ → ℝ) (v : ℝ → ℝ)
    (x : Fin d → ℝ) (t : ℝ) : ℝ := ∑ k, ch02_lgLogDensity (m k) v (x k) t

/-- The `ℝ^d` density of the linear-Gaussian family. -/
noncomputable def ch02_lgDensityNd {d : ℕ} (m : Fin d → ℝ → ℝ) (v : ℝ → ℝ)
    (x : Fin d → ℝ) (t : ℝ) : ℝ := Real.exp (ch02_lgLogDensityNd m v x t)

/-- It is the product of the `d` scalar channels: "the coordinates evolve independently". -/
theorem ch02_lgDensityNd_eq_prod {d : ℕ} (m : Fin d → ℝ → ℝ) (v : ℝ → ℝ)
    (x : Fin d → ℝ) (t : ℝ) :
    ch02_lgDensityNd m v x t = ∏ k, ch02_lgDensity (m k) v (x k) t := by
  simp only [ch02_lgDensityNd, ch02_lgLogDensityNd, ch02_lgDensity, Real.exp_sum]

/-- And it is the isotropic Gaussian of the boxed display noname-3. -/
theorem ch02_lgDensityNd_eq_gauss {d : ℕ} (m : Fin d → ℝ → ℝ) (v : ℝ → ℝ)
    (x : Fin d → ℝ) {t : ℝ} (hv : 0 < v t) :
    ch02_lgDensityNd m v x t
      = Real.exp (-(∑ k, (x k - m k t) ^ 2) / (2 * v t))
          / Real.sqrt (2 * Real.pi * v t) ^ d := by
  rw [ch02_lgDensityNd_eq_prod,
    Finset.prod_congr rfl (fun k _ => ch02_lgDensity_eq_gauss (m k) v (x k) t hv)]
  exact ch02_noname_3_factorises (v t) x (fun k => m k t)

theorem ch02_lgDensityNd_pos {d : ℕ} (m : Fin d → ℝ → ℝ) (v : ℝ → ℝ)
    (x : Fin d → ℝ) (t : ℝ) : 0 < ch02_lgDensityNd m v x t := Real.exp_pos _

/-- A function of the single coordinate `x k` is differentiable, with the expected
directional derivative. -/
theorem ch02_hasFDerivAt_coordFun {d : ℕ} {c : ℝ → ℝ} {c' : ℝ} (k : Fin d) (x : Fin d → ℝ)
    (h : HasDerivAt c c' (x k)) :
    HasFDerivAt (fun y : Fin d → ℝ => c (y k))
      (c' • ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Fin d => ℝ) k) x :=
  h.comp_hasFDerivAt x (hasFDerivAt_apply k x)

/-- Evaluating a sum of scaled coordinate projections at the `j`-th basis vector picks out
the `j`-th coefficient: this is how `ch02_partial` reads off a gradient. -/
theorem ch02_sum_proj_single {d : ℕ} (c : Fin d → ℝ) (j : Fin d) :
    (∑ k, c k • ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Fin d => ℝ) k)
        (Pi.single j (1 : ℝ)) = c j := by
  simp [ContinuousLinearMap.proj_apply, Pi.single_apply]

/-- `∂ₖ` of a function of the single coordinate `x k`. -/
theorem ch02_partial_coordFun_self {d : ℕ} {c : ℝ → ℝ} {c' : ℝ} (k : Fin d) (x : Fin d → ℝ)
    (h : HasDerivAt c c' (x k)) : ch02_partial (fun y : Fin d → ℝ => c (y k)) k x = c' := by
  simp only [ch02_partial, (ch02_hasFDerivAt_coordFun k x h).fderiv,
    ContinuousLinearMap.smul_apply, ContinuousLinearMap.proj_apply, smul_eq_mul,
    Pi.single_eq_same, mul_one]

/-- The `x`-derivative of the scalar log-density: `∂ₓ log p = -(x - m)/v`. -/
theorem ch02_lgLogDensity_hasDerivAt_x (m v : ℝ → ℝ) {t : ℝ} (hv : 0 < v t) (y : ℝ) :
    HasDerivAt (fun z => ch02_lgLogDensity m v z t) (-((y - m t) / v t)) y := by
  have hy : HasDerivAt (fun z : ℝ => z - m t) 1 y := (hasDerivAt_id y).sub_const (m t)
  have hp2 : HasDerivAt (fun z : ℝ => (z - m t) ^ 2) (2 * (y - m t)) y := by
    simpa using hy.fun_pow 2
  have h1 : HasDerivAt (fun z : ℝ => -(z - m t) ^ 2 / (2 * v t))
      (-(2 * (y - m t)) / (2 * v t)) y := hp2.neg.div_const (2 * v t)
  have h2 := h1.sub_const (Real.log (2 * Real.pi * v t) / 2)
  have hsimp : -(2 * (y - m t)) / (2 * v t) = -((y - m t) / v t) := by
    rw [neg_div, mul_div_mul_left _ _ (by norm_num : (2:ℝ) ≠ 0)]
  rw [hsimp] at h2
  unfold ch02_lgLogDensity
  exact h2

/-- The gradient of the `ℝ^d` log-density, coordinate by coordinate. -/
theorem ch02_lgLogDensityNd_hasFDerivAt {d : ℕ} (m : Fin d → ℝ → ℝ) (v : ℝ → ℝ) {t : ℝ}
    (hv : 0 < v t) (y : Fin d → ℝ) :
    HasFDerivAt (fun z : Fin d → ℝ => ch02_lgLogDensityNd m v z t)
      (∑ k, (-((y k - m k t) / v t)) •
        ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Fin d => ℝ) k) y := by
  unfold ch02_lgLogDensityNd
  exact HasFDerivAt.fun_sum fun k _ =>
    ch02_hasFDerivAt_coordFun k y (ch02_lgLogDensity_hasDerivAt_x (m k) v hv (y k))

theorem ch02_lgDensityNd_hasFDerivAt {d : ℕ} (m : Fin d → ℝ → ℝ) (v : ℝ → ℝ) {t : ℝ}
    (hv : 0 < v t) (y : Fin d → ℝ) :
    HasFDerivAt (fun z : Fin d → ℝ => ch02_lgDensityNd m v z t)
      (ch02_lgDensityNd m v y t •
        ∑ k, (-((y k - m k t) / v t)) •
          ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Fin d => ℝ) k) y :=
  (ch02_lgLogDensityNd_hasFDerivAt m v hv y).exp

/-- `∂ₖp = -((xₖ - mₖ)/v) p`: the score of the `ℝ^d` channel. -/
theorem ch02_lgDensityNd_partial {d : ℕ} (m : Fin d → ℝ → ℝ) (v : ℝ → ℝ) {t : ℝ}
    (hv : 0 < v t) (j : Fin d) (y : Fin d → ℝ) :
    ch02_partial (fun z => ch02_lgDensityNd m v z t) j y
      = -((y j - m j t) / v t) * ch02_lgDensityNd m v y t := by
  simp only [ch02_partial, (ch02_lgDensityNd_hasFDerivAt m v hv y).fderiv,
    ContinuousLinearMap.smul_apply, smul_eq_mul, ch02_sum_proj_single]
  ring

/-- **eq:sm-fokkerplanck on `ℝ^d`** for the linear-Gaussian family: with drift
`f(x,t) = -θ(t)x` and diffusion `g(t)`, the Gaussian density with per-coordinate means
`m k` and shared variance `v` satisfies `∂ₜp = -∇·[f p] + ½g²Δp` at every point of `ℝ^d`
WHENEVER the means and the variance solve the moment ODEs `mₖ' = -θ mₖ` and
`v' = -2θv + g²`.  (In one dimension the converse also holds, so there the relation is
an equivalence: `ch02_fokkerplanck_linear_gaussian_converse`.) -/
theorem ch02_fokkerplanck_linear_gaussian_nd {d : ℕ} (m : Fin d → ℝ → ℝ) (v θ g : ℝ → ℝ)
    (x : Fin d → ℝ) {t : ℝ} (hv : 0 < v t)
    (hm : ∀ k, HasDerivAt (m k) (-(θ t) * m k t) t)
    (hvd : HasDerivAt v (-(2 * θ t * v t) + g t ^ 2) t) :
    HasDerivAt (fun s => ch02_lgDensityNd m v x s)
      (-ch02_div (fun y k => (-(θ t) * y k) * ch02_lgDensityNd m v y t) x
        + g t ^ 2 / 2 * ch02_laplacian (fun y => ch02_lgDensityNd m v y t) x) t := by
  have hvne : v t ≠ 0 := ne_of_gt hv
  have hPdiff : ∀ y : Fin d → ℝ,
      DifferentiableAt ℝ (fun z : Fin d → ℝ => ch02_lgDensityNd m v z t) y :=
    fun y => (ch02_lgDensityNd_hasFDerivAt m v hv y).differentiableAt
  -- the score factor `-((y j - m j t)/v t)` and its own `j`-th partial derivative
  have hcderiv : ∀ (j : Fin d) (z : Fin d → ℝ),
      HasDerivAt (fun s : ℝ => -((s - m j t) / v t)) (-(1 / v t)) (z j) := by
    intro j z
    have h : HasDerivAt (fun s : ℝ => (s - m j t) / v t) (1 / v t) (z j) :=
      ((hasDerivAt_id (z j)).sub_const (m j t)).div_const (v t)
    exact h.fun_neg
  have hcdiff : ∀ (j : Fin d) (z : Fin d → ℝ),
      DifferentiableAt ℝ (fun y : Fin d → ℝ => -((y j - m j t) / v t)) z :=
    fun j z => (ch02_hasFDerivAt_coordFun j z (hcderiv j z)).differentiableAt
  have hcpart : ∀ (j : Fin d) (z : Fin d → ℝ),
      ch02_partial (fun y : Fin d → ℝ => -((y j - m j t) / v t)) j z = -(1 / v t) :=
    fun j z => ch02_partial_coordFun_self j z (hcderiv j z)
  -- the Laplacian of the density
  have hpfun : ∀ j : Fin d,
      ch02_partial (fun z : Fin d → ℝ => ch02_lgDensityNd m v z t) j
        = fun y => -((y j - m j t) / v t) * ch02_lgDensityNd m v y t :=
    fun j => funext fun y => ch02_lgDensityNd_partial m v hv j y
  have hlap : ch02_laplacian (fun y : Fin d → ℝ => ch02_lgDensityNd m v y t) x
      = ∑ j, (-(1 / v t) + ((x j - m j t) / v t) ^ 2) * ch02_lgDensityNd m v x t := by
    simp only [ch02_laplacian]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [hpfun j, ch02_partial_mul (fun y : Fin d → ℝ => -((y j - m j t) / v t))
        (fun z : Fin d → ℝ => ch02_lgDensityNd m v z t) j x (hcdiff j x) (hPdiff x),
      hcpart j x, ch02_lgDensityNd_partial m v hv j x]
    ring
  -- the divergence of the drift flux
  have hdiv : ch02_div (fun y k => (-(θ t) * y k) * ch02_lgDensityNd m v y t) x
      = ∑ k, (-(θ t) * ch02_lgDensityNd m v x t
          + (-(θ t) * x k)
            * (-((x k - m k t) / v t) * ch02_lgDensityNd m v x t)) := by
    simp only [ch02_div]
    refine Finset.sum_congr rfl fun k _ => ?_
    have hlin : HasDerivAt (fun s : ℝ => -(θ t) * s) (-(θ t)) (x k) := by
      simpa using (hasDerivAt_id (x k)).const_mul (-(θ t))
    have hlindiff : DifferentiableAt ℝ (fun y : Fin d → ℝ => -(θ t) * y k) x :=
      (ch02_hasFDerivAt_coordFun k x hlin).differentiableAt
    rw [ch02_partial_mul (fun y : Fin d → ℝ => -(θ t) * y k)
        (fun z : Fin d → ℝ => ch02_lgDensityNd m v z t) k x hlindiff (hPdiff x),
      ch02_partial_coordFun_self k x hlin, ch02_lgDensityNd_partial m v hv k x]
  -- the time derivative, coordinate by coordinate
  have hL : HasDerivAt (fun s => ch02_lgLogDensityNd m v x s)
      (∑ k, ((x k - m k t) * (-(θ t) * m k t) / v t
        + (x k - m k t) ^ 2 * (-(2 * θ t * v t) + g t ^ 2) / (2 * v t ^ 2)
        - (-(2 * θ t * v t) + g t ^ 2) / (2 * v t))) t := by
    unfold ch02_lgLogDensityNd
    exact HasDerivAt.fun_sum fun k _ =>
      ch02_lgLogDensity_deriv (m k) v (x k) hv (hm k) hvd
  have hexp : HasDerivAt (fun s => ch02_lgDensityNd m v x s)
      (ch02_lgDensityNd m v x t *
        ∑ k, ((x k - m k t) * (-(θ t) * m k t) / v t
          + (x k - m k t) ^ 2 * (-(2 * θ t * v t) + g t ^ 2) / (2 * v t ^ 2)
          - (-(2 * θ t * v t) + g t ^ 2) / (2 * v t))) t := hL.exp
  rw [hlap, hdiv]
  convert hexp using 1
  rw [← Finset.sum_neg_distrib, Finset.mul_sum, ← Finset.sum_add_distrib, Finset.mul_sum]
  refine Finset.sum_congr rfl fun k _ => ?_
  field_simp
  ring

/-- The Ornstein-Uhlenbeck channel of eq:sm-ou and of the boxed display noname-3, on
`ℝ^d`: `θ ≡ 1`, `g ≡ √2`, mean `e^{-t}a` with `a ∈ ℝ^d`, variance `Δ_t = 1 - e^{-2t}`. -/
theorem ch02_fokkerplanck_ou_nd {d : ℕ} (a : Fin d → ℝ) (x : Fin d → ℝ) {t : ℝ}
    (ht : 0 < t) :
    HasDerivAt (fun s => ch02_lgDensityNd (fun k r => Real.exp (-r) * a k) ch02_Delta x s)
      (-ch02_div (fun y k => (-(1 : ℝ) * y k)
            * ch02_lgDensityNd (fun k r => Real.exp (-r) * a k) ch02_Delta y t) x
        + Real.sqrt 2 ^ 2 / 2
          * ch02_laplacian (fun y =>
              ch02_lgDensityNd (fun k r => Real.exp (-r) * a k) ch02_Delta y t) x) t := by
  have hΔ : 0 < ch02_Delta t := ch02_sm_ou_Delta_pos ht
  have hs2 : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  have hm : ∀ k : Fin d, HasDerivAt (fun r : ℝ => Real.exp (-r) * a k)
      (-(1 : ℝ) * (Real.exp (-t) * a k)) t := by
    intro k
    have h : HasDerivAt (fun r : ℝ => -r) (-1) t := (hasDerivAt_id t).neg
    have h2 : HasDerivAt (fun r : ℝ => Real.exp (-r)) (Real.exp (-t) * -1) t := h.exp
    have h3 := h2.mul_const (a k)
    have heq : Real.exp (-t) * -1 * a k = -(1 : ℝ) * (Real.exp (-t) * a k) := by ring
    rwa [heq] at h3
  have hvd : HasDerivAt ch02_Delta
      (-(2 * 1 * ch02_Delta t) + Real.sqrt 2 ^ 2) t := by
    have h := ch02_sm_ou_var_ode t
    have heq : (2 : ℝ) - 2 * ch02_Delta t = -(2 * 1 * ch02_Delta t) + Real.sqrt 2 ^ 2 := by
      rw [hs2]; ring
    rwa [heq] at h
  exact ch02_fokkerplanck_linear_gaussian_nd _ _ (fun _ => (1 : ℝ)) (fun _ => Real.sqrt 2)
    x hΔ hm hvd

/-! #### eq:sm-fokkerplanck: the converse direction, in one dimension

The note attached to eq:sm-fokkerplanck used to say that the Gaussian solves the equation
"exactly when" the moments solve the ODEs, while only the forward direction was proved.
The converse is proved here, in one dimension and for the whole linear-Gaussian class:
if the Gaussian density with mean `m` and variance `v` satisfies the Fokker-Planck
identity at EVERY `x`, then `m` and `v` MUST solve `m' = -θm` and `v' = -2θv + g²`.  So
the displayed `Δ_t = 1 - e^{-2t}` and `e^{-t}a` of eq:sm-ou really are THE solutions the
equation forces, not merely solutions that work. -/

/-- The Fokker-Planck right-hand side of the linear-Gaussian family, computed: this is the
algebra shared by the two directions. -/
theorem ch02_fokkerplanck_rhs_eq (m v θ g : ℝ → ℝ) {t : ℝ} (hv : 0 < v t) (x : ℝ) :
    -deriv (fun y => (-(θ t) * y) * ch02_lgDensity m v y t) x
        + g t ^ 2 / 2 * deriv (deriv (fun y => ch02_lgDensity m v y t)) x
      = ((x - m t) * (-(θ t) * m t) / v t
          + (x - m t) ^ 2 * (-(2 * θ t * v t) + g t ^ 2) / (2 * v t ^ 2)
          - (-(2 * θ t * v t) + g t ^ 2) / (2 * v t)) * ch02_gauss (v t) x (m t) := by
  have hvne : v t ≠ 0 := ne_of_gt hv
  have hfun : (fun y => ch02_lgDensity m v y t) = fun y => ch02_gauss (v t) y (m t) := by
    funext y; exact ch02_lgDensity_eq_gauss m v y t hv
  have hd1 : deriv (fun y => ch02_lgDensity m v y t)
      = fun y => -((y - m t) / v t) * ch02_gauss (v t) y (m t) := by
    rw [hfun]; funext y; exact (ch02_kernelgrad hv y _).deriv
  have hd2 : deriv (deriv (fun y => ch02_lgDensity m v y t)) x
      = (-(1 / v t) + ((x - m t) / v t) ^ 2) * ch02_gauss (v t) x (m t) := by
    rw [hd1]; exact (ch02_gauss_deriv2 hv x _).deriv
  have hd3 : deriv (fun y => (-(θ t) * y) * ch02_lgDensity m v y t) x
      = -(θ t) * 1 * ch02_gauss (v t) x (m t)
        + -(θ t) * x * (-((x - m t) / v t) * ch02_gauss (v t) x (m t)) := by
    have hxfun : (fun y => (-(θ t) * y) * ch02_lgDensity m v y t)
        = fun y => (-(θ t) * y) * ch02_gauss (v t) y (m t) := by
      funext y; rw [ch02_lgDensity_eq_gauss m v y t hv]
    rw [hxfun]
    exact (((hasDerivAt_id x).const_mul (-(θ t))).mul (ch02_kernelgrad hv x _)).deriv
  rw [hd2, hd3]
  field_simp
  ring

/-- **The converse of `ch02_fokkerplanck_linear_gaussian`** (one dimension): satisfying the
Fokker-Planck equation at every `x` FORCES the moment ODEs. -/
theorem ch02_fokkerplanck_linear_gaussian_converse (m v θ g : ℝ → ℝ) {t m' v' : ℝ}
    (hv : 0 < v t) (hm : HasDerivAt m m' t) (hvd : HasDerivAt v v' t)
    (hfp : ∀ x : ℝ, HasDerivAt (fun s => ch02_lgDensity m v x s)
      (-deriv (fun y => (-(θ t) * y) * ch02_lgDensity m v y t) x
        + g t ^ 2 / 2 * deriv (deriv (fun y => ch02_lgDensity m v y t)) x) t) :
    m' = -(θ t) * m t ∧ v' = -(2 * θ t * v t) + g t ^ 2 := by
  have hvne : v t ≠ 0 := ne_of_gt hv
  have key : ∀ x : ℝ,
      (x - m t) * m' / v t + (x - m t) ^ 2 * v' / (2 * v t ^ 2) - v' / (2 * v t)
        = (x - m t) * (-(θ t) * m t) / v t
            + (x - m t) ^ 2 * (-(2 * θ t * v t) + g t ^ 2) / (2 * v t ^ 2)
            - (-(2 * θ t * v t) + g t ^ 2) / (2 * v t) := by
    intro x
    have hA := (ch02_lgLogDensity_deriv m v x hv hm hvd).exp
    have hGexp : Real.exp (ch02_lgLogDensity m v x t) = ch02_gauss (v t) x (m t) :=
      ch02_lgDensity_eq_gauss m v x t hv
    rw [hGexp] at hA
    have heq := hA.unique (hfp x)
    rw [ch02_fokkerplanck_rhs_eq m v θ g hv x] at heq
    refine mul_right_cancel₀ (ne_of_gt (ch02_gauss_pos hv x (m t))) ?_
    linear_combination heq
  have hv'eq : v' = -(2 * θ t * v t) + g t ^ 2 := by
    have h := key (m t)
    have h2 : v' / (2 * v t) = (-(2 * θ t * v t) + g t ^ 2) / (2 * v t) := by
      linear_combination -h
    field_simp at h2
    linarith
  refine ⟨?_, hv'eq⟩
  have h := key (m t + 1)
  rw [hv'eq] at h
  have h2 : m' / v t = (-(θ t) * m t) / v t := by linear_combination h
  field_simp at h2
  linarith

/-! #### eq:sm-forward (tex 215-218): the terminal law on ℝ^d

Pass 3 proved the terminal-law claim in one dimension: the time-`t` marginal of an
arbitrary finitely supported data law converges, at every point and as `t → ∞`, to the
standard Gaussian density.  The chapter's `X_t` lives in `ℝ^d`, so the same statement is
proved here on `ℝ^d`, for an arbitrary finite number of atoms at arbitrary positions
`A i ∈ ℝ^d` with arbitrary weights summing to one.  The display itself is still the SDE
and is still not formalised; see the note on the Itô boundary. -/

/-- The time-`t` marginal of eq:marg on `ℝ^d`: a finitely supported data law pushed
through the isotropic Gaussian channel of the boxed display noname-3. -/
noncomputable def ch02_margNd {N d : ℕ} (lam : Fin N → ℝ) (A : Fin N → Fin d → ℝ)
    (Δ c : ℝ) (x : Fin d → ℝ) : ℝ :=
  ∑ i, lam i * ∏ k, ch02_gauss Δ (x k) (c * A i k)

theorem ch02_sm_forward_terminal_nd {N d : ℕ} (lam : Fin N → ℝ) (A : Fin N → Fin d → ℝ)
    (hlam : ∑ i, lam i = 1) (x : Fin d → ℝ) :
    Filter.Tendsto (fun t : ℝ => ch02_margNd lam A (ch02_Delta t) (Real.exp (-t)) x)
      Filter.atTop (nhds (∏ k, ch02_gauss 1 (x k) 0)) := by
  have hmean : ∀ (i : Fin N) (k : Fin d),
      Filter.Tendsto (fun t : ℝ => Real.exp (-t) * A i k) Filter.atTop (nhds 0) := by
    intro i k
    simpa using Real.tendsto_exp_neg_atTop_nhds_zero.mul_const (A i k)
  have hprod : ∀ i : Fin N, Filter.Tendsto
      (fun t : ℝ => ∏ k, ch02_gauss (ch02_Delta t) (x k) (Real.exp (-t) * A i k))
      Filter.atTop (nhds (∏ k, ch02_gauss 1 (x k) 0)) := fun i =>
    tendsto_finsetProd _ fun k _ =>
      ch02_gauss_tendsto (x k) ch02_Delta_tendsto_one (hmean i k)
  have hterm : ∀ i : Fin N, Filter.Tendsto
      (fun t : ℝ => lam i * ∏ k, ch02_gauss (ch02_Delta t) (x k) (Real.exp (-t) * A i k))
      Filter.atTop (nhds (lam i * ∏ k, ch02_gauss 1 (x k) 0)) :=
    fun i => (hprod i).const_mul (lam i)
  have hval : ∑ _i : Fin N, lam _i * ∏ k, ch02_gauss 1 (x k) 0
      = ∏ k, ch02_gauss 1 (x k) 0 := by
    rw [← Finset.sum_mul, hlam, one_mul]
  have hsum := tendsto_finsetSum (Finset.univ : Finset (Fin N)) (fun i _ => hterm i)
  rw [hval] at hsum
  have hfun : (fun t : ℝ => ch02_margNd lam A (ch02_Delta t) (Real.exp (-t)) x)
      = fun t : ℝ =>
        ∑ i, lam i * ∏ k, ch02_gauss (ch02_Delta t) (x k) (Real.exp (-t) * A i k) := rfl
  rw [hfun]
  exact hsum

/-- The terminal law of the previous theorem IS the standard Gaussian density on `ℝ^d`,
the "simple terminal measure" whose partition function is noname-4. -/
theorem ch02_sm_forward_terminal_nd_gauss {d : ℕ} (x : Fin d → ℝ) :
    ∏ k, ch02_gauss 1 (x k) 0
      = Real.exp (-(∑ k, x k ^ 2) / 2) / Real.sqrt (2 * Real.pi) ^ d := by
  simpa using ch02_noname_3_factorises (d := d) 1 x (fun _ => 0)

/-! #### eq:sm-probflow-continuity (tex 723-735) on ℝ^d

The continuity-equation rewriting of the Fokker-Planck right-hand side.  Pass 3 proved it
in one dimension; with the divergence of noname-9 and the field identity of noname-14
available it is proved here at arbitrary `d`, for an arbitrary vector field `f` and an
arbitrary positive differentiable density, exactly as `ch02_sm_reverse_nd` does for the
reverse drift. -/

theorem ch02_sm_probflow_continuity_nd {d : ℕ} (p : (Fin d → ℝ) → ℝ)
    (f : (Fin d → ℝ) → Fin d → ℝ) (g2 : ℝ) (x : Fin d → ℝ)
    (hp : ∀ y, 0 < p y) (hd : Differentiable ℝ p)
    (hfld : ∀ k, DifferentiableAt ℝ (fun y => f y k * p y) x)
    (hpk : ∀ k, DifferentiableAt ℝ (ch02_partial p k) x) :
    -ch02_div (fun y k =>
        (f y k - g2 / 2 * ch02_partial (fun z => Real.log (p z)) k y) * p y) x
      = -ch02_div (fun y k => f y k * p y) x + g2 / 2 * ch02_laplacian p x := by
  have hfield : ∀ k : Fin d,
      (fun y => (f y k - g2 / 2 * ch02_partial (fun z => Real.log (p z)) k y) * p y)
        = fun y => f y k * p y - g2 / 2 * ch02_partial p k y := by
    intro k; funext y
    have hy := congrFun (ch02_noname_14_score_field p hp hd k) y
    calc (f y k - g2 / 2 * ch02_partial (fun z => Real.log (p z)) k y) * p y
        = f y k * p y - g2 / 2 * (p y * ch02_partial (fun z => Real.log (p z)) k y) := by
          ring
      _ = f y k * p y - g2 / 2 * ch02_partial p k y := by rw [hy]
  have hdiv : ch02_div (fun y k =>
        (f y k - g2 / 2 * ch02_partial (fun z => Real.log (p z)) k y) * p y) x
      = ch02_div (fun y k => f y k * p y) x - g2 / 2 * ch02_laplacian p x := by
    simp only [ch02_div, ch02_laplacian]
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [hfield k, ch02_partial_sub _ _ k x (hfld k) ((hpk k).const_mul (g2 / 2)),
      ch02_partial_const_mul _ (g2 / 2) k x (hpk k)]
  rw [hdiv]
  ring

/-! #### noname-8 (tex 610-616) and eq:sm-reverse (tex 773-786) on ℝ^d

The two remaining one-dimensional statements of the toolbox.  Neither restriction was an
obstruction: Mathlib's dominated differentiation-under-the-integral theorem is stated for
an ARBITRARY measure space, so Step 3 costs no more on `ℝ^d` than on the line, and the
time-reversal corollary of eq:sm-reverse only needed the `ℝ^d` identity `ch02_sm_reverse_nd`
that Pass 4 already proved. -/

/-- **noname-8 on `ℝ^d`**: `d/dt E[φ(X_t)] = d/dt ∫ φ p_t = ∫ φ ∂ₜp_t`, by dominated
convergence, with the text's own hypotheses (`φ` bounded of compact support supplies the
dominating function `bd`). -/
theorem ch02_noname_8_dt_under_integral_nd {d : ℕ}
    (φ : (Fin d → ℝ) → ℝ) (p pt : ℝ → (Fin d → ℝ) → ℝ) (bd : (Fin d → ℝ) → ℝ)
    {t₀ : ℝ} {S : Set ℝ}
    (hS : S ∈ nhds t₀)
    (hmeas : ∀ᶠ t in nhds t₀,
      AEStronglyMeasurable (fun x => φ x * p t x) (volume : Measure (Fin d → ℝ)))
    (hint : Integrable (fun x => φ x * p t₀ x) (volume : Measure (Fin d → ℝ)))
    (hmeas' : AEStronglyMeasurable (fun x => φ x * pt t₀ x) (volume : Measure (Fin d → ℝ)))
    (hbound : ∀ᵐ x ∂(volume : Measure (Fin d → ℝ)), ∀ t ∈ S, ‖φ x * pt t x‖ ≤ bd x)
    (hbint : Integrable bd (volume : Measure (Fin d → ℝ)))
    (hdiff : ∀ᵐ x ∂(volume : Measure (Fin d → ℝ)), ∀ t ∈ S,
      HasDerivAt (fun τ => φ x * p τ x) (φ x * pt t x) t) :
    HasDerivAt (fun τ => ∫ x : Fin d → ℝ, φ x * p τ x)
      (∫ x : Fin d → ℝ, φ x * pt t₀ x) t₀ :=
  (hasDerivAt_integral_of_dominated_loc_of_deriv_le (F := fun τ x => φ x * p τ x)
    (F' := fun τ x => φ x * pt τ x) (bound := bd) hS hmeas hint hmeas' hbound hbint hdiff).2

/-- **eq:sm-reverse on `ℝ^d`, at the level of densities**: if the forward marginal `p_t`
solves the Fokker-Planck equation with drift `f(·,t)` and diffusion `g(t)²`, then the
reversed density `q_τ := p_{T-τ}` solves the Fokker-Planck equation of the reverse SDE,
whose drift in forward time is `-(f - g²∇log p)`.  Arbitrary `d`, arbitrary
time-dependent `f` and `g`, arbitrary positive differentiable `p`.  The pathwise
statement (Anderson's theorem) remains out of reach: it needs a stochastic integral. -/
theorem ch02_sm_reverse_time_reversal_nd {d : ℕ} (p : ℝ → (Fin d → ℝ) → ℝ)
    (f : ℝ → (Fin d → ℝ) → Fin d → ℝ) (g2 : ℝ → ℝ) (T τ : ℝ) (x : Fin d → ℝ) (dtp : ℝ)
    (hp : ∀ y, 0 < p (T - τ) y) (hd : Differentiable ℝ (p (T - τ)))
    (hfld : ∀ k, DifferentiableAt ℝ (fun y => f (T - τ) y k * p (T - τ) y) x)
    (hpk : ∀ k, DifferentiableAt ℝ (ch02_partial (p (T - τ)) k) x)
    (hdt : HasDerivAt (fun s => p s x) dtp (T - τ))
    (hfwd : dtp = -ch02_div (fun y k => f (T - τ) y k * p (T - τ) y) x
        + g2 (T - τ) / 2 * ch02_laplacian (p (T - τ)) x) :
    HasDerivAt (fun s => p (T - s) x)
      (-ch02_div (fun y k => (-(f (T - τ) y k)
            + g2 (T - τ) * ch02_partial (fun z => Real.log (p (T - τ) z)) k y)
            * p (T - τ) y) x
        + g2 (T - τ) / 2 * ch02_laplacian (p (T - τ)) x) τ := by
  have hlin : HasDerivAt (fun s : ℝ => T - s) (-1) τ := (hasDerivAt_id τ).const_sub T
  have hchain : HasDerivAt (fun s => p (T - s) x) (-dtp) τ := by
    have h := hdt.comp τ hlin
    simpa [Function.comp_def] using h
  have hrev := ch02_sm_reverse_nd (p (T - τ)) (f (T - τ)) (g2 (T - τ)) x hp hd hfld hpk
  rw [hrev, ← hfwd]
  exact hchain

/-- **eq:sm-probability-flow (tex 737-745) on `ℝ^d`**: the velocity field of the
probability-flow ODE, `f(x,t) - ½g(t)²∇ₓlog p_t(x)`, as a vector field.  A definition; the
claim that it reproduces the marginals `p_t` is `ch02_sm_probflow_continuity_nd`. -/
noncomputable def ch02_sm_probability_flow_velocity_nd {d : ℕ}
    (f : (Fin d → ℝ) → ℝ → Fin d → ℝ) (g : ℝ → ℝ) (s : (Fin d → ℝ) → ℝ → Fin d → ℝ)
    (x : Fin d → ℝ) (t : ℝ) (k : Fin d) : ℝ :=
  f x t k - g t ^ 2 / 2 * s x t k

end ThesisAudit
