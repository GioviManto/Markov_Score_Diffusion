import Mathlib

/-!
Audit of the display-math formulas in
`thesis/chapters/ch04-ring.tex`, lines 1-671
("The Rotating Ring: A Two-Dimensional Dynamic Object").

Conventions used throughout this file:

* Plane vectors are `Fin 2 → ℝ`; the rotation `R_ψ` is the literal matrix
  `!![cos ψ, -sin ψ; sin ψ, cos ψ]` (`ch04_R`) acting by `Matrix.mulVec`.
* A trajectory is `Fin L → Fin 2 → ℝ` (block form) or `(Fin L × Fin 2) → ℝ`
  (Kronecker form); `ch04_blockMulVec` is the action of `C ⊗ I₂` on the block
  form and is proved to agree with `Matrix.kronecker` in general `L`.
* Stochastic differential equations cannot be stated in mathlib as such.
  Itô statements are therefore checked at the level of the *coefficients*:
  for `dX = b dU + s dB` the drift of `V(X)` is `V' b + s² V''/2` and its
  diffusion coefficient is `s V'`.  This is the whole content of the algebra
  in \eqref{eq:ring-ito-energy}; the Itô formula itself is not re-proved.
* Laws (`∼ N(m, Σ)`) are checked through their defining first two moments,
  i.e. as covariance identities.
* Gradients/scores are stated as `HasDerivAt` in one coordinate at a time
  (the partial derivatives), which is exactly what the chapter's `∇_{z_k}`
  displays assert componentwise.
* Where the chapter integrates over a latent variable, the audit uses a finite
  (discrete) latent alphabet.  Then "differentiation under the integral sign"
  is a finite sum of derivatives and Fisher's identity, the Tweedie identity
  and the exponential-family identities are honest theorems rather than
  statements needing dominated-convergence side conditions.  Every such
  instance says so in its comment.
-/

namespace ThesisAudit

open Real Finset Matrix
open scoped Kronecker

noncomputable section

/-! ### Shared definitions and helpers -/

/-- `Δ_t = 1 - e^{-2t}`, the channel variance of Section~\ref{sec:ou}. -/
def ch04_Delta (t : ℝ) : ℝ := 1 - Real.exp (-2 * t)

lemma ch04_Delta_pos {t : ℝ} (ht : 0 < t) : 0 < ch04_Delta t := by
  have h : Real.exp (-2 * t) < 1 := by
    rw [Real.exp_lt_one_iff]; linarith
  unfold ch04_Delta; linarith

lemma ch04_Delta_nonneg {t : ℝ} (ht : 0 ≤ t) : 0 ≤ ch04_Delta t := by
  have h : Real.exp (-2 * t) ≤ 1 := by
    rw [Real.exp_le_one_iff]; linarith
  unfold ch04_Delta; linarith

/-- Derivative of a quadratic polynomial; the workhorse for every partial
derivative in this file (all the log-densities here are quadratics in each
single coordinate). -/
lemma ch04_hasDerivAt_quadratic (a b c x : ℝ) :
    HasDerivAt (fun y : ℝ => a * y ^ 2 + b * y + c) (2 * a * x + b) x := by
  have h1 : HasDerivAt (fun y : ℝ => y ^ 2) ((2 : ℝ) * x ^ 1) x := by
    simpa using hasDerivAt_pow 2 x
  have hid : HasDerivAt (fun y : ℝ => y) 1 x := hasDerivAt_id' x
  have h2 := ((h1.const_mul a).add (hid.const_mul b)).add_const c
  have hval : a * ((2 : ℝ) * x ^ 1) + b * 1 = 2 * a * x + b := by ring
  first
    | (rw [hval] at h2; exact h2)
    | (convert h2 using 1; ring)
    | (convert h2 using 2; ring)

/-- Derivative of a finite sum, in the shape used throughout this file. -/
lemma ch04_hasDerivAt_sum {ι : Type*} [Fintype ι] (A : ι → ℝ → ℝ) (A' : ι → ℝ) (x : ℝ)
    (h : ∀ i, HasDerivAt (A i) (A' i) x) :
    HasDerivAt (fun y => ∑ i, A i y) (∑ i, A' i) x :=
  HasDerivAt.fun_sum (fun i _ => h i)

/-! ### §4.1  Model and notation -/

/-- eq:ring-potential (ch04-ring.tex lines 64-67):
`V(r) = (κ/2)(r-1)²`, the confining quadratic potential. -/
def ch04_ring_V (κ r : ℝ) : ℝ := κ / 2 * (r - 1) ^ 2

/-- The potential is nonnegative (for `κ ≥ 0`) and vanishes exactly at the
stable radius `r = 1`. -/
theorem ch04_ring_potential_min {κ : ℝ} (hκ : 0 ≤ κ) (r : ℝ) :
    0 ≤ ch04_ring_V κ r ∧ ch04_ring_V κ 1 = 0 := by
  constructor
  · unfold ch04_ring_V; positivity
  · unfold ch04_ring_V; norm_num

/-- `V' (r) = κ (r-1)`. -/
theorem ch04_ring_V_hasDerivAt (κ r : ℝ) :
    HasDerivAt (ch04_ring_V κ) (κ * (r - 1)) r := by
  have h : (ch04_ring_V κ) = fun y : ℝ => (κ / 2) * y ^ 2 + (-κ) * y + κ / 2 := by
    funext y; unfold ch04_ring_V; ring
  rw [h]
  have := ch04_hasDerivAt_quadratic (κ / 2) (-κ) (κ / 2) r
  convert this using 1
  ring

/-- eq:ring-force (ch04-ring.tex lines 69-72):
`F(r) = -V'(r) = -κ (r-1)`.  Both equalities are checked: the derivative of
`ch04_ring_V` is `κ(r-1)`, hence `-V'(r)` really is `-κ(r-1)`. -/
def ch04_ring_F (κ r : ℝ) : ℝ := -κ * (r - 1)

theorem ch04_ring_force (κ r : ℝ) :
    ch04_ring_F κ r = -deriv (ch04_ring_V κ) r ∧ ch04_ring_F κ r = -(κ * (r - 1)) := by
  have h : deriv (ch04_ring_V κ) r = κ * (r - 1) := (ch04_ring_V_hasDerivAt κ r).deriv
  unfold ch04_ring_F
  rw [h]
  constructor <;> ring

/-- The sign statement following eq:ring-force: the force points inward above
the stable radius and outward below it. -/
theorem ch04_ring_force_sign {κ : ℝ} (hκ : 0 < κ) (r : ℝ) :
    (1 < r → ch04_ring_F κ r < 0) ∧ (r < 1 → 0 < ch04_ring_F κ r) := by
  constructor <;> intro h <;> unfold ch04_ring_F <;> nlinarith

/-- eq:ring-radial-sde (ch04-ring.tex lines 29-31):
`dR_u = -κ(R_u-1) du + √(2 D_r) dB_u`.  Recorded as its drift and diffusion
coefficients; the identification of the drift with `-V'` (the overdamped
Langevin form quoted on line 62) is the lemma. -/
def ch04_ring_radial_drift (κ r : ℝ) : ℝ := -κ * (r - 1)

def ch04_ring_radial_diffusion (Dr : ℝ) : ℝ := Real.sqrt (2 * Dr)

theorem ch04_ring_radial_sde_is_langevin (κ r : ℝ) :
    ch04_ring_radial_drift κ r = -deriv (ch04_ring_V κ) r := by
  rw [ch04_ring_radial_drift, (ch04_ring_V_hasDerivAt κ r).deriv]; ring

/-- The stationary law quoted after eq:ring-force is `N(1, D_r/κ)`:
the Boltzmann weight `exp(-V/D_r)` is the Gaussian kernel with mean `1` and
variance `D_r/κ`. -/
theorem ch04_ring_stationary_law {κ Dr : ℝ} (hκ : 0 < κ) (hD : 0 < Dr) (r : ℝ) :
    Real.exp (-(ch04_ring_V κ r) / Dr)
      = Real.exp (-(r - 1) ^ 2 / (2 * (Dr / κ))) := by
  congr 1
  rw [ch04_ring_V]
  field_simp
  try ring

/-- eq:ring-angular-sde (ch04-ring.tex lines 32-33):
`dΘ_u = ω du + √(2 D_θ) dB_u`.  Recorded as its coefficients; the lemma is the
solution of the deterministic part used in the proof of marginal blindness
(`Θ_u = Θ_0 + ω u + √(2D_θ) B_u`). -/
def ch04_ring_angular_drift (ω : ℝ) : ℝ := ω

theorem ch04_ring_angular_sde_solution (Θ₀ ω u : ℝ) :
    HasDerivAt (fun v : ℝ => Θ₀ + ω * v) (ch04_ring_angular_drift ω) u := by
  have := ((hasDerivAt_id u).const_mul ω).const_add Θ₀
  simpa [ch04_ring_angular_drift] using this

/-- eq:ring-embedding (ch04-ring.tex lines 36-39):
`a_u = R_u (cos Θ_u, sin Θ_u) ∈ ℝ²`. -/
def ch04_ring_embed (R θ : ℝ) : Fin 2 → ℝ := ![R * Real.cos θ, R * Real.sin θ]

/-- The embedding has squared length `R²`: `‖a‖ = |R|`, which is the sense in
which `R` is a *signed* radius (lines 93-103). -/
theorem ch04_ring_embedding_norm (R θ : ℝ) :
    (∑ i, ch04_ring_embed R θ i ^ 2) = R ^ 2 := by
  have h := Real.sin_sq_add_cos_sq θ
  simp [ch04_ring_embed, Fin.sum_univ_two]
  nlinarith [h]

/-- The negative-radius reading of line 96: radius `-R` at phase `θ` is the
same plane point as radius `R` at phase `θ + π`. -/
theorem ch04_ring_embedding_signed (R θ : ℝ) :
    ch04_ring_embed (-R) θ = ch04_ring_embed R (θ + Real.pi) := by
  funext i
  fin_cases i <;>
    simp [ch04_ring_embed, Real.cos_add_pi, Real.sin_add_pi] <;> ring

/-- eq:ring-moving-frame (ch04-ring.tex lines 47-51):
`e_r = (cos θ, sin θ)`, `e_θ = (-sin θ, cos θ)`. -/
def ch04_ring_er (θ : ℝ) : Fin 2 → ℝ := ![Real.cos θ, Real.sin θ]

def ch04_ring_eθ (θ : ℝ) : Fin 2 → ℝ := ![-Real.sin θ, Real.cos θ]

/-- eq:ring-rotation (ch04-ring.tex lines 186-190):
`R_ψ = !![cos ψ, -sin ψ; sin ψ, cos ψ]`. -/
def ch04_R (ψ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![Real.cos ψ, -Real.sin ψ; Real.sin ψ, Real.cos ψ]

@[simp] lemma ch04_R_apply_00 (ψ : ℝ) : ch04_R ψ 0 0 = Real.cos ψ := rfl
@[simp] lemma ch04_R_apply_01 (ψ : ℝ) : ch04_R ψ 0 1 = -Real.sin ψ := rfl
@[simp] lemma ch04_R_apply_10 (ψ : ℝ) : ch04_R ψ 1 0 = Real.sin ψ := rfl
@[simp] lemma ch04_R_apply_11 (ψ : ℝ) : ch04_R ψ 1 1 = Real.cos ψ := rfl

/-- The frame of eq:ring-moving-frame is orthonormal, and `e_θ` is `e_r` turned
through a quarter revolution — the three assertions made in the surrounding
text (lines 52-54). -/
theorem ch04_ring_moving_frame (θ : ℝ) :
    (∑ i, ch04_ring_er θ i * ch04_ring_er θ i) = 1 ∧
    (∑ i, ch04_ring_eθ θ i * ch04_ring_eθ θ i) = 1 ∧
    (∑ i, ch04_ring_er θ i * ch04_ring_eθ θ i) = 0 ∧
    ch04_ring_eθ θ = ch04_R (Real.pi / 2) *ᵥ ch04_ring_er θ := by
  have h := Real.sin_sq_add_cos_sq θ
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp [ch04_ring_er, Fin.sum_univ_two]; nlinarith [h]
  · simp [ch04_ring_eθ, Fin.sum_univ_two]; nlinarith [h]
  · simp [ch04_ring_er, ch04_ring_eθ, Fin.sum_univ_two]; ring
  · funext i
    fin_cases i <;>
      simp [ch04_ring_er, ch04_ring_eθ, ch04_R, Matrix.mulVec, dotProduct,
        Fin.sum_univ_two]

/-- The embedding of eq:ring-embedding is `R` times the radial frame vector. -/
theorem ch04_ring_embed_eq_smul (R θ : ℝ) :
    ch04_ring_embed R θ = R • ch04_ring_er θ := by
  funext i; fin_cases i <;> simp [ch04_ring_embed, ch04_ring_er]

/-- The first derivative of the chapter's potential, as a function:
`V' = κ(r-1)`.  (`ch04_ring_V_hasDerivAt` is the pointwise statement.) -/
theorem ch04_ring_V_deriv (κ : ℝ) : deriv (ch04_ring_V κ) = fun r => κ * (r - 1) := by
  funext r
  exact (ch04_ring_V_hasDerivAt κ r).deriv

/-- The second derivative of the chapter's potential: `V'' = κ`, the constant
curvature that supplies the `D_r V''` term of eq:ring-ito-energy. -/
theorem ch04_ring_V_deriv2 (κ : ℝ) : deriv (deriv (ch04_ring_V κ)) = fun _ => κ := by
  rw [ch04_ring_V_deriv]
  funext r
  have h : HasDerivAt (fun y : ℝ => κ * (y - 1)) κ r := by
    simpa using ((hasDerivAt_id r).sub_const 1).const_mul κ
  exact h.deriv

/-- eq:ring-ito-energy (ch04-ring.tex lines 78-83), for an *arbitrary* potential:
for the overdamped Langevin equation `dR = -V'(R) du + √(2D_r) dB` Itô's
coefficients — drift `V' b + s² V''/2` and diffusion `s V'` — are exactly the two
displayed ones, `-V'² + D_r V''` and `√(2D_r) V'`.

Nothing here is a free parameter: `V'` and `V''` are `deriv V` and
`deriv (deriv V)` of one and the same arbitrary `V : ℝ → ℝ`, the drift `b` is
minus the first of them (which is what eq:ring-radial-sde and eq:ring-force say),
and `s` is `ch04_ring_radial_diffusion D_r`.  Itô's formula itself is *assumed*
(mathlib has no Itô formula in this form); what is proved is the chapter's
algebra from the generic Itô coefficients to the display. -/
theorem ch04_ring_ito_energy {Dr : ℝ} (hD : 0 ≤ Dr) (V : ℝ → ℝ) (r : ℝ) :
    deriv V r * (-(deriv V r))
        + (ch04_ring_radial_diffusion Dr) ^ 2 / 2 * deriv (deriv V) r
      = -(deriv V r) ^ 2 + Dr * deriv (deriv V) r
    ∧ ch04_ring_radial_diffusion Dr * deriv V r = Real.sqrt (2 * Dr) * deriv V r := by
  have hs : (ch04_ring_radial_diffusion Dr) ^ 2 = 2 * Dr := by
    rw [ch04_ring_radial_diffusion, Real.sq_sqrt (by linarith)]
  refine ⟨by rw [hs]; ring, ?_⟩
  rw [ch04_ring_radial_diffusion]

/-- The same display at the chapter's own potential, where nothing is left open:
`V'` and `V''` are computed from `ch04_ring_V` (they are `κ(r-1)` and `κ`), the
drift is the `ch04_ring_radial_drift` of eq:ring-radial-sde and really equals
`-V'`, and the two Itô coefficients of `V(R_u)` are `-κ²(r-1)² + D_r κ` and
`√(2D_r) κ(r-1)`.  General in `κ`, `D_r` and `r`. -/
theorem ch04_ring_ito_energy_quadratic {Dr : ℝ} (hD : 0 ≤ Dr) (κ r : ℝ) :
    deriv (ch04_ring_V κ) r = κ * (r - 1)
    ∧ deriv (deriv (ch04_ring_V κ)) r = κ
    ∧ ch04_ring_radial_drift κ r = -deriv (ch04_ring_V κ) r
    ∧ deriv (ch04_ring_V κ) r * ch04_ring_radial_drift κ r
        + (ch04_ring_radial_diffusion Dr) ^ 2 / 2 * deriv (deriv (ch04_ring_V κ)) r
      = -(κ * (r - 1)) ^ 2 + Dr * κ
    ∧ ch04_ring_radial_diffusion Dr * deriv (ch04_ring_V κ) r
      = Real.sqrt (2 * Dr) * (κ * (r - 1)) := by
  have h1 : deriv (ch04_ring_V κ) r = κ * (r - 1) := by simp only [ch04_ring_V_deriv]
  have h2 : deriv (deriv (ch04_ring_V κ)) r = κ := by simp only [ch04_ring_V_deriv2]
  have hs : (ch04_ring_radial_diffusion Dr) ^ 2 = 2 * Dr := by
    rw [ch04_ring_radial_diffusion, Real.sq_sqrt (by linarith)]
  refine ⟨h1, h2, ?_, ?_, ?_⟩
  · rw [h1, ch04_ring_radial_drift]; ring
  · rw [h1, h2, hs, ch04_ring_radial_drift]; ring
  · rw [h1, ch04_ring_radial_diffusion]

/-- The deterministic descent statement of lines 73-75:
along `ṙ = -V'(r)` the energy decreases, `d/du V(r_u) = -V'(r)² ≤ 0`. -/
theorem ch04_ring_energy_descent (V' : ℝ) : V' * (-V') = -V' ^ 2 ∧ -V' ^ 2 ≤ 0 := by
  constructor
  · ring
  · nlinarith [sq_nonneg V']

/-- eq:ring-datum (ch04-ring.tex lines 109-112):
`a = (a_0, …, a_{L-1}) ∈ ℝ^{2L}`, i.e. the stacked datum has `2L` real
coordinates. -/
def ch04_ring_datum (L : ℕ) : Type := Fin L × Fin 2

theorem ch04_ring_datum_dim (L : ℕ) : Fintype.card (Fin L × Fin 2) = 2 * L := by
  simp [mul_comm]

/-- eq:ring-channel (ch04-ring.tex lines 117-120):
`x = e^{-t} a + √Δ_t ξ`, `ξ ∼ N(0, I_{2L})`. -/
def ch04_ring_channel {ι : Type*} (t : ℝ) (a ξ : ι → ℝ) : ι → ℝ :=
  fun i => Real.exp (-t) * a i + Real.sqrt (ch04_Delta t) * ξ i

/-- The channel's residual is `√Δ_t ξ`, so given `a` the noise has variance
`Δ_t` in each coordinate, and the channel is variance preserving:
`e^{-2t} + Δ_t = 1`. -/
theorem ch04_ring_channel_moments {ι : Type*} {t : ℝ} (ht : 0 ≤ t) (a ξ : ι → ℝ) (i : ι) :
    ch04_ring_channel t a ξ i - Real.exp (-t) * a i = Real.sqrt (ch04_Delta t) * ξ i ∧
    (Real.sqrt (ch04_Delta t)) ^ 2 = ch04_Delta t ∧
    (Real.exp (-t)) ^ 2 + ch04_Delta t = 1 := by
  refine ⟨by simp [ch04_ring_channel], Real.sq_sqrt (ch04_Delta_nonneg ht), ?_⟩
  have h2 : (Real.exp (-t)) ^ 2 = Real.exp (-2 * t) := by
    rw [← Real.exp_nat_mul]
    norm_num
  rw [h2, ch04_Delta]; ring

/-! ### §4.2  The gauge, and the exact score -/

/-- eq:ring-sur-prior (ch04-ring.tex lines 162-165):
`p_λ(z) ∝ exp[-(‖z‖-1)²/(2λ)]`, the ring prior for `z_0`. -/
def ch04_ring_prior (lam : ℝ) (z : Fin 2 → ℝ) : ℝ :=
  Real.exp (-((Real.sqrt (∑ i, z i ^ 2) - 1) ^ 2) / (2 * lam))

theorem ch04_ring_prior_pos (lam : ℝ) (z : Fin 2 → ℝ) : 0 < ch04_ring_prior lam z :=
  Real.exp_pos _

/-- `R_ψ` preserves squared length: `‖R_ψ z‖² = ‖z‖²`, the statement quoted on
line 200. -/
theorem ch04_R_preserves_normSq (ψ : ℝ) (z : Fin 2 → ℝ) :
    (∑ i, (ch04_R ψ *ᵥ z) i ^ 2) = ∑ i, z i ^ 2 := by
  have h := Real.sin_sq_add_cos_sq ψ
  simp [ch04_R, Matrix.mulVec, dotProduct, Fin.sum_univ_two]
  nlinarith [h, sq_nonneg (z 0), sq_nonneg (z 1)]

/-- The ring prior is rotationally invariant — the statement used in the proof
of Theorem~\ref{thm:ring-marginal-blindness} (lines 531-534). -/
theorem ch04_ring_prior_rotation_invariant (lam ψ : ℝ) (z : Fin 2 → ℝ) :
    ch04_ring_prior lam (ch04_R ψ *ᵥ z) = ch04_ring_prior lam z := by
  unfold ch04_ring_prior
  rw [ch04_R_preserves_normSq]

/-- eq:ring-sur-dynamics (ch04-ring.tex lines 166-168):
`z_{k+1} = R_ψ z_k + σ η_{k+1}`. -/
def ch04_ring_step (ψ σ : ℝ) (z η : Fin 2 → ℝ) : Fin 2 → ℝ :=
  ch04_R ψ *ᵥ z + σ • η

/-- Without innovation the surrogate dynamics stays on its circle. -/
theorem ch04_ring_step_noiseless (ψ : ℝ) (z η : Fin 2 → ℝ) :
    (∑ i, ch04_ring_step ψ 0 z η i ^ 2) = ∑ i, z i ^ 2 := by
  have : ch04_ring_step ψ 0 z η = ch04_R ψ *ᵥ z := by
    funext i; simp [ch04_ring_step]
  rw [this, ch04_R_preserves_normSq]

/-- eq:ring-rot-props (ch04-ring.tex lines 194-198), first claim:
`R_ψᵀ R_ψ = I₂`. -/
theorem ch04_ring_rot_props_orthogonal (ψ : ℝ) : (ch04_R ψ)ᵀ * ch04_R ψ = 1 := by
  have h := Real.sin_sq_add_cos_sq ψ
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [ch04_R, Matrix.mul_apply, Fin.sum_univ_two, Matrix.one_apply] <;>
    nlinarith [h]

/-- eq:ring-rot-props, second claim: `det R_ψ = 1`. -/
theorem ch04_ring_rot_props_det (ψ : ℝ) : (ch04_R ψ).det = 1 := by
  have h := Real.sin_sq_add_cos_sq ψ
  rw [ch04_R, Matrix.det_fin_two_of]
  nlinarith [h]

/-- eq:ring-rot-props, third claim: `R_ψᵀ = R_{-ψ}`. -/
theorem ch04_ring_rot_props_transpose (ψ : ℝ) : (ch04_R ψ)ᵀ = ch04_R (-ψ) := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [ch04_R, Real.cos_neg, Real.sin_neg]

/-- eq:ring-rot-props, fourth claim: `R_ψ R_φ = R_{ψ+φ}`. -/
theorem ch04_ring_rot_props_comp (ψ φ : ℝ) : ch04_R ψ * ch04_R φ = ch04_R (ψ + φ) := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [ch04_R, Matrix.mul_apply, Fin.sum_univ_two, Real.cos_add, Real.sin_add] <;>
    ring

/-- The composition law transported to the action on vectors. -/
theorem ch04_R_mulVec_comp (ψ φ : ℝ) (z : Fin 2 → ℝ) :
    ch04_R ψ *ᵥ (ch04_R φ *ᵥ z) = ch04_R (ψ + φ) *ᵥ z := by
  rw [Matrix.mulVec_mulVec, ch04_ring_rot_props_comp]

/-- noname-1 (ch04-ring.tex lines 245-251), the computation in the proof of
Lemma~\ref{lem:ring-gauge}:
`R_{-(k+1)ψ}(R_ψ z_k + σ η) = R_{-kψ} z_k + σ R_{-(k+1)ψ} η`, the underbraced
step being `R_{-(k+1)ψ} R_ψ = R_{-kψ}`. -/
theorem ch04_ring_gauge_step (ψ σ : ℝ) (k : ℕ) (zk η : Fin 2 → ℝ) :
    ch04_R (-((k : ℝ) + 1) * ψ) *ᵥ (ch04_ring_step ψ σ zk η)
      = ch04_R (-(k : ℝ) * ψ) *ᵥ zk + σ • (ch04_R (-((k : ℝ) + 1) * ψ) *ᵥ η) := by
  have hcomp : ch04_R (-((k : ℝ) + 1) * ψ) * ch04_R ψ = ch04_R (-(k : ℝ) * ψ) := by
    rw [ch04_ring_rot_props_comp]
    congr 1
    ring
  unfold ch04_ring_step
  rw [Matrix.mulVec_add, Matrix.mulVec_smul, Matrix.mulVec_mulVec, hcomp]

/-- eq:ring-gauged-rw (ch04-ring.tex lines 227-231):
with `y_k = R_{-kψ} z_k` the de-rotated chain is the plain random walk
`y_{k+1} = y_k + σ η̃_{k+1}`, `η̃_{k+1} = R_{-(k+1)ψ} η_{k+1}`. -/
theorem ch04_ring_gauged_rw (ψ σ : ℝ) (k : ℕ) (zk η : Fin 2 → ℝ) :
    ch04_R (-((k : ℝ) + 1) * ψ) *ᵥ (ch04_ring_step ψ σ zk η)
      = (ch04_R (-(k : ℝ) * ψ) *ᵥ zk) + σ • (ch04_R (-((k : ℝ) + 1) * ψ) *ᵥ η) :=
  ch04_ring_gauge_step ψ σ k zk η

/-- Lemma~\ref{lem:ring-isotropy} (lines 203-216) at the level of the two
moments it is proved by: a rotation kills no mass (`R·0 = 0`) and preserves the
identity covariance (`R I₂ Rᵀ = I₂`). -/
theorem ch04_ring_isotropy (ψ : ℝ) :
    ch04_R ψ *ᵥ (0 : Fin 2 → ℝ) = 0 ∧ ch04_R ψ * (1 : Matrix (Fin 2) (Fin 2) ℝ) * (ch04_R ψ)ᵀ = 1 := by
  refine ⟨by simp, ?_⟩
  have h := Real.sin_sq_add_cos_sq ψ
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [ch04_R, Matrix.mul_apply, Fin.sum_univ_two, Matrix.one_apply] <;>
    nlinarith [h]

/-- The per-frame gauge `U_ψ = diag(I₂, R_{-ψ}, …, R_{-(L-1)ψ})` of
Lemma~\ref{lem:ring-gauge} (lines 224-226), as a block-diagonal matrix on
`Fin 2 × Fin L` (mathlib's `blockDiagonal` indexes as inner × outer). -/
def ch04_ring_U (L : ℕ) (ψ : ℝ) : Matrix (Fin 2 × Fin L) (Fin 2 × Fin L) ℝ :=
  Matrix.blockDiagonal (fun k : Fin L => ch04_R (-(k : ℝ) * ψ))

/-- `U_ψ` is orthogonal with unit determinant (lines 254-257). -/
theorem ch04_ring_U_orthogonal (L : ℕ) (ψ : ℝ) :
    (ch04_ring_U L ψ)ᵀ * ch04_ring_U L ψ = 1 ∧ (ch04_ring_U L ψ).det = 1 := by
  constructor
  · rw [ch04_ring_U, Matrix.blockDiagonal_transpose, ← Matrix.blockDiagonal_mul]
    have hone : (fun k : Fin L => (ch04_R (-(k : ℝ) * ψ))ᵀ * ch04_R (-(k : ℝ) * ψ))
        = (1 : Fin L → Matrix (Fin 2) (Fin 2) ℝ) := by
      funext k; exact ch04_ring_rot_props_orthogonal _
    rw [hone, Matrix.blockDiagonal_one]
  · rw [ch04_ring_U, Matrix.det_blockDiagonal]
    simp [ch04_ring_rot_props_det]

/-- noname-2 (ch04-ring.tex lines 266-271), the chain-rule bookkeeping in the
proof of Lemma~\ref{lem:ring-gauge}:
`∑_b U_{bc} v_b = (Uᵀ v)_c`, i.e. a gradient transforms with `Uᵀ` while a point
transforms with `U`. -/
theorem ch04_ring_gauge_chain_rule {n : ℕ} (U : Matrix (Fin n) (Fin n) ℝ) (v : Fin n → ℝ)
    (c : Fin n) : (∑ b, U b c * v b) = (Uᵀ *ᵥ v) c := by
  simp [Matrix.mulVec, dotProduct, Matrix.transpose_apply]

/-! ### Linear algebra of Toolbox 4.1: `C ⊗ I₂`, quadratic forms, Gaussian scores -/

/-- The action of `C ⊗ I₂` on a block vector: `[(C ⊗ I₂) x]_k = ∑_j C_{kj} x_j`. -/
def ch04_blockMulVec {L : ℕ} (C : Matrix (Fin L) (Fin L) ℝ) (x : Fin L → Fin 2 → ℝ) :
    Fin L → Fin 2 → ℝ := fun k i => ∑ j, C k j * x j i

/-- noname-3 (ch04-ring.tex lines 359-363): the chapter's block reading of
`C_t^{-1}` agrees with the honest Kronecker product `C ⊗ I₂` acting on the
flattened `2L`-vector, for every `L`. -/
theorem ch04_ring_block_action {L : ℕ} (C : Matrix (Fin L) (Fin L) ℝ)
    (x : Fin L → Fin 2 → ℝ) (k : Fin L) (i : Fin 2) :
    ((C ⊗ₖ (1 : Matrix (Fin 2) (Fin 2) ℝ)) *ᵥ fun p : Fin L × Fin 2 => x p.1 p.2) (k, i)
      = ch04_blockMulVec C x k i := by
  simp only [Matrix.mulVec, dotProduct, Fintype.sum_prod_type, Matrix.kronecker_apply,
    Matrix.one_apply, ch04_blockMulVec]
  refine Finset.sum_congr rfl fun j _ => ?_
  simp [mul_ite, ite_mul]

/-- `Q_A(x) = xᵀ A x`. -/
def ch04_quad {ι : Type*} [Fintype ι] (A : Matrix ι ι ℝ) (x : ι → ℝ) : ℝ := x ⬝ᵥ (A *ᵥ x)

lemma ch04_symm_apply {ι : Type*} {A : Matrix ι ι ℝ} (hA : Aᵀ = A) (a b : ι) : A b a = A a b := by
  have h := congrFun (congrFun hA a) b
  simpa [Matrix.transpose_apply] using h

lemma ch04_dot_symm {ι : Type*} [Fintype ι] {A : Matrix ι ι ℝ} (hA : Aᵀ = A) (x y : ι → ℝ) :
    y ⬝ᵥ (A *ᵥ x) = x ⬝ᵥ (A *ᵥ y) := by
  simp only [dotProduct, Matrix.mulVec, Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun b _ => ?_
  rw [ch04_symm_apply hA a b]
  ring

lemma ch04_dot_transpose {ι : Type*} [Fintype ι] (N : Matrix ι ι ℝ) (x y : ι → ℝ) :
    x ⬝ᵥ (Nᵀ *ᵥ y) = y ⬝ᵥ (N *ᵥ x) := by
  simp only [dotProduct, Matrix.mulVec, Matrix.transpose_apply, Finset.mul_sum]
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun r _ => Finset.sum_congr rfl fun i _ => by ring

/-- Polarisation of a symmetric quadratic form. -/
theorem ch04_quad_sub {ι : Type*} [Fintype ι] {A : Matrix ι ι ℝ} (hA : Aᵀ = A) (x y : ι → ℝ) :
    ch04_quad A (x - y) = ch04_quad A x - 2 * (x ⬝ᵥ (A *ᵥ y)) + ch04_quad A y := by
  have hs := ch04_dot_symm hA x y
  simp only [ch04_quad, Matrix.mulVec_sub, sub_dotProduct, dotProduct_sub]
  rw [hs]
  ring

/-- Expansion of `Q_A` along a single coordinate direction. -/
theorem ch04_quad_single {ι : Type*} [Fintype ι] [DecidableEq ι] (A : Matrix ι ι ℝ)
    (x : ι → ℝ) (c : ι) (u : ℝ) :
    ch04_quad A (x + Pi.single c u)
      = ch04_quad A x + ((x ⬝ᵥ fun a => A a c) + (A *ᵥ x) c) * u + A c c * u ^ 2 := by
  have h1 : A *ᵥ (Pi.single c u) = fun a => A a c * u := by
    funext a
    simp [Matrix.mulVec]
  have h2 : (x ⬝ᵥ fun a => A a c * u) = (x ⬝ᵥ fun a => A a c) * u := by
    simp only [dotProduct, Finset.sum_mul]
    exact Finset.sum_congr rfl fun a _ => by ring
  simp only [ch04_quad, Matrix.mulVec_add, h1, add_dotProduct, dotProduct_add,
    single_dotProduct, h2]
  ring

lemma ch04_smul_one_mulVec {ι : Type*} [Fintype ι] [DecidableEq ι] (a : ℝ) (v : ι → ℝ)
    (c : ι) : ((a • (1 : Matrix ι ι ℝ)) *ᵥ v) c = a * v c := by
  rw [Matrix.smul_mulVec, Matrix.one_mulVec]
  simp

/-- The Gaussian log-density kernel with precision `A` and mean `μ`
(the additive normalising constant is dropped: it does not depend on `x`). -/
def ch04_logGauss {ι : Type*} [Fintype ι] (A : Matrix ι ι ℝ) (μ x : ι → ℝ) : ℝ :=
  -(1/2) * ch04_quad A (x - μ)

/-- The partial derivative of a Gaussian log-density: `∂_c log p = -(A(x-μ))_c`.
This is the calculus behind every score display of the chapter. -/
theorem ch04_hasDerivAt_logGauss {ι : Type*} [Fintype ι] [DecidableEq ι]
    {A : Matrix ι ι ℝ} (hA : Aᵀ = A) (μ x : ι → ℝ) (c : ι) :
    HasDerivAt (fun s => ch04_logGauss A μ (Function.update x c s))
      (-((A *ᵥ (x - μ)) c)) (x c) := by
  have hupd : ∀ s : ℝ, Function.update x c s - μ = (x - μ) + Pi.single c (s - x c) := by
    intro s
    funext a
    by_cases h : a = c
    · subst h
      simp [Function.update_apply, Pi.single_apply]
    · simp [Function.update_apply, Pi.single_apply, h]
  have hK : ((x - μ) ⬝ᵥ fun a => A a c) = (A *ᵥ (x - μ)) c := by
    simp only [dotProduct, Matrix.mulVec]
    exact Finset.sum_congr rfl fun a _ => by rw [ch04_symm_apply hA c a]; ring
  have hfun : (fun s => ch04_logGauss A μ (Function.update x c s))
      = fun s => (-(1/2) * A c c) * s ^ 2
          + ((-((A *ᵥ (x - μ)) c)) - 2 * (-(1/2) * A c c) * (x c)) * s
          + ((-(1/2) * A c c) * (x c) ^ 2 - (-((A *ᵥ (x - μ)) c)) * (x c)
              + (-(1/2) * ch04_quad A (x - μ))) := by
    funext s
    simp only [ch04_logGauss, hupd s, ch04_quad_single, hK]
    ring
  rw [hfun]
  have hq := ch04_hasDerivAt_quadratic (-(1/2) * A c c)
      ((-((A *ᵥ (x - μ)) c)) - 2 * (-(1/2) * A c c) * (x c))
      ((-(1/2) * A c c) * (x c) ^ 2 - (-((A *ᵥ (x - μ)) c)) * (x c)
        + (-(1/2) * ch04_quad A (x - μ))) (x c)
  convert hq using 1
  ring

/-! ### §4.2 (Toolbox 4.1)  `C_t`, the anchor statistic, and the exact score -/

/-- eq:ring-Ct-def (ch04-ring.tex lines 330-336), walk part: `M_{ij} = min(i,j)`,
zero-based. -/
def ch04_M (L : ℕ) : Matrix (Fin L) (Fin L) ℝ := fun i j => (min (i : ℕ) (j : ℕ) : ℝ)

/-- eq:ring-Ct-def: `C_t = e^{-2t} σ² M + Δ_t I_L`. -/
def ch04_Ct (L : ℕ) (σ t : ℝ) : Matrix (Fin L) (Fin L) ℝ :=
  (Real.exp (-2 * t) * σ ^ 2) • ch04_M L + (ch04_Delta t) • (1 : Matrix (Fin L) (Fin L) ℝ)

/-- The indicator matrix of "increment `r` has already happened by frame `i`". -/
def ch04_N (L : ℕ) : Matrix (Fin L) (Fin L) ℝ :=
  fun r i => if (r : ℕ) < (i : ℕ) then 1 else 0

lemma ch04_sum_lt_indicator (L : ℕ) : ∀ m : ℕ, m ≤ L →
    (∑ r ∈ Finset.range L, (if r < m then (1:ℝ) else 0)) = (m : ℝ) := by
  induction L with
  | zero =>
      intro m hm
      have : m = 0 := Nat.le_zero.mp hm
      subst this
      simp
  | succ L ih =>
      intro m hm
      rw [Finset.sum_range_succ]
      rcases Nat.lt_or_ge L m with hL | hL
      · have hm' : m = L + 1 := by omega
        subst hm'
        have hall : ∀ r ∈ Finset.range L, (if r < L + 1 then (1:ℝ) else 0) = 1 := by
          intro r hr
          rw [Finset.mem_range] at hr
          simp [Nat.lt_succ_of_lt hr]
        rw [Finset.sum_congr rfl hall]
        simp
      · rw [ih m hL]
        simp [Nat.not_lt.mpr hL]

/-- The anchored-walk covariance quoted before eq:ring-Ct-def (lines 321-327):
`y_k = A + σ ∑_{r ≤ k} η̃_r`, so frames `i` and `j` share exactly `min(i,j)`
increments and `Cov(y_i, y_j | A) = σ² min(i,j) I₂`; the first frame is pinned
to the anchor, `min(0,j) = 0`. -/
theorem ch04_ring_walk_covariance (L : ℕ) (i j : Fin L) :
    (∑ r : Fin L, (if (r : ℕ) < (i : ℕ) then (1:ℝ) else 0) *
        (if (r : ℕ) < (j : ℕ) then (1:ℝ) else 0)) = ch04_M L i j := by
  have hcollapse : ∀ r : Fin L,
      (if (r : ℕ) < (i : ℕ) then (1:ℝ) else 0) * (if (r : ℕ) < (j : ℕ) then (1:ℝ) else 0)
        = (if (r : ℕ) < min (i : ℕ) (j : ℕ) then (1:ℝ) else 0) := by
    intro r
    by_cases hi : (r : ℕ) < (i : ℕ) <;> by_cases hj : (r : ℕ) < (j : ℕ) <;>
      simp [hi, hj, lt_min_iff]
  rw [Finset.sum_congr rfl fun r _ => hcollapse r]
  rw [Fin.sum_univ_eq_sum_range (fun r => if r < min (i : ℕ) (j : ℕ) then (1:ℝ) else 0) L]
  rw [ch04_sum_lt_indicator L (min (i : ℕ) (j : ℕ))
    (le_of_lt (lt_of_le_of_lt (min_le_left _ _) i.isLt))]
  simp [ch04_M]

/-- `M` is the Gram matrix `NᵀN` of those indicators. -/
theorem ch04_ring_M_gram (L : ℕ) : (ch04_N L)ᵀ * ch04_N L = ch04_M L := by
  ext i j
  rw [Matrix.mul_apply]
  simpa [ch04_N, Matrix.transpose_apply] using ch04_ring_walk_covariance L i j

/-- Hence `M` is positive semidefinite. -/
theorem ch04_ring_M_psd (L : ℕ) (x : Fin L → ℝ) : 0 ≤ x ⬝ᵥ (ch04_M L *ᵥ x) := by
  rw [← ch04_ring_M_gram, ← Matrix.mulVec_mulVec, ch04_dot_transpose]
  simp only [dotProduct]
  exact Finset.sum_nonneg fun i _ => mul_self_nonneg _

/-- `M` is singular: its zeroth row vanishes (the first frame is pinned to the
anchor), exactly as stated after eq:ring-Ct-def. -/
theorem ch04_ring_M_singular (L : ℕ) (hL : 0 < L) : (ch04_M L).det = 0 := by
  refine Matrix.det_eq_zero_of_row_eq_zero (⟨0, hL⟩ : Fin L) ?_
  intro j
  simp [ch04_M]

/-- eq:ring-Ct-def, the positivity claim: `C_t` is positive definite for every
`t > 0` — thanks to the `Δ_t I_L` term — even though `M` is singular. -/
theorem ch04_ring_Ct_posdef {L : ℕ} (σ : ℝ) {t : ℝ} (ht : 0 < t) (x : Fin L → ℝ)
    (hx : x ≠ 0) : 0 < x ⬝ᵥ (ch04_Ct L σ t *ᵥ x) := by
  have hM : 0 ≤ x ⬝ᵥ (ch04_M L *ᵥ x) := ch04_ring_M_psd L x
  have hc : 0 ≤ Real.exp (-2 * t) * σ ^ 2 := by positivity
  have hD : 0 < ch04_Delta t := ch04_Delta_pos ht
  have hxx : 0 < x ⬝ᵥ x := by
    obtain ⟨i, hi⟩ := Function.ne_iff.mp hx
    have hne : x i ≠ 0 := by simpa using hi
    simp only [dotProduct]
    refine Finset.sum_pos' (fun j _ => mul_self_nonneg (x j)) ⟨i, Finset.mem_univ i, ?_⟩
    exact mul_self_pos.mpr hne
  have hexp : x ⬝ᵥ (ch04_Ct L σ t *ᵥ x)
      = (Real.exp (-2 * t) * σ ^ 2) * (x ⬝ᵥ (ch04_M L *ᵥ x)) + (ch04_Delta t) * (x ⬝ᵥ x) := by
    rw [ch04_Ct, Matrix.add_mulVec, dotProduct_add, Matrix.smul_mulVec, Matrix.smul_mulVec,
      Matrix.one_mulVec, dotProduct_smul, dotProduct_smul, smul_eq_mul, smul_eq_mul]
  rw [hexp]
  nlinarith [mul_nonneg hc hM, mul_pos hD hxx]

/-- `M` is symmetric, because `min` is. -/
theorem ch04_M_symm (L : ℕ) : (ch04_M L)ᵀ = ch04_M L := by
  ext i j
  simp only [ch04_M, Matrix.transpose_apply]
  first
    | (rw [min_comm]; done)
    | (congr 1; exact min_comm _ _)
    | (push_cast; exact min_comm _ _)
    | simp [min_comm]

/-- `C_t` is symmetric — so the toolbox's standing hypothesis `Pᵀ = P` is really
satisfied by the chapter's matrix. -/
theorem ch04_Ct_symm (L : ℕ) (σ t : ℝ) : (ch04_Ct L σ t)ᵀ = ch04_Ct L σ t := by
  rw [ch04_Ct, Matrix.transpose_add, Matrix.transpose_smul, Matrix.transpose_smul,
    Matrix.transpose_one, ch04_M_symm]

/-- The coefficient matrix `A = [e^{-t}σ Nᵀ | √Δ_t I_L]` of the noised
de-rotated trajectory, seen as an affine map of the white sources
`(η̃_1,…,η̃_L, ξ_1,…,ξ_L)`. -/
def ch04_ring_channelCoeff (L : ℕ) (σ t : ℝ) : Matrix (Fin L) (Fin L ⊕ Fin L) ℝ :=
  Matrix.of fun i => Sum.elim
    (fun r : Fin L => Real.exp (-t) * σ * ch04_N L r i)
    (fun m : Fin L => Real.sqrt (ch04_Delta t) * (if i = m then 1 else 0))

/-- eq:ring-conditional-noised-law (lines 340-348), step one: the noised
de-rotated trajectory really is that affine map.  Frame `i` of the anchored walk
of eq:ring-gauged-rw is `y_i = α + σ ∑_{r<i} η̃_r`; pushing it through the
channel of eq:ring-channel gives `e^{-t}α + (A (η̃,ξ))_i`, one plane coordinate
at a time.  General in `L`, `σ`, `t`, `α` and the sources. -/
theorem ch04_ring_channel_is_linear (L : ℕ) (σ t α : ℝ) (η ξ : Fin L → ℝ) (i : Fin L) :
    Real.exp (-t) * (α + σ * ∑ r, ch04_N L r i * η r) + Real.sqrt (ch04_Delta t) * ξ i
      = Real.exp (-t) * α + (ch04_ring_channelCoeff L σ t *ᵥ Sum.elim η ξ) i := by
  rw [Matrix.mulVec, dotProduct, Fintype.sum_sum_type]
  simp only [ch04_ring_channelCoeff, Matrix.of_apply, Sum.elim_inl, Sum.elim_inr]
  have h1 : ∑ r : Fin L, (Real.exp (-t) * σ * ch04_N L r i) * η r
      = Real.exp (-t) * σ * ∑ r, ch04_N L r i * η r := by
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun r _ => by ring
  have h2 : ∑ m : Fin L, (Real.sqrt (ch04_Delta t) * (if i = m then (1:ℝ) else 0)) * ξ m
      = Real.sqrt (ch04_Delta t) * ξ i := by
    rw [Finset.sum_eq_single i]
    · simp
    · intro m _ hm
      simp [hm.symm]
    · intro h
      exact absurd (Finset.mem_univ i) h
  rw [h1, h2]
  ring

/-- eq:ring-conditional-noised-law, step two — the covariance the display
asserts, which is the step eq:ring-Ct-def actually makes.  The covariance of an
affine image of white sources is the Gram matrix `A Aᵀ` of its coefficients, and
here that Gram matrix is exactly `C_t = e^{-2t}σ²M + Δ_t I_L`: passing the
anchored walk's `σ² min(i,j)` through the channel scales it by `e^{-2t}` and adds
`Δ_t I`.  General in `L`, `σ`, and every `t ≥ 0`. -/
theorem ch04_ring_channel_covariance (L : ℕ) (σ : ℝ) {t : ℝ} (ht : 0 ≤ t) :
    ch04_ring_channelCoeff L σ t * (ch04_ring_channelCoeff L σ t)ᵀ = ch04_Ct L σ t := by
  have hexp : Real.exp (-t) * Real.exp (-t) = Real.exp (-2 * t) := by
    rw [← Real.exp_add]; congr 1; ring
  have hsq : Real.sqrt (ch04_Delta t) * Real.sqrt (ch04_Delta t) = ch04_Delta t :=
    Real.mul_self_sqrt (ch04_Delta_nonneg ht)
  ext i j
  rw [Matrix.mul_apply, Fintype.sum_sum_type]
  simp only [ch04_ring_channelCoeff, Matrix.of_apply, Matrix.transpose_apply,
    Sum.elim_inl, Sum.elim_inr]
  have hwalk : ∑ r : Fin L,
      (Real.exp (-t) * σ * ch04_N L r i) * (Real.exp (-t) * σ * ch04_N L r j)
      = (Real.exp (-2 * t) * σ ^ 2) * ch04_M L i j := by
    rw [← ch04_ring_walk_covariance L i j, Finset.mul_sum]
    refine Finset.sum_congr rfl fun r _ => ?_
    simp only [ch04_N]
    rw [← hexp]
    ring
  have hnoise : ∑ m : Fin L,
      (Real.sqrt (ch04_Delta t) * (if i = m then (1:ℝ) else 0)) *
        (Real.sqrt (ch04_Delta t) * (if j = m then (1:ℝ) else 0))
      = ch04_Delta t * (1 : Matrix (Fin L) (Fin L) ℝ) i j := by
    rw [Finset.sum_eq_single i]
    · by_cases hij : i = j
      · subst hij
        simpa [Matrix.one_apply] using hsq
      · simp [hij, Ne.symm hij, Matrix.one_apply]
    · intro m _ hm
      simp [hm.symm]
    · intro h
      exact absurd (Finset.mem_univ i) h
  rw [hwalk, hnoise, ch04_Ct]
  simp [Matrix.add_apply, Matrix.smul_apply, smul_eq_mul]

/-- eq:ring-Ct-def's positivity claim, in the form the toolbox needs: for every
`t > 0` the matrix `C_t` is invertible. -/
theorem ch04_ring_Ct_det_ne_zero {L : ℕ} (σ : ℝ) {t : ℝ} (ht : 0 < t) :
    (ch04_Ct L σ t).det ≠ 0 := by
  intro hdet
  obtain ⟨u, hu, hu0⟩ := Matrix.exists_mulVec_eq_zero_iff.mpr hdet
  have hpos := ch04_ring_Ct_posdef σ ht u hu
  rw [hu0] at hpos
  simp at hpos

theorem ch04_ring_Ct_isUnit_det {L : ℕ} (σ : ℝ) {t : ℝ} (ht : 0 < t) :
    IsUnit (ch04_Ct L σ t).det :=
  isUnit_iff_ne_zero.mpr (ch04_ring_Ct_det_ne_zero σ ht)

theorem ch04_ring_Ct_mul_inv {L : ℕ} (σ : ℝ) {t : ℝ} (ht : 0 < t) :
    ch04_Ct L σ t * (ch04_Ct L σ t)⁻¹ = 1 :=
  Matrix.mul_nonsing_inv _ (ch04_ring_Ct_isUnit_det σ ht)

/-- `C_t^{-1}` is symmetric, so the chapter's precision matrix satisfies the
hypothesis `Pᵀ = P` carried through the whole toolbox. -/
theorem ch04_ring_Ct_inv_symm {L : ℕ} (σ t : ℝ) :
    ((ch04_Ct L σ t)⁻¹)ᵀ = (ch04_Ct L σ t)⁻¹ := by
  rw [Matrix.transpose_nonsing_inv, ch04_Ct_symm]

/-- eq:ring-anchor-statistic (lines 350-357), first half: `q = C_t^{-1} 1`.
Throughout the toolbox the chapter's `C_t^{-1}` is carried as a symmetric
matrix `P`: no display of the toolbox uses anything about `P` beyond its
symmetry and `q = P 1`. -/
def ch04_ring_q {L : ℕ} (P : Matrix (Fin L) (Fin L) ℝ) : Fin L → ℝ := P *ᵥ (fun _ => 1)

lemma ch04_ring_q_apply {L : ℕ} (P : Matrix (Fin L) (Fin L) ℝ) (k : Fin L) :
    ch04_ring_q P k = ∑ j, P k j := by
  simp [ch04_ring_q, Matrix.mulVec, dotProduct]

/-- The flattened trajectory index: `L` frames of two coordinates. -/
abbrev ch04_Ix (L : ℕ) := Fin L × Fin 2

/-- `P ⊗ I₂` on the `2L` coordinates. -/
def ch04_kron2 {L : ℕ} (P : Matrix (Fin L) (Fin L) ℝ) : Matrix (ch04_Ix L) (ch04_Ix L) ℝ :=
  P ⊗ₖ (1 : Matrix (Fin 2) (Fin 2) ℝ)

lemma ch04_kron2_symm {L : ℕ} {P : Matrix (Fin L) (Fin L) ℝ} (hP : Pᵀ = P) :
    (ch04_kron2 P)ᵀ = ch04_kron2 P := by
  ext p q
  obtain ⟨k, i⟩ := p
  obtain ⟨j, i'⟩ := q
  simp only [ch04_kron2, Matrix.transpose_apply, Matrix.kronecker_apply, Matrix.one_apply]
  rw [ch04_symm_apply hP k j]
  by_cases h : i = i'
  · simp [h]
  · simp [h, Ne.symm h]

lemma ch04_kron2_mulVec {L : ℕ} (P : Matrix (Fin L) (Fin L) ℝ) (x : ch04_Ix L → ℝ)
    (k : Fin L) (i : Fin 2) :
    (ch04_kron2 P *ᵥ x) (k, i) = ∑ j, P k j * x (j, i) := by
  simp only [ch04_kron2, Matrix.mulVec, dotProduct, Fintype.sum_prod_type,
    Matrix.kronecker_apply, Matrix.one_apply]
  refine Finset.sum_congr rfl fun j _ => ?_
  simp [mul_ite, ite_mul]

lemma ch04_kron2_mul {L : ℕ} (A B : Matrix (Fin L) (Fin L) ℝ) :
    ch04_kron2 A * ch04_kron2 B = ch04_kron2 (A * B) := by
  rw [ch04_kron2, ch04_kron2, ch04_kron2, ← Matrix.mul_kronecker_mul, Matrix.one_mul]

lemma ch04_kron2_one {L : ℕ} : ch04_kron2 (1 : Matrix (Fin L) (Fin L) ℝ) = 1 := by
  rw [ch04_kron2, Matrix.one_kronecker_one]

/-- `(C ⊗ I₂)^{-1} = C^{-1} ⊗ I₂`: the Kronecker factorisation the toolbox uses
whenever it writes `(C_t^{-1} ⊗ I₂) x̃` for the action of the inverse of the
trajectory covariance `C_t ⊗ I₂`.  Certified at arbitrary size `L` by
`Matrix.inv_eq_right_inv`, not by computing an inverse. -/
theorem ch04_kron2_inv {L : ℕ} {C : Matrix (Fin L) (Fin L) ℝ} (hC : IsUnit C.det) :
    (ch04_kron2 C)⁻¹ = ch04_kron2 C⁻¹ := by
  refine Matrix.inv_eq_right_inv ?_
  rw [ch04_kron2_mul, Matrix.mul_nonsing_inv _ hC, ch04_kron2_one]

/-- eq:ring-conditional-noised-law, step two in the plane: the two plane
coordinates of a frame are driven by their own copies of the same white sources,
so the coefficient matrix of the whole trajectory is `A ⊗ I₂` and its Gram matrix
is `C_t ⊗ I₂` — the covariance exactly as displayed, Kronecker factor included. -/
theorem ch04_ring_channel_covariance_kron (L : ℕ) (σ : ℝ) {t : ℝ} (ht : 0 ≤ t) :
    (ch04_ring_channelCoeff L σ t ⊗ₖ (1 : Matrix (Fin 2) (Fin 2) ℝ)) *
        (ch04_ring_channelCoeff L σ t ⊗ₖ (1 : Matrix (Fin 2) (Fin 2) ℝ))ᵀ
      = ch04_kron2 (ch04_Ct L σ t) := by
  have hT : (ch04_ring_channelCoeff L σ t ⊗ₖ (1 : Matrix (Fin 2) (Fin 2) ℝ))ᵀ
      = (ch04_ring_channelCoeff L σ t)ᵀ ⊗ₖ (1 : Matrix (Fin 2) (Fin 2) ℝ)ᵀ :=
    (Matrix.kroneckerMap_transpose _ _ _).symm
  rw [hT, ← Matrix.mul_kronecker_mul, ch04_ring_channel_covariance L σ ht,
    Matrix.transpose_one, Matrix.mul_one, ch04_kron2]

/-- `q` as a row matrix and as a column matrix, so that the chapter's
`q^T ⊗ I₂` and `q ⊗ I₂` can be read as honest Kronecker products. -/
def ch04_rowVec {L : ℕ} (q : Fin L → ℝ) : Matrix (Fin 1) (Fin L) ℝ := Matrix.of fun _ j => q j

def ch04_colVec {L : ℕ} (q : Fin L → ℝ) : Matrix (Fin L) (Fin 1) ℝ := Matrix.of fun j _ => q j

/-- eq:ring-anchor-statistic (lines 350-357), second half: the Kronecker form
`(q^T ⊗ I₂) x̃` really is the frame sum `∑_j q_j x̃_j`, componentwise. -/
theorem ch04_ring_h_kron {L : ℕ} (q : Fin L → ℝ) (x : Fin L × Fin 2 → ℝ) (i : Fin 2) :
    ((ch04_rowVec q ⊗ₖ (1 : Matrix (Fin 2) (Fin 2) ℝ)) *ᵥ x) (0, i) = ∑ j, q j * x (j, i) := by
  simp only [Matrix.mulVec, dotProduct, Fintype.sum_prod_type, Matrix.kronecker_apply,
    Matrix.one_apply, ch04_rowVec, Matrix.of_apply]
  refine Finset.sum_congr rfl fun j _ => ?_
  simp [mul_ite, ite_mul]

/-- eq:ring-score-vector-form (lines 474-479): the Kronecker form `(q ⊗ I₂) m`
of the rank-one score correction has components `q_k m_i`. -/
theorem ch04_ring_q_kron {L : ℕ} (q : Fin L → ℝ) (m : Fin 2 → ℝ) (k : Fin L) (i : Fin 2) :
    ((ch04_colVec q ⊗ₖ (1 : Matrix (Fin 2) (Fin 2) ℝ)) *ᵥ fun p : Fin 1 × Fin 2 => m p.2) (k, i)
      = q k * m i := by
  simp only [Matrix.mulVec, dotProduct, Fintype.sum_prod_type, Matrix.kronecker_apply,
    Matrix.one_apply, ch04_colVec, Matrix.of_apply]
  simp [mul_ite, ite_mul]

/-- eq:ring-conditional-noised-law (lines 340-348): conditionally on the anchor
the de-rotated noised trajectory is Gaussian with mean `e^{-t} 1 ⊗ α` and
covariance `C_t ⊗ I₂`; here is its log-density, written through the precision
`P = C_t^{-1}`. -/
def ch04_ring_condMean {L : ℕ} (t : ℝ) (α : Fin 2 → ℝ) : ch04_Ix L → ℝ :=
  fun p => Real.exp (-t) * α p.2

def ch04_ring_condLog {L : ℕ} (P : Matrix (Fin L) (Fin L) ℝ) (t : ℝ) (α : Fin 2 → ℝ)
    (x : ch04_Ix L → ℝ) : ℝ :=
  ch04_logGauss (ch04_kron2 P) (ch04_ring_condMean t α) x

/-- eq:ring-conditional-score (lines 385-396):
`∇_{x̃_k} log p(x̃|α) = -∑_j (C_t^{-1})_{kj} x̃_j + e^{-t} q_k α`. -/
theorem ch04_ring_conditional_score {L : ℕ} {P : Matrix (Fin L) (Fin L) ℝ} (hP : Pᵀ = P)
    (t : ℝ) (α : Fin 2 → ℝ) (x : ch04_Ix L → ℝ) (k : Fin L) (i : Fin 2) :
    HasDerivAt (fun s => ch04_ring_condLog P t α (Function.update x (k, i) s))
      (-(∑ j, P k j * x (j, i)) + Real.exp (-t) * ch04_ring_q P k * α i) (x (k, i)) := by
  have h := ch04_hasDerivAt_logGauss (ch04_kron2_symm hP)
    (ch04_ring_condMean (L := L) t α) x (k, i)
  have hval : -(((ch04_kron2 P) *ᵥ (x - ch04_ring_condMean (L := L) t α)) (k, i))
      = -(∑ j, P k j * x (j, i)) + Real.exp (-t) * ch04_ring_q P k * α i := by
    rw [ch04_kron2_mulVec]
    have hterm : ∀ j : Fin L, P k j * ((x - ch04_ring_condMean (L := L) t α) (j, i))
        = P k j * x (j, i) - P k j * (Real.exp (-t) * α i) := by
      intro j
      simp [ch04_ring_condMean]
      ring
    rw [Finset.sum_congr rfl fun j _ => hterm j, Finset.sum_sub_distrib, ← Finset.sum_mul,
      ← ch04_ring_q_apply]
    ring
  rw [← hval]
  exact h

/-- eq:ring-conditional-noised-law, step three: the conditional mean of the
display is what the channel of eq:ring-channel does to the anchor — `e^{-t}α` in
every frame and both plane coordinates, i.e. `e^{-t}(1 ⊗ α)`. -/
theorem ch04_ring_condMean_eq_channel {L : ℕ} (t : ℝ) (α : Fin 2 → ℝ) :
    ch04_ring_channel t (fun p : ch04_Ix L => α p.2) (fun _ => 0)
      = ch04_ring_condMean t α := by
  funext p
  simp [ch04_ring_channel, ch04_ring_condMean]

/-- The toolbox's symmetric precision `P` at the chapter's own matrix: for
`t > 0`, `C_t` is invertible, `C_t^{-1}` is symmetric, the inverse of the
trajectory covariance `C_t ⊗ I₂` is `C_t^{-1} ⊗ I₂`, and eq:ring-conditional-score
holds with `P = C_t^{-1}` and `q = C_t^{-1} 1`.  This is the link between
eq:ring-Ct-def and every score display of the toolbox. -/
theorem ch04_ring_conditional_score_Ct {L : ℕ} (σ : ℝ) {t : ℝ} (ht : 0 < t)
    (α : Fin 2 → ℝ) (x : ch04_Ix L → ℝ) (k : Fin L) (i : Fin 2) :
    (ch04_kron2 (ch04_Ct L σ t))⁻¹ = ch04_kron2 (ch04_Ct L σ t)⁻¹
    ∧ HasDerivAt
        (fun s => ch04_ring_condLog (ch04_Ct L σ t)⁻¹ t α (Function.update x (k, i) s))
        (-(∑ j, (ch04_Ct L σ t)⁻¹ k j * x (j, i))
          + Real.exp (-t) * ch04_ring_q (ch04_Ct L σ t)⁻¹ k * α i) (x (k, i)) :=
  ⟨ch04_kron2_inv (ch04_ring_Ct_isUnit_det σ ht),
    ch04_ring_conditional_score (ch04_ring_Ct_inv_symm σ t) t α x k i⟩

/-! ### Toolbox 4.1: Fisher's identity and the exact score

The chapter's anchor integral is replaced throughout by a *finite* anchor
alphabet `Λ` with weights `w` and values `v : Λ → ℝ²`.  Differentiation under
the integral sign is then a finite sum of derivatives, so Fisher's identity
below is an honest theorem with no dominated-convergence side conditions. -/

/-- noname-4 (lines 368-372): the latent-variable marginal
`p_t(x̃) = ∫ p(x̃|α) p(α) dα`. -/
def ch04_ring_marginal {Λ ι : Type*} [Fintype Λ] (w : Λ → ℝ) (f : Λ → (ι → ℝ) → ℝ)
    (x : ι → ℝ) : ℝ := ∑ l, w l * f l x

/-- The anchor posterior `p(α | x̃)`. -/
def ch04_ring_post {Λ ι : Type*} [Fintype Λ] (w : Λ → ℝ) (f : Λ → (ι → ℝ) → ℝ)
    (x : ι → ℝ) (l : Λ) : ℝ := w l * f l x / ch04_ring_marginal w f x

lemma ch04_ring_post_sum {Λ ι : Type*} [Fintype Λ] (w : Λ → ℝ) (f : Λ → (ι → ℝ) → ℝ)
    (x : ι → ℝ) (h : ch04_ring_marginal w f x ≠ 0) :
    ∑ l, ch04_ring_post w f x l = 1 := by
  have hdef : ∑ l, w l * f l x = ch04_ring_marginal w f x := rfl
  simp only [ch04_ring_post, ← Finset.sum_div, hdef]
  exact div_self h

/-- Averaging an affine function of the anchor against the posterior. -/
lemma ch04_ring_post_affine {Λ ι : Type*} [Fintype Λ] (w : Λ → ℝ) (f : Λ → (ι → ℝ) → ℝ)
    (x : ι → ℝ) (h : ch04_ring_marginal w f x ≠ 0) (a b : ℝ) (φ : Λ → ℝ) :
    (∑ l, ch04_ring_post w f x l * (a + b * φ l))
      = a + b * ∑ l, ch04_ring_post w f x l * φ l := by
  have hsum := ch04_ring_post_sum w f x h
  have hterm : ∀ l : Λ, ch04_ring_post w f x l * (a + b * φ l)
      = ch04_ring_post w f x l * a + b * (ch04_ring_post w f x l * φ l) := fun l => by ring
  rw [Finset.sum_congr rfl fun l _ => hterm l, Finset.sum_add_distrib, ← Finset.sum_mul, hsum,
    ← Finset.mul_sum]
  ring

/-- eq:ring-fisher-identity (lines 374-383): the score of the marginal is the
posterior average of the conditional score. -/
theorem ch04_ring_fisher_identity {Λ ι : Type*} [Fintype Λ] [Fintype ι] [DecidableEq ι]
    (w : Λ → ℝ) (f : Λ → (ι → ℝ) → ℝ) (g : Λ → ℝ) (x : ι → ℝ) (c : ι)
    (hpos : ch04_ring_marginal w f x ≠ 0)
    (hd : ∀ l, HasDerivAt (fun s => f l (Function.update x c s)) (f l x * g l) (x c)) :
    HasDerivAt (fun s => Real.log (ch04_ring_marginal w f (Function.update x c s)))
      (∑ l, ch04_ring_post w f x l * g l) (x c) := by
  have hupd : Function.update x c (x c) = x := Function.update_eq_self c x
  have hpos' : (∑ l, w l * f l x) ≠ 0 := hpos
  have h1 := ch04_hasDerivAt_sum (fun l s => w l * f l (Function.update x c s))
      (fun l => w l * (f l x * g l)) (x c) (fun l => (hd l).const_mul (w l))
  have hne : (∑ l, w l * f l (Function.update x c (x c))) ≠ 0 := by
    rw [hupd]; exact hpos'
  have h2 := h1.log hne
  rw [hupd] at h2
  have hsplit : (∑ l, ch04_ring_post w f x l * g l)
      = (∑ l, w l * (f l x * g l)) / (∑ l, w l * f l x) := by
    rw [eq_div_iff hpos', Finset.sum_mul]
    refine Finset.sum_congr rfl fun l _ => ?_
    rw [ch04_ring_post, ch04_ring_marginal]
    field_simp
  show HasDerivAt (fun s => Real.log (∑ l, w l * f l (Function.update x c s))) _ (x c)
  rw [hsplit]
  exact h2

/-- eq:ring-recap-score (lines 298-302) and the boxed display of lines 398-405:
`S_k^{(0)}(x̃,t) = -∑_j (C_t^{-1})_{kj} x̃_j + e^{-t} q_k E[A | x̃]`, obtained by
averaging eq:ring-conditional-score over the anchor posterior. -/
theorem ch04_ring_recap_score {L : ℕ} {Λ : Type*} [Fintype Λ]
    {P : Matrix (Fin L) (Fin L) ℝ} (hP : Pᵀ = P) (t : ℝ) (w : Λ → ℝ) (v : Λ → Fin 2 → ℝ)
    (x : ch04_Ix L → ℝ) (k : Fin L) (i : Fin 2)
    (hpos : ch04_ring_marginal w (fun l y => Real.exp (ch04_ring_condLog P t (v l) y)) x ≠ 0) :
    HasDerivAt
      (fun s => Real.log (ch04_ring_marginal w
        (fun l y => Real.exp (ch04_ring_condLog P t (v l) y)) (Function.update x (k, i) s)))
      (-(∑ j, P k j * x (j, i))
        + Real.exp (-t) * ch04_ring_q P k *
            (∑ l, ch04_ring_post w (fun l y => Real.exp (ch04_ring_condLog P t (v l) y)) x l
              * v l i))
      (x (k, i)) := by
  have hd : ∀ l : Λ,
      HasDerivAt (fun s => (fun l y => Real.exp (ch04_ring_condLog P t (v l) y)) l
          (Function.update x (k, i) s))
        ((fun l y => Real.exp (ch04_ring_condLog P t (v l) y)) l x *
          (-(∑ j, P k j * x (j, i)) + Real.exp (-t) * ch04_ring_q P k * v l i)) (x (k, i)) := by
    intro l
    have hc := (ch04_ring_conditional_score hP t (v l) x k i).exp
    simpa [Function.update_eq_self] using hc
  have hfisher := ch04_ring_fisher_identity w (fun l y => Real.exp (ch04_ring_condLog P t (v l) y))
      (fun l => -(∑ j, P k j * x (j, i)) + Real.exp (-t) * ch04_ring_q P k * v l i) x (k, i)
      hpos hd
  have haff := ch04_ring_post_affine w (fun l y => Real.exp (ch04_ring_condLog P t (v l) y)) x hpos
      (-(∑ j, P k j * x (j, i))) (Real.exp (-t) * ch04_ring_q P k) (fun l => v l i)
  rw [← haff]
  exact hfisher

/-- eq:ring-anchor-posterior (lines 413-420): the anchor posterior depends on the
whole trajectory only through the two-dimensional statistic `h(x̃)`.  The exact
splitting of the exponent is checked here: the `x̃`-dependence that survives is
exactly `h(x̃)ᵀα`, and everything else is either independent of `α` or
independent of `x̃`. -/
def ch04_ring_h {L : ℕ} (P : Matrix (Fin L) (Fin L) ℝ) (t : ℝ) (x : ch04_Ix L → ℝ)
    (i : Fin 2) : ℝ := Real.exp (-t) * ∑ j, ch04_ring_q P j * x (j, i)

theorem ch04_ring_anchor_posterior {L : ℕ} {P : Matrix (Fin L) (Fin L) ℝ} (hP : Pᵀ = P)
    (t : ℝ) (α : Fin 2 → ℝ) (x : ch04_Ix L → ℝ) :
    ch04_ring_condLog P t α x
      = (-(1/2) * ch04_quad (ch04_kron2 P) x)
        + (∑ i, ch04_ring_h P t x i * α i)
        + (-(1/2) * Real.exp (-2 * t) * (∑ k, ch04_ring_q P k) * (∑ i, α i ^ 2)) := by
  have hmean : (ch04_ring_condMean t α : ch04_Ix L → ℝ) = fun p => Real.exp (-t) * α p.2 := rfl
  have hcross : (x ⬝ᵥ (ch04_kron2 P *ᵥ (ch04_ring_condMean t α)))
      = ∑ i, ch04_ring_h P t x i * α i := by
    simp only [dotProduct, Fintype.sum_prod_type, ch04_kron2_mulVec, ch04_ring_condMean,
      ch04_ring_h, Finset.sum_mul, Finset.mul_sum]
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
    rw [ch04_ring_q_apply, Finset.sum_mul, Finset.mul_sum, Finset.sum_mul]
    exact Finset.sum_congr rfl fun j' _ => by ring
  have hmm : ch04_quad (ch04_kron2 P) (ch04_ring_condMean t α)
      = Real.exp (-2 * t) * (∑ k, ch04_ring_q P k) * (∑ i, α i ^ 2) := by
    simp only [ch04_quad, dotProduct, Fintype.sum_prod_type, ch04_kron2_mulVec,
      ch04_ring_condMean, Finset.sum_mul, Finset.mul_sum]
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun i _ => ?_
    have hee : Real.exp (-t) * Real.exp (-t) = Real.exp (-2 * t) := by
      rw [← Real.exp_add]
      congr 1
      ring
    have hq : ∀ k : Fin L, (∑ j, Real.exp (-t) * α i * (P k j * (Real.exp (-t) * α i)))
        = Real.exp (-2 * t) * ch04_ring_q P k * α i ^ 2 := by
      intro k
      have hstep : ∀ j : Fin L, Real.exp (-t) * α i * (P k j * (Real.exp (-t) * α i))
          = (Real.exp (-t) * Real.exp (-t) * α i ^ 2) * P k j := fun j => by ring
      rw [Finset.sum_congr rfl fun j _ => hstep j, ← Finset.mul_sum, ← ch04_ring_q_apply, hee]
      ring
    rw [Finset.sum_congr rfl fun k _ => hq k]
  rw [ch04_ring_condLog, ch04_logGauss, ch04_quad_sub (ch04_kron2_symm hP), hcross, hmm]
  ring

/-! ### Toolbox 4.1: the response

The anchor posterior of eq:ring-anchor-posterior is the exponential family
`p(α|h) ∝ exp(g(α) + hᵀα)` in the two-dimensional natural parameter `h`.  With
a finite alphabet the partition function is a finite sum. -/

/-- noname-6 (lines 425-429): `Z(h) = ∫ exp(g(α) + hᵀα) dα`. -/
def ch04_ring_Z {Λ : Type*} [Fintype Λ] (g : Λ → ℝ) (v : Λ → Fin 2 → ℝ) (h0 h1 : ℝ) : ℝ :=
  ∑ l, Real.exp (g l + (h0 * v l 0 + h1 * v l 1))

/-- noname-7 (lines 431-437): `m(h) = E[A|h]`. -/
def ch04_ring_m {Λ : Type*} [Fintype Λ] (g : Λ → ℝ) (v : Λ → Fin 2 → ℝ) (h0 h1 : ℝ)
    (i : Fin 2) : ℝ :=
  (∑ l, Real.exp (g l + (h0 * v l 0 + h1 * v l 1)) * v l i) / ch04_ring_Z g v h0 h1

/-- The posterior covariance `Cov(A|h)`. -/
def ch04_ring_cov {Λ : Type*} [Fintype Λ] (g : Λ → ℝ) (v : Λ → Fin 2 → ℝ) (h0 h1 : ℝ)
    (i j : Fin 2) : ℝ :=
  (∑ l, Real.exp (g l + (h0 * v l 0 + h1 * v l 1)) * (v l i * v l j)) / ch04_ring_Z g v h0 h1
    - ch04_ring_m g v h0 h1 i * ch04_ring_m g v h0 h1 j

lemma ch04_ring_Z_pos {Λ : Type*} [Fintype Λ] [Nonempty Λ] (g : Λ → ℝ) (v : Λ → Fin 2 → ℝ)
    (h0 h1 : ℝ) : 0 < ch04_ring_Z g v h0 h1 := by
  refine Finset.sum_pos (fun l _ => Real.exp_pos _) ?_
  exact Finset.univ_nonempty

lemma ch04_ring_exp_hasDerivAt {Λ : Type*} (g : Λ → ℝ) (v : Λ → Fin 2 → ℝ) (h0 h1 : ℝ)
    (l : Λ) :
    HasDerivAt (fun s : ℝ => Real.exp (g l + (s * v l 0 + h1 * v l 1)))
      (Real.exp (g l + (h0 * v l 0 + h1 * v l 1)) * v l 0) h0 := by
  have hin : HasDerivAt (fun s : ℝ => g l + (s * v l 0 + h1 * v l 1)) (v l 0) h0 := by
    have h := (((hasDerivAt_id h0).mul_const (v l 0)).add_const (h1 * v l 1)).const_add (g l)
    simpa using h
  simpa using hin.exp

/-- `∂_{h_0} Z = ∑ e^{...} v_0`. -/
lemma ch04_ring_Z_hasDerivAt {Λ : Type*} [Fintype Λ] (g : Λ → ℝ) (v : Λ → Fin 2 → ℝ)
    (h0 h1 : ℝ) :
    HasDerivAt (fun s => ch04_ring_Z g v s h1)
      (∑ l, Real.exp (g l + (h0 * v l 0 + h1 * v l 1)) * v l 0) h0 :=
  ch04_hasDerivAt_sum (fun l s => Real.exp (g l + (s * v l 0 + h1 * v l 1)))
    (fun l => Real.exp (g l + (h0 * v l 0 + h1 * v l 1)) * v l 0) h0
    (fun l => ch04_ring_exp_hasDerivAt g v h0 h1 l)

/-- noname-7 (lines 431-437): `m(h) = ∇_h log Z(h)` — here the first component. -/
theorem ch04_ring_mean_is_grad_logZ {Λ : Type*} [Fintype Λ] [Nonempty Λ] (g : Λ → ℝ)
    (v : Λ → Fin 2 → ℝ) (h0 h1 : ℝ) :
    HasDerivAt (fun s => Real.log (ch04_ring_Z g v s h1)) (ch04_ring_m g v h0 h1 0) h0 := by
  have hZ := ch04_ring_Z_hasDerivAt g v h0 h1
  have hpos := ch04_ring_Z_pos g v h0 h1
  simpa [ch04_ring_m] using hZ.log (ne_of_gt hpos)

/-- eq:ring-mean-covariance-identity (lines 439-446):
`∇_h m(h) = ∇²_h log Z(h) = Cov(A|h)` — here the derivative in the first natural
parameter, for both components of `m`. -/
theorem ch04_ring_mean_covariance {Λ : Type*} [Fintype Λ] [Nonempty Λ] (g : Λ → ℝ)
    (v : Λ → Fin 2 → ℝ) (h0 h1 : ℝ) (i : Fin 2) :
    HasDerivAt (fun s => ch04_ring_m g v s h1 i) (ch04_ring_cov g v h0 h1 i 0) h0 := by
  have hpos := ch04_ring_Z_pos g v h0 h1
  have hnum : HasDerivAt
      (fun s => ∑ l, Real.exp (g l + (s * v l 0 + h1 * v l 1)) * v l i)
      (∑ l, Real.exp (g l + (h0 * v l 0 + h1 * v l 1)) * (v l 0 * v l i)) h0 := by
    refine ch04_hasDerivAt_sum (fun l s => Real.exp (g l + (s * v l 0 + h1 * v l 1)) * v l i)
      (fun l => Real.exp (g l + (h0 * v l 0 + h1 * v l 1)) * (v l 0 * v l i)) h0 (fun l => ?_)
    have h := (ch04_ring_exp_hasDerivAt g v h0 h1 l).mul_const (v l i)
    have hrw : Real.exp (g l + (h0 * v l 0 + h1 * v l 1)) * v l 0 * v l i
        = Real.exp (g l + (h0 * v l 0 + h1 * v l 1)) * (v l 0 * v l i) := by ring
    rw [hrw] at h
    exact h
  have hZ := ch04_ring_Z_hasDerivAt g v h0 h1
  have hdiv := hnum.div hZ (ne_of_gt hpos)
  have hval : ((∑ l, Real.exp (g l + (h0 * v l 0 + h1 * v l 1)) * (v l 0 * v l i)) *
        ch04_ring_Z g v h0 h1
        - (∑ l, Real.exp (g l + (h0 * v l 0 + h1 * v l 1)) * v l i) *
          (∑ l, Real.exp (g l + (h0 * v l 0 + h1 * v l 1)) * v l 0))
      / ch04_ring_Z g v h0 h1 ^ 2
      = ch04_ring_cov g v h0 h1 i 0 := by
    rw [ch04_ring_cov, ch04_ring_m, ch04_ring_m]
    field_simp
    try ring
  rw [← hval]
  exact hdiv

lemma ch04_ring_exp_hasDerivAt_one {Λ : Type*} (g : Λ → ℝ) (v : Λ → Fin 2 → ℝ) (h0 h1 : ℝ)
    (l : Λ) :
    HasDerivAt (fun s : ℝ => Real.exp (g l + (h0 * v l 0 + s * v l 1)))
      (Real.exp (g l + (h0 * v l 0 + h1 * v l 1)) * v l 1) h1 := by
  have hin : HasDerivAt (fun s : ℝ => g l + (h0 * v l 0 + s * v l 1)) (v l 1) h1 := by
    have h := ((((hasDerivAt_id h1).mul_const (v l 1)).const_add (h0 * v l 0)).const_add (g l))
    simpa using h
  simpa using hin.exp

/-- `∂_{h_1} Z = ∑ e^{...} v_1`. -/
lemma ch04_ring_Z_hasDerivAt_one {Λ : Type*} [Fintype Λ] (g : Λ → ℝ) (v : Λ → Fin 2 → ℝ)
    (h0 h1 : ℝ) :
    HasDerivAt (fun s => ch04_ring_Z g v h0 s)
      (∑ l, Real.exp (g l + (h0 * v l 0 + h1 * v l 1)) * v l 1) h1 :=
  ch04_hasDerivAt_sum (fun l s => Real.exp (g l + (h0 * v l 0 + s * v l 1)))
    (fun l => Real.exp (g l + (h0 * v l 0 + h1 * v l 1)) * v l 1) h1
    (fun l => ch04_ring_exp_hasDerivAt_one g v h0 h1 l)

/-- eq:ring-mean-covariance-identity (lines 439-446), the *second* column of the
Jacobian: the derivative of the posterior mean in the second natural parameter is
`Cov(A|h)_{i1}`.  With `ch04_ring_mean_covariance` (the first column) this is the
full `2×2` identity `∇_h m(h) = Cov(A|h)`, both rows and both columns. -/
theorem ch04_ring_mean_covariance_one {Λ : Type*} [Fintype Λ] [Nonempty Λ] (g : Λ → ℝ)
    (v : Λ → Fin 2 → ℝ) (h0 h1 : ℝ) (i : Fin 2) :
    HasDerivAt (fun s => ch04_ring_m g v h0 s i) (ch04_ring_cov g v h0 h1 i 1) h1 := by
  have hpos := ch04_ring_Z_pos g v h0 h1
  have hnum : HasDerivAt
      (fun s => ∑ l, Real.exp (g l + (h0 * v l 0 + s * v l 1)) * v l i)
      (∑ l, Real.exp (g l + (h0 * v l 0 + h1 * v l 1)) * (v l 1 * v l i)) h1 := by
    refine ch04_hasDerivAt_sum (fun l s => Real.exp (g l + (h0 * v l 0 + s * v l 1)) * v l i)
      (fun l => Real.exp (g l + (h0 * v l 0 + h1 * v l 1)) * (v l 1 * v l i)) h1 (fun l => ?_)
    have h := (ch04_ring_exp_hasDerivAt_one g v h0 h1 l).mul_const (v l i)
    have hrw : Real.exp (g l + (h0 * v l 0 + h1 * v l 1)) * v l 1 * v l i
        = Real.exp (g l + (h0 * v l 0 + h1 * v l 1)) * (v l 1 * v l i) := by ring
    rw [hrw] at h
    exact h
  have hZ := ch04_ring_Z_hasDerivAt_one g v h0 h1
  have hdiv := hnum.div hZ (ne_of_gt hpos)
  have hval : ((∑ l, Real.exp (g l + (h0 * v l 0 + h1 * v l 1)) * (v l 1 * v l i)) *
        ch04_ring_Z g v h0 h1
        - (∑ l, Real.exp (g l + (h0 * v l 0 + h1 * v l 1)) * v l i) *
          (∑ l, Real.exp (g l + (h0 * v l 0 + h1 * v l 1)) * v l 1))
      / ch04_ring_Z g v h0 h1 ^ 2
      = ch04_ring_cov g v h0 h1 i 1 := by
    rw [ch04_ring_cov, ch04_ring_m, ch04_ring_m]
    field_simp
    try ring
  rw [← hval]
  exact hdiv

/-- The derivative of a linear statistic `∑_k c_k x̃_{k,i}` in the single
coordinate `x̃_{j,i'}`: it is `c_j` when `i = i'` and `0` otherwise.  This is the
`I₂` in noname-8 and the `(C_t^{-1})_{kj} I₂` in the response. -/
theorem ch04_ring_linear_coord_deriv {L : ℕ} (c : Fin L → ℝ) (x : ch04_Ix L → ℝ)
    (j : Fin L) (i i' : Fin 2) :
    HasDerivAt (fun s => ∑ k, c k * (Function.update x (j, i') s) (k, i))
      (if i = i' then c j else 0) (x (j, i')) := by
  have hterm : ∀ k : Fin L, HasDerivAt
      (fun s => c k * (Function.update x (j, i') s) (k, i))
      (if ((k, i) : ch04_Ix L) = (j, i') then c k else 0) (x (j, i')) := by
    intro k
    by_cases hk : ((k, i) : ch04_Ix L) = (j, i')
    · rw [if_pos hk]
      have hfun : (fun s => c k * (Function.update x (j, i') s) (k, i)) = fun s => c k * s := by
        funext s
        rw [hk, Function.update_self]
      rw [hfun]
      simpa using (hasDerivAt_id (x (j, i'))).const_mul (c k)
    · rw [if_neg hk]
      have hfun : (fun s => c k * (Function.update x (j, i') s) (k, i))
          = fun _ => c k * x (k, i) := by
        funext s
        rw [Function.update_of_ne hk]
      rw [hfun]
      exact hasDerivAt_const _ _
  have hsum := ch04_hasDerivAt_sum (fun k s => c k * (Function.update x (j, i') s) (k, i))
      (fun k => if ((k, i) : ch04_Ix L) = (j, i') then c k else 0) (x (j, i')) hterm
  have hval : (∑ k, (if ((k, i) : ch04_Ix L) = (j, i') then c k else 0))
      = (if i = i' then c j else 0) := by
    by_cases hii : i = i'
    · subst hii
      simp
    · have : ∀ k : Fin L, ((k, i) : ch04_Ix L) = (j, i') ↔ False := by
        intro k
        simp [Prod.ext_iff, hii]
      simp [this, hii]
  rw [← hval]
  exact hsum

/-- noname-8 (lines 448-452): `∂h/∂x̃_j = e^{-t} q_j I₂`. -/
theorem ch04_ring_h_hasDerivAt {L : ℕ} (P : Matrix (Fin L) (Fin L) ℝ) (t : ℝ)
    (x : ch04_Ix L → ℝ) (j : Fin L) (i i' : Fin 2) :
    HasDerivAt (fun s => ch04_ring_h P t (Function.update x (j, i') s) i)
      (Real.exp (-t) * (if i = i' then ch04_ring_q P j else 0)) (x (j, i')) := by
  have h := (ch04_ring_linear_coord_deriv (ch04_ring_q P) x j i i').const_mul (Real.exp (-t))
  simpa [ch04_ring_h] using h

/-- A natural parameter does not move when a *different* plane coordinate of a
frame is perturbed: `h_i` depends on `x̃_{·,i}` only. -/
lemma ch04_ring_h_const {L : ℕ} (P : Matrix (Fin L) (Fin L) ℝ) (t : ℝ) (x : ch04_Ix L → ℝ)
    (j : Fin L) (i i' : Fin 2) (hne : i ≠ i') (s : ℝ) :
    ch04_ring_h P t (Function.update x (j, i') s) i = ch04_ring_h P t x i := by
  simp only [ch04_ring_h]
  congr 1
  refine Finset.sum_congr rfl fun k _ => ?_
  have hne' : ((k, i) : ch04_Ix L) ≠ (j, i') := by
    simp [Prod.ext_iff, hne]
  rw [Function.update_of_ne hne']

/-- eq:ring-anchor-response (lines 454-460):
`∂E[A|x̃]/∂x̃_j = e^{-t} q_j Cov(A|x̃)`, by the chain rule through `h`.  The full
`2×2` matrix identity: the perturbed plane coordinate `i'` and the component `i`
of the posterior mean are both arbitrary, and the answer is the `(i,i')` entry of
the posterior covariance. -/
theorem ch04_ring_anchor_response {L : ℕ} {Λ : Type*} [Fintype Λ] [Nonempty Λ]
    (P : Matrix (Fin L) (Fin L) ℝ) (t : ℝ) (g : Λ → ℝ) (v : Λ → Fin 2 → ℝ)
    (x : ch04_Ix L → ℝ) (j : Fin L) (i i' : Fin 2) :
    HasDerivAt
      (fun s => ch04_ring_m g v (ch04_ring_h P t (Function.update x (j, i') s) 0)
        (ch04_ring_h P t (Function.update x (j, i') s) 1) i)
      (Real.exp (-t) * ch04_ring_q P j *
        ch04_ring_cov g v (ch04_ring_h P t x 0) (ch04_ring_h P t x 1) i i') (x (j, i')) := by
  have hinner : ∀ a : Fin 2,
      HasDerivAt (fun s => ch04_ring_h P t (Function.update x (j, a) s) a)
        (Real.exp (-t) * ch04_ring_q P j) (x (j, a)) := by
    intro a
    simpa using ch04_ring_h_hasDerivAt P t x j a a
  have hpoint : ∀ a : Fin 2,
      ch04_ring_h P t (Function.update x (j, a) (x (j, a))) a = ch04_ring_h P t x a := by
    intro a
    rw [Function.update_eq_self]
  have hcases : ∀ a : Fin 2, a = 0 ∨ a = 1 := by decide
  rcases hcases i' with hi' | hi'
  · -- perturb the first plane coordinate of frame `j`
    subst hi'
    have hfun : (fun s => ch04_ring_m g v (ch04_ring_h P t (Function.update x (j, 0) s) 0)
          (ch04_ring_h P t (Function.update x (j, 0) s) 1) i)
        = fun s => (fun u => ch04_ring_m g v u (ch04_ring_h P t x 1) i)
            (ch04_ring_h P t (Function.update x (j, 0) s) 0) := by
      funext s
      rw [ch04_ring_h_const P t x j 1 0 (by decide)]
    rw [hfun]
    have houter' : HasDerivAt (fun u => ch04_ring_m g v u (ch04_ring_h P t x 1) i)
        (ch04_ring_cov g v (ch04_ring_h P t x 0) (ch04_ring_h P t x 1) i 0)
        ((fun s => ch04_ring_h P t (Function.update x (j, 0) s) 0) (x (j, 0))) := by
      simpa only [hpoint 0] using
        ch04_ring_mean_covariance g v (ch04_ring_h P t x 0) (ch04_ring_h P t x 1) i
    have hcomp := houter'.comp (x (j, 0)) (hinner 0)
    simp only [Function.comp_def] at hcomp
    have hrw : ch04_ring_cov g v (ch04_ring_h P t x 0) (ch04_ring_h P t x 1) i 0 *
        (Real.exp (-t) * ch04_ring_q P j)
        = Real.exp (-t) * ch04_ring_q P j *
          ch04_ring_cov g v (ch04_ring_h P t x 0) (ch04_ring_h P t x 1) i 0 := by ring
    rw [hrw] at hcomp
    exact hcomp
  · -- perturb the second plane coordinate of frame `j`
    subst hi'
    have hfun : (fun s => ch04_ring_m g v (ch04_ring_h P t (Function.update x (j, 1) s) 0)
          (ch04_ring_h P t (Function.update x (j, 1) s) 1) i)
        = fun s => (fun u => ch04_ring_m g v (ch04_ring_h P t x 0) u i)
            (ch04_ring_h P t (Function.update x (j, 1) s) 1) := by
      funext s
      rw [ch04_ring_h_const P t x j 0 1 (by decide)]
    rw [hfun]
    have houter' : HasDerivAt (fun u => ch04_ring_m g v (ch04_ring_h P t x 0) u i)
        (ch04_ring_cov g v (ch04_ring_h P t x 0) (ch04_ring_h P t x 1) i 1)
        ((fun s => ch04_ring_h P t (Function.update x (j, 1) s) 1) (x (j, 1))) := by
      simpa only [hpoint 1] using
        ch04_ring_mean_covariance_one g v (ch04_ring_h P t x 0) (ch04_ring_h P t x 1) i
    have hcomp := houter'.comp (x (j, 1)) (hinner 1)
    simp only [Function.comp_def] at hcomp
    have hrw : ch04_ring_cov g v (ch04_ring_h P t x 0) (ch04_ring_h P t x 1) i 1 *
        (Real.exp (-t) * ch04_ring_q P j)
        = Real.exp (-t) * ch04_ring_q P j *
          ch04_ring_cov g v (ch04_ring_h P t x 0) (ch04_ring_h P t x 1) i 1 := by ring
    rw [hrw] at hcomp
    exact hcomp

/-- eq:ring-recap-response (lines 303-306) and the boxed display of lines
462-471: differentiating the score of eq:ring-recap-score once more,
`J_{kj}^{(0)} = -(C_t^{-1})_{kj} I₂ + e^{-2t} q_k q_j Cov(A|x̃)`.

This is the full `2×2` block identity: the frames `k, j`, the score component `i`
and the perturbed plane coordinate `i'` are all arbitrary, the `I₂` appears as the
Kronecker delta `if i = i'`, and the rank-one correction carries the `(i,i')`
entry of the posterior covariance. -/
theorem ch04_ring_recap_response {L : ℕ} {Λ : Type*} [Fintype Λ] [Nonempty Λ]
    (P : Matrix (Fin L) (Fin L) ℝ) (t : ℝ) (g : Λ → ℝ) (v : Λ → Fin 2 → ℝ)
    (x : ch04_Ix L → ℝ) (k j : Fin L) (i i' : Fin 2) :
    HasDerivAt
      (fun s => -(∑ j', P k j' * (Function.update x (j, i') s) (j', i))
        + Real.exp (-t) * ch04_ring_q P k *
          ch04_ring_m g v (ch04_ring_h P t (Function.update x (j, i') s) 0)
            (ch04_ring_h P t (Function.update x (j, i') s) 1) i)
      (-(P k j) * (if i = i' then (1:ℝ) else 0)
        + Real.exp (-2 * t) * ch04_ring_q P k * ch04_ring_q P j *
          ch04_ring_cov g v (ch04_ring_h P t x 0) (ch04_ring_h P t x 1) i i')
      (x (j, i')) := by
  have hee : Real.exp (-t) * Real.exp (-t) = Real.exp (-2 * t) := by
    rw [← Real.exp_add]
    congr 1
    ring
  have hlin : HasDerivAt (fun s => -(∑ j', P k j' * (Function.update x (j, i') s) (j', i)))
      (-(if i = i' then P k j else 0)) (x (j, i')) :=
    (ch04_ring_linear_coord_deriv (fun j' => P k j') x j i i').neg
  have hresp : HasDerivAt
      (fun s => Real.exp (-t) * ch04_ring_q P k *
        ch04_ring_m g v (ch04_ring_h P t (Function.update x (j, i') s) 0)
          (ch04_ring_h P t (Function.update x (j, i') s) 1) i)
      (Real.exp (-t) * ch04_ring_q P k *
        (Real.exp (-t) * ch04_ring_q P j *
          ch04_ring_cov g v (ch04_ring_h P t x 0) (ch04_ring_h P t x 1) i i')) (x (j, i')) :=
    (ch04_ring_anchor_response P t g v x j i i').const_mul (Real.exp (-t) * ch04_ring_q P k)
  have hsum : HasDerivAt
      (fun s => -(∑ j', P k j' * (Function.update x (j, i') s) (j', i))
        + Real.exp (-t) * ch04_ring_q P k *
          ch04_ring_m g v (ch04_ring_h P t (Function.update x (j, i') s) 0)
            (ch04_ring_h P t (Function.update x (j, i') s) 1) i)
      ((-(if i = i' then P k j else 0))
        + Real.exp (-t) * ch04_ring_q P k *
          (Real.exp (-t) * ch04_ring_q P j *
            ch04_ring_cov g v (ch04_ring_h P t x 0) (ch04_ring_h P t x 1) i i')) (x (j, i')) :=
    hlin.add hresp
  convert hsum using 1
  by_cases hi : i = i'
  · rw [if_pos hi, if_pos hi]
    linear_combination (ch04_ring_q P k * ch04_ring_q P j *
      ch04_ring_cov g v (ch04_ring_h P t x 0) (ch04_ring_h P t x 1) i i') * hee.symm
  · rw [if_neg hi, if_neg hi]
    linear_combination (ch04_ring_q P k * ch04_ring_q P j *
      ch04_ring_cov g v (ch04_ring_h P t x 0) (ch04_ring_h P t x 1) i i') * hee.symm

/-- eq:ring-score-vector-form (lines 474-479): the vector form
`S^{(0)} = -(C_t^{-1} ⊗ I₂) x̃ + e^{-t}(q ⊗ I₂) E[A|x̃]` has exactly the
components of eq:ring-recap-score. -/
theorem ch04_ring_score_vector_form {L : ℕ} (P : Matrix (Fin L) (Fin L) ℝ) (t : ℝ)
    (x : ch04_Ix L → ℝ) (m : Fin 2 → ℝ) (k : Fin L) (i : Fin 2) :
    (-(ch04_kron2 P *ᵥ x) + fun p : ch04_Ix L => Real.exp (-t) * ch04_ring_q P p.1 * m p.2)
        (k, i)
      = -(∑ j, P k j * x (j, i)) + Real.exp (-t) * ch04_ring_q P k * m i := by
  simp [ch04_kron2_mulVec]

/-- eq:ring-response-vector-form (lines 480-486): the vector form
`J^{(0)} = -(C_t^{-1} ⊗ I₂) + e^{-2t}(qqᵀ ⊗ Cov)` has exactly the entries of
eq:ring-recap-response, and the correction is rank one in the frame indices
(every row of `qqᵀ` is a multiple of `qᵀ`). -/
theorem ch04_ring_response_vector_form {L : ℕ} (P : Matrix (Fin L) (Fin L) ℝ) (t : ℝ)
    (Cov : Matrix (Fin 2) (Fin 2) ℝ) (k j : Fin L) (i i' : Fin 2) :
    (-(ch04_kron2 P) + (Real.exp (-2 * t)) •
        (Matrix.vecMulVec (ch04_ring_q P) (ch04_ring_q P) ⊗ₖ Cov)) (k, i) (j, i')
      = -(P k j * (if i = i' then (1:ℝ) else 0))
        + Real.exp (-2 * t) * (ch04_ring_q P k * ch04_ring_q P j) * Cov i i' := by
  simp [ch04_kron2, Matrix.kronecker_apply, Matrix.one_apply, Matrix.vecMulVec_apply,
    smul_eq_mul]
  ring

lemma ch04_ring_qqT_rank_one {L : ℕ} (P : Matrix (Fin L) (Fin L) ℝ) (k j : Fin L) :
    Matrix.vecMulVec (ch04_ring_q P) (ch04_ring_q P) k j
      = ch04_ring_q P k * ch04_ring_q P j := by
  simp [Matrix.vecMulVec_apply]

/-! ### §4.4  The clean surrogate score at an interior frame -/

/-- noname-11 (lines 568-574): the two terms of `log p(z)` that contain `z_k`,
for an interior frame `1 ≤ k ≤ L-2`. -/
def ch04_ring_cleanLocal (R : Matrix (Fin 2) (Fin 2) ℝ) (σ : ℝ)
    (zprev zk znext : Fin 2 → ℝ) : ℝ :=
  -(1/(2*σ^2)) * (∑ i, (zk i - (R *ᵥ zprev) i)^2)
    - (1/(2*σ^2)) * (∑ i, (znext i - (R *ᵥ zk) i)^2)

/-- eq:ring-rotation-in-score (lines 548-557) at the first coordinate. -/
theorem ch04_ring_rotation_in_score_zero {σ : ℝ} (hσ : σ ≠ 0) (ψ : ℝ)
    (zprev zk znext : Fin 2 → ℝ) :
    HasDerivAt (fun s => ch04_ring_cleanLocal (ch04_R ψ) σ zprev (Function.update zk 0 s) znext)
      (1/σ^2 * ((ch04_R ψ *ᵥ zprev) 0 + ((ch04_R ψ)ᵀ *ᵥ znext) 0 - 2 * zk 0)) (zk 0) := by
  have hcd : Real.cos ψ ^ 2 + Real.sin ψ ^ 2 = 1 := Real.cos_sq_add_sin_sq ψ
  have hσ2 : σ ^ 2 ≠ 0 := pow_ne_zero 2 hσ
  have hfun :
      (fun s => ch04_ring_cleanLocal (ch04_R ψ) σ zprev (Function.update zk 0 s) znext)
      = fun s => (-(1/(2*σ^2)) * 2) * s ^ 2
          + ((1/σ^2) * ((Real.cos ψ * zprev 0 - Real.sin ψ * zprev 1)
              + (Real.cos ψ * znext 0 + Real.sin ψ * znext 1))) * s
          + ch04_ring_cleanLocal (ch04_R ψ) σ zprev (Function.update zk 0 0) znext := by
    funext s
    simp [ch04_ring_cleanLocal, ch04_R, Matrix.mulVec, dotProduct, Fin.sum_univ_two,
      Function.update_apply]
    first
      | linear_combination (-(s ^ 2) / (2 * σ ^ 2)) * hcd
      | linear_combination (-(s ^ 2)) * hcd
      | (field_simp; linear_combination (-(s ^ 2)) * hcd)
      | (field_simp; linear_combination (-(2 * s ^ 2)) * hcd)
      | (field_simp; linear_combination (-(s ^ 2) * σ ^ 2) * hcd)
      | (field_simp; ring)
      | ring
  rw [hfun]
  have hq := ch04_hasDerivAt_quadratic (-(1/(2*σ^2)) * 2)
      ((1/σ^2) * ((Real.cos ψ * zprev 0 - Real.sin ψ * zprev 1)
        + (Real.cos ψ * znext 0 + Real.sin ψ * znext 1)))
      (ch04_ring_cleanLocal (ch04_R ψ) σ zprev (Function.update zk 0 0) znext) (zk 0)
  convert hq using 1
  simp [ch04_R, Matrix.mulVec, dotProduct, Fin.sum_univ_two, Matrix.transpose_apply]
  first
    | ring
    | (field_simp; ring)

/-- eq:ring-rotation-in-score (lines 548-557) at the second coordinate. -/
theorem ch04_ring_rotation_in_score_one {σ : ℝ} (hσ : σ ≠ 0) (ψ : ℝ)
    (zprev zk znext : Fin 2 → ℝ) :
    HasDerivAt (fun s => ch04_ring_cleanLocal (ch04_R ψ) σ zprev (Function.update zk 1 s) znext)
      (1/σ^2 * ((ch04_R ψ *ᵥ zprev) 1 + ((ch04_R ψ)ᵀ *ᵥ znext) 1 - 2 * zk 1)) (zk 1) := by
  have hcd : Real.cos ψ ^ 2 + Real.sin ψ ^ 2 = 1 := Real.cos_sq_add_sin_sq ψ
  have hσ2 : σ ^ 2 ≠ 0 := pow_ne_zero 2 hσ
  have hfun :
      (fun s => ch04_ring_cleanLocal (ch04_R ψ) σ zprev (Function.update zk 1 s) znext)
      = fun s => (-(1/(2*σ^2)) * 2) * s ^ 2
          + ((1/σ^2) * ((Real.sin ψ * zprev 0 + Real.cos ψ * zprev 1)
              + (-Real.sin ψ * znext 0 + Real.cos ψ * znext 1))) * s
          + ch04_ring_cleanLocal (ch04_R ψ) σ zprev (Function.update zk 1 0) znext := by
    funext s
    simp [ch04_ring_cleanLocal, ch04_R, Matrix.mulVec, dotProduct, Fin.sum_univ_two,
      Function.update_apply]
    first
      | linear_combination (-(s ^ 2) / (2 * σ ^ 2)) * hcd
      | linear_combination (-(s ^ 2)) * hcd
      | (field_simp; linear_combination (-(s ^ 2)) * hcd)
      | (field_simp; linear_combination (-(2 * s ^ 2)) * hcd)
      | (field_simp; linear_combination (-(s ^ 2) * σ ^ 2) * hcd)
      | (field_simp; ring)
      | ring
  rw [hfun]
  have hq := ch04_hasDerivAt_quadratic (-(1/(2*σ^2)) * 2)
      ((1/σ^2) * ((Real.sin ψ * zprev 0 + Real.cos ψ * zprev 1)
        + (-Real.sin ψ * znext 0 + Real.cos ψ * znext 1)))
      (ch04_ring_cleanLocal (ch04_R ψ) σ zprev (Function.update zk 1 0) znext) (zk 1)
  convert hq using 1
  simp [ch04_R, Matrix.mulVec, dotProduct, Fin.sum_univ_two, Matrix.transpose_apply]
  first
    | ring
    | (field_simp; ring)

/-- eq:ring-rotation-in-score (lines 548-557) together with the intermediate
displays noname-12, noname-13 and noname-14 of Toolbox 4.2:
`S_k(z,0) = σ^{-2}(R_ψ z_{k-1} + R_ψᵀ z_{k+1} - 2 z_k)` for an interior frame,
checked in both coordinates. -/
theorem ch04_ring_rotation_in_score {σ : ℝ} (hσ : σ ≠ 0) (ψ : ℝ)
    (zprev zk znext : Fin 2 → ℝ) (i : Fin 2) :
    HasDerivAt (fun s => ch04_ring_cleanLocal (ch04_R ψ) σ zprev (Function.update zk i s) znext)
      (1/σ^2 * ((ch04_R ψ *ᵥ zprev) i + ((ch04_R ψ)ᵀ *ᵥ znext) i - 2 * zk i)) (zk i) := by
  fin_cases i
  · exact ch04_ring_rotation_in_score_zero hσ ψ zprev zk znext
  · exact ch04_ring_rotation_in_score_one hσ ψ zprev zk znext

/-- noname-12 (lines 581-584): the gradient of the first term,
`∇_{z_k}[-‖z_k - R_ψ z_{k-1}‖²/(2σ²)] = -σ^{-2}(z_k - R_ψ z_{k-1})`. -/
theorem ch04_ring_clean_term1 {σ : ℝ} (hσ : σ ≠ 0) (R : Matrix (Fin 2) (Fin 2) ℝ)
    (zprev zk : Fin 2 → ℝ) (i : Fin 2) :
    HasDerivAt (fun s => -(1/(2*σ^2)) * ∑ r, ((Function.update zk i s) r - (R *ᵥ zprev) r) ^ 2)
      (-((1/σ^2) * (zk i - (R *ᵥ zprev) i))) (zk i) := by
  have hsym : ((1/σ^2) • (1 : Matrix (Fin 2) (Fin 2) ℝ))ᵀ
      = (1/σ^2) • (1 : Matrix (Fin 2) (Fin 2) ℝ) := by
    simp [Matrix.transpose_smul, Matrix.transpose_one]
  have h := ch04_hasDerivAt_logGauss hsym (R *ᵥ zprev) zk i
  have hval : -((((1/σ^2) • (1 : Matrix (Fin 2) (Fin 2) ℝ)) *ᵥ (zk - R *ᵥ zprev)) i)
      = -((1/σ^2) * (zk i - (R *ᵥ zprev) i)) := by
    rw [ch04_smul_one_mulVec]
    simp
  have hfun : (fun s => -(1/(2*σ^2)) * ∑ r, ((Function.update zk i s) r - (R *ᵥ zprev) r) ^ 2)
      = fun s => ch04_logGauss ((1/σ^2) • (1 : Matrix (Fin 2) (Fin 2) ℝ)) (R *ᵥ zprev)
          (Function.update zk i s) := by
    funext s
    rw [ch04_logGauss, ch04_quad, Matrix.smul_mulVec, Matrix.one_mulVec, dotProduct_smul,
      smul_eq_mul]
    simp only [dotProduct, Pi.sub_apply]
    have hsq : ∀ r : Fin 2,
        ((Function.update zk i s) r - (R *ᵥ zprev) r) * ((Function.update zk i s) r - (R *ᵥ zprev) r)
          = ((Function.update zk i s) r - (R *ᵥ zprev) r) ^ 2 := fun r => by ring
    rw [Finset.sum_congr rfl fun r _ => hsq r]
    ring
  rw [hfun, ← hval]
  exact h

/-- noname-13 (lines 589-593): the gradient of the second term,
`∇_{z_k}[-‖z_{k+1} - R_ψ z_k‖²/(2σ²)] = σ^{-2}(R_ψᵀ z_{k+1} - z_k)`, where
`R_ψᵀR_ψ = I₂` has already collapsed `R_ψᵀR_ψ z_k` to `z_k`. -/
theorem ch04_ring_clean_term2 {σ : ℝ} (hσ : σ ≠ 0) (ψ : ℝ)
    (zprev zk znext : Fin 2 → ℝ) (i : Fin 2) :
    HasDerivAt
      (fun s => -(1/(2*σ^2)) * ∑ r, (znext r - (ch04_R ψ *ᵥ (Function.update zk i s)) r) ^ 2)
      ((1/σ^2) * (((ch04_R ψ)ᵀ *ᵥ znext) i - zk i)) (zk i) := by
  have hfull := ch04_ring_rotation_in_score hσ ψ zprev zk znext i
  have h1 := ch04_ring_clean_term1 hσ (ch04_R ψ) zprev zk i
  have hsub := hfull.sub h1
  have hfe :
      (fun s => -(1/(2*σ^2)) * ∑ r, (znext r - (ch04_R ψ *ᵥ (Function.update zk i s)) r) ^ 2)
      = fun s => ch04_ring_cleanLocal (ch04_R ψ) σ zprev (Function.update zk i s) znext
          - (-(1/(2*σ^2)) * ∑ r, ((Function.update zk i s) r - (ch04_R ψ *ᵥ zprev) r) ^ 2) := by
    funext s
    rw [ch04_ring_cleanLocal]
    ring
  rw [hfe]
  have hrw : (1/σ^2) * (((ch04_R ψ)ᵀ *ᵥ znext) i - zk i)
      = 1/σ^2 * ((ch04_R ψ *ᵥ zprev) i + ((ch04_R ψ)ᵀ *ᵥ znext) i - 2 * zk i)
        - (-((1/σ^2) * (zk i - (ch04_R ψ *ᵥ zprev) i))) := by ring
  rw [hrw]
  exact hsub

/-- noname-10 (lines 562-566): the clean surrogate's joint law factorises along
the chain; its logarithm is the prior term plus the chain of squared
increments, of which exactly two involve `z_k`. -/
def ch04_ring_cleanJoint (R : Matrix (Fin 2) (Fin 2) ℝ) (σ lam : ℝ) (L : ℕ)
    (z : ℕ → Fin 2 → ℝ) : ℝ :=
  ch04_ring_prior lam (z 0) *
    ∏ i ∈ Finset.range (L - 1),
      Real.exp (-(1/(2*σ^2)) * ∑ r, (z (i+1) r - (R *ᵥ z i) r) ^ 2)

theorem ch04_ring_cleanJoint_log (R : Matrix (Fin 2) (Fin 2) ℝ) (σ lam : ℝ) (L : ℕ)
    (z : ℕ → Fin 2 → ℝ) :
    Real.log (ch04_ring_cleanJoint R σ lam L z)
      = Real.log (ch04_ring_prior lam (z 0))
        + ∑ i ∈ Finset.range (L - 1), (-(1/(2*σ^2)) * ∑ r, (z (i+1) r - (R *ᵥ z i) r) ^ 2) := by
  have hprod : (∏ i ∈ Finset.range (L - 1),
      Real.exp (-(1/(2*σ^2)) * ∑ r, (z (i+1) r - (R *ᵥ z i) r) ^ 2)) ≠ 0 :=
    ne_of_gt (Finset.prod_pos fun i _ => Real.exp_pos _)
  rw [ch04_ring_cleanJoint, Real.log_mul (ne_of_gt (ch04_ring_prior_pos lam (z 0))) hprod]
  congr 1
  rw [Real.log_prod (fun i _ => ne_of_gt (Real.exp_pos _))]
  exact Finset.sum_congr rfl fun i _ => Real.log_exp _

/-- Lemma~\ref{lem:ring-gauge} at the level of the *clean* chain density, for
every `L` and every `ψ`: the rotating chain's joint density at `z` is the
non-rotating chain's joint density at the de-rotated trajectory
`(U_ψ z)_k = R_{-kψ} z_k`.  The prior term is rotation invariant and each
increment exponent is preserved because
`R_{-(k+1)ψ}(z_{k+1} - R_ψ z_k) = (U_ψ z)_{k+1} - (U_ψ z)_k`, which is
eq:ring-gauged-rw.  This is the clean-law half of the density identity
eq:ring-gauge-score; `ch04_gauge_noisedDensity` is the noised half. -/
theorem ch04_ring_cleanJoint_gauge (ψ σ lam : ℝ) (L : ℕ) (z : ℕ → Fin 2 → ℝ) :
    ch04_ring_cleanJoint (ch04_R ψ) σ lam L z
      = ch04_ring_cleanJoint (1 : Matrix (Fin 2) (Fin 2) ℝ) σ lam L
          (fun k => ch04_R (-(k : ℝ) * ψ) *ᵥ z k) := by
  have hstep : ∀ i : ℕ,
      (∑ r, (z (i + 1) r - (ch04_R ψ *ᵥ z i) r) ^ 2)
        = ∑ r, ((ch04_R (-((i + 1 : ℕ) : ℝ) * ψ) *ᵥ z (i + 1)) r
            - (ch04_R (-((i : ℕ) : ℝ) * ψ) *ᵥ z i) r) ^ 2 := by
    intro i
    have hang : -((i + 1 : ℕ) : ℝ) * ψ + ψ = -((i : ℕ) : ℝ) * ψ := by
      push_cast
      ring
    have hcomp : ch04_R (-((i + 1 : ℕ) : ℝ) * ψ) *ᵥ (ch04_R ψ *ᵥ z i)
        = ch04_R (-((i : ℕ) : ℝ) * ψ) *ᵥ z i := by
      rw [ch04_R_mulVec_comp, hang]
    have h1 : (fun r => z (i + 1) r - (ch04_R ψ *ᵥ z i) r) = z (i + 1) - ch04_R ψ *ᵥ z i := by
      funext r
      simp
    have hsub : ch04_R (-((i + 1 : ℕ) : ℝ) * ψ) *ᵥ (fun r => z (i + 1) r - (ch04_R ψ *ᵥ z i) r)
        = fun r => (ch04_R (-((i + 1 : ℕ) : ℝ) * ψ) *ᵥ z (i + 1)) r
            - (ch04_R (-((i : ℕ) : ℝ) * ψ) *ᵥ z i) r := by
      rw [h1, Matrix.mulVec_sub, hcomp]
      funext r
      simp
    have h := ch04_R_preserves_normSq (-((i + 1 : ℕ) : ℝ) * ψ)
      (fun r => z (i + 1) r - (ch04_R ψ *ᵥ z i) r)
    rw [hsub] at h
    exact h.symm
  have hprior : ch04_ring_prior lam (z 0)
      = ch04_ring_prior lam (ch04_R (-((0 : ℕ) : ℝ) * ψ) *ᵥ z 0) :=
    (ch04_ring_prior_rotation_invariant lam _ (z 0)).symm
  unfold ch04_ring_cleanJoint
  simp only [Matrix.one_mulVec]
  rw [hprior]
  congr 1
  refine Finset.prod_congr rfl fun i _ => ?_
  rw [hstep i]

/-! ### §4.4  The polar model: denoising score and posterior -/

/-- The isotropic channel log-density `log N(x; e^{-t}a, Δ_t I)`. -/
def ch04_ring_channelLog {ι : Type*} [Fintype ι] [DecidableEq ι] (t : ℝ) (a x : ι → ℝ) : ℝ :=
  ch04_logGauss ((1 / ch04_Delta t) • (1 : Matrix ι ι ℝ)) (fun i => Real.exp (-t) * a i) x

/-- The conditional (Tweedie) score of the channel: `(e^{-t}a_c - x_c)/Δ_t`. -/
theorem ch04_ring_channel_score {ι : Type*} [Fintype ι] [DecidableEq ι] {t : ℝ} (ht : 0 < t)
    (a x : ι → ℝ) (c : ι) :
    HasDerivAt (fun s => ch04_ring_channelLog t a (Function.update x c s))
      ((Real.exp (-t) * a c - x c) / ch04_Delta t) (x c) := by
  have hD : ch04_Delta t ≠ 0 := ne_of_gt (ch04_Delta_pos ht)
  have hsym : ((1 / ch04_Delta t) • (1 : Matrix ι ι ℝ))ᵀ = (1 / ch04_Delta t) • (1 : Matrix ι ι ℝ) := by
    simp [Matrix.transpose_smul, Matrix.transpose_one]
  have h := ch04_hasDerivAt_logGauss hsym (fun i => Real.exp (-t) * a i) x c
  have hval : -((((1 / ch04_Delta t) • (1 : Matrix ι ι ℝ)) *ᵥ (x - fun i => Real.exp (-t) * a i)) c)
      = (Real.exp (-t) * a c - x c) / ch04_Delta t := by
    rw [ch04_smul_one_mulVec]
    simp only [Pi.sub_apply]
    field_simp
    ring
  rw [← hval]
  simpa only [ch04_ring_channelLog] using h

/-- eq:ring-polar-denoising-score (lines 614-618):
`S_{ω,k}(x,t) = (e^{-t} E_ω[a_k | x] - x_k)/Δ_t`, the denoising identity, for a
finite trajectory alphabet. -/
theorem ch04_ring_polar_denoising_score {Λ ι : Type*} [Fintype Λ] [Fintype ι] [DecidableEq ι]
    {t : ℝ} (ht : 0 < t) (w : Λ → ℝ) (a : Λ → ι → ℝ) (x : ι → ℝ) (c : ι)
    (hpos : ch04_ring_marginal w (fun l y => Real.exp (ch04_ring_channelLog t (a l) y)) x ≠ 0) :
    HasDerivAt
      (fun s => Real.log (ch04_ring_marginal w
        (fun l y => Real.exp (ch04_ring_channelLog t (a l) y)) (Function.update x c s)))
      ((Real.exp (-t) *
          (∑ l, ch04_ring_post w (fun l y => Real.exp (ch04_ring_channelLog t (a l) y)) x l
            * a l c) - x c) / ch04_Delta t)
      (x c) := by
  have hD : ch04_Delta t ≠ 0 := ne_of_gt (ch04_Delta_pos ht)
  have hd : ∀ l : Λ,
      HasDerivAt (fun s => (fun l y => Real.exp (ch04_ring_channelLog t (a l) y)) l
          (Function.update x c s))
        ((fun l y => Real.exp (ch04_ring_channelLog t (a l) y)) l x *
          (-(x c) / ch04_Delta t + (Real.exp (-t) / ch04_Delta t) * a l c)) (x c) := by
    intro l
    have hc := (ch04_ring_channel_score ht (a l) x c).exp
    have hrw : (Real.exp (-t) * a l c - x c) / ch04_Delta t
        = -(x c) / ch04_Delta t + (Real.exp (-t) / ch04_Delta t) * a l c := by
      field_simp
      ring
    rw [hrw] at hc
    simpa [Function.update_eq_self] using hc
  have hfisher := ch04_ring_fisher_identity w
      (fun l y => Real.exp (ch04_ring_channelLog t (a l) y))
      (fun l => -(x c) / ch04_Delta t + (Real.exp (-t) / ch04_Delta t) * a l c) x c hpos hd
  have haff := ch04_ring_post_affine w (fun l y => Real.exp (ch04_ring_channelLog t (a l) y)) x
      hpos (-(x c) / ch04_Delta t) (Real.exp (-t) / ch04_Delta t) (fun l => a l c)
  rw [haff] at hfisher
  have hval : -(x c) / ch04_Delta t + Real.exp (-t) / ch04_Delta t *
        (∑ l, ch04_ring_post w (fun l y => Real.exp (ch04_ring_channelLog t (a l) y)) x l * a l c)
      = (Real.exp (-t) *
          (∑ l, ch04_ring_post w (fun l y => Real.exp (ch04_ring_channelLog t (a l) y)) x l
            * a l c) - x c) / ch04_Delta t := by
    field_simp
    ring
  rw [← hval]
  exact hfisher

/-- eq:ring-polar-posterior (lines 620-628):
`p_ω(a|x) ∝ p(a_0) ∏_u K_ω(a_{u+1}|a_u) ∏_u N(x_u; e^{-t}a_u, Δ_t I₂)`.
Encoded as the normalised joint over a finite alphabet of trajectories. -/
def ch04_ring_polarJoint {A : Type*} (L : ℕ) (p0 : A → ℝ) (K : A → A → ℝ) (N : ℕ → A → ℝ)
    (a : ℕ → A) : ℝ :=
  p0 (a 0) * (∏ u ∈ Finset.range (L - 1), K (a u) (a (u + 1))) * ∏ u ∈ Finset.range L, N u (a u)

def ch04_ring_polarPost {Λ A : Type*} [Fintype Λ] (L : ℕ) (p0 : A → ℝ) (K : A → A → ℝ)
    (N : ℕ → A → ℝ) (traj : Λ → ℕ → A) (l : Λ) : ℝ :=
  ch04_ring_polarJoint L p0 K N (traj l) / ∑ l', ch04_ring_polarJoint L p0 K N (traj l')

/-- The posterior of eq:ring-polar-posterior is a probability vector proportional
to the joint: the proportionality constant is the same for every trajectory. -/
theorem ch04_ring_polarPost_sum {Λ A : Type*} [Fintype Λ] (L : ℕ) (p0 : A → ℝ) (K : A → A → ℝ)
    (N : ℕ → A → ℝ) (traj : Λ → ℕ → A)
    (h : (∑ l', ch04_ring_polarJoint L p0 K N (traj l')) ≠ 0) :
    (∑ l, ch04_ring_polarPost L p0 K N traj l) = 1 ∧
      ∀ l, ch04_ring_polarPost L p0 K N traj l
        = (1 / ∑ l', ch04_ring_polarJoint L p0 K N (traj l'))
          * ch04_ring_polarJoint L p0 K N (traj l) := by
  constructor
  · simp only [ch04_ring_polarPost, ← Finset.sum_div]
    exact div_self h
  · intro l
    rw [ch04_ring_polarPost]
    ring

/-! ### §4.2  The gauge on densities and scores -/

/-- The linear functional `v ↦ ∑ i, g i * v i`, the representation of a gradient
by a vector. -/
def ch04_dual {n : ℕ} (g : Fin n → ℝ) : (Fin n → ℝ) →L[ℝ] ℝ :=
  ∑ i, (g i) • (ContinuousLinearMap.proj i)

@[simp] lemma ch04_dual_apply {n : ℕ} (g v : Fin n → ℝ) :
    ch04_dual g v = ∑ i, g i * v i := by
  simp [ch04_dual, ContinuousLinearMap.sum_apply]

/-- A gradient pairs with a point through the transpose:
`∑_b S_b (U v)_b = ∑_c (Uᵀ S)_c v_c`. -/
lemma ch04_dual_comp {n : ℕ} (U : Matrix (Fin n) (Fin n) ℝ) (S v : Fin n → ℝ) :
    (∑ b, S b * (U *ᵥ v) b) = ∑ c, (Uᵀ *ᵥ S) c * v c := by
  have h := ch04_dot_transpose U v S
  simp only [dotProduct] at h
  rw [← h]
  exact Finset.sum_congr rfl fun c _ => by ring

/-- eq:ring-gauge-score (lines 233-238), second identity, and eq:ring-recap-gauge
(lines 307-308): if the non-rotating log-density `f` has gradient `S` at `U z`,
then `z ↦ f(U z)` has gradient `Uᵀ S(U z)` — a point transforms with `U`, a
gradient with `Uᵀ`. -/
theorem ch04_ring_gauge_score {n : ℕ} (U : Matrix (Fin n) (Fin n) ℝ)
    (f : (Fin n → ℝ) → ℝ) (S : Fin n → ℝ) (z : Fin n → ℝ)
    (hf : HasFDerivAt f (ch04_dual S) (U *ᵥ z)) :
    HasFDerivAt (fun w => f (U *ᵥ w)) (ch04_dual (Uᵀ *ᵥ S)) z := by
  have hlin : HasFDerivAt (fun w : Fin n → ℝ => U *ᵥ w)
      ((Matrix.mulVecLin U).toContinuousLinearMap) z :=
    (Matrix.mulVecLin U).toContinuousLinearMap.hasFDerivAt
  have hCLM : (ch04_dual S).comp ((Matrix.mulVecLin U).toContinuousLinearMap)
      = ch04_dual (Uᵀ *ᵥ S) := by
    apply ContinuousLinearMap.ext
    intro v
    show ch04_dual S ((Matrix.mulVecLin U).toContinuousLinearMap v) = ch04_dual (Uᵀ *ᵥ S) v
    rw [ch04_dual_apply, ch04_dual_apply]
    exact ch04_dual_comp U S v
  rw [← hCLM]
  exact hf.comp z hlin

/-- An orthogonal matrix preserves the squared norm — the one analytic fact the
change of variables of Lemma~\ref{lem:ring-gauge} needs. -/
lemma ch04_orth_normSq {ι : Type*} [Fintype ι] [DecidableEq ι] {U : Matrix ι ι ℝ}
    (hU : Uᵀ * U = 1)
    (u : ι → ℝ) : (∑ i, ((U *ᵥ u) i) ^ 2) = ∑ i, (u i) ^ 2 := by
  have h := ch04_dot_transpose U u (U *ᵥ u)
  rw [Matrix.mulVec_mulVec, hU, Matrix.one_mulVec] at h
  simp only [dotProduct] at h
  calc (∑ i, ((U *ᵥ u) i) ^ 2)
      = ∑ i, (U *ᵥ u) i * (U *ᵥ u) i := Finset.sum_congr rfl fun i _ => by ring
    _ = ∑ i, u i * u i := h.symm
    _ = ∑ i, (u i) ^ 2 := Finset.sum_congr rfl fun i _ => by ring

/-- An orthogonal matrix is invertible with `U⁻¹ = Uᵀ`, so `U Uᵀ = I` as well. -/
lemma ch04_orth_mul_transpose {ι : Type*} [Fintype ι] [DecidableEq ι] {U : Matrix ι ι ℝ}
    (hU : Uᵀ * U = 1) : U * Uᵀ = 1 := by
  have hdetU : IsUnit U.det := by
    have h := congrArg Matrix.det hU
    rw [Matrix.det_mul, Matrix.det_transpose, Matrix.det_one] at h
    refine isUnit_iff_ne_zero.mpr ?_
    intro h0
    rw [h0] at h
    norm_num at h
  rw [← Matrix.inv_eq_left_inv hU]
  exact Matrix.mul_nonsing_inv U hdetU

/-- The isotropic Gaussian density `N(x; m, Δ I)` on a finite index set, with its
normalising constant. -/
def ch04_gaussDensity {ι : Type*} [Fintype ι] (Δ : ℝ) (m x : ι → ℝ) : ℝ :=
  Real.rpow (2 * Real.pi * Δ) (-(Fintype.card ι : ℝ) / 2) *
    Real.exp (-(∑ i, (x i - m i) ^ 2) / (2 * Δ))

/-- The noised law of eq:ring-channel over a finite clean alphabet:
`P_t(x) = ∑_l w_l N(x; e^{-t}a_l, Δ_t I)`, the mixture form of the marginal
`p_t(x) = ∫ p(x|a)p(a) da` used throughout this audit. -/
def ch04_ring_noisedDensity {ι Λ : Type*} [Fintype ι] [Fintype Λ] (w : Λ → ℝ)
    (a : Λ → ι → ℝ) (t : ℝ) (x : ι → ℝ) : ℝ :=
  ∑ l, w l * ch04_gaussDensity (ch04_Delta t) (fun i => Real.exp (-t) * a l i) x

/-- eq:ring-gauge-score (lines 233-238), first identity, in general:
`P_t(z|ψ) = P^{(0)}_t(U_ψ z)`.

By the first half of Lemma~\ref{lem:ring-gauge} (eq:ring-gauged-rw, proved here
as `ch04_ring_traj_gauge` and `ch04_ring_cleanJoint_gauge`) the clean
trajectories of the rotating model are the de-rotated ones read backwards,
`a^{(ψ)} = Uᵀ a^{(0)}`.  Its noised density at `z` is then literally the
non-rotating noised density at `U z`: the de-rotation commutes with the channel,
because `U` preserves the isotropic Gaussian exponent.  General in the index set,
in the clean alphabet, in `t`, and for every orthogonal `U`. -/
theorem ch04_gauge_noisedDensity {ι Λ : Type*} [Fintype ι] [DecidableEq ι] [Fintype Λ]
    {U : Matrix ι ι ℝ} (hU : Uᵀ * U = 1) (w : Λ → ℝ) (a : Λ → ι → ℝ) (t : ℝ) (z : ι → ℝ) :
    ch04_ring_noisedDensity w (fun l => Uᵀ *ᵥ a l) t z
      = ch04_ring_noisedDensity w a t (U *ᵥ z) := by
  have hUU : U * Uᵀ = 1 := ch04_orth_mul_transpose hU
  have key : ∀ l : Λ, (∑ i, (z i - Real.exp (-t) * (Uᵀ *ᵥ a l) i) ^ 2)
      = ∑ i, ((U *ᵥ z) i - Real.exp (-t) * a l i) ^ 2 := by
    intro l
    have h1 : (fun i => z i - Real.exp (-t) * (Uᵀ *ᵥ a l) i)
        = z - Real.exp (-t) • (Uᵀ *ᵥ a l) := by
      funext i
      simp [Pi.sub_apply, Pi.smul_apply, smul_eq_mul]
    have hvec : (U *ᵥ (fun i => z i - Real.exp (-t) * (Uᵀ *ᵥ a l) i))
        = fun i => (U *ᵥ z) i - Real.exp (-t) * a l i := by
      rw [h1, Matrix.mulVec_sub, Matrix.mulVec_smul, Matrix.mulVec_mulVec, hUU,
        Matrix.one_mulVec]
      funext i
      simp [Pi.sub_apply, Pi.smul_apply, smul_eq_mul]
    have h := ch04_orth_normSq hU (fun i => z i - Real.exp (-t) * (Uᵀ *ᵥ a l) i)
    rw [hvec] at h
    exact h.symm
  simp only [ch04_ring_noisedDensity, ch04_gaussDensity]
  exact Finset.sum_congr rfl fun l _ => by rw [key l]

/-- eq:ring-gauge-score, first identity, for the chapter's own gauge
`U_ψ = diag(I₂, R_{-ψ}, …, R_{-(L-1)ψ})`: the Jacobian factor of the change of
variables is one, and the rotating model's noised density at `z` is the
non-rotating one at `U_ψ z`.  General in `L`, `ψ`, `t` and the alphabet. -/
theorem ch04_ring_gauge_density (L : ℕ) (ψ : ℝ) {Λ : Type*} [Fintype Λ] (w : Λ → ℝ)
    (a : Λ → (Fin 2 × Fin L) → ℝ) (t : ℝ) (z : Fin 2 × Fin L → ℝ) :
    |(ch04_ring_U L ψ).det| = 1 ∧
      ch04_ring_noisedDensity w (fun l => (ch04_ring_U L ψ)ᵀ *ᵥ a l) t z
        = ch04_ring_noisedDensity w a t (ch04_ring_U L ψ *ᵥ z) := by
  refine ⟨?_, ch04_gauge_noisedDensity (ch04_ring_U_orthogonal L ψ).1 w a t z⟩
  rw [(ch04_ring_U_orthogonal L ψ).2]
  norm_num

/-- Theorem 4.1 (marginal blindness, lines 518-536) for the surrogate, at the
*initial* frame: the ring prior is rotation invariant, so the law of the rotated
first frame does not depend on `ψ`; the same invariance is what carries through
the isotropic innovations and the isotropic channel.  This is only the base
case.  The statement for an arbitrary frame `k` is
`ch04_ring_one_frame_blind_surrogate`, proved from the whole-chain gauge
`ch04_ring_traj_gauge`; the polar model's version is `ch04_ring_one_frame_blind`. -/
theorem ch04_ring_marginal_blindness (lam ψ : ℝ) (z : Fin 2 → ℝ) :
    ch04_ring_prior lam (ch04_R ψ *ᵥ z) = ch04_ring_prior lam z :=
  ch04_ring_prior_rotation_invariant lam ψ z

/-! ### §4.5  Chapter conclusion: the one-frame / joint-trajectory separation

`eq:ring-chapter-message` (lines 663-672) is a boxed slogan,

    one-frame score        ⟶ geometry,
    joint trajectory score ⟶ geometry and dynamics,

so it is not itself an identity.  What it asserts is a *dichotomy*, and the
chapter states both halves precisely in the two sections the box summarises:
Theorem~\ref{thm:ring-marginal-blindness} (lines 518-536) and the scope notes
of lines 633-643.  Both halves are proved below, in the strongest form the
chapter claims and general in every parameter:

* the *blind* half is an equality, not a bound: a one-frame average is
  literally the same number for every value of the angular drift, so its
  derivative in the drift is `0` (zero Fisher information);
* the *informative* half is an identifiability statement: the interior-frame
  joint score **determines** the rotation, exactly up to the `2π` aliasing the
  chapter allows, and it separates `ψ` from `-ψ` precisely when the chapter
  says it does.
-/

/-- `R_0 = I₂`. -/
@[simp] lemma ch04_R_zero : ch04_R (0 : ℝ) = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [ch04_R]

/-- Scope note, lines 634-637: "`ψ` and `ψ + 2π` generate identical laws".  The
parameter enters only through the one-step rotation, and that rotation is
`2π`-periodic, so from discrete frames `ψ` is identifiable at most modulo `2π`. -/
theorem ch04_ring_rotation_alias_two_pi (ψ : ℝ) :
    ch04_R (ψ + 2 * Real.pi) = ch04_R ψ := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [ch04_R, Real.cos_add_two_pi, Real.sin_add_two_pi]

/-! #### The blind half: one frame sees only the geometry -/

/-- The embedded plane point of `eq:ring-embedding` is `2π`-periodic in the
phase — the fact that makes a uniform phase absorb any angular shift. -/
theorem ch04_ring_embed_periodic (r : ℝ) :
    Function.Periodic (ch04_ring_embed r) (2 * Real.pi) := by
  intro θ
  funext i
  fin_cases i <;>
    simp [ch04_ring_embed, Real.cos_add_two_pi, Real.sin_add_two_pi]

/-- The mechanism of Theorem~\ref{thm:ring-marginal-blindness} (lines 527-530):
"adding any independent angle to a uniform circular variable returns a uniform
variable".  For a `2π`-periodic observable the average over a uniform phase is
unchanged by an arbitrary shift `c` of that phase.  Stated for the unnormalised
integral over one period; the uniform expectation is this divided by `2π`, the
same constant on both sides. -/
theorem ch04_ring_uniform_phase_shift_invariant (F : ℝ → ℝ)
    (hF : Function.Periodic F (2 * Real.pi)) (c : ℝ) :
    (∫ θ in (0 : ℝ)..(2 * Real.pi), F (θ + c))
      = ∫ θ in (0 : ℝ)..(2 * Real.pi), F θ := by
  have h := hF.intervalIntegral_add_eq c 0
  rw [zero_add] at h
  simp only [intervalIntegral.integral_comp_add_right, zero_add]
  rw [add_comm (2 * Real.pi) c]
  exact h

/-- Theorem~\ref{thm:ring-marginal-blindness} for the polar model, lines 518-536.
Condition on the radius `r = R_u` and on the angular-noise realisation
`b = √(2D_θ)B_u` (both are independent of `Θ₀` and carry no `ω`); then
`Θ_u = Θ₀ + ωu + b` with `Θ₀` uniform on the circle, and the average of *any*
one-frame observable `g(a_u)` is the **same number for every** angular drift
`ω`.  This is an equality for all `ω`, i.e. the strongest form of "the one-frame
marginal is independent of the angular drift".  The remaining step — averaging
this equality over `r` and `b` — averages a function that is already constant
in `ω`, and is the only measure-theoretic ingredient not carried here. -/
theorem ch04_ring_one_frame_blind (g : (Fin 2 → ℝ) → ℝ) (r ω u b : ℝ) :
    (∫ θ in (0 : ℝ)..(2 * Real.pi), g (ch04_ring_embed r (θ + (ω * u + b))))
      = ∫ θ in (0 : ℝ)..(2 * Real.pi), g (ch04_ring_embed r θ) :=
  ch04_ring_uniform_phase_shift_invariant (fun θ => g (ch04_ring_embed r θ))
    (fun θ => congrArg g (ch04_ring_embed_periodic r θ)) (ω * u + b)

/-- The consequence drawn at lines 534-536: "every one-frame score carries zero
Fisher information about the rotation".  The one-frame expectation is constant
in `ω`, so its `ω`-derivative is `0` at every `ω`. -/
theorem ch04_ring_one_frame_zero_fisher (g : (Fin 2 → ℝ) → ℝ) (r u b ω : ℝ) :
    HasDerivAt (fun w : ℝ => ∫ θ in (0 : ℝ)..(2 * Real.pi),
        g (ch04_ring_embed r (θ + (w * u + b)))) 0 ω := by
  have hconst : (fun w : ℝ => ∫ θ in (0 : ℝ)..(2 * Real.pi),
      g (ch04_ring_embed r (θ + (w * u + b))))
      = fun _ : ℝ => ∫ θ in (0 : ℝ)..(2 * Real.pi), g (ch04_ring_embed r θ) := by
    funext w
    exact ch04_ring_one_frame_blind g r w u b
  rw [hconst]
  exact hasDerivAt_const ω _

/-- The surrogate trajectory generated by `eq:ring-sur-dynamics` from an initial
frame `z₀` and innovations `η`. -/
def ch04_ring_traj (ψ σ : ℝ) (z0 : Fin 2 → ℝ) (η : ℕ → Fin 2 → ℝ) : ℕ → (Fin 2 → ℝ)
  | 0 => z0
  | k + 1 => ch04_ring_step ψ σ (ch04_ring_traj ψ σ z0 η k) (η (k + 1))

@[simp] lemma ch04_ring_traj_zero (ψ σ : ℝ) (z0 : Fin 2 → ℝ) (η : ℕ → Fin 2 → ℝ) :
    ch04_ring_traj ψ σ z0 η 0 = z0 := rfl

@[simp] lemma ch04_ring_traj_succ (ψ σ : ℝ) (z0 : Fin 2 → ℝ) (η : ℕ → Fin 2 → ℝ) (k : ℕ) :
    ch04_ring_traj ψ σ z0 η (k + 1)
      = ch04_ring_step ψ σ (ch04_ring_traj ψ σ z0 η k) (η (k + 1)) := rfl

/-- With `ψ = 0` the surrogate step is a plain random-walk step. -/
lemma ch04_ring_step_zero (σ : ℝ) (z η : Fin 2 → ℝ) :
    ch04_ring_step 0 σ z η = z + σ • η := by
  simp [ch04_ring_step]

/-- The de-rotated innovations of `eq:ring-gauged-rw`, `η̃_j = R_{-jψ} η_j`. -/
def ch04_ring_gaugedNoise (ψ : ℝ) (η : ℕ → Fin 2 → ℝ) : ℕ → Fin 2 → ℝ :=
  fun j => ch04_R (-(j : ℝ) * ψ) *ᵥ η j

@[simp] lemma ch04_ring_gaugedNoise_apply (ψ : ℝ) (η : ℕ → Fin 2 → ℝ) (j : ℕ) :
    ch04_ring_gaugedNoise ψ η j = ch04_R (-(j : ℝ) * ψ) *ᵥ η j := rfl

/-- Lemma~\ref{lem:ring-gauge} run over the **whole chain** rather than one step:
for every frame index `k`, the `ψ`-chain is the non-rotating (`ψ = 0`) chain
driven by the de-rotated innovations `R_{-jψ}η_j`, then turned by `R_{kψ}`.
Proved by induction on `k`, so it is general in the trajectory length — the
one-step statement `ch04_ring_gauge_step` is the case that feeds the induction. -/
theorem ch04_ring_traj_gauge (ψ σ : ℝ) (z0 : Fin 2 → ℝ) (η : ℕ → Fin 2 → ℝ) (k : ℕ) :
    ch04_ring_traj ψ σ z0 η k
      = ch04_R ((k : ℝ) * ψ) *ᵥ ch04_ring_traj 0 σ z0 (ch04_ring_gaugedNoise ψ η) k := by
  induction k with
  | zero => simp
  | succ k ih =>
    have hang1 : ψ + (k : ℝ) * ψ = ((k + 1 : ℕ) : ℝ) * ψ := by push_cast; ring
    have hang2 : ((k + 1 : ℕ) : ℝ) * ψ + -((k + 1 : ℕ) : ℝ) * ψ = 0 := by ring
    have hL : ch04_ring_traj ψ σ z0 η (k + 1)
        = ch04_R (((k + 1 : ℕ) : ℝ) * ψ)
            *ᵥ ch04_ring_traj 0 σ z0 (ch04_ring_gaugedNoise ψ η) k
          + σ • η (k + 1) := by
      rw [ch04_ring_traj_succ, ih]
      simp only [ch04_ring_step]
      rw [ch04_R_mulVec_comp, hang1]
    have hR : ch04_R (((k + 1 : ℕ) : ℝ) * ψ)
          *ᵥ ch04_ring_traj 0 σ z0 (ch04_ring_gaugedNoise ψ η) (k + 1)
        = ch04_R (((k + 1 : ℕ) : ℝ) * ψ)
            *ᵥ ch04_ring_traj 0 σ z0 (ch04_ring_gaugedNoise ψ η) k
          + σ • η (k + 1) := by
      rw [ch04_ring_traj_succ, ch04_ring_step_zero, Matrix.mulVec_add, Matrix.mulVec_smul,
        ch04_ring_gaugedNoise_apply, ch04_R_mulVec_comp, hang2, ch04_R_zero,
        Matrix.one_mulVec]
    rw [hL, hR]

/-- Theorem~\ref{thm:ring-marginal-blindness} for the surrogate (lines 531-533).
By `ch04_ring_traj_gauge`, `ψ` enters frame `k` of the trajectory **only** as
the rotation `R_{kψ}`, so every rotation-invariant one-frame statistic takes the
same value on the `ψ`-chain as on the non-rotating chain driven by the
de-rotated innovations.  The chapter's remaining ingredients are already proved:
the initial law is rotation invariant (`ch04_ring_prior_rotation_invariant`) and
the innovation law is isotropic (`ch04_ring_isotropy`), so the de-rotated chain
has the same law as the original one and the `ψ`-dependence disappears. -/
theorem ch04_ring_one_frame_blind_surrogate (ψ σ : ℝ) (z0 : Fin 2 → ℝ)
    (η : ℕ → Fin 2 → ℝ) (k : ℕ) (g : (Fin 2 → ℝ) → ℝ)
    (hg : ∀ (φ : ℝ) (z : Fin 2 → ℝ), g (ch04_R φ *ᵥ z) = g z) :
    g (ch04_ring_traj ψ σ z0 η k)
      = g (ch04_ring_traj 0 σ z0 (ch04_ring_gaugedNoise ψ η) k) := by
  rw [ch04_ring_traj_gauge, hg]

/-! #### The informative half: the trajectory sees the dynamics -/

/-- The interior-frame joint score of `eq:ring-rotation-in-score`, named as a
function of the rotation parameter.  `ch04_ring_jointScore_is_score` below
records that this really is the gradient of the clean surrogate log-density. -/
def ch04_ring_jointScore (ψ σ : ℝ) (zprev zk znext : Fin 2 → ℝ) : Fin 2 → ℝ :=
  fun i => 1/σ^2 * ((ch04_R ψ *ᵥ zprev) i + ((ch04_R ψ)ᵀ *ᵥ znext) i - 2 * zk i)

/-- `ch04_ring_jointScore ψ σ` is the interior-frame score: this is
`ch04_ring_rotation_in_score` restated through the named function. -/
theorem ch04_ring_jointScore_is_score {σ : ℝ} (hσ : σ ≠ 0) (ψ : ℝ)
    (zprev zk znext : Fin 2 → ℝ) (i : Fin 2) :
    HasDerivAt
      (fun s => ch04_ring_cleanLocal (ch04_R ψ) σ zprev (Function.update zk i s) znext)
      (ch04_ring_jointScore ψ σ zprev zk znext i) (zk i) :=
  ch04_ring_rotation_in_score hσ ψ zprev zk znext i

/-- The `2π` aliasing at the level of the score: `ψ` and `ψ + 2π` give the same
interior-frame score at every configuration (lines 634-637). -/
theorem ch04_ring_jointScore_alias_two_pi (ψ σ : ℝ) (zprev zk znext : Fin 2 → ℝ) :
    ch04_ring_jointScore (ψ + 2 * Real.pi) σ zprev zk znext
      = ch04_ring_jointScore ψ σ zprev zk znext := by
  unfold ch04_ring_jointScore
  rw [ch04_ring_rotation_alias_two_pi]

/-- The informative half of `eq:ring-chapter-message`, in its strongest form:
the interior-frame joint score **determines** the rotation parameter.  If two
parameters produce the same score at every configuration of the three frames,
they differ by an integer multiple of `2π` — exactly the aliasing the chapter
allows (lines 634-637) and nothing more.  Together with
`ch04_ring_jointScore_alias_two_pi` this is an exact characterisation: the joint
score sees `ψ` modulo `2π`, and sees it completely. -/
theorem ch04_ring_joint_score_identifies_rotation {σ : ℝ} (hσ : σ ≠ 0) (ψ ψ' : ℝ)
    (h : ∀ zprev zk znext : Fin 2 → ℝ,
      ch04_ring_jointScore ψ σ zprev zk znext = ch04_ring_jointScore ψ' σ zprev zk znext) :
    ∃ n : ℤ, ψ' - ψ = n * (2 * Real.pi) := by
  have hσ2 : σ ^ 2 ≠ 0 := pow_ne_zero 2 hσ
  have hone : (1 : ℝ) / σ ^ 2 ≠ 0 := one_div_ne_zero hσ2
  have key : ∀ i : Fin 2,
      (1 / σ ^ 2) * ((ch04_R ψ *ᵥ (![1, 0] : Fin 2 → ℝ)) i)
        = (1 / σ ^ 2) * ((ch04_R ψ' *ᵥ (![1, 0] : Fin 2 → ℝ)) i) := by
    intro i
    have hi := congrFun (h ![1, 0] 0 0) i
    simpa [ch04_ring_jointScore] using hi
  have hc : Real.cos ψ = Real.cos ψ' := by
    have h0 := mul_left_cancel₀ hone (key 0)
    simpa [ch04_R, Matrix.mulVec, dotProduct, Fin.sum_univ_two] using h0
  have hs : Real.sin ψ = Real.sin ψ' := by
    have h1 := mul_left_cancel₀ hone (key 1)
    simpa [ch04_R, Matrix.mulVec, dotProduct, Fin.sum_univ_two] using h1
  have hcos : Real.cos (ψ' - ψ) = 1 := by
    rw [Real.cos_sub, ← hc, ← hs]
    linear_combination Real.sin_sq_add_cos_sq ψ
  obtain ⟨n, hn⟩ := (Real.cos_eq_one_iff (ψ' - ψ)).1 hcos
  exact ⟨n, hn.symm⟩

/-- Scope note, lines 637-639: "within a fundamental interval `(-π, π]` the joint
law does distinguish `ψ` from `-ψ`, which is the sense in which the direction of
turning is knowable".  Exactly: the interior-frame score fails to separate `ψ`
from `-ψ` **iff** `sin ψ = 0`, and on `(-π, π]` the solutions of `sin ψ = 0` are
the two angles `ψ = 0, π`, at which `ψ` and `-ψ` are the same rotation anyway.
So the score distinguishes the two senses of rotation at every `ψ` where they
differ. -/
theorem ch04_ring_joint_score_orientation {σ : ℝ} (hσ : σ ≠ 0) (ψ : ℝ) :
    (∀ zprev zk znext : Fin 2 → ℝ,
        ch04_ring_jointScore ψ σ zprev zk znext
          = ch04_ring_jointScore (-ψ) σ zprev zk znext)
      ↔ Real.sin ψ = 0 := by
  constructor
  · intro h
    obtain ⟨n, hn⟩ := ch04_ring_joint_score_identifies_rotation hσ ψ (-ψ) h
    refine Real.sin_eq_zero_iff.2 ⟨-n, ?_⟩
    have : ((-n : ℤ) : ℝ) = -(n : ℝ) := by push_cast; ring
    rw [this]
    nlinarith [Real.pi_pos, hn]
  · intro h zprev zk znext
    have hR : ch04_R (-ψ) = ch04_R ψ := by
      ext i j
      fin_cases i <;> fin_cases j <;>
        simp [ch04_R, Real.cos_neg, Real.sin_neg, h]
    unfold ch04_ring_jointScore
    rw [hR]

/-- eq:ring-chapter-message (ch04-ring.tex lines 663-672), the boxed closing
slogan, stated as the dichotomy it asserts and proved in both halves.

`⟨i⟩` *One frame ⟶ geometry only.*  A one-frame average is literally the same
number for every angular drift `ω`: conditionally on the radius and the angular
noise, the drift is a shift of a uniform phase, and a uniform phase absorbs it.

`⟨ii⟩` *Joint trajectory ⟶ geometry and dynamics.*  The interior-frame joint
score determines the rotation parameter, exactly up to the `2π` aliasing the
chapter allows.

The box itself is prose; the pair of statements below is the mathematical
content it summarises, and neither half is weakened: the first is an equality
for all `ω` (not a smallness bound) and the second is identifiability (not a
mere "depends on `ψ`"). -/
theorem ch04_ring_chapter_message {σ : ℝ} (hσ : σ ≠ 0) :
    (∀ (g : (Fin 2 → ℝ) → ℝ) (r ω u b : ℝ),
        (∫ θ in (0 : ℝ)..(2 * Real.pi), g (ch04_ring_embed r (θ + (ω * u + b))))
          = ∫ θ in (0 : ℝ)..(2 * Real.pi), g (ch04_ring_embed r θ))
      ∧ (∀ ψ ψ' : ℝ,
          (∀ zprev zk znext : Fin 2 → ℝ,
              ch04_ring_jointScore ψ σ zprev zk znext
                = ch04_ring_jointScore ψ' σ zprev zk znext)
            → ∃ n : ℤ, ψ' - ψ = n * (2 * Real.pi)) :=
  ⟨fun g r ω u b => ch04_ring_one_frame_blind g r ω u b,
   fun ψ ψ' h => ch04_ring_joint_score_identifies_rotation hσ ψ ψ' h⟩

end

end ThesisAudit
