# Formal audit of *Markov Score Diffusion* — final report

Lean 4 (`leanprover/lean4:v4.33.0`) + Mathlib. Twelve modules under `ThesisAudit/`, 20,153 lines and
1,509 top-level declarations; per-formula verdicts in `AUDIT_RESULTS/*.json`.

Regenerated **14 September 2026**, after an adversarial fidelity pass, a strengthening pass, and a
second verification pass over the newest proofs. All totals below were recomputed from the JSON files
on this date; none were carried over from an earlier version of this report.

---

## Bottom line

**Every display formula in the thesis that carries a `\label` — all 299 of them across the ten
mathematical chapters — has been read against a Lean statement**, and 173 unlabelled displays and
inline claims were itemised as well, for **480 audited items** in total. All twelve Lean modules and
the root `ThesisAudit` build **green**, with **no `sorry`, no `axiom` and no `native_decide`**
(grep-verified, not merely inferred from the build).

Of the 480 items:

| verdict | count | what it means |
|---|---:|---|
| **proved** | **336** | Lean accepts a formalisation of the thesis's identity, in the generality the thesis states it (under the thesis's own modelling hypotheses) |
| **instance_checked** | 13 | only a special case, a weaker form, or a conditional version is established |
| **defined** | 125 | the display is a definition or a notational restatement; it asserts nothing on its own |
| **skipped** | 4 | not formalised at all |
| **discrepancy** | 2 | the thesis is wrong |

**125 of the 480 items assert nothing** (they are definitions). Of the **355 items that do assert
something, 336 are proved — 94.6%.** Read the other way, across all 480 recorded items, 70% are
proved outright and a further 26% are definitional transcriptions that were never candidates for
proof.

**Was any real mathematical error found? Yes.** The display-math audit found **two**, and both are
minor — neither touches a derivation, a theorem, or a reported experimental conclusion. (A companion
audit of *prose* found twelve more, summarised immediately below; read the two counts together, not
the first one alone.) The two are:

1. an **inline** formula in Chapter 8 for the stationary variance of the fitted chain that omits a
   squared-mean term (factor-of-2 error at the counterexample; the conclusion the sentence draws is
   unaffected);
2. a **linear-interpolation figure** in Chapter 10 that should read $3.2$ rather than $3.0$ (6%; the
   sentence itself says these interpolations "are quoted nowhere as results").

Both fixes are one-line edits and are given in full below.

**A second, separate audit of inline math and quoted numbers found considerably more.** The Lean pass
covers display math; because *both* errors it stumbled on were in prose, a companion pass checked the
~446 decimal-bearing inline claims directly. It found **37 further items, 12 of them outright
errors**, including a results table whose asymptotic column is 29% too large. Those are **not**
included in the counts above. See
[`PROSE_FINDINGS.md`](/Users/gloriabagnato/Code/Thesis/Diffusion/formal/PROSE_FINDINGS.md); the five
most consequential are summarised on this page.

**What this report does not claim.** 17 items are `instance_checked` or `skipped`, and every one of
them is listed below with its concrete obstruction. Eleven of the seventeen have the same cause and
it is not a defect in the thesis: **this Mathlib has no stochastic integral**, so Itô's lemma, the
generator identity, the Fokker–Planck derivation and Anderson's time-reversal theorem cannot be
*stated* here, let alone proved.

---

## DISCREPANCIES — the findings that require an edit to the thesis

### 1. Chapter 8 — the stationary variance of the fitted chain omits the squared mean

**File:** `thesis/chapters/ch08-em-parameters.tex`, **lines 58–60**.
Inline math in the paragraph attached to `eq:em-parameter-space`.
Recorded as `AUDIT_RESULTS/Ch08a.json` item `noname-inline-1`; Lean witnesses
`ch08a_stationary_variance`, `ch08a_stationary_variance_not_second_moment`.

**Claim as written:**

> with $(\pi_c,\nu_c,s_c)$ free the fitted chain's stationary variance is
> $\sum_c \pi_c(\nu_c^2+s_c^2)/(1-\alpha^2)$, which equals one only under a variance-matching
> constraint this family deliberately does not impose, and its stationary mean is zero only if
> $\sum_c \pi_c \nu_c = 0$, which it likewise does not impose

**Why it is wrong.** For $a_i = \alpha a_{i-1} + \varepsilon_i$ the stationary variance $v$ satisfies
$v = \alpha^2 v + \operatorname{Var}(\varepsilon)$, hence
$v = \operatorname{Var}(\varepsilon)/(1-\alpha^2)$ with

$$\operatorname{Var}(\varepsilon)\;=\;\sum_c \pi_c(\nu_c^2+s_c^2)\;-\;\Big(\sum_c \pi_c \nu_c\Big)^{2}.$$

The printed expression is the innovation **second moment** over $1-\alpha^2$. It equals the
stationary variance only when $\sum_c \pi_c\nu_c = 0$ — *precisely the constraint that the very next
clause of the same sentence says the family does not impose*. Nor is it the stationary second moment,
which is $\big[\mathbb{E}\varepsilon^2 + 2\alpha m^2/(1-\alpha)\big]/(1-\alpha^2)$ with
$m=\sum_c\pi_c\nu_c$.

**Counterexample, verified in Lean.** Take $C=1$, $\pi_1=1$, $\nu_1=1$, $s_1=1$, $\alpha=0$.

* printed value: $\;1\cdot(1^2+1^2)/(1-0)=\mathbf{2}$
* true stationary variance: $\;[\,1\cdot(1^2+1^2)-(1\cdot 1)^2\,]/(1-0)=\mathbf{1}$

A factor of two, with $m = 1 \neq 0$. (`PROSE_FINDINGS.md` records the same error with a two-component
counterexample, $\pi=(\tfrac12,\tfrac12)$, $\nu=(1,1)$, $s^2=(1,1)$, $\alpha=0$.)

**Corrected statement:**

> with $(\pi_c,\nu_c,s_c)$ free the fitted chain's stationary variance is
> $\big[\sum_c \pi_c(\nu_c^2+s_c^2)-\big(\sum_c \pi_c\nu_c\big)^2\big]/(1-\alpha^2)$, …

Equivalently, keep the printed expression but say explicitly that $\sum_c \pi_c(\nu_c^2+s_c^2)$ is the
innovation *second moment*, so the formula is the stationary variance only under the mean condition
$\sum_c\pi_c\nu_c = 0$.

**Severity: minor.** The point the sentence is making — that the stationary variance need not equal
one — survives unchanged.

---

### 2. Chapter 10 — the break-even interpolation: the MLP crossing is 3.2, not 3.0

**File:** `thesis/chapters/ch10-comparison.tex`, **lines 1084–1086**.
Prose attached to `eq:em-breakeven`.
Recorded as `AUDIT_RESULTS/Ch10.json` item `eq:em-breakeven-b`; Lean witnesses
`ch10_em_breakeven_mlp_crossing`, `ch10_em_breakeven_mlp_discrepancy`,
`ch10_em_breakeven_cnn_crossing`, `ch10_em_breakeven_cnn_rounds`.

**Claim as written:**

> the CNN ratio falling from $1.2$ to $0.97$ across its bracket and the MLP one from $1.2$ to $0.84$
> across its own. A straight line through those endpoints crosses at $2.1$ and $3.0\times10^{-2}$
> respectively

**Reading.** "Those endpoints" are the two bracket endpoint pairs just named, so the sentence
describes two-point linear interpolation of the ratio against $\delta$.

**Counterexample, verified in Lean in exact rational arithmetic.** For the MLP, the line through
$(\delta,R) = (2.2\times10^{-2},\,1.2)$ and $(4.0\times10^{-2},\,0.84)$ has slope $-0.2$ per
$10^{-2}$ nats/site and reaches $R=1$ at

$$\delta \;=\; 2.2 + \frac{1.2-1.0}{1.2-0.84}\,(4.0-2.2) \;=\; 2.2 + 0.5556\times 1.8 \;=\; \mathbf{3.2}\times 10^{-2}$$

exactly (`ch10_em_breakeven_mlp_crossing`: `ch10_lin 2.2 1.2 4.0 0.84 3.2 = 1`). At the printed
$\delta = 3.0\times10^{-2}$ the interpolant is $\mathbf{1.04}$ — still **above** break-even, not at it
(`ch10_em_breakeven_mlp_discrepancy`: `ch10_lin 2.2 1.2 4.0 0.84 3.0 = 1.04` and `≠ 1`).

**The CNN number is right.** The line through $(1.1, 1.2)$ and $(2.2, 0.97)$ crosses $1$ at
$1.1 + 22/23 = 2.0565\times10^{-2}$, which rounds to the printed $2.1$.

**Corrected statement:**

> … A straight line through those endpoints crosses at $2.1$ and $3.2\times10^{-2}$ respectively

**Note for the author: no single interpolation convention reproduces the printed pair $(2.1, 3.0)$.**
Plain linear interpolation — which is what "a straight line through those endpoints" describes — gives
$(2.1, 3.2)$. Linear in $\log\delta$ gives $(2.0, 3.1)$. Log–log gives $(2.0, 3.0)$, so $3.0$ is
reproducible only under a convention that then contradicts the printed $2.1$. Under the reading the
text itself states, only the second number changes.

**Severity: minor.** The error is 6%, and the sentence explicitly says these interpolations "are
quoted nowhere as results". But the printed pair is internally inconsistent.

---

### Also read: `PROSE_FINDINGS.md` — 37 further items, 12 of them errors

The Lean audit covers **display math**. Because both errors it found were in *prose*, a companion pass
checked the ~446 decimal-bearing **inline** claims and quoted numbers against the thesis's own data
files and code. It filed **37 items: 12 errors, 16 inconsistencies, 8 rounding slips, 1 unverifiable
claim**. Full text and arithmetic for every one:
[`PROSE_FINDINGS.md`](/Users/gloriabagnato/Code/Thesis/Diffusion/formal/PROSE_FINDINGS.md).

The five most consequential:

1. **Ch. 9, Table 9.1 (`tab:em-rates`), lines 182–198 — a results table whose asymptotic column is
   29% too large.** The table prints the settling tolerances as *absolute* ($\tau = 1.2\times10^{-3}$,
   $6.7\times10^{-3}$); the sweep that produced every number in the chapter used *relative*
   tolerances ($10^{-3}$ for $\alpha$, $2\times10^{-2}$ for excess kurtosis). Recomputing
   $k(\tau,\lambda)=\ln\tau/\ln\lambda$ with the tolerances actually used gives $216$ and $976$, not
   the printed $211$ and $1256$, and the asymptotic ratio becomes $4.5$, not $5.9$. Fix: $\tau =
   1.0\times10^{-3}$ and $2.0\times10^{-2}$, $k = 216$ and $976$, ratio $4.5$, and propagate
   $5.9\to4.5$ at line 219. **Root cause** is the definition at lines 262–265, which states the
   settling criterion as an absolute distance; the absolute reading does not reproduce the chapter's
   own medians (it gives 416 updates for the shape coordinate against the reported 229).

2. **Ch. 9, lines 147–148 — the damping argument is stated backwards.** The text says (9.6) means the
   channel "removes an $e^{-4t}$ fraction" of the identifying structure. Equation (9.6) is
   $\kappa_4(X_t) = e^{-4t}\kappa_4(A)$: $e^{-4t}$ is the fraction that **survives**; the fraction
   removed is $1-e^{-4t}$. As written the sentence says the damage *shrinks* as $t$ grows — the
   opposite of the argument being made. Fix: "leaves only an $e^{-4t}$ fraction".

3. **Ch. 7, lines 367–369 — an order-of-magnitude claim that contradicts the chapter's own caption.**
   The text says the self-convergence error is "$10^4$ to $10^7$ times smaller"; the actual ratio over
   the 14 sweep points of `ch07_metrics.csv` runs from $1.0\times10^{3}$ to $8.1\times10^{4}$ — it
   never reaches $10^5$, and drops below $10^4$ at the two largest $t$. The caption of the same panel
   (lines 710–712) already states the correct range.

4. **Ch. 10, lines 893–899 (`eq:em-longrange`) — the printed matrix is not the matrix the code
   builds.** The diagonal compensation is printed as a sum over all $j\neq i$, but the off-diagonal it
   compensates is zero at $|i-j|=1$, and `thesis/tools/fig_ch10.py` sums only over $|i-j|\geq2$. Taken
   literally, the printed equation gives $\delta = 7.2\times10^{-3},\,1.24,\,1.81,\,2.34\times10^{-2}$
   against the table's $1.1,\,2.2,\,4.0,\,6.0\times10^{-2}$, and it destroys the "factor of 1.7"
   separation the section rests on (it becomes 1.10). Fix: change the diagonal subscript to
   $\sum_{|i-j|\geq2}$.

5. **Ch. 6, lines 571–575 — the claimed signature of inter-frame evidence sharing is false, and the
   thesis says so elsewhere.** The text argues that a positive $S_1$ at $x_1<0$ cannot come from a
   per-frame marginal score. But the per-frame marginal is $\mathcal{N}(0,1)$ at every $t$, so its
   score is exactly $-x_k$ — which at $x_1=-0.4$ is $+0.4$, i.e. positive, *necessarily*. Chapter 5
   states this verbatim at line 803. What is actually distinctive is the **magnitude** ($S_1 = +3.92$
   at $\alpha=0.8,\,t=0.05$ against the marginal's $+0.40$).

---

## Per-chapter table

`env` counts display environments in the chapter source
(`grep -c 'begin{equation\|begin{align\|begin{gather\|begin{multline'`); `items` counts recorded audit
items. Items may exceed environments because `align` blocks are split into one item per asserted
identity and because unlabelled and inline claims were itemised; they may fall short of a *file* count
where two modules split one file.

| Chapter | module | env | items | proved | instance_checked | defined | skipped | discrepancy | build |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 Introduction | `ThesisAudit.Ch01` | 1 | 1 | 0 | 0 | 1 | 0 | 0 | green |
| 2 Statistical mechanics & diffusion | `ThesisAudit.Ch02` | 57 | 56 | 32 | 10 | 10 | 4 | 0 | green |
| 3 Model | `ThesisAudit.Ch03` | 38 | 41 | 26 | 0 | 15 | 0 | 0 | green |
| 4 Ring | `ThesisAudit.Ch04` | 32 | 50 | 37 | 0 | 13 | 0 | 0 | green |
| 5 Gaussian matrix (part a) | `ThesisAudit.Ch05a` | 110 | 65 | 53 | 1 | 11 | 0 | 0 | green |
| 5 Gaussian matrix (part b) | `ThesisAudit.Ch05b` | *(same file)* | 68 | 59 | 1 | 8 | 0 | 0 | green |
| 6 Gaussian BP | `ThesisAudit.Ch06` | 42 | 49 | 35 | 1 | 13 | 0 | 0 | green |
| 7 Laplace | `ThesisAudit.Ch07` | 20 | 32 | 19 | 0 | 13 | 0 | 0 | green |
| 8 EM parameters (part a) | `ThesisAudit.Ch08a` | 71 | 40 | 20 | 0 | 19 | 0 | **1** | green |
| 8 EM parameters (part b) | `ThesisAudit.Ch08b` | *(same file)* | 40 | 30 | 0 | 10 | 0 | 0 | green |
| 9 EM kernel | `ThesisAudit.Ch09` | 4 | 4 | 4 | 0 | 0 | 0 | 0 | green |
| 10 Comparison | `ThesisAudit.Ch10` | 34 | 34 | 21 | 0 | 12 | 0 | **1** | green |
| **Total** | | **409** | **480** | **336** | **13** | **125** | **4** | **2** | **green** |

Root `lake build ThesisAudit` (all twelve modules plus `BasicIdentities`): **green**, 8,720 jobs,
zero errors.

*(`./audit_status.py` prints an `exp` column totalling 393, not 409. It is a different and older
coverage measure — a hand-maintained per-module expectation, counting labelled displays for some
chapters and all displays for others, and splitting the shared ch05 and ch08 files between their two
modules. The `env` column above is the raw `grep` count in the chapter source, one row per chapter,
and is the figure to check against the `.tex` files.)*

Two bookkeeping notes, so the table can be checked against the files:

* **Chapter 9's four verdicts live in `AUDIT_RESULTS/Ch01.json`**, which declares that it covers both
  `ch01-introduction.tex` and `ch09-em-kernel.tex`; that file's `modules` field records which
  theorems sit in `Ch01.lean` and which in `Ch09.lean`. `./audit_status.py` therefore prints a single
  `Ch01` row with 5 items where the table above shows Chapter 1 (1 item) and Chapter 9 (4 items)
  separately.
* **`ThesisAudit/ProseCh08.lean` is green but is *not* imported by the root**, so
  `lake build ThesisAudit` does not cover it. It holds the Lean witnesses for `PROSE_FINDINGS.md`,
  including `stationary_variance_counterexample`. Build it explicitly
  (`./safe_build.sh ThesisAudit.ProseCh08`) if you want those checked; the display-math discrepancy
  #1 above has its own witnesses inside `Ch08a.lean`, which *is* in the root build.

**Coverage cross-check (recomputed 14 September 2026).** Extracting every `\label` that sits inside a
display environment, chapter by chapter, gives 299 labels: ch01 1, ch02 42, ch03 24, ch04 32, ch05 75,
ch06 29, ch07 19, ch08 43, ch09 4, ch10 30. **Every one has a recorded item.** Three labels are split
across several items because the display asserts several identities (`eq:em-breakeven` → `-a`/`-b`,
`eq:ring-gauge-score` → `-a`/`-b`, `eq:ring-rot-props` → `-a`…`-d`), which is why 307 item ids carry a
label prefix against 299 labels. The remaining 173 items are unlabelled displays and inline claims.
Chapter 2's 14 unlabelled displays — a real gap in an earlier version of this report — are now
itemised as `noname-1` … `noname-14`.

**Status vocabulary.**
*proved* — Lean accepts a formalisation of the thesis's identity in the generality the thesis states
it (possibly under the thesis's own modelling hypotheses).
*instance_checked* — only a special case, a single instance, a side condition, a weaker form, or a
version conditional on assumed premises is established.
*defined* — the display is a definition, or the Lean statement is `rfl` against the audit's own
definitions and carries no independent content.
*skipped* — not formalised.
*discrepancy* — the thesis is wrong.

---

## What is NOT fully proved, and why

This section is the point of the audit, and it is deliberately not compressed. **17 items** are
`instance_checked` or `skipped`. Every one is listed here with its concrete obstruction, grouped by
cause. A further **125 items are `defined`** and assert nothing; that box is discussed at the end,
because "480 items audited" must not be read as "480 claims verified".

### A. No stochastic integral in Mathlib — 11 items

This is the single largest cause, it accounts for **11 of the 17**, and **it is not a defect in the
thesis**. The obstruction is absence, not difficulty:

> This Mathlib (toolchain `leanprover/lean4:v4.33.0`) has **no stochastic integral**.
> `Mathlib/Probability/BrownianMotion/Basic.lean` defines Brownian motion, but a search of the whole
> library for `stochasticIntegral` or `itoIntegral` returns nothing; there is no Itô formula, no
> "an Itô integral is a martingale so its expectation vanishes" lemma, and no solution concept for
> $dX = f\,dt + g\,dW$.

So the objects these displays are *about* — the solution of an SDE, its law, its quadratic variation
— cannot be written down at all. **Nothing weaker was substituted in their place**; where a fragment
*was* provable it is recorded as such and named.

**Not formalised at all (`skipped`, 4 items — all in Chapter 2's toolbox `tb:sm-fokkerplanck`):**

| item | tex lines | what it says | what exists |
|---|---|---|---|
| `noname-5` | 544–551 | Itô's lemma, $d\varphi(X_t) = \nabla\varphi\cdot dX_t + \tfrac12\sum_{ij}\partial_i\partial_j\varphi\,d\langle X^i,X^j\rangle_t$ | nothing — $dW_t$ and $d\langle X^i,X^j\rangle_t$ have no statements in this library |
| `noname-6` | 557–571 | the Itô expansion with isotropic noise, split into finite-variation and martingale parts | one step is pure algebra and **is** proved at arbitrary dimension: `ch02_noname_6_isotropic_collapse` collapses the double sum onto the Laplacian given $d\langle X^i,X^j\rangle = g^2\delta_{ij}dt$. The drift/martingale split is not formalised |
| `noname-7` | 575–583 | the integrated expansion $\mathbb{E}\varphi(X_t) = \mathbb{E}\varphi(X_0) + \mathbb{E}\!\int\!\dots ds + \mathbb{E}\!\int\!\dots dW_s$ | nothing — the text's next move, "the last term is an Itô integral … so its expectation is zero", is exactly the missing lemma |
| `eq:sm-generator` | 589–600 | $\tfrac{d}{dt}\mathbb{E}[\varphi(X_t)] = \mathbb{E}[(\mathcal{L}_t\varphi)(X_t)]$ | nothing — the display cannot even be *stated*: there is no object "the law of the solution of $dX = f\,dt + g\,dW$" to take $\mathbb{E}$ over |

**Checked in a weaker or conditional form (`instance_checked`, 7 items):**

* **`eq:sm-forward`** (ch02, 215–218) — the forward SDE $dX_t = f\,dt + g\,dW_t$. The SDE itself is
  unformalised. What *is* checked is the claim the surrounding text attaches to it: for the thesis's
  own $f(x,t)=-x$, $g=\sqrt2$, `ch02_sm_forward_terminal` proves the time-$t$ marginal converges
  pointwise to the standard Gaussian as $t\to\infty$, for an **arbitrary finitely supported** data law
  (any number of atoms, arbitrary locations and weights). Still not `proved` because it is checked for
  the thesis's $f$ and $g$, not a general drift/diffusion.
* **`eq:sm-ou`** (ch02, 240–245) — the OU transition law $x = e^{-t}a + \sqrt{\Delta_t}\varepsilon$,
  $\Delta_t = 1-e^{-2t}$. Checked four independent ways (both moment ODEs, variance preservation, and
  the full time-dependent density satisfying the OU Fokker–Planck equation), and the Fokker–Planck
  cross-check is now a corollary of the general linear-Gaussian theorem. **Two gaps remain:** nothing
  formalises that the solution of $dX = -X\,dt+\sqrt2\,dW$ *has* this transition law, and the initial
  condition $p_t\to\delta_a$ as $t\to0$ is not checked.
* **`noname-3`** (ch02, 286–292) — a boxed verbatim restatement of `eq:sm-ou`; inherits its status and
  its obstruction. What it *adds* — that the channel may be read coordinatewise — **is** proved at
  arbitrary dimension $d$ (`ch02_noname_3_factorises`).
* **`eq:sm-weakform`** (ch02, 619–626) — the weak form. The display itself is Steps 1–3 of the
  toolbox and is out of reach. What is proved is the load-bearing *use* the chapter makes of it:
  `ch02_sm_weakform_implies_fokkerplanck` derives `eq:sm-fokkerplanck` from the weak form plus the two
  integrations by parts plus the fundamental lemma of the calculus of variations, general in $f$, in
  $g$ and in the density. The item's own note is candid that this is "a conditional proof, and
  `instance_checked` is the nearest honest label in this audit's vocabulary".
* **`eq:sm-fokkerplanck`** (ch02, 688–698) — the Fokker–Planck equation. Substantially generalised:
  `ch02_fokkerplanck_linear_gaussian` covers the whole linear-Gaussian class (arbitrary
  time-dependent $f(x,t) = -\theta(t)x$, arbitrary $g(t)$, every $x$), with OU, the
  variance-preserving schedule and the variance-exploding schedule all instances. **Still missing:
  Steps 1–3**, i.e. the stochastic integral — the equation is not *derived* from the SDE, only shown
  equivalent to the weak form and verified on the linear-Gaussian solutions. *Corrected on 14
  September:* the moment ODEs are **hypotheses** of that theorem; the converse is never stated and
  neither is uniqueness, so the earlier note's "solves it exactly when" and "THE solutions of the
  moment ODEs the equation forces" overstated the Lean. "Solves it **whenever** the moment ODEs hold"
  is what was proved.
* **`eq:sm-reverse`** (ch02, 773–786) — Anderson's reverse-time SDE. Generalised to arbitrary
  dimension $d$: `ch02_sm_reverse_nd` shows the Fokker–Planck right-hand side of the reverse drift
  $-(f - g^2\nabla\log p)$ equals minus the forward one, so the factor $g^2$ in the boxed drift
  (against the probability flow's $g^2/2$) is pinned in every dimension. **Status is not about
  dimension:** the display is a *pathwise* statement about a process, and Anderson's theorem cannot be
  stated here at all.
* **`thm:g-decouple-iii`** (ch05, 1229–1244) — "the reverse-time dynamics decouple into $L$
  independent scalar linear Gaussian diffusions". The entire **coefficient-level** content of the
  thesis's own proof is formalised at arbitrary chain length: the eigen-coordinate drift field is
  coordinatewise scalar and linear (`ch05b_g_decouple_reverse`), the transformed noise is still
  isotropic (`ch05b_g_decouple_noise`), and the scalar diffusions really are time-inhomogeneous
  (`ch05b_g_decouple_inhomog`). **The remaining step — "the coefficients separate, therefore the
  solution processes are $L$ independent scalar diffusions" — is a statement about solutions of a
  time-inhomogeneous linear SDE, and is simply not formalised.**

### B. Proved on the line where the thesis states $\mathbb{R}^d$ — 4 items

All four are in Chapter 2's toolbox `tb:sm-fokkerplanck`, inside a derivation whose governing sentence
(tex line 641) says **"Integrate over $\mathbb{R}^d$"**. All four were recorded `proved` and were
**downgraded to `instance_checked` on 14 September** by the second verification pass. In each case the
mathematics is correct, the hypotheses are not weakened, and the one-dimensionality was disclosed in
the note — but `proved` is what a reader scans, and the $\nabla$/$\nabla\cdot$/$\Delta$ summaries read
as the $\mathbb{R}^d$ claim.

| item | tex | the display | the Lean |
|---|---|---|---|
| `noname-10` | 645–649 | $\int f\cdot\nabla\varphi\,p_t\,dx = -\int \varphi\,\nabla_x\!\cdot\![f p_t]\,dx$ | `ch02_noname_10_ibp_drift`, Ch02.lean 1882–1887: `∫ x, φ' x * fp x = -∫ x, φ x * fp' x`, a one-line instance of `ch02_ibp_real` |
| `noname-11` | 654–658 | $\int \Delta\varphi\,p_t\,dx = -\int\nabla\varphi\cdot\nabla p_t\,dx$ | `ch02_noname_11_ibp_lap1`, 1890–1895, same shape |
| `noname-12` | 661–665 | $-\int\nabla\varphi\cdot\nabla p_t\,dx = \int\varphi\,\Delta_x p_t\,dx$ | `ch02_noname_12_ibp_lap2`, 1898–1903 |
| `noname-13` | 672–682 | $\int\varphi\big[\partial_t p_t + \nabla\!\cdot\![fp_t] - \tfrac12 g^2\Delta_x p_t\big]dx = 0$ for every test function ⟹ the bracket vanishes | `ch02_noname_13`, 1914–1916: for `h : ℝ → ℝ` continuous with `∫ x, φ x * h x = 0` for all smooth compactly supported `φ`, `h = 0` |

In $\mathbb{R}^d$ the divergence is a sum over $d$ coordinates and the Laplacian a sum of $d$ second
derivatives; each identity needs Fubini in each coordinate, and none of that appears anywhere in the
module. Consequently the chaining the text draws from `noname-11` and `noname-12` — **formal
self-adjointness of the Laplacian on $\mathbb{R}^d$** — is established only on the line.

**`noname-13` deserves a separate word, because there the restriction is convenience, not
obstruction.** The Mathlib lemma actually invoked is already general:
`.lake/packages/mathlib/Mathlib/Analysis/Distribution/AEEqOfIntegralContDiff.lean:186`,
`ae_eq_zero_of_integral_contDiff_smul_eq_zero`, is stated for `E` an **arbitrary finite-dimensional
real normed space**. The $\mathbb{R}^d$ statement was available at almost no cost and was not taken.

**A convention that the reader should know about.** $d = 1$ is a module-wide encoding convention in
Chapter 2, and several items that remain `proved` rest on it — `eq:sm-probflow-continuity` carries the
note "Proved in one dimension", and `eq:sm-dsm` and `noname-8` were re-noted on 14 September to
disclose the same restriction (see the methodology note). The four items above were singled out for
downgrade because their displays sit inside an explicitly $\mathbb{R}^d$ derivation whose whole point
is the multi-coordinate bookkeeping.

### C. Only part of the display, or a weaker object — 2 items

* **`eq:g-gaussian-cond-precision`** (ch05, 550–556) —
  $\mathbb{E}[a_k\mid a_{\setminus k}] = -(Q_{kk})^{-1}\sum_{l\neq k}Q_{kl}a_l$.
  `ch05a_g_gaussian_cond_precision` proves that $-(Q_{kk})^{-1}S$ minimises $Q_{kk}x^2 + 2xS$ for
  $Q_{kk}>0$, with $S=\sum_{l\neq k}Q_{kl}a_l$. **Two gaps.** The objective is written by hand rather
  than obtained as the $k$-section of $a\cdot Q\,a$; and **no symmetry hypothesis is imposed on $Q$** —
  for non-symmetric $Q$ the $k$-section of $a^\top Q a$ is
  $Q_{kk}x^2 + x\sum_{l\neq k}(Q_{kl}+Q_{lk})a_l$, so the displayed minimiser is the thesis's only when
  $Q = Q^\top$. As stated, it is the true but content-free fact that $x=-S/q$ minimises $qx^2+2xS$ for
  $q>0$, with $Q$ supplying two scalars. (Contrast `ch05a_g_general_quadratic_form_three/four`, which
  *do* carry explicit symmetry hypotheses.)
* **`eq:bp-error-numerator`** (ch06, 666–677) —
  $\mathbb{E}\|(M-Q_t)x\|^2 = \operatorname{tr}[(M-Q_t)^\top(M-Q_t)\Sigma_t] = \operatorname{tr}[(M-Q_t)\Sigma_t(M-Q_t)^\top]$.
  The tex is a two-step chain; `ch06_bp_error_numerator` proves **only the second step** — pure trace
  cyclicity, with `Sg` an arbitrary matrix never tied to a covariance. The first and substantive
  equality, where the expectation and the law $x\sim\mathcal{N}(0,\Sigma_t)$ enter, appears in no Lean
  statement: `ch06_normSq_as_quad` and `ch06_bp_quadratic_trace` exist separately but are never
  composed into it, and `ch06_bp_quadratic_trace` models $\mathbb{E}$ as an unnormalised finite sample
  sum with $\Sigma$ *defined* as $\sum_i x_ix_i^\top$.

### D. The 125 `defined` items — counted for coverage, not verification

`defined` items assert nothing on their own. They are the thesis's definitions, transcribed, plus Lean
statements that hold by `rfl` against definitions the audit itself wrote. **They must not be read as
verification**, and they are 26% of the item count. Representative examples, each moved into this box
by the adversarial pass precisely because the Lean was `rfl` on the audit's own definition:
`ch08a_post P z = P z / ch08a_marg P` (Bayes' rule cannot fail when `ch08a_post` *is* that ratio);
`ch08a_marg P = ∑ z, P z`; `noname-9-b` in Chapter 3 (`mul_comm` on the audit's own abbreviation);
`noname-13` in Chapter 3 (a constant function is constant — it says nothing about the *model's*
one-frame marginal); `noname-17` in Chapter 6, where `eq:tweedie-joint` is imported from Chapter 2 as
an **unformalised premise**, so the chapter's final score formula is not verified there.

### E. Two items keep `proved` but with their content narrowed — read them with care

* **`eq:dev-propagator`** (ch03) is the chapter's **rejected** conjecture
  $S_{:,u}(x)\approx P_t(S_{:,u-1}(x))$. What Lean proves is its *negation*, in an exact form: it is a
  verified refutation, not a verified identity. Refuting *exact* propagation also does not refute the
  *approximate* ($\approx$) conjecture the chapter actually wrote, and the refutation covers only the
  pointwise, frame-local operator the display literally writes. Readers counting verified identities
  should exclude this one.
* **`eq:em-indicator-responsibility`** (ch08b) is
  `∑ d, (if d = c then 1 else 0) * r d = r c`, by `simp`. The variable `r` is an **arbitrary**
  real-valued function on `Fin C`: not `ch08b_em_responsibility`, not required to be nonnegative or to
  sum to one, with no hypothesis relating it to the label posterior. The statement is true of any `r`
  whatsoever, so it does not carry the display's content — that the conditional law of the label given
  $(a_{i-1},a_i,x)$ *is* the responsibility.

One further convention worth naming: **`eq:g-score-matrix` (ch05b) is `proved` without any Mathlib
derivative operator.** No `fderiv`/`gradient` is used; the gradient claim is formalised as an exact
first-order expansion with its explicit quadratic remainder, which is that module's declared
convention. Identifying that expansion with `gradient` would additionally require an
$o(\|v\|)$ estimate, which is not stated.

---

## Methodology

**How the audit was produced.** Each chapter's LaTeX source was enumerated display by display; each
display became an item in `AUDIT_RESULTS/ChNN.json` with a status, a Lean witness name, a summary and
a note. The Lean lives in `ThesisAudit/ChNN.lean` (twelve modules plus `BasicIdentities`, 20,153 lines,
1,509 top-level declarations),
all built against Mathlib at `leanprover/lean4:v4.33.0` through a lock-serialised `./safe_build.sh`.
Where the thesis's object was out of Mathlib's reach, the rule was to record the obstruction rather
than substitute a weaker statement and call it the display.

**Three review passes ran over the first draft of the audit, and each one lowered the headline.**

1. **An adversarial fidelity pass** re-read every recorded item of all twelve modules against *both*
   the LaTeX and the Lean, looking for records that claim more than the Lean establishes. It filed
   **58 issues** — 20 *vacuous* (the theorem assumes its own conclusion, or is `rfl` on the audit's own
   definitions), 19 *unfaithful* (the Lean statement is not the tex statement), 13 *inflated* (part of
   a display proved, all of it claimed) and 6 *missing* (a labelled display with no item at all) —
   recorded in `AUDIT_RESULTS/FIDELITY_*.json`. **33 items were downgraded** as a result (22
   *proved* → *instance_checked*, 10 *proved* → *defined*, 1 *defined* → *skipped*), 6 items were
   added, and 87 stale tex line pointers were regenerated against the current sources.
2. **A strengthening pass** then wrote the proofs the downgrades had asked for. Of the 31 downgrades
   still traceable as `[FIDELITY DOWNGRADE 2026-09-13]` paragraphs in the item notes, **20 are back at
   `proved`** — each carrying both the downgrade note and the repair note, so the round trip is
   visible — while 8 remain `defined` and 3 `instance_checked`. Several items were also generalised
   well past the thesis's own statement (the Fokker–Planck equation to the whole linear-Gaussian
   class; the reverse-drift identity to arbitrary dimension).
3. **A second verification pass (13–14 September)** re-checked the newest proofs — the ones written
   *after* the adversarial pass, which no adversary had yet seen — in Chapters 2, 5b and 8a, recorded
   in `AUDIT_RESULTS/VERIFY2_*.json`. Chapters 5b and 8a came back clean on all 14 verdicts. Chapter 2
   produced seven findings, all applied on 14 September: **four downgrades** (`noname-10`, `noname-11`,
   `noname-12`, `noname-13`, *proved* → *instance_checked*, Section B above) and **three note
   corrections where the status survived but the prose around it did not** — `eq:sm-dsm` (the proof
   survived an attempt to break it, but its "residual encoding restrictions" list omitted that the
   formalisation is one-dimensional and at a single fixed $t$, while claiming everything else was
   arbitrary), `noname-8` (same undisclosed restriction to the line) and `eq:sm-fokkerplanck` (the
   note claimed an "exactly when" equivalence the Lean proves in one direction only). Every edited
   item carries a `[VERIFY PASS 2, 2026-09-14]` paragraph giving the reason, and the pre-downgrade
   status is preserved in `status_before_verify2`.

**Nothing was quietly adjusted.** Every status change in this audit's history is recoverable from the
item's own `note` field and its `status_original` / `status_before_verify2` fields.

### What a green build does and does not mean

**It does mean:** Lean's kernel accepted these proofs. The audit contains no `sorry`, no `axiom` and
no `native_decide` — verified by grep over `ThesisAudit/*.lean`, not merely inferred from the build,
because a `sorry` only produces a *warning* and a build with one still goes green. The only matches
for those strings anywhere in the tree are in the root module's own header comment (`ThesisAudit.lean`
lines 14–15), which states that the audit contains none; the English word "admits" appears in three
docstrings and is not a tactic.

**It does not mean the formalisation captures every nuance of the prose.** A Lean theorem is only as
faithful as the encoding chosen for it, and the encoding is chosen by the same process that writes the
proof. This is not hypothetical: an adversarial pass over *this* audit found 58 places where the
record claimed more than the Lean established, and a second pass over the newest proofs found seven
more. Specific limits a reader should carry:

* **Dimension.** Chapter 2 formalises on $\mathbb{R}$ where the thesis writes $\mathbb{R}^d$
  (Section B). Several Chapter 2 items remain `proved` on that convention.
* **Integrals as finite sums.** Chapter 2's score-matching items use a finitely supported data prior
  and finite sums in place of expectations; Chapter 6 models $\mathbb{E}$ as an unnormalised sample
  sum.
* **Derivatives by expansion.** Chapter 5b's score claims are exact first-order expansions with
  explicit remainders, not Mathlib `gradient` statements.
* **Definitions are not theorems.** 125 items are `defined`; `rfl` against the audit's own definition
  proves nothing about the thesis.
* **Prose is outside this audit.** The Lean pass covers display math. `PROSE_FINDINGS.md` covers the
  inline claims, and it found 12 errors the display-math audit could not have seen.

The honest summary is that the audit is strong evidence about the *algebra* of the thesis and weaker
evidence about its *probability*, and that its two years of headline results — none of which any pass
managed to break — rest on identities that Lean now checks.

---

## How to reproduce

```bash
cd /Users/gloriabagnato/Code/Thesis/Diffusion/formal
lake build ThesisAudit     # all twelve modules + BasicIdentities; expect green, 8720 jobs
./audit_status.py          # per-chapter table, totals, and the discrepancies in full
```

`./audit_status.py -v` prints the discrepancy notes untruncated.

Individual modules build with `./safe_build.sh ThesisAudit.Ch02` and so on — use that script rather
than calling `lake` directly, since it serialises concurrent builds behind a lock.

To re-run the cheat sweep:

```bash
grep -n "sorry\|native_decide" ThesisAudit/*.lean ThesisAudit.lean
#   -> the only two hits are ThesisAudit.lean:14 and :15, the root module's own header
#      comment saying the audit contains none.  ThesisAudit/*.lean alone returns nothing.
grep -n "axiom" ThesisAudit/*.lean ThesisAudit.lean
#   -> the only hit is that same header comment, ThesisAudit.lean:14.
```

The prose witnesses build separately, since `ThesisAudit/ProseCh08.lean` is not imported by the root:

```bash
./safe_build.sh ThesisAudit.ProseCh08
```
