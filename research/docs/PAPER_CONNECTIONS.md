# Layer 6 — what the two papers imply for this project

Two papers were handed to the project as things to study and take inspiration
from. Both turn out to be by the advisors themselves, and together they define
almost exactly the two axes this package has been measuring blind.

- **P1** — Jérôme Garnier-Brun, Marc Mézard, Emanuele Moscato, Luca Saglietti,
  *How transformers learn structured data: insights from hierarchical
  filtering*, arXiv:2408.15138; ICML 2025 (PMLR 267:18831–18847).
- **P2** — Giulio Biroli, Tony Bonnaire, Valentin de Bortoli, Marc Mézard,
  *Dynamical regimes of diffusion models*, Nature Communications **15**, 9957
  (2024), doi:10.1038/s41467-024-54281-3; arXiv:2402.18491.

---

## 0. How these papers were read — and a correction

**P1 has now been read in full.** Direct HTTP is blocked at this environment's
gateway (`arxiv.org`, `www.nature.com`, PMC, HAL, PMLR all 403 on `CONNECT`),
but the Hugging Face Hub exposes `hf://papers/2408.15138/paper.md` through a
server-side route, which returns the complete LaTeXML rendering including the
appendices. Everything about P1 below is from the source.

**P2 has still not been read.** It is not on that route, and every direct route
is blocked. What is said about P2 here comes from web search — abstract-level
statements about the two cross-overs and the entropic criterion — and *nothing
mathematical is quoted from it*. The speciation crossover used throughout is
derived from scratch in §3.1 in this package's conventions and checked against
40 000 sampled forward trajectories. (One search summary rendered the condition
with a reciprocal, `t_S = ½ log(1/Λ)`, contradicting its own statement that
`t_S` diverges with dimension; deriving it independently avoided importing that
slip.) **Read P2 before citing it.**

### What reading P1 changed

Two things in an earlier version of this note were wrong. Both are corrected in
place below, and both were substantive rather than cosmetic:

1. **I had the filtering construction wrong.** I implemented "filter at level
   `k`" as *`b^k` independent subtrees*. P1 §2.2 instead draws the depth-`k`
   nodes **conditionally independently given the root**, with the marginals the
   unfiltered model gives them (their Eq. 1, obtained by tracing the transition
   tensor along the unique root→node path). The blocks therefore stay
   correlated *through the root*; at `k = L` the leaves become conditionally
   i.i.d. given the root, which is the regime where P1 notes a Naive Bayes
   classifier is optimal and attention is superfluous. Independent blocks would
   have put a zero where the correct construction puts `ρ^{2L}`. The
   implementation now follows the paper and a test pins the distinction
   (`test_filtered_sampler_matches_the_analytic_covariance`).

2. **I claimed the hierarchy × diffusion intersection was "untouched by either
   paper". It is not.** Sclocchi, Favero and Wyart, *A phase transition in
   diffusion models reveals the hierarchical nature of data*, PNAS 122(1)
   e2408799121 (2025), arXiv:2402.16991 — **cited by P1 itself** — studies
   exactly diffusion on hierarchically generated data and finds a phase
   transition at which the probability of reconstructing high-level features
   drops sharply, while low-level features evolve smoothly across the whole
   process. §2(c) is rewritten accordingly. The measurements here turn out to
   be *consistent* with that picture rather than novel against it, and saying
   so is more useful than the claim it replaces (see §5).

**A methodological correction that follows from (1).** P1's probe for
sequential learning is not "resolve the error onto levels and watch it move" —
which is what exp_13 `ordering` did, and it found nothing. It is to compare the
model's predictions against the *family* of mismatched oracles `BP_k`, and ask
which one it currently agrees with. `exp_15` implements that.

---

## 1. What the two papers say

### P1 — hierarchical filtering (read in full)

**The data model.** A root symbol `x_0` drawn from a vocabulary of size `q`,
then a transition *tensor* `M ∈ ℝ_+^{q×q×q}` with `M_{abc} = P(children (b,c) |
parent a)` applied for `ℓ` generations down a binary tree, yielding `2^ℓ`
leaves. Entries are log-normally sampled and **non-overlapping** — if
`M_{abc} > 0` then `M_{a'bc} = 0` for all `a' ≠ a` — so production rules are
unambiguous and, at `k = 0`, the whole tree including the root is
deterministically recoverable from the leaves. They call it a simplified PCFG,
and relate it to the Random Hierarchy Model of Cagnetta et al. (uniform rates,
layer-dependent rules) — the differences being log-normal versus uniform rates
and a single tensor throughout.

**Filtering (§2.2), stated precisely because I first got it wrong.** At
level `k`, the `2^k` nodes at depth `k` are drawn **conditionally independently
given the root**, with the marginals the unfiltered model assigns them —
Eq. 1, `P(x_j = b | x_0 = a) = (p_0 M^{σ_0(j)} ⋯ M^{σ_{k−1}(j)})_{ab}`, tracing
the tensor along the unique root→`j` path with `σ_m ∈ {L,R}` recording the
turns. Levels `1..k−1` are skipped. Below `k` the ordinary process resumes, so
correlations survive inside blocks of `2^{ℓ−k}` leaves **and between blocks
through the root**. At `k = ℓ` the leaves are conditionally i.i.d. given the
root, and the paper notes that a Naive Bayes classifier is then optimal and
every attention layer superfluous. Marginals are preserved at every `k`; what
filtering removes is the intermediate ancestry, not the root.

**Inference.** The graph is a tree at every `k`, so BP is exact and linear-time,
converging in `2(ℓ − k + 1)` steps — an upward and a downward pass. `BP_k` is
therefore a *family* of exact algorithms, one per filtering level, and running
`BP_k` on data generated at a different level is a well-defined mismatched
oracle. That family is the paper's measuring instrument.

**Findings.** (i) Transformers match `BP_{k_train}`'s accuracy *and its
calibrated probabilities*, in-sample and out-of-sample, including where `BP_k`
is not optimal — evidence they implement the algorithm rather than merely
scoring well. (ii) **Sequential acquisition**: during training the predictions
align first with `BP_ℓ` (leaf-to-root only) and then with *decreasing* `k`, a
clean "staircase" in the out-of-sample accuracy — shortest-range correlations
first. (iii) Attention maps develop blocks of size `~2^{ℓ−k}`, organized
hierarchically across layers; layer-wise probes recover ancestors up to the
probed depth. (iv) Appendix F gives an explicit construction embedding BP in
`ℓ` transformer layers, using `O(q²)` memory slots per token to fold the
downward pass into the upward one. (v) MLM pre-training sharply reduces the
labelled data needed for root classification.

### P2 — dynamical regimes

Diffusion models are analysed with statistical-physics methods in the regime of
large dimension and large sample count, with an optimally trained score. The
backward dynamics passes through two cross-overs:

- **Speciation** — the coarse structure of the sample is decided, by a mechanism
  analogous to symmetry breaking. Its time scale is set by the **top eigenvalue
  of the data covariance**; when that eigenvalue grows with dimension, the
  speciation time diverges logarithmically.
- **Collapse** — trajectories are captured by a single training point, by a
  mechanism analogous to glassy condensation. Its time scale is set by an
  **excess entropy** of the data, and the consequence is a curse of
  dimensionality: memorization is avoidable at finite times **only if the
  dataset is exponentially large in the dimension**.

The collapse statement is about the **empirical score** — the exact score of
the measure putting mass `1/N` on each training point, which is what an
unconstrained model with a perfectly minimized training loss converges to.

---

## 2. Why this is directly load-bearing here, not just adjacent

Three things line up, and one of them is a genuine gap in this project's story
that P2 fills.

**(a) P1's reference algorithm is this project's method.** P1 measures a
transformer against exact BP on a tree. Layers 1–5 here compute a diffusion
score by exact BP on a chain and measure a network against it. The difference is
the task, not the epistemics; the "structure first, then compute" stance is
identical. What P1 adds is the *hierarchy*: on a chain there is one length scale
and "which correlations has the model learned" has no graded answer. On a tree
it does. That is why `src/hierarchy.py` exists.

**(b) P2 explains the sample-efficiency result that Layer 5 measured but did
not explain.** Layer 5's headline is that EM-BP matches a score network with
≳64× less data, and that its error falls at the parametric `N^{−1/2}` rate
(measured slope −0.500 ± 0.048 at 12 replicates). That was reported as an
empirical fact. P2 supplies the mechanism, and it is sharper than "BP has an
inductive bias":

> The empirical score's sufficient statistic **is the training set** — `N × n`
> numbers, growing in both dataset size and dimension, which is why it must
> collapse onto that set unless `N` is exponential in `n`. A BP score's
> sufficient statistic is the fitted kernel: two numbers for a Gaussian AR(1),
> `M × M` for a nonparametric one, and **independent of `n`**. The data reaches
> it only through `Ξ`, which is an average. There is no training point for a
> trajectory to be captured by.

So the prediction is not "BP memorizes less by some margin". It is that **the
memorization axis does not exist for a BP score**, at any `n` and any `N`, while
for the empirical score it is governed by an exactly computable entropy. That
is falsifiable, and `exp_14` measures it.

**(c) The two devices compose — but the composition is not new, and an earlier
version of this note wrongly said it was.** P1's data model has a ladder of
length scales; P2's speciation time is set by a covariance eigenvalue. Putting
them together suggests a hierarchical prior should show not one speciation time
but `L + 1`, resolved coarse-to-fine.

**Sclocchi, Favero and Wyart (PNAS 2025, arXiv:2402.16991) already study
diffusion on hierarchical data**, and P1 cites them. They find a phase
transition at which the probability of reconstructing *high-level* features
drops sharply, while low-level features evolve *smoothly* across the whole
process. So the intersection is occupied, and the correct question is not "does
a hierarchy produce structure in diffusion time" — it does, and they showed it —
but what the Gaussian setting adds. Three things, and they are modest:

1. **The whole ladder is analytic.** The ultrametric covariance has a
   closed-form spectrum, so every crossover time is known before any experiment
   (§3.2), rather than located empirically. That is what makes `r = 0.9998`
   against prediction a meaningful check.
2. **Filtering acts on the ladder in a computable way** (§3.6): filtering at
   level `k` merges the top `k` rungs into one. That is P1's knob and P2's time
   scale in the same formula.
3. **The sharp/smooth split of Sclocchi et al. is reproduced and localized.**
   In a purely Gaussian tree every level gives a *smooth* crossover — there is
   no class to speciate into. Making only the root bimodal, with the covariance
   held bit-identical, produces genuine symmetry breaking at the coarsest level
   and leaves the rest smooth (§5, result 5). That is their qualitative finding
   in a setting where the control is exact.

None of this is a novelty claim against them. It is a claim that the Gaussian
tree is a useful calibration instrument for statements they make in a harder
setting.

---

## 3. What was derived, in this package's conventions

All of it verified in `tests/test_hierarchy.py` (25 tests) rather than asserted.

### 3.1 The speciation crossover

Along an eigendirection of the clean covariance with eigenvalue `Λ`, the
variance-preserving forward process splits into signal and noise,

    Var(⟨v, x_t⟩) = α_t² Λ + Δ_t,     α_t² = e^{−2t},  Δ_t = 1 − e^{−2t}.

The mode is decided while signal dominates, undecided once noise does, so

    **t_S(Λ) = ½ log(1 + Λ)**,

and the measurable statement is that the correlation between the projection at
time `t` and at time `0`,

    commitment(t, Λ) = α_t √Λ / √(α_t² Λ + Δ_t),

passes through `1/√2` exactly at `t_S`. Checked analytically
(`test_commitment_crosses_one_over_sqrt_two_at_the_speciation_time`) and against
40 000 sampled trajectories of the forward process, agreeing to < 0.02
(`test_commitment_matches_the_forward_process_empirically`).

**A distinction that must not be blurred.** In P2 speciation is a *symmetry
breaking*: the dynamics chooses between data classes, and the phenomenon needs
a multimodal distribution to have something to choose between. The prior used
here is Gaussian, so there is no class to speciate into. What `commitment`
measures is the **information crossover along an eigendirection** — the point
at which that mode's value stops being determined by the data and starts being
determined by the noise. This coincides with P2's *criterion* (it is the same
signal-versus-noise balance, set by the same eigenvalue) but it is not the same
*phenomenon*, and the results below should be read as locating the crossover,
not as observing symmetry breaking.

Making it a genuine speciation needs only a multimodal root: `tree_bp_grid`
already accepts an arbitrary `log_root`, so a two-component root prior is a
one-line change, and the level −1 transition would then be a real choice
between classes. That is listed as untried work rather than claimed.

### 3.2 A chain has no speciation cascade; a tree does

For a stationary AR(1) chain with `Cov = ρ^{|i−j|}`, the covariance spectrum is
bounded for every length by the spectral density at zero frequency,

    **Λ_max → (1 + ρ)/(1 − ρ)**,

so `t_S` *saturates*: 64× the chain length changes the top eigenvalue by under
2× (`test_chain_top_eigenvalue_saturates_at_the_limit`). **A stationary Markov
chain therefore has no diverging speciation time and no coarse-to-fine cascade
to resolve** — it sits outside the regime P2 describes for image data, and that
is a property of the data model, not of the method.

For the balanced-tree prior (`z_child = ρ z_parent + innovation`, leaves
observed) the leaf covariance is ultrametric, `Cov = ρ^{2(L − d_LCA)}`, and it
has exactly `L + 1` distinct eigenvalues, derived in closed form in
`GaussianTree.level_eigenvalues`:

    Λ_d = S_{d+1} − b^{L−d−1} ρ^{2(L−d)},   multiplicity (b−1) b^d,
    S_d = 1 + (b−1) Σ_{m=1}^{L−d} b^{m−1} ρ^{2m},

with the uniform mode at `Λ = S_0`. Verified against `eigh` of the dense
covariance to 1e-10 for three (branching, depth, ρ) combinations, with
multiplicities summing to the number of leaves and the trace to it as well.
When `bρ² > 1` the top eigenvalue grows geometrically in depth, hence
**logarithmically in dimension** — which is the regime P2's analysis describes,
now with an exactly known spectrum rather than an empirical one.

### 3.3 A closed-form collapse criterion for the chain

For a Gaussian AR(1) chain the differential entropy is exactly extensive, so the
per-site excess entropy relative to the terminal measure `N(0,1)` is

    **s = −½ log(1 − ρ²)**,

exactly, with no fitting — and the dataset size below which memorization is
forced is `exp(n s)`. At `ρ = 0.85` that is **0.641 nats per site**, so 33 sites
already demand ~10⁹ chains. Checked against `−½ log det C`
(`test_excess_entropy_matches_the_gaussian_determinant`).

At finite `t` the noised law is `N(0, α_t² C + Δ_t I)`, so

    s(t) = −(1/2n) log det(α_t² C + Δ_t I)

is available in closed form too, and `n s(t_C) = log N` gives a **predicted
collapse time with no fitted constants**. `exp_14` part `time` tests it.

### 3.4 Exact BP on a tree, two ways

- `tree_bp_gaussian` — information form `(h, λ)`, exact, `O(n)`. The upward and
  downward updates touch only those two numbers per message: the audit's
  finding F2 (that the previous Gaussian BP was doing "something strange"
  instead of updating mean and variance) applies here too, and the code is
  written so that there is nowhere for a projection step to hide. Matches a
  dense linear solve to **< 1e-10** at four noise levels and for branching 2
  and 3.
- `tree_bp_grid` — grid messages, any innovation law. Matches the information
  form to **< 2e-6**, and returns finite, visibly different answers on a
  Laplace-innovation tree where no closed form exists. The leave-one-out
  sibling product is computed by prefix/suffix scans rather than by dividing
  out a message, since message entries are legitimately ~1e-16 in the tails.

### 3.5 EM on a tree

`tree_e_step` produces the **same `Ξ` matrix** as the chain E-step — the
topology stops being visible at exactly that point, so **every kernel in
`src/kernels.py` consumes it unmodified**. A chain-trained M-step and a
tree-trained M-step are the same code. The evidence is recovered by accumulating
the log-scales discarded at each renormalization, and is verified against the
closed-form Gaussian marginal likelihood to a **relative 1e-6**, which is what
makes the monotone-ascent check available on trees as well.

**Measured, and worth knowing before running anything:** tree EM converges much
more slowly than chain EM. At depth 3 with 512 trees the estimate is
`ρ̂ = 0.7360` at 50 iterations, `0.7483` at 100 and `0.7487` (converged, true
value 0.75) at 150, with **zero monotone violation throughout**. On a chain
every site is observed; on a tree the internal nodes never are, the missing
information fraction is far larger, and EM's linear rate is correspondingly
slower. An estimate read off at 40 iterations looks like a broken M-step and is
merely unconverged — this cost real time to diagnose here and is now asserted
as a *rate* property in the tests rather than as a single-budget accuracy claim.

---

### 3.6 Filtering removes rungs from the speciation ladder

This is where P1's knob and P2's time scale meet in one formula. Under the
correct filtering (§0), the leaf covariance is block-diagonal-plus-constant: the
depth-`(L−k)` subtree covariance inside each of the `b^k` blocks, and `ρ^{2L}`
between them. Its eigenvectors are

1. **within-block contrasts** — the subtree's own non-uniform modes, each
   multiplicity multiplied by `b^k`;
2. **between-block contrasts**, constant on each block and summing to zero
   across blocks, with the single eigenvalue

       Λ_between = S_0^{(L−k)} − b^{L−k} ρ^{2L},   multiplicity b^k − 1;

3. the **uniform mode**, `Λ = S_0^{(L−k)} + (b^k − 1) b^{L−k} ρ^{2L}`.

So **the top `k` rungs merge into one**, and the number of distinct speciation
times falls from `L + 1` to `L − k + 2`, reaching 2 at `k = L` where the
covariance is equicorrelated. Measured at `L = 4`: **5, 5, 4, 3, 2** rungs for
`k = 0…4`.

Two checks fall out of this for free. The analytic spectrum matches `eigh` of
the dense covariance to **1e-15** at every `(L, k)` tried, and `BP_k` matches a
dense linear solve to **1e-15**. And `k = 0` and `k = 1` giving the same count
independently reproduces P1's remark that those two cases share a tree topology
and differ only in the top transition probabilities — which in the Gaussian
case means they do not differ at all (§5, result 7).

## 4. What was built

| Added | Purpose |
|---|---|
| `src/hierarchy.py` | Tree prior, analytic ultrametric spectrum, exact tree BP (information form + grid), tree E-step, `fit_em_tree` |
| `src/spectral.py` | Speciation time, commitment, chain spectrum and its limit, excess entropy, collapse dataset size |
| `experiments/exp_13_speciation_cascade.py` | `spectra`, `cascade`, `levels`, `ordering` |
| `experiments/exp_14_memorization_collapse.py` | `budget`, `collapse`, `time` |
| `experiments/exp_15_bp_oracles.py` | `ladder`, `alignment`, `mismatch` -- the mismatched-oracle probe |
| `tests/test_hierarchy.py` | 51 tests (suite 50 -> 101) |

---

## 5. What the experiments showed

Full tables are in `../compendium/` (the current compendium; the Aug 6 draft this
section originally cited is archived) and the CSVs under
`research/nongaussian-bp/outputs/`. The four results, in the order they matter:

**1. The speciation ladder exists and sits where predicted.** Depth-5 tree,
ρ = 0.9: six distinct transitions spanning 16× in time within one dataset, each
within 3.5% of `t_S = ½ log(1 + Λ)` along the forward process, and reproduced
by the reverse SDE under the exact tree-BP score. The reverse dynamics resolves
the hierarchy coarse-to-fine. **This is the question neither paper asks.**

**2. The chain studied in Layers 1–5 is outside that regime.** Its top
eigenvalue is bounded by `(1+ρ)/(1−ρ)` at any length, so `t_S` saturates and
there is no cascade. This bounds how far the project generalizes and is the
less comfortable of the two findings.

**3. Memorization does not have a BP analogue.** The empirical score sits at
0.33–0.66 on the nearest-training-neighbour ratio and recovers monotonically as
`N` crosses its entropic wall; EM-BP sits at 0.97–1.08 at every `n` and `N`,
tracking the true-prior reference within 0.03. This is the *mechanism* behind
Layer 5's headline, which had been reported as an empirical fact.

**4. The network's error is blind to the hierarchy; BP's is not.** Against the
null of uniform per-mode error, the concentration ratio spreads by 1.6–2.2×
across levels for the ε-network — flat, as if the levels did not exist — and by
224–6957× for EM-BP, migrating from the finest levels at small `t` to the single
coarsest mode at large `t`. The x₀-network is genuinely intermediate (up to
12.6×) and is reported as such.

**5. Symmetry breaking happens before the information crossover, and the
crossover itself does not move.** A two-component root keeps the covariance and
therefore every predicted time bit-identical — a controlled comparison, asserted
in a test. The correlation crossing lands at 1.155 (bimodal) and 1.031
(Gaussian) against a predicted 1.136, both inside the ±10% the integrator costs
at this resolution; the *class choice* for the bimodal root locks in at 1.265,
earlier in the generation and 35% earlier than the Gaussian root's. Two
separated classes can be told apart while noise still swamps the within-class
detail. This is the one measurement here that is speciation in P2's sense rather
than a crossover sharing its criterion.

**6. The advantage does not shrink as correlations reach further** (filtering,
**PARTIAL**). Sweeping the filter level `k`, the network's error is flat
(0.414 → 0.347) while EM-BP's advantage runs 29–99×. But fixing the number of
sequences does not fix the number of edges — 16, 24, 28, 30 for k = 1…4 — so
larger `k` hands EM more data at the same nominal budget and the sweep moves two
things at once. The *rising* advantage is therefore not attributable to
correlation range; only the negative is safe.

**7. A diffusion denoiser trained on filtered data implements the oracle matched
to its training distribution** (`exp_15 mismatch`) — P1's Fig. 2 finding,
transplanted. Train a DSM network on data filtered at `k_train`, test on
*unfiltered* data, and ask which `BP_k` its posterior mean is closest to. It
picks `BP_{k_train}`, not `BP_0`, in **16 of 16** cells once two degeneracies
are set aside, and the margin over the runner-up grows with `k_train`: +1.6–4.3%
at `k=2`, +5.9–29% at `k=3`, **+6.2–63%** at `k=4`.

The two set-aside cases are both explained rather than excluded by hand:

- **`k_train = 1` is exactly degenerate with `k_train = 0` in a Gaussian tree.**
  Measured `dist_k0` and `dist_k1` agree to five decimals, gap 0.00%. This is a
  real limitation of the Gaussian analogue, not noise: P1's transition *tensor*
  `M_{abc}` can correlate the two children given the parent, whereas a
  linear-Gaussian edge makes siblings conditionally independent given the
  parent by construction. So the `k = 0` versus `k = 1` distinction — the one
  place their filtering changes probabilities without changing topology —
  **cannot be represented here at all.** Reproducing it needs the discrete
  tensor model.
- **At `t = 1.6` the argmin stops measuring alignment.** Not because the oracles
  converge (their spread is still 0.22) but because the network's distance to
  *every* oracle is ~1.50, seven times that spread. The argmin is then reading
  network error, not implied correlation range. Every row carries
  `oracle_spread` so this is checkable rather than assumed.

### What did not come out

**Sequential acquisition: a weak version, not the staircase** (`exp_15
alignment`; supersedes the wrong probe in `exp_13 ordering`). Training a
denoiser on unfiltered data and tracking which `BP_k` it is closest to, the
argmin does move in the predicted direction at the two intermediate noise
levels — `2 → 2 → 0` at `t = 0.4` and `2 → 1 → 1` at `t = 0.8` over the first
~1000 gradient steps — and then **saturates for the remaining 31 000 steps**.
At `t = 0.1`, `0.2` and `1.6` it never leaves `k = 0`.

So: the direction is right, the effect is early and small, and nothing like the
clean staircase P1 reports. Three reasons, all plausible and none verified
here: their oracles are far more distinguishable (a discrete non-overlapping
transition tensor versus a Gaussian edge — see the `k=0`/`k=1` degeneracy
above); a denoising regression loss converges in far fewer steps than their
MLM, which needs ~10³ epochs; and the architecture and task are different. The
honest summary is that **this setting is too easy to resolve the phenomenon**,
not that the phenomenon is absent.

The earlier `exp_13 ordering` probe — resolve the error onto hierarchy levels
and watch it move — found nothing at all, and is retained only as a record of a
measurement that could not have answered the question.

**EM, by contrast, does show an ordering**, with a transparent mechanism: it
starts with ρ̂ far too small, which destroys long-range structure first, so the
coarsest level carries almost all the error early (level −1 error 1.29 against
0.02 at level 1 after one iteration) and the levels equalize as ρ̂ climbs. That
is a property of the estimator's parameterization, not an inductive bias worth
generalizing.

---

## 6. Untried, in priority order

1. **Read P2.** P1 is now read in full; the Nature Communications paper is not,
   and every route to it from this environment is blocked. Nothing here depends
   on its notation — the derivations are self-contained and independently
   verified — but a write-up that cites it must use its symbols, and the
   correspondence has not been checked against the source.
1b. **Read Sclocchi, Favero and Wyart (arXiv:2402.16991)** before writing
   anything about hierarchy and diffusion. It is the closest prior work, P1
   cites it, and an earlier version of this note claimed its territory was
   empty.
2. **The nonparametric collapse test.** The obvious objection to result 3 is
   that a two-parameter kernel *cannot* memorize. `exp_14 --set
   include_mdn=True` runs the same comparison with a mixture-density kernel of
   several hundred parameters and the same `n`-independent statistic, which
   separates the mechanism from the parameter count.
3. **A discrete-alphabet tree** — closest to the actual data model of the
   hierarchical-filtering paper. `src/discrete.py` does chains,
   `src/hierarchy.py` does continuous trees; the two have not been crossed.
4. **The cascade under a learned score**, rather than the exact one: which
   levels of the ladder each method gets right *dynamically*.
5. **A non-Gaussian tree.** `tree_bp_grid` and `fit_em_tree` already support it
   with no new machinery; the Layer-3 question (recover an unknown innovation
   law from noisy data alone) transfers directly.
