# Three claims, one script

Everything below comes from one command:

```bash
python3.12 simple/run_simple.py
```

About 73 minutes on eight cores, no cluster — roughly half spent fitting both arms to convergence
and half on claim 3's reverse integrations. It writes `results.csv`, which contains every number
quoted here, and the three figures in `figures/`. `--quick` runs the same code paths in under a
minute for smoke-testing; its numbers are meaningless and the script says so.

The script imports every numerical routine from `src/`, which is covered by the 220-test suite.
Only the orchestration is new. Nothing in the full package was deleted — this is a second, shorter
entry point to the same code.

## The idea, in one paragraph

Coordinatewise OU noising contributes one unary likelihood factor per site and adds no edges. So
the posterior of a noised Markov chain is *still a chain*; a chain is a tree; sum–product on a
tree is exact. That gives a **computable population score** — the object every diffusion model
approximates and normally cannot see. Having it lets us ask two questions that usually have no
ground truth and get real answers.

---

## Claim 1 — the kernel is learned from noised data alone

`fig1_recovery.pdf` — three panels: (a) fitted ρ against EM iteration, (b) fitted innovation
excess kurtosis against the same axis, (c) the learned innovation density against the true
Laplace. EM–BP is started at ρ = 0.0, 0.3 and 0.6 on a Laplace AR(1) chain with true ρ = 0.85, and
never sees a clean sample: every observation it is fitted to has already been noised. All three
initialisations converge to the same place.

| initialisation | fitted ρ | innovation variance | innovation excess kurtosis |
|---|---|---|---|
| 0.0 | 0.8554 | 0.2754 | 2.152 |
| 0.3 | 0.8553 | 0.2754 | 2.147 |
| 0.6 | 0.8553 | 0.2754 | 2.143 |
| **truth** | **0.85** | **0.2775** | **3.0** |

Three initialisations spanning 0.0 to 0.6 agree to four decimal places, the evidence is monotone
at every EM step (violation exactly 0.0), and the variance is recovered to 0.8%.

Read this carefully, because the columns do not say the same thing — and because the third one
nearly fooled us twice.

**ρ and the variance are recovered tightly.** Three initialisations spanning 0.0 to 0.6 land within
0.0001 of each other. Note what that does *not* establish: they share one dataset, so their
agreement bounds the optimisation variance and says nothing about the estimation error.

**The kurtosis column is a single draw at 120 iterations, and neither of those is enough.**
Panels (a) and (b) show why the iteration count is not: they are the same three runs on the same
axis, and (a) is flat from iteration 25 while (b) is still climbing. A ρ trace that has plateaued
is not evidence of a converged kernel.

Varying the *data* rather than the initialisation, at 200 iterations, ρ = 0.85, N = 512, C = 8,
eight independent draws:

| | mean | sd across draws | s.e. | range |
|---|---|---|---|---|
| fitted from clean chains | 2.972 | 0.209 | 0.074 | [2.592, 3.293] |
| fitted through the channel | 3.024 | 0.577 | 0.204 | [2.396, 3.993] |
| **truth** | **3.0** | | | |

So **at convergence the estimator recovers the innovation shape essentially exactly**, and a single
draw is worth very little: the channel-fitted value ranges from 2.40 to 3.99 across draws. The
2.15 in the table above is one low draw at an iteration count it had not converged at, and an
earlier version of this note built an argument on it — "shape is recovered to 72%", with the
missing 28% attributed to a contest between mixture capacity and channel information. Neither
factor was needed; the deficit was optimisation and sampling noise.

**The channel costs convergence rate, not accuracy.** Paired on the same chains, clean against
noised: the paired difference in recovered kurtosis is **−0.052 ± 0.217**, i.e. nothing. What does
differ is how fast each gets there — ρ converges in a single iteration on clean data against
roughly 30–60 through the channel, and the shape takes about 20–60 clean against 60–120 noised.
That is the missing-information effect of Dempster–Laird–Rubin showing up where it belongs, in the
rate, and it explains why every under-iterated run in this project produced a *deficit* rather
than noise.

One consequence worth stating plainly, because it has now caught us three times: at 40 iterations
it manufactured a pointwise/generative dissociation and a capacity effect (see the last section);
at 120 it manufactured this shape deficit; and a paired clean-vs-channel comparison at a fixed
120 iterations manufactured a channel penalty of +0.22 ± 0.15 that is −0.05 ± 0.22 once both arms
are run to convergence. The lesson is not "use more iterations" but that **a fixed iteration
budget compares convergence rates, not estimators.**

---

## Claim 2 — knowing the chain structure buys a large sample-efficiency advantage

`fig2_efficiency.pdf`. This is Marc's claim. EM–BP and a convolutional denoiser are fitted to the
**same chains**, and both are scored by relative L2 error against the exact BP score. Averaged
over five noise levels and three seeds:

| training chains N | EM–BP | convolution | ratio |
|---|---|---|---|
| 32 | 0.0553 | 0.1992 | **3.60×** |
| 128 | 0.0313 | 0.1117 | **3.57×** |
| 512 | 0.0202 | 0.1114 | **5.52×** |

Note the shape of it, not just the size. EM–BP's error falls by 2.7× across the range, while the
convolution stops improving after N = 128 and sits at 0.111. So the advantage **widens with data**
rather than being a fixed offset: knowing the structure means the extra chains go into estimating
13 numbers, whereas the network is still spending them on discovering the structure.

The estimator is better at every budget and the error bars over seeds do not overlap.

Two things make this a fair fight rather than a rigged one. First, the network is trained to its
own convergence, checked the same way EM was: at 4000 gradient steps its score error is 0.0950
(ε) and 0.0869 (x₀); at 8000, 0.0884 and 0.0817; at 16000, 0.0890 and 0.0762. The runs here use
8000, where ε has flattened. Second, the network redraws its noise at every gradient step, so it
effectively sees unlimited noisy views of the training chains, while EM sees one noisy realisation
per chain and never a clean one — an asymmetry that favours the network.

One caveat in the other direction, which claim 3 explains. The network arm is the parameterisation
chosen on a validation split, and validation selects x₀ at every budget. At N = 32 and 128 x₀ is
also the better arm on the score metric reported above, so the ratio there is conservative. At
N = 512 it is **not**: ε scores 0.0981 against x₀'s 0.1114, so a score-optimal network would give
a ratio of 4.86× rather than the 5.52× quoted. Read the last row as 4.9–5.5× depending on which
network you think the baseline should be allowed to pick.

The mechanism is not subtle. EM–BP fits **13 numbers** — one correlation plus an eight-component
innovation mixture — because it has been told the dependency structure. The network has to
discover that structure from data before it can exploit it.

---

## Claim 3 — pointwise accuracy does not predict generative fidelity

`fig3_generation.pdf`. This is Jérôme's objection, and it is the result worth arguing about.

The same fitted estimators are now run **through reverse diffusion** and judged on what they
generate. The statistic is the AR-filtered residual excess kurtosis. The reference is real data
noised to `t_min`, not clean data, because the reverse SDE stops at `t_min` — comparing against
clean data would measure the stopping floor rather than the score.

Generated AR-residual excess kurtosis on the **Laplace** chain, mean ± standard error over **six**
seeds (twice the replicates of claims 1–2: this is a fourth moment and much noisier than a score
error, so the extra runs go where the noise is):

| training chains N | EM–BP | convolution (ε) | convolution (x₀) | target |
|---|---|---|---|---|
| 32 | 1.538 ± 0.177 | 3.089 ± 1.145 | 0.916 ± 0.162 | 1.889 |
| 128 | 1.359 ± 0.239 | 2.072 ± 0.426 | 1.186 ± 0.078 | 1.889 |
| 512 | **1.649 ± 0.118** | 1.794 ± 0.189 | 1.249 ± 0.080 | 1.889 |

Take only what clears the intervals. Distance from target, in standard errors:

| N | EM–BP | convolution (ε) | convolution (x₀) | ε − x₀ |
|---|---|---|---|---|
| 32 | 2.0σ | 1.0σ | **6.0σ** | 1.9σ |
| 128 | 2.2σ | 0.4σ | **9.1σ** | 2.1σ |
| 512 | 2.0σ | 0.5σ | **8.0σ** | 2.6σ |

- **x₀ falls short at every budget, by 6–9σ.** This is the firmest result in the table.
- **ε is consistent with the target at every budget** (0.4–1.0σ).
- **EM–BP sits about 2σ low, and stays there** — it does not drift with budget, so this looks like
  a small genuine bias rather than noise, and it is the residual shape deficit of claim 1 showing
  up downstream.
- **EM–BP and ε are not distinguishable from each other** (0.145 apart at N = 512, 0.65σ). No claim
  that either beats the other survives, and an earlier draft of this note wrongly made one.

There is a second effect in the table that the means alone hide: **ε is wildly unstable across
seeds.** Its standard error is 1.145 at N = 32 against x₀'s 0.162 — seven times larger — and still
2.4× larger at N = 512. So the two parameterisations fail differently: x₀ is tightly clustered and
biased low, ε is roughly unbiased and unreliable. EM–BP is the only arm that is both near the
target and low-variance. That is a real point in its favour, and a weaker one than "it wins".

Now put that next to how the same two networks look pointwise. Two reasonable pointwise metrics
**disagree with each other about which network is better**, at N = 512:

| | posterior-mean MSE | relative score L2 | generated kurtosis |
|---|---|---|---|
| ε | 0.02442 | **0.0981** | 1.794 |
| x₀ | **0.01001** | 0.1114 | 1.249 |

The validation rule sums posterior-mean MSE, where x₀ wins by a factor of 2.4, and it duly selects
x₀ at every budget and every seed. On score error — the thing the reverse SDE actually integrates —
ε is 12% better. And generatively x₀ misses by 8.0σ while ε is within noise of the target.

The two metrics differ because they weight noise levels differently. Tempting as it is to conclude
that generation simply follows the score metric, the table does not support that either: at
N = 32 both metrics prefer x₀ and x₀ *is* the closer arm generatively (0.973 from target against
ε's 1.200), while at N = 128 they disagree and ε is closer. There is no rule here — which is the
point. A selection procedure that nobody would think twice about picks the generatively worse arm
at the largest budget, and no pointwise quantity available at selection time would have told you.

So the honest statement of Jérôme's objection is not "the structured estimator generates worse" —
that was the earlier, unconverged answer. It is that **the pointwise number does not order the
generative one.** A choice nobody thinks of as a modelling decision, made on a validation split by
a rule that is itself ambiguous, moves the generated law more than the entire estimator-versus-
network gap does.

### Why there are two network arms and not one

The convolutional denoiser can be trained to regress either the noise (ε) or the clean signal
(x₀). This is normally treated as an implementation detail chosen by validation. It is not one.
The two parameterisations are trained on identical data and differ only in the regression target,
and:

- Which one validation prefers **depends on how you aggregate across noise levels**. Summing the
  per-level MSE picks x₀ (0.054 against 0.193); a majority vote over the five levels picks ε
  (it wins at t = 0.1, 0.2, 0.4 and loses at 0.8, 1.6). Both rules are defensible and they
  disagree.
- It also depends on **which pointwise metric you use at all**: posterior-mean MSE prefers x₀ by
  2.4×, relative score L2 prefers ε by 12%, on the same two networks at N = 512.
- What they generate is **not close**: 1.794 against 1.249 at N = 512, a gap of 0.545 ± 0.205.

So reporting the validation-selected network would have let an arbitrary choice decide the
headline — and note which way it would have gone: the summed-MSE rule selects x₀, the arm that
misses the target by 8.0σ. Claim 2 must pick one network, because it compares pointwise error and
there has to be a single number; claim 3 reports both, because there the choice *is* the finding.

### What the control establishes

On a Gaussian chain the fitted mixture family **contains** the truth. If the spread on the Laplace
panel were caused by the channel destroying information, or by the integrator, or by the residual
statistic, it would appear here too.

| N | EM–BP | convolution (ε) | convolution (x₀) | spread | target |
|---|---|---|---|---|---|
| 32 | 0.381 ± 0.174 | 0.094 ± 0.027 | 0.014 ± 0.024 | 0.366 | 0.038 |
| 128 | 0.130 ± 0.069 | 0.123 ± 0.021 | 0.049 ± 0.024 | 0.081 | 0.038 |
| 512 | **0.042 ± 0.026** | 0.124 ± 0.016 | 0.060 ± 0.008 | **0.082** | 0.038 |

Two things to read here, and the first is the one that licenses the conclusion.

**The spread collapses.** Across the three arms at N = 512 it is 0.545 on the Laplace chain and
0.082 here — a factor of 6.7. And **EM–BP specifically goes from 2.0σ low on Laplace to 0.2σ on
Gaussian**: 0.042 against a target of 0.038, essentially exact, converging onto it as the budget
grows (0.381 → 0.130 → 0.042). The estimator misses precisely when its model class does not
contain the truth and hits when it does. That is what a misspecification account predicts and an
information-loss account does not.

**But "all three sit on the target" would be too strong.** The network arms keep small biases that
do not shrink with data — ε sits at 0.124 regardless of budget, which is 5.2σ at N = 512 only
because the standard error there is 0.016. In absolute terms it is a bias of 0.086 against a
Laplace-panel effect of 0.545, so it does not threaten the reading; it is a property of the
networks rather than of the chain, and it is worth knowing about.

So the Laplace spread is **model misspecification** — a property of the fitted classes, in
principle fixable — and not a property of the observation model, which would not be. How to fix it
is *not* settled here: the capacity sweep in the full package would be the obvious appeal, but that
sweep ran at 40 EM iterations and is entangled with the convergence effect below.

---

## What this does and does not show

**Does.** That the exact posterior gives a computable reference; that an unknown kernel is
recoverable from noised data alone; that structure is worth 3.6–5.5× in sample efficiency at these
budgets, widening with data; and that two pointwise metrics can rank the same pair of networks in
opposite orders while the generative gap between them is 72%, demonstrated with a control that
rules out the boring explanations.

**Does not.** These are 32-site AR(1) chains, not images. Claim 2's ratio is measured against one
convolutional baseline at one receptive field, not against a tuned modern denoiser. Claim 3 rests
on six seeds, which resolves x₀'s shortfall (8.0σ) and the ε-versus-x₀ gap (2.6σ) but **not** any
ordering between EM–BP and ε (0.65σ) — this evidence does not say the structured estimator
generates better, only that it is the one arm that is simultaneously near the target and stable.
And it establishes that pointwise error fails to *order* the generative result; it does not say how
to choose between the arms, which is the open question.

## Relation to the full apparatus

The simple version must reproduce the full runs or it is a different experiment. Three of the four
checks pass. **The fourth does not, and the discrepancy is the point.**

| quantity | full apparatus | here | |
|---|---|---|---|
| recovered ρ | 0.850 | 0.8553–0.8554 | ✅ |
| EM–BP vs convolution, score error | ≈ 3.9× | 3.6–5.5× (4.9× at N=512 against the score-optimal arm) | ✅ same order |
| Gaussian control | no gap | no gap | ✅ |
| Laplace generated kurtosis, N = 512 | EM–BP 0.841, CNN 1.410 | EM–BP 1.649, CNN 1.249–1.794 | ❌ **does not reproduce** |

The committed run has EM–BP generating well *below* the convolution and far below the target
(0.841 against 1.9); here it lands within 2σ of the target and level with the better network arm.
That is a genuine contradiction, not a tolerance issue, and the cause is identified: **`exp_16`
runs `em_iters=40`**, at which the fitted kernel carries roughly a third of the true innovation
excess kurtosis (see claim 1), and the generated statistic tracks the kernel it came from.

### The decisive check

Rather than argue from a differently-configured script, `exp_16`'s own configuration was rerun
with the iteration count as the only variable — C ∈ {4, 8}, composite protocol, Laplace, N = 512,
grid 401 — at 40 against 200 iterations:

| C | EM iterations | fitted innovation kurtosis | generated kurtosis |
|---|---|---|---|
| 4 | 40 | 1.079 | 0.625 |
| 4 | **200** | **2.538** | **1.464** |
| 8 | 40 | 1.621 | 0.871 |
| 8 | **200** | **2.560** | **1.380** |

with the convolution at 1.367 and the target at 1.888. Two things fall out, and both contradict
what is currently written in ch11.

**The capacity effect is a convergence effect.** Going from C = 4 to C = 8 buys **0.542** in fitted
kurtosis at 40 iterations and **0.022** at 200. Roughly 96% of the apparent benefit of a larger
mixture is not capacity at all — it is that a larger mixture gets further within a fixed iteration
budget. At convergence, C = 4 fits the innovation law as well as C = 8 (2.538 against 2.560).

**The dissociation goes with it.** At 40 iterations EM–BP generates 0.625 (C = 4) and 0.871
(C = 8), both below the convolution's 1.367 — the reported dissociation. At 200 iterations it
generates 1.464 and 1.380, both at or above it. The ordering that ch11 is built on reverses when
the estimator is fitted to convergence.

So:

1. **ch11 §"the dissociation is a capacity effect" is not supported.** The capacity sweep varied
   C at a fixed 40 iterations, which confounds the two, and unconfounding them removes the effect.
2. **The C = 8 choice inherits the problem**, since it came from that sweep. C = 4 appears to be
   sufficient once EM is run to convergence.
3. **The dissociation reported in ch11 §"pointwise accuracy and generative fidelity can disagree"
   is itself in question**, not merely its explanation.

Caveats on this table, so it is not over-read: one seed, 800 generated chains, and 200 integration
steps against `exp_16`'s 400. It is strong enough to say the committed conclusions cannot be
defended as they stand, and not a substitute for rerunning the sweep properly on the cluster.
Claims 1 and 2 are unaffected.

Remaining differences are ordinary and do not change orderings: this runs at a 301-point grid and
200 integration steps against 401 points and 400 steps, and tops out at N = 512 rather than
N = 2048. Nothing here should be quoted as a headline figure in preference to full-scale numbers —
except the convergence finding, which supersedes them.
