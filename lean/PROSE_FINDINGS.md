# Prose & numerical-claim audit — actionable findings

Companion to `AUDIT_REPORT.md` (which covers the 460 display-math formulas).

**Why this pass exists:** the Lean audit found the display equations sound, but both errors it did
stumble on were in *prose* — one inline formula, one quoted number. This pass therefore checked the
~446 decimal-bearing inline claims directly. It found **37 items**.

Every finding below states the arithmetic, so each can be checked in well under a minute.

| severity | count |
|---|---|
| error | 12 |
| inconsistency | 16 |
| rounding | 8 |
| unverifiable | 1 |


---

## ERRORS — the statement is wrong as written

### App. A (Reproducibility) — `30-31`

**As written:** two are Ti\emph{k}Z schematics drawn in the source

**Problem:** There are THREE TikZ figures, not two. Counting over thesis/chapters/: 31 figure environments and 31 \label{fig:...}. Of these, 28 use \includegraphics (21 .pdf = exactly the 21 rows of Table~\ref{tab:figure-map}; 7 .png = exactly the seven literature figures named in this same sentence) and 3 use \begin{tikzpicture}: fig:thesis-pipeline (ch01-introduction.tex, 1 tikzpicture), fig:rbm-graph and fig:posterior-factor-graph (ch03-model.tex, 2 tikzpictures). Arithmetic: 21 + 7 + 2 = 30, but the thesis has 31 figures; 21 + 7 + 3 = 31. Per-file tikzpicture counts are ch01=1, ch03=2, all other chapters 0.

**Fix:** Change "two are Ti\emph{k}Z schematics drawn in the source" to "three are Ti\emph{k}Z schematics drawn in the source".

### Ch. 6 (Gaussian BP) — `571-575`

**As written:** One qualitative detail from a worked $L = 3$ instance: at $x = (1.2, -0.4, 0.8)$ the score component $S_1$ is positive although $x_1 < 0$, because the neighbours pull the posterior mean of the middle frame positive. No per-frame marginal score can produce this; it is inter-frame evidence sharing in numbers.

**Problem:** The last sentence is false, and it contradicts this thesis's own ch05. The per-frame marginal of x_k is N(0, e^{-2t}*1 + Delta_t) = N(0,1) at every t, so the per-frame marginal score is exactly -x_k -- ch05-gaussian-matrix.tex line 803 says so verbatim ('the per-frame marginal score stays $-x_k$ at every $t$'). At x_1 = -0.4 that marginal score is -(-0.4) = +0.4 > 0, i.e. POSITIVE. So a positive S_1 when x_1 < 0 is precisely what a per-frame marginal score does produce (necessarily, since it always has the opposite sign to x_k); it carries no inter-frame information at all and cannot be the signature claimed. What is actually distinctive is the MAGNITUDE: e.g. L=3, alpha=0.8, t=0.05 gives exact S_1 = +3.915 against the marginal +0.400 (alpha=0.9, t=0.05: S_1 = +5.711). Secondary: the stated mechanism also fails for much of the chapter's own audit grid -- at alpha=0.8, t=0.05 the posterior mean is m_1 = -0.029 < 0 (not 'pulled positive') while S_1 is still +3.92, and for alpha=0.2 m_1 < 0 at every t in {0.05,0.3,1,3}; the instance's alpha and t are not stated, so only alpha near 0.9 supports the 'because' clause.

**Fix:** Replace the sign argument with the magnitude argument, and state alpha and t. E.g. 'at $x=(1.2,-0.4,0.8)$ with $\alpha=0.9$, $t=0.05$ the middle score is $S_1=+5.71$, fourteen times the $-x_1=+0.40$ a per-frame marginal score would give: the neighbours, not the frame itself, set the size of the correction.'

### Ch. 7 (Laplace) — `367-369`

**As written:** the self-convergence error between the two remains $10^4$ to $10^7$ times smaller than every effect measured below (Figure~\ref{fig:bp-closure}, left)

**Problem:** Both ends are wrong. The two curves plotted in fig:bp-closure (left) are score_med and grid_selfconv_med of tools/data/ch07_metrics.csv, both relative score errors. Ratio score_med/grid_selfconv_med over the 14 sweep points: 1.36e4 (t=0.02), 1.77e4, 2.38e4, 3.15e4, 3.78e4, 5.55e4, 8.02e4, 8.07e4 (t=0.3215, the maximum), 7.78e4, 5.21e4, 3.14e4, 1.40e4, 5.27e3 (t=2.07), 1.02e3 (t=3). So the range is ~1e3 to ~8e4, i.e. three to just under five orders of magnitude; it never reaches 1e5, let alone 1e7, and it drops BELOW 1e4 at the two largest t. The chapter's own caption for the same panel (lines 710-712) states the correct range: 'at least three orders of magnitude everywhere and by four to five over most of the range'.

**Fix:** Replace '$10^4$ to $10^7$ times smaller' with '$10^3$ to $10^5$ times smaller' (or, to match the caption, 'at least three orders of magnitude smaller everywhere and four to five over most of the range').

### Ch. 7 (Laplace) — `713-715`

**As written:** Directional error is one to two orders of magnitude smaller than magnitude error at every diffusion time. (caption of Figure~\ref{fig:bp-closure}, right)

**Problem:** The right panel plots score_med ('relative magnitude') and 1 - cos_med ('$1-\cos$ angle') against t. Ratio score_med/(1-cos_med): t=0.02 -> 0.3770/0.07149 = 5.3; t=0.032 -> 0.2996/0.04350 = 6.9; t=0.05 -> 0.2686/0.03646 = 7.4 (all UNDER one order, and t=0.05 is the anchor the body text quotes, cos = 0.964); then 10.4, 17.2, 23.6, 42, 57, 87, 160, 336, 1.1e3 (t=1.425), 9.2e3 (t=2.07), 2.9e5 (t=3). The gap widens monotonically from a factor of ~5 to more than five orders; 'one to two orders' holds only for t in roughly [0.07, 0.5], not 'at every diffusion time'.

**Fix:** State the real behaviour, e.g. 'Directional error is smaller than magnitude error at every diffusion time, by a factor of about five at $t=0.02$ and by three to five orders of magnitude once $t\gtrsim1$.'

### Ch. 8 (EM parameters) — `57-62`

**As written:** with $(\pi_c, \nu_c, s_c)$ free the fitted chain's stationary variance is $\sum_c \pi_c(\nu_c^2 + s_c^2)/(1-\alpha^2)$, which equals one only under a variance-matching constraint this family deliberately does not impose, and its stationary mean is zero only if $\sum_c \pi_c \nu_c = 0$, which it likewise does not impose

**Problem:** The inline formula silently assumes the innovation mean is zero, which the same sentence then says is NOT imposed. For a_i = alpha*a_{i-1} + eps_i with eps ~ sum_c pi_c N(nu_c, s_c^2): the innovation SECOND MOMENT is sum_c pi_c(nu_c^2+s_c^2), but its VARIANCE is sum_c pi_c(nu_c^2+s_c^2) - m^2 with m = sum_c pi_c nu_c. Hence the stationary variance is [sum_c pi_c(nu_c^2+s_c^2) - m^2]/(1-alpha^2). The printed expression is neither the stationary variance nor the stationary second moment when m != 0 (the stationary second moment is that variance plus mu^2 = m^2/(1-alpha)^2). Counterexample (Lean, stationary_variance_counterexample): pi=(1/2,1/2), nu=(1,1), s^2=(1,1), alpha=0 gives printed value 2 but true stationary variance 1, a factor-2 error, with m = 1 != 0.

**Fix:** Write the stationary variance as $\big[\sum_c \pi_c(\nu_c^2 + s_c^2) - (\sum_c \pi_c\nu_c)^2\big]/(1-\alpha^2)$, or say explicitly that $\sum_c \pi_c(\nu_c^2+s_c^2)$ is the innovation second moment and the expression is the stationary variance only when the mean condition $\sum_c\pi_c\nu_c=0$ holds.

*Lean witness:* `ThesisAudit.ProseCh08.stationary_variance_counterexample`

### Ch. 9 (EM kernel) — `182-198 (Table 9.1, tab:em-rates: caption, tau column, asymptotic k column, ratio row)`

**As written:** rel. tolerance tau = 1.2e-3 (alpha) and 6.7e-3 (kappa_4); asymptotic k(tau,lambda) = 211 and 1256; ratio row 'asymptotic 5.9'. Caption: 'The asymptotic column applies (9.4) with the relative tolerance actually used.'

**Problem:** The tolerances actually used in the sweep are RELATIVE 1e-3 (alpha) and RELATIVE 2e-2 (excess kurtosis), not absolute. Verified two ways: (i) research/nongaussian-bp/tools/make_convergence_numbers.py docstring, 'stays within a relative tolerance of its final value (rho 1e-3, innovation variance 1e-2, excess kurtosis 2e-2)'; (ii) recomputing settle_* from exp_27_seed16/shape_trace.csv reproduces all 16 CSV values exactly under the relative rule and none of them under the absolute rule. The printed taus are the absolute values divided by the endpoints (1e-3/0.8504 = 1.18e-3 -> '1.2e-3'; 2e-2/2.98 = 6.71e-3 -> '6.7e-3'), i.e. a conversion that should not have been made. With the tolerances actually used: k(1e-3, 0.9686) = ln(1e-3)/ln(0.9686) = -6.9078/-0.031903 = 216 (not 211), and k(2e-2, 0.9960) = ln(0.02)/ln(0.9960) = -3.9120/-0.0040080 = 976 (not 1256). The asymptotic ratio becomes (ln 0.02/ln 0.001) x (ln 0.9686/ln 0.9960) = 0.5663 x 7.960 = 4.5, not 5.9. (Unrounded lambdas from the trace, 0.968561 and 0.996018, give 216, 981, 4.5.) The kappa_4 asymptotic count is therefore 29% too large.

**Fix:** Set tau = 1.0e-3 and 2.0e-2 in the table, asymptotic k = 216 and 976, ratio 4.5; and propagate 5.9 -> 4.5 at line 219.

### Ch. 9 (EM kernel) — `262-265`

**As written:** "Settled" is trace stability --- the first update after which a coordinate stays within $10^{-3}$ of its end-of-run value for $\alpha$, or $2\times10^{-2}$ for the excess kurtosis

**Problem:** Stated as an absolute distance; the criterion that produced every settling number in this chapter is relative to the end-of-run value. The absolute reading does not reproduce the reported medians: recomputing from exp_27_seed16/shape_trace.csv with an absolute 2e-2 band on the excess kurtosis gives a median settling time of 416 updates (per-seed values 92...686), whereas the relative 2e-2 band reproduces the CSV exactly and gives the reported median of 229. For alpha the two conventions nearly coincide (|alpha| ~ 0.85), which is why only the shape coordinate shows the discrepancy. This mis-statement is the root of the tau column error above.

**Fix:** "...stays within a relative $10^{-3}$ of its end-of-run value for $\alpha$, or a relative $2\times10^{-2}$ for the excess kurtosis".

### Ch. 9 (EM kernel) — `219-221`

**As written:** the asymptotic counts over-predict both individual settling times by a factor of two to three

**Problem:** Only alpha is in that range. As printed: 211/80 = 2.64 for alpha but 1256/229 = 5.48 for the shape. With the corrected taus of the first finding: 216/80 = 2.7 and 976/229 = 4.3. Either way the two over-prediction factors are not both in [2,3], and the sentence is self-contradictory with its own first half: the same sentence says the predicted ratio is 5.9 while the measured ratio is 2.8, and 5.9/2.8 = 2.1 forces the two individual over-prediction factors to differ by a factor of ~2 (they do: 5.48/2.64 = 2.08). A common 2-3x over-prediction would have left the ratio intact.

**Fix:** "...over-predict the correlation's settling time by a factor of about three and the shape's by a factor of about five" (or, with the corrected taus, "about three and about four").

### Ch. 9 (EM kernel) — `147-148`

**As written:** (9.6) says the channel removes an $e^{-4t}$ fraction of exactly the structure that identifies it

**Problem:** Backwards relative to the display equation it cites. Equation (9.6) is kappa_4(X_t) = e^{-4t} kappa_4(A), so e^{-4t} is the fraction that SURVIVES; the fraction removed is 1 - e^{-4t}. As written the sentence says the damage shrinks as t grows (at t = 0.2, e^{-0.8} = 0.449, so it claims 45% is removed when in fact 55% is removed; as t -> infinity it claims almost nothing is removed, the opposite of the argument being made). The surrounding argument -- that the channel concentrates its information loss on the fourth-order coordinate -- needs the complementary fraction.

**Fix:** "...the channel leaves only an $e^{-4t}$ fraction of exactly the structure that identifies it" (equivalently, removes a $1-e^{-4t}$ fraction).

### Ch. 10 (Comparison) — `1085-1087`

**As written:** the CNN ratio falling from $1.2$ to $0.97$ across its bracket and the MLP one from $1.2$ to $0.84$ across its own. A straight line through those endpoints crosses at $2.1$ and $3.0 \times 10^{-2}$ respectively

**Problem:** The MLP crossing is 3.2, not 3.0. Under the convention the sentence itself states (a straight line through the two bracket endpoints, linear in delta) the MLP bracket is (delta=2.2e-2, R=1.2) -> (delta=4.0e-2, R=0.84); crossing R=1 at delta = 2.2 + (1.2-1.0)/(1.2-0.84)*(4.0-2.2) = 2.2 + 0.5556*1.8 = 3.2 (x1e-2). The CNN number is right under that same convention: 1.1 + (1.2-1.0)/(1.2-0.97)*(2.2-1.1) = 1.1 + 0.8696*1.1 = 2.06 -> 2.1. No single convention produces the printed pair: a straight line in log(delta) gives 2.01 and 3.07, i.e. (2.0, 3.1). So the second printed value is wrong whichever of the two readings is intended.

**Fix:** Replace '3.0' with '3.2' (keeping '2.1'), or state the log-delta convention and print '2.0 and 3.1'.

### Ch. 10 (Comparison) — `893-899 (eq:em-longrange) vs table at 1009-1021`

**As written:** $[Q_{\mathrm{long}}]_{ij} = -e^{-|i-j|/\ell}\ (|i-j| \geq 2)$, $[Q_{\mathrm{long}}]_{ii} = 1.05 \sum_{j\neq i} e^{-|i-j|/\ell}$

**Problem:** The diagonal compensation as printed sums over all j != i, but the off-diagonal it is compensating is zero at |i-j| = 1, and the code that produces the delta column of Table 10.9 (thesis/tools/fig_ch10.py, longrange_cov: off[dist<2]=0 then Ql[i,i] = -off.sum(axis=1)*1.05) sums only over |i-j| >= 2. The two are not the same matrix: e^{-1/8}=0.8825, so the printed diagonal is larger by 1.05*2*0.8825 ~ 1.85 at interior sites. Taken literally the printed equation gives delta = 7.2e-3, 1.24e-2, 1.81e-2, 2.34e-2 at gamma = 0.05, 0.10, 0.20, 0.40, against the table's 1.1e-2, 2.2e-2, 4.0e-2, 6.0e-2 (which reproduce the code exactly: 1.079e-2, 2.244e-2, 3.963e-2, 6.037e-2). It also destroys the sentence the section rests on: under the literal reading delta(gamma=0.05)/delta(beta=1) = 7.2e-3/6.53e-3 = 1.10, not the stated 'factor of 1.7' (true value 1.65), so beta=1 and gamma=0.05 would no longer be cleanly separated on the delta axis, and the break-even brackets of eq:em-breakeven would move.

**Fix:** Change the diagonal subscript to match the implementation: $[Q_{\mathrm{long}}]_{ii} = 1.05 \sum_{|i-j| \geq 2} e^{-|i-j|/\ell}$.

### Cross-chapter — `ch11:18 and ch11:57 vs ch06:16-20 and ch01:248-251`

**As written:** RQ1: '... $q(\alpha,t)$ sets the posterior correlation length ... (Chapter~\ref{ch:gaussian})'; RQ4: 'the influence of evidence at distance $d$ decays exactly as $q(\alpha,t)^d$ ... (Chapter~\ref{ch:gaussian-bp})'

**Problem:** Both pointers send an examiner to a chapter that does not contain the result, and the thesis says so itself twice. ch06:16-20: 'The closed form of the bulk posterior itself, and the geometric influence-decay rate $q(\alpha,t)$ it implies, are derived in Chapter~\ref{ch:em-results}, Section~\ref{sec:em-locality-prediction}'. ch01:248-251 repeats it: 'the closed-form influence-decay rate is derived where it is used, in Chapter~\ref{ch:em-results}'. grep confirms q(\corr,t)/q(\alpha,t) appears in ch10 (eq:em-bulk-params, eq:em-q-limits, eq:em-oracle, Table at ch10:672) and nowhere in ch05; ch05's only correlation length is the PRIOR's, |gamma(d)| = e^{-d/tau} at ch05:271-272, a different object. The ch11:57 numbers that follow (12.3% vs 21.0%) DO come from ch06 (tab:g-truncation, r=1 -> 0.123, b=1 -> 0.210, at L=40, alpha=0.8), so the RQ4 paragraph merges two chapters under one pointer.

**Fix:** ch11:18 -> '(Chapters~\ref{ch:gaussian} and~\ref{ch:em-results})'; ch11:57 -> '(Chapters~\ref{ch:gaussian-bp} and~\ref{ch:em-results})', or move the $q^d$ sentence's citation to \ref{ch:em-results} and leave \ref{ch:gaussian-bp} on the 12.3/21.0 clause. Consider adding 'at $\alpha = 0.8$' to the 12.3/21.0 parenthesis, since every other number in the thesis is quoted at $\corr = 0.85$.


---

## INCONSISTENCIES — two places in the thesis disagree, or a claim overstates

### App. A (Reproducibility) — `146-151`

**As written:** written there by one of the thirteen generators below ... Nine live under \texttt{research/tools/}; three are thesis-side scripts ... \texttt{make\_ref\_energy} is the fourteenth name and writes no macro at all

**Problem:** Off-by-one: make_ref_energy is counted twice. The table that follows lists exactly 13 names (col 1: make_tab_efficiency, make_tab_structured, make_tab_screening, make_tab_capacity, make_tab_nonmarkov, fig_ch10, make_build_manifest = 7; col 2: make_aggregation_robustness, make_aggregation_structured, make_convergence_numbers, make_grid_diagnostics, make_ref_energy, make_laplace_oracle = 6). The sentence's own breakdown gives 9 + 3 = 12 macro-writing generators, not thirteen. And since make_ref_energy IS one of the 13 listed names, it is the thirteenth name, not the fourteenth (13 macro-writers + make_ref_energy would require 14 rows in the table; there are 13). Confirmed against the generated files: thesis/sections/ has exactly 12 files containing \newcommand, written by 11 distinct scripts (fig_ch10 writes two of them), and thesis/tools/ contains exactly the three thesis-side names fig_ch10.py, make_laplace_oracle.py, make_build_manifest.py.

**Fix:** Read "one of the twelve generators below" and "\texttt{make\_ref\_energy} is the thirteenth name". (9 research-side + 3 thesis-side = 12 macro writers; make_ref_energy is the 13th listed name and writes no macro.)

### Ch. 6 (Gaussian BP) — `737, 755`

**As written:** removed by the first hop: $85\%$, $59\%$, $33\%$, $21\%$ (table row, repeated in prose: 'That fraction falls with diffusion time ($85\%, 59\%, 33\%, 21\%$)')

**Problem:** The t = 0.3 entry should be 58%, not 59%. Exact closed-form values (tools/fig_ch06.py, L=40, alpha=0.8): r0 = 0.5665192369, r1 = 0.2351592105, so 1 - r1/r0 = 0.5849052 = 58.4905% -> 58%. The other three entries are computed from exact values and are right (84.925% -> 85%, 32.943% -> 33%, 21.336% -> 21%), so the row's convention is 'exact values, rounded to whole percent' -- and under that convention 58.49% cannot print as 59%. The competing convention (derive from the rounded cells shown above) is the one that would give 59% ((0.567-0.235)/0.567 = 58.55%), but it is ruled out by the t=3 column, where it gives (0.0045-0.0035)/0.0045 = 22.2% -> 22%, not the 21% printed. So no single convention yields all four printed numbers; 59% is the odd one out. (Most likely cause: fig_ch06.py prints this quantity as '0.585' to three decimals and 58.5% was then rounded up.)

**Fix:** Change 59% to 58% in the table row (line 737) and in the prose list (line 755): '($85\%, 58\%, 33\%, 21\%$)'.

### Ch. 7 (Laplace) — `691-693`

**As written:** It is not monotone: it peaks at $t\simeq0.07$ and is no smaller at $t=0.02$ than at $t=0.15$. (caption of Figure~\ref{fig:l-metrics}, left)

**Problem:** The comparison is reversed. eps_PM = 0.03552 at t=0.02 and 0.04253 at t=0.1527 (ch07_metrics.csv; Table~\ref{tab:l-closure} prints 0.036 and 0.043, ratio column 1.00 vs 1.20). So eps_PM at t=0.02 IS smaller than at t=0.15, by ~20%. The body text at line 639-640 says it correctly: 'is still larger at $t=0.15$ than at $t=0.02$'.

**Fix:** Swap the two times: 'is no smaller at $t=0.15$ than at $t=0.02$'.

### Ch. 7 (Laplace) — `341-345 (and 365-366)`

**As written:** the working configuration clears it with room to spare ($3h = 0.12$ against $\sqrt{2t} = 0.32$ at the smallest reported $t$) ... It is \eqref{eq:l-resolution}, not the innovation law, that sets the smallest diffusion time this chapter reports.

**Problem:** $\sqrt{2t}=0.32$ is t = 0.05, but the smallest $t$ this chapter reports is 0.02 (first row of Table~\ref{tab:l-closure}; ch07_metrics.csv starts at t=0.02, and t=0.032 is also reported). At t=0.02, $\sqrt{2t} = \sqrt{0.04} = 0.200$, not 0.32. The gate is still cleared there (0.200 > 3h = 6A/(N_g-1) = 48/400 = 0.12), so no result is invalidated -- but the quoted number belongs to t=0.05. The same slip appears at line 366, 'gate satisfied for all $t \geq 0.05$', while the sweep reported in Table 7.1 runs down to t=0.02; and the gate itself permits $t \gtrsim 0.12^2/2 = 0.0072$ at $N_g=401$, so it is not what sets the chapter's smallest reported time.

**Fix:** Either quote the true smallest reported time ('$3h = 0.12$ against $\sqrt{2t} = 0.20$ at the smallest reported $t$, $t=0.02$') and change 'for all $t \geq 0.05$' to 'for all $t \geq 0.02$', or drop the claim that the gate sets the smallest reported diffusion time.

### Ch. 10 (Comparison) — `835-839 (and figure caption, 826-828)`

**As written:** no capacity buys anything: every paired mean is negative, the $\nseq = 128$ deficit at the largest capacity is resolved against zero ($-4.4 \times 10^{-4}$, CI $[-9.2\times10^{-4},\,-1.1\times10^{-4}]$)

**Problem:** Those numbers are the predeclared C=16 vs C=8 equivalence test (sections/capacity-numbers.tex says so in its own comment, and Table 10.12's verdict row states it correctly as 'C=16 resolved worse than C=8'), but the sentence sits inside a list of statements about the pairing against C=1, so a reader checks it against Table 10.8 and finds the C=16 / nseq=128 entry to be -2.35e-2 +/- 1.18e-2 -- a factor of 53 larger, and only marginally resolved against zero (|mean|/s.e. = 1.99). The figure 10.5(a) caption has the same problem ('at nseq = 128 the C = 16 deficit is resolved against zero').

**Fix:** Say what the contrast is: '... the nseq = 128 deficit of C = 16 against C = 8 is resolved (...)', in both the prose and the figure caption.

### Ch. 10 (Comparison) — `948-952`

**As written:** the correction in \eqref{eq:em-longrange} is $\gamma Q_{\mathrm{long}}$, dense by construction, with numerical rank $\nsites = 32$ and $30$ to $31$ eigenvalues above $1\%$ of its largest at the strengths tested

**Problem:** The spectrum of gamma*Q_long is scale-invariant in gamma, so no count from it can vary 'at the strengths tested'; and the count at a 1% threshold is 32, not 30-31 (|eig| sorted: 13.248 ... 4.416, 0.469; the smallest is 3.54% of the largest, so all 32 clear 1%). 31 clear 5% and 30 clear 33%. The printed 30-31 does reproduce a different object -- the realised departure of the rescaled contaminated precision from the chain precision, Q_gamma - Q_AR, which gives 30, 30, 30, 31 eigenvalues above 1% of its largest at gamma = 0.05, 0.10, 0.20, 0.40 and is the true analogue of the rank-one statement in Prop. 10.2 (which compares Q_beta with (1+beta^2)Q_AR, not with beta^2 * 11^T).

**Fix:** Attribute the count to the object that produces it: '... the departure $Q_\gamma - Q_{\mathrm{AR}}$ is dense, of numerical rank 32, with 30 to 31 eigenvalues above 1% of its largest at the strengths tested'.

### Ch. 10 (Comparison) — `808-811`

**As written:** the generic-MLP ratio overstates what can be attributed specifically to the supplied Markov model, and by roughly a factor of three

**Problem:** Nothing a reader can compute gives three. Size-matched, headline / structured is 6.9/2.34 = 2.95 (n=32), 7.9/2.82 = 2.80 (n=128), 8.2/3.58 = 2.29 (n=512), 8.6/6.14 = 1.40 (n=2048); the ratio of the two quoted ranges is 6.6/2.3 = 2.87 at the low end and 10.9/6.1 = 1.79 at the high end. The overstatement factor shrinks by more than two-fold across the tested budgets because the structured-baseline ratio grows faster (2.34 -> 6.14, a factor of 2.6) than the headline (6.9 -> 8.6, a factor of 1.25). 'Roughly a factor of three' is true only at the smallest training set.

**Fix:** '... and by a factor that runs from about three at the smallest training set to about 1.4 at the largest' -- or simply 'by roughly a factor of two to three'.

### Ch. 10 (Comparison) — `839-841`

**As written:** above $C = \capcappedfrom$ every cell stops at the $\capiterlimit$-iteration cap rather than at tolerance

**Problem:** Off by one against the generator's own count. sections/capacity-numbers.tex has \capcappedfrom = 4 with \capcappedcells = \capcappedtotal = 96 out of \capcells = 160; with 16 seeds x 2 sizes = 32 cells per capacity, 96 = three capacities, i.e. C in {4, 8, 16}. 'Above C = 4' names only C in {8, 16} = 64 cells. The macro name (\capcappedfrom) and the count both say capping starts at C = 4, not above it. The same sentence appears in ch11-conclusions.tex line 153.

**Fix:** 'from $C = \capcappedfrom$ upward every cell stops at the cap' (fix both chapters).

### Ch. 10 (Comparison) — `sections/tab-efficiency.tex:24 (rendered inside ch10 at line 350) vs ch10 lines 141-143 and 347-349`

**As written:** Appendix~\ref{app:aggregation} reports the same cells under five alternative aggregations.

**Problem:** The chapter prose says 'four alternative summaries' twice (lines 141-143 and 347-349), Appendix B says 'four alternatives', and Table B.1 carries one reported estimand (0) plus four alternatives (1)-(4); its column (5) is a bootstrap CI on (0), not an aggregation. Five estimands, four alternatives -- the table caption is the only place that says five alternatives.

**Fix:** In sections/tab-efficiency.tex change 'five alternative aggregations' to 'four alternative aggregations' (or 'five estimands').

### Ch. 11 (Conclusions) — `ch11-conclusions.tex:15-16`

**As written:** A single bulk quantity $q(\alpha,t) \in (0,|\alpha|]$ sets the posterior correlation length.

**Problem:** The upper endpoint should be open: q(alpha,t) < |alpha| strictly at every finite t, and |alpha| is only the t -> infinity limit. From ch10 Eq. (em-bulk-params)/(em-q-limits), q is the in-disc root of beta q^2 - J_d q + beta = 0, so |q| = (J_d - sqrt(J_d^2 - 4 beta^2))/(2|beta|) with J_d(t) = e^{-2t}/Delta_t + (1+a^2)/(1-a^2) and |beta| = |a|/(1-a^2). |q| is strictly decreasing in J_d (d|q|/dJ_d = (1 - J_d/sqrt(J_d^2-4beta^2))/(2|beta|) < 0) and J_d(t) is strictly decreasing to its t=infinity value, so |q| increases strictly to |alpha| without reaching it. Numerically at alpha=0.85: q(0.05)=0.20293, q(0.3)=0.52737, q(1)=0.75759, q(3)=0.84791, q(10)=0.849999998 -- always < 0.85. ch10 Eq. (em-q-limits) itself states 'q -> |corr|' as a limit and the prose there says the correlation length 'climbs monotonically from zero to the prior's own'. The asymmetry gives the slip away: 0 is EXCLUDED although it is attained at t=0 (Delta_0 = 0, sites pinned), while |alpha| is INCLUDED although it is attained nowhere.

**Fix:** Write $q(\alpha,t) \in (0,|\alpha|)$ (for $t>0$), or say 'rising from $0$ towards $|\alpha|$' if the endpoints are meant as limits.

### Cross-chapter — `ch11:16 vs ch10:530-539 (eq:em-q-limits)`

**As written:** A single bulk quantity $q(\alpha,t) \in (0,|\alpha|]$ sets the posterior correlation length.

**Problem:** ch10 states the same quantity as a strict limit, not an attained value: eq:em-q-limits gives 't -> infinity: q -> |corr|', and the sentence under it reads 'the posterior correlation length climbs monotonically from zero to the prior's own'. q = (J_d - sqrt(J_d^2 - 4 beta^2))/(2|beta|) is strictly decreasing in J_d and J_d(t) = e^{-2t}/Delta_t + (1+a^2)/(1-a^2) decreases strictly to its limit, so |alpha| is approached and never reached (alpha=0.85: q(1)=0.7576, q(3)=0.8479, q(10)=0.8499999). The closed endpoint in ch11 turns ch10's limit into an attainment. Same underlying error the ch11 agent reported; confirmed here against the other side.

**Fix:** ch11:16 -> $q(\alpha,t) \in (0,|\alpha|)$, or 'rising monotonically from $0$ towards $|\alpha|$ without attaining it'.

### Cross-chapter — `abstract:44-47 and ch11:79-80 vs ch10:1256-1259, tab-efficiency.tex:33-40, tab-structured.tex:26-29`

**As written:** abstract: 'it attains 6.6--10.9$\times$ lower schedule-pooled relative score error than a denoising score-matching network given no structural information, and 2.3--6.1$\times$ against a baseline given locality and weight sharing'; ch11: the same two ranges, with 'across \nsizesused{} sizes' attached only to the first.

**Problem:** The two ranges are not measured on comparable grids, and the abstract gives the reader nothing to stop him dividing them. The headline range spans 8 sizes, 32-4096 (ratios 6.9, 6.6, 7.9, 8.2, 8.2, 9.0, 8.6, 10.9); the structured range spans 4 sizes, 32-2048 (2.34, 2.82, 3.58, 6.14) AND a different scoring region and test bundle (ch10:176 protocol row: 1,024/2,048 for structured runs vs 256/256 for headline; ch10:366-368 caption: 'not on a common scale ... no number is carried between them'). 10.9 is attained at nseq=4096, where the structured arm was never run. Size-matched the two ratios give 6.9/2.34=2.95, 7.9/2.82=2.80, 8.2/3.58=2.29, 8.6/6.14=1.40 -- not a constant factor. ch11 compounds it by putting '\nsizesused{} sizes' (=8) next to a 4-size number in the same sentence. This is the abstract/conclusions face of the ch10 agent's 'roughly a factor of three' finding at ch10:808-811.

**Fix:** In ch11:79-80 state both grids ('across \nsizesused{} sizes; against a baseline given locality and weight sharing, $\structratiolo$--$\structratiohi\times$ over the \structsizes{} sizes of the all-site protocol'). In the abstract add 'over the four sizes of a separate all-site protocol' to the second range, or drop the second range's endpoints and say 'roughly 2--6x'.

### Cross-chapter — `ch10:777-783 vs ch10:176 (tab:em-protocol) and ch10:366-368 (caption of fig:em-value-of-structure)`

**As written:** 'Against the screened baseline --- the radius-$\screenwinnerradius$, width-$\screenwinnerwidth$ window head, $4{,}801$ parameters --- at the headline protocol throughout: the same twelve noise levels and $\structseeds$ seeds as Table~\ref{tab:pointwise}, every site scored rather than the centre alone ...'

**Problem:** 'at the headline protocol throughout' contradicts two other descriptions of the same run. The protocol table at ch10:176 gives 'validation / test: 1,024 / 2,048 sequences (structured runs); 256 / 256 (headline)' and training sizes '{32,...,4096} (headline) or {32,128,512,2048}', and the figure caption at ch10:366-368 says the two panels 'use different scoring regions and different test bundles --- and no number is carried between them'. The clause after the colon names one of those departures itself ('every site scored rather than the centre alone'), so the sentence asserts sameness and then lists a difference. This is the sentence a reader consults to decide whether 6.6--10.9 and 2.3--6.1 are on one scale.

**Fix:** ch10:778-779 -> 'at the shared generating protocol of Table~\ref{tab:em-protocol}, with the structured runs' own scoring region and test bundle:' and keep the list as the statement of what differs.

### Cross-chapter — `abstract:49-51 and ch11:126-127 vs ch10:1063-1067`

**As written:** abstract: 'a long-range one defeats the estimator at coupling strengths near one tenth'; ch11: 'while failing on a long-range one at a coupling strength of roughly one tenth'.

**Problem:** ch10 resolves that into two different strengths depending on the baseline: 'the advantage against the convolutional baseline is gone by $\gamma = \nmcrossover$ [0.1], having survived $\gamma = \nmlastholding$ [0.05], and by $\gamma = 0.2$ the estimator loses to both baselines and not to one'. So one tenth is the CNN-only crossing; against the MLP the estimator is still ahead at 0.1 and loses only at 0.2. 'Defeats the estimator' and 'failing' read as unconditional. The same split shows in the bracket ch10 actually reports: delta* in [1.1,2.2]e-2 (vs CNN) against [2.2,4.0]e-2 (vs MLP), i.e. the two crossings are a full tested strength apart.

**Fix:** abstract:50-51 -> 'a long-range one costs the estimator its advantage over the convolutional baseline near coupling strength one tenth, and over both baselines by one fifth'; mirror in ch11:126-127.

### Cross-chapter — `ch10:343 (and ch10:13) vs ch10:16-17, ch10:356, abstract:44-45, ch11:76-78`

**As written:** 'the structured estimator attains lower relative denoising error by a factor of $\ratiolo$--$\ratiohi$ across the \nsizesused{} training-set sizes'

**Problem:** The chapter's own estimand section says 'Every headline number in this chapter is a ratio of two relative $L^2$ score errors' (ch10:16-17, eq:em-relerr/eq:em-pooling), the figure caption calls it 'schedule-pooled relative score error' (ch10:356), the summary row calls it 'pooled error' (ch10:1256), and the abstract and conclusions both call it 'schedule-pooled relative score error'. Only the sentence that carries the headline factor calls it 'denoising error' -- a name the thesis uses elsewhere for a different quantity (ch07:660-670 measures the denoiser reading as excess MSE over the MMSE floor, which is not eq:em-relerr).

**Fix:** ch10:343 -> 'lower schedule-pooled relative score error \eqref{eq:em-pooling} by a factor of ...'; ch10:13 -> 'reduce score error'.

### Cross-chapter — `ch11:153 and ch10:840 vs sections/capacity-numbers.tex:26-28`

**As written:** ch11: 'Every capacity cell above $C = \capcappedfrom$ stops at an iteration cap rather than at tolerance'; ch10:840: 'above $C = \capcappedfrom$ every cell stops at the $\capiterlimit$-iteration cap'.

**Problem:** Both chapters say 'above C = 4', so they agree with each other but appear to be off by one grid point against the generator. The sweep is \capcomps = {1,2,4,8,16} at \capsizes = {128,512} with \capseeds = 16, i.e. 32 cells per capacity out of \capcells = 160. The generator exports \capcappedcells = \capcappedtotal = 96 = 3 x 32, which is C in {4,8,16} (at and above 4); strictly above 4 would be 64. Neither chapter quotes \capcappedcells, so nothing in the built text contradicts itself -- but if 96 is the capped count, both sentences exclude C = 4 wrongly, and C = 4 is inside the range the capacity claim rests on.

**Fix:** Confirm against make_tab_capacity.py which capacities are censored; if 96, change both to 'from $C = \capcappedfrom$ upward' (ch11:152-155 and ch10:840), and consider quoting \capcappedcells of \capcells so the count travels with the claim.


---

## ROUNDING / PRECISION — small numerical slips

### Ch. 6 (Gaussian BP) — `734`

**As written:** banded answer, $b = 1$ ... $t = 0.3$: $0.302$

**Problem:** The exact value is 0.3014606, which rounds to 0.301 at three decimals, not 0.302. Every other cell of the table matches its exact value (r0: 0.814737, 0.566519, 0.194083, 0.004482; r1: 0.122825, 0.235159, 0.130146, 0.003525; r2: 0.019627, 0.106840, 0.089213, 0.002773; b1: 0.209650, ., 0.134609, 0.003525; b2: 0.031847, 0.134423, 0.092402, 0.002773), so this is a one-in-the-last-place slip. It is also inconsistent with the table's own derived row: the printed penalty 1.28x equals the exact 0.3014606/0.2351592 = 1.28194, whereas 0.302/0.235 = 1.2851 would print as 1.29x.

**Fix:** Change $0.302$ to $0.301$ in the b = 1 row, t = 0.3 column.

### Ch. 6 (Gaussian BP) — `744-746`

**As written:** At $t = 0.05$ the isolated-frame score is wrong by $82\%$ in relative RMS

**Problem:** The table three lines above gives 0.815 for this same quantity, i.e. 81.5%, and the exact closed-form value is 0.8147374 = 81.47%, which rounds to 81%. '82%' is a double rounding (0.81474 -> 0.815 -> 82%) and reads as inconsistent with the table on the same page. The dependent sentences are unaffected either way: (0.8147374 - 0.1228250)/0.8147374 = 84.9% -> '85%' and (0.8147374 - 0.0196270)/0.8147374 = 97.6% -> '98%' both stand.

**Fix:** Write 'wrong by $81.5\%$ in relative RMS' (or '$0.815$ in relative RMS') to agree with Table~\ref{tab:g-truncation}.

### Ch. 7 (Laplace) — `115-116`

**As written:** At $t = 0.05$ the Laplace score saturates near $\pm 5.1$ while the Gaussian one grows without bound

**Problem:** Re-running the chapter's own generator (tools/fig_ch07.py, grid_bp_score with alpha=0.85, L=9, k=4, A=8, t=0.05, x_k swept over [-4,4] with the other frames at zero) gives a plateau of 5.2212, already flat from |x_k| ~ 2.5 to the plot edge, and identical at N_g = 401, 801 and 1601. The matched Gaussian reaches 14.42 at x_k = 4. So the saturation level is +/-5.2, not +/-5.1 (a 2.3% error in a value quoted to one decimal).

**Fix:** Change '$\pm 5.1$' to '$\pm 5.2$'.

### Ch. 7 (Laplace) — `622`

**As written:** Table~\ref{tab:l-closure}, row $t=0.072$: $10$--$90\%$ band $(0.038,\,0.082)$

**Problem:** ch07_metrics.csv row t=0.0725 has eps_pm_q90 = 0.0814660249996671, which rounds to 0.081 at the three decimals the table uses, not 0.082. (The lower endpoint 0.038057... -> 0.038 is right, and every other band entry in the table rounds correctly.)

**Fix:** Change $(0.038,\,0.082)$ to $(0.038,\,0.081)$.

### Ch. 8 (EM parameters) — `1183-1187`

**As written:** increasing the inner budget from four to sixteen sweeps moves the fitted innovation excess kurtosis from $1.68$ to $2.14$, a $28\%$ change relative to the former value

**Problem:** (2.14 - 1.68)/1.68 = 0.46/1.68 = 23/84 = 0.27380952..., i.e. 27.4%, which rounds to 27%, not 28%. A 28% change from 1.68 would land at 1.68*1.28 = 2.1504, and reaching 2.14 as a 28% rise would require a former value of 2.14/1.28 = 1.6719. No other reading gives 28% either: relative to the latter value it is 0.46/2.14 = 21.5%, relative to the true Laplace value 0.46/3.0 = 15.3%.

**Fix:** Change "$28\%$" to "$27\%$" (or state it as "about a quarter").

*Lean witness:* `ThesisAudit.ProseCh08.inner_budget_relative_change`

### Ch. 8 (EM parameters) — `1377`

**As written:** Table~\ref{tab:em-recovery}, row $C=8$, $\nseq=2048$: $\widehat\ivar = 0.2793$ with $|\Delta\ivar|/\ivar = 0.7\%$

**Problem:** Against the caption's true value $\ivar = 0.2775$, |0.2793 - 0.2775|/0.2775 = 0.0018/0.2775 = 18/2775 = 0.0064865 = 0.649%, which rounds to 0.6%, not 0.7%. Recomputed from the frozen source (research/nongaussian-bp/outputs/frozen/exp_18/innovation_summary.csv, innovation_var = 0.2792983440826006) the exact figure is 0.64805%, again 0.6%. Every other cell of that column rounds correctly to one decimal under round-to-nearest (3.197->3.2, 2.503->2.5, 1.532->1.5, 1.770->1.8, 2.916->2.9), so this row is the only one out of step; it also slightly overstates the best variance recovery in the table, which the following paragraph leans on.

**Fix:** Change the last row's $|\Delta\ivar|/\ivar$ entry from $0.7\%$ to $0.6\%$.

*Lean witness:* `ThesisAudit.ProseCh08.recovery_var_relative_error`

### Ch. 9 (EM kernel) — `296-297`

**As written:** the two arms finish ... within a fifth of a unit on excess kurtosis

**Problem:** Table 9.3 (tab-convergence) gives 2.939 (clean) and 2.729 (through the channel); 2.939 - 2.729 = 0.210, which is not within 0.2. Confirmed against the source: pooled medians of exp_06/clean_vs_noised_shape.csv are 2.9390 and 2.7286, difference 0.2104. The correlation half of the sentence is fine (0.8526 - 0.8509 = 0.0017 < 0.002).

**Fix:** "within about a fifth of a unit" or "within a quarter of a unit".

### Ch. 10 (Comparison) — `1213-1221 (caption of tab:em-information) vs 1224-1236`

**As written:** Entries are ratios of outer-product estimates at $D = 512$ sequences, so only the order is meaningful and two significant digits are printed for that reason.

**Problem:** The table prints more than two significant digits in three of its columns: loss in $\ivar$ is 235, 392, 459, 501, 522 (three s.f.), the contrast column is 112, 236, 776, 867, 1222 (three and four s.f.), and the summary rows print 465 and 493 (three s.f.). Only 'loss, rho' (78, 87, 83, 75, 56) and 'loss, p' (8.7e3, 2.0e4, 6.4e4, 6.5e4, 6.8e4) obey the stated rule. The contrast column is the one the caption says only the order of is meaningful, yet it is the one printed to four figures.

**Fix:** Either round the sigma-eta, contrast and summary entries to two significant digits (240, 390, 460, 500, 520; 110, 240, 780, 870, 1200; 470 against 490), or drop the 'two significant digits' claim from the caption.


---

## UNVERIFIABLE — cannot be checked from the submitted document

### Ch. 11 (Conclusions) — `ch11-conclusions.tex:137-139`

**As written:** no low-rank structure is asserted for the Laplace $\nsites = 2$ posterior Hessian beyond the narrow regime where it holds

**Problem:** Nothing in the submitted document supports 'the narrow regime where it holds': grep over thesis/chapters and thesis/sections finds 'low-rank'/'low rank' ONLY on this line, 'Hessian' only here and at ch05:545 (the Gaussian constant Hessian), and no L=2 / two-site Laplace analysis anywhere. The negative half of the sentence ('no low-rank structure is asserted') is self-consistent with that absence, but the trailing clause points at a result an examiner cannot find. Likely a leftover from material that lives in the research package / compendium rather than the thesis.

**Fix:** Either drop 'beyond the narrow regime where it holds', or add a pointer to where that regime is established (compendium section), since the thesis contains no L=2 Laplace Hessian analysis.

