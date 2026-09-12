# Independent replication (from a separate implementation)

These three CSVs came from `research/em-learning/`, a from-scratch reimplementation of the
chain-model / grid-BP / EM machinery that once lived as a separate, parallel effort in this
repo. That effort's own audit (`ASKS-AUDIT.md`, archived at
`../../../../../archive/em-learning-superseded-2026-08-13/ASKS-AUDIT.md`, outside this repo)
concluded it was a redundant reimplementation of what is now `research/nongaussian-bp/src/` —
every gap it reported (reverse diffusion, MLP-vs-BP, receptive field/depth) was already closed
here, in more depth (`src/reverse.py`, `experiments/exp_05_reverse_dynamics.py`,
`experiments/exp_07_em_vs_score_network.py`, `experiments/exp_12_receptive_field.py`).

One thing from that effort is worth keeping: a 5,530-row cluster sweep (48 CPU-hours, 0 failed
cells; 5 innovation families x 4 rho x 3 n x 3 M x 3 reps), because it independently
**reproduces** this project's central correction on the Gaussian-closure error — that it decays
monotonically in `t` with no large-`t` blowup (0.200 -> 9.2e-5 for Laplace). A second,
independently-written implementation landing on the same number is real corroborating evidence.
It is not a new result and should not be cited as one — treat it as a replication check, not as
additional science.

| file | what it covers |
|---|---|
| `identifiability_all.csv` | EM parameter recovery (`rho`, `q`) across families/regimes, 4,320 fits |
| `nonlinearity_gap_all.csv` | Gaussian-closure vs exact-BP score gap as a function of `t` — the replicated finding |
| `sample_efficiency_all.csv` | EM parameter recovery error vs number of training chains |

The code that produced these, and the rest of that effort's documentation, is archived outside
this repository at `../archive/em-learning-superseded-2026-08-13/` (sibling of the repo root) —
not deleted, just not part of the tracked tree.
