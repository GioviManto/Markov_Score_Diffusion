# experiments — the runs behind every number

29 numbered experiments. Each is a script with `--quick` (smoke) and `--only`
(run one part), writing CSVs and a `params_*.json` manifest into `outputs/`.

## frozen_config.py is the point

Every experiment imports `FROZEN`. It fixes ρ, the grid, the noise schedule, the
replicate count, the innovation-mixture size and the iteration budgets in **one**
place, so no run can quietly diverge from what the documents describe.
`tools/check_paper.sh` fails if an experiment sets any of them locally.

This exists because it went wrong: replicate counts of 3, 6 and 16 coexisted, and
the efficiency experiment carried its own size list while `FROZEN.sizes` held a
different one and the paper's appendix quoted a third. The config now records
`efficiency_sizes` separately, with the reason it differs.

## The ones that reach the documents

| Experiment | What it establishes |
|---|---|
| `exp_01`, `exp_18` | discretisation control — truncation and quadrature, separated and measured |
| `exp_02`, `exp_03` | what Gaussian message closure costs, against the true kernel |
| `exp_06`, `exp_08` | parameter recovery; gradient ascent against the exact M-step |
| `exp_07` | **the efficiency comparison** — EM–BP against trained denoisers, both budgets validation-selected |
| `exp_27` | shape convergence: ρ settles by ~25 iterations, the innovation shape needs ~120 |
| `exp_28` | the rotating ring, joint against marginal |
| `exp_29` | EM overfitting — training evidence rises while held-out score error turns |

The rest are exploratory branches (wavelets, video, hierarchies, discrete
alphabets) or superseded sweeps kept for the record.

## Reading an output

`outputs/frozen/` holds the runs the documents cite. Each carries a
`params_*.json` with the configuration, the seeds, the git commit (or the deploy
stamp, on the cluster) and the environment.

Since 18 Aug 2026 `exp_07` also writes a **certificate** per cell:
`em_resolved` (is the narrowest mixture component wider than two grid cells),
`em_inner_converged`, `em_outer_stop_reason`. `make_tab_efficiency.py` refuses to
build the table from cells that fail the resolution test, and says "NOT
CERTIFIED" rather than passing quietly for older outputs that predate the column.

## Conventions

- One noise view per chain: a training set of N chains yields N observations,
  split across the twelve levels. So N counts *independent latent sequences*.
- Seeds come from `utils.rng_for(...)`; a cell is reproducible from its name.
- A run that hits its cap is **censored**, and says so.
