# The research code behind the thesis

This directory is the research package the thesis was built from: the belief
propagation and EM implementations, the experiments that produced every
reported number, the generators that turn frozen outputs into the document's
tables and figures, and the frozen outputs themselves.

The thesis at the repository root is the *document*; this is the *evidence*.

## Layout

| Path | What it is |
| --- | --- |
| `src/` | the library — grid and Gaussian BP, the EM/ECM kernels, priors, the noising channel, the network baselines |
| `experiments/` | one script per experiment (`exp_07_…`, `exp_31_…`); these wrote `outputs/frozen/` |
| `outputs/frozen/` | the frozen experiment outputs the thesis cites, 870 files |
| `tools/` | the generators that emit the thesis's `sections/*.tex` from those outputs, plus provenance and check scripts |
| `tests/` | the test suite, including `test_audit_fixes.py`, which pins the corrected claims |
| `audit/` | standalone audit scripts written to check specific claims |
| `hpc/` | SLURM batch scripts and the clean-deploy tooling the sweeps ran under |
| `notebooks/` | the exploratory notebooks, kept as the record of how the questions were approached |
| `simple/`, `docs/` | a minimal reference implementation, and design notes |

## Running it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q
```

The regression suite in `tests/test_audit_fixes.py` is the one to run first if
you want to know whether the corrected claims still hold: it pins the pooled
estimand against the retired one, the finite low-noise limit of
`tr(Sigma_t^-1)`, the exact windowed-oracle values, the ECM-versus-EM Jacobian,
the parameter counts, and the quadrature weights.

## The rule that matters

Every number in the thesis comes from a generator in `tools/`, reading
`outputs/frozen/`. None is typed by hand. Each generated `.tex` says so on its
first line, and editing one is silently undone the next time its generator
runs — change the generator instead.

To check that for yourself, the headline is the shortest path: the ratio the
thesis reports as 6.6–10.9× is what `tools/make_tab_efficiency.py` computes
from `outputs/frozen/exp_07_*` under the schedule-pooled paired per-seed
estimand of Eq. (10.2).

## Provenance, stated plainly

Appendix A of the thesis classifies each experiment by how strongly its
provenance is established, and the honest summary is that they differ: some
runs carry a certified environment digest, one rests on a legacy digest with
an empty intersection, and six record configuration and seed but no source
revision. That last group includes experiments the thesis cites. Rerunning
today's code reproduces today's code; it does not retroactively certify what
produced a frozen output in July. The appendix says which runs are in which
category rather than presenting a uniform guarantee.

`REDACTIONS.md` records the one substitution applied to this tree before it
was made public.
