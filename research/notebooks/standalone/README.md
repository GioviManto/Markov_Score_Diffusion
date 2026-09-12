# Standalone notebooks

Notebooks in this directory are **self-contained**: they define their own
implementation rather than importing from `src/`, and read no files from
`outputs/`. They are documents you can hand to someone with a Python
environment and nothing else.

They live in a subdirectory because `tools/check_all.sh` globs
`notebooks/*.ipynb` and re-executes every match. These cannot participate in
that: `EM_BP_theory_and_experiments.ipynb` imports `torch`, which is not in the
project venv (`.venv`), so the check would fail on a missing dependency rather
than on anything being wrong. Excluding them by location is honest about that;
teaching the checker to skip notebooks it cannot run would not be.

The consequence is worth stating plainly: **the outputs committed here are not
verified against a fresh run**, unlike every notebook one level up. Treat their
numbers as a record of one execution rather than as something the repository
re-derives.

## Contents

- **`EM_BP_theory_and_experiments.ipynb`** — the full theoretical development
  with ten experiments, executed. Part I derives the method (why the noised
  posterior is still a chain, EM from the observed likelihood, Fisher's
  identity, missing information and EM's convergence rate, the mixture
  innovation model); Part II is a minimal from-scratch implementation; Part III
  runs ten experiments against it, from grid-BP validation through parameter
  recovery, the exact M-step against gradient ascent, the price of noising,
  sample-size scaling, learning a non-Gaussian innovation without naming it,
  and a small neural comparator; Part IV states what it establishes.

  Overlaps deliberately with `../05_em_from_scratch.ipynb`, which builds the
  same machinery in fewer steps against the repository's own code. Read 05 to
  understand the method quickly; read this one for the derivations.

  *Known defect, left as received:* the heading "Experiment 9 — Does the learned
  kernel yield a useful denoiser and score?" appears twice. The second instance
  precedes an end-of-notebook audit table, not a repeat of Experiment 9, so the
  heading is wrong rather than the content duplicated. Renumbering is the
  author's call.
