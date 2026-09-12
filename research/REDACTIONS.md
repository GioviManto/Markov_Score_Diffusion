# Redactions applied before publication

One substitution was applied to this tree when it was made public. It is
recorded here rather than left for a reader to find by diffing.

## What was replaced

The author's university cluster account number, wherever it appeared **as a
filesystem path component**:

* `hpc/` batch scripts (26 files): it formed the SLURM log paths and the
  project root, which now read `$HOME` and `$BP_CLUSTER_USER` -- what those
  paths meant, and portable. `hpc/deploy_clean.sh`, `hpc/sync_to_cluster.sh`
  and `tools/pull_and_check.sh` likewise take the cluster host and user from
  `BP_CLUSTER_HOST` / `BP_CLUSTER_USER` and exit if they are unset, rather
  than defaulting to an internal hostname.

* three frozen configuration records, inside absolute paths recorded at run
  time, where the account now reads `<cluster-user>`:

  * `outputs/frozen/exp_31_confirm_merged/params_confirm.json`
  * `outputs/frozen/exp_31_confirm_merged/params_confirm_b.json`
  * `outputs/frozen/exp_31_confirm_merged/params_confirm_c.json`

**No measured value was touched.** Those three files are the only ones in
`outputs/frozen/` that differ from the working tree; the other 867 are
byte-identical, verified by recursive diff.

## A trap worth recording

The first attempt replaced the account number by plain string match across the
whole tree. The account number is a seven-digit string, and seven-digit strings
also occur inside the mantissas of recorded floating-point results: that pass
rewrote a measured score error of `0.6529684316454211` into
`0.6529684<cluster-user>11` in two result CSVs, and would have published
corrupted evidence under a note claiming nothing had been altered. The
substitution above therefore matches only `/home/<account>` and `<account>@`,
never the bare digits, and the three CSVs it had damaged are byte-identical to
the working tree again.

## What is absent rather than redacted

The Python virtual environment, the non-frozen scratch outputs, and internal
correspondence with advisors. None of those is code or evidence.
