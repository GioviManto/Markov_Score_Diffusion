# audit — independent checks

Scripts that re-derive a result by a different route than the code under test, plus `independent_replication/` — a from-scratch implementation used to check the main one.

`AUDIT_NOTE.md` is the running record. These have caught claims that code inspection did not: an M-step budget that under-converged the innovation shape, a boundary statistic quoted as a bound, a sign error in a gradient that survived every internal diagnostic.

The best of these become permanent tests in `../tests/`.
