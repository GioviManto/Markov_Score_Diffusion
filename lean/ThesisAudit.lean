/-
Formal audit of the thesis "Markov Score Diffusion".

Each `ThesisAudit.ChNN` module formalises the display-math formulas of the
corresponding chapter; the per-formula verdicts live in
`AUDIT_RESULTS/ChNN.json`.  Chapter 9's four verdicts are recorded inside
`AUDIT_RESULTS/Ch01.json` (which declares that it covers both
`ch01-introduction.tex` and `ch09-em-kernel.tex`), but the Lean theorems they
cite live in `ThesisAudit.Ch09`, which is why that module is imported and
built in its own right below.

All twelve chapter modules build cleanly and are imported here, so
`lake build ThesisAudit` is a single green-or-red signal for the audited
portion of the thesis.  The audit contains no `sorry`, no `axiom` and no
`native_decide`.

A green build means Lean accepted these proofs.  It does NOT mean every
formalisation captures the full content of the thesis's prose: see
AUDIT_REPORT.md for the two discrepancies the author must fix, for the
fidelity review of the formalisations themselves, and for the complete list
of claims that are instance-checked, definitional or skipped rather than
proved in the generality the thesis states them.
-/
import ThesisAudit.BasicIdentities
import ThesisAudit.Ch01
import ThesisAudit.Ch02
import ThesisAudit.Ch03
import ThesisAudit.Ch04
import ThesisAudit.Ch05a
import ThesisAudit.Ch05b
import ThesisAudit.Ch06
import ThesisAudit.Ch07
import ThesisAudit.Ch08a
import ThesisAudit.Ch08b
import ThesisAudit.Ch09
import ThesisAudit.Ch10
