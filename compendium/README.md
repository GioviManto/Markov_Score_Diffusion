# Companion Compendium — the extended research record

This folder is a self-contained LaTeX project: the complete, extended
version (about 230 pages) of the MSc thesis *Learning Scores for Markovian
Dynamics in Generative Diffusion*, of which the thesis at the repository
root is the concise, submitted form.

It carries every derivation written out, the step-by-step toolbox
derivations, the rotating-ring precursor study (its Chapter 4), and the two
original appendices (reproducibility; how the reported ratios were
aggregated). Its content is that of the extended build preserved at the git
tag `extended-thesis-pre-compression`; the only changes are the front matter
(a title page and a note in place of the university's four mandatory pages)
and the wording of the references to the earlier technical compendium, which
is superseded by this volume and kept in `../archive/`.

Chapter, section and equation numbers differ from those of the submitted
thesis. Every number in `sections/` is generated from the frozen experiment
outputs in `../research/outputs/frozen/` by the generators documented in the
thesis's Appendix A; the figures in `figures/` are the frozen copies the
extended build used, including the three rotating-ring figures and the six
literature reproductions that the concise thesis dropped.

## Building

```bash
tectonic main.tex
```

`./check.sh` builds and reports unresolved references, undefined citations,
duplicate labels, `??` in the PDF and overfull boxes. The document is
pdfLaTeX-compatible (Latin Modern), so on Overleaf the default compiler
works.
