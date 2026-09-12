#!/usr/bin/env bash
# Build the thesis and report every defect class, refusing to look at a stale PDF.
#
# WHY THE EXIT CODE IS CHECKED FIRST. tectonic leaves the previous main.pdf in
# place when a run fails. A check that compiles with output suppressed and then
# inspects main.pdf will happily report "0 unresolved references, 144 pages" from
# a PDF built before the error -- which is exactly what happened on 4 September
# 2026 while adding Appendix B: a `Double subscript` error halted the build for
# three consecutive "clean" reports. Build first, check the status, and only then
# read the artefact.
#
#   ./check.sh          build and report
#   ./check.sh --quiet  report only the summary line
#
# Exits non-zero if the build fails or any defect class is non-empty.

set -uo pipefail
cd "$(dirname "$0")"

QUIET=0
[[ ${1:-} == "--quiet" ]] && QUIET=1
say() { [[ $QUIET -eq 1 ]] || echo "$@"; }

log=$(mktemp)
if ! tectonic -X compile main.tex --keep-intermediates --keep-logs --reruns 4 \
        >"$log" 2>&1; then
    echo "BUILD FAILED -- main.pdf is stale and was not inspected:" >&2
    grep -iE '^error|^!' "$log" | head -10 | sed 's/^/    /' >&2
    exit 1
fi
say "build ok"

python3 - "$QUIET" <<'PY'
import re, subprocess, sys, pathlib
quiet = sys.argv[1] == "1"
log = pathlib.Path("main.log").read_text(errors="ignore")
last = log.rsplit("Running TeX", 1)[-1]        # final pass only
pdf = subprocess.run(["pdftotext", "main.pdf", "-"],
                     capture_output=True, text=True).stdout
info = subprocess.run(["pdfinfo", "main.pdf"], capture_output=True, text=True).stdout

overfull = [float(x) for x in
            re.findall(r'Overfull \\hbox \(([\d.]+)pt too wide\)', last)]
# The title the university registers, carried by title.tex and stamped into
# the PDF metadata. A submitted artefact whose metadata names a different
# document is a provenance defect, and it is invisible in the rendered pages,
# so it is checked here rather than trusted.
title = pathlib.Path("title.tex").read_text()
want = title.split("{Learning", 1)[-1]
want = "Learning" + want.split("}")[0] if "Learning" in title else None
got = ""
for line in info.splitlines():
    if line.startswith("Title:"):
        got = line.split(":", 1)[1].strip()

defects = {
    "undefined references": len(set(re.findall(r"Reference `([^']+)' on page", last))),
    "PDF title != title.tex": 0 if (want and got == want) else 1,
    "undefined citations":  len(set(re.findall(r"Citation `([^']+)' on page", last))),
    "multiply-defined labels": len(set(re.findall(r"Label `([^']+)' multiply", last))),
    "'??' printed in the PDF": pdf.count("??"),
    "missing glyphs": len(re.findall(r'Missing character', last)),
    # Stray LaTeX tokens typeset as prose. This class got through twice before
    # ("noindentInterior diagonal" on p.78, a ```latex fence in ch02), because
    # nonstop mode happily sets a mangled control sequence as words. Lowercase
    # with word boundaries: \LaTeX renders as uppercase LATEX, so the logo does
    # not trip it.
    # Leading \\b only for the glued forms: the failure this catches is a
    # control sequence run into the following word, so requiring a trailing
    # boundary would miss exactly it. Every alternative below is chosen not to
    # be the prefix of an English word ("emph" was, and is not listed).
    "stray source tokens": len(re.findall(
        r'(?:\b(?:noindent|textbf|textit|eqref|includegraphics|citep|citet'
        r'|newcommand|renewcommand|vspace|hspace|linewidth|textwidth)'
        r'|\b(?:latex|begin\{|end\{)\b)', pdf)),
    # A few points over is invisible; 10pt is a word in the margin.
    "overfull hboxes > 10pt": sum(1 for v in overfull if v > 10),
}
# abstract.tex is NOT \input by the build -- the submission system ingests it
# separately -- so its headline figures are the one place in the bundle where a
# generated number must be transcribed by hand. Check the transcription against
# the macros rather than trusting it.
macros = {}
for f in pathlib.Path("sections").glob("*-numbers.tex"):
    for name, val in re.findall(r'\\newcommand\{\\(\w+)\}\{([^}]*)\}', f.read_text()):
        macros[name] = val
abstract = pathlib.Path("abstract.tex").read_text()
abstract_bad = []
for lo, hi in (("ratiolo", "ratiohi"), ("structratiolo", "structratiohi")):
    if lo in macros and hi in macros:
        want = f"{macros[lo]}--{macros[hi]}"
        if want not in abstract:
            abstract_bad.append(want)
defects["abstract vs generated macros"] = len(abstract_bad)

pages = info.split("Pages:")[1].split()[0]
# \s+ between the word and the number: pdftotext breaks a caption line
# wherever the typeset line broke, so "Figure\n7.1:" is the same caption as
# "Figure 7.1:" and must not read as one figure fewer. Two builds of this
# document disagreed by exactly that, which looked like a lost figure.
figs = len(set(re.findall(r'Figure\s+(\d+\.\d+):', pdf)))
tabs = len(set(re.findall(r'Table\s+(\d+\.\d+):', pdf)))
cites = len(re.findall(r'.bibitem', pathlib.Path("main.bbl").read_text()))

if not quiet:
    for k, v in defects.items():
        print(f"  {k:<26}{v}")
    print(f"  {'overfull (any size)':<26}{len(overfull)} "
          f"(worst {max(overfull, default=0):.1f}pt)")
    print(f"  {'pages':<26}{pages}")
    print(f"  {'figures / tables':<26}{figs} / {tabs}")
    print(f"  {'citations':<26}{cites}")

bad = {k: v for k, v in defects.items() if v}
if bad:
    print("DEFECTS: " + ", ".join(f"{k}={v}" for k, v in bad.items()))
    sys.exit(1)
print(f"clean: {pages} pp, {figs} figures, {tabs} tables, {cites} citations")
PY
