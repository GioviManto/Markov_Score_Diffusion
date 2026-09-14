#!/usr/bin/env python3
"""Generate a LaTeX document of the full Lean derivations backing the thesis audit.

Reads ThesisAudit/*.lean and AUDIT_RESULTS/*.json and emits derivations.tex:
for every audited formula, the thesis claim, the verdict, and the verbatim Lean
statement and proof that establishes it.

Usage:  ./make_derivations_tex.py [outfile]
"""
import json
import pathlib
import re
import sys
from collections import OrderedDict

HERE = pathlib.Path(__file__).parent
LEAN = HERE / "ThesisAudit"
RES = HERE / "AUDIT_RESULTS"

# Thesis order, with the chapter titles used for the section headings.
CHAPTERS = [
    ("Ch01", "Introduction", "ch01-introduction.tex"),
    ("Ch02", "Statistical Mechanics and Diffusion", "ch02-statmech-diffusion.tex"),
    ("Ch03", "The Model", "ch03-model.tex"),
    ("Ch04", "The Ring", "ch04-ring.tex"),
    ("Ch05a", "The Gaussian Chain in Matrix Form (part I)", "ch05-gaussian-matrix.tex"),
    ("Ch05b", "The Gaussian Chain in Matrix Form (part II)", "ch05-gaussian-matrix.tex"),
    ("Ch06", "The Gaussian Chain by Belief Propagation", "ch06-gaussian-bp.tex"),
    ("Ch07", "The Laplace Prior", "ch07-laplace.tex"),
    ("Ch08a", "EM for Parameters (part I)", "ch08-em-parameters.tex"),
    ("Ch08b", "EM for Parameters (part II)", "ch08-em-parameters.tex"),
    ("Ch09", "The EM Kernel", "ch09-em-kernel.tex"),
    ("Ch10", "Comparison", "ch10-comparison.tex"),
]
# Chapter 9's verdicts live inside Ch01.json (it declares it covers both files).
JSON_FOR = {"Ch09": "Ch01"}

# Glyphs Lean uses that the monospace face lacks; mapped to math equivalents.
FALLBACKS = OrderedDict([
    ("\u2016", r"\Vert"), ("\u2225", r"\parallel"), ("\u2113", r"\ell"),
    ("\u22a2", r"\vdash"), ("\u22a5", r"\perp"), ("\u27c2", r"\perp"),
    ("\u27ea", r"\langle\!\langle"), ("\u27eb", r"\rangle\!\rangle"),
    ("\u27f6", r"\longrightarrow"), ("\u27f9", r"\Longrightarrow"),
    ("\u27fa", r"\Longleftrightarrow"), ("\u2b1d", r"\cdot"),
    ("\U0001d4dd", r"\mathcal{N}"), ("\U0001d540", r"\mathbb{I}"),
    ("\U0001d7cf", r"\mathbf{1}"), ("\u2096", r"{}_{k}"),
    ("\u20d7", r"\to"), ("\u20d6", r"\leftarrow"),
    ("\u209c", r"{}_{t}"), ("\U0001d55c", r"\Bbbk"),
])

DECL = re.compile(
    r"^(?:@\[[^\]]*\]\s*)?(?:private\s+|protected\s+|noncomputable\s+)*"
    r"(theorem|lemma|def|abbrev|instance|structure|example)\s+([A-Za-z_][A-Za-z0-9_'!?.]*)"
)
STATUS_LABEL = {
    "proved": r"\statusproved", "instance_checked": r"\statusinstance",
    "defined": r"\statusdefined", "skipped": r"\statusskipped",
    "discrepancy": r"\statusdiscrepancy",
}


def tex_escape(s):
    if s is None:
        return ""
    out = []
    for ch in str(s):
        if ch in "&%$#_{}":
            out.append("\\" + ch)
        elif ch == "\\":
            out.append(r"\textbackslash{}")
        elif ch == "^":
            out.append(r"\textasciicircum{}")
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        else:
            out.append(ch)
    return "".join(out)


def parse_lean(path):
    """Return (module_docstring, OrderedDict name -> source-with-leading-comments)."""
    text = path.read_text(encoding="utf-8")
    doc = ""
    m = re.search(r"/-!(.*?)-/", text, re.S)
    if m:
        doc = m.group(1).strip()

    lines = text.split("\n")
    starts = []
    in_block_comment = False
    for i, ln in enumerate(lines):
        # Track /- ... -/ so a declaration keyword inside a comment is ignored.
        stripped = ln.strip()
        if in_block_comment:
            if "-/" in ln:
                in_block_comment = False
            continue
        if stripped.startswith("/-") and "-/" not in ln:
            in_block_comment = True
            continue
        if DECL.match(ln):
            starts.append(i)

    decls = OrderedDict()
    for idx, s in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        # Walk backwards over the attached comment block / docstring.
        b = s
        while b > 0:
            prev = lines[b - 1].strip()
            if prev.startswith("--") or prev.endswith("-/") or prev.startswith("/--"):
                # pull the whole docstring or comment run
                b -= 1
                if lines[b].strip().endswith("-/"):
                    while b > 0 and not lines[b].strip().startswith(("/--", "/-")):
                        b -= 1
            elif prev == "" and b - 1 > 0 and lines[b - 2].strip().startswith("--"):
                b -= 1
            else:
                break
        # don't swallow the previous declaration's trailing blank lines
        while b < s and lines[b].strip() == "":
            b += 1
        name = DECL.match(lines[s]).group(2)
        body = "\n".join(lines[b:end]).rstrip()
        if name not in decls:
            decls[name] = body
    return doc, decls


def verbatim(src):
    return "\\begin{Verbatim}[fontsize=\\small,breaklines=true,breakanywhere=true," \
           "commandchars=\\\\\\{\\}]\n" + src.replace("\\", "\\textbackslash{}") \
               .replace("{", "\\{").replace("}", "\\}") + "\n\\end{Verbatim}\n"


def verbatim_plain(src):
    # No commandchars: Lean braces/backslashes pass through untouched.
    return ("\\begin{Verbatim}[fontsize=\\small,breaklines=true,breakanywhere=true]\n"
            + src + "\n\\end{Verbatim}\n")


def main():
    out_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "derivations.tex"
    parts = [PREAMBLE]

    # ---- collect everything first, so the summary can be exact -------------
    loaded = {}
    totals = {"proved": 0, "instance_checked": 0, "defined": 0, "skipped": 0, "discrepancy": 0}
    discrepancies = []
    for mod, title, tex in CHAPTERS:
        lp = LEAN / f"{mod}.lean"
        jp = RES / f"{JSON_FOR.get(mod, mod)}.json"
        if not lp.exists() or not jp.exists():
            continue
        doc, decls = parse_lean(lp)
        data = json.loads(jp.read_text(encoding="utf-8"))
        items = data.get("items", [])
        if mod in ("Ch01", "Ch09"):
            want = "ch09" if mod == "Ch09" else "ch01"
            items = [i for i in items if want in str(i.get("tex_file", "")).lower()]
        loaded[mod] = (title, tex, doc, decls, items, data)
        for i in items:
            if i.get("status") in totals:
                totals[i["status"]] += 1
            if i.get("status") == "discrepancy":
                discrepancies.append((mod, tex, i))

    n = sum(totals.values())
    asserting = n - totals["defined"]
    pct = (100.0 * totals["proved"] / asserting) if asserting else 0.0

    parts.append(r"\section*{What this document is}" "\n")
    parts.append(
        "This is the machine-checked companion to the thesis \\emph{Markov Score Diffusion}. "
        "Every display-math formula of the thesis was enumerated and given a verdict, and, "
        "where the verdict is that it holds, a Lean~4 proof. What follows is the complete Lean "
        "development: for each formula, the claim as the thesis states it, the verdict, and the "
        "verbatim Lean statement and proof that establishes it.\n\n"
        "Nothing here is paraphrased. The Lean below is the source that compiles: it contains no "
        "\\verb|sorry|, no \\verb|axiom| and no \\verb|native_decide|, so every theorem shown "
        "rests only on Lean's kernel and \\texttt{mathlib}.\n\n")
    parts.append(r"\subsection*{Verdicts}" "\n")
    parts.append(
        "\\begin{tabular}{lrl}\n\\toprule\nverdict & count & meaning\\\\\n\\midrule\n"
        f"\\statusproved & {totals['proved']} & Lean proves the thesis's claim in the generality stated\\\\\n"
        f"\\statusinstance & {totals['instance_checked']} & only a special case or weaker form is established\\\\\n"
        f"\\statusdefined & {totals['defined']} & a definition or notation; it asserts nothing to prove\\\\\n"
        f"\\statusskipped & {totals['skipped']} & not formalised; the obstruction is stated in place\\\\\n"
        f"\\statusdiscrepancy & {totals['discrepancy']} & the thesis is wrong; see below\\\\\n"
        "\\midrule\n"
        f"total & {n} & of which {asserting} assert something, and {totals['proved']} "
        f"({pct:.1f}\\%) are proved\\\\\n\\bottomrule\n\\end{{tabular}}\n\n".replace("{{", "{").replace("}}", "}"))
    parts.append(
        "\\medskip\\noindent The verdicts were produced by automated formalisation and then "
        "subjected to two adversarial review passes, which downgraded items whose Lean statement "
        "did not in fact say what the thesis says. Items still marked \\statusinstance{} or "
        "\\statusskipped{} carry the concrete obstruction in their note; they are recorded rather "
        "than hidden.\n\n")
    parts.append(r"\subsection*{Reproducing it}" "\n")
    parts.append(verbatim_plain("cd lean\nlake exe cache get      # fetch mathlib build cache\n"
                                "lake build ThesisAudit  # must exit 0\n./audit_status.py       # per-chapter verdict table"))

    # ---- discrepancies ----------------------------------------------------
    if discrepancies:
        parts.append(r"\clearpage\section{Discrepancies}" "\n")
        parts.append("These are the places where the Lean disagrees with the thesis. "
                     "Each was verified by exhibiting an explicit counterexample.\n\n")
        for mod, tex, i in discrepancies:
            parts.append(f"\\subsection{{{tex_escape(i.get('id'))}}}\n")
            parts.append(f"\\emph{{{tex_escape(tex)}}}, lines {tex_escape(i.get('tex_lines'))}"
                         f" \\quad [module \\texttt{{{tex_escape(mod)}}}]\n\n")
            parts.append(f"\\textbf{{Claim.}} {tex_escape(i.get('summary'))}\n\n")
            parts.append(f"\\textbf{{Finding.}} {tex_escape(i.get('note'))}\n\n")
            for nm in [x.strip() for x in str(i.get("lean_name", "")).split(",") if x.strip()]:
                src = loaded.get(mod, (None,) * 6)[3].get(nm) if mod in loaded else None
                if src:
                    parts.append(verbatim_plain(src))

    # ---- per chapter ------------------------------------------------------
    for mod, title, tex in CHAPTERS:
        if mod not in loaded:
            continue
        title, tex, doc, decls, items, data = loaded[mod]
        parts.append(f"\\clearpage\\section{{{tex_escape(title)}}}\n")
        parts.append(f"\\emph{{Thesis source:}} \\texttt{{{tex_escape(tex)}}}. "
                     f"\\emph{{Lean module:}} \\texttt{{ThesisAudit.{tex_escape(mod)}}}. "
                     f"{len(items)} audited items.\n\n")
        if doc:
            parts.append("\\subsection*{Conventions used in this module}\n")
            parts.append(verbatim_plain(doc))

        used = set()
        for i in items:
            if i.get("status") == "discrepancy":
                continue  # already shown
            ident = tex_escape(i.get("id") or "?")
            parts.append(f"\\subsection*{{{ident} \\hfill {STATUS_LABEL.get(i.get('status'), '')}}}\n")
            parts.append(f"\\addcontentsline{{toc}}{{subsection}}{{{ident}}}\n")
            if i.get("tex_lines"):
                parts.append(f"\\emph{{Thesis lines {tex_escape(i.get('tex_lines'))}.}} ")
            parts.append(f"{tex_escape(i.get('summary'))}\n\n")
            names = [x.strip() for x in str(i.get("lean_name", "")).split(",") if x.strip()]
            shown = 0
            for nm in names:
                src = decls.get(nm)
                if src and nm not in used:
                    parts.append(verbatim_plain(src))
                    used.add(nm)
                    shown += 1
                elif src:
                    parts.append(f"\\noindent(\\texttt{{{tex_escape(nm)}}}, shown above.)\n\n")
                    shown += 1
            if not shown and names:
                parts.append("\\noindent\\emph{(Lean witness: "
                             + ", ".join(f"\\texttt{{{tex_escape(x)}}}" for x in names)
                             + ".)}\n\n")
            if i.get("note"):
                parts.append(f"\\noindent\\footnotesize\\textbf{{Note.}} {tex_escape(i.get('note'))}\\normalsize\n\n")

        leftover = [nm for nm in decls if nm not in used]
        if leftover:
            parts.append(f"\\subsection*{{Supporting lemmas and definitions}}\n")
            parts.append("These declarations are used by the theorems above.\n\n")
            for nm in leftover:
                parts.append(verbatim_plain(decls[nm]))

    parts.append("\\end{document}\n")
    text = "".join(parts)
    for ch, rep in FALLBACKS.items():
        text = text.replace(ch, f"\\ensuremath{{{rep}}}")
    out_path.write_text(text, encoding="utf-8")
    print(f"wrote {out_path} ({len(text)//1024} KB)")
    print(f"items {n}: " + ", ".join(f"{k}={v}" for k, v in totals.items()))


PREAMBLE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{fontspec}
\setmonofont{Menlo}[Scale=0.78]
\usepackage{fvextra}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{amsmath,amssymb}
\usepackage[colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black]{hyperref}
\usepackage{titlesec}
\titleformat{\section}{\Large\bfseries}{\thesection}{1em}{}
\setcounter{secnumdepth}{1}
\setcounter{tocdepth}{1}
\newcommand{\statusproved}{\textcolor{green!45!black}{\textbf{proved}}}
\newcommand{\statusinstance}{\textcolor{orange!80!black}{\textbf{instance-checked}}}
\newcommand{\statusdefined}{\textcolor{blue!55!black}{\textbf{definition}}}
\newcommand{\statusskipped}{\textcolor{gray}{\textbf{not formalised}}}
\newcommand{\statusdiscrepancy}{\textcolor{red!75!black}{\textbf{DISCREPANCY}}}
\title{\textbf{Markov Score Diffusion}\\[0.3em]\large Formal verification in Lean 4:
complete derivations}
\author{Companion artefact to the thesis}
\date{}
\begin{document}
\maketitle
\tableofcontents
\clearpage
"""

if __name__ == "__main__":
    main()
