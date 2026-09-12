"""Render the project's advisor Markdown documents to LaTeX, and thence to PDF.

Deliberately narrow: it handles exactly the Markdown this project writes -- ATX
headings, pipe tables, bullet and numbered lists, block quotes, horizontal rules,
emphasis, inline code, and `$...$` maths passed through untouched. It is not a
general converter and will not silently mangle constructs it does not know: it
raises on an unclosed table and leaves anything unrecognised as a paragraph.

    python3 tools/md2tex.py EXECUTIVE_SUMMARY.md            # -> .tex and .pdf
    python3 tools/md2tex.py --no-pdf ANSWERS_AND_QUESTIONS_FOR_ADVISORS.md
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.4cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{microtype}
\usepackage{enumitem}
\usepackage[colorlinks=true,linkcolor=black,urlcolor=black,citecolor=black]{hyperref}
\setlist{topsep=3pt,itemsep=1.5pt,parsep=0pt}
\renewcommand{\arraystretch}{1.15}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\begin{document}
"""

# Characters that must be escaped outside maths and verbatim spans.
_ESCAPES = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "$": r"\$"}


def _protect(text: str) -> tuple[str, list[str]]:
    """Replace `code` and $maths$ spans by placeholders so escaping skips them."""
    stash: list[str] = []

    def keep(m: re.Match) -> str:
        stash.append(m.group(0))
        return f"\0{len(stash) - 1}\0"

    text = re.sub(r"`[^`]*`", keep, text)
    text = re.sub(r"\$[^$]*\$", keep, text)
    return text, stash


def _restore(text: str, stash: list[str]) -> str:
    def put(m: re.Match) -> str:
        raw = stash[int(m.group(1))]
        if raw.startswith("`"):
            body = raw[1:-1].replace("\\", r"\textbackslash{}")
            for ch, esc in _ESCAPES.items():
                body = body.replace(ch, esc)
            body = body.replace("{", r"\{").replace("}", r"\}")
            return r"\texttt{\small " + body + "}"
        return raw  # maths passes through verbatim

    return re.sub("\0(\\d+)\0", put, text)


def inline(text: str) -> str:
    text, stash = _protect(text)
    for ch, esc in _ESCAPES.items():
        text = text.replace(ch, esc)
    text = text.replace("---", "\u2014").replace("--", "\u2013")
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<![\w*])\*([^*]+?)\*(?![\w*])", r"\\emph{\1}", text)
    text = re.sub(r"\[(.+?)\]\((.+?)\)", r"\\href{\2}{\1}", text)
    text = text.replace("\u2014", "---").replace("\u2013", "--")
    return _restore(text, stash)


def _table(rows: list[str]) -> str:
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    header, body = cells[0], cells[2:]        # cells[1] is the alignment rule
    n = len(header)
    spec = "@{}" + "l" * n + "@{}"
    out = [r"\begin{center}", r"\footnotesize",
           r"\begin{tabular}{" + spec + "}", r"\toprule",
           " & ".join(inline(c) for c in header) + r" \\", r"\midrule"]
    for row in body:
        row = (row + [""] * n)[:n]
        out.append(" & ".join(inline(c) for c in row) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    return "\n".join(out)


def convert(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    i, list_env = 0, None

    def close_list() -> None:
        nonlocal list_env
        if list_env:
            out.append(rf"\end{{{list_env}}}")
            list_env = None

    while i < len(lines):
        line = lines[i]

        if not line.strip():
            close_list()
            out.append("")
            i += 1
            continue

        if re.fullmatch(r"-{3,}|\*{3,}", line.strip()):
            close_list()
            out.append(r"\medskip\hrule\medskip")
            i += 1
            continue

        m = re.match(r"(#{1,4})\s+(.*)", line)
        if m:
            close_list()
            level = len(m.group(1))
            cmd = {1: "section*", 2: "subsection*", 3: "subsubsection*",
                   4: "paragraph"}[level]
            out.append(rf"\{cmd}{{{inline(m.group(2))}}}")
            i += 1
            continue

        if line.lstrip().startswith("|"):
            close_list()
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i])
                i += 1
            if len(block) < 2:
                raise ValueError(f"malformed table near line {i}")
            out.append(_table(block))
            continue

        if line.lstrip().startswith("> "):
            close_list()
            block = []
            while i < len(lines) and lines[i].lstrip().startswith(">"):
                block.append(lines[i].lstrip()[1:].strip())
                i += 1
            out.append(r"\begin{quote}\small " + inline(" ".join(block)) + r"\end{quote}")
            continue

        def absorb(first: str) -> str:
            """Take a list item or paragraph together with its continuation lines."""
            nonlocal i
            parts = [first]
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if not nxt.strip():
                    break
                if re.match(r"\s*([-*]\s+|\d+\.\s+|#{1,4}\s+|\|)", nxt):
                    break
                if re.fullmatch(r"-{3,}|\*{3,}", nxt.strip()):
                    break
                parts.append(nxt.strip())
                i += 1
            return " ".join(parts)

        m = re.match(r"\s*[-*]\s+(.*)", line)
        if m:
            if list_env != "itemize":
                close_list()
                out.append(r"\begin{itemize}")
                list_env = "itemize"
            out.append(r"\item " + inline(absorb(m.group(1))))
            continue

        m = re.match(r"\s*\d+\.\s+(.*)", line)
        if m:
            if list_env != "enumerate":
                close_list()
                out.append(r"\begin{enumerate}")
                list_env = "enumerate"
            out.append(r"\item " + inline(absorb(m.group(1))))
            continue

        close_list()
        out.append(inline(absorb(line)))

    close_list()
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()

    for src in args.sources:
        path = Path(src)
        tex = path.with_suffix(".tex")
        tex.write_text(PREAMBLE + convert(path.read_text()) + "\n\\end{document}\n")
        print(f"wrote {tex}")
        if args.no_pdf:
            continue
        r = subprocess.run(
            ["tectonic", "-X", "compile", tex.name, "--outdir", "."],
            cwd=tex.parent, capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(r.stderr[-2000:], file=sys.stderr)
            raise SystemExit(f"tectonic failed on {tex}")
        print(f"wrote {tex.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
