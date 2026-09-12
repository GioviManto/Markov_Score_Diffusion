#!/usr/bin/env python3
"""Tie the submitted PDF to the sources and frozen outputs it was built from.

WHY THIS EXISTS. A reproducibility appendix that lists commands is a promise;
a manifest is a check. Without one the submitted PDF cannot be tied to a
repository state at all: it could have been built from edited working-tree
files and would look internally consistent, because every consistency check
the document runs is internal to it. This records what the artefact actually
came from -- the repository revision, the last commit touching each of the
two subtrees and whether that subtree was dirty at build time, a digest over
the generated inputs, and the digests of the figures and of the PDF itself --
so a reader with the repository can verify rather than trust.

WHAT IT DOES NOT CLAIM. A manifest fixes the SOURCE side only. It says which
code and which frozen outputs produced this document; it does not certify the
provenance of the frozen outputs themselves, which is Appendix A's business
and is disclosed there route by route, including the runs whose recorded
revision fields are empty.

    python3 tools/make_build_manifest.py

Writes sections/build-manifest.tex (generated; do not hand-edit). Run it after
the final build, then rebuild once so the manifest that ships describes the
PDF that ships -- the PDF digest is necessarily of the previous build, and the
file says so rather than pretending otherwise.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

THESIS = Path(__file__).resolve().parents[1]
# The research package sits beside the documents in the working tree and
# inside the thesis in the published repository. An absolute guess at one of
# them crashed in the other, which made the manifest -- the one artefact whose
# job is to be re-runnable by a reader -- the one script a reader could not run.
RESEARCH = next(
    (c for c in (THESIS / "research",
                 THESIS.parents[1] / "research" / "nongaussian-bp")
     if (c / "outputs").is_dir()),
    THESIS / "research")
OUT = THESIS / "sections" / "build-manifest.tex"


def git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), *args],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return ""


def head(repo: Path) -> str:
    sha = git(repo, "rev-parse", "HEAD")
    return sha[:12] if sha else "unrecorded"


def subtree(repo: Path, path: Path) -> tuple[str, bool]:
    """(last commit touching this subtree, is the subtree dirty).

    Path-limited on both counts. An earlier version ran `rev-parse HEAD` and
    an unlimited `status` for each of two directories that live in ONE
    repository: the two rows could never disagree, and the dirty flag tripped
    on edits anywhere in the tree -- including documents not on either code
    path. Per-subtree is the fact the manifest is supposed to record.
    """
    try:
        rel = str(path.relative_to(repo)) if path != repo else "."
    except ValueError:                       # outside this repository entirely
        return "outside repository", False
    sha = git(repo, "log", "-1", "--format=%H", "--", rel)
    dirty = bool(git(repo, "status", "--porcelain", "--", rel))
    return (sha[:12] if sha else "unrecorded"), dirty


def digest_of(paths) -> str:
    """One digest over a set of files, order-independent and path-aware."""
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()[:16]


def main() -> None:
    # One repository holds both trees, so the repository revision is one fact;
    # what differs, and is worth recording, is the last commit that touched
    # each subtree and whether that subtree has uncommitted edits.
    repo = THESIS
    while repo != repo.parent and not (repo / ".git").exists():
        repo = repo.parent
    repo_sha = head(repo)
    # In the published repository the thesis IS the repository root, so a
    # "thesis subtree" row would restate the repository revision. Report the
    # document subtrees that actually exist as distinct paths, and say which
    # case we are in rather than printing one fact three times.
    doc = THESIS / "chapters"
    thesis_sha, thesis_dirty = subtree(repo, doc if doc.is_dir() else THESIS)
    research_sha, research_dirty = subtree(repo, RESEARCH)
    single = (repo.resolve() == THESIS.resolve())

    generated = sorted((THESIS / "sections").glob("*.tex"))
    # The manifest cannot include itself: writing it changes the digest.
    generated = [p for p in generated if p.name != OUT.name]
    figures = sorted((THESIS / "figures").glob("*"))

    pdf = THESIS / "main.pdf"
    pdf_digest = (hashlib.sha256(pdf.read_bytes()).hexdigest()[:16]
                  if pdf.exists() else "not built")

    tect = subprocess.run(["tectonic", "--version"], capture_output=True,
                          text=True).stdout.strip() or "unrecorded"

    def flag(d):
        return r"\emph{dirty}" if d else "clean"

    body = f"""%% GENERATED by thesis/tools/make_build_manifest.py -- do not hand-edit.
\\newcommand{{\\bmreposha}}{{\\texttt{{{repo_sha}}}}}
\\newcommand{{\\bmlayout}}{{{"the published repository, whose root is the thesis" if single else "the working tree, where the documents sit beside the research package"}}}
\\newcommand{{\\bmthesissha}}{{\\texttt{{{thesis_sha}}}}}
\\newcommand{{\\bmthesisstate}}{{{flag(thesis_dirty)}}}
\\newcommand{{\\bmresearchsha}}{{\\texttt{{{research_sha}}}}}
\\newcommand{{\\bmresearchstate}}{{{flag(research_dirty)}}}
\\newcommand{{\\bmgenerated}}{{\\texttt{{{digest_of(generated)}}}}}
\\newcommand{{\\bmgeneratedcount}}{{{len(generated)}}}
\\newcommand{{\\bmfigures}}{{\\texttt{{{digest_of(figures)}}}}}
\\newcommand{{\\bmfigurecount}}{{{len(figures)}}}
\\newcommand{{\\bmpdf}}{{\\texttt{{{pdf_digest}}}}}
\\newcommand{{\\bmtectonic}}{{{tect}}}
"""
    OUT.write_text(body)
    print(f"repository {repo_sha}")
    print(f"thesis     {thesis_sha} last touched, {'dirty' if thesis_dirty else 'clean'}")
    print(f"research   {research_sha} last touched, {'dirty' if research_dirty else 'clean'}")
    print(f"generated {digest_of(generated)} over {len(generated)} files")
    print(f"figures   {digest_of(figures)} over {len(figures)} files")
    print(f"pdf       {pdf_digest} (of the PREVIOUS build)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
