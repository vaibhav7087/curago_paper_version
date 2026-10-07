#!/usr/bin/env python3
"""paper/export_md.py — generate paper/main.md from main.tex.

Deterministic Markdown rendering of the IEEE draft so the paper is readable
as plain text (review, diffing, AI tooling). Rules:

  * numbers come from paper_numbers.json via make_numbers (never hand-typed)
  * figure/table numbers follow declaration order; section refs resolve to
    their IEEE numbers
  * citations resolve to their main.bbl order (same as the PDF's [n])
  * fail-closed: any LaTeX command left unhandled aborts with exit 1

Usage:  python paper/export_md.py        # writes paper/main.md
        python paper/export_md.py --check  # verify main.md is current
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN_TEX = HERE / "main.tex"
BBL = HERE / "main.bbl"
OUT = HERE / "main.md"
FIG_DIR = Path("..") / "ds" / "models" / "paper"

sys.path.insert(0, str(HERE))
from make_numbers import load_numbers, resolve_all  # noqa: E402

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]

GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "lambda": "λ", "mu": "μ", "pi": "π", "rho": "ρ", "sigma": "σ",
    "tau": "τ", "phi": "φ", "omega": "ω",
    "times": "×", "leq": "≤", "geq": "≥", "neq": "≠", "pm": "±",
    "approx": "≈", "rightarrow": "→", "infty": "∞",
    "lceil": "⌈", "rceil": "⌉", "lfloor": "⌊", "rfloor": "⌋",
    "cdot": "·", "sum": "Σ", "in": "∈",
}


def die(msg: str) -> None:
    sys.exit(f"FAIL-CLOSED: export_md: {msg}")


def strip_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%[^\n]*", "", text)


def inline_inputs(text: str) -> str:
    r"""Inline \input{...} files; numbers.tex is not needed (macros substituted)."""

    def repl(m: re.Match) -> str:
        name = m.group(1)
        if name == "numbers":
            return ""
        path = HERE / name
        if not path.suffix:
            path = path.with_suffix(".tex")
        if not path.exists():
            path = HERE.parent / name
            if not path.suffix:
                path = path.with_suffix(".tex")
        if not path.exists():
            die(f"missing input file: {name}")
        return path.read_text(encoding="utf-8")

    return re.sub(r"\\input\{([^}]+)\}", repl, text)


def sub_macros(text: str) -> str:
    macros = resolve_all(load_numbers())
    for name in sorted(macros, key=len, reverse=True):
        value = macros[name][1]
        text = re.sub(
            rf"\\{re.escape(name)}(?:\{{\}})?(?![A-Za-z])", lambda _m: value, text
        )
    return text


def bbl_cite_map(bbl_text: str) -> dict[str, int]:
    keys = re.findall(r"\\bibitem\{([^}]+)\}", bbl_text)
    if not keys:
        die("no \\bibitem entries in main.bbl (run build.bat first)")
    return {k: i + 1 for i, k in enumerate(keys)}


def bbl_references(bbl_text: str) -> list[str]:
    parts = re.split(r"\\bibitem\{[^}]+\}", bbl_text)[1:]
    refs = []
    for body in parts:
        body = re.sub(r"\\end\{thebibliography\}.*", "", body, flags=re.S)
        body = re.sub(r"\\hskip.*?\\relax", " ", body, flags=re.S)
        body = re.sub(r"\\BIBentry\w+", " ", body)
        body = re.sub(r"\\BIBdecl", " ", body)
        body = re.sub(r"\\url\{([^}]+)\}", r"\1", body)
        body = re.sub(
            r"\\emph\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"*\1*", body
        )
        body = body.replace("``", '"').replace("''", '"')
        body = re.sub(r"\{([^{}]*)\}", r"\1", body)
        body = body.replace(r"\&", "&").replace("~", " ")
        body = body.replace("--", "–")
        for src, dst in {"\\'c": "ć", "\\'e": "é", "\\'E": "É", "\\'o": "ó",
                         "\\'a": "á", "\\'u": "ú"}.items():
            body = body.replace(src, dst)
        body = re.sub(r"\\'([A-Za-z])", r"\1", body)
        body = re.sub(r"\s+", " ", body).strip()
        refs.append(body)
    return refs


def collect_labels(text: str) -> tuple[dict[str, str], dict[str, int], dict[str, int]]:
    """Return (sec label->title, fig label->number, tab label->number)."""
    secs: dict[str, str] = {}
    for m in re.finditer(
        r"\\section\{([^}]+)\}\s*\\label\{(sec:[^}]+)\}", text
    ):
        secs[m.group(2)] = m.group(1)
    figs: dict[str, int] = {}
    tabs: dict[str, int] = {}
    f = t = 0
    for m in re.finditer(
        r"\\begin\{(figure\*?|table\*?)\}(.*?)\\end\{\1\}", text, flags=re.S
    ):
        lbl = re.search(r"\\label\{([^}]+)\}", m.group(2))
        if not lbl:
            die(f"{m.group(1)} environment without label")
        if m.group(1).startswith("figure"):
            f += 1
            figs[lbl.group(1)] = f
        else:
            t += 1
            tabs[lbl.group(1)] = t
    return secs, figs, tabs


def tables_to_md(text: str) -> str:
    """Convert each table environment into a Markdown block."""

    def repl(m: re.Match) -> str:
        body = m.group(1)
        cap = re.search(r"\\caption\{(.+?)\}\s*\\label\{([^}]+)\}", body, flags=re.S)
        if not cap:
            die("table without caption/label")
        tab = re.search(
            r"\\begin\{tabular\}\{[^}]+\}(.*?)\\end\{tabular\}", body, flags=re.S
        )
        if not tab:
            die("table without tabular")
        raw = tab.group(1)
        header_part, sep, body_part = raw.partition(r"\midrule")
        if not sep:
            die("table without midrule")
        header_part = header_part.replace(r"\toprule", "")
        body_part = body_part.replace(r"\bottomrule", "")
        header_rows = [r.strip() for r in re.split(r"\\\\", header_part) if r.strip()]
        body_rows = [r.strip() for r in re.split(r"\\\\", body_part) if r.strip()]
        if not header_rows:
            die("table header not found")
        cells = [c.strip() for c in header_rows[-1].split("&")]
        md = ["| " + " | ".join(cells) + " |",
              "|" + "|".join([" \x01 "] * len(cells)) + "|"]
        for row in body_rows:
            md.append("| " + " | ".join(c.strip() for c in row.split("&")) + " |")
        note = body.split(r"\end{adjustbox}")[-1]
        note = re.sub(r"\\par\b|\\vspace\{[^}]+\}|\\raggedright|\\footnotesize",
                      " ", note)
        note = re.sub(r"\\textbf\{([^}]+)\}", r"**\1**", note)
        note = re.sub(r"\\[a-zA-Z]+", " ", note)
        note = note.replace("{", "").replace("}", "")
        note = re.sub(r"\s+", " ", note).strip()
        caption = re.sub(r"\s+", " ", cap.group(1)).strip()
        out = [f"**Table ⟨{cap.group(2)}⟩. {caption}**", ""]
        out += md
        if note:
            out += ["", note]
        out += [""]
        return "\n".join(out)

    text = re.sub(
        r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", repl, text, flags=re.S
    )
    return text


def figures_to_md(text: str) -> str:
    def repl(m: re.Match) -> str:
        body = m.group(1)
        cap = re.search(r"\\caption\{(.+?)\}\s*\\label\{(fig:[^}]+)\}", body, flags=re.S)
        inc = re.search(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", body)
        if not cap or not inc:
            die("figure missing caption or includegraphics")
        caption = re.sub(r"\s+", " ", cap.group(1)).strip()
        name = inc.group(1)
        if not name.endswith(".png"):
            name += ".png"
        return f"\n![{caption}]({FIG_DIR.as_posix()}/{name})\n"

    return re.sub(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", repl, text,
                  flags=re.S)


def resolve_refs(text: str, secs: dict[str, str], figs: dict[str, int],
                  tabs: dict[str, int]) -> str:
    def fig_ref(m: re.Match) -> str:
        lbl = "fig:" + m.group(1)
        if lbl not in figs:
            die(f"unresolved figure ref: {lbl}")
        return str(figs[lbl])

    def tab_ref(m: re.Match) -> str:
        lbl = "tab:" + m.group(1)
        if lbl not in tabs:
            die(f"unresolved table ref: {lbl}")
        return ROMAN[tabs[lbl] - 1]

    def sec_ref(m: re.Match) -> str:
        lbl = "sec:" + m.group(1)
        if lbl not in secs:
            die(f"unresolved section ref: {lbl}")
        title = secs[lbl]
        order = list(secs.values())
        return ROMAN[order.index(title)]

    text = re.sub(r"\\ref\{fig:([^}]+)\}", fig_ref, text)
    text = re.sub(r"\\ref\{tab:([^}]+)\}", tab_ref, text)
    text = re.sub(r"\\ref\{sec:([^}]+)\}", sec_ref, text)
    return text


def sub_cites(text: str, cite_map: dict[str, int]) -> str:
    def repl(m: re.Match) -> str:
        keys = [k.strip() for k in m.group(1).split(",")]
        nums = []
        for k in keys:
            if k not in cite_map:
                die(f"cite key not in main.bbl: {k}")
            n = cite_map[k]
            if n not in nums:
                nums.append(n)
        return " [" + ", ".join(str(n) for n in nums) + "]"

    return re.sub(r"\s*~?\s*\\cite\{([^}]+)\}", repl, text)


def structure(text: str) -> str:
    # title + author
    title = re.search(r"\\title\{(.+?)\}\s*\\author\{(.*?)\}\s*\\maketitle",
                      text, flags=re.S)
    if not title:
        die("title/author block not found")
    t_text = re.sub(r"\s+", " ", title.group(1)).strip()
    a_body = title.group(2)
    a_name = re.search(r"\\IEEEauthorblockN\{([^}]+)\}", a_body)
    a_aff = re.search(r"\\IEEEauthorblockA\{(.*)\}", a_body, flags=re.S)
    authors = a_name.group(1).strip() if a_name else "Authors"
    aff = ""
    if a_aff:
        aff = a_aff.group(1)
        aff = aff.replace("\\\\", ", ")
        aff = re.sub(r"\s+", " ", aff)
        aff = aff.replace(r"\{", "{").replace(r"\}", "}").strip()
    head = [f"# {t_text}", "", f"**{authors}** — {aff}", ""]
    # drop preamble through \maketitle + pagestyle lines
    text = text[title.end():]
    text = re.sub(r"\\pagestyle\{plain\}|\\thispagestyle\{plain\}", "", text)

    # abstract / keywords
    text = re.sub(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        lambda m: "## Abstract\n\n" + re.sub(r"\s+", " ", m.group(1)).strip() + "\n",
        text, flags=re.S,
    )
    text = re.sub(
        r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}",
        lambda m: "**Keywords:** " + re.sub(r"\s+", " ", m.group(1)).strip() + "\n",
        text, flags=re.S,
    )

    # sections
    text = re.sub(r"\\section\{([^}]+)\}(?:\s*\\label\{[^}]+\})?",
                  lambda m: f"\n## {m.group(1)}\n", text)
    text = re.sub(r"\\subsection\{([^}]+)\}(?:\s*\\label\{[^}]+\})?",
                  lambda m: f"\n### {m.group(1)}\n", text)
    # run-in paragraphs
    text = re.sub(r"\\paragraph\{([^}]+)\}", lambda m: f"**{m.group(1)}.**", text)
    # lists
    for env in ("itemize", "enumerate"):
        text = text.replace(f"\\begin{{{env}}}", "\n")
        text = text.replace(f"\\end{{{env}}}", "\n")
    text = re.sub(r"\\item\s+", "- ", text)
    # LaTeX straight quotes -> typographic quotes (before texttt adds backticks)
    text = text.replace("``", '"').replace("''", '"')
    # accents
    for src, dst in {"\\'c": "ć", "\\'e": "é", "\\'E": "É", "\\'o": "ó",
                     "\\'a": "á", "\\'u": "ú"}.items():
        text = text.replace(src, dst)
    text = re.sub(r"\\'([A-Za-z])", r"\1", text)  # fallback: keep base letter
    # inline
    text = re.sub(r"\\texttt\{([^}]+)\}", r"`\1`", text)
    text = re.sub(r"\\text\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\\(?:textbf|emph|textit)\{([^}]+)\}", r"*\1*", text)
    text = re.sub(r"\\url\{([^}]+)\}", r"<\1>", text)
    text = text.replace(r"\{", "{").replace(r"\}", "}")
    # bibliography hooks
    text = re.sub(r"\\bibliographystyle\{[^}]+\}", "", text)
    text = re.sub(r"\\bibliography\{[^}]+\}", lambda _m: "\\BIBHERE", text)
    # misc decorations
    text = re.sub(r"\\label\{[^}]+\}", "", text)
    text = re.sub(r"\\maketitle", "", text)
    text = re.sub(r"\\(?:centering|pagestyle|thispagestyle)\b", "", text)
    text = text.replace(r"\%", "%").replace(r"\&", "&").replace(r"\_", "_")
    text = text.replace("---", "—").replace("--", "–")
    text = text.replace("~", " ")
    # math
    for cmd, ch in GREEK.items():
        text = text.replace(f"\\{cmd}", ch)
    text = text.replace("$", "")
    text = text.replace("{}", "")
    return "\n".join(head) + "\n" + text


def cleanup(text: str) -> str:
    text = text.replace("\x01", "---")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def main() -> None:
    if not MAIN_TEX.exists():
        die("main.tex missing")
    if not BBL.exists():
        die("main.bbl missing — run `cmd /c paper\\build.bat` first")

    bbl = BBL.read_text(encoding="utf-8")
    cite_map = bbl_cite_map(bbl)
    refs = bbl_references(bbl)

    text = MAIN_TEX.read_text(encoding="utf-8")
    text = strip_comments(text)
    text = inline_inputs(text)
    text = sub_macros(text)

    secs, figs, tabs = collect_labels(text)
    text = tables_to_md(text)
    for lbl, n in sorted(tabs.items(), key=lambda kv: kv[1]):
        text = text.replace(f"⟨{lbl}⟩", ROMAN[n - 1])

    text = figures_to_md(text)
    text = resolve_refs(text, secs, figs, tabs)
    text = sub_cites(text, cite_map)
    text = structure(text)

    bib_block = "\n## References\n\n" + "\n".join(
        f"[{i + 1}] {r}" for i, r in enumerate(refs)
    ) + "\n"
    text = text.replace("\\BIBHERE", bib_block)
    text = re.sub(r"\\end\{document\}", "", text)
    text = cleanup(text)

    leftover = sorted(set(re.findall(r"\\[A-Za-z]+", text)))
    if leftover:
        die("unhandled LaTeX commands: " + ", ".join(leftover))

    out = (
        "<!-- GENERATED by paper/export_md.py from main.tex + "
        "paper_numbers.json + main.bbl. DO NOT EDIT BY HAND. -->\n\n" + text
    )
    if "--check" in sys.argv:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != out:
            sys.exit("main.md is OUT OF DATE — run `python paper/export_md.py`")
        print("main.md is current")
        return
    OUT.write_text(out, encoding="utf-8", newline="\n")
    print(f"wrote {OUT} ({len(out.split())} words)")


if __name__ == "__main__":
    main()
