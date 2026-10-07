#!/usr/bin/env python3
"""paper/export_docx.py — generate paper/main.docx from paper/main.md.

Produces a clean, fully editable Word document (native headings, tables,
figures, bullet lists) from the verified Markdown export. Word wrapping is
content-faithful single-column; the PDF remains the layout-faithful version.

Usage:  python paper/export_docx.py         # writes paper/main.docx
        python paper/export_docx.py --check  # validate existing main.docx
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MD = HERE / "main.md"
OUT = HERE / "main.docx"
IMG_W = 6.0  # inches; letter width minus margins


def die(msg: str) -> None:
    sys.exit(f"FAIL-CLOSED: export_docx: {msg}")


INLINE = re.compile(r"(\*\*.+?\*\*|\*[^*]+?\*|`[^`]+?`)")


def add_runs(paragraph, text: str) -> None:
    """Add inline-formatted runs (**bold**, *italic*, `code`)."""
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            r = paragraph.add_run(part[2:-2])
            r.bold = True
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            r = paragraph.add_run(part[1:-1])
            r.italic = True
        elif part.startswith("`") and part.endswith("`"):
            r = paragraph.add_run(part[1:-1])
            r.font.name = "Consolas"
        else:
            paragraph.add_run(part.replace("<", "").replace(">", ""))


def parse(md_text: str) -> None:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt

    doc = Document()
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(0.75)
        s.top_margin = s.bottom_margin = Inches(0.8)

    lines = md_text.splitlines()
    if lines and lines[0].startswith("<!--"):
        lines = lines[1:]
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # figures
        m = re.fullmatch(r"!\[(.*)\]\((.*)\)", stripped)
        if m:
            alt, path = m.group(1), m.group(2)
            img = (HERE / path).resolve()
            if not img.exists():
                die(f"image not found: {path}")
            doc.add_picture(str(img), width=Inches(IMG_W))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap = doc.add_paragraph()
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = cap.add_run(alt)
            r.italic = True
            r.font.size = Pt(9)
            i += 1
            continue

        # tables
        if stripped.startswith("|"):
            block = []
            while i < n and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            rows = [[c.strip() for c in r.strip("|").split("|")] for r in block]
            rows = [r for r in rows if not all(set(c) <= set("- :") and c for c in r)]
            if not rows:
                die("empty markdown table")
            tbl = doc.add_table(rows=len(rows), cols=len(rows[0]))
            tbl.style = "Table Grid"
            for ri, row in enumerate(rows):
                for ci, cell in enumerate(row[: len(rows[0])]):
                    p = tbl.rows[ri].cells[ci].paragraphs[0]
                    add_runs(p, cell)
                    if ri == 0:
                        for r in p.runs:
                            r.bold = True
            doc.add_paragraph()
            continue

        # headings
        if stripped.startswith("# ") or stripped.startswith("## ") or stripped.startswith("### "):
            level = len(stripped) - len(stripped.lstrip("#"))
            text = stripped.lstrip("# ").strip()
            if level == 1:
                doc.add_heading(text, level=0)
            else:
                doc.add_heading(text, level=min(level - 1, 4))
            i += 1
            continue

        # list items
        if stripped.startswith("- "):
            while i < n and lines[i].strip().startswith("- "):
                item = lines[i].strip()[2:]
                p = doc.add_paragraph(style="List Bullet")
                add_runs(p, item)
                i += 1
            continue

        # reference entries
        if re.match(r"^\[\d+\] ", stripped):
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.left_indent = Inches(0.3)
            p.paragraph_format.first_line_indent = Inches(-0.3)
            add_runs(p, stripped)
            i += 1
            continue

        # wrapped paragraph: join until blank line / structural line
        buf = [stripped]
        i += 1
        while i < n:
            nxt = lines[i].strip()
            if (not nxt or nxt.startswith("#") or nxt.startswith("|")
                    or nxt.startswith("![") or nxt.startswith("- ")
                    or re.match(r"^\[\d+\] ", nxt)):
                break
            buf.append(nxt)
            i += 1
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(8)
        add_runs(p, " ".join(buf))

    doc.save(str(OUT))


def validate() -> dict:
    if not OUT.exists() or OUT.stat().st_size < 20_000:
        die("main.docx missing or implausibly small")
    from docx import Document

    d = Document(str(OUT))
    parts = [p.text for p in d.paragraphs]
    for tbl in d.tables:
        for row in tbl.rows:
            for cell in row.cells:
                parts.append(cell.text)
    text = "\n".join(parts)
    stats = {
        "paragraphs": len(d.paragraphs),
        "tables": len(d.tables),
        "images": len(d.inline_shapes),
        "chars": len(text),
        "bytes": OUT.stat().st_size,
    }
    if stats["tables"] < 4:
        die(f"expected >=4 tables, found {stats['tables']}")
    if stats["images"] < 5:
        die(f"expected >=5 images, found {stats['images']}")
    for needle in ("Abstract", "Data Availability", "References",
                   "Rural telemedicine platforms", "7.0", "0.732",
                   "What holds"):
        if needle not in text:
            die(f"docx missing expected content: {needle!r}")
    return stats


def main() -> None:
    if not MD.exists():
        die("main.md missing — run `python paper/export_md.py` first")
    if "--check" not in sys.argv:
        parse(MD.read_text(encoding="utf-8"))
    stats = validate()
    print(f"{'validated' if '--check' in sys.argv else 'wrote'} {OUT} ({stats})")


if __name__ == "__main__":
    main()
