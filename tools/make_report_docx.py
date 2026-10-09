"""docs/report.md → docs/report.docx（Word 交付版）。

md 是 source of truth；本腳本把報告用到的 md 子集轉成 Word 原生格式：
標題層級用內建 Heading 樣式（可導覽/可生成目錄）、表格用真表格（Table Grid + 表頭底紋）、
程式碼塊（架構圖）用 Consolas + 灰底、行內 **粗體**/*斜體*/*斜體*//`程式碼`。

Usage: .venv/Scripts/python.exe tools/make_report_docx.py [src.md] [out.docx]
"""
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]


def _arg(n: int, default: Path) -> Path:
    if len(sys.argv) > n:
        p = Path(sys.argv[n])
        return p if p.is_absolute() else ROOT / p
    return default


SRC = _arg(1, ROOT / "docs" / "report.md")
OUT = _arg(2, ROOT / "docs" / "report.docx")

MONO = "Consolas"
BODY = "Calibri"
ACCENT = RGBColor(0x1F, 0x4E, 0x79)  # 深藍，標題用


def shade(par_or_cell, fill: str) -> None:
    """段落/儲存格底紋。"""
    el = par_or_cell._p if hasattr(par_or_cell, "_p") else par_or_cell._tc
    pr = el.get_or_add_pPr() if hasattr(par_or_cell, "_p") else el.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), fill)
    pr.append(shd)


def add_runs(par, text: str, base_size: Pt | None = None) -> None:
    """行內格式：先剝連結，再切 `code`，再切 **粗**/*斜*。"""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    for chunk in re.split(r"(`[^`]+`)", text):
        if chunk.startswith("`") and chunk.endswith("`") and len(chunk) > 2:
            r = par.add_run(chunk[1:-1])
            r.font.name = MONO
            r.font.size = (base_size or Pt(10.5)) - Pt(1.5)
            shade(par, "F2F2F2")
        else:
            for tok in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", chunk):
                if not tok:
                    continue
                if tok.startswith("**") and tok.endswith("**"):
                    r = par.add_run(tok[2:-2]); r.bold = True
                elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
                    r = par.add_run(tok[1:-1]); r.italic = True
                else:
                    r = par.add_run(tok)
                if base_size is not None:
                    r.font.size = base_size


def add_code_block(doc, lines: list[str]) -> None:
    for i, line in enumerate(lines):
        par = doc.add_paragraph()
        pf = par.paragraph_format
        pf.space_before = Pt(6) if i == 0 else Pt(0)
        pf.space_after = Pt(6) if i == len(lines) - 1 else Pt(0)
        pf.left_indent = Cm(0.3)
        run = par.add_run(line if line else " ")
        run.font.name = MONO
        run.font.size = Pt(7.5)
        shade(par, "F5F5F5")


def add_table(doc, header: list[str], rows: list[list[str]]) -> None:
    t = doc.add_table(rows=len(rows) + 1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, cell_text in enumerate(header):
        cell = t.rows[0].cells[j]
        par = cell.paragraphs[0]
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_runs(par, f"**{cell_text}**", base_size=Pt(9.5))
        shade(cell, "DEEAF6")  # 淡藍表頭
    for i, row in enumerate(rows, start=1):
        for j, cell_text in enumerate(row):
            par = t.rows[i].cells[j].paragraphs[0]
            add_runs(par, cell_text, base_size=Pt(9.5))
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def main() -> None:
    lines = SRC.read_text(encoding="utf-8").splitlines()

    doc = Document()
    # 頁面：A4 + 適中邊界
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.2)
    sec.top_margin = sec.bottom_margin = Cm(2.0)

    # 內文字型與標題色
    normal = doc.styles["Normal"]
    normal.font.name = BODY
    normal.font.size = Pt(10.5)
    for h, size in (("Heading 1", 15), ("Heading 2", 12.5), ("Heading 3", 11.5)):
        st = doc.styles[h]
        st.font.name = BODY
        st.font.size = Pt(size)
        st.font.color.rgb = ACCENT

    i = 0
    title_done = False
    while i < len(lines):
        line = lines[i]

        if line.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            add_code_block(doc, block)

        elif line.startswith("|"):
            tbl = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):  # 跳過對齊列
                    tbl.append(cells)
                i += 1
            if tbl:
                width = max(len(r) for r in tbl)
                add_table(doc, tbl[0], [r + [""] * (width - len(r)) for r in tbl[1:]])
            continue

        elif line.startswith("### "):
            doc.add_heading(line[4:].strip(), level=2)
        elif line.startswith("## "):
            doc.add_heading(line[3:].strip(), level=1)
        elif line.startswith("# "):
            h = doc.add_heading(line[2:].strip(), level=0)
            h.alignment = WD_ALIGN_PARAGRAPH.CENTER
            title_done = True
        elif re.match(r"^\d+\.\s", line):
            par = doc.add_paragraph(style="List Number")
            add_runs(par, re.sub(r"^\d+\.\s+", "", line))
        elif line.startswith("- "):
            par = doc.add_paragraph(style="List Bullet")
            add_runs(par, line[2:])
        elif line.startswith("> "):
            if "Word deliverable" in line:  # provenance 註記只給 md 讀者，不進交付版
                pass
            else:
                par = doc.add_paragraph()
                par.alignment = WD_ALIGN_PARAGRAPH.CENTER
                add_runs(par, line[2:])
                for r in par.runs:
                    r.font.size = Pt(9.5)
                    r.font.color.rgb = RGBColor(0x59, 0x59, 0x59)
        elif line.strip() in ("---", ""):
            pass
        else:
            par = doc.add_paragraph()
            # 標題後第一行斜體簡介 → 副標
            if title_done and line.startswith("*") and line.endswith("*"):
                par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_runs(par, line)

        i += 1

    doc.save(OUT)
    print(f"-> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
