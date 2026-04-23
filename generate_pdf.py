"""
Convert all project Markdown files to PDF using ReportLab.
Run from the Queue-Management directory:
    python generate_pdf.py
"""

import re
import os
import sys
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Preformatted,
    Table, TableStyle, HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# All project Markdown files to convert (relative to SCRIPT_DIR)
MD_FILES = [
    "IQMS_FULL_DOC.md",
    "DEVELOPER_DOC.md",
    "SIMULATOR_OVERVIEW.md",
    "BUG_FIXES.md",
    "Head-Detector/README.md",
    (
        "Queue-Management-System-v2-main/"
        "Queue-Management-System-v2-main/README.md"
    ),
]

# ── Styles ─────────────────────────────────────────────────────────────────────
BASE = getSampleStyleSheet()

def _style(name, parent="Normal", **kw):
    s = ParagraphStyle(name, parent=BASE[parent], **kw)
    return s

H1 = _style("H1", "Heading1", fontSize=18, textColor=colors.HexColor("#1a237e"),
            spaceAfter=8, spaceBefore=18, leading=22)
H2 = _style("H2", "Heading2", fontSize=14, textColor=colors.HexColor("#283593"),
            spaceAfter=6, spaceBefore=14, leading=18)
H3 = _style("H3", "Heading3", fontSize=11, textColor=colors.HexColor("#3949ab"),
            spaceAfter=4, spaceBefore=10, leading=14)
H4 = _style("H4", "Heading4", fontSize=10, textColor=colors.HexColor("#546e7a"),
            spaceAfter=3, spaceBefore=8, leading=13, fontName="Helvetica-Bold")
BODY = _style("Body", fontSize=9, leading=13, spaceAfter=4)
BULLET = _style("Bullet", fontSize=9, leading=13, spaceAfter=2,
                leftIndent=14, bulletIndent=4)
CODE = _style("Code", fontName="Courier", fontSize=7.5, leading=11,
              leftIndent=10, rightIndent=10,
              backColor=colors.HexColor("#f5f5f5"),
              borderColor=colors.HexColor("#cccccc"),
              borderPadding=4)
BLOCKQUOTE = _style("Blockquote", fontSize=8.5, leading=12, leftIndent=20,
                    textColor=colors.HexColor("#555555"),
                    borderColor=colors.HexColor("#aaaaaa"),
                    borderPadding=(2, 0, 2, 8))
TBL_HEADER = _style("TblHdr", fontName="Helvetica-Bold", fontSize=8,
                    textColor=colors.white, alignment=TA_LEFT)
TBL_CELL   = _style("TblCell", fontSize=8, leading=11, alignment=TA_LEFT)
TBL_CODE   = _style("TblCode", fontName="Courier", fontSize=7.5,
                    leading=10, alignment=TA_LEFT)
TITLE = _style("Title", fontSize=22, textColor=colors.HexColor("#0d47a1"),
               alignment=TA_CENTER, spaceAfter=6, leading=28,
               fontName="Helvetica-Bold")
SUBTITLE = _style("Subtitle", fontSize=12, textColor=colors.HexColor("#37474f"),
                  alignment=TA_CENTER, spaceAfter=4)

# ── Helper: escape XML chars ───────────────────────────────────────────────────
def _esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

# ── Inline markdown: bold and code spans ──────────────────────────────────────
def _inline(text):
    text = _esc(text)
    # **bold**
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # `code`
    text = re.sub(r"`([^`]+)`", r'<font name="Courier" size="8">\1</font>', text)
    # [text](link) → just text
    text = re.sub(r"\[([^\]]+)\]\([^\)]*\)", r"\1", text)
    return text

# ── Parse markdown to flowables ───────────────────────────────────────────────
def parse_md(md_text):
    flowables = []
    lines = md_text.splitlines()
    i = 0

    # Title block (first 3 non-empty lines)
    non_empty = [l for l in lines[:15] if l.strip()]
    if non_empty:
        flowables.append(Paragraph(_inline(non_empty[0].lstrip("# ")), TITLE))
    if len(non_empty) > 1:
        flowables.append(Paragraph(_inline(non_empty[1].lstrip("# ")), SUBTITLE))
    if len(non_empty) > 2:
        flowables.append(Paragraph(_inline(non_empty[2]), SUBTITLE))
    flowables.append(Spacer(1, 0.4*cm))
    flowables.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a237e")))
    flowables.append(Spacer(1, 0.3*cm))

    # Skip the title block lines
    skip = set()
    for j, l in enumerate(lines[:15]):
        if l.strip() in [non_empty[k].strip() for k in range(min(3, len(non_empty)))]:
            skip.add(j)
    i = max(skip) + 1 if skip else 0

    in_code  = False
    in_table = False
    code_lines = []
    table_rows = []
    table_has_sep = False

    def flush_code():
        nonlocal code_lines
        if code_lines:
            text = "\n".join(code_lines)
            flowables.append(Preformatted(text, CODE))
            flowables.append(Spacer(1, 0.15*cm))
            code_lines = []

    def flush_table():
        nonlocal table_rows, table_has_sep
        if len(table_rows) < 1:
            table_rows = []
            return
        col_count = max(len(r) for r in table_rows)
        col_w = (A4[0] - 4*cm) / col_count

        data = []
        header_row = None
        for ri, row in enumerate(table_rows):
            # pad
            while len(row) < col_count:
                row.append("")
            cells = []
            for ci, cell in enumerate(row):
                s = TBL_HEADER if ri == 0 else (TBL_CODE if cell.startswith("`") else TBL_CELL)
                cells.append(Paragraph(_inline(cell.strip("`")), s))
            data.append(cells)

        t = Table(data, colWidths=[col_w]*col_count, repeatRows=1)
        ts = TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#283593")),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f8f9ff"), colors.white]),
            ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#cccccc")),
            ("FONTSIZE", (0,0), (-1,-1), 8),
            ("TOPPADDING",    (0,0), (-1,-1), 3),
            ("BOTTOMPADDING", (0,0), (-1,-1), 3),
            ("LEFTPADDING",   (0,0), (-1,-1), 5),
            ("RIGHTPADDING",  (0,0), (-1,-1), 5),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
        ])
        t.setStyle(ts)
        flowables.append(t)
        flowables.append(Spacer(1, 0.2*cm))
        table_rows = []
        table_has_sep = False

    while i < len(lines):
        line = lines[i]

        # ── Fenced code block ─────────────────────────────────────────────────
        if line.strip().startswith("```"):
            if not in_code:
                in_code = True
                code_lines = []
            else:
                in_code = False
                flush_code()
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        # ── Table row ─────────────────────────────────────────────────────────
        if line.strip().startswith("|"):
            # separator row (|---|---|)
            if re.match(r"^[\|\s\-:]+$", line):
                table_has_sep = True
                i += 1
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            table_rows.append(cells)
            i += 1
            continue
        else:
            if table_rows:
                flush_table()

        # ── Horizontal rule ───────────────────────────────────────────────────
        if line.strip() in ("---", "***", "___"):
            flowables.append(Spacer(1, 0.1*cm))
            flowables.append(HRFlowable(width="100%", thickness=0.5,
                                        color=colors.HexColor("#aaaaaa")))
            flowables.append(Spacer(1, 0.1*cm))
            i += 1
            continue

        # ── Headings ──────────────────────────────────────────────────────────
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            level = len(m.group(1))
            text = _inline(m.group(2))
            style = {1: H1, 2: H2, 3: H3, 4: H4}.get(level, BODY)
            if level == 1:
                flowables.append(PageBreak())
            flowables.append(Paragraph(text, style))
            i += 1
            continue

        # ── Blockquote ────────────────────────────────────────────────────────
        if line.strip().startswith(">"):
            text = _inline(line.strip().lstrip("> "))
            flowables.append(Paragraph(text, BLOCKQUOTE))
            i += 1
            continue

        # ── Bullet / numbered list ────────────────────────────────────────────
        m = re.match(r"^(\s*)[-*+]\s+(.*)", line)
        if m:
            indent_level = len(m.group(1)) // 2
            bullet = "\u2022" if indent_level == 0 else "\u25e6"
            style = ParagraphStyle(
                f"BulletL{indent_level}", parent=BULLET,
                leftIndent=14 + indent_level*14,
                bulletIndent=4 + indent_level*14,
            )
            flowables.append(Paragraph(f"{bullet} {_inline(m.group(2))}", style))
            i += 1
            continue

        m = re.match(r"^\d+\.\s+(.*)", line)
        if m:
            flowables.append(Paragraph(f"\u2022 {_inline(m.group(1))}", BULLET))
            i += 1
            continue

        # ── Blank line ────────────────────────────────────────────────────────
        if not line.strip():
            flowables.append(Spacer(1, 0.15*cm))
            i += 1
            continue

        # ── Normal paragraph ──────────────────────────────────────────────────
        flowables.append(Paragraph(_inline(line), BODY))
        i += 1

    flush_code()
    flush_table()
    return flowables


# ── Build a single PDF ────────────────────────────────────────────────────────
def build_pdf(md_path, pdf_path):
    with open(md_path, encoding="utf-8") as f:
        md = f.read()

    title = os.path.splitext(os.path.basename(md_path))[0].replace("_", " ")

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2.2*cm,
        title=title,
        author="IQMS",
    )

    story = parse_md(md)
    doc.build(story)
    size_kb = os.path.getsize(pdf_path) // 1024
    print(f"  OK  {os.path.relpath(pdf_path, SCRIPT_DIR):50s}  {size_kb} KB")


# ── Convert all MDs ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Converting Markdown files to PDF...\n")
    errors = []
    for rel_md in MD_FILES:
        md_path  = os.path.join(SCRIPT_DIR, rel_md)
        if not os.path.isfile(md_path):
            print(f"  SKIP {rel_md}  (not found)")
            continue
        # Place PDF next to the MD file
        pdf_path = os.path.splitext(md_path)[0] + ".pdf"
        try:
            build_pdf(md_path, pdf_path)
        except Exception as exc:
            print(f"  FAIL {rel_md}: {exc}")
            errors.append((rel_md, exc))

    print(f"\nDone — {len(MD_FILES) - len(errors)} PDF(s) written.")
    if errors:
        sys.exit(1)
