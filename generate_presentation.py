"""
Convert presentation_rex_poc.md to a beautifully formatted landscape PDF presentation using ReportLab.
Run from the Queue-Management directory:
    python generate_presentation.py
"""

import os
import re
import sys
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Preformatted,
    Table, TableStyle, PageBreak, Image
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.pdfgen import canvas

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_MD = os.path.join(SCRIPT_DIR, "presentation_rex_poc.md")
OUTPUT_PDF = os.path.join(SCRIPT_DIR, "presentation_rex_poc.pdf")

# ── Custom Canvas for Slide Background and Footer ──────────────────────────────
class SlideCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pages = []

    def showPage(self):
        # Save page properties for the two-pass page numbering
        self.pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        page_count = len(self.pages)
        for page in self.pages:
            self.__dict__.update(page)
            self.draw_slide_decorations(page_count)
            super().showPage()
        super().save()

    def draw_slide_decorations(self, page_count):
        self.saveState()
        width, height = landscape(A4)

        # 1. Top Accent Bar (Blue)
        self.setFillColor(colors.HexColor("#0071e3"))
        self.rect(0, height - 6, width, 6, fill=True, stroke=False)

        # 2. Bottom Gray Divider
        self.setStrokeColor(colors.HexColor("#e5e5e7"))
        self.setLineWidth(0.5)
        self.line(2 * cm, 1.4 * cm, width - 2 * cm, 1.4 * cm)

        # 3. Footer Title
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#1d1d1f"))
        self.drawString(2 * cm, 0.8 * cm, "REX POC — Système Intelligent de Gestion de File d'Attente (IQMS)")
        
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#86868b"))
        self.drawString(11.8 * cm, 0.8 * cm, "|   Synthèse d'Intégration & REX")

        # 4. Slide Number (Page Number / Total Pages)
        slide_str = f"{self._pageNumber} / {page_count}"
        self.drawRightString(width - 2 * cm, 0.8 * cm, slide_str)
        
        self.restoreState()

# ── Styles ─────────────────────────────────────────────────────────────────────
BASE = getSampleStyleSheet()

def _style(name, parent="Normal", **kw):
    return ParagraphStyle(name, parent=BASE[parent], **kw)

# Presentation scaled typography
TITLE = _style("Title", fontSize=26, textColor=colors.HexColor("#0071e3"),
               alignment=TA_CENTER, spaceAfter=8, leading=32,
               fontName="Helvetica-Bold")
SUBTITLE = _style("Subtitle", fontSize=14, textColor=colors.HexColor("#515154"),
                  alignment=TA_CENTER, spaceAfter=4, leading=18)
H1 = _style("H1", "Heading1", fontSize=20, textColor=colors.HexColor("#0071e3"),
            spaceAfter=12, spaceBefore=8, leading=24, fontName="Helvetica-Bold")
H2 = _style("H2", "Heading2", fontSize=15, textColor=colors.HexColor("#1d1d1f"),
            spaceAfter=10, spaceBefore=6, leading=19, fontName="Helvetica-Bold")
BODY = _style("Body", fontSize=11, leading=16, spaceAfter=6, textColor=colors.HexColor("#1d1d1f"))
BULLET = _style("Bullet", fontSize=11, leading=16, spaceAfter=4,
                leftIndent=24, bulletIndent=10, textColor=colors.HexColor("#1d1d1f"))
CODE = _style("Code", fontName="Courier", fontSize=9, leading=13,
              leftIndent=15, rightIndent=15,
              backColor=colors.HexColor("#f5f5f7"),
              borderColor=colors.HexColor("#d2d2d7"),
              borderPadding=8)
TBL_HEADER = _style("TblHdr", fontName="Helvetica-Bold", fontSize=9,
                    textColor=colors.white, alignment=TA_LEFT)
TBL_CELL   = _style("TblCell", fontSize=9, leading=12, alignment=TA_LEFT, textColor=colors.HexColor("#1d1d1f"))
TBL_CODE   = _style("TblCode", fontName="Courier", fontSize=8.5,
                    leading=11, alignment=TA_LEFT, textColor=colors.HexColor("#1d1d1f"))

# ── Character replacements for ReportLab compatibility ────────────────────────
def _esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _clean_diagram(text):
    # Replace unicode box-drawing elements with ascii counterparts for PDF rendering safety
    replacements = {
        "──►": "-->",
        "──>": "-->",
        "─►": "->",
        "►": ">",
        "│": "|",
        "▼": "v",
        "▲": "^",
        "┌": "+",
        "┐": "+",
        "└": "+",
        "┘": "+",
        "├": "+",
        "┤": "+",
        "┬": "+",
        "┴": "+",
        "┼": "+",
        "─": "-",
        "═": "=",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def _inline(text):
    text = _esc(text)
    # **bold**
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # *italic*
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    # `code`
    text = re.sub(r"`([^`]+)`", r'<font name="Courier" size="10">\1</font>', text)
    # [text](link) → just text
    text = re.sub(r"\[([^\]]+)\]\([^\)]*\)", r"\1", text)
    return text

# ── Main Parser ────────────────────────────────────────────────────────────────
def parse_markdown_presentation(md_text):
    flowables = []
    
    # Split slides by horizontal dividers
    slides_raw = re.split(r'\n---\n|\r\n---\r\n', md_text)
    
    is_first_slide = True
    
    for slide_idx, slide_content in enumerate(slides_raw):
        slide_content = slide_content.strip()
        if not slide_content:
            continue
            
        # Skip Marp Frontmatter block if detected
        if "marp:" in slide_content and "theme:" in slide_content:
            continue
            
        if not is_first_slide:
            flowables.append(PageBreak())
        else:
            is_first_slide = False
            
        lines = slide_content.splitlines()
        
        # Check if it is the cover slide
        if slide_idx == 0 or (len(lines) > 0 and lines[0].startswith("# ") and any("Restitution" in l or "Rapport" in l for l in lines)):
            # Draw Cover slide
            flowables.append(Spacer(1, 3.5 * cm))
            for line in lines:
                line_str = line.strip()
                if not line_str:
                    continue
                if line_str.startswith("# "):
                    flowables.append(Paragraph(_inline(line_str.lstrip("# ")), TITLE))
                    flowables.append(Spacer(1, 0.4 * cm))
                elif line_str.startswith("## "):
                    flowables.append(Paragraph(_inline(line_str.lstrip("## ")), SUBTITLE))
                    flowables.append(Spacer(1, 0.2 * cm))
                else:
                    flowables.append(Paragraph(_inline(line_str.strip("* ")), SUBTITLE))
            continue

        # Normal slide processing
        in_code = False
        in_table = False
        code_lines = []
        table_rows = []
        table_has_sep = False
        
        def flush_code():
            nonlocal code_lines
            if code_lines:
                text = "\n".join(code_lines)
                text = _clean_diagram(text)
                flowables.append(Preformatted(text, CODE))
                flowables.append(Spacer(1, 0.2 * cm))
                code_lines = []
                
        def flush_table():
            nonlocal table_rows, table_has_sep
            if not table_rows:
                return
            col_count = max(len(r) for r in table_rows)
            # A4 Landscape width is 841.89 pt. Printable width ~ width - 4cm
            width, _ = landscape(A4)
            col_w = (width - 4.5 * cm) / col_count
            
            data = []
            for ri, row in enumerate(table_rows):
                while len(row) < col_count:
                    row.append("")
                cells = []
                for ci, cell in enumerate(row):
                    s = TBL_HEADER if ri == 0 else (TBL_CODE if cell.startswith("`") else TBL_CELL)
                    cells.append(Paragraph(_inline(cell.strip("`")), s))
                data.append(cells)
                
            t = Table(data, colWidths=[col_w] * col_count)
            ts = TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0071e3")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f5f5f7"), colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d2d2d7")),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ])
            t.setStyle(ts)
            flowables.append(Spacer(1, 0.2 * cm))
            flowables.append(t)
            flowables.append(Spacer(1, 0.3 * cm))
            table_rows = []
            table_has_sep = False

        # Add initial slide title spacing
        flowables.append(Spacer(1, 0.5 * cm))

        i = 0
        while i < len(lines):
            line = lines[i]
            line_stripped = line.strip()
            
            # Fenced code blocks
            if line_stripped.startswith("```"):
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
                
            # Tables
            if line_stripped.startswith("|"):
                if re.match(r"^[\|\s\-:]+$", line_stripped):
                    table_has_sep = True
                    i += 1
                    continue
                cells = [c.strip() for c in line_stripped.strip("|").split("|")]
                table_rows.append(cells)
                i += 1
                continue
            else:
                if table_rows:
                    flush_table()
                    
            # Headings
            m = re.match(r"^(#{1,4})\s+(.*)", line)
            if m:
                level = len(m.group(1))
                text = _inline(m.group(2))
                style = H1 if level == 2 else (H2 if level == 3 else H2)
                flowables.append(Paragraph(text, style))
                i += 1
                continue
                
            # Bullets
            if line_stripped.startswith("* ") or line_stripped.startswith("- "):
                text = _inline(line_stripped[2:])
                flowables.append(Paragraph(f"&bull; {text}", BULLET))
                i += 1
                continue
                
            # Images
            m_img = re.match(r"^!\[(.*?)\]\((.*?)\)", line_stripped)
            if m_img:
                img_path = m_img.group(2)
                # Skip absolute paths starting with /C:/ or /Users/ for local generator since they are for the brain artifact,
                # but handle local filename
                if img_path.startswith("/"):
                    img_name = os.path.basename(img_path)
                else:
                    img_name = img_path
                full_path = os.path.join(SCRIPT_DIR, img_name)
                if os.path.exists(full_path):
                    flowables.append(Spacer(1, 0.2 * cm))
                    flowables.append(Image(full_path, width=280, height=210))
                    flowables.append(Spacer(1, 0.2 * cm))
                else:
                    print(f"Warning: Image file '{full_path}' not found.")
                i += 1
                continue

            # Paragraph text
            if line_stripped:
                text = _inline(line_stripped)
                flowables.append(Paragraph(text, BODY))
                
            i += 1
            
        # Flush any remaining table/code block
        if table_rows:
            flush_table()
        if code_lines:
            flush_code()
            
    return flowables

# ── Main Runner ────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(INPUT_MD):
        print(f"Error: Input Markdown file '{INPUT_MD}' not found.")
        sys.exit(1)
        
    print(f"Reading markdown presentation from '{INPUT_MD}'...")
    with open(INPUT_MD, "r", encoding="utf-8") as f:
        md_text = f.read()
        
    flowables = parse_markdown_presentation(md_text)
    
    print(f"Generating landscape PDF to '{OUTPUT_PDF}'...")
    width, height = landscape(A4)
    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=(width, height),
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=2 * cm
    )
    
    doc.build(flowables, canvasmaker=SlideCanvas)
    print("Presentation PDF successfully generated.")

if __name__ == "__main__":
    main()
