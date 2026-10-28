
from pathlib import Path
import re
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "phase14"

source = REPORT_DIR / "final_report.md"
output = REPORT_DIR / "final_report.pdf"

if not source.exists():
    raise FileNotFoundError(f"Markdown report not found: {source}")

text = source.read_text(encoding="utf-8")
if "Phase 15–17: Additional Validation Findings" not in text:
    raise ValueError("Phase 15–17 addendum is missing; PDF not generated.")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="ReportTitle", parent=styles["Title"],
    fontSize=19, leading=23, alignment=TA_CENTER,
    textColor=colors.HexColor("#174a6e"), spaceAfter=12
))
styles.add(ParagraphStyle(
    name="SectionHeading", parent=styles["Heading1"],
    fontSize=13, leading=16,
    textColor=colors.HexColor("#174a6e"),
    spaceBefore=10, spaceAfter=6
))
styles.add(ParagraphStyle(
    name="SubHeading", parent=styles["Heading2"],
    fontSize=10.5, leading=13,
    textColor=colors.HexColor("#28627d"),
    spaceBefore=8, spaceAfter=4
))
styles.add(ParagraphStyle(
    name="ReportBody", parent=styles["BodyText"],
    fontSize=8.5, leading=11, spaceAfter=5
))
styles.add(ParagraphStyle(
    name="ReportBullet", parent=styles["BodyText"],
    fontSize=8.5, leading=11, leftIndent=12,
    firstLineIndent=-8, spaceAfter=4
))
styles.add(ParagraphStyle(
    name="TableCell", parent=styles["BodyText"],
    fontSize=7, leading=8.5
))

def inline_markup(value):
    value = escape(value.strip())
    value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", value)
    return value

story = []
lines = text.splitlines()
i = 0

while i < len(lines):
    line = lines[i].strip()

    if not line:
        i += 1
        continue

    if line.startswith("|"):
        block = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            block.append(lines[i].strip())
            i += 1

        rows = []
        for row in block:
            cells = [c.strip() for c in row.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            rows.append([
                Paragraph(inline_markup(cell), styles["TableCell"])
                for cell in cells
            ])

        if rows:
            col_count = max(len(row) for row in rows)
            for row in rows:
                while len(row) < col_count:
                    row.append(Paragraph("", styles["TableCell"]))

            table = Table(
                rows, repeatRows=1,
                colWidths=[176 * mm / col_count] * col_count
            )
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0),
                 colors.HexColor("#dceaf2")),
                ("TEXTCOLOR", (0, 0), (-1, 0),
                 colors.HexColor("#173f5f")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.extend([table, Spacer(1, 7)])
        continue

    if line == "---":
        story.extend([Spacer(1, 4), HRFlowable(
            width="100%", color=colors.lightgrey
        ), Spacer(1, 4)])
    elif line.startswith("# "):
        story.append(Paragraph(
            inline_markup(line[2:]), styles["ReportTitle"]
        ))
    elif line.startswith("## "):
        story.append(Paragraph(
            inline_markup(line[3:]), styles["SectionHeading"]
        ))
    elif line.startswith("### "):
        story.append(Paragraph(
            inline_markup(line[4:]), styles["SubHeading"]
        ))
    elif line.startswith("- "):
        story.append(Paragraph(
            "&#8226; " + inline_markup(line[2:]),
            styles["ReportBullet"]
        ))
    elif re.match(r"^\d+\.\s+", line):
        item = re.sub(r"^\d+\.\s+", "", line)
        story.append(Paragraph(
            inline_markup(line.split(".", 1)[0] + ".") + " "
            + inline_markup(item), styles["ReportBullet"]
        ))
    else:
        story.append(Paragraph(
            inline_markup(line), styles["ReportBody"]
        ))

    i += 1

doc = SimpleDocTemplate(
    str(output), pagesize=A4,
    rightMargin=17 * mm, leftMargin=17 * mm,
    topMargin=16 * mm, bottomMargin=16 * mm,
    title="FreshBasket Customer Churn Prediction — Final Report"
)
doc.build(story)

print("PDF generated:", output.relative_to(ROOT))
print("Source Markdown preserved:", source.relative_to(ROOT))
print("Includes Phase 15–17 findings.")
