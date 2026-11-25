#!/usr/bin/env python3
"""
src/export_benchmark_pdf_dataset.py
===================================
Converts the 36 FLORES-200 curated multilingual sentence groups (EN, DE, FR)
into a landscape A4 PDF document for distribution, audit, and documentation.
"""

import os
import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.pdfgen import canvas

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILOT_JSONL = os.path.join(WORKSPACE_ROOT, "benchmark", "data", "pilot_dataset.jsonl")
PDF_DATASET_OUTPUT = os.path.join(WORKSPACE_ROOT, "results", "TollMatcher_Benchmark_Datasets.pdf")
DOCS_PDF_DATASET = os.path.join(WORKSPACE_ROOT, "benchmark", "docs", "TollMatcher_Benchmark_Datasets.pdf")


class NumberedLandscapeCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        # Header
        if self._pageNumber > 1:
            self.drawString(15 * mm, 198 * mm, "TollMatcher FLORES-200 Multilingual Benchmark Dataset — Ground-Truth Table")
            self.drawRightString(282 * mm, 198 * mm, "EN ↔ DE ↔ FR (216 Evaluation Directions)")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(15 * mm, 195 * mm, 282 * mm, 195 * mm)

        # Footer
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(15 * mm, 12 * mm, 282 * mm, 12 * mm)
        self.drawString(15 * mm, 8 * mm, "Ostfalia University of Applied Sciences • Department of Computer Science • CC-BY-SA 4.0")
        self.drawRightString(282 * mm, 8 * mm, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


def build_dataset_pdf():
    os.makedirs(os.path.dirname(PDF_DATASET_OUTPUT), exist_ok=True)
    os.makedirs(os.path.dirname(DOCS_PDF_DATASET), exist_ok=True)

    with open(PILOT_JSONL, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    doc = SimpleDocTemplate(
        PDF_DATASET_OUTPUT,
        pagesize=landscape(A4),
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=18 * mm,
        bottomMargin=16 * mm
    )

    styles = getSampleStyleSheet()
    navy = colors.HexColor("#0F172A")
    primary_blue = colors.HexColor("#1D4ED8")
    dark_text = colors.HexColor("#1E293B")
    border_color = colors.HexColor("#CBD5E1")
    alt_bg = colors.HexColor("#F8FAFC")

    title_style = ParagraphStyle(
        'Title_Landscape',
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=navy,
        spaceAfter=3
    )
    subtitle_style = ParagraphStyle(
        'Subtitle_Landscape',
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=primary_blue,
        spaceAfter=8
    )
    th_style = ParagraphStyle(
        'TH_Style',
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=1
    )
    td_style = ParagraphStyle(
        'TD_Style',
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
        textColor=dark_text
    )
    td_bold_center = ParagraphStyle(
        'TD_Bold_Center',
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=10,
        textColor=navy,
        alignment=1
    )
    td_center = ParagraphStyle(
        'TD_Center',
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
        textColor=dark_text,
        alignment=1
    )

    elements = []

    # Title & Metadata
    elements.append(Paragraph("TollMatcher — FLORES-200 Multilingual Benchmark Dataset", title_style))
    elements.append(Paragraph(
        "36 Curated Multi-Way Aligned Sentence Groups in English, German, and French (216 Directional Evaluation Pairs) "
        "• Stratified across Length (Short, Medium, Long) and Difficulty (Easy, Medium, Hard, Stress-Test)",
        subtitle_style
    ))
    elements.append(HRFlowable(width="100%", thickness=1, color=primary_blue, spaceBefore=0, spaceAfter=6))

    # Main Table
    headers = [
        Paragraph("ID", th_style),
        Paragraph("Split", th_style),
        Paragraph("Length", th_style),
        Paragraph("Difficulty", th_style),
        Paragraph("Topic", th_style),
        Paragraph("English Ground Truth (eng_Latn)", th_style),
        Paragraph("German Ground Truth (deu_Latn)", th_style),
        Paragraph("French Ground Truth (fra_Latn)", th_style)
    ]

    table_data = [headers]

    for r in records:
        gid = r["sentence_group_id"]
        split = r["original_split"]
        lcat = r["sentence_length_category"].capitalize()
        dcat = r["linguistic_difficulty_category"].replace("_", " ").capitalize()
        topic = r.get("topic", "")

        row = [
            Paragraph(gid, td_bold_center),
            Paragraph(split, td_center),
            Paragraph(lcat, td_center),
            Paragraph(dcat, td_center),
            Paragraph(topic, td_style),
            Paragraph(r["text_en"], td_style),
            Paragraph(r["text_de"], td_style),
            Paragraph(r["text_fr"], td_style)
        ]
        table_data.append(row)

    col_widths = [15 * mm, 12 * mm, 15 * mm, 18 * mm, 24 * mm, 62 * mm, 62 * mm, 61 * mm]
    t = Table(table_data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), navy),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, alt_bg])
    ]))

    elements.append(t)
    doc.build(elements, canvasmaker=NumberedLandscapeCanvas)
    print(f"[SUCCESS] Generated Dataset PDF at: {PDF_DATASET_OUTPUT}")

    import shutil
    shutil.copyfile(PDF_DATASET_OUTPUT, DOCS_PDF_DATASET)
    print(f"[SUCCESS] Copied Dataset PDF to: {DOCS_PDF_DATASET}")


if __name__ == "__main__":
    build_dataset_pdf()
