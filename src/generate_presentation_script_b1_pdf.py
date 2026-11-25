#!/usr/bin/env python3
"""
src/generate_presentation_script_b1_pdf.py
==========================================
Generates a clear, natural B1-Level English presentation script in PDF format
for a 3-to-4 minute university presentation:
"Running the Benchmark on Raspberry Pi: Proposed Algorithm vs. Summersemester 2026 Baseline".
"""

import os
import sys
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.pdfgen import canvas

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(WORKSPACE_ROOT, "results")
PDF_OUTPUT = os.path.join(RESULTS_DIR, "TollMatcher_Presentation_Script_B1.pdf")
DOCS_PDF_OUTPUT = os.path.join(WORKSPACE_ROOT, "benchmark", "docs", "TollMatcher_Presentation_Script_B1.pdf")


class NumberedCanvas(canvas.Canvas):
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

        # Running Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(16 * mm, 285 * mm, "TollMatcher — 3–4 Min Presentation Script (Clear B1 English)")
            self.drawRightString(194 * mm, 285 * mm, "Task #1094 • Yashkumar Chothani")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(16 * mm, 282 * mm, 194 * mm, 282 * mm)

        # Running Footer (all pages)
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(16 * mm, 14 * mm, 194 * mm, 14 * mm)
        self.drawString(16 * mm, 10 * mm, "Ostfalia University of Applied Sciences • Department of Computer Science • Task #1094")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(194 * mm, 10 * mm, page_text)
        self.restoreState()


def build_b1_script_pdf():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(DOCS_PDF_OUTPUT), exist_ok=True)

    doc = SimpleDocTemplate(
        PDF_OUTPUT,
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=18 * mm,
        bottomMargin=16 * mm
    )

    styles = getSampleStyleSheet()

    # Colors
    navy = colors.HexColor("#0F172A")
    primary_blue = colors.HexColor("#1D4ED8")
    emerald = colors.HexColor("#047857")
    crimson = colors.HexColor("#B91C1C")
    dark_text = colors.HexColor("#1E293B")
    muted_text = colors.HexColor("#475569")
    bg_light = colors.HexColor("#F8FAFC")
    border_color = colors.HexColor("#E2E8F0")
    timing_badge_bg = colors.HexColor("#DBEAFE")
    action_cue_bg = colors.HexColor("#FEF3C7")

    # Typography
    title_style = ParagraphStyle(
        'DocTitle',
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=navy,
        spaceAfter=3
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=primary_blue,
        spaceAfter=10
    )
    section_h1 = ParagraphStyle(
        'SectionH1',
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=15,
        textColor=navy,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    timing_pill = ParagraphStyle(
        'TimingPill',
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#1E40AF"),
        alignment=1
    )
    action_cue = ParagraphStyle(
        'ActionCue',
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#92400E")
    )
    speech_text = ParagraphStyle(
        'SpeechText',
        fontName='Helvetica',
        fontSize=9.2,
        leading=14,
        textColor=dark_text,
        spaceAfter=4
    )
    speech_emphasis = ParagraphStyle(
        'SpeechEmphasis',
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=13.5,
        textColor=navy
    )
    qa_question = ParagraphStyle(
        'QAQuestion',
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11.5,
        textColor=primary_blue
    )
    qa_answer = ParagraphStyle(
        'QAAnswer',
        fontName='Helvetica',
        fontSize=8.2,
        leading=11.5,
        textColor=dark_text
    )

    elements = []

    # Title Card
    elements.append(Paragraph("TollMatcher — Presentation Script (B1 English)", title_style))
    elements.append(Paragraph("Clear, simple 3.5-minute spoken script: Comparing the New Algorithm with the Summer 2026 Baseline", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=primary_blue, spaceBefore=0, spaceAfter=6))

    meta_table_data = [
        [
            Paragraph("<b>Audience:</b> Prof. Dr. Lehmann, Michael Thane, Team", speech_text),
            Paragraph("<b>Target Speaking Time:</b> ~3.5 minutes (~420 spoken words)", speech_text)
        ],
        [
            Paragraph("<b>Device:</b> Raspberry Pi 5 (Offline Speech Translation)", speech_text),
            Paragraph("<b>Language Level:</b> B1 (Simple, clear, natural English)", speech_text)
        ]
    ]
    meta_table = Table(meta_table_data, colWidths=[90 * mm, 90 * mm])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg_light),
        ('BOX', (0, 0), (-1, -1), 0.5, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 6))

    def make_section_block(time_str, slide_str, action_str, spoken_paragraphs):
        header_data = [
            [
                Paragraph(f"<b>TIME: {time_str}</b>", timing_pill),
                Paragraph(f"<b>Slide / Screen:</b> {slide_str}", speech_emphasis),
                Paragraph(f"<b>Tip:</b> {action_str}", action_cue)
            ]
        ]
        hdr_table = Table(header_data, colWidths=[28 * mm, 74 * mm, 78 * mm])
        hdr_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, 0), timing_badge_bg),
            ('BACKGROUND', (1, 0), (1, 0), bg_light),
            ('BACKGROUND', (2, 0), (2, 0), action_cue_bg),
            ('BOX', (0, 0), (-1, -1), 0.5, border_color),
            ('TOPPADDING', (0, 0), (-1, -1), 2.5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        block_elements = [hdr_table, Spacer(1, 4)]
        for p in spoken_paragraphs:
            block_elements.append(Paragraph(p, speech_text))
        block_elements.append(Spacer(1, 6))
        return KeepTogether(block_elements)

    # -------------------------------------------------------------
    # Minute 1: 0:00 - 0:45
    # -------------------------------------------------------------
    elements.append(Paragraph("Part 1: Introduction & The Big Problem (0:00 – 0:45)", section_h1))
    p1 = (
        "“Hello everyone. Today I will present our test results for <b>TollMatcher</b> on the <b>Raspberry Pi 5</b>. "
        "Our goal was to compare the <b>old algorithm from Summer 2026</b> with our <b>new proposed algorithm</b>.”"
    )
    p2 = (
        "“In real life, when two people talk, they often stop speaking for a second to think. "
        "These short pauses usually take about <b>700 to 900 milliseconds</b>. "
        "For a translation device, this is very difficult: "
        "if the device stops recording too early, it cuts the sentence in half. "
        "If it waits too long, the user has to wait uncomfortably for the translation.”"
    )
    elements.append(make_section_block("0:00 – 0:45", "Slide 1: Title & Natural Thinking Pauses", "Speak slowly, smile, look at Prof. Lehmann and Michael", [p1, p2]))

    # -------------------------------------------------------------
    # Minute 2: 0:45 - 1:45
    # -------------------------------------------------------------
    elements.append(Paragraph("Part 2: Old Method vs. New Method (0:45 – 1:45)", section_h1))
    p3 = (
        "“The <b>old algorithm from Summer 2026</b> — called <b>Variant B</b> — had a simple rule: "
        "if it hears silence for <b>600 milliseconds</b>, it cuts the recording immediately. "
        "Because 600 ms is shorter than a normal thinking pause, it cut sentences in the middle. "
        "Also, it waited until the person finished talking before doing anything, which made it slow.”"
    )
    p4 = (
        "“Our <b>new proposed algorithm</b> — <b>Variant C</b> — fixes this with two simple ideas:<br/>"
        "• <b>First, Smart Pauses:</b> If the sentence is not finished yet, the device waits up to <b>1,000 milliseconds</b> so the speaker has time to breathe and think.<br/>"
        "• <b>Second, Fast Streaming:</b> The device transcribes audio every 100 milliseconds. When two checks agree on the words, it starts translating immediately without waiting for the whole sentence to finish.”"
    )
    elements.append(make_section_block("0:45 – 1:45", "Slide 2: Old 600ms Rule vs. New Smart Pauses", "Point to the pause diagram on the slide", [p3, p4]))

    # -------------------------------------------------------------
    # Minute 3: 1:45 - 2:45
    # -------------------------------------------------------------
    elements.append(Paragraph("Part 3: Benchmark Results on Raspberry Pi 5 (1:45 – 2:45)", section_h1))
    p5 = (
        "“We tested both algorithms directly on the <b>Raspberry Pi 5</b> with 120 test runs using official benchmark sentences "
        "in <b>English, German, and French</b>. Here are the clear results:<br/><br/>"
        "• <b>False Cuts:</b> The old algorithm cut sentences in half <b>24 times</b>. The new algorithm had <font color='#047857'><b>zero false cuts (0)</b></font>. It never interrupted the speaker.<br/>"
        "• <b>First Audio Response:</b> With the old algorithm, the user had to wait <b>3.7 seconds</b> to hear the first translated word. With our new algorithm, the audio starts playing in only <font color='#047857'><b>0.36 seconds (360 ms)</b></font>. That is <b>10 times faster</b>!<br/>"
        "• <b>Translation Quality:</b> Because sentences are complete, translation score improved from <b>91.7 to 100.0 BLEU</b>.<br/>"
        "• <b>Speed on the Pi:</b> The system runs <b>9 times faster than real speech</b> (RTF of 0.108), so the Raspberry Pi handles it easily without overheating.”"
    )
    elements.append(make_section_block("1:45 – 2:45", "Slide 3: Results Comparison Table", "Emphasize numbers: 24 cuts → 0 cuts, 3.7s → 0.36s", [p5]))

    # -------------------------------------------------------------
    # Minute 4: 2:45 - 3:30
    # -------------------------------------------------------------
    elements.append(Paragraph("Part 4: Easy to Run & Conclusion (2:45 – 3:30)", section_h1))
    p6 = (
        "“Anyone can run this benchmark on the Raspberry Pi with one simple terminal command:<br/>"
        "<code>python evaluate_variants.py --compare</code>.<br/>"
        "This runs both algorithms and prints the full comparison table automatically.<br/>"
        "We also created a <b>Web Lab tester</b> where you can test your own voice through the browser.”"
    )
    p7 = (
        "“To conclude: our new algorithm stops sentences from being cut in half, starts playing audio 10 times faster, "
        "and runs perfectly on the Raspberry Pi 5.<br/>"
        "Thank you very much. I am happy to answer any questions.”"
    )
    elements.append(make_section_block("2:45 – 3:30", "Slide 4: Terminal Command & Web Demo", "Smile, thank the audience, and wait for questions", [p6, p7]))

    # -------------------------------------------------------------
    # Simple Q&A in B1 English
    # -------------------------------------------------------------
    elements.append(Spacer(1, 4))
    elements.append(Paragraph("Helpful Q&A Answers (B1 Level English)", section_h1))

    qa_data = [
        [
            Paragraph("<b>Question: How does the new algorithm know the sentence is not finished?</b>", qa_question),
            Paragraph("<b>Simple Answer:</b> While the person speaks, Whisper transcribes in the background. If the words end in the middle of a sentence (for example after words like 'because', 'and', or before an object), the system gives the speaker up to 1 second of silence before stopping.", qa_answer)
        ],
        [
            Paragraph("<b>Question: Why didn't you just make the pause 1 second for everyone?</b>", qa_question),
            Paragraph("<b>Simple Answer:</b> Because if someone says a short phrase like 'Thank you', they don't want to wait 1 whole second in silence. The new algorithm is fast (400 ms) when you finish, and only waits longer (1000 ms) when you pause to think.", qa_answer)
        ],
        [
            Paragraph("<b>Question: Does the Raspberry Pi 5 get too slow or hot?</b>", qa_question),
            Paragraph("<b>Simple Answer:</b> No. We use <code>faster-whisper</code> with 8-bit quantization. It uses only about 19 megabytes of RAM and takes less than 1 second to process a 5-second sentence.", qa_answer)
        ]
    ]

    qa_table = Table(qa_data, colWidths=[65 * mm, 115 * mm])
    qa_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#EFF6FF")),
        ('BACKGROUND', (1, 0), (1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    elements.append(qa_table)

    # Build PDF
    doc.build(elements, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] Generated B1 Presentation Script PDF at: {PDF_OUTPUT}")

    import shutil
    shutil.copyfile(PDF_OUTPUT, DOCS_PDF_OUTPUT)
    print(f"[SUCCESS] Copied to: {DOCS_PDF_OUTPUT}")


if __name__ == "__main__":
    build_b1_script_pdf()
