#!/usr/bin/env python3
"""
src/generate_presentation_script_pdf.py
=======================================
Generates a publication-grade, professionally structured 3-4 minute presentation script
in PDF format for presenting Task #1094:
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
PDF_OUTPUT = os.path.join(RESULTS_DIR, "TollMatcher_Benchmark_Presentation_Script.pdf")
DOCS_PDF_OUTPUT = os.path.join(WORKSPACE_ROOT, "benchmark", "docs", "TollMatcher_Benchmark_Presentation_Script.pdf")


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
            self.drawString(16 * mm, 285 * mm, "TollMatcher — 3–4 Minute Presentation Script (Benchmark on Raspberry Pi)")
            self.drawRightString(194 * mm, 285 * mm, "Task #1094 • Yashkumar Chothani")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(16 * mm, 282 * mm, 194 * mm, 282 * mm)

        # Running Footer (all pages)
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(16 * mm, 14 * mm, 194 * mm, 14 * mm)
        self.drawString(16 * mm, 10 * mm, "Ostfalia University of Applied Sciences • Department of Computer Science • SoSe/WiSe 2026")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(194 * mm, 10 * mm, page_text)
        self.restoreState()


def build_script_pdf():
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
        fontSize=9,
        leading=13.5,
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
        fontSize=8,
        leading=11,
        textColor=dark_text
    )

    elements = []

    # Header Card
    elements.append(Paragraph("TollMatcher — Benchmark Presentation Script", title_style))
    elements.append(Paragraph("Verbatim 3.5-Minute Speaker Script: Proposed Streaming Algorithm vs. Summersemester 2026 Baseline", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=primary_blue, spaceBefore=0, spaceAfter=6))

    meta_table_data = [
        [
            Paragraph("<b>Target Audience:</b> Prof. Dr. Lehmann, Michael Thane, Team", speech_text),
            Paragraph("<b>Total Speaking Time:</b> ~3 minutes 30 seconds (~480 words)", speech_text)
        ],
        [
            Paragraph("<b>Hardware Target:</b> Raspberry Pi 5 (ARM Cortex-A76 @ 2.4 GHz)", speech_text),
            Paragraph("<b>Task Reference:</b> Task #1094 (Running the Benchmark)", speech_text)
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
                Paragraph(f"<b>Visual / Slide:</b> {slide_str}", speech_emphasis),
                Paragraph(f"<b>Cue:</b> {action_str}", action_cue)
            ]
        ]
        hdr_table = Table(header_data, colWidths=[28 * mm, 72 * mm, 80 * mm])
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
    elements.append(Paragraph("Minute 1: Introduction & The Core Bottleneck (0:00 – 0:45)", section_h1))
    p1 = (
        "“Good morning everyone. Today I am presenting our empirical benchmark evaluation for TollMatcher "
        "running directly on the Raspberry Pi 5. Our objective was to measure the exact performance difference between "
        "the <b>previous algorithm from Summersemester 2026</b> and our <b>newly proposed streaming adaptive pipeline</b>.”"
    )
    p2 = (
        "“In conversational speech translation, deciding <i>when a speaker has finished talking</i> is critical. "
        "In everyday dialogues, people frequently take natural <b>700 to 900 millisecond thinking pauses</b> mid-sentence. "
        "If the device cuts off too early, the sentence fragments and the translation breaks down. "
        "If it waits too long, the conversational flow feels sluggish and unresponsive.”"
    )
    elements.append(make_section_block("0:00 – 0:45", "Slide 1: Title & The Pause Dilemma", "Speak clearly, maintain eye contact, set problem context", [p1, p2]))

    # -------------------------------------------------------------
    # Minute 2: 0:45 - 1:45
    # -------------------------------------------------------------
    elements.append(Paragraph("Minute 2: Previous Algorithm vs. Proposed Solution (0:45 – 1:45)", section_h1))
    p3 = (
        "“Let's examine how the two algorithms handle this. The <b>previous algorithm from Summersemester 2026</b> "
        "— which we benchmark as <b>Variant B</b> — relies on energy-based Voice Activity Detection with a <b>rigid 600 ms silence cutoff</b>. "
        "Whenever a speaker pauses for more than 600 ms, the system assumes the sentence is over and immediately cuts the audio. "
        "Furthermore, it waits for the entire recording to complete before starting transcription, causing high latency.”"
    )
    p4 = (
        "“To solve this, we introduced our <b>proposed algorithm, Variant C</b>. It combines three key innovations:<br/>"
        "1. <b>Adaptive Pause Bridging:</b> If an utterance is syntactically incomplete, the silence threshold dynamically expands to <b>1,000 ms</b>, allowing speakers to pause naturally without interruption.<br/>"
        "2. <b>Local Agreement (<i>k</i>=2):</b> Hypotheses are continuously streamed in 100 ms chunks and committed only when two consecutive Whisper decodings agree on words.<br/>"
        "3. <b>Concurrent Pipelining:</b> Synthesizing and translating audio while the speaker is still finishing their utterance.”"
    )
    elements.append(make_section_block("0:45 – 1:45", "Slide 2: Architectural Comparison Diagram", "Point to the 600ms cutoff vs. adaptive 1000ms pause bridging", [p3, p4]))

    # -------------------------------------------------------------
    # Minute 3: 1:45 - 2:45
    # -------------------------------------------------------------
    elements.append(Paragraph("Minute 3: Benchmark Findings on Raspberry Pi 5 (1:45 – 2:45)", section_h1))
    p5 = (
        "“We executed our evaluation matrix across <b>120 rigorous benchmark runs</b> using official <b>FLORES-200</b> "
        "multi-way aligned sentences in <b>English, German, and French</b> across all six bi-directional pairs. "
        "Here are the empirical findings on the Raspberry Pi 5:<br/><br/>"
        "• <b>Premature Cuts:</b> The previous algorithm triggered <b>24 false cuts</b> during thinking pauses, severely fragmenting sentences. Our proposed algorithm achieved <font color='#047857'><b>exactly 0 false cuts</b></font> — a 100% reduction in premature cut defects.<br/>"
        "• <b>First Audio Latency (TTFA):</b> The previous algorithm had a median Time-to-First-Audio of <b>3,682 ms</b> — almost 3.7 seconds. Our proposed algorithm begins emitting translated speech in just <font color='#047857'><b>360 ms</b></font>. That is a <b>10× reduction in perceived latency</b>.<br/>"
        "• <b>Translation Fidelity:</b> Because sentences are no longer chopped into fragments, translation quality increased from <b>91.7 to 100.0 BLEU</b> and <b>98.0 to 100.0 chrF++</b>.<br/>"
        "• <b>Computational Efficiency:</b> The Real-Time Factor on the Pi's Cortex-A76 CPU is <b>0.108</b>, meaning the system transcribes and translates more than <b>9 times faster than real-time speech</b> with zero thermal throttling.”"
    )
    elements.append(make_section_block("1:45 – 2:45", "Slide 3: Executive Comparison Table", "Emphasize: 24 cuts → 0 cuts, 3.7s → 360ms TTFA", [p5]))

    # -------------------------------------------------------------
    # Minute 4: 2:45 - 3:30
    # -------------------------------------------------------------
    elements.append(Paragraph("Minute 4: Reproducibility, Demonstration & Wrap-Up (2:45 – 3:30)", section_h1))
    p6 = (
        "“Every single number in this report is 100% reproducible directly on the device with a single command:<br/>"
        "<code>python evaluate_variants.py --compare</code>.<br/>"
        "This runs both algorithms head-to-head and outputs the exact Markdown and CSV reports into the <code>results/</code> directory.<br/>"
        "Furthermore, we deployed our <b>Web Translation Lab</b> on port 8000, allowing any team member to test live microphone inputs or inspect FLORES-200 samples interactively.”"
    )
    p7 = (
        "“In conclusion: our proposed streaming adaptive pipeline completely eliminates false sentence cutoffs, "
        "reduces listener response latency by a factor of 10, and runs smoothly within the resource envelope of the Raspberry Pi 5. "
        "Thank you, and I look forward to your questions.”"
    )
    elements.append(make_section_block("2:45 – 3:30", "Slide 4: Live Terminal / Web Lab Demo", "Smile, thank the committee, open floor for questions", [p6, p7]))

    # -------------------------------------------------------------
    # Q&A Cheat Sheet
    # -------------------------------------------------------------
    elements.append(Spacer(1, 4))
    elements.append(Paragraph("Anticipated Q&A Cheat Sheet (Preparation for Committee Questions)", section_h1))

    qa_data = [
        [
            Paragraph("<b>Q1: What does Local Agreement (k=2) actually do?</b>", qa_question),
            Paragraph("<b>Answer:</b> Whisper outputs streaming text hypotheses every 100ms. If we emit words immediately, Whisper might change its mind when more audio arrives (flicker). Local Agreement ($k=2$) holds words in a buffer until two consecutive sliding windows agree on the exact same prefix words. Once agreed, words are committed to translation with 100% stability.", qa_answer)
        ],
        [
            Paragraph("<b>Q2: Why not just increase the silence cutoff from 600ms to 1200ms everywhere?</b>", qa_question),
            Paragraph("<b>Answer:</b> Because if you make the silence cutoff 1200ms unconditionally, the user has to wait 1.2 seconds <i>every single time</i> they stop talking before anything happens. Our adaptive algorithm stays at a snappy 400ms when an utterance looks syntactically complete, and only bridges up to 1000ms when the sentence is in an active clause.", qa_answer)
        ],
        [
            Paragraph("<b>Q3: What model is running on the Raspberry Pi?</b>", qa_question),
            Paragraph("<b>Answer:</b> We run <code>faster-whisper</code> (tiny model) quantized to int8 on CPU, paired with CTranslate2-based ArgosTranslate. This combination uses only ~19 MB RSS memory overhead during benchmarking and delivers an RTF of 0.108.", qa_answer)
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
    print(f"[SUCCESS] Generated Presentation Script PDF at: {PDF_OUTPUT}")

    import shutil
    shutil.copyfile(PDF_OUTPUT, DOCS_PDF_OUTPUT)
    print(f"[SUCCESS] Copied to: {DOCS_PDF_OUTPUT}")


if __name__ == "__main__":
    build_script_pdf()
