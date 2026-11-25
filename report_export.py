"""Export the comparison result as a Word (.docx) or PDF report.
Both reports keep factual accuracy and readability in SEPARATE sections, as PS 86 requires."""

import io

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from xml.sax.saxutils import escape

VERDICT = {"supported": "Supported", "contradicted": "Contradicted",
           "not_in_reference": "Not in reference", "unverified": "Unverified evidence"}


def _source_line(run):
    ps = run.get("passages") or []
    if not ps:
        return None
    seen = []
    for p in ps:
        s = p["source"] + (f" (p. {p['page']})" if p.get("page") else "")
        if s not in seen:
            seen.append(s)
    return "; ".join(seen)


def _acc(i):
    a = i["accuracy"]
    return ("n/a" if a["accuracy"] is None else f"{a['accuracy']}%"), a


# ------------------------------------------------------------------ Word
def to_docx(run: dict) -> bytes:
    doc = Document()
    st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(11)

    t = doc.add_heading("Explanation Comparison Report", level=0)
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph(f"PS 86 – Academic Explanation Comparison Platform · Generated {run['time']}")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading("1. Question and reference", level=1)
    doc.add_paragraph().add_run("Question: ").bold = True
    doc.paragraphs[-1].add_run(run["question"])
    src = _source_line(run)
    if src:
        para = doc.add_paragraph(); para.add_run("Reference sources: ").bold = True; para.add_run(src)
    doc.add_paragraph().add_run("Reference text used:").bold = True
    doc.add_paragraph(run["reference"])

    doc.add_heading("2. Explanations compared", level=1)
    for i in run["items"]:
        doc.add_heading(i["name"], level=2)
        doc.add_paragraph(i["text"])

    doc.add_heading("3. Factual accuracy (Gemini judge vs reference)", level=1)
    doc.add_paragraph("Accuracy = supported claims ÷ all claims. Writing style is ignored.")
    tb = doc.add_table(rows=1, cols=5); tb.style = "Light Grid Accent 1"
    for c, h in zip(tb.rows[0].cells, ["Explanation", "Accuracy", "Supported", "Contradicted", "Not in reference"]):
        c.text = h
    for i in run["items"]:
        score, a = _acc(i)
        for c, v in zip(tb.add_row().cells, [i["name"], score, a["supported"], a["contradicted"], a["not_in_reference"]]):
            c.text = str(v)
    for i in run["items"]:
        doc.add_heading(f"Claim check – {i['name']}", level=2)
        ct = doc.add_table(rows=1, cols=4); ct.style = "Light Grid Accent 1"
        for c, h in zip(ct.rows[0].cells, ["Claim", "Verdict", "Evidence from reference", "Correction"]):
            c.text = h
        for r in i["accuracy"]["results"]:
            cells = ct.add_row().cells
            cells[0].text, cells[2].text, cells[3].text = r["claim"], r["evidence"], r["correction"]
            run_v = cells[1].paragraphs[0].add_run(VERDICT[r["verdict"]])
            if r["verdict"] == "contradicted":
                run_v.bold = True; run_v.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)
            elif r["verdict"] == "supported":
                run_v.font.color.rgb = RGBColor(0x1E, 0x7B, 0x34)

    doc.add_heading("4. Readability (textstat formulas, no LLM)", level=1)
    doc.add_paragraph("Reported separately from accuracy. Higher Flesch Reading Ease = easier to read.")
    rt = doc.add_table(rows=1, cols=5); rt.style = "Light Grid Accent 1"
    for c, h in zip(rt.rows[0].cells, ["Explanation", "Flesch ease", "Grade level", "Avg sentence length", "Label"]):
        c.text = h
    for i in run["items"]:
        m = i["readability"]
        for c, v in zip(rt.add_row().cells, [i["name"], m["flesch_reading_ease"], m["grade_level"],
                                              m["avg_sentence_length"], m["label"]]):
            c.text = str(v)

    doc.add_heading("5. Summary", level=1)
    best_a = max(run["items"], key=lambda i: i["accuracy"]["accuracy"] or 0)
    best_r = max(run["items"], key=lambda i: i["readability"]["flesch_reading_ease"])
    doc.add_paragraph(f"Most accurate: {best_a['name']}. Most readable: {best_r['name']}. "
                      "Accuracy and readability are scored independently and never combined.")
    buf = io.BytesIO(); doc.save(buf)
    return buf.getvalue()


# ------------------------------------------------------------------ PDF
def to_pdf(run: dict) -> bytes:
    buf = io.BytesIO()
    pdf = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm,
                            bottomMargin=2 * cm, title="Explanation Comparison Report")
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=10.5, leading=14)
    cell = ParagraphStyle("c", parent=body, fontSize=9, leading=11.5)
    h1, h2 = ss["Heading2"], ss["Heading3"]
    E = lambda s: escape(str(s))
    P = lambda s, st=body: Paragraph(E(s), st)

    def table(header, rows, widths):
        data = [[Paragraph(f"<b>{E(h)}</b>", cell) for h in header]] + \
               [[c if isinstance(c, Paragraph) else Paragraph(E(c), cell) for c in r] for r in rows]
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
                               ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        return t

    W = A4[0] - 4 * cm
    s = [Paragraph("Explanation Comparison Report", ParagraphStyle("t", parent=ss["Title"], fontSize=18)),
         Paragraph(E(f"PS 86 – Academic Explanation Comparison Platform | Generated {run['time']}"),
                   ParagraphStyle("st", parent=body, alignment=TA_CENTER, textColor=colors.grey)),
         Spacer(1, 10),
         Paragraph("1. Question and reference", h1),
         Paragraph(f"<b>Question:</b> {E(run['question'])}", body)]
    src = _source_line(run)
    if src:
        s.append(Paragraph(f"<b>Reference sources:</b> {E(src)}", body))
    s += [Spacer(1, 4), Paragraph("<b>Reference text used:</b>", body), P(run["reference"]),
          Paragraph("2. Explanations compared", h1)]
    for i in run["items"]:
        s += [Paragraph(E(i["name"]), h2), P(i["text"])]

    s += [Paragraph("3. Factual accuracy (Gemini judge vs reference)", h1),
          P("Accuracy = supported claims ÷ all claims. Writing style is ignored.")]
    rows = []
    for i in run["items"]:
        score, a = _acc(i)
        rows.append([i["name"], score, a["supported"], a["contradicted"], a["not_in_reference"]])
    s.append(table(["Explanation", "Accuracy", "Supported", "Contradicted", "Not in ref."], rows,
                   [W * .32, W * .17, W * .17, W * .17, W * .17]))
    color = {"supported": "#1E7B34", "contradicted": "#C00000"}
    for i in run["items"]:
        s.append(Paragraph(E(f"Claim check – {i['name']}"), h2))
        rows = [[r["claim"],
                 Paragraph(f"<font color='{color.get(r['verdict'], '#555555')}'><b>{VERDICT[r['verdict']]}</b></font>", cell),
                 r["evidence"], r["correction"]] for r in i["accuracy"]["results"]]
        s.append(table(["Claim", "Verdict", "Evidence from reference", "Correction"], rows,
                       [W * .3, W * .16, W * .3, W * .24]))

    s += [Paragraph("4. Readability (textstat formulas, no LLM)", h1),
          P("Reported separately from accuracy. Higher Flesch Reading Ease = easier to read.")]
    rows = [[i["name"], i["readability"]["flesch_reading_ease"], i["readability"]["grade_level"],
             i["readability"]["avg_sentence_length"], i["readability"]["label"]] for i in run["items"]]
    s.append(table(["Explanation", "Flesch ease", "Grade level", "Avg sentence len.", "Label"], rows,
                   [W * .28, W * .16, W * .16, W * .18, W * .22]))

    best_a = max(run["items"], key=lambda i: i["accuracy"]["accuracy"] or 0)
    best_r = max(run["items"], key=lambda i: i["readability"]["flesch_reading_ease"])
    s += [Paragraph("5. Summary", h1),
          P(f"Most accurate: {best_a['name']}. Most readable: {best_r['name']}. "
            "Accuracy and readability are scored independently and never combined.")]
    pdf.build(s)
    return buf.getvalue()