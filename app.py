"""PS 86 - Academic Explanation Comparison Platform (owner: U Rahul Sagar - UI & integration)
Groq generates explanations from the QUESTION ONLY; Gemini extracts and verifies claims against the
REFERENCE; textstat scores readability with no LLM. Accuracy and readability are reported separately."""

import json
import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types as gtypes
from groq import Groq

from generator import GROQ_MODELS, STYLES, generate
from judge import GEMINI_MODELS, extract_claims, reset_judge, verify
from readability import readability
from report_export import to_docx, to_pdf
from reference_loader import build_reference, chunk, label, load_file, load_url, retrieve

load_dotenv()  # reads GROQ_API_KEY / GEMINI_API_KEY from the .env file

st.set_page_config(page_title="PS 86 - Explanation Comparison", page_icon="🔬", layout="wide")

MAX_FILE_MB = 10
MAX_FILES = 5
MAX_QUESTION_CHARS = 300
MAX_REF_CHARS = 12000

# Used only if samples.csv has not been created yet (run prepare_dataset.py to build it from SciQ)
DEMO_SAMPLES = [
    {"subject": "Physics", "question": "Why do we have seasons on Earth?",
     "reference": "Seasons are caused by the 23.5 degree tilt of Earth's rotational axis relative to its orbital plane, "
                  "not by changes in distance from the Sun. The hemisphere tilted toward the Sun receives more direct "
                  "sunlight and longer days, causing summer. The hemisphere tilted away receives more oblique light and "
                  "shorter days, causing winter. Earth is actually closest to the Sun in early January."},
    {"subject": "Physics", "question": "Why is the sky blue?",
     "reference": "Sunlight contains all visible wavelengths. Gas molecules in the atmosphere scatter shorter wavelengths "
                  "much more than longer ones (Rayleigh scattering), so blue light is scattered across the sky and reaches "
                  "our eyes from all directions, making the sky appear blue."},
    {"subject": "Biology", "question": "How does photosynthesis work?",
     "reference": "Photosynthesis converts light energy into chemical energy in chloroplasts. Chlorophyll absorbs light. "
                  "In the light-dependent reactions water is split, releasing oxygen, and ATP and NADPH are made. In the "
                  "Calvin cycle, ATP and NADPH are used to fix carbon dioxide into glucose."},
]


@st.cache_data
def load_samples():
    if os.path.exists("samples.csv"):
        return pd.read_csv("samples.csv").to_dict("records"), "SciQ (samples.csv)"
    return DEMO_SAMPLES, "built-in demo questions (run prepare_dataset.py to load SciQ)"


def secret(name):
    try:
        v = st.secrets.get(name, "")
    except Exception:
        v = ""
    # .env / environment first, then Streamlit secrets (for Streamlit Cloud), then the sidebar box
    return os.environ.get(name, "") or v or st.session_state.get(name, "")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    for key, key_name in [("GROQ_API_KEY", "Groq API key"), ("GEMINI_API_KEY", "Gemini API key")]:
        if secret(key):
            st.success(f"{key_name} loaded")
        else:
            st.session_state[key] = st.text_input(key_name, type="password")
    groq_model = st.selectbox("Generator (Groq)", GROQ_MODELS)
    judge_mode = st.radio("Judge", ["Auto: Gemini, then Groq backup", "Groq only (fast)"],
                          help="Use 'Groq only' if Gemini is slow or out of quota.")
    gemini_model = st.selectbox("Judge model (Gemini)", GEMINI_MODELS)
    styles = st.multiselect("Explanation styles", list(STYLES), default=list(STYLES))
    st.caption("**Accuracy**: Gemini extracts claims and checks each against the reference (temperature 0).\n\n"
               "**Readability**: Flesch formulas via textstat. No LLM.\n\nThe two scores are never combined.")
    with st.expander("🛡️ Responsible AI safeguards"):
        st.markdown(
            "- **Separate scores**: accuracy (AI judge) and readability (formula) are never combined\n"
            "- **No self-grading**: a different model family judges than the one that writes\n"
            "- **Evidence required**: a verdict whose quote is not in the reference is shown as *unverified*\n"
            "- **Low-confidence flags** when the reference does not cover most claims\n"
            "- **Transparency**: the judge model used and the reference passages are always shown\n"
            "- **Scope guard**: non-science or harmful questions are refused\n"
            "- **Prompt-injection guard**: text inside references is treated as data, not instructions\n"
            "- **Input limits**: question length, file size and number of files\n"
            "- **Privacy**: nothing is stored; uploads stay in memory for this session only; keys stay in `.env`")

st.title("🔬 Academic Explanation Comparison Platform")
st.caption("PS 86 · Generate alternative explanations and compare **factual accuracy** and **readability**, reported separately.")
st.info("**Educational aid, not an authority.** Verdicts are only as reliable as the reference you provide, and AI "
        "judges can make mistakes. Check important facts with your teacher or textbook.", icon="ℹ️")

# ---------------------------------------------------------------- input modes
samples, source = load_samples()
mode = st.radio("Input mode", ["Mode 1: Dataset", "Mode 2: Custom"], horizontal=True)
if mode.startswith("Mode 1"):
    st.caption(f"Source: {source}")
    labels = [f"[{s.get('subject', '')}] {s['question']}" for s in samples]
    idx = st.selectbox("Select a question", range(len(samples)), format_func=lambda i: labels[i])
    question = samples[idx]["question"]
    reference = samples[idx]["reference"]
    st.markdown(f"**Question:** {question}")
    with st.expander("Reference (fetched automatically)"):
        st.write(reference)
else:
    question = st.text_input("Science question")
    ref_source = st.radio("Reference source", ["Type / paste notes", "Upload files (PDF, Word, TXT)", "Web link"],
                          horizontal=True)
    reference, uploads, urls = "", [], []
    if ref_source.startswith("Type"):
        reference = st.text_area("Reference explanation / supporting notes (trusted source)", height=130)
    elif ref_source.startswith("Upload"):
        uploads = st.file_uploader("Upload textbook chapters, notes or slides", type=["pdf", "docx", "txt", "md"],
                                   accept_multiple_files=True)
        st.caption("Long files are fine: the app picks the passages most relevant to the question (RAG).")
    else:
        urls = [u.strip() for u in st.text_area("Web link(s), one per line", height=80).splitlines() if u.strip()]
        st.caption("Use trusted pages (e.g. NCERT, Britannica, Khan Academy). Some sites block automated reading.")

own = st.text_area("Optional: paste your own explanation to evaluate (e.g. a student's answer)", height=80)

if st.button("Generate & Compare", type="primary", width="stretch"):
    if not secret("GROQ_API_KEY") or (judge_mode.startswith("Auto") and not secret("GEMINI_API_KEY")):
        st.error("Add the API keys in the sidebar (Gemini is only needed in Auto mode)."); st.stop()
    passages = []
    if mode.startswith("Mode 2") and (uploads or urls):
        if not question.strip():
            st.error("Enter the question first, so the right passages can be found."); st.stop()
        if uploads and len(uploads) > MAX_FILES:
            st.error(f"Please upload at most {MAX_FILES} files."); st.stop()
        too_big = [f.name for f in uploads or [] if f.size > MAX_FILE_MB * 1024 * 1024]
        if too_big:
            st.error(f"Files over {MAX_FILE_MB} MB are not allowed: {', '.join(too_big)}"); st.stop()
        bad_urls = [u for u in urls if not u.lower().startswith(("http://", "https://"))]
        if bad_urls:
            st.error(f"Links must start with http:// or https:// : {', '.join(bad_urls)}"); st.stop()
        docs = []
        for f in uploads or []:
            try:
                docs += load_file(f.name, f.getvalue())
            except Exception as e:
                st.warning(f"Could not read {f.name}: {e}")
        for u in urls:
            try:
                docs += load_url(u)
            except Exception as e:
                st.warning(f"Could not read {u}: {e}")
        if not docs:
            st.error("No readable text found. Scanned PDFs (images) have no text layer - paste the notes instead.")
            st.stop()
        passages = retrieve(question, chunk(docs))
        if not passages:
            st.error("Your sources don't seem to cover this question (no relevant passages found)."); st.stop()
        reference = build_reference(passages)
    if len(question) > MAX_QUESTION_CHARS:
        st.error(f"Question is too long (max {MAX_QUESTION_CHARS} characters)."); st.stop()
    if len(reference) > MAX_REF_CHARS:
        reference = reference[:MAX_REF_CHARS]
        st.warning(f"Reference was long, so only the first {MAX_REF_CHARS} characters are used.")
    if not question.strip() or not reference.strip():
        st.error("A question and a reference are both required."); st.stop()
    if not styles and not own.strip():
        st.error("Choose at least one style or paste an explanation."); st.stop()

    groq_client = Groq(api_key=secret("GROQ_API_KEY"), timeout=40, max_retries=1)
    gem_client = None if judge_mode.startswith("Groq") else genai.Client(
        api_key=secret("GEMINI_API_KEY"), http_options=gtypes.HttpOptions(timeout=25000))  # 25 s per call
    reset_judge()
    items = []
    try:
        with st.status("Running pipeline...", expanded=True) as status:
            for s in styles:
                st.write(f"✍️ Groq: generating **{s}** explanation (question only)")
                text = generate(groq_client, question, s, groq_model)
                if text.strip().upper().startswith("NOT_SCIENCE"):
                    status.update(label="Stopped", state="error")
                    st.error("This doesn't look like a science or academic question (or it asks for unsafe content), "
                             "so the app won't generate explanations for it.")
                    st.stop()
                items.append({"name": s, "text": text})
            if own.strip():
                items.append({"name": "User-provided", "text": own.strip()})
            for it in items:
                st.write(f"🔎 Gemini: extracting and verifying claims for **{it['name']}**")
                claims = extract_claims(gem_client, it["text"], gemini_model, groq_client)
                it["accuracy"] = verify(gem_client, question, reference, claims, gemini_model, groq_client)
                it["readability"] = readability(it["text"])
            status.update(label="Done", state="complete", expanded=False)
    except Exception as e:
        st.error(f"Pipeline error: {e}"); st.stop()
    st.session_state["run"] = {"question": question, "reference": reference, "items": items, "passages": passages,
                               "time": datetime.now().strftime("%Y-%m-%d %H:%M")}

# ---------------------------------------------------------------- results
if "run" in st.session_state:
    run = st.session_state["run"]
    items = run["items"]

    if run.get("passages"):
        with st.expander(f"📚 Reference used: {len(run['passages'])} passages retrieved from your sources"):
            for p in run["passages"]:
                score = f" · relevance {p['score']}" if p.get("score") is not None else ""
                st.markdown(f"**{label(p)}**{score}")
                st.caption(p["text"])

    st.subheader("📄 Explanations side by side")
    for col, it in zip(st.columns(len(items)), items):
        with col:
            st.markdown(f"#### {it['name']}")
            st.info(it["text"])

    t_acc, t_read, t_exp = st.tabs(["✅ Accuracy report", "📖 Readability report", "⬇️ Export"])

    ICON = {"supported": "✅ supported", "contradicted": "❌ contradicted",
            "not_in_reference": "❔ not in reference", "unverified": "⚠️ unverified evidence"}
    with t_acc:
        used = sorted({i["accuracy"].get("judge_model", "") for i in items} - {""})
        st.caption("Judge vs. the reference. Writing style is ignored. Accuracy = supported claims / all claims. "
                   f"Judge model used: {', '.join(used) or 'n/a'}")
        if any(u.startswith("groq:") for u in used) and judge_mode.startswith("Auto"):
            st.warning("Gemini was unavailable (quota or high demand), so a different Groq model family judged instead.")
        for col, it in zip(st.columns(len(items)), items):
            a = it["accuracy"]
            with col:
                st.metric(it["name"], "n/a" if a["accuracy"] is None else f"{a['accuracy']}%",
                          f"{a['contradicted']} contradicted", delta_color="inverse")
                uncertain = a.get("not_in_reference", 0) + a.get("unverified", 0)
                if a["total"] == 0:
                    st.warning("No factual claims found to check.")
                elif uncertain / a["total"] >= 0.5:
                    st.warning(f"Low confidence: {uncertain} of {a['total']} claims are not covered by the "
                               "reference or lack valid evidence. Add a fuller reference.")
        for it in items:
            st.markdown(f"**{it['name']}**")
            rows = [{"Claim": r["claim"], "Verdict": ICON[r["verdict"]], "Evidence from reference": r["evidence"],
                     "Correction": r["correction"]} for r in it["accuracy"]["results"]]
            st.table(pd.DataFrame(rows).set_index("Claim") if rows else pd.DataFrame({"Claim": ["(no claims)"]}).set_index("Claim"))
        acc_df = pd.DataFrame({"Explanation": [i["name"] for i in items],
                               "Accuracy (%)": [i["accuracy"]["accuracy"] or 0 for i in items]})
        st.plotly_chart(px.bar(acc_df, x="Explanation", y="Accuracy (%)", range_y=[0, 100], text="Accuracy (%)",
                               title="Factual accuracy"), width="stretch")

    with t_read:
        st.caption("textstat formulas only. Independent of the accuracy judge.")
        rdf = pd.DataFrame([{"Explanation": i["name"], **i["readability"]} for i in items]).rename(columns={
            "flesch_reading_ease": "Flesch ease (higher = easier)", "grade_level": "Grade level", "words": "Words",
            "sentences": "Sentences", "avg_sentence_length": "Avg sentence length",
            "difficult_words": "Difficult words", "label": "Readability"})
        st.table(rdf.set_index("Explanation"))
        st.plotly_chart(px.bar(rdf, x="Explanation", y="Flesch ease (higher = easier)", range_y=[0, 100],
                               text="Flesch ease (higher = easier)", title="Flesch Reading Ease"), width="stretch")
        best_a = max(items, key=lambda i: i["accuracy"]["accuracy"] or 0)
        best_r = max(items, key=lambda i: i["readability"]["flesch_reading_ease"])
        st.markdown(f"**Most accurate:** {best_a['name']}  |  **Most readable:** {best_r['name']}")

    with t_exp:
        lines = [f"# Explanation Comparison Report", f"_Generated {run['time']}_", "",
                 f"**Question:** {run['question']}", "", f"**Reference:** {run['reference']}", "",
                 "## Accuracy (Gemini judge vs reference)", "| Explanation | Accuracy % | Supported | Contradicted | Not in ref |",
                 "|---|---|---|---|---|"]
        lines += [f"| {i['name']} | {i['accuracy']['accuracy']} | {i['accuracy']['supported']} | "
                  f"{i['accuracy']['contradicted']} | {i['accuracy']['not_in_reference']} |" for i in items]
        lines += ["", "## Readability (textstat, independent)", "| Explanation | Flesch | Grade | Label |", "|---|---|---|---|"]
        lines += [f"| {i['name']} | {i['readability']['flesch_reading_ease']} | {i['readability']['grade_level']} | "
                  f"{i['readability']['label']} |" for i in items]
        report = "\n".join(lines)
        c1, c2, c3, c4 = st.columns(4)
        c1.download_button("📄 Word report (.docx)", to_docx(run), "comparison_report.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        c2.download_button("📕 PDF report (.pdf)", to_pdf(run), "comparison_report.pdf", mime="application/pdf")
        c3.download_button("Markdown (.md)", report, "comparison_report.md")
        c4.download_button("Raw results (.json)", json.dumps(run, indent=2), "comparison_results.json")
        st.markdown(report)
