# PS 86 – Academic Explanation Comparison Platform

GenAI app that generates alternative explanations for a science question and reports
**factual accuracy** (claim-level, Gemini judge vs. reference) **separately** from
**readability** (Flesch formulas, no LLM).

Team (Venue 206): V Rakesh Reddy · Shaik Mohammad Adil · K Santosh Reddy · U Rahul Sagar

## Files and owners

| File | What it does | Owner |
|---|---|---|
| `prepare_dataset.py` | Downloads SciQ, filters it, saves `samples.csv` | V Rakesh Reddy |
| `readability.py` | Flesch Reading Ease, grade level, label (no LLM) | V Rakesh Reddy |
| `generator.py` | Groq: 3 explanation styles from the **question only** | Shaik Mohammad Adil |
| `judge.py` | Gemini: extract claims, verify each vs reference (temp 0, JSON) | K Santosh Reddy |
| `test_judge.py` | Planted-error test: % of wrong answers the judge catches | Rakesh / Santosh |
| `app.py` | Streamlit UI: Mode 1 (SciQ) / Mode 2 (custom), side-by-side reports | U Rahul Sagar |

## Run locally

```bash
pip install -r requirements.txt
python prepare_dataset.py          # creates samples.csv from SciQ (needs internet)
mkdir -p .streamlit && cp secrets.toml.example .streamlit/secrets.toml   # then paste your keys
streamlit run app.py
```

Keys: Groq → https://console.groq.com/keys · Gemini → https://aistudio.google.com/apikey

Test each module on its own:

```bash
export GROQ_API_KEY=...  GEMINI_API_KEY=...
python readability.py
python generator.py
python judge.py
python test_judge.py --n 10       # writes judge_test_results.csv
```

## Deploy (Streamlit Community Cloud)

share.streamlit.io → Create app → this repo, `app.py` → Advanced settings → Secrets:
paste the two key lines → Deploy. Commit `samples.csv` so Mode 1 uses SciQ online.

## Pipeline

Input (SciQ or custom) → **Groq** generator (question only) → 3 explanations →
**Gemini** claim extraction + verification against the reference (accuracy report) and
**textstat** (readability report) in parallel → side-by-side dashboard + export.

Safeguard: a verdict whose evidence quote is not found in the reference is shown as
"unverified" instead of being trusted.
