"""Initial testing of the accuracy judge (owner: V Rakesh Reddy / K Santosh Reddy).
For each SciQ item we build two short explanations:
  - CORRECT:  states the correct answer
  - PLANTED:  states a wrong answer taken from a SciQ distractor
The judge should mark the planted claim "contradicted" and the correct one "supported".
Run:  python test_judge.py --n 10      (needs samples.csv and GEMINI_API_KEY)
Writes judge_test_results.csv and prints detection rate."""
import argparse
import os
import time
import pandas as pd
from google import genai
from judge import extract_claims, verify


def run(n: int, model: str):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    df = pd.read_csv("samples.csv").head(n)
    rows = []
    for _, r in df.iterrows():
        for kind, answer in [("correct", r["correct_answer"]), ("planted_error", r["distractor1"])]:
            explanation = f"{r['question']} The answer is {answer}."
            try:
                claims = extract_claims(client, explanation, model)
                res = verify(client, r["question"], r["reference"], claims, model)
                flagged = res["contradicted"] > 0
                rows.append({"id": r["id"], "kind": kind, "explanation": explanation, "claims": len(claims),
                             "contradicted": res["contradicted"], "supported": res["supported"],
                             "accuracy": res["accuracy"], "flagged_error": flagged})
                print(f"#{r['id']:>3} {kind:<14} flagged={flagged}  accuracy={res['accuracy']}")
            except Exception as e:
                print(f"#{r['id']:>3} {kind:<14} ERROR {e}")
            time.sleep(1)  # stay under free-tier rate limits
    out = pd.DataFrame(rows)
    out.to_csv("judge_test_results.csv", index=False)
    planted = out[out["kind"] == "planted_error"]
    correct = out[out["kind"] == "correct"]
    if len(planted):
        print(f"\nPlanted errors caught: {planted['flagged_error'].sum()} / {len(planted)} "
              f"({100 * planted['flagged_error'].mean():.0f}%)")
    if len(correct):
        print(f"False alarms on correct answers: {correct['flagged_error'].sum()} / {len(correct)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--model", default="gemini-3.5-flash")
    a = p.parse_args()
    run(a.n, a.model)
