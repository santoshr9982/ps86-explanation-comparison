"""Dataset preparation (owner: V Rakesh Reddy).
Downloads SciQ from Hugging Face, cleans it and saves samples.csv for the app.
Run once:  python prepare_dataset.py            (default 30 questions)
           python prepare_dataset.py --n 50
"""
import argparse
import pandas as pd
from datasets import load_dataset

# Simple keyword buckets to balance subjects (SciQ has no subject column).
SUBJECTS = {
    "Physics": ["force", "energy", "light", "wave", "motion", "electric", "magnet", "heat", "gravity", "sound", "velocity", "pressure"],
    "Chemistry": ["atom", "molecule", "chemical", "reaction", "acid", "base", "element", "compound", "bond", "electron", "ion", "solution"],
    "Biology": ["cell", "organism", "plant", "animal", "gene", "dna", "protein", "species", "blood", "body", "photosynthesis", "evolution"],
}


def subject_of(text: str) -> str:
    t = text.lower()
    scores = {s: sum(k in t for k in kws) for s, kws in SUBJECTS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Other"


def main(n: int, seed: int, out: str):
    ds = load_dataset("allenai/sciq", split="train")
    df = ds.to_pandas()
    print("Raw rows:", len(df))

    # 1. Keep rows that have a usable support paragraph (many SciQ rows have none)
    df["support"] = df["support"].fillna("").str.strip()
    df = df[df["support"].str.split().str.len().between(40, 300)]
    print("After support-length filter (40-300 words):", len(df))

    # 2. Keep rows whose support actually mentions the correct answer (reference states the answer)
    df = df[df.apply(lambda r: str(r["correct_answer"]).lower() in r["support"].lower(), axis=1)]
    print("After 'answer appears in support' filter:", len(df))

    # 3. Balance across subjects
    df["subject"] = (df["question"] + " " + df["support"]).apply(subject_of)
    df = df[df["subject"] != "Other"]
    per = max(1, n // 3)
    parts = [g.sample(min(per, len(g)), random_state=seed) for _, g in df.groupby("subject")]
    out_df = pd.concat(parts).head(n).reset_index(drop=True)

    # 4. Build the reference = support + explicit answer
    out_df["reference"] = out_df["support"] + " Answer: " + out_df["correct_answer"].astype(str) + "."
    out_df.insert(0, "id", range(1, len(out_df) + 1))
    cols = ["id", "subject", "question", "reference", "correct_answer", "distractor1", "distractor2", "distractor3"]
    out_df[cols].to_csv(out, index=False)
    print(f"Saved {len(out_df)} rows to {out}")
    print(out_df["subject"].value_counts().to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="samples.csv")
    a = p.parse_args()
    main(a.n, a.seed, a.out)
