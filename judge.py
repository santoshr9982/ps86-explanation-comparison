"""Accuracy engine (owner: K Santosh Reddy).
Gemini extracts atomic claims from an explanation, then verifies each claim against the
reference. A different model family from the generator avoids self-evaluation bias.
Temperature 0 + JSON output keep verdicts consistent."""
import json
import re
from google import genai
from google.genai import types

GEMINI_MODELS = ["gemini-3.5-flash", "gemini-flash-latest"]

EXTRACT_SYSTEM = """Split the given explanation into short atomic factual claims.
Each claim = one piece of scientific information, written as a standalone sentence.
Skip greetings, analogies' story details and opinions; keep the scientific facts they map to.
Return ONLY JSON: {"claims": ["...", "..."]}"""

VERIFY_SYSTEM = """You are a strict science fact-checker. Judge each CLAIM against the REFERENCE only.
Verdicts:
- "supported": the reference states or directly implies it
- "contradicted": the reference states something incompatible with it
- "not_in_reference": the reference does not cover it (do not use your own knowledge)
Simplifications are fine if they are not false. Ignore writing style completely.
For "supported" and "contradicted", "evidence" MUST be an exact quote copied from the reference.
For "contradicted", give a short "correction" based on the reference.
Return ONLY JSON:
{"results": [{"claim": "...", "verdict": "supported|contradicted|not_in_reference", "evidence": "...", "correction": "..."}]}"""


def _parse(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


def _call(client: genai.Client, model: str, system: str, user: str) -> dict:
    order = [model] + [m for m in GEMINI_MODELS if m != model]
    last_error = None
    for m in order:
        try:
            resp = client.models.generate_content(
                model=m,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system, temperature=0, response_mime_type="application/json"
                ),
            )
            return _parse(resp.text)
        except Exception as e:
            last_error = e
    raise RuntimeError(f"All Gemini models failed: {last_error}")


def extract_claims(client: genai.Client, text: str, model: str = GEMINI_MODELS[0]) -> list:
    data = _call(client, model, EXTRACT_SYSTEM, f"EXPLANATION:\n{text}")
    return [c.strip() for c in data.get("claims", []) if isinstance(c, str) and c.strip()]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())).strip()


def verify(client: genai.Client, question: str, reference: str, claims: list, model: str = GEMINI_MODELS[0]) -> dict:
    if not claims:
        return {"results": [], "accuracy": None, "supported": 0, "contradicted": 0, "not_in_reference": 0, "total": 0}
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    data = _call(client, model, VERIFY_SYSTEM, f"QUESTION:\n{question}\n\nREFERENCE:\n{reference}\n\nCLAIMS:\n{numbered}")
    results = []
    ref_norm = _norm(reference)
    for r in data.get("results", []):
        verdict = r.get("verdict", "not_in_reference")
        if verdict not in ("supported", "contradicted", "not_in_reference"):
            verdict = "not_in_reference"
        evidence = r.get("evidence", "") or ""
        # Safeguard: evidence must really appear in the reference, otherwise the verdict is not trusted.
        evidence_ok = bool(evidence) and _norm(evidence) in ref_norm
        if verdict in ("supported", "contradicted") and not evidence_ok:
            verdict_shown = "unverified"
        else:
            verdict_shown = verdict
        results.append({
            "claim": r.get("claim", ""),
            "verdict": verdict_shown,
            "evidence": evidence,
            "correction": r.get("correction", "") or "",
        })
    counts = {k: sum(1 for r in results if r["verdict"] == k) for k in ("supported", "contradicted", "not_in_reference", "unverified")}
    total = len(results)
    return {
        "results": results,
        "accuracy": round(100 * counts["supported"] / total, 1) if total else None,
        "total": total,
        **counts,
    }


if __name__ == "__main__":
    import os
    c = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    ref = ("Seasons are caused by the 23.5 degree tilt of Earth's axis, not by changes in distance from the Sun. "
           "The hemisphere tilted toward the Sun receives more direct sunlight and longer days, causing summer.")
    exp = "Earth is tilted. When your half leans toward the Sun, you get summer. Also, Earth is closer to the Sun in summer."
    claims = extract_claims(c, exp)
    print("Claims:", claims)
    print(json.dumps(verify(c, "Why do we have seasons?", ref, claims), indent=2))
