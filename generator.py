"""Explanation generator (owner: Shaik Mohammad Adil).
Uses Groq. The generator receives ONLY the question - never the reference."""
import re
from groq import Groq

GROQ_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.6-27b"]

STYLES = {
    "Beginner-friendly": "Explain for a 12-year-old. Use short sentences, everyday words and one simple example.",
    "Technical": "Explain for a university student. Use correct scientific terminology and precise detail.",
    "Analogy-based": "Explain using one clear real-world analogy, then connect the analogy back to the science.",
}


def _clean(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def generate(client: Groq, question: str, style: str, model: str = GROQ_MODELS[0]) -> str:
    """Return one explanation of `question` in the given style.
    If the chosen model fails (e.g. rate limit), the other Groq models are tried in turn."""
    system = (
        "You are a science teacher. Answer the question in 120-180 words. "
        "Plain text only, no headings, no bullet points. " + STYLES[style] +
        " Safety rule: if the input is not a genuine science/academic question, or asks for harmful, unsafe or "
        "personal content, reply with exactly NOT_SCIENCE and nothing else."
    )
    order = [model] + [m for m in GROQ_MODELS if m != model]
    try:  # add any other chat models this key can use, so a missing or rate-limited model is not fatal
        skip = ("whisper", "guard", "tts", "playai", "orpheus", "compound", "safeguard")
        order += [m.id for m in client.models.list().data
                  if m.id not in order and not any(k in m.id.lower() for k in skip)]
    except Exception:
        pass
    order = order[:5]
    last_error = None
    for m in order:
        try:
            resp = client.chat.completions.create(
                model=m,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": f"Question: {question}"}],
                temperature=0.7,
            )
            return _clean(resp.choices[0].message.content)
        except Exception as e:  # rate limit, model unavailable, etc.
            last_error = e
    raise RuntimeError(f"All Groq models failed: {last_error}")


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    c = Groq(api_key=os.environ["GROQ_API_KEY"])
    for s in STYLES:
        print(f"--- {s} ---\n{generate(c, 'Why do we have seasons on Earth?', s)}\n")
