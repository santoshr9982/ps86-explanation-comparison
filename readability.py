"""Readability module (owner: V Rakesh Reddy).
Formula-based only - no LLM - so it stays independent of the accuracy judge."""
import textstat


def label(ease: float) -> str:
    if ease >= 80:
        return "Very easy"
    if ease >= 60:
        return "Easy / plain English"
    if ease >= 50:
        return "Fairly difficult"
    if ease >= 30:
        return "Difficult"
    return "Very difficult"


def readability(text: str) -> dict:
    ease = textstat.flesch_reading_ease(text)
    words = textstat.lexicon_count(text)
    sentences = max(1, textstat.sentence_count(text))
    return {
        "flesch_reading_ease": round(ease, 1),
        "grade_level": round(textstat.flesch_kincaid_grade(text), 1),
        "words": words,
        "sentences": sentences,
        "avg_sentence_length": round(words / sentences, 1),
        "difficult_words": textstat.difficult_words(text),
        "label": label(ease),
    }


if __name__ == "__main__":
    for t in [
        "Sunlight has many colors. Air bounces blue light around the most. So the sky looks blue.",
        "Rayleigh scattering, proportional to the inverse fourth power of wavelength, preferentially "
        "redirects shorter wavelengths, producing the characteristic azure appearance of the sky.",
    ]:
        print(readability(t))
