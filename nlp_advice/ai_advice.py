import re
import hashlib
from typing import Tuple

_GENERATOR = None

def _get_generator():
    global _GENERATOR
    if _GENERATOR is not None:
        return _GENERATOR
    try:
        from transformers import pipeline
        _GENERATOR = pipeline("text-generation", model="distilgpt2")
        return _GENERATOR
    except Exception:
        return None

SAFE_FALLBACK = (
    "Increase hydration, prioritize 7–8 hours of sleep, "
    "do light movement (walk/stretch), and monitor how you feel."
)

UNSAFE_PATTERNS = re.compile(
    r"\b(prescription|medicine|drug|dose|mg|antibiotic|steroid|opioid|"
    r"diagnose|diagnosis|cure|treat|contraindicat|side\s*effect|"
    r"pregnan|breastfeed|insulin|antidepressant|chemotherapy|"
    r"take\s+\d+mg|pill|tablet)\b",
    flags=re.IGNORECASE
)

def _seed_from_values(hr: float, slp: float, act: float, temp: float, jitter: int) -> int:
    s = f"{hr:.2f}-{slp:.2f}-{act:.2f}-{temp:.2f}-{jitter}"
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2**31)

def _clean_text(text: str, max_sentences: int = 2) -> str:
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = text.replace("\n", " ").strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    text = " ".join(parts[:max_sentences]).strip()
    return text[:300].strip()

def generate_dynamic_advice(
    hr: float, slp: float, act: float, temp: float, anomaly: int,
    temperature: float = 0.9, top_p: float = 0.92, jitter: int = 0
) -> Tuple[str, bool]:
    """
    Returns (advice, used_fallback).
    'temperature' and 'top_p' control creativity.
    'jitter' changes the seed to vary phrasing even for same inputs.
    """
    gen = _get_generator()
    base_prompt = (
        "You are a helpful wellness assistant. "
        "Give short, safe, lifestyle-only suggestions (no medicines, no diagnoses). "
        "1–2 sentences, friendly tone.\n"
    )
    case_tone = "User may be under stress. " if anomaly else "User appears within normal range. "
    user_stats = (
        f"Heart rate: {hr} bpm. Sleep: {slp} hours. "
        f"Activity: {act}/10. Temperature: {temp} °C.\n"
        "Give personalized, actionable advice."
    )
    prompt = base_prompt + case_tone + user_stats

    if gen is None:
        return (SAFE_FALLBACK, True)

    try:
        from transformers import set_seed
        seed = _seed_from_values(hr, slp, act, temp, jitter)
        set_seed(seed)
        out = gen(
            prompt,
            max_new_tokens=64,
            do_sample=True,
            temperature=float(max(0.1, min(2.0, temperature))),
            top_p=float(max(0.1, min(1.0, top_p))),
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=50256
        )[0]["generated_text"]
        text = _clean_text(out)

        if not text or UNSAFE_PATTERNS.search(text):
            return (SAFE_FALLBACK, True)
        if any(w in text.lower() for w in ["diagnose", "prescription", "dose", "mg"]):
            return (SAFE_FALLBACK, True)

        return (text, False)
    except Exception:
        return (SAFE_FALLBACK, True)
