# api/main.py
from fastapi import FastAPI, Query
from pydantic import BaseModel
import joblib, json
import numpy as np
from pathlib import Path

# Optional LLM (advice)
try:
    from transformers import pipeline
    generator = pipeline('text-generation', model='distilgpt2')
except Exception:
    generator = None
    print("Note: transformers not available; using safe fallback advice.")

MODEL = joblib.load("models/anomaly_model.pkl")
CONFIG = json.loads(Path("models/config.json").read_text())

FEATURES = CONFIG["feature_order"]
THRESHOLD = float(CONFIG["threshold"])
RULES = CONFIG.get("rule_safety_net", {"hr_high": 110, "sleep_low": 5.0, "temp_high": 37.8})

app = FastAPI(title="Smart Health Monitoring API (Calibrated)")

class Features(BaseModel):
    heart_rate: float
    sleep_hours: float
    activity_level: float
    temperature: float

def rule_safety_net(f: Features) -> bool:
    # Simple override for obvious anomalies
    # Severe sleep deprivation rule:
    if f.sleep_hours < 2:
        return True

    # Heart + sleep combined:
    if f.heart_rate > 100 and f.sleep_hours < 4:
        return True

    # High temp always anomaly:
    if f.temperature > RULES["temp_high"]:
        return True
    return False

def predict_proba_ordered(vals):
    # Ensure feature order consistency
    arr = np.array([[vals[FEATURES.index("heart_rate")],
                     vals[FEATURES.index("sleep_hours")],
                     vals[FEATURES.index("activity_level")],
                     vals[FEATURES.index("temperature")]]])
    return float(MODEL.predict_proba(arr)[:, 1][0])

def get_advice(f: Features):
    if generator is None:
        return ("Increase hydration, prioritize 7–8 hours of sleep, "
                "do light activity (walk/stretch), and monitor temperature.")
    prompt = (f"User heart rate: {f.heart_rate} bpm, sleep: {f.sleep_hours} hours, "
              f"activity level: {f.activity_level}/10, temperature: {f.temperature} °C. "
              "Give concise, safe, general lifestyle advice only (no medicines). "
              "One or two sentences.")
    out = generator(prompt, max_new_tokens=60, do_sample=False, pad_token_id=50256)[0]["generated_text"]
    return out.strip()

@app.get("/predict")
def predict(features: str = Query(..., description="Comma-separated: heart_rate,sleep_hours,activity_level,temperature")):
    vals = [float(x) for x in features.split(",")]
    f = Features(heart_rate=vals[0], sleep_hours=vals[1], activity_level=vals[2], temperature=vals[3])

    proba = predict_proba_ordered(vals)
    pred = int(proba >= THRESHOLD)

    if rule_safety_net(f):
        pred = 1  # override to anomaly

    advice = get_advice(f)
    return {"anomaly": pred, "probability": round(proba, 4), "threshold": THRESHOLD, "advice": advice}

@app.post("/predict_json")
def predict_json(f: Features):
    vals = [f.heart_rate, f.sleep_hours, f.activity_level, f.temperature]
    proba = predict_proba_ordered(vals)
    pred = int(proba >= THRESHOLD)

    if rule_safety_net(f):
        pred = 1  # override

    advice = get_advice(f)
    return {"anomaly": pred, "probability": round(proba, 4), "threshold": THRESHOLD, "advice": advice}

from nlp_advice.ai_advice import generate_dynamic_advice  # at top if not already

@app.get("/predict_rich")
def predict_rich(
    features: str = Query(..., description="Comma-separated: heart_rate,sleep_hours,activity_level,temperature"),
    t: float = 0.9,           # temperature
    p: float = 0.92,          # top_p
    j: int = 0                # jitter
):
    vals = [float(x) for x in features.split(",")]
    f = Features(heart_rate=vals[0], sleep_hours=vals[1], activity_level=vals[2], temperature=vals[3])

    proba = predict_proba(vals)
    pred = int(proba >= THRESHOLD)
    if rule_safety_net(f):
        pred = 1

    causes = derive_causes(f)
    topf = shap_top_factors(vals)
    advice, used_fallback = generate_dynamic_advice(f.heart_rate, f.sleep_hours, f.activity_level, f.temperature, pred, t, p, j)

    return {
        "anomaly": pred,
        "probability": round(proba, 4),
        "threshold": THRESHOLD,
        "causes": causes,
        "top_factors": topf,
        "advice": advice,
        "advice_fallback": used_fallback,
        "temperature": t,
        "top_p": p,
        "jitter": j
    }
