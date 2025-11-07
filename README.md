# Smart Health Monitoring System (Federated Learning + TinyML + XAI + Prompt Engineering)

This repository contains a complete, end-to-end final year project that showcases:
- **Federated Learning** (Flower + TensorFlow demo)
- **TinyML** (TensorFlow Lite conversion and edge-ready model)
- **Explainable AI (XAI)** (SHAP for RandomForest model interpretability)
- **Prompt Engineering** (Hugging Face `transformers` to generate personalized advice)
- **API** (FastAPI to serve predictions + advice)

## 1) Setup (Windows PowerShell / macOS / Linux)

```bash
python -m venv ai_project_env
# Windows
ai_project_env\Scripts\activate
# macOS/Linux
# source ai_project_env/bin/activate

pip install -r requirements.txt
```

> Tip: First-time use of `transformers` will download model weights (internet required).

## 2) Dataset

A synthetic dataset is included at `data/health_data.csv` with columns:  
`heart_rate, sleep_hours, activity_level, temperature, anomaly`.

## 3) Train the core ML model (RandomForest)

```bash
python scripts/train_model.py
```
- Saves: `models/anomaly_model.pkl`

## 4) Explainable AI (SHAP)

```bash
python explainable_ai/shap_explain.py
```
- Opens a SHAP summary plot showing feature importance for the RandomForest model.

## 5) Prompt Engineering demo

```bash
python nlp_advice/prompt_engineering.py
```

## 6) Serve API (FastAPI)

```bash
uvicorn api.main:app --reload
```
Open: `http://127.0.0.1:8000/docs` (Swagger UI)  
Try: `GET /predict?features=90,5,4,36.8` → `[heart_rate, sleep_hours, activity_level, temperature]`

## 7) TinyML (TensorFlow model + TFLite)

Train a lightweight TF model and export TFLite:
```bash
python tinyml/train_tf_model.py
python tinyml/convert_to_tflite.py
```
- Outputs: `tinyml/anomaly_model.tflite`

## 8) Federated Learning (Flower)

**Terminal 1 – Start server**
```bash
python federated/server.py
```

**Terminal 2/3 – Start two clients (in separate terminals)**
```bash
python federated/client.py --client_id 1
python federated/client.py --client_id 2
```
- The server aggregates client updates and prints metrics.

## 9) Docker (optional)

```dockerfile
# Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build & run:
```bash
docker build -t smart_health_app .
docker run -p 8000:8000 smart_health_app
```

## 10) Project Order (Recommended)

1. `python scripts/train_model.py`  
2. `python explainable_ai/shap_explain.py`  
3. `python nlp_advice/prompt_engineering.py`  
4. `uvicorn api.main:app --reload`  
5. `python tinyml/train_tf_model.py` → `python tinyml/convert_to_tflite.py`  
6. Federated Learning: `python federated/server.py` + two `python federated/client.py --client_id X`

## Viva / Resume Highlights
- Privacy-preserving training with FL (Flower)
- Edge-ready TFLite model (TinyML)
- Interpretable model with SHAP
- Prompt-engineered recommendations via LLM
- FastAPI microservice + (optional) Docker
