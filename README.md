# Smart Health Monitoring System

A connected IoT-based health monitoring system that continuously tracks vital parameters (e.g., heart rate, body temperature, ambient conditions) and sends data to the cloud for remote monitoring and alerting. This repository contains hardware schematics/notes, microcontroller firmware, cloud integration code and a simple dashboard example.

---

## Overview
This project demonstrates an end-to-end Smart Health Monitoring System:
- Hardware reads vitals (pulse, body temp, optional SpO₂/ECG) using sensors attached to a microcontroller.
- Data is packaged and sent to a cloud endpoint (HTTP/MQTT).
- Cloud stores timeseries data, runs threshold checks and sends alerts.
- A web/mobile dashboard displays live values and historical trends.

---

## How It Works (Step-by-step)

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

---

## OUTPUT - How It Works (Step-by-step)

### 1. Sensor & Hardware Layer
- Microcontroller (e.g., Arduino, ESP32) reads sensors at regular intervals:
  - Pulse / heart-rate sensor (e.g., MAX30100/30102, pulse sensor)
  - Body temperature (e.g., DS18B20, LM35)
  - Ambient temp & humidity (DHT11 / DHT22 / BME280)
- Microcontroller formats readings into JSON and sends via Wi-Fi (ESP modules) or GSM (SIM module).

### 2. Data Transmission & Cloud Layer
- Device posts JSON payloads to a cloud API or publishes to an MQTT broker:
```json
{
  "device_id": "device_001",
  "timestamp": "2025-11-07T09:30:00Z",
  "heart_rate": 78,
  "body_temp": 36.7,
  "ambient_temp": 24.1,
  "humidity": 48
}

Heart Rate: 85 bpm
Body Temperature: 36.9 °C
Humidity: 48 %
Sending data to cloud...
Data uploaded successfully!

Heart Rate: 120 bpm ⚠️ (High)
Body Temperature: 36.8 °C ✅
Humidity: 46% ✅
Alert triggered: High heart rate detected at 10:02 AM

Subject: ⚠️ Health Alert - High Heart Rate Detected
Message: Patient Device #001 - Heart rate recorded at 120 bpm (10:02 AM).
Recommended: Immediate attention.

---
---
## Alert Notifications

When abnormal readings occur:

The cloud function sends an alert notification via email or SMS.

The dashboard highlights the parameter in red.

Alert messages like:

⚠️ ALERT: Patient heart rate exceeded 120 bpm at 10:02 AM.


The caregiver/doctor can log in to check detailed data.

