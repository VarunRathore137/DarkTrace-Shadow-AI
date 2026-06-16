# DarkTrace Shadow AI — Drug Transaction Detection System

<div align="center">
  <img src="assets/logo.png" alt="DarkTrace Shadow AI" height="180px" width="180px"/>
</div>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-FF6600?style=for-the-badge)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-blueviolet?style=for-the-badge)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_RAG-FF4081?style=for-the-badge)
![Build](https://img.shields.io/badge/build-passing-brightgreen?style=for-the-badge)

**AI-powered real-time detection of drug-related conversations on encrypted platforms.**  
XGBoost · SHAP Explainability · Semantic RAG Search · Live Network Graph · FastAPI + Vanilla JS UI

</div>

---

## 🌟 Overview

DarkTrace Shadow AI is an end-to-end machine learning pipeline designed to detect illicit drug transaction patterns in chat messages. The system combines a trained **XGBoost classifier** with **explainable AI (SHAP)**, **Retrieval-Augmented Generation (RAG)** for semantic case retrieval, and a live **D3.js network graph** — all served through a blazing-fast **FastAPI** backend and a premium dark-themed web UI.

> Trained exclusively on **synthetic data** — no real user messages were used. Built for research, law enforcement simulation, and cybersecurity education.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔍 **Live Message Scanner** | Real-time XGBoost classification with risk score gauge |
| 🧠 **SHAP Explainability** | Per-feature influence bars showing *why* a message was flagged |
| 📡 **Semantic Similar Messages** | ChromaDB RAG retrieves the most contextually similar past cases |
| 🕸️ **Network Graph** | D3.js ego-network visualization of suspect interaction clusters |
| 📊 **Live Dashboard** | Total analyses, risk split donut chart, API latency tracker |
| 🌐 **Batch Analysis** | Analyze up to 100 messages simultaneously |
| 💬 **Multi-Platform Support** | Handles Telegram, Discord, Reddit, WhatsApp context |
| ⚡ **High Performance** | Sub-100ms inference via FastAPI + pre-loaded model |

---

## 🛠 Tech Stack

### Backend
- **FastAPI** — Async REST API server
- **XGBoost** — Primary classifier (trained on 50,000+ synthetic messages)
- **scikit-learn** — Feature engineering pipeline & preprocessor
- **SHAP** — TreeExplainer for model explainability
- **ChromaDB** — Local vector database for semantic similarity search
- **SentenceTransformers** (`all-MiniLM-L6-v2`) — Text embeddings for RAG
- **NetworkX** — Graph analytics engine
- **Pandas / NumPy** — Data processing

### Frontend
- **Vanilla HTML/CSS/JavaScript** — Zero-framework, zero-dependency UI
- **D3.js** — Interactive network graph visualization
- **Chart.js** — Risk split donut chart & dashboard stats
- Served via Python's built-in `http.server`

---

## 🖥️ Screenshots

> The cyberpunk-themed UI was built entirely with Vanilla CSS — no frameworks.

*(Add screenshots here)*

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Git

### 1. Clone the repo
```bash
git clone https://github.com/VarunRathore137/darktrace-ai-.git
cd darktrace-ai-
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate datasets & ingest into ChromaDB
> These large files are excluded from git. Run once to populate locally.
```bash
python generate_dataset.py
python ingest.py
```

### 4. (Optional) Retrain the model
> Pre-trained model artifacts are included in `artifacts/`. Skip this if you just want to run.
```bash
python train_xgboost.py
```

### 5. Launch the app
**One command to start both servers:**
```bash
start_all.bat        # Windows
```
Or manually in two terminals:
```bash
# Terminal 1 — Backend (port 8000)
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2 — Frontend (port 8501)
cd frontend && python -m http.server 8501
```

Open **http://localhost:8501** in your browser.

---

## 📁 Project Structure

```
darktrace-ai-/
│
├── api/
│   └── main.py                     # FastAPI backend — all endpoints
│
├── frontend/
│   └── index.html                  # Complete UI (single-file, Vanilla JS)
│
├── artifacts/                      # Trained model files (not in git for large ones)
│   ├── xgboost_pipeline.joblib     ✅ Active classifier pipeline
│   ├── xgboost_preprocessor.joblib ✅ Feature preprocessor (TF-IDF + custom)
│   └── final_pipeline.joblib       (large, local only)
│
├── create_final_model.py           # Feature engineering logic (shared by API)
├── train_xgboost.py                # XGBoost training script
├── generate_dataset.py             # Synthetic dataset generator
├── ingest.py                       # ChromaDB ingestion script
├── locustfile.py                   # Load testing config
│
├── drug_emoji_dictionary.csv       # 🔑 Custom emoji → risk mapping
├── drug_slang_dictionary.csv       # 🔑 Custom slang → risk mapping
│
├── DATASET_DOCUMENTATION.pdf       # Dataset design & methodology
├── requirements.txt
├── Dockerfile                      # Backend container
├── Dockerfile.frontend             # Frontend container
├── docker-compose.yml              # Orchestrates both services
├── start_all.bat                   # Windows one-click launcher
└── README.md
```

---

## 🧠 How It Works

### Feature Engineering
Each incoming message is transformed into **30+ custom features** before being scored:

| Category | Features |
|---|---|
| **Emoji Analysis** | `emoji_count`, `drug_emoji_count`, `drug_emoji_ratio`, `has_drug_emoji` |
| **Slang Detection** | `slang_count`, `has_slang`, `slang_ratio` |
| **Text Metrics** | `char_count`, `word_count`, `avg_word_length`, `text_complexity` |
| **Keyword Signals** | `has_price`, `has_quantity`, `has_location`, `has_urgency`, `has_payment` |
| **Combo Signals** | `suspicious_combo_1..7` — high-confidence rule-based flags |
| **Risk Score** | `weighted_risk_score` — pre-classification heuristic |
| **TF-IDF** | Drug slang bigrams (420, molly, dm, selling weed, etc.) |

### Classification Pipeline
```
message text
    │
    ├─ TF-IDF vectorizer (drug slang bigrams)
    ├─ One-hot encoder (platform, message_type)
    └─ Custom feature extractor (30+ signals)
          │
          └─► XGBoost Classifier ──► Risk Score (0.0 – 1.0)
                    │
                    └─► SHAP TreeExplainer ──► Feature influence bars
```

### Risk Thresholds
| Score | Level | Label |
|---|---|---|
| ≥ 0.70 | 🔴 HIGH | Critical Intercept |
| 0.40 – 0.69 | 🟡 MEDIUM | Elevated Activity |
| < 0.40 | 🟢 LOW | Routine Traffic |

---

## 🌐 API Reference

All endpoints are served at `http://127.0.0.1:8000`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server health check |
| `POST` | `/predict?explain=true` | Classify a single message + SHAP values |
| `POST` | `/batch_predict` | Classify up to 100 messages |
| `POST` | `/semantic_search` | Find similar messages in ChromaDB |
| `POST` | `/network_analysis` | Generate ego-network graph data |
| `GET` | `/benchmark` | Throughput & latency benchmark |
| `GET` | `/docs` | Interactive Swagger UI |

### Example: Single Prediction
```bash
curl -X POST "http://127.0.0.1:8000/predict?explain=true" \
  -H "Content-Type: application/json" \
  -d '{"message": "yo got fire 💊 dm me $30 cash only asap"}'
```

**Response:**
```json
{
  "prediction": 0.9995,
  "risk_level": "High",
  "features": {
    "drug_emoji_count": 1,
    "slang_count": 1,
    "has_price": true,
    "has_urgency": true,
    "weighted_risk_score": 20
  },
  "shap_values": [...],
  "shap_base_value": 0.14
}
```

---

## 🐳 Docker Deployment

```bash
docker-compose up --build
```

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:8501`

---

## 📊 Performance

| Metric | Value |
|---|---|
| Model Accuracy | **~93%** on synthetic test set |
| Inference Latency | **< 50ms** per message (FastAPI) |
| Feature Dimensions | **130+** (TF-IDF + custom + one-hot) |
| Training Data | **50,000+** synthetic messages |
| Platforms Covered | Telegram, Discord, Reddit, WhatsApp |

---

## 🧪 Testing

```bash
# Unit tests
pytest tests/

# Load testing (requires locust)
locust -f locustfile.py --host http://127.0.0.1:8000
```

---

## ⚖️ Ethics & Compliance

- **No real user data** was collected or used at any stage.
- All training data is **synthetically generated** using rule-based patterns.
- Designed for **law enforcement simulation**, **research**, and **cybersecurity education** only.
- Not intended for production surveillance without appropriate legal oversight.

---

## 🔮 Future Enhancements

- [ ] Transformer-based text encoder (fine-tuned BERT)
- [ ] Real-time WebSocket streaming scanner
- [ ] Admin authentication layer
- [ ] Multi-language slang support
- [ ] Export flagged messages to PDF report

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  Built with ❤️ for cybersecurity research<br/>
  <strong>DarkTrace Shadow AI</strong> — Illuminate the dark web
</div>
