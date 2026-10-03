# 🚗 Used Car Price Predictor & Valuation Platform

A full-stack machine learning application and production microservice architecture that predicts the resale market value of used cars in India with high precision and sub-10ms inference latency.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19.0-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.4-38B2AC?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![CatBoost](https://img.shields.io/badge/CatBoost-Regressor-FFCC00?logo=yandex&logoColor=black)](https://catboost.ai/)
[![Docker](https://img.shields.io/badge/Docker-Multi--stage-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Render](https://img.shields.io/badge/Render-Deployment-46E3B7?logo=render&logoColor=white)](https://render.com/)

---

## 👥 The Team & Contributions

* **[Nishita1652](https://github.com/Nishita1652)**: Machine Learning Engineering, CatBoost Model Training & Tuning, FastAPI Microservice Architecture, Containerization & Render Infrastructure.
* **[salmon6934](https://github.com/salmon6934)**: Frontend Development, UI/UX Design, and Initial API Integration.

---

## 🚀 Key Highlights & Performance

* **Machine Learning Model**: CatBoost Regressor trained on 12,000+ cleaned vehicle records from CarDekho (12,328 train split of 15,411 total records).
* **High Predictive Accuracy**: Achieved test **$R^2 = 0.9367$** using target log-transformation ($\ln(1 + y)$) and tuned tree ensembles.
* **Ultra-Low Latency**: FastAPI backend delivers **sub-10ms response times** (empirical average ~4.38ms), over 20x faster than the 100ms SLA.
* **Modern Reactive Interface**: Built with React 19, Vite, and Tailwind CSS featuring dark mode glassmorphism, instant vehicle presets, dynamic currency formatting (₹ Lakhs & INR), and real-time latency diagnostics.
* **Production Containerization**: Multi-stage Docker builds with Nginx web server, dynamic `$PORT` routing, and Infrastructure-as-Code via `render.yaml`.

---

## 🛠️ Tech Stack

* **Backend / API**: Python 3.10+, FastAPI, Uvicorn, Pydantic v2
* **Machine Learning**: CatBoost, Scikit-Learn, Pandas, NumPy, Joblib
* **Frontend**: React 19, Vite, Tailwind CSS, Lucide Icons
* **DevOps & Cloud**: Docker (multi-stage), Nginx Alpine, Render Cloud Blueprint (`render.yaml`)

---

## 📂 Project Structure

```text
car_prediction_model/
├── car-api-deploy/                 # FastAPI ML Microservice
│   ├── Dockerfile                  # Container definition for Python/FastAPI
│   ├── api_server.py               # REST API endpoints (/predict/, /selling_price, /model-info)
│   ├── best_car_price_model.pkl    # Serialized CatBoost regression model (9.5MB)
│   └── requirements.txt            # Python dependencies
├── frontend/                       # Modern Containerized React Frontend
│   ├── Dockerfile                  # Multi-stage build (Node -> Nginx Alpine)
│   ├── nginx.conf.template         # Nginx dynamic port template
│   ├── package.json                # Dependencies (React 19, Tailwind CSS, Lucide)
│   ├── tailwind.config.js          # Tailwind theme and styling tokens
│   ├── vite.config.js              # Vite bundler configuration
│   └── src/                        # React components and valuation engine
├── render.yaml                     # Render Infrastructure-as-Code Blueprint
├── cardekho_dataset.csv            # Raw dataset
├── cleaned_cardekho_dataset.csv    # Preprocessed dataset (15,411 rows)
├── Dockerfile                      # Root Dockerfile for direct container deploys
├── api_server.py                   # Root FastAPI entrypoint
└── requirements.txt                # Root requirements
```

---

## 🧠 Machine Learning Pipeline

1. **Data Preprocessing & Cleaning**:
   - Extracted numeric engine displacement (`engine_cleaned`), fuel efficiency (`mileage_cleaned`), and horsepower (`max_power_cleaned`).
   - Derived `vehicle_age` from registration years and normalized transmission & seller types.
2. **Target Normalization**:
   - Log-transformed selling price using $y_{log} = \ln(1 + \text{price})$ to stabilize variance and handle right-skewed pricing distributions.
3. **Model Selection**:
   - Evaluated Linear Regression, Random Forest, XGBoost, and CatBoost.
   - Selected CatBoost Regressor for native categorical feature support and superior generalization ($R^2 = 0.9367$).

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Microservice health check |
| `GET` | `/model-info` | Returns model metadata, $R^2$ metrics, and latency SLA |
| `POST` | `/predict/` | Main valuation endpoint returning `predicted_price_inr` |
| `POST` | `/selling_price` | Compatibility alias for `/predict/` |
| `GET` | `/docs` | Interactive Swagger API documentation |

### Example Request Body:
```json
{
  "brand": "Maruti",
  "model": "Swift Dzire",
  "vehicle_age": 5,
  "km_driven": 65000,
  "seller_type": "Individual",
  "fuel_type": "Diesel",
  "transmission_type": "Manual",
  "mileage_cleaned": 22.3,
  "engine_cleaned": 1248.0,
  "max_power_cleaned": 88.7,
  "seats": 5
}
```

### Example Response:
```json
{
  "predicted_price_inr": 624311.5,
  "model_input": { ... }
}
```

---

## 💻 Local Development Setup

### 1. Run the FastAPI Backend
```bash
cd car-api-deploy
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python api_server.py
# Server runs at http://127.0.0.1:8000 (Swagger docs at http://127.0.0.1:8000/docs)
```

### 2. Run the React Frontend
```bash
cd frontend
npm install
npm run dev
# Frontend runs at http://127.0.0.1:5173
```

---

## 🚀 Deployment on Render

This repository includes a ready-to-use [`render.yaml`](render.yaml) Blueprint:

1. Connect this GitHub repository to your [Render Dashboard](https://dashboard.render.com).
2. Click **Blueprints** -> **New Blueprint Instance**.
3. Select this repository; Render will automatically detect `render.yaml` and provision:
   - `car-price-api` (FastAPI Docker service)
   - `car-price-frontend` (Nginx + React SPA Docker service with automatic backend host wiring)
