# SmartFlood Mumbai — Bayesian AI Flood Prediction System

An end-to-end flood risk system for Mumbai with a FastAPI backend, AI/ML models, real-time data services, and a React-based interactive dashboard.

## 🌊 Overview

SmartFlood combines:

- **Probabilistic AI** (Random Forest, Bayesian model, ward clustering)
- **Real-time data** (weather + tide, with graceful fallbacks)
- **Interactive UI** (Mumbai ward map, predictions, routing)

## 🚀 Key Features

- **Ward-level predictions** with probabilities and confidence
- **Live weather integration** via OpenWeather (optional API key)
- **Routing that avoids high-risk areas** (A\* over OSMnx road graph)
- **Batch predictions** for all wards
- **Clean React frontend** with a global glassmorphic preloader

## 🏗️ Architecture

- **Backend**: FastAPI + Uvicorn, models in `Backend/models`, services in `Backend/services`
- **Frontend**: React + Leaflet (via `react-leaflet`), proxy to FastAPI during dev

## 📁 Project Structure

```
SmartFlood-Mumbai-Bayesian-AI-main/
├── Backend/
│   ├── api/
│   │   └── flood_prediction_api.py     # FastAPI app + all endpoints
│   ├── services/
│   │   ├── routing_service.py          # Flood-aware A* routing over road graph
│   │   └── weather_service_fixed.py    # Weather + tide service with fallbacks
│   ├── models/
│   │   ├── train_models.py             # Model training/initialization helpers
│   │   ├── flood_prediction_models.py  # Base model defs
│   │   ├── trained/                    # Saved models (PKL/CSV)
│   │   └── mumbai_drive.graphml        # Cached road graph (auto-built)
│   ├── Dataset/
│   │   ├── enriched_flood_dataset.csv
│   │   └── mumbai-wards-cleaned.geojson
│   └── run_api.py                      # API server runner (port 8000)
├── flood-frontend/
│   ├── public/
│   │   └── mumbai-wards-cleaned.geojson
│   └── src/
│       ├── App.js, Dashboard.js, MapComponent.js, api.js, ...
│       ├── Preloader.js / Preloader.css      # Global preloader
│       └── RoutingMap.jsx / RoutingMap.css   # Routing UI
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🧰 Prerequisites

- Python 3.9+
- Node.js 18+
- npm (or yarn)

Note: Geo/graph stack (geopandas, shapely, rtree, osmnx) is included in `requirements.txt`. On Windows, ensure you have build tools or prebuilt wheels available.

## ⚙️ Backend Setup (FastAPI)

1. Create and activate a virtual env (recommended)

```
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Optional: Configure environment variables (create `Backend/.env`)

- OPENWEATHER_API_KEY=<your_key> (enables live weather)
- API_WORKERS=2 (default 2; set 1 if using reload)
- API_RELOAD=1 (dev hot-reload; forces workers=1)
- ROUTING_API_BASE=http://127.0.0.1:8000 (routing service self-calls)

4. Start the API

```
python Backend/run_api.py
```

- API: http://localhost:8000
- Docs: http://localhost:8000/docs

First start may train/load models and build the road graph (if missing). The graph build is cached at `Backend/models/mumbai_drive.graphml` and can take several minutes initially.

## 💻 Frontend Setup (React)

1. Install dependencies

```
cd flood-frontend
npm install
```

2. Start the dev server

```
npm start
```

- App: http://localhost:3000

The frontend proxy (package.json) points to `http://127.0.0.1:8000`. You can override with `REACT_APP_API_BASE_URL`.

## 🔌 API Endpoints (summary)

- Health & info
  - GET `/` — API status summary
  - GET `/health` — Detailed health (models/services)
  - GET `/models/info` — Model configuration and dataset info
- Predictions
  - POST `/predict/ward/{ward}` — Predict for a specific ward (uses current weather)
  - GET `/predict/all-wards` — Batch prediction for all wards
  - POST `/predict/custom` — Predict with custom weather payload
- Weather
  - GET `/weather/current/{ward}` — Current weather for a ward
- Clustering
  - GET `/wards/clusters` — Ward clustering + summary
- Routing
  - POST `/route` — Flood-aware route between two coordinates
  - POST `/route/demo` — Demo scenarios (e.g., `central_flood`)
  - GET `/graph/refresh` — Rebuild/refresh the cached road graph

Example ward prediction (Python):

```python
import requests
resp = requests.post("http://localhost:8000/predict/ward/H/E")
print(resp.json())
```

## 🧠 Models

- Random Forest + Bayesian probability + Ward clustering (K-means)
- Uses `Backend/Dataset/enriched_flood_dataset.csv`
- Trained artifacts saved to `Backend/models/trained/`
- Fallback predictions are returned if models aren’t ready

## 🗺️ Routing Notes

- Builds a drivable OSM graph clipped to Mumbai wards
- Edge weights include flood risk: `length * (1 + alpha * risk)`
- Edges in high-risk wards (>= avoid_threshold) are removed
- First route may take longer due to model/graph warm-up

## 🧪 Quick Start

```
# Backend
pip install -r requirements.txt
python Backend/run_api.py

# Frontend
cd flood-frontend
npm install
npm start
```

## 🤝 Contributing

- Fork → Branch → Commit → PR

## 📝 License

MIT — see LICENSE if present.

## 👤 Author

- Aditya Kate — https://github.com/adityajkate

## 👥 Contributors

- Tanmay Harmalkar - https://github.com/Tanmay-25032006
- Suman Manik - https://github.com/SumanManik

## 🙏 Acknowledgments

- OpenWeather, Mumbai ward GeoJSON sources
- OSMnx, GeoPandas, Shapely, NetworkX

---

Need a lighter or branded preloader variant? Share colors and I’ll update quickly.
