from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import json
import os
import sys
from datetime import datetime
import asyncio

# Routing imports
try:
    from services.routing_service import GRAPH_MANAGER, RouteParams
except Exception as e:
    GRAPH_MANAGER = None
    RouteParams = None

# Add project root and models directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))  # Backend/api
project_root = os.path.dirname(current_dir)  # Backend
backend_dir = project_root  # Backend
models_dir = os.path.join(backend_dir, "models")
services_dir = os.path.join(backend_dir, "services")

# Add directories to sys.path for imports
for path in [project_root, backend_dir, models_dir, services_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Now use absolute imports
try:
    from models.train_models import (
        RealTimeFloodPredictionSystem,
        OptimizedRandomForestPredictor,
        OptimizedWardClusteringModel,
        OptimizedBayesianModel,
    )
except ImportError:
    try:
        from models.train_models import (
            RealTimeFloodPredictionSystem,
            OptimizedRandomForestPredictor,
            OptimizedWardClusteringModel,
            OptimizedBayesianModel,
        )
    except ImportError:
        from Backend.models.train_models import (
            RealTimeFloodPredictionSystem,
            OptimizedRandomForestPredictor,
            OptimizedWardClusteringModel,
            OptimizedBayesianModel,
        )

try:
    from services.weather_service_fixed import RealTimeDataService
except ImportError:
    try:
        from services.weather_service_fixed import RealTimeDataService
    except ImportError:
        from Backend.services.weather_service_fixed import RealTimeDataService

app = FastAPI(
    title="Mumbai Flood Prediction API",
    description="Real-time flood prediction using Random Forest, K-means Clustering, and Bayesian Networks",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Global variables for models (loaded once)
flood_system: Optional[RealTimeFloodPredictionSystem] = None
weather_service: Optional[RealTimeDataService] = None


# Define Pydantic models for request/response
class WeatherInput(BaseModel):
    rainfall_mm: float
    rainfall_24hr: float
    tide_level_m: float
    temperature_c: float
    humidity_percent: float
    wind_speed_kmh: float
    season: str


class FloodPredictionResponse(BaseModel):
    ward_code: str
    ward_name: Optional[str] = None
    ward_risk_zone: str
    random_forest: Dict[str, Any]
    bayesian_probability: float
    combined_assessment: Dict[str, Any]
    weather_data: Dict[str, Any]
    timestamp: str


class BatchPredictionResponse(BaseModel):
    predictions: List[FloodPredictionResponse]
    summary: Dict[str, Any]
    timestamp: str


# Routing request model
class RouteRequest(BaseModel):
    from_lat: float
    from_lng: float
    to_lat: float
    to_lng: float
    avoid_threshold: Optional[float] = 0.7
    alpha: Optional[float] = 10.0


def load_trained_models(flood_system: RealTimeFloodPredictionSystem, model_dir: str):
    """Load pre-trained models from disk"""
    import joblib
    import pandas as pd

    try:
        # Set the correct dataset path (relative to flood_prediction_api.py)
        dataset_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "Dataset",
            "enriched_flood_dataset.csv",
        )
        print(f"Attempting to load dataset from: {dataset_path}")
        print(f"Dataset exists: {os.path.exists(dataset_path)}")

        # Ensure BaseFloodModel uses the correct dataset path
        if hasattr(flood_system.base_model, "dataset_path"):
            flood_system.base_model.dataset_path = dataset_path
        else:
            print(
                "Warning: BaseFloodModel does not have dataset_path attribute. Update train_models.py if needed."
            )

        # Load base data
        flood_system.base_model.load_and_preprocess_data()

        # Initialize model components
        flood_system.rf_model = OptimizedRandomForestPredictor(flood_system.base_model)
        flood_system.clustering_model = OptimizedWardClusteringModel(
            flood_system.base_model
        )
        flood_system.bayesian_model = OptimizedBayesianModel(flood_system.base_model)

        # Load Random Forest model
        rf_model_path = os.path.join(model_dir, "random_forest_model.pkl")
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        encoders_path = os.path.join(model_dir, "label_encoders.pkl")

        if os.path.exists(rf_model_path):
            flood_system.rf_model.model = joblib.load(rf_model_path)
        if os.path.exists(scaler_path):
            flood_system.base_model.scaler = joblib.load(scaler_path)
        if os.path.exists(encoders_path):
            flood_system.base_model.label_encoders = joblib.load(encoders_path)

        # Load clustering model
        kmeans_path = os.path.join(model_dir, "kmeans_model.pkl")
        clusters_path = os.path.join(model_dir, "ward_clusters.csv")

        if os.path.exists(kmeans_path):
            flood_system.clustering_model.model = joblib.load(kmeans_path)
        if os.path.exists(clusters_path):
            flood_system.clustering_model.ward_features = pd.read_csv(clusters_path)

        # Load Bayesian model if available
        bayesian_model_path = os.path.join(model_dir, "bayesian_model.pkl")
        if os.path.exists(bayesian_model_path):
            try:
                flood_system.bayesian_model.model = joblib.load(bayesian_model_path)
                if flood_system.bayesian_model.model is not None:
                    from pgmpy.inference import VariableElimination

                    flood_system.bayesian_model.inference = VariableElimination(
                        flood_system.bayesian_model.model
                    )
            except Exception as e:
                print(f"Could not load Bayesian model: {e}")
                flood_system.bayesian_model.model = None

        # Set flag to indicate models are loaded
        flood_system.base_model.models_loaded = True
        print("Pre-trained models loaded successfully!")

    except Exception as e:
        print(f"Error loading models: {e}")
        if flood_system and flood_system.base_model:
            flood_system.base_model.models_loaded = False


@app.on_event("startup")
async def startup_event():
    """Initialize models and services on startup"""
    global flood_system, weather_service

    try:
        print("Initializing flood prediction system...")
        flood_system = RealTimeFloodPredictionSystem(use_pretrained=True)

        # Print paths for debugging
        dataset_path = os.path.join(
            backend_dir, "Dataset", "enriched_flood_dataset.csv"
        )
        print(f"Expected dataset path: {dataset_path}")
        print(f"Dataset exists: {os.path.exists(dataset_path)}")

        # Initialize async
        if hasattr(flood_system, "initialize_async"):
            await flood_system.initialize_async()

        # Check if models are already trained
        model_dir = os.path.join(backend_dir, "models", "trained")
        os.makedirs(model_dir, exist_ok=True)
        print(f"Model directory: {model_dir}")

        rf_model_path = os.path.join(model_dir, "random_forest_model.pkl")
        if os.path.exists(rf_model_path):
            print("Loading pre-trained models...")
            load_trained_models(flood_system, model_dir)
        else:
            print("Training models for the first time...")
            try:
                flood_system.train_all_models()
                flood_system.save_models()
                print("Models trained and saved successfully!")
            except Exception as e:
                print(f"Error training models: {e}")
                if flood_system and flood_system.base_model:
                    flood_system.base_model.models_loaded = False

        print("Flood prediction system initialization completed!")

    except Exception as e:
        print(f"Error loading flood models: {e}")
        flood_system = None

    # Initialize weather service separately
    try:
        print("Initializing weather service...")
        weather_service = RealTimeDataService()
        print("Weather service initialized successfully!")
    except Exception as e:
        print(f"Error initializing weather service: {e}")
        print("Weather service will use fallback data")
        weather_service = None

    # Pre-warm routing graph to avoid delays on first routing request
    if GRAPH_MANAGER is not None:
        try:
            print("Pre-warming routing graph (this may take a moment)...")
            await asyncio.to_thread(GRAPH_MANAGER.ensure_graph)
            print("Routing graph pre-warmed and ready!")
        except Exception as e:
            print(f"Warning: Could not pre-warm routing graph: {e}")
            print("Routing will initialize on first request instead.")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Mumbai Flood Prediction API",
        "status": "running",
        "models_loaded": flood_system is not None
        and hasattr(flood_system, "base_model")
        and hasattr(flood_system.base_model, "models_loaded")
        and flood_system.base_model.models_loaded,
        "weather_service": weather_service is not None,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    routing_status = "not_installed"
    routing_error = None
    
    if GRAPH_MANAGER is not None:
        try:
            # Check if graph is loaded
            if GRAPH_MANAGER.G is not None:
                routing_status = "ready"
            else:
                routing_status = "available_not_initialized"
        except Exception as e:
            routing_status = "error"
            routing_error = str(e)
    
    return {
        "api_status": "healthy",
        "models": {
            "flood_system": flood_system is not None,
            "random_forest": (
                hasattr(flood_system, "rf_model") and flood_system.rf_model is not None
                if flood_system
                else False
            ),
            "clustering": (
                hasattr(flood_system, "clustering_model")
                and flood_system.clustering_model is not None
                if flood_system
                else False
            ),
            "bayesian": (
                hasattr(flood_system, "bayesian_model")
                and flood_system.bayesian_model is not None
                if flood_system
                else False
            ),
        },
        "services": {
            "weather_service": weather_service is not None,
            "routing_available": GRAPH_MANAGER is not None,
            "routing_status": routing_status,
            "routing_error": routing_error,
        },
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/predict/ward/{ward_code}", response_model=FloodPredictionResponse)
async def predict_flood_for_ward(ward_code: str):
    """Predict flood risk for a specific ward using real-time weather data"""
    try:
        # Get weather data
        if weather_service:
            try:
                weather_data = weather_service.get_complete_weather_data(
                    ward_code.upper()
                )
            except Exception as e:
                print(f"Weather service error: {e}")
                weather_data = get_fallback_weather_data(ward_code)
        else:
            weather_data = get_fallback_weather_data(ward_code)

        # Try to get prediction from models, fallback if any error
        try:
            if (
                flood_system
                and hasattr(flood_system, "base_model")
                and hasattr(flood_system.base_model, "models_loaded")
                and flood_system.base_model.models_loaded
                and hasattr(flood_system, "predict_flood_risk_async")
            ):
                prediction = await flood_system.predict_flood_risk_async(
                    weather_data, ward_code.upper()
                )
            else:
                prediction = get_fallback_prediction(weather_data, ward_code)
        except Exception as e:
            print(f"Model prediction error: {e}")
            prediction = get_fallback_prediction(weather_data, ward_code)

        return FloodPredictionResponse(
            ward_code=ward_code.upper(),
            ward_name=weather_data.get("ward_name", f"Ward {ward_code}"),
            ward_risk_zone=prediction.get("ward_risk_zone", "Medium Risk"),
            random_forest=prediction.get(
                "random_forest",
                {
                    "flood_risk_level": 0,
                    "will_flood": False,
                    "risk_probabilities": {"low": 0.7, "medium": 0.2, "high": 0.1},
                },
            ),
            bayesian_probability=prediction.get("bayesian_probability", 0.3),
            combined_assessment=prediction.get(
                "combined_assessment",
                {
                    "high_risk": False,
                    "confidence": "Medium",
                    "confidence_score": 0.5,
                },
            ),
            weather_data=weather_data,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


def get_fallback_weather_data(ward_code: str) -> Dict[str, Any]:
    """Generate fallback weather data when weather service is unavailable"""
    return {
        "Rainfall_mm": 5.0,
        "Temperature_C": 28.0,
        "Humidity_%": 80,
        "Wind_Speed_kmh": 15.0,
        "season": "Monsoon",
        "weather_description": "Fallback weather data",
        "timestamp": datetime.now().isoformat(),
        "Rainfall_24hr": 10.0,
        "Tide_Level_m": 2.0,
        "High_Tide_m": 4.2,
        "Low_Tide_m": 0.6,
        "ward_code": ward_code.upper(),
        "ward_name": f"Ward {ward_code}",
        "coordinates": {"lat": 19.0760, "lon": 72.8777},
    }


def get_fallback_prediction(
    weather_data: Dict[str, Any], ward_code: str
) -> Dict[str, Any]:
    """Generate fallback prediction when models are not loaded"""
    rainfall = weather_data.get("Rainfall_mm", 0)
    rainfall_24hr = weather_data.get("Rainfall_24hr", 0)
    tide_level = weather_data.get("Tide_Level_m", 2.0)

    # Start with baseline probability of 10-12%
    risk_score = 0.12
    
    if rainfall > 50:
        risk_score += 0.22
    elif rainfall > 20:
        risk_score += 0.12

    if rainfall_24hr > 100:
        risk_score += 0.20
    elif rainfall_24hr > 50:
        risk_score += 0.08

    if tide_level > 3.5:
        risk_score += 0.15
    elif tide_level > 2.5:
        risk_score += 0.06

    # Cap at 1.0
    risk_score = min(risk_score, 1.0)

    will_flood = risk_score > 0.6
    flood_risk_level = 2 if risk_score > 0.65 else (1 if risk_score > 0.35 else 0)

    return {
        "ward_risk_zone": (
            "High Risk"
            if risk_score > 0.6
            else ("Medium Risk" if risk_score > 0.30 else "Low Risk")
        ),
        "random_forest": {
            "flood_risk_level": flood_risk_level,
            "will_flood": will_flood,
            "risk_probabilities": {
                "low": max(0, 1 - risk_score - 0.15),
                "medium": 0.20,
                "high": min(1, risk_score),
            },
        },
        "bayesian_probability": risk_score,
        "combined_assessment": {
            "high_risk": will_flood,
            "confidence": "Low (Fallback Mode)",
            "confidence_score": 0.3,
        },
    }


@app.post("/predict/custom", response_model=FloodPredictionResponse)
async def predict_flood_custom_weather(ward_code: str, weather_input: WeatherInput):
    """Predict flood risk using custom weather data"""
    if not flood_system:
        raise HTTPException(status_code=503, detail="Flood system not initialized")

    try:
        weather_data = {
            "Rainfall_mm": weather_input.rainfall_mm,
            "Rainfall_24hr": weather_input.rainfall_24hr,
            "Tide_Level_m": weather_input.tide_level_m,
            "Temperature_C": weather_input.temperature_c,
            "Humidity_%": weather_input.humidity_percent,
            "Wind_Speed_kmh": weather_input.wind_speed_kmh,
            "season": weather_input.season,
            "ward_code": ward_code.upper(),
            "ward_name": f"Ward {ward_code}",
            "timestamp": datetime.now().isoformat(),
        }

        models_loaded = (
            flood_system is not None
            and hasattr(flood_system, "base_model")
            and hasattr(flood_system.base_model, "models_loaded")
            and flood_system.base_model.models_loaded
        )

        if models_loaded and hasattr(flood_system, "predict_flood_risk_async"):
            prediction = await flood_system.predict_flood_risk_async(
                weather_data, ward_code.upper()
            )
        else:
            prediction = get_fallback_prediction(weather_data, ward_code)

        return FloodPredictionResponse(
            ward_code=ward_code.upper(),
            ward_name=f"Ward {ward_code}",
            ward_risk_zone=prediction["ward_risk_zone"],
            random_forest=prediction["random_forest"],
            bayesian_probability=prediction["bayesian_probability"],
            combined_assessment=prediction["combined_assessment"],
            weather_data=weather_data,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/predict/all-wards", response_model=BatchPredictionResponse)
async def predict_all_wards():
    """Get flood predictions for all Mumbai wards"""

    try:
        if weather_service:
            try:
                all_weather_data = weather_service.get_weather_for_all_wards()
            except Exception as e:
                print(f"Error getting weather data: {e}")
                all_weather_data = {}
                # Use actual Mumbai ward codes
                mumbai_wards = [
                    "A",
                    "B",
                    "C",
                    "D",
                    "E",
                    "F/N",
                    "F/S",
                    "G/N",
                    "G/S",
                    "H/E",
                    "H/W",
                    "K/E",
                    "K/W",
                    "L",
                    "M/E",
                    "M/W",
                    "N",
                    "P/N",
                    "P/S",
                    "R/C",
                    "R/N",
                    "R/S",
                    "S",
                    "T",
                ]
                for ward_code in mumbai_wards:
                    all_weather_data[ward_code] = get_fallback_weather_data(ward_code)
        else:
            all_weather_data = {}
            # Use actual Mumbai ward codes
            mumbai_wards = [
                "A",
                "B",
                "C",
                "D",
                "E",
                "F/N",
                "F/S",
                "G/N",
                "G/S",
                "H/E",
                "H/W",
                "K/E",
                "K/W",
                "L",
                "M/E",
                "M/W",
                "N",
                "P/N",
                "P/S",
                "R/C",
                "R/N",
                "R/S",
                "S",
                "T",
            ]
            for ward_code in mumbai_wards:
                all_weather_data[ward_code] = get_fallback_weather_data(ward_code)

        predictions = []
        high_risk_count = 0

        models_loaded = (
            flood_system is not None
            and hasattr(flood_system, "base_model")
            and hasattr(flood_system.base_model, "models_loaded")
            and flood_system.base_model.models_loaded
        )

        for ward_code, weather_data in all_weather_data.items():
            try:
                # Try to get prediction from models, fallback if any error
                try:
                    if models_loaded and hasattr(
                        flood_system, "predict_flood_risk_async"
                    ):
                        prediction = await flood_system.predict_flood_risk_async(
                            weather_data, ward_code
                        )
                    else:
                        prediction = get_fallback_prediction(weather_data, ward_code)
                except Exception as e:
                    print(f"Model prediction error for ward {ward_code}: {e}")
                    prediction = get_fallback_prediction(weather_data, ward_code)

                pred_response = FloodPredictionResponse(
                    ward_code=ward_code,
                    ward_name=weather_data.get("ward_name", f"Ward {ward_code}"),
                    ward_risk_zone=prediction["ward_risk_zone"],
                    random_forest=prediction["random_forest"],
                    bayesian_probability=prediction["bayesian_probability"],
                    combined_assessment=prediction["combined_assessment"],
                    weather_data=weather_data,
                    timestamp=datetime.now().isoformat(),
                )

                predictions.append(pred_response)

                if prediction["combined_assessment"]["high_risk"]:
                    high_risk_count += 1

            except Exception as e:
                print(f"Error predicting for ward {ward_code}: {e}")
                continue

        summary = {
            "total_wards": len(predictions),
            "high_risk_wards": high_risk_count,
            "medium_risk_wards": len(
                [p for p in predictions if p.random_forest["flood_risk_level"] == 1]
            ),
            "low_risk_wards": len(
                [p for p in predictions if p.random_forest["flood_risk_level"] == 0]
            ),
            "average_bayesian_probability": (
                sum([p.bayesian_probability for p in predictions]) / len(predictions)
                if predictions
                else 0
            ),
            "models_loaded": models_loaded,
        }

        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.post("/route")
async def compute_route(req: RouteRequest):
    if GRAPH_MANAGER is None or RouteParams is None:
        raise HTTPException(status_code=503, detail="Routing service not initialized")
    try:
        params = RouteParams(
            avoid_threshold=(
                req.avoid_threshold if req.avoid_threshold is not None else 0.7
            ),
            alpha=req.alpha if req.alpha is not None else 10.0,
        )
        feature = GRAPH_MANAGER.astar_route(
            from_lat=req.from_lat,
            from_lng=req.from_lng,
            to_lat=req.to_lat,
            to_lng=req.to_lng,
            params=params,
            show_blocked_geoms=False,
            blocked_geom_limit=0,
        )
        return feature
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Routing error: {str(e)}")


class DemoRouteRequest(RouteRequest):
    scenario: Optional[str] = "central_flood"


@app.post("/route/demo")
async def compute_route_demo(req: DemoRouteRequest):
    if GRAPH_MANAGER is None or RouteParams is None:
        raise HTTPException(status_code=503, detail="Routing service not initialized")
    try:
        params = RouteParams(
            avoid_threshold=(
                req.avoid_threshold if req.avoid_threshold is not None else 0.7
            ),
            alpha=req.alpha if req.alpha is not None else 10.0,
        )
        feature = GRAPH_MANAGER.astar_route_demo(
            from_lat=req.from_lat,
            from_lng=req.from_lng,
            to_lat=req.to_lat,
            to_lng=req.to_lng,
            params=params,
            scenario=req.scenario or "central_flood",
            show_blocked_geoms=False,
            blocked_geom_limit=0,
        )
        return feature
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Routing demo error: {str(e)}")


@app.get("/graph/refresh")
async def refresh_graph():
    if GRAPH_MANAGER is None:
        raise HTTPException(status_code=503, detail="Routing service not initialized")
    try:
        result = GRAPH_MANAGER.refresh_graph()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph refresh error: {str(e)}")


@app.get("/wards/clusters")
async def get_ward_clusters():
    """Get ward clustering information"""

    try:
        if (
            flood_system
            and hasattr(flood_system, "clustering_model")
            and flood_system.clustering_model
        ):
            clusters = flood_system.clustering_model.get_all_ward_clusters()
            return {
                "ward_clusters": clusters.to_dict("records"),
                "cluster_summary": clusters.groupby("risk_zone").size().to_dict(),
                "timestamp": datetime.now().isoformat(),
            }
        else:
            raise Exception("Clustering model not available")

    except Exception as e:
        # Use actual Mumbai ward codes for fallback
        mumbai_wards = [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F/N",
            "F/S",
            "G/N",
            "G/S",
            "H/E",
            "H/W",
            "K/E",
            "K/W",
            "L",
            "M/E",
            "M/W",
            "N",
            "P/N",
            "P/S",
            "R/C",
            "R/N",
            "R/S",
            "S",
            "T",
        ]

        fallback_clusters = []
        risk_zones = ["Very High Risk", "High Risk", "Medium Risk", "Low Risk"]

        for i, ward_code in enumerate(mumbai_wards):
            fallback_clusters.append(
                {
                    "Ward": ward_code,
                    "Ward_Name": f"Ward {ward_code}",
                    "ward_code": ward_code,
                    "risk_zone": risk_zones[i % len(risk_zones)],
                    "cluster": i % 4,
                }
            )

        return {
            "ward_clusters": fallback_clusters,
            "cluster_summary": {zone: 6 for zone in risk_zones},
            "timestamp": datetime.now().isoformat(),
            "note": "Fallback cluster data - clustering model not available",
        }


@app.get("/weather/current/{ward_code}")
async def get_current_weather(ward_code: str):
    """Get current weather data for a ward"""
    try:
        if weather_service:
            weather_data = weather_service.get_complete_weather_data(ward_code.upper())
        else:
            weather_data = get_fallback_weather_data(ward_code.upper())

        return weather_data

    except Exception as e:
        return get_fallback_weather_data(ward_code.upper())


@app.post("/models/retrain")
async def retrain_models(background_tasks: BackgroundTasks):
    """Retrain all models (background task)"""

    def retrain():
        global flood_system
        try:
            print("Starting model retraining...")
            flood_system = RealTimeFloodPredictionSystem()
            flood_system.train_all_models()
            flood_system.save_models()
            print("Model retraining completed!")
        except Exception as e:
            print(f"Error during retraining: {e}")

    background_tasks.add_task(retrain)
    return {"message": "Model retraining started in background"}


@app.get("/models/info")
async def get_model_info():
    """Get information about the trained models"""
    if not flood_system:
        raise HTTPException(status_code=503, detail="Flood system not initialized")

    models_loaded = (
        hasattr(flood_system, "base_model")
        and hasattr(flood_system.base_model, "models_loaded")
        and flood_system.base_model.models_loaded
    )

    try:
        info = {
            "models_loaded": models_loaded,
            "random_forest": {
                "n_estimators": 50,
                "max_depth": 8,
                "features": (
                    flood_system.base_model.feature_columns
                    if models_loaded
                    and hasattr(flood_system.base_model, "feature_columns")
                    else [
                        "Rainfall_mm",
                        "Tide_Level_m",
                        "Temperature_C",
                        "Humidity_%",
                        "Wind_Speed_kmh",
                    ]
                ),
            },
            "clustering": {"n_clusters": 4, "algorithm": "K-means"},
            "bayesian_network": {
                "available": (
                    models_loaded
                    and hasattr(flood_system, "bayesian_model")
                    and hasattr(flood_system.bayesian_model, "model")
                    and flood_system.bayesian_model.model is not None
                ),
                "nodes": [
                    "Rainfall_Category",
                    "Tide_Category",
                    "Ward_Risk_Zone",
                    "Season",
                    "Flood",
                ],
            },
            "dataset_info": {
                "total_records": (
                    len(flood_system.base_model.data)
                    if models_loaded
                    and hasattr(flood_system.base_model, "data")
                    and flood_system.base_model.data is not None
                    else 0
                ),
                "features": (
                    list(flood_system.base_model.data.columns)
                    if models_loaded
                    and hasattr(flood_system.base_model, "data")
                    and flood_system.base_model.data is not None
                    else []
                ),
            },
            "fallback_mode": not models_loaded,
        }

        return info

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model info error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    print("Starting Mumbai Flood Prediction API...")
    print("API will be available at: http://localhost:8000")
    print("API documentation available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
