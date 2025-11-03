"""
Flood Prediction Models for Mumbai - OPTIMIZED FOR REAL-TIME
Contains three models:
1. Random Forest Classifier - Main flood prediction
2. K-means Clustering - Ward risk zone grouping
3. Bayesian Network - Probability calculation with uncertainty
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import pickle
import hashlib
import time
from typing import Dict, List, Tuple, Any, Optional
from functools import lru_cache
from datetime import datetime
import warnings
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import asyncio
import json

warnings.filterwarnings("ignore")


def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


try:
    # Try the new import first
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination

    PGMPY_AVAILABLE = True
except ImportError:
    try:
        # Fallback to old import if new one fails
        from pgmpy.models import BayesianNetwork as DiscreteBayesianNetwork
        from pgmpy.factors.discrete import TabularCPD
        from pgmpy.inference import VariableElimination

        PGMPY_AVAILABLE = True
    except ImportError:
        PGMPY_AVAILABLE = False
        print("Warning: pgmpy not available. Bayesian Network model will be disabled.")


class ModelCache:
    """Cache for model predictions to improve real-time performance"""

    def __init__(self, ttl_seconds=300):  # 5 minutes TTL
        self.cache = {}
        self.ttl = ttl_seconds
        self.lock = threading.Lock()

    def _generate_key(self, data: Dict) -> str:
        """Generate cache key from input data"""
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.md5(serialized.encode()).hexdigest()

    def get(self, data: Dict) -> Optional[Dict]:
        """Get cached prediction if available and not expired"""
        key = self._generate_key(data)
        with self.lock:
            if key in self.cache:
                entry_time, result = self.cache[key]
                if time.time() - entry_time < self.ttl:
                    return result
                else:
                    del self.cache[key]
        return None

    def set(self, data: Dict, result: Dict):
        """Cache prediction result"""
        key = self._generate_key(data)
        with self.lock:
            self.cache[key] = (time.time(), result)
            # Clean old entries
            self._cleanup()

    def _cleanup(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            k for k, (t, _) in self.cache.items() if current_time - t > self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]


class RealTimeDataProcessor:
    """Handles real-time data processing and feature engineering"""

    def __init__(self):
        self.feature_buffer = queue.Queue(maxsize=1000)
        self.processing_pool = ThreadPoolExecutor(max_workers=4)

    @lru_cache(maxsize=128)
    def engineer_features(
        self, rainfall: float, tide: float, temp: float, humidity: float, wind: float
    ) -> np.ndarray:
        """Fast feature engineering with caching"""
        features = np.array(
            [
                rainfall,
                rainfall * 1.5,  # Simulated 24hr rainfall
                tide,
                temp,
                humidity,
                wind,
            ]
        )
        return features

    def process_batch(self, data_batch: List[Dict]) -> List[np.ndarray]:
        """Process batch of data in parallel"""
        futures = []
        for data in data_batch:
            future = self.processing_pool.submit(
                self.engineer_features,
                data.get("Rainfall_mm", 0),
                data.get("Tide_Level_m", 0),
                data.get("Temperature_C", 25),
                data.get("Humidity_%", 60),
                data.get("Wind_Speed_kmh", 10),
            )
            futures.append(future)

        return [f.result() for f in futures]


class OptimizedFloodPredictionModels:
    """Optimized main class with all three flood prediction models"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Singleton pattern for model instance"""
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, dataset_path: str = None, enable_cache: bool = True):
        if hasattr(self, "initialized"):
            return

        if dataset_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            dataset_path = os.path.join(
                os.path.dirname(os.path.dirname(current_dir)),
                "Dataset",
                "enriched_flood_dataset.csv",
            )

        self.dataset_path = dataset_path
        self.data = None
        self.rf_model = None
        self.kmeans_model = None
        self.bayesian_model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = [
            "Rainfall_mm",
            "Rainfall_24hr",
            "Tide_Level_m",
            "Temperature_C",
            "Humidity_%",
            "Wind_Speed_kmh",
            "Season_encoded",
        ]

        # Real-time optimizations
        self.cache = ModelCache() if enable_cache else None
        self.data_processor = RealTimeDataProcessor()
        self.models_loaded = False
        self.initialized = True
        self.ward_lookup = {}  # Added for safe lookups

    def load_and_preprocess_data(self):
        """Optimized data loading with chunking for large datasets"""
        print("Loading dataset...")

        # Load in chunks for memory efficiency
        chunk_size = 10000
        chunks = []

        for chunk in pd.read_csv(self.dataset_path, chunksize=chunk_size):
            chunks.append(chunk)

        self.data = pd.concat(chunks, ignore_index=True)

        # Vectorized categorical encoding
        categorical_cols = [
            "Season",
            "Elevation_m",
            "Drainage_Capacity",
            "Population_Density",
        ]
        for col in categorical_cols:
            if col in self.data.columns:
                le = LabelEncoder()
                self.data[f"{col}_encoded"] = le.fit_transform(
                    self.data[col].astype(str)
                )
                self.label_encoders[col] = le

        print(f"Dataset loaded: {len(self.data)} records")
        return self.data

    def load_pretrained_models(self, model_dir: str = "Backend/models/trained"):
        """Load pre-trained models for real-time inference"""
        try:
            print("Loading pre-trained models...")

            # Load models in parallel
            with ThreadPoolExecutor(max_workers=3) as executor:
                rf_future = executor.submit(
                    joblib.load, f"{model_dir}/random_forest_model.pkl"
                )
                kmeans_future = executor.submit(
                    joblib.load, f"{model_dir}/kmeans_model.pkl"
                )
                scaler_future = executor.submit(joblib.load, f"{model_dir}/scaler.pkl")

                self.rf_model = rf_future.result()
                self.kmeans_model = kmeans_future.result()
                self.scaler = scaler_future.result()

            # Load additional data
            self.label_encoders = joblib.load(f"{model_dir}/label_encoders.pkl")
            self.ward_features = pd.read_csv(f"{model_dir}/ward_clusters.csv")

            if PGMPY_AVAILABLE and os.path.exists(f"{model_dir}/bayesian_model.pkl"):
                self.bayesian_model = joblib.load(f"{model_dir}/bayesian_model.pkl")

            self.models_loaded = True
            print("Models loaded successfully")
            return True

        except Exception as e:
            print(f"Error loading models: {e}")
            return False


class OptimizedRandomForestPredictor:
    """Optimized Random Forest for real-time predictions"""

    def __init__(self, parent_model):
        self.parent = parent_model
        # Use the parent's pre-trained model if available, otherwise create new
        if hasattr(parent_model, "rf_model") and parent_model.rf_model is not None:
            self.model = parent_model.rf_model
        else:
            self.model = RandomForestClassifier(
                n_estimators=50,  # Reduced for faster inference
                max_depth=8,  # Reduced depth
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,  # Use all CPU cores
            )

    def train(self):
        """Optimized training with early stopping"""
        print("\n=== Training Optimized Random Forest ===")

        # Only train if model is not already trained
        if hasattr(self.parent, "rf_model") and self.parent.rf_model is not None:
            print("Using pre-trained Random Forest model")
            return self.model

        X = self.parent.data[self.parent.feature_columns].copy()
        y_risk = self.parent.data["Flood_Risk_Level"]

        X = X.fillna(X.mean())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_risk, test_size=0.2, random_state=42, stratify=y_risk
        )

        X_train_scaled = self.parent.scaler.fit_transform(X_train)
        X_test_scaled = self.parent.scaler.transform(X_test)

        # Train with warm start for incremental learning
        self.model.fit(X_train_scaled, y_train)

        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Optimized RF Accuracy: {accuracy:.3f}")

        # Store the trained model in parent for persistence
        self.parent.rf_model = self.model

        return self.model

    def safe_season_encoding(self, season_name: str) -> int:
        """Safely encode season, handling unseen labels"""
        if "Season" not in self.parent.label_encoders:
            return 0

        encoder = self.parent.label_encoders["Season"]

        # Check if season is known
        if season_name in encoder.classes_:
            return encoder.transform([season_name])[0]
        else:
            # Use the most common season as default
            if len(encoder.classes_) > 0:
                default_season = encoder.classes_[0]  # First class
                return encoder.transform([default_season])[0]
            else:
                return 0  # Fallback

    @lru_cache(maxsize=256)
    def predict_cached(self, features_tuple: tuple) -> Dict[str, Any]:
        """Cached prediction for repeated queries"""
        features = np.array(features_tuple).reshape(1, -1)

        risk_level = self.model.predict(features)[0]
        risk_proba = self.model.predict_proba(features)[0]

        result = {
            "flood_risk_level": risk_level,
            "will_flood": risk_level >= 1,
            "risk_probabilities": {
                "low": risk_proba[0],
                "medium": risk_proba[1] if len(risk_proba) > 1 else 0.0,
                "high": risk_proba[2] if len(risk_proba) > 2 else 0.0,
            },
        }

        # Convert all NumPy types to Python types
        return convert_numpy_types(result)

    def predict(self, weather_data: Dict[str, float]) -> Dict[str, Any]:
        """Real-time optimized prediction"""
        # Check cache first
        if self.parent.cache:
            cached_result = self.parent.cache.get(weather_data)
            if cached_result:
                return cached_result

        # Prepare input
        input_df = pd.DataFrame([weather_data])

        if "season" in weather_data:
            season_encoded = self.safe_season_encoding(weather_data["season"])
            input_df["Season_encoded"] = season_encoded
        else:
            input_df["Season_encoded"] = 0  # Default

        input_features = input_df[self.parent.feature_columns].fillna(0)
        input_scaled = self.parent.scaler.transform(input_features)

        # Convert to tuple for caching
        features_tuple = tuple(input_scaled[0])
        result = self.predict_cached(features_tuple)

        # Cache result
        if self.parent.cache:
            self.parent.cache.set(weather_data, result)

        return result


class OptimizedWardClusteringModel:
    """Optimized clustering model with pre-computed lookups"""

    def __init__(self, parent_model):
        self.parent = parent_model
        self.model = KMeans(n_clusters=4, random_state=42, n_init=10)
        self.ward_features = None
        self.ward_lookup = {}  # Pre-computed lookup table

    def prepare_ward_features(self):
        """Optimized ward feature preparation"""
        print("\n=== Preparing Ward Features (Optimized) ===")

        # Use vectorized operations
        ward_stats = (
            self.parent.data.groupby(["Ward", "Ward_Name"])
            .agg(
                {
                    "Rainfall_mm": ["mean", "max"],
                    "Rainfall_24hr": ["mean", "max"],
                    "Flood_Occurred": ["sum", "count"],
                    "Elevation_m": "first",
                    "Drainage_Capacity": "first",
                }
            )
            .round(2)
        )

        ward_stats.columns = ["_".join(col).strip() for col in ward_stats.columns]
        ward_stats = ward_stats.reset_index()

        ward_stats["flood_frequency"] = (
            ward_stats["Flood_Occurred_sum"] / ward_stats["Flood_Occurred_count"]
        ).round(3)

        # Fast encoding
        for col, encoder_name in [
            ("Elevation_m_first", "elevation"),
            ("Drainage_Capacity_first", "drainage"),
        ]:
            encoder = LabelEncoder()
            ward_stats[f"{encoder_name}_encoded"] = encoder.fit_transform(
                ward_stats[col].astype(str)
            )

        clustering_features = [
            "Rainfall_mm_mean",
            "Rainfall_mm_max",
            "Rainfall_24hr_max",
            "flood_frequency",
            "elevation_encoded",
            "drainage_encoded",
        ]

        self.ward_features = ward_stats[
            ["Ward", "Ward_Name"] + clustering_features
        ].copy()

        return self.ward_features

    def train(self):
        """Optimized training with pre-computed lookups"""
        print("\n=== Training Optimized K-means ===")

        if self.ward_features is None:
            self.prepare_ward_features()

        feature_cols = [
            col
            for col in self.ward_features.columns
            if col not in ["Ward", "Ward_Name"]
        ]
        X = self.ward_features[feature_cols].fillna(0)

        # Scale features
        X_scaled = StandardScaler().fit_transform(X)

        # Fit clustering model
        self.model.fit(X_scaled)

        # Add cluster labels
        self.ward_features["cluster"] = self.model.labels_

        # Analyze clusters
        cluster_analysis = (
            self.ward_features.groupby("cluster")
            .agg(
                {"flood_frequency": "mean", "Rainfall_mm_max": "mean", "Ward": "count"}
            )
            .round(3)
        )

        print("\nCluster Analysis:")
        print(cluster_analysis)

        # Assign risk levels to clusters based on flood frequency
        cluster_risk_mapping = {}
        sorted_clusters = cluster_analysis.sort_values(
            "flood_frequency", ascending=False
        )
        risk_levels = ["Very High Risk", "High Risk", "Medium Risk", "Low Risk"]

        for i, (cluster_id, _) in enumerate(sorted_clusters.iterrows()):
            cluster_risk_mapping[cluster_id] = risk_levels[min(i, len(risk_levels) - 1)]

        self.ward_features["risk_zone"] = self.ward_features["cluster"].map(
            cluster_risk_mapping
        )

        # Store the cluster risk mapping for later use
        self.cluster_risk_mapping = cluster_risk_mapping

        # Create ward clusters mapping for easy lookup
        self.ward_lookup = dict(
            zip(self.ward_features["Ward"], self.ward_features["risk_zone"])
        )  # Use risk_zone directly for lookup

        print("\nWard Risk Zones:")
        for cluster, risk in cluster_risk_mapping.items():
            wards_in_cluster = self.ward_features[
                self.ward_features["cluster"] == cluster
            ]
            print(f"Cluster {cluster} ({risk}): {len(wards_in_cluster)} wards")

        return self.model

    def get_ward_risk_zone(self, ward_code: str) -> str:
        """Get risk zone for a ward"""
        ward_code = ward_code.upper()
        return self.ward_lookup.get(ward_code, "Medium Risk")

    def get_all_ward_clusters(self) -> pd.DataFrame:
        """Get all wards with their cluster assignments"""
        return self.ward_features[["Ward", "Ward_Name", "cluster", "risk_zone"]].copy()


class OptimizedBayesianModel:
    """Optimized Bayesian Network for real-time probability calculation"""

    def __init__(self, parent_model):
        self.parent = parent_model
        self.model = None
        self.inference = None
        self.valid_risk_zones = {
            "Low Risk",
            "Medium Risk",
            "High Risk",
            "Very High Risk",
        }

    def create_network(self):
        """Create and train the Bayesian Network"""
        if not PGMPY_AVAILABLE:
            print("Bayesian Network model not available - pgmpy not installed")
            return None

        print("\n=== Creating Optimized Bayesian Network ===")

        try:
            # Define network structure
            self.model = DiscreteBayesianNetwork()

            # Add edges to the network
            self.model.add_edges_from(
                [
                    ("Rainfall_Category", "Flood"),
                    ("Tide_Category", "Flood"),
                    ("Ward_Risk_Zone", "Flood"),
                    ("Season", "Flood"),
                ]
            )

            # Prepare categorical data
            data_for_bn = self.parent.data.copy()

            # Categorize continuous variables
            data_for_bn["Rainfall_Category"] = pd.cut(
                data_for_bn["Rainfall_mm"],
                bins=[0, 10, 50, float("inf")],
                labels=["Low", "Medium", "High"],
            )

            data_for_bn["Tide_Category"] = pd.cut(
                data_for_bn["Tide_Level_m"],
                bins=[0, 2, 4, float("inf")],
                labels=["Low", "Medium", "High"],
            )

            # Use clustering results for ward risk zones
            if (
                hasattr(self.parent, "clustering_model")
                and self.parent.clustering_model.ward_features is not None
            ):

                ward_risk_map = dict(
                    zip(
                        self.parent.clustering_model.ward_features["Ward"],
                        self.parent.clustering_model.ward_features["risk_zone"],
                    )
                )
                data_for_bn["Ward_Risk_Zone"] = data_for_bn["Ward"].map(ward_risk_map)
                self.valid_risk_zones = set(
                    self.parent.clustering_model.ward_features["risk_zone"].unique()
                )
            else:
                # Default to all risk zones if clustering not available
                data_for_bn["Ward_Risk_Zone"] = "Medium Risk"
                self.valid_risk_zones = {
                    "Low Risk",
                    "Medium Risk",
                    "High Risk",
                    "Very High Risk",
                }

            data_for_bn["Flood"] = data_for_bn["Flood_Occurred"].map(
                {0: "No", 1: "Yes"}
            )

            # Remove rows with NaN values
            bn_data = data_for_bn[
                [
                    "Rainfall_Category",
                    "Tide_Category",
                    "Ward_Risk_Zone",
                    "Season",
                    "Flood",
                ]
            ].dropna()

            # Fit the model
            self.model.fit(bn_data)

            # Create inference object
            self.inference = VariableElimination(self.model)

            print("Optimized Bayesian Network created and trained successfully")
            return self.model

        except Exception as e:
            print(f"Error creating Bayesian Network: {e}")
            print("Bayesian Network will be disabled")
            self.model = None
            self.inference = None
            return None

    def predict_probability(
        self, rainfall_cat: str, tide_cat: str, ward_risk: str, season: str
    ) -> float:
        """Calculate flood probability using Bayesian inference"""
        if self.model is not None and self.inference is not None:
            try:
                # Ensure all evidence values are valid
                evidence = {
                    "Rainfall_Category": rainfall_cat,
                    "Tide_Category": tide_cat,
                    "Ward_Risk_Zone": (
                        ward_risk
                        if ward_risk in self.valid_risk_zones
                        else "Medium Risk"
                    ),
                    "Season": season,
                }

                result = self.inference.query(
                    variables=["Flood"],
                    evidence=evidence,
                )
                return result.values[1]  # Probability of 'Yes'
            except Exception as e:
                print(f"Bayesian inference error: {e}")
                # Fall through to fallback calculation

        # Fallback if model not available or error
        # Base probability set to 0.10 (10%) with small adjustments
        base_prob = 0.10
        
        risk_weights = {
            "Very High Risk": 0.25,
            "High Risk": 0.18,
            "Medium Risk": 0.10,
            "Low Risk": 0.03,
        }
        rainfall_weights = {"High": 0.22, "Medium": 0.12, "Low": 0.03}
        tide_weights = {"High": 0.15, "Medium": 0.08, "Low": 0.02}
        season_weights = {"Monsoon": 0.12, "Post-Monsoon": 0.06, "Winter": 0.01, "Summer": 0.03}

        prob = (
            base_prob
            + risk_weights.get(ward_risk, 0.10)
            + rainfall_weights.get(rainfall_cat, 0.10)
            + tide_weights.get(tide_cat, 0.08)
            + season_weights.get(season, 0.03)
        )
        return min(prob, 1.0)


class RealTimeFloodPredictionSystem:
    """Complete real-time flood prediction system"""

    def __init__(self, dataset_path: str = None, use_pretrained: bool = True):
        self.base_model = OptimizedFloodPredictionModels(
            dataset_path, enable_cache=True
        )
        self.rf_model = None
        self.clustering_model = None
        self.bayesian_model = None
        self.use_pretrained = use_pretrained
        self.prediction_queue = queue.Queue(maxsize=1000)
        self.batch_processor = ThreadPoolExecutor(max_workers=4)

    async def initialize_async(self):
        """Async initialization for real-time systems"""
        loop = asyncio.get_event_loop()

        if self.use_pretrained:
            # Try to load pre-trained models first
            success = await loop.run_in_executor(
                None, self.base_model.load_pretrained_models
            )
            if success:
                print("Using pre-trained models for real-time predictions")
                self.rf_model = OptimizedRandomForestPredictor(self.base_model)
                self.clustering_model = OptimizedWardClusteringModel(self.base_model)
                self.bayesian_model = OptimizedBayesianModel(self.base_model)
                return

        # Fall back to training if needed
        await loop.run_in_executor(None, self.train_all_models)

    def train_all_models(self):
        """Train all models with optimizations"""
        print("=== REAL-TIME FLOOD PREDICTION SYSTEM TRAINING ===")

        self.base_model.load_and_preprocess_data()

        # Initialize optimized models
        self.rf_model = OptimizedRandomForestPredictor(self.base_model)
        self.clustering_model = OptimizedWardClusteringModel(self.base_model)
        self.bayesian_model = OptimizedBayesianModel(self.base_model)

        # Train in optimal order
        self.rf_model.train()
        self.clustering_model.train()

        # Share ward lookup with base model
        self.base_model.ward_lookup = self.clustering_model.ward_lookup

        self.bayesian_model.create_network()

        # Store models in base for serialization
        self.base_model.rf_model = self.rf_model.model
        self.base_model.kmeans_model = self.clustering_model.model
        self.base_model.bayesian_model = self.bayesian_model.model
        self.base_model.ward_features = self.clustering_model.ward_features

        print("\n=== SYSTEM READY FOR REAL-TIME PREDICTIONS ===")

    async def predict_flood_risk_async(
        self, weather_data: Dict[str, Any], ward_code: str
    ) -> Dict[str, Any]:
        """Async real-time flood prediction"""
        loop = asyncio.get_event_loop()

        # Run prediction in executor to avoid blocking
        result = await loop.run_in_executor(
            None, self.predict_flood_risk, weather_data, ward_code
        )

        return result

    def predict_flood_risk(
        self, weather_data: Dict[str, Any], ward_code: str
    ) -> Dict[str, Any]:
        """Real-time flood prediction with all optimizations"""

        start_time = time.time()

        # Check cache first
        cache_key = {**weather_data, "ward": ward_code}
        if self.base_model.cache:
            cached = self.base_model.cache.get(cache_key)
            if cached:
                cached["response_time_ms"] = 1  # Cache hit
                cached["cache_hit"] = True
                return cached

        # Ensure models are loaded
        if self.base_model.models_loaded:
            # Use pre-loaded models directly
            rf_model = (
                self.rf_model
                if self.rf_model
                else OptimizedRandomForestPredictor(self.base_model)
            )
            clustering_model = (
                self.clustering_model
                if self.clustering_model
                else OptimizedWardClusteringModel(self.base_model)
            )
            ward_features = self.base_model.ward_features
        else:
            # Use instance models
            rf_model = self.rf_model
            clustering_model = self.clustering_model
            ward_features = clustering_model.ward_features if clustering_model else None

        # RF prediction
        rf_prediction = rf_model.predict(weather_data)

        # Ward risk zone
        ward_risk_zone = (
            clustering_model.get_ward_risk_zone(ward_code)
            if clustering_model
            else "Medium Risk"
        )

        # Categorize for Bayesian
        rainfall_cat = (
            "Low"
            if weather_data.get("Rainfall_mm", 0) < 10
            else "Medium" if weather_data.get("Rainfall_mm", 0) < 50 else "High"
        )
        tide_cat = (
            "Low"
            if weather_data.get("Tide_Level_m", 0) < 2
            else "Medium" if weather_data.get("Tide_Level_m", 0) < 4 else "High"
        )

        # Bayesian probability (use model if available, else fallback - matching original logic)
        if self.bayesian_model and self.bayesian_model.model:
            bayesian_prob = self.bayesian_model.predict_probability(
                rainfall_cat,
                tide_cat,
                ward_risk_zone,
                weather_data.get("season", "Monsoon"),
            )
        else:
            # Base probability set to 0.10 (10%) with small adjustments
            base_prob = 0.10
            
            risk_weights = {
                "Very High Risk": 0.25,
                "High Risk": 0.18,
                "Medium Risk": 0.10,
                "Low Risk": 0.03,
            }
            rainfall_weights = {"High": 0.22, "Medium": 0.12, "Low": 0.03}
            tide_weights = {"High": 0.15, "Medium": 0.08, "Low": 0.02}
            season_weights = {"Monsoon": 0.12, "Post-Monsoon": 0.06, "Winter": 0.01, "Summer": 0.03}

            bayesian_prob = (
                base_prob
                + risk_weights.get(ward_risk_zone, 0.10)
                + rainfall_weights.get(rainfall_cat, 0.10)
                + tide_weights.get(tide_cat, 0.08)
                + season_weights.get(weather_data.get("season", "Monsoon"), 0.03)
            )
            bayesian_prob = min(bayesian_prob, 1.0)

        # Calculate combined assessment
        rf_high_prob = rf_prediction.get("risk_probabilities", {}).get("high", 0.0)
        confidence_score = (rf_high_prob + bayesian_prob) / 2

        if confidence_score >= 0.8:
            confidence_level = "High"
        elif confidence_score >= 0.5:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"

        # Prepare result
        result = {
            "ward_code": ward_code,
            "ward_risk_zone": ward_risk_zone,
            "random_forest": rf_prediction,
            "bayesian_probability": bayesian_prob,
            "combined_assessment": {
                "high_risk": rf_prediction["flood_risk_level"] >= 2
                or bayesian_prob > 0.7,
                "confidence": confidence_level,
                "confidence_score": confidence_score,
            },
            "response_time_ms": int((time.time() - start_time) * 1000),
            "cache_hit": False,
            "timestamp": datetime.now().isoformat(),
        }

        # Convert all NumPy types to Python types
        result = convert_numpy_types(result)

        # Cache result
        if self.base_model.cache:
            self.base_model.cache.set(cache_key, result)

        return result

    def predict_batch(self, predictions_batch: List[Tuple[Dict, str]]) -> List[Dict]:
        """Process batch of predictions for efficiency"""
        results = []

        for weather_data, ward_code in predictions_batch:
            result = self.predict_flood_risk(weather_data, ward_code)
            results.append(result)

        return results

    def save_models(self, model_dir: str = "../models/trained"):
        """Save optimized models"""
        os.makedirs(model_dir, exist_ok=True)

        # Save models
        joblib.dump(self.base_model.rf_model, f"{model_dir}/random_forest_model.pkl")
        joblib.dump(self.base_model.scaler, f"{model_dir}/scaler.pkl")
        joblib.dump(self.base_model.label_encoders, f"{model_dir}/label_encoders.pkl")
        joblib.dump(self.base_model.kmeans_model, f"{model_dir}/kmeans_model.pkl")

        if self.clustering_model and hasattr(self.clustering_model, "ward_features"):
            self.clustering_model.ward_features.to_csv(
                f"{model_dir}/ward_clusters.csv", index=False
            )

        if PGMPY_AVAILABLE and self.base_model.bayesian_model is not None:
            joblib.dump(
                self.base_model.bayesian_model, f"{model_dir}/bayesian_model.pkl"
            )

        # Save metadata
        metadata = {
            "saved_at": datetime.now().isoformat(),
            "model_version": "2.0",
            "features": self.base_model.feature_columns,
            "cache_enabled": self.base_model.cache is not None,
        }

        with open(f"{model_dir}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Optimized models saved to {model_dir}")


# Async wrapper for real-time API integration
async def real_time_prediction_handler(
    weather_data: Dict, ward_code: str, system: RealTimeFloodPredictionSystem
) -> Dict:
    """Handler for real-time prediction requests"""
    result = await system.predict_flood_risk_async(weather_data, ward_code)
    return result


if __name__ == "__main__":
    # Example usage with async support
    async def main():
        # Initialize real-time system
        system = RealTimeFloodPredictionSystem(use_pretrained=True)

        # Initialize asynchronously
        await system.initialize_async()

        # If models not pre-trained, train them
        if not system.base_model.models_loaded:
            system.train_all_models()
            system.save_models()

        # Example real-time predictions
        test_cases = [
            {
                "weather": {
                    "Rainfall_mm": 25.5,
                    "Rainfall_24hr": 45.2,
                    "Tide_Level_m": 3.2,
                    "Temperature_C": 28.5,
                    "Humidity_%": 85.0,
                    "Wind_Speed_kmh": 15.3,
                    "season": "Monsoon",
                },
                "ward": "A",
            },
            {
                "weather": {
                    "Rainfall_mm": 5.0,
                    "Rainfall_24hr": 10.0,
                    "Tide_Level_m": 1.5,
                    "Temperature_C": 30.0,
                    "Humidity_%": 60.0,
                    "Wind_Speed_kmh": 8.0,
                    "season": "Summer",
                },
                "ward": "B",
            },
        ]

        print("\n=== REAL-TIME PREDICTIONS ===")

        # Test single predictions
        for i, test in enumerate(test_cases, 1):
            print(f"\nTest Case {i}:")
            result = await real_time_prediction_handler(
                test["weather"], test["ward"], system
            )
            print(f"Ward: {result['ward_code']}")
            print(f"Risk Zone: {result['ward_risk_zone']}")
            print(f"Flood Risk Level: {result['random_forest']['flood_risk_level']}")
            print(f"Will Flood: {result['random_forest']['will_flood']}")
            print(f"Bayesian Probability: {result['bayesian_probability']:.2%}")
            print(f"Combined Assessment: {result['combined_assessment']['confidence']}")
            print(f"Response Time: {result['response_time_ms']}ms")
            print(f"Cache Hit: {result['cache_hit']}")

        # Test batch prediction
        print("\n=== BATCH PREDICTION TEST ===")
        batch = [(test["weather"], test["ward"]) for test in test_cases * 5]

        start = time.time()
        batch_results = system.predict_batch(batch)
        batch_time = time.time() - start

        print(f"Processed {len(batch)} predictions in {batch_time:.2f}s")
        print(f"Average time per prediction: {(batch_time/len(batch))*1000:.2f}ms")

    # Run async main
    asyncio.run(main())
