"""
Flood Prediction Models for Mumbai
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
from typing import Dict, List, Tuple, Any
import warnings

warnings.filterwarnings("ignore")

try:
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination

    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False
    print("Warning: pgmpy not available. Bayesian Network model will be disabled.")


class FloodPredictionModels:
    """Main class containing all three flood prediction models"""

    def __init__(self, dataset_path: str = None):
        if dataset_path is None:
            # Get the directory of this file and construct the path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            dataset_path = os.path.join(
                os.path.dirname(current_dir), "Dataset", "enriched_flood_dataset.csv"
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
            "Season",
        ]

    def load_and_preprocess_data(self):
        """Load and preprocess the flood dataset"""
        print("Loading dataset...")
        self.data = pd.read_csv(self.dataset_path)

        # Handle categorical variables
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

        # Update feature columns to include encoded versions
        self.feature_columns = [
            "Rainfall_mm",
            "Rainfall_24hr",
            "Tide_Level_m",
            "Temperature_C",
            "Humidity_%",
            "Wind_Speed_kmh",
            "Season_encoded",
        ]

        print(f"Dataset loaded: {len(self.data)} records")
        print(f"Columns: {list(self.data.columns)}")
        return self.data


class RandomForestFloodPredictor:
    """Model 1: Random Forest Classifier for main flood prediction"""

    def __init__(self, parent_model):
        self.parent = parent_model
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, class_weight="balanced"
        )

    def train(self):
        """Train the Random Forest model"""
        print("\n=== Training Random Forest Classifier ===")

        # Prepare features and targets
        X = self.parent.data[self.parent.feature_columns].copy()
        y_risk = self.parent.data["Flood_Risk_Level"]
        y_flood = self.parent.data["Flood_Occurred"]

        # Handle missing values
        X = X.fillna(X.mean())

        # Split data
        X_train, X_test, y_risk_train, y_risk_test, y_flood_train, y_flood_test = (
            train_test_split(
                X, y_risk, y_flood, test_size=0.2, random_state=42, stratify=y_risk
            )
        )

        # Scale features
        X_train_scaled = self.parent.scaler.fit_transform(X_train)
        X_test_scaled = self.parent.scaler.transform(X_test)

        # Train model for risk level prediction
        self.model.fit(X_train_scaled, y_risk_train)

        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_risk_test, y_pred)

        print(f"Random Forest Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_risk_test, y_pred))

        # Feature importance
        feature_importance = pd.DataFrame(
            {
                "feature": self.parent.feature_columns,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        print("\nFeature Importance:")
        print(feature_importance)

        return self.model

    def predict(self, weather_data: Dict[str, float]) -> Dict[str, Any]:
        """Predict flood risk for current weather conditions"""
        # Prepare input data
        input_df = pd.DataFrame([weather_data])

        # Handle categorical encoding
        if "season" in weather_data:
            season_encoded = self.parent.label_encoders["Season"].transform(
                [weather_data["season"]]
            )[0]
            input_df["Season_encoded"] = season_encoded

        # Select and order features
        input_features = input_df[self.parent.feature_columns].fillna(0)
        input_scaled = self.parent.scaler.transform(input_features)

        # Make predictions
        risk_level = self.model.predict(input_scaled)[0]
        risk_proba = self.model.predict_proba(input_scaled)[0]
        will_flood = risk_level >= 1  # Medium or High risk

        return {
            "flood_risk_level": int(risk_level),
            "will_flood": bool(will_flood),
            "risk_probabilities": {
                "low": float(risk_proba[0]),
                "medium": float(risk_proba[1]) if len(risk_proba) > 1 else 0.0,
                "high": float(risk_proba[2]) if len(risk_proba) > 2 else 0.0,
            },
        }


class WardClusteringModel:
    """Model 2: K-means Clustering for ward risk zone grouping"""

    def __init__(self, parent_model):
        self.parent = parent_model
        self.model = KMeans(n_clusters=4, random_state=42)
        self.ward_features = None

    def prepare_ward_features(self):
        """Prepare ward-level aggregated features"""
        print("\n=== Preparing Ward Features for Clustering ===")

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

        # Flatten column names
        ward_stats.columns = ["_".join(col).strip() for col in ward_stats.columns]
        ward_stats = ward_stats.reset_index()

        # Calculate flood frequency
        ward_stats["flood_frequency"] = (
            ward_stats["Flood_Occurred_sum"] / ward_stats["Flood_Occurred_count"]
        ).round(3)

        # Encode categorical features
        elevation_encoder = LabelEncoder()
        drainage_encoder = LabelEncoder()

        ward_stats["elevation_encoded"] = elevation_encoder.fit_transform(
            ward_stats["Elevation_m_first"].astype(str)
        )
        ward_stats["drainage_encoded"] = drainage_encoder.fit_transform(
            ward_stats["Drainage_Capacity_first"].astype(str)
        )

        # Select clustering features
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

        print(f"Ward features prepared: {len(self.ward_features)} wards")
        print(f"Clustering features: {clustering_features}")

        return self.ward_features

    def train(self):
        """Train the K-means clustering model"""
        print("\n=== Training K-means Clustering ===")

        if self.ward_features is None:
            self.prepare_ward_features()

        # Prepare features for clustering
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
        self.ward_clusters = dict(
            zip(self.ward_features["Ward"], self.ward_features["cluster"])
        )
        print("\nWard Risk Zones:")
        for cluster, risk in cluster_risk_mapping.items():
            wards_in_cluster = self.ward_features[
                self.ward_features["cluster"] == cluster
            ]
            print(f"Cluster {cluster} ({risk}): {len(wards_in_cluster)} wards")

        return self.model

    def get_ward_risk_zone(self, ward_code: str) -> str:
        """Get risk zone for a ward"""
        if not hasattr(self, "ward_features") or self.ward_features is None:
            return "Medium Risk"

        ward_code = ward_code.upper()

        # Try to find the ward in ward_features
        ward_match = self.ward_features[self.ward_features["Ward"] == ward_code]
        if not ward_match.empty:
            return ward_match.iloc[0]["risk_zone"]

        # Fallback to cluster mapping if available
        if hasattr(self, "ward_clusters") and self.ward_clusters is not None:
            cluster = self.ward_clusters.get(ward_code, -1)
            if cluster != -1 and hasattr(self, "cluster_risk_mapping"):
                return self.cluster_risk_mapping.get(cluster, "Medium Risk")

        return "Medium Risk"

    def get_all_ward_clusters(self) -> pd.DataFrame:
        """Get all wards with their cluster assignments"""
        return self.ward_features[["Ward", "Ward_Name", "cluster", "risk_zone"]].copy()


class BayesianFloodModel:
    """Model 3: Bayesian Network for probability calculation with uncertainty"""

    def __init__(self, parent_model):
        self.parent = parent_model
        self.model = None
        self.inference = None

    def create_network(self):
        """Create and train the Bayesian Network"""
        if not PGMPY_AVAILABLE:
            print("Bayesian Network model not available - pgmpy not installed")
            return None

        print("\n=== Creating Bayesian Network ===")

        # Define network structure
        self.model = DiscreteBayesianNetwork(
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

        # Initialize ward clusters mapping
        self.ward_clusters = {}
        self.cluster_risk_mapping = {}

        if hasattr(self.parent, "clustering_model") and self.parent.clustering_model:
            try:
                ward_data = self.parent.clustering_model.get_all_ward_clusters()
                for _, row in ward_data.iterrows():
                    self.ward_clusters[row["Ward"]] = row["cluster"]
                    self.cluster_risk_mapping[row["cluster"]] = row["risk_zone"]
            except Exception as e:
                print(f"Warning: Could not load ward clusters: {e}")
                # Store the cluster risk mapping for later use
        self.cluster_risk_mapping = {
            0: "Low Risk",
            1: "Medium Risk",
            2: "High Risk",
            3: "Very High Risk",
        }

        # Create ward clusters mapping for easy lookup (use clustering_model's ward_features if available)
        if (
            hasattr(self.parent, "clustering_model")
            and getattr(self.parent.clustering_model, "ward_features", None) is not None
        ):
            self.ward_clusters = dict(
                zip(
                    self.parent.clustering_model.ward_features["Ward"],
                    self.parent.clustering_model.ward_features["cluster"],
                )
            )
        else:
            self.ward_clusters = {}

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
            # Store the valid risk zones for later validation
            self.valid_risk_zones = set(
                self.parent.clustering_model.ward_features["risk_zone"].unique()
            )
        else:
            data_for_bn["Ward_Risk_Zone"] = "Medium Risk"  # Default
            self.valid_risk_zones = {
                "Low Risk",
                "Medium Risk",
                "High Risk",
                "Very High Risk",
            }

        data_for_bn["Flood"] = data_for_bn["Flood_Occurred"].map({0: "No", 1: "Yes"})

        # Remove rows with NaN values and ensure all risk zones are represented
        bn_data = data_for_bn[
            ["Rainfall_Category", "Tide_Category", "Ward_Risk_Zone", "Season", "Flood"]
        ].dropna()

        # Print unique values for debugging
        print(
            f"Unique Ward Risk Zones in training data: {sorted(bn_data['Ward_Risk_Zone'].unique())}"
        )

        # Fit the model
        self.model.fit(bn_data)

        # Create inference object
        self.inference = VariableElimination(self.model)

        print("Bayesian Network created and trained successfully")
        return self.model

    def predict_probability(
        self, rainfall_cat: str, tide_cat: str, ward_risk: str, season: str
    ) -> float:
        """Calculate flood probability using Bayesian inference"""
        if not PGMPY_AVAILABLE or self.model is None:
            # Fallback simple probability calculation
            risk_weights = {
                "Very High Risk": 0.8,
                "High Risk": 0.6,
                "Medium Risk": 0.4,
                "Low Risk": 0.2,
            }
            rainfall_weights = {"High": 0.7, "Medium": 0.4, "Low": 0.1}
            tide_weights = {"High": 0.6, "Medium": 0.3, "Low": 0.1}
            season_weights = {"Monsoon": 0.8, "Winter": 0.2, "Summer": 0.3}

            prob = (
                risk_weights.get(ward_risk, 0.4) * 0.4
                + rainfall_weights.get(rainfall_cat, 0.3) * 0.3
                + tide_weights.get(tide_cat, 0.2) * 0.2
                + season_weights.get(season, 0.3) * 0.1
            )
            return min(prob, 1.0)


# Main model manager class
class FloodPredictionSystem:
    """Complete flood prediction system with all three models"""

    def __init__(self, dataset_path: str = None):
        self.base_model = FloodPredictionModels(dataset_path)
        self.rf_model = None
        self.clustering_model = None
        self.bayesian_model = None

    def train_all_models(self):
        """Train all three models"""
        print("=== FLOOD PREDICTION SYSTEM TRAINING ===")

        # Load data
        self.base_model.load_and_preprocess_data()

        # Initialize models
        self.rf_model = RandomForestFloodPredictor(self.base_model)
        self.clustering_model = WardClusteringModel(self.base_model)
        self.bayesian_model = BayesianFloodModel(self.base_model)

        # Train models in correct order (clustering first, then Bayesian)
        self.rf_model.train()
        self.clustering_model.train()

        # Pass clustering model to Bayesian model so it can access ward risk zones
        self.base_model.clustering_model = self.clustering_model
        self.bayesian_model.create_network()

        print("\n=== ALL MODELS TRAINED SUCCESSFULLY ===")

    def predict_flood_risk(
        self, weather_data: Dict[str, Any], ward_code: str
    ) -> Dict[str, Any]:
        """Complete flood prediction using all three models"""

        # Model 1: Random Forest prediction
        rf_prediction = self.rf_model.predict(weather_data)

        # Model 2: Ward risk zone
        ward_risk_zone = self.clustering_model.get_ward_risk_zone(ward_code)

        # Model 3: Bayesian probability
        # Categorize input data for Bayesian model
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

        bayesian_prob = self.bayesian_model.predict_probability(
            rainfall_cat,
            tide_cat,
            ward_risk_zone,
            weather_data.get("season", "Monsoon"),
        )

        # Calculate confidence safely
        rf_high_prob = rf_prediction.get("risk_probabilities", {}).get("high", 0.0)
        confidence_score = (rf_high_prob + bayesian_prob) / 2

        # Determine confidence level
        if confidence_score >= 0.8:
            confidence_level = "High"
        elif confidence_score >= 0.5:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"

        return {
            "ward_code": ward_code,
            "ward_risk_zone": ward_risk_zone,
            "random_forest": rf_prediction,
            "bayesian_probability": float(bayesian_prob),
            "combined_assessment": {
                "high_risk": rf_prediction["flood_risk_level"] >= 2
                or bayesian_prob > 0.7,
                "confidence": confidence_level,
                "confidence_score": float(confidence_score),
            },
        }

    def save_models(self, model_dir: str = "Backend/models/trained"):
        """Save all trained models"""
        os.makedirs(model_dir, exist_ok=True)

        # Save Random Forest model
        joblib.dump(self.rf_model.model, f"{model_dir}/random_forest_model.pkl")
        joblib.dump(self.base_model.scaler, f"{model_dir}/scaler.pkl")
        joblib.dump(self.base_model.label_encoders, f"{model_dir}/label_encoders.pkl")

        # Save clustering model
        joblib.dump(self.clustering_model.model, f"{model_dir}/kmeans_model.pkl")
        self.clustering_model.ward_features.to_csv(
            f"{model_dir}/ward_clusters.csv", index=False
        )

        # Save Bayesian model (if available)
        if PGMPY_AVAILABLE and self.bayesian_model.model is not None:
            joblib.dump(self.bayesian_model.model, f"{model_dir}/bayesian_model.pkl")

        print(f"All models saved to {model_dir}")


if __name__ == "__main__":
    # Example usage
    system = FloodPredictionSystem()
    system.train_all_models()

    # Example prediction
    sample_weather = {
        "Rainfall_mm": 25.5,
        "Rainfall_24hr": 45.2,
        "Tide_Level_m": 3.2,
        "Temperature_C": 28.5,
        "Humidity_%": 85.0,
        "Wind_Speed_kmh": 15.3,
        "season": "Monsoon",
    }

    prediction = system.predict_flood_risk(sample_weather, "A")
    print("\n=== SAMPLE PREDICTION ===")
    print(prediction)

    # Save models
    system.save_models()
