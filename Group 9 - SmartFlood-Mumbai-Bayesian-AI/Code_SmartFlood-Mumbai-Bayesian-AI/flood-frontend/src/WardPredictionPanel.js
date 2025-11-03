import React, { useState, useEffect } from "react";
import {
  predictSingleWard,
  getWardCurrentStatus,
  getRiskColor,
  getWardCoordinates,
} from "./api";
import "./WardPredictionPanel.css";

const WardPredictionPanel = ({
  selectedWard,
  onWardSelect,
  availableWards = [],
}) => {
  const [prediction, setPrediction] = useState(null);
  const [currentStatus, setCurrentStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [inputData, setInputData] = useState({
    rainfall_mm: "",
    tide_height_m: "",
    season: "monsoon",
  });

  // Load current status when ward is selected
  useEffect(() => {
    if (selectedWard) {
      loadCurrentStatus();
    }
  }, [selectedWard]);

  const loadCurrentStatus = async () => {
    if (!selectedWard) return;

    try {
      const status = await getWardCurrentStatus(selectedWard);
      setCurrentStatus(status);
    } catch (error) {
      console.error("Failed to load current status:", error);
      setCurrentStatus(null);
    }
  };

  const handlePredictWard = async () => {
    if (!selectedWard) {
      setError("Please select a ward first");
      return;
    }

    // Validate input data
    if (!inputData.rainfall_mm || inputData.rainfall_mm === "") {
      setError("Please enter rainfall amount");
      return;
    }

    if (!inputData.tide_height_m || inputData.tide_height_m === "") {
      setError("Please enter tide height");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await predictSingleWard(
        selectedWard,
        parseFloat(inputData.rainfall_mm),
        parseFloat(inputData.tide_height_m),
        inputData.season
      );

      setPrediction(result);
      // Refresh current status after new prediction
      await loadCurrentStatus();
    } catch (error) {
      console.error("Prediction failed:", error);
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (field, value) => {
    setInputData((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const getWardDisplayName = (wardId) => {
    const ward = availableWards.find((w) => w.ward_id === wardId);
    return ward ? `${ward.ward_name} (${wardId})` : wardId;
  };

  const getRiskLevelClass = (riskLevel) => {
    switch (riskLevel?.toLowerCase()) {
      case "critical":
        return "risk-critical";
      case "high":
        return "risk-high";
      case "medium":
        return "risk-medium";
      case "low":
        return "risk-low";
      default:
        return "risk-unknown";
    }
  };

  return (
    <div className="ward-prediction-panel">
      <div className="panel-header">
        <h3>🎯 Ward Flood Prediction</h3>
        <p>
          Click on a ward or select from dropdown to get detailed flood risk
          analysis
        </p>
      </div>

      {/* Ward Selection */}
      <div className="ward-selection">
        <label htmlFor="ward-select">Select Ward:</label>
        <select
          id="ward-select"
          value={selectedWard || ""}
          onChange={(e) => onWardSelect(e.target.value)}
          className="ward-dropdown"
        >
          <option value="">Choose a ward...</option>
          {availableWards.map((ward) => (
            <option key={ward.ward_id} value={ward.ward_id}>
              {ward.ward_name} ({ward.ward_id}) - {ward.zone}
            </option>
          ))}
        </select>
      </div>

      {selectedWard && (
        <>
          {/* Current Status Display */}
          {currentStatus && currentStatus.status !== "no_recent_data" && (
            <div className="current-status">
              <h4>📊 Current Status</h4>
              <div
                className={`status-card ${getRiskLevelClass(
                  currentStatus.risk_level
                )}`}
              >
                <div className="status-header">
                  <span className="ward-name">
                    {getWardDisplayName(selectedWard)}
                  </span>
                  <span
                    className={`risk-badge ${getRiskLevelClass(
                      currentStatus.risk_level
                    )}`}
                  >
                    {currentStatus.risk_level}
                  </span>
                </div>
                <div className="status-details">
                  <div className="status-item">
                    <span>Flood Probability:</span>
                    <span>
                      {(currentStatus.current_flood_probability * 100).toFixed(
                        1
                      )}
                      %
                    </span>
                  </div>
                  <div className="status-item">
                    <span>Last Updated:</span>
                    <span>
                      {new Date(currentStatus.last_updated).toLocaleString()}
                    </span>
                  </div>
                  <div className="status-item">
                    <span>Data Freshness:</span>
                    <span
                      className={
                        currentStatus.data_freshness === "current"
                          ? "fresh"
                          : "stale"
                      }
                    >
                      {currentStatus.data_freshness}
                    </span>
                  </div>
                </div>
                <div className="recommendation">
                  <strong>Recommendation:</strong>
                  <p>{currentStatus.recommendation}</p>
                </div>
              </div>
            </div>
          )}

          {/* Input Parameters */}
          <div className="input-section">
            <h4>🌧️ Prediction Parameters</h4>
            <div className="input-grid">
              <div className="input-group">
                <label htmlFor="rainfall">Rainfall (mm):</label>
                <input
                  id="rainfall"
                  type="number"
                  min="0"
                  max="200"
                  step="0.1"
                  value={inputData.rainfall_mm}
                  onChange={(e) =>
                    handleInputChange("rainfall_mm", e.target.value)
                  }
                  placeholder="Enter rainfall in mm (e.g., 25.5)"
                  className="input-field"
                  required
                />
              </div>

              <div className="input-group">
                <label htmlFor="tide">Tide Height (m):</label>
                <input
                  id="tide"
                  type="number"
                  min="0"
                  max="5"
                  step="0.1"
                  value={inputData.tide_height_m}
                  onChange={(e) =>
                    handleInputChange("tide_height_m", e.target.value)
                  }
                  placeholder="Enter tide height in m (e.g., 2.1)"
                  className="input-field"
                  required
                />
              </div>

              <div className="input-group">
                <label htmlFor="season">Season:</label>
                <select
                  id="season"
                  value={inputData.season}
                  onChange={(e) => handleInputChange("season", e.target.value)}
                  className="input-field"
                >
                  <option value="monsoon">Monsoon</option>
                  <option value="winter">Winter</option>
                  <option value="summer">Summer</option>
                  <option value="post-monsoon">Post-Monsoon</option>
                </select>
              </div>
            </div>

            <button
              onClick={handlePredictWard}
              disabled={loading || !selectedWard}
              className="predict-button"
            >
              {loading ? "🔄 Analyzing..." : "🎯 Predict Flood Risk"}
            </button>
          </div>

          {/* Error Display */}
          {error && (
            <div className="error-section">
              <div className="error-message">
                <span className="error-icon">❌</span>
                <span>{error}</span>
              </div>
            </div>
          )}

          {/* Prediction Results */}
          {prediction && (
            <div className="prediction-results">
              <h4>🤖 ML Prediction Results</h4>

              {/* Overall Assessment */}
              {prediction.overall_assessment && (
                <div
                  className={`assessment-card ${getRiskLevelClass(
                    prediction.overall_assessment.combined_risk_level
                  )}`}
                >
                  <div className="assessment-header">
                    <span className="ward-name">
                      {getWardDisplayName(selectedWard)}
                    </span>
                    <span
                      className={`risk-badge ${getRiskLevelClass(
                        prediction.overall_assessment.combined_risk_level
                      )}`}
                    >
                      {prediction.overall_assessment.combined_risk_level}
                    </span>
                  </div>

                  <div className="assessment-metrics">
                    <div className="metric">
                      <span className="metric-label">
                        Combined Probability:
                      </span>
                      <span className="metric-value">
                        {(
                          prediction.overall_assessment
                            .combined_flood_probability * 100
                        ).toFixed(1)}
                        %
                      </span>
                    </div>
                    <div className="metric">
                      <span className="metric-label">Models Used:</span>
                      <span className="metric-value">
                        {prediction.overall_assessment.models_used}
                      </span>
                    </div>
                  </div>

                  <div className="recommendation">
                    <strong>Recommendation:</strong>
                    <p>{prediction.overall_assessment.recommendation}</p>
                  </div>
                </div>
              )}

              {/* Individual Model Results */}
              <div className="model-results">
                {/* Random Forest */}
                {prediction.random_forest && (
                  <div className="model-card">
                    <h5>🌳 Random Forest Model</h5>
                    <div className="model-metrics">
                      <div className="metric">
                        <span>Flood Probability:</span>
                        <span>
                          {(
                            prediction.random_forest.flood_probability * 100
                          ).toFixed(1)}
                          %
                        </span>
                      </div>
                      <div className="metric">
                        <span>Risk Level:</span>
                        <span
                          className={getRiskLevelClass(
                            prediction.random_forest.risk_level
                          )}
                        >
                          {prediction.random_forest.risk_level}
                        </span>
                      </div>
                      <div className="metric">
                        <span>Confidence:</span>
                        <span>{prediction.random_forest.confidence}</span>
                      </div>
                    </div>
                  </div>
                )}

                {/* Bayesian Network */}
                {prediction.bayesian_network && (
                  <div className="model-card">
                    <h5>🧠 Bayesian Network</h5>
                    <div className="model-metrics">
                      <div className="metric">
                        <span>Flood Probability:</span>
                        <span>
                          {(
                            prediction.bayesian_network.flood_probability * 100
                          ).toFixed(1)}
                          %
                        </span>
                      </div>
                      {prediction.bayesian_network.confidence_interval && (
                        <div className="metric">
                          <span>Confidence Interval:</span>
                          <span>
                            [
                            {(
                              prediction.bayesian_network
                                .confidence_interval[0] * 100
                            ).toFixed(1)}
                            % -
                            {(
                              prediction.bayesian_network
                                .confidence_interval[1] * 100
                            ).toFixed(1)}
                            %]
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Clustering Analysis */}
                {prediction.cluster_analysis && (
                  <div className="model-card">
                    <h5>🎯 Cluster Analysis</h5>
                    <div className="model-metrics">
                      <div className="metric">
                        <span>Cluster:</span>
                        <span>{prediction.cluster_analysis.cluster}</span>
                      </div>
                      <div className="metric">
                        <span>Cluster Risk Level:</span>
                        <span
                          className={getRiskLevelClass(
                            prediction.cluster_analysis.cluster_risk_level
                          )}
                        >
                          {prediction.cluster_analysis.cluster_risk_level}
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Route Status */}
              {prediction.route_status && (
                <div className="route-status">
                  <h5>🛣️ Route Status</h5>
                  <div className="route-info">
                    <div className="route-item">
                      <span>High Risk Area:</span>
                      <span
                        className={
                          prediction.route_status.is_high_risk_area
                            ? "risk-high"
                            : "risk-low"
                        }
                      >
                        {prediction.route_status.is_high_risk_area
                          ? "Yes"
                          : "No"}
                      </span>
                    </div>
                    <div className="route-item">
                      <span>Safe Zones Nearby:</span>
                      <span
                        className={
                          prediction.route_status.safe_zones_nearby
                            ? "available"
                            : "unavailable"
                        }
                      >
                        {prediction.route_status.safe_zones_nearby
                          ? "Available"
                          : "Limited"}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Timestamp */}
              <div className="prediction-timestamp">
                <small>
                  Prediction generated:{" "}
                  {new Date(prediction.timestamp).toLocaleString()}
                </small>
              </div>
            </div>
          )}
        </>
      )}

      {!selectedWard && (
        <div className="no-selection">
          <div className="no-selection-icon">🗺️</div>
          <p>
            Select a ward from the dropdown above or click on a ward in the map
            to get started
          </p>
        </div>
      )}
    </div>
  );
};

export default WardPredictionPanel;
