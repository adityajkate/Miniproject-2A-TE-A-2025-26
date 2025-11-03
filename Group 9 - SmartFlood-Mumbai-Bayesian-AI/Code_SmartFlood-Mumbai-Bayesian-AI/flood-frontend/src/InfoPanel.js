import React from "react";
import "./InfoPanel.css";

const InfoPanel = ({ selectedWard, prediction, loading }) => {
  const getRiskColor = (riskLevel) => {
    switch (riskLevel?.toLowerCase()) {
      case "low":
        return "#28a745";
      case "medium":
        return "#ffc107";
      case "high":
        return "#fd7e14";
      case "critical":
        return "#dc3545";
      default:
        return "#6c757d";
    }
  };

  return (
    <div className="info-panel">
      <h3>Mumbai Flood Risk Prediction</h3>

      {!selectedWard && (
        <p className="instruction">
          Click on a ward to see flood risk prediction
        </p>
      )}

      {selectedWard && (
        <div className="ward-info">
          <h4>Ward: {selectedWard}</h4>

          {loading && (
            <div className="loading">
              <p>Analyzing flood risk...</p>
            </div>
          )}

          {prediction && !loading && (
            <div className="prediction-result">
              <div className="risk-level">
                <span
                  className="risk-badge"
                  style={{
                    backgroundColor: getRiskColor(prediction.risk_level),
                  }}
                >
                  {prediction.risk_level} Risk
                </span>
              </div>

              <div className="probability">
                <strong>Probability:</strong>{" "}
                {(prediction.flood_risk_probability * 100).toFixed(1)}%
              </div>

              <div className="description">
                {prediction.risk_level === "Low" &&
                  "Minimal flood risk expected"}
                {prediction.risk_level === "Medium" &&
                  "Moderate flood risk - stay alert"}
                {prediction.risk_level === "High" &&
                  "High flood risk - take precautions"}
                {prediction.risk_level === "Critical" &&
                  "Critical flood risk - avoid area if possible"}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default InfoPanel;
