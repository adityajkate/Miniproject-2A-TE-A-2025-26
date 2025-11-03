import React from "react";
import "./WardDetails.css";

const WardDetails = ({
  selectedWard,
  prediction,
  loading,
  currentWeather,
  weatherAlerts,
}) => {
  // Utility function to safely handle numeric values
  const safeNumber = (value, defaultValue = 0) => {
    if (value === null || value === undefined || isNaN(value) || value === "") {
      return defaultValue;
    }
    const num = Number(value);
    return isNaN(num) ? defaultValue : num;
  };

  // Utility function to safely handle string values
  const safeString = (value, defaultValue = "Unknown") => {
    if (value === null || value === undefined || value === "") {
      return defaultValue;
    }
    return String(value);
  };
  const getRiskColor = (riskLevel) => {
    switch (riskLevel?.toLowerCase()) {
      case "low":
        return "#34a853";
      case "medium":
        return "#fbbc04";
      case "high":
        return "#ea4335";
      case "critical":
        return "#ea4335";
      default:
        return "#6c757d";
    }
  };

  const getRiskDescription = (riskLevel) => {
    switch (riskLevel?.toLowerCase()) {
      case "low":
        return "Minimal flood risk expected";
      case "medium":
        return "Moderate flood risk - stay alert";
      case "high":
        return "High flood risk - take precautions";
      case "critical":
        return "Critical flood risk - avoid area if possible";
      default:
        return "Risk assessment unavailable";
    }
  };

  return (
    <div className="ward-details">
      <h2>Ward Details</h2>

      {!selectedWard ? (
        <div className="no-selection">
          <div className="no-selection-icon">
            <svg
              width="48"
              height="48"
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M21 10C21 17 12 23 12 23S3 17 3 10A9 9 0 0 1 21 10Z"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <circle
                cx="12"
                cy="10"
                r="3"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </div>
          <p>Click on a ward to get real-time flood risk prediction</p>
          <div className="instruction-note">
            <small>
              🤖 All predictions are generated in real-time using our ML model
            </small>
          </div>
        </div>
      ) : (
        <div className="ward-info">
          <div className="ward-header">
            <h3>{selectedWard}</h3>
            {loading && (
              <div className="loading-spinner">
                <div className="spinner"></div>
              </div>
            )}
          </div>

          {!prediction && !loading && (
            <div className="no-prediction">
              <div className="info-message">
                <h4>🤖 Ready for ML Prediction</h4>
                <p>
                  Click this ward again to get a real-time flood risk prediction
                  from our machine learning model.
                </p>
                <div className="ml-info">
                  <small>
                    • Uses 15-day rainfall pattern analysis
                    <br />
                    • Real-time risk assessment
                    <br />• Confidence level included
                  </small>
                </div>
              </div>
            </div>
          )}

          {prediction && !loading && (
            <div className="prediction-details">
              {console.log("WardDetails received prediction:", prediction)}
              {/* Overall Assessment */}
              {prediction.overall_assessment && (
                <div className="overall-assessment">
                  <div className="risk-status">
                    <span
                      className="risk-badge"
                      style={{
                        backgroundColor: getRiskColor(
                          prediction.overall_assessment.combined_risk_level ||
                            "Low"
                        ),
                      }}
                    >
                      {prediction.overall_assessment.combined_risk_level ||
                        "Low"}{" "}
                      Risk
                    </span>
                    <div className="probability">
                      {console.log(
                        "Probability calculation:",
                        prediction.overall_assessment
                          .combined_flood_probability,
                        "->",
                        (
                          safeNumber(
                            prediction.overall_assessment
                              .combined_flood_probability
                          ) * 100
                        ).toFixed(1)
                      )}
                      {(
                        safeNumber(
                          prediction.overall_assessment
                            .combined_flood_probability
                        ) * 100
                      ).toFixed(1)}
                      % probability
                    </div>
                  </div>
                  <div className="risk-description">
                    {prediction.overall_assessment.recommendation ||
                      "No specific recommendation available"}
                  </div>
                </div>
              )}

              {/* ML Model Results */}
              <div className="ml-models-section">
                <h4>🤖 ML Model Analysis</h4>

                {/* Random Forest Results */}
                {prediction.random_forest && (
                  <div className="model-result">
                    <div className="model-header">
                      <strong>🌳 Random Forest Classifier</strong>
                      <span className="confidence-badge">
                        {prediction.random_forest.confidence || "Medium"}{" "}
                        Confidence
                      </span>
                    </div>
                    <div className="model-details">
                      <span
                        className="risk-level"
                        style={{
                          color: getRiskColor(
                            safeString(
                              prediction.random_forest.risk_level,
                              "Low"
                            )
                          ),
                        }}
                      >
                        {safeString(prediction.random_forest.risk_level, "Low")}{" "}
                        Risk (
                        {(
                          safeNumber(
                            prediction.random_forest.flood_probability
                          ) * 100
                        ).toFixed(1)}
                        %)
                      </span>
                    </div>
                  </div>
                )}

                {/* Bayesian Network Results */}
                {prediction.bayesian_network && (
                  <div className="model-result">
                    <div className="model-header">
                      <strong>🧠 Bayesian Network</strong>
                      <span className="confidence-badge">
                        {prediction.bayesian_network.confidence_interval
                          ? `±${(
                              (prediction.bayesian_network
                                .confidence_interval || 0) * 100
                            ).toFixed(1)}%`
                          : "High Confidence"}
                      </span>
                    </div>
                    <div className="model-details">
                      <span
                        className="risk-level"
                        style={{
                          color: getRiskColor(
                            safeString(
                              prediction.bayesian_network.risk_level,
                              "Low"
                            )
                          ),
                        }}
                      >
                        {safeString(
                          prediction.bayesian_network.risk_level,
                          "Low"
                        )}{" "}
                        Risk (
                        {(
                          safeNumber(
                            prediction.bayesian_network.flood_probability
                          ) * 100
                        ).toFixed(1)}
                        %)
                      </span>
                    </div>
                  </div>
                )}

                {/* Cluster Analysis removed per request */}
              </div>

              <div className="additional-info">
                <div className="info-item">
                  <strong>Models Used:</strong>
                  <span>
                    {prediction.overall_assessment?.models_used || 1} ML Models
                  </span>
                </div>
                <div className="info-item">
                  <strong>Assessment Time:</strong>
                  <span>
                    {new Date(prediction.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <div className="info-item">
                  <strong>Data Source:</strong>
                  <span>Live Weather + ML Models</span>
                </div>
                <div className="info-item">
                  <strong>Input Rainfall:</strong>
                  <span>{prediction.input_data?.rainfall_mm || 0}mm/hour</span>
                </div>
              </div>

              {/* Ward Geographic Information */}
              {prediction.wardDetails && (
                <div className="ward-geographic-info">
                  <h4>Ward Information</h4>
                  <div className="geo-info-grid">
                    <div className="info-item">
                      <strong>Ward Code:</strong>
                      <span>{prediction.wardDetails.code}</span>
                    </div>
                    <div className="info-item">
                      <strong>Object ID:</strong>
                      <span>{prediction.wardDetails.objectId}</span>
                    </div>
                    {prediction.wardDetails.coordinates && (
                      <>
                        <div className="info-item">
                          <strong>Latitude:</strong>
                          <span>
                            {prediction.wardDetails.coordinates.latitude.toFixed(
                              6
                            )}
                          </span>
                        </div>
                        <div className="info-item">
                          <strong>Longitude:</strong>
                          <span>
                            {prediction.wardDetails.coordinates.longitude.toFixed(
                              6
                            )}
                          </span>
                        </div>
                      </>
                    )}
                    <div className="info-item">
                      <strong>Area:</strong>
                      <span>{prediction.wardDetails.area} sq.m</span>
                    </div>
                    <div className="info-item">
                      <strong>Perimeter:</strong>
                      <span>{prediction.wardDetails.perimeter} m</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Risk Level Legend */}
      <div className="risk-legend">
        <h3>Risk Level Legend</h3>
        <div className="legend-items">
          <div className="legend-item">
            <div className="legend-color low"></div>
            <span>Low Risk (&lt;30mm)</span>
          </div>
          <div className="legend-item">
            <div className="legend-color medium"></div>
            <span>Medium Risk (30-60mm)</span>
          </div>
          <div className="legend-item">
            <div className="legend-color high"></div>
            <span>High Risk (&gt;60mm)</span>
          </div>
        </div>
      </div>

      {/* Emergency Contacts */}
      <div className="emergency-contacts">
        <h3>Emergency Contacts</h3>
        <div className="contact-list">
          <div className="contact-item">
            <strong>Mumbai Fire Brigade:</strong> 101
          </div>
          <div className="contact-item">
            <strong>Police:</strong> 100
          </div>
          <div className="contact-item">
            <strong>Ambulance:</strong> 108
          </div>
          <div className="contact-item">
            <strong>Disaster Management:</strong> 1077
          </div>
        </div>
      </div>
    </div>
  );
};

export default WardDetails;
