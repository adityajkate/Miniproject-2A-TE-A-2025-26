import React, { useState, useEffect } from "react";
import { getWardCurrentWeather, getWardFloodPrediction } from "./api";
import "./WardWeatherDetails.css";

const WardWeatherDetails = ({ selectedWard, wardCoordinates }) => {
  const [liveData, setLiveData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  const fetchLiveWeatherPrediction = async () => {
    if (!selectedWard) return;

    setLoading(true);
    setError(null);

    try {
      // Get both weather data and flood prediction
      const [weatherData, floodPrediction] = await Promise.all([
        getWardCurrentWeather(selectedWard),
        getWardFloodPrediction(selectedWard)
      ]);

      // Transform the data to match the expected format with safe property access
      const combinedAssessment = floodPrediction.combined_assessment || {};
      const randomForest = floodPrediction.random_forest || {};
      const bayesianProbability = floodPrediction.bayesian_probability || 0;
      
      const transformedData = {
        warning: {
          level: combinedAssessment.high_risk ? 'high' : 'low',
          message: combinedAssessment.high_risk 
            ? 'High flood risk detected in this area' 
            : 'Low flood risk in this area',
          flood_probability: bayesianProbability
        },
        live_weather: {
          temperature: weatherData.Temperature_C || 25,
          humidity: weatherData['Humidity_%'] || 70,
          wind_speed: weatherData.Wind_Speed_kmh ? (weatherData.Wind_Speed_kmh / 3.6).toFixed(1) : 5,
          weather_description: weatherData.weather_description || 'Clear sky',
          current_rainfall_1h: weatherData.Rainfall_mm || 0,
          current_rainfall_3h: weatherData.Rainfall_24hr || 0,
          location: {
            name: weatherData.ward_name || selectedWard,
            country: 'India'
          }
        },
        flood_prediction: {
          overall_assessment: {
            combined_risk_level: randomForest.flood_risk_level === 2 ? 'High' : 
                                randomForest.flood_risk_level === 1 ? 'Medium' : 'Low',
            models_used: 'Random Forest + Bayesian Network + K-means',
            recommendation: combinedAssessment.high_risk 
              ? 'Avoid travel to this area. Monitor conditions closely.'
              : 'Normal conditions. Continue monitoring weather updates.'
          },
          random_forest: {
            confidence: combinedAssessment.confidence || 'Medium'
          }
        },
        weather_alerts: [],
        timestamp: floodPrediction.timestamp || new Date().toISOString()
      };

      setLiveData(transformedData);
      setLastUpdated(new Date());
    } catch (err) {
      console.error("Live weather prediction error:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (selectedWard) {
      fetchLiveWeatherPrediction();
      
      // Auto-refresh every 5 minutes
      const interval = setInterval(fetchLiveWeatherPrediction, 5 * 60 * 1000);
      return () => clearInterval(interval);
    }
  }, [selectedWard, wardCoordinates]);

  const getWarningIcon = (level) => {
    switch (level) {
      case "critical": return "🚨";
      case "high": return "⚠️";
      case "medium": return "🟡";
      case "low": return "🟢";
      default: return "✅";
    }
  };

  const getWarningClass = (level) => {
    switch (level) {
      case "critical": return "warning-critical";
      case "high": return "warning-high";
      case "medium": return "warning-medium";
      case "low": return "warning-low";
      default: return "warning-none";
    }
  };

  const getRainfallIntensity = (rainfall) => {
    if (rainfall > 50) return { text: "Extreme", class: "extreme" };
    if (rainfall > 20) return { text: "Heavy", class: "heavy" };
    if (rainfall > 10) return { text: "Moderate", class: "moderate" };
    if (rainfall > 2) return { text: "Light", class: "light" };
    if (rainfall > 0) return { text: "Drizzle", class: "drizzle" };
    return { text: "No Rain", class: "none" };
  };

  if (!selectedWard) {
    return (
      <div className="ward-weather-details">
        <div className="no-ward-selected">
          <div className="no-selection-icon">🌦️</div>
          <h3>Live Weather & Flood Risk</h3>
          <p>Click on a ward to get live weather data and real-time flood risk assessment</p>
        </div>
      </div>
    );
  }

  return (
    <div className="ward-weather-details">
      <div className="weather-header">
        <h3>🌦️ Live Weather & Flood Risk</h3>
        <div className="ward-name">{selectedWard}</div>
        {lastUpdated && (
          <div className="last-updated">
            Updated: {lastUpdated.toLocaleTimeString()}
          </div>
        )}
      </div>

      {loading && (
        <div className="loading-section">
          <div className="loading-spinner"></div>
          <p>Fetching live weather data...</p>
        </div>
      )}

      {error && (
        <div className="error-section">
          <div className="error-icon">❌</div>
          <p>Weather service unavailable</p>
          <small>{error}</small>
          <button onClick={fetchLiveWeatherPrediction} className="retry-btn">
            Retry
          </button>
        </div>
      )}

      {liveData && !loading && (
        <div className="live-data-content">
          {/* Warning Alert */}
          <div className={`flood-warning ${getWarningClass(liveData.warning.level)}`}>
            <div className="warning-header">
              <span className="warning-icon">{getWarningIcon(liveData.warning.level)}</span>
              <span className="warning-title">
                {liveData.warning.level.toUpperCase()} FLOOD RISK
              </span>
            </div>
            <div className="warning-message">{liveData.warning.message}</div>
            <div className="flood-probability">
              Flood Probability: {(liveData.warning.flood_probability * 100).toFixed(1)}%
            </div>
          </div>

          {/* Current Weather */}
          <div className="current-weather-section">
            <h4>Current Weather Conditions</h4>
            <div className="weather-grid">
              <div className="weather-item">
                <span className="weather-label">Temperature</span>
                <span className="weather-value">
                  {Math.round(liveData.live_weather.temperature)}°C
                </span>
              </div>
              <div className="weather-item">
                <span className="weather-label">Humidity</span>
                <span className="weather-value">{liveData.live_weather.humidity}%</span>
              </div>
              <div className="weather-item">
                <span className="weather-label">Wind Speed</span>
                <span className="weather-value">{liveData.live_weather.wind_speed} m/s</span>
              </div>
              <div className="weather-item">
                <span className="weather-label">Conditions</span>
                <span className="weather-value">
                  {liveData.live_weather.weather_description}
                </span>
              </div>
            </div>
          </div>

          {/* Rainfall Analysis */}
          <div className="rainfall-section">
            <h4>Rainfall Analysis</h4>
            <div className="rainfall-grid">
              <div className="rainfall-item">
                <div className="rainfall-header">
                  <span>Current Hour</span>
                  <span className={`rainfall-badge ${getRainfallIntensity(liveData.live_weather.current_rainfall_1h).class}`}>
                    {getRainfallIntensity(liveData.live_weather.current_rainfall_1h).text}
                  </span>
                </div>
                <div className="rainfall-value">
                  {liveData.live_weather.current_rainfall_1h.toFixed(1)} mm
                </div>
              </div>
              <div className="rainfall-item">
                <div className="rainfall-header">
                  <span>Last 3 Hours</span>
                  <span className={`rainfall-badge ${getRainfallIntensity(liveData.live_weather.current_rainfall_3h / 3).class}`}>
                    {getRainfallIntensity(liveData.live_weather.current_rainfall_3h / 3).text}
                  </span>
                </div>
                <div className="rainfall-value">
                  {liveData.live_weather.current_rainfall_3h.toFixed(1)} mm
                </div>
              </div>
            </div>
          </div>

          {/* ML Prediction Details */}
          {liveData.flood_prediction.overall_assessment && (
            <div className="ml-prediction-section">
              <h4>🤖 ML Model Analysis</h4>
              <div className="prediction-grid">
                <div className="prediction-item">
                  <span className="prediction-label">Combined Risk Level</span>
                  <span className="prediction-value">
                    {liveData.flood_prediction.overall_assessment.combined_risk_level}
                  </span>
                </div>
                <div className="prediction-item">
                  <span className="prediction-label">Models Used</span>
                  <span className="prediction-value">
                    {liveData.flood_prediction.overall_assessment.models_used}
                  </span>
                </div>
                <div className="prediction-item">
                  <span className="prediction-label">Confidence</span>
                  <span className="prediction-value">
                    {liveData.flood_prediction.random_forest?.confidence || "Medium"}
                  </span>
                </div>
              </div>
              <div className="recommendation">
                <strong>Recommendation:</strong>
                <p>{liveData.flood_prediction.overall_assessment.recommendation}</p>
              </div>
            </div>
          )}

          {/* Weather Alerts */}
          {liveData.weather_alerts && liveData.weather_alerts.length > 0 && (
            <div className="weather-alerts-section">
              <h4>⚠️ Weather Alerts</h4>
              <div className="alerts-list">
                {liveData.weather_alerts.map((alert, index) => (
                  <div key={index} className={`alert-item alert-${alert.severity || 'info'}`}>
                    <div className="alert-title">{alert.title}</div>
                    <div className="alert-description">{alert.description}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Location Info */}
          <div className="location-section">
            <h4>📍 Location Details</h4>
            <div className="location-info">
              <div className="location-item">
                <span>Area:</span>
                <span>{liveData.live_weather.location.name}, {liveData.live_weather.location.country}</span>
              </div>
              <div className="location-item">
                <span>Data Source:</span>
                <span>Live Weather API</span>
              </div>
              <div className="location-item">
                <span>Last Update:</span>
                <span>{new Date(liveData.timestamp).toLocaleString()}</span>
              </div>
            </div>
          </div>

          {/* Refresh Button */}
          <div className="actions-section">
            <button 
              onClick={fetchLiveWeatherPrediction} 
              disabled={loading}
              className="refresh-btn"
            >
              {loading ? "Updating..." : "🔄 Refresh Live Data"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default WardWeatherDetails;