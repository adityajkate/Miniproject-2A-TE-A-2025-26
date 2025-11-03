import React, { useState, useEffect } from "react";
import { getWardCurrentWeather } from "./api";
import "./WeatherWidget.css";

const WeatherWidget = ({ lat, lon, selectedWard, onWeatherUpdate }) => {
  const [weather, setWeather] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  const fetchWeatherData = async () => {
    setLoading(true);
    setError(null);

    try {
      // Use the selected ward, or default to 'A' (Colaba) if no ward is selected
      const wardToFetch = selectedWard || "A";
      const weatherData = await getWardCurrentWeather(wardToFetch);

      // Transform the data to match the expected format with proper formatting
      const actualTemp = weatherData?.Temperature_C || 0;
      const humidity = weatherData?.["Humidity_%"] || 0;
      const feelsLikeTemp = calculateFeelsLike(actualTemp, humidity);

      const transformedWeather = {
        location: {
          name: weatherData?.ward_name || "Mumbai",
          country: "India",
        },
        current: {
          temperature: Math.round(actualTemp),
          feels_like: Math.round(feelsLikeTemp),
          humidity: Math.round(humidity),
          pressure: 1013, // Default pressure if not available
          wind_speed: weatherData?.Wind_Speed_kmh
            ? Math.round((weatherData.Wind_Speed_kmh / 3.6) * 10) / 10 // Convert to m/s and round to 1 decimal
            : 0,
          visibility: 10, // Default visibility
          weather: {
            description: weatherData?.weather_description || "Clear",
            icon: "01d", // Default icon
          },
          rainfall: {
            last_1h: Math.round((weatherData?.Rainfall_mm || 0) * 10) / 10,
            last_3h: Math.round((weatherData?.Rainfall_24hr || 0) * 10) / 10,
          },
        },
      };

      // Validate that essential weather data is available
      if (
        !weatherData?.Temperature_C ||
        !weatherData?.["Humidity_%"] ||
        weatherData?.Rainfall_mm === undefined
      ) {
        throw new Error("Essential weather data not available from backend");
      }

      setWeather(transformedWeather);
      setAlerts([]); // No alerts for now
      setLastUpdated(new Date());

      // Notify parent component of weather update
      if (onWeatherUpdate) {
        onWeatherUpdate(transformedWeather, []);
      }
    } catch (err) {
      console.error("Weather fetch error:", err);
      setError(err.message);
      // Don't set any fallback data - let the component show error state
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchWeatherData();

    // Auto-refresh every 10 minutes
    const interval = setInterval(fetchWeatherData, 10 * 60 * 1000);
    return () => clearInterval(interval);
  }, [lat, lon, selectedWard]);

  const getWeatherIcon = (iconCode) => {
    return `https://openweathermap.org/img/wn/${iconCode}@2x.png`;
  };

  // Calculate "feels like" temperature using heat index formula
  const calculateFeelsLike = (tempC, humidity) => {
    // Convert Celsius to Fahrenheit for heat index calculation
    const tempF = (tempC * 9) / 5 + 32;

    // Heat index formula (simplified version)
    if (tempF < 80) {
      // For temperatures below 80°F, feels like is approximately the same as actual temperature
      return tempC;
    }

    // Heat index calculation for higher temperatures
    const hi =
      -42.379 +
      2.04901523 * tempF +
      10.14333127 * humidity -
      0.22475541 * tempF * humidity -
      6.83783e-3 * tempF * tempF -
      5.481717e-2 * humidity * humidity +
      1.22874e-3 * tempF * tempF * humidity +
      8.5282e-4 * tempF * humidity * humidity -
      1.99e-6 * tempF * tempF * humidity * humidity;

    // Convert back to Celsius
    const feelsLikeC = ((hi - 32) * 5) / 9;

    // Return the higher of actual temperature or heat index
    return Math.max(tempC, feelsLikeC);
  };

  const getRainfallStatus = (rainfall) => {
    if (rainfall > 20) return { level: "extreme", text: "Extreme" };
    if (rainfall > 10) return { level: "heavy", text: "Heavy" };
    if (rainfall > 2) return { level: "moderate", text: "Moderate" };
    if (rainfall > 0) return { level: "light", text: "Light" };
    return { level: "none", text: "No Rain" };
  };

  const getAlertSeverityClass = (severity) => {
    switch (severity) {
      case "high":
        return "alert-high";
      case "medium":
        return "alert-medium";
      case "low":
        return "alert-low";
      default:
        return "alert-info";
    }
  };

  if (loading && !weather) {
    return (
      <div className="weather-widget loading">
        <div className="weather-header">
          <h3>Current Weather</h3>
        </div>
        <div className="loading-spinner">Loading weather data...</div>
      </div>
    );
  }

  if (error && !weather) {
    return (
      <div className="weather-widget error">
        <div className="weather-header">
          <h3>Current Weather</h3>
        </div>
        <div className="error-message">
          <p>Weather service unavailable</p>
          <button onClick={fetchWeatherData} className="retry-btn">
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!weather) {
    return null;
  }

  const current = weather.current;
  const rainfallStatus = getRainfallStatus(current.rainfall.last_1h);

  return (
    <div className="weather-widget">
      <div className="weather-header">
        <h3>Current Weather</h3>
        <div className="weather-location">
          {weather.location.name}, {weather.location.country}
        </div>
        {lastUpdated && (
          <div className="last-updated">
            Updated: {lastUpdated.toLocaleTimeString()}
          </div>
        )}
      </div>

      <div className="weather-main">
        <div className="weather-primary">
          <div className="weather-icon">
            <img
              src={getWeatherIcon(current.weather.icon)}
              alt={current.weather.description}
            />
          </div>
          <div className="weather-temp">
            <span className="temp-value">{current.temperature}°C</span>
            <span className="temp-feels">
              Feels like {current.feels_like}°C
            </span>
          </div>
        </div>

        <div className="weather-description">
          {current.weather.description.charAt(0).toUpperCase() +
            current.weather.description.slice(1)}
        </div>
      </div>

      <div className="weather-details">
        <div className="weather-detail-row">
          <div className="weather-detail">
            <span className="detail-label">Humidity</span>
            <span className="detail-value">{current.humidity}%</span>
          </div>
          <div className="weather-detail">
            <span className="detail-label">Wind</span>
            <span className="detail-value">{current.wind_speed} m/s</span>
          </div>
        </div>

        <div className="weather-detail-row">
          <div className="weather-detail">
            <span className="detail-label">Pressure</span>
            <span className="detail-value">{current.pressure} hPa</span>
          </div>
          <div className="weather-detail">
            <span className="detail-label">Visibility</span>
            <span className="detail-value">{current.visibility} km</span>
          </div>
        </div>

        <div className="rainfall-section">
          <div className="rainfall-header">
            <span className="detail-label">Rainfall</span>
            <span className={`rainfall-status ${rainfallStatus.level}`}>
              {rainfallStatus.text}
            </span>
          </div>
          <div className="rainfall-details">
            <div className="rainfall-item">
              <span>Last hour:</span>
              <span>{current.rainfall.last_1h} mm</span>
            </div>
            <div className="rainfall-item">
              <span>Last 3 hours:</span>
              <span>{current.rainfall.last_3h} mm</span>
            </div>
          </div>
        </div>
      </div>

      {alerts.length > 0 && (
        <div className="weather-alerts">
          <div className="alerts-header">
            <span className="alert-icon">⚠️</span>
            Weather Alerts ({alerts.length})
          </div>
          <div className="alerts-list">
            {alerts.map((alert, index) => (
              <div
                key={index}
                className={`alert-item ${getAlertSeverityClass(
                  alert.severity
                )}`}
              >
                <div className="alert-title">{alert.title}</div>
                <div className="alert-description">{alert.description}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="weather-actions">
        <button
          onClick={fetchWeatherData}
          disabled={loading}
          className="refresh-weather-btn"
        >
          {loading ? "Updating..." : "Refresh"}
        </button>
      </div>
    </div>
  );
};

export default WeatherWidget;
