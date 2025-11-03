import React, { useState } from "react";
import MapComponent from "./MapComponent";
import StatsCards from "./StatsCards";
import WardDetails from "./WardDetails";
import WeatherWidget from "./WeatherWidget";
import WardWeatherDetails from "./WardWeatherDetails";

import FloodPredictionPanel from "./FloodPredictionPanel";
import RoutingMap from "./RoutingMap.jsx";
import "./Dashboard.css";

const Dashboard = ({
  predictions,
  setPredictions,
  selectedWard,
  setSelectedWard,
  loading,
  setLoading,
  lastUpdated,
  onRefresh,
  apiStatus,
}) => {
  const [currentWeather, setCurrentWeather] = useState(null);
  const [weatherAlerts, setWeatherAlerts] = useState([]);
  const [wardCoordinates, setWardCoordinates] = useState(null);
  const [activeTab, setActiveTab] = useState("map"); // 'map' | 'routing'

  const handleWeatherUpdate = (weatherData, alerts) => {
    setCurrentWeather(weatherData);
    setWeatherAlerts(alerts);
  };

  const handleWardCoordinatesChange = (coordinates) => {
    setWardCoordinates(coordinates);
  };
  return (
    <div className="dashboard">
      {/* Header */}
      <header className="dashboard-header">
        <div className="header-left">
          <div className="header-icon">
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <circle
                cx="12"
                cy="12"
                r="3"
                stroke="currentColor"
                strokeWidth="2"
              />
              <path
                d="M12 1v6m0 6v6m11-7h-6m-6 0H1"
                stroke="currentColor"
                strokeWidth="2"
              />
            </svg>
          </div>
          <div className="header-text">
            <h1>Mumbai Flood Risk Monitor</h1>
            <p>Real-time ML-powered flood risk predictions</p>
          </div>
        </div>
        <div className="header-right">
          <div className="last-updated">
            Last Updated:{" "}
            {lastUpdated.toLocaleString("en-US", {
              month: "numeric",
              day: "numeric",
              year: "numeric",
              hour: "numeric",
              minute: "2-digit",
              hour12: true,
            })}
          </div>
          <button className="refresh-btn" onClick={onRefresh}>
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M4 4V10H10"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path
                d="M20 12A8 8 0 1 1 11.93 4.06L10 6"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
            Refresh
          </button>
        </div>
      </header>

      {/* Stats Cards */}
      <StatsCards predictions={predictions} />

      {/* Tab Navigation */}
      <div className="tab-navigation">
        <button
          className={`tab-button ${activeTab === "map" ? "active" : ""}`}
          onClick={() => setActiveTab("map")}
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M21 10C21 17 12 23 12 23S3 17 3 10A9 9 0 0 1 21 10Z"
              stroke="currentColor"
              strokeWidth="2"
            />
            <circle
              cx="12"
              cy="10"
              r="3"
              stroke="currentColor"
              strokeWidth="2"
            />
          </svg>
          Interactive Map
        </button>
        <button
          className={`tab-button ${activeTab === "routing" ? "active" : ""}`}
          onClick={() => setActiveTab("routing")}
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M3 12h6l3-6 3 12 3-6h3"
              stroke="currentColor"
              strokeWidth="2"
              fill="none"
            />
          </svg>
          Flood-aware Routing
        </button>
      </div>

      {/* Main Content */}
      <div className="dashboard-content">
        {activeTab === "map" && (
          <div className="map-tab-content">
            {/* Map Section */}
            <div className="map-section">
              <div className="section-header">
                <div className="section-icon">
                  <svg
                    width="18"
                    height="18"
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
                <h2>Interactive Flood Risk Map</h2>
              </div>
              <div className="map-container">
                <MapComponent
                  predictions={predictions}
                  setPredictions={setPredictions}
                  selectedWard={selectedWard}
                  setSelectedWard={setSelectedWard}
                  loading={loading}
                  setLoading={setLoading}
                  onWardCoordinatesChange={handleWardCoordinatesChange}
                />
              </div>
            </div>

            {/* Ward Details Section */}
            <div className="details-section">
              {/* Weather Widget */}
              <WeatherWidget
                lat={19.076} // Mumbai coordinates
                lon={72.8777}
                selectedWard={selectedWard}
                onWeatherUpdate={handleWeatherUpdate}
              />

              {/* Live Ward Weather Details */}
              <WardWeatherDetails
                selectedWard={selectedWard}
                wardCoordinates={wardCoordinates}
              />

              {/* New Flood Prediction Panel */}
              <FloodPredictionPanel
                selectedWard={selectedWard}
                onPredictionUpdate={(wardCode, prediction) => {
                  setPredictions((prev) => ({
                    ...prev,
                    [wardCode]: prediction,
                  }));
                }}
              />

              <WardDetails
                selectedWard={selectedWard}
                prediction={selectedWard ? predictions[selectedWard] : null}
                loading={loading}
                currentWeather={currentWeather}
                weatherAlerts={weatherAlerts}
              />
            </div>
          </div>
        )}

        {activeTab === "routing" && (
          <div className="routing-section" style={{ paddingTop: 12 }}>
            <RoutingMap />
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
