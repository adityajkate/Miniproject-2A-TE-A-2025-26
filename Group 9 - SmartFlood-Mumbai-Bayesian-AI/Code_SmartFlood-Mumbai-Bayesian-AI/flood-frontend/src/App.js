import React, { useState, useEffect, useRef } from "react";
import Dashboard from "./Dashboard";
import {
  checkFloodApiHealth,
  getAllWardsFloodPrediction,
  formatFloodPrediction,
} from "./api";
import "./App.css";
import Preloader from "./Preloader";

function App() {
  const [predictions, setPredictions] = useState({});
  const [selectedWard, setSelectedWard] = useState(null);
  const [loading, setLoading] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(new Date());
  const [apiStatus, setApiStatus] = useState("checking");
  const [error, setError] = useState(null);
  // Ensure preloader stays visible for a short initial boot period
  const [initialHold, setInitialHold] = useState(true);
  useEffect(() => {
    const t = setTimeout(() => setInitialHold(false), 1200); // adjust duration as needed
    return () => clearTimeout(t);
  }, []);

  // Check API health on startup
  useEffect(() => {
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      setApiStatus("checking");
      const health = await checkFloodApiHealth();
      setApiStatus("connected");
      console.log("Flood Prediction API connected:", health);

      // Load initial predictions if API is healthy
      if (health.models_loaded) {
        await loadAllPredictions();
      }
    } catch (error) {
      setApiStatus("error");
      setError(error.message);
      console.error("API health check failed:", error);
    }
  };

  const MIN_PRELOAD_MS = 900; // keep preloader visible at least this long
  const preloadStartRef = useRef(null);

  const loadAllPredictions = async () => {
    try {
      setError(null);
      preloadStartRef.current = Date.now();
      setLoading(true);

      const response = await getAllWardsFloodPrediction();

      // Format predictions for the UI
      const formattedPredictions = {};
      response.predictions.forEach((prediction) => {
        const formatted = formatFloodPrediction(prediction);
        formattedPredictions[formatted.wardCode] = formatted;
      });

      setPredictions(formattedPredictions);
      setLastUpdated(new Date());

      console.log(
        "Loaded predictions for",
        Object.keys(formattedPredictions).length,
        "wards"
      );
      console.log(
        "Sample prediction:",
        formattedPredictions[Object.keys(formattedPredictions)[0]]
      );
      console.log("All predictions:", formattedPredictions);
    } catch (error) {
      setError(error.message);
      console.error("Error loading predictions:", error);
    } finally {
      const elapsed = Date.now() - (preloadStartRef.current || Date.now());
      const remaining = Math.max(0, MIN_PRELOAD_MS - elapsed);
      setTimeout(() => setLoading(false), remaining);
    }
  };

  const handleRefresh = () => {
    if (apiStatus === "connected") {
      loadAllPredictions();
    } else {
      checkApiHealth();
    }
  };

  const handleWardSelect = (wardCode) => {
    setSelectedWard(wardCode);
  };

  return (
    <div className="App">
      {/* Global preloader during initial connect or bulk load */}
      <Preloader
        show={
          initialHold ||
          apiStatus === "checking" ||
          (apiStatus === "connected" && loading)
        }
        statusText={
          apiStatus === "checking"
            ? "Connecting to backend and loading models..."
            : loading
            ? "Fetching flood predictions for all wards..."
            : "Launching dashboard..."
        }
      />

      <header className="app-header">
        <h1>Mumbai Flood Prediction System</h1>
        <div className="api-status">
          <span className={`status-indicator ${apiStatus}`}>
            {apiStatus === "checking" && "🔄 Connecting..."}
            {apiStatus === "connected" && "✅ Connected"}
            {apiStatus === "error" && "❌ Disconnected"}
          </span>
          {apiStatus === "connected" && (
            <span className="model-info">
              AI Models: Random Forest + Bayesian Network
            </span>
          )}
        </div>
      </header>

      {error && (
        <div className="error-banner">
          <div className="error-content">
            <span className="error-icon">⚠️</span>
            <div className="error-details">
              <strong>Backend Connection Error:</strong> {error}
              <div className="error-suggestions">
                <p>Please ensure:</p>
                <ul>
                  <li>
                    ✅ Backend server is running on{" "}
                    <code>http://localhost:8000</code>
                  </li>
                  <li>✅ All ML models are trained and loaded</li>
                  <li>✅ Weather service is configured properly</li>
                </ul>
              </div>
            </div>
          </div>
          <button onClick={checkApiHealth} className="retry-btn">
            🔄 Retry Connection
          </button>
        </div>
      )}

      <Dashboard
        predictions={predictions}
        setPredictions={setPredictions}
        selectedWard={selectedWard}
        setSelectedWard={handleWardSelect}
        loading={loading}
        setLoading={setLoading}
        lastUpdated={lastUpdated}
        onRefresh={handleRefresh}
        apiStatus={apiStatus}
      />
    </div>
  );
}

export default App;
