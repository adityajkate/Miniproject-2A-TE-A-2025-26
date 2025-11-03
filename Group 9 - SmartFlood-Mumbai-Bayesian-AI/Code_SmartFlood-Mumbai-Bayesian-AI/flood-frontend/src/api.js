import axios from "axios";

// Prefer explicit env var, fall back to local FastAPI dev server
const API_BASE_URL =
  process.env.REACT_APP_API_BASE_URL || "http://127.0.0.1:8000";

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000, // 10 second timeout
  headers: {
    "Content-Type": "application/json",
  },
});

// ===== WORKING BACKEND ENDPOINTS ONLY =====

// Check if the API server is running
export const checkServerHealth = async () => {
  try {
    const response = await api.get("/health");
    return response.data;
  } catch (error) {
    console.error("Server health check failed:", error);
    throw new Error(
      "Backend server is not responding. Make sure your FastAPI server is running on http://127.0.0.1:8000"
    );
  }
};

// Get API and model information
export const getApiInfo = async () => {
  try {
    const response = await api.get("/");
    return response.data;
  } catch (error) {
    console.error("Error getting API info:", error);
    throw error;
  }
};

// Get model information
export const getModelInfo = async () => {
  try {
    const response = await api.get("/models/info");
    return response.data;
  } catch (error) {
    console.error("Error getting model info:", error);
    throw error;
  }
};

// Ward-specific prediction
export const getWardFloodPrediction = async (wardCode) => {
  try {
    const encodedWardCode = encodeURIComponent(wardCode);
    const response = await api.post(`/predict/ward/${encodedWardCode}`);
    return response.data;
  } catch (error) {
    console.error(
      `Error getting flood prediction for ward ${wardCode}:`,
      error
    );

    if (error.response?.status === 503) {
      throw new Error(
        "Flood prediction models not initialized. Please check server status."
      );
    } else if (error.response?.status === 404) {
      throw new Error(`Weather data not available for ward ${wardCode}`);
    } else if (error.response?.status === 500) {
      throw new Error(
        `Prediction error for ward ${wardCode}. Please try again.`
      );
    }
    throw error;
  }
};

// Get flood predictions for all Mumbai wards
export const getAllWardsFloodPrediction = async () => {
  try {
    const response = await api.get("/predict/all-wards");
    return response.data;
  } catch (error) {
    console.error("Error getting all wards flood prediction:", error);

    if (error.response?.status === 503) {
      throw new Error(
        "Flood prediction models not initialized. Please check server status."
      );
    }
    throw error;
  }
};

// Custom weather flood prediction
export const predictFloodWithCustomWeather = async (wardCode, weatherData) => {
  try {
    const response = await api.post(`/predict/custom`, {
      ward_code: wardCode,
      ...weatherData,
    });
    return response.data;
  } catch (error) {
    console.error("Error with custom weather prediction:", error);

    if (error.response?.status === 503) {
      throw new Error("Flood prediction models not initialized.");
    } else if (error.response?.status === 400) {
      throw new Error("Invalid weather data provided.");
    }
    throw error;
  }
};

// Get ward clustering information
export const getWardClusters = async () => {
  try {
    const response = await api.get("/wards/clusters");
    return response.data;
  } catch (error) {
    console.error("Error getting ward clusters:", error);

    if (error.response?.status === 503) {
      throw new Error("Clustering model not initialized.");
    }
    throw error;
  }
};

// Get current weather for a ward
export const getCurrentWeather = async (wardCode = "A") => {
  try {
    const response = await api.get(`/weather/current/${wardCode}`);
    return response.data;
  } catch (error) {
    console.error(`Error getting current weather for ward ${wardCode}:`, error);

    if (error.response?.status === 503) {
      throw new Error("Weather service not initialized.");
    }
    throw error;
  }
};

// Retrain models (admin function)
export const retrainModels = async () => {
  try {
    const response = await api.post("/models/retrain");
    return response.data;
  } catch (error) {
    console.error("Error retraining models:", error);
    throw error;
  }
};

// ===== Routing API =====
// Allow longer timeout for routing (graph warm-up and model loading can take time)
const ROUTE_TIMEOUT_MS = Number(
  process.env.REACT_APP_ROUTE_TIMEOUT_MS || 90000
);

export const computeRoute = async ({
  from_lat,
  from_lng,
  to_lat,
  to_lng,
  avoid_threshold = 0.7,
  alpha = 10,
}) => {
  try {
    const response = await api.post(
      "/route",
      {
        from_lat,
        from_lng,
        to_lat,
        to_lng,
        avoid_threshold,
        alpha,
      },
      { timeout: ROUTE_TIMEOUT_MS }
    );
    return response.data;
  } catch (error) {
    console.error("Routing error details:", {
      message: error.message,
      response: error.response?.data,
      status: error.response?.status,
      code: error.code,
    });
    throw error;
  }
};

export const computeDemoRoute = async ({
  from_lat,
  from_lng,
  to_lat,
  to_lng,
  avoid_threshold = 0.7,
  alpha = 10,
  scenario = "central_flood",
}) => {
  try {
    const response = await api.post(
      "/route/demo",
      {
        from_lat,
        from_lng,
        to_lat,
        to_lng,
        avoid_threshold,
        alpha,
        scenario,
      },
      { timeout: ROUTE_TIMEOUT_MS }
    );
    return response.data;
  } catch (error) {
    console.error("Demo routing error details:", {
      message: error.message,
      response: error.response?.data,
      status: error.response?.status,
      code: error.code,
    });
    throw error;
  }
};

const GRAPH_REFRESH_TIMEOUT_MS = Number(
  process.env.REACT_APP_GRAPH_REFRESH_TIMEOUT_MS || 120000
);

export const refreshRoutingGraph = async () => {
  const response = await api.get("/graph/refresh", {
    timeout: GRAPH_REFRESH_TIMEOUT_MS,
  });
  return response.data;
};

// ===== UTILITY FUNCTIONS FOR FRONTEND =====

// Helper function to determine risk color for map visualization
export const getRiskColor = (riskLevel) => {
  switch (riskLevel?.toLowerCase()) {
    case "critical":
      return "#FF0000"; // Red
    case "high":
      return "#FF6600"; // Orange
    case "medium":
      return "#FFFF00"; // Yellow
    case "low":
      return "#00FF00"; // Green
    default:
      return "#CCCCCC"; // Gray
  }
};

// Helper function to get risk level from probability
export const getRiskLevel = (probability) => {
  if (probability >= 0.7) return "Critical";
  if (probability >= 0.3) return "High";
  if (probability >= 0.1) return "Medium";
  return "Low";
};

// Helper function to format flood prediction response for UI
export const formatFloodPrediction = (prediction) => {
  if (!prediction) return null;

  // Safe property access with fallbacks
  const randomForest = prediction.random_forest || {};
  const combinedAssessment = prediction.combined_assessment || {};
  const bayesianProbability = prediction.bayesian_probability ?? 0;
  const wardRiskZone = prediction.ward_risk_zone || "Unknown";

  // Normalize confidence to numeric [0,1]
  const rawConf = combinedAssessment?.confidence;
  const confidence =
    typeof rawConf === "number"
      ? Math.max(0, Math.min(1, rawConf))
      : typeof rawConf === "string"
      ? {
          "Very High": 0.9,
          High: 0.75,
          Medium: 0.5,
          Low: 0.25,
        }[rawConf] ?? 0.5
      : 0.5;

  const rfRiskLevelNum = randomForest.flood_risk_level ?? 0;
  const rfRiskProbs = randomForest.risk_probabilities || {};
  const riskProbabilities = {
    low: rfRiskProbs.low ?? 0,
    medium: rfRiskProbs.medium ?? 0,
    high: rfRiskProbs.high ?? 0,
  };

  const riskLevelToString = (lvl) =>
    lvl === 2 ? "high" : lvl === 1 ? "medium" : "low";

  return {
    wardCode: prediction.ward_code,
    wardName: prediction.ward_name,
    riskZone: wardRiskZone,
    riskLevel: rfRiskLevelNum,

    // Added fields for map compatibility
    risk_level: riskLevelToString(rfRiskLevelNum),
    flood_risk_probability: bayesianProbability,

    willFlood: randomForest.will_flood || false,
    riskProbabilities,
    bayesianProbability,
    highRisk: combinedAssessment.high_risk || false,
    confidence,
    timestamp: prediction.timestamp || new Date().toISOString(),
    weatherData: prediction.weather_data || {},
  };
};

// Helper function to get risk color based on flood prediction
export const getFloodRiskColor = (riskLevel) => {
  switch (riskLevel) {
    case 2:
      return "#FF0000"; // High - Red
    case 1:
      return "#FF6600"; // Medium - Orange
    case 0:
      return "#00FF00"; // Low - Green
    default:
      return "#CCCCCC"; // Unknown - Gray
  }
};

// Helper function to get risk zone color
export const getRiskZoneColor = (riskZone) => {
  switch (riskZone) {
    case "Very High Risk":
      return "#8B0000"; // Dark Red
    case "High Risk":
      return "#FF4500"; // Orange Red
    case "Medium Risk":
      return "#FFD700"; // Gold
    case "Low Risk":
      return "#32CD32"; // Lime Green
    default:
      return "#CCCCCC"; // Gray
  }
};

// Mumbai ward coordinates (approximate centroids)
export const MUMBAI_WARD_COORDINATES = {
  A: { latitude: 18.9067, longitude: 72.8147, name: "Colaba" },
  B: { latitude: 18.9322, longitude: 72.8264, name: "Fort" },
  C: { latitude: 18.948, longitude: 72.8258, name: "Marine Lines" },
  D: { latitude: 18.9696, longitude: 72.8448, name: "Mazgaon" },
  E: { latitude: 18.975, longitude: 72.8342, name: "Byculla" },
  "F/N": { latitude: 19.0138, longitude: 72.8452, name: "Parel" },
  "F/S": { latitude: 19.0008, longitude: 72.83, name: "Lower Parel" },
  "G/N": { latitude: 19.0176, longitude: 72.8562, name: "Dadar" },
  "G/S": { latitude: 19.033, longitude: 72.857, name: "Mahim" },
  "H/E": { latitude: 19.0596, longitude: 72.8656, name: "Bandra East" },
  "H/W": { latitude: 19.0596, longitude: 72.8295, name: "Bandra West" },
  "K/E": { latitude: 19.1136, longitude: 72.8697, name: "Andheri East" },
  "K/W": { latitude: 19.1197, longitude: 72.8464, name: "Andheri West" },
  L: { latitude: 19.0728, longitude: 72.8826, name: "Kurla" },
  "M/E": { latitude: 19.033, longitude: 72.899, name: "Chembur" },
  "M/W": { latitude: 19.027, longitude: 72.95, name: "Trombay" },
  N: { latitude: 19.0896, longitude: 72.9081, name: "Ghatkopar" },
  "P/N": { latitude: 19.1872, longitude: 72.8495, name: "Malad" },
  "P/S": { latitude: 19.2094, longitude: 72.8526, name: "Kandivali" },
  "R/C": { latitude: 19.2307, longitude: 72.8567, name: "Borivali" },
  "R/N": { latitude: 19.2544, longitude: 72.8656, name: "Dahisar" },
  "R/S": { latitude: 19.2094, longitude: 72.87, name: "Kandivali East" },
  S: { latitude: 19.145, longitude: 72.9342, name: "Bhandup" },
  T: { latitude: 19.1728, longitude: 72.9342, name: "Mulund" },
};

// Helper function to get ward coordinates
export const getWardCoordinates = (wardId) => {
  return (
    MUMBAI_WARD_COORDINATES[wardId] || {
      latitude: 19.076,
      longitude: 72.8777,
      name: "Mumbai Center",
    }
  );
};

// Mumbai ward codes for dropdown
export const MUMBAI_WARD_CODES = [
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
];

// Ward names mapping
export const WARD_NAMES = {
  A: "Colaba, Churchgate",
  B: "Masjid Bunder",
  C: "Marine Lines",
  D: "Grant Road",
  E: "Byculla, Mazgaon",
  "F/N": "Matunga",
  "F/S": "Parel",
  "G/N": "Dadar",
  "G/S": "Worli",
  "H/E": "Santacruz East",
  "H/W": "Bandra West",
  "K/E": "Vile Parle East",
  "K/W": "Andheri West",
  L: "Kurla",
  "M/E": "Mankhurd",
  "M/W": "Chembur",
  N: "Ghatkopar",
  "P/N": "Malad",
  "P/S": "Goregaon",
  "R/C": "Borivali",
  "R/N": "Dahisar",
  "R/S": "Kandivali",
  S: "Powai",
  T: "Mulund",
};

// Get current season based on date (for Mumbai)
export const getCurrentSeason = () => {
  const month = new Date().getMonth() + 1; // 1-12

  if (month >= 6 && month <= 9) {
    return "monsoon"; // June to September
  } else if (month >= 10 && month <= 11) {
    return "post-monsoon"; // October to November
  } else if (month >= 12 || month <= 2) {
    return "winter"; // December to February
  } else {
    return "summer"; // March to May
  }
};

// ===== LEGACY FUNCTION ALIASES FOR COMPATIBILITY =====

// Aliases for backward compatibility with existing components
export const checkFloodApiHealth = checkServerHealth;
export const getFloodModelInfo = getModelInfo;
export const getWardCurrentWeather = getCurrentWeather;
export const predictSingleWard = getWardFloodPrediction;
export const getWardClustering = getWardClusters;

// Simplified functions for common use cases
export const predictMultipleWards = async () => {
  try {
    const response = await getAllWardsFloodPrediction();

    // Transform to expected format for existing components
    const ward_predictions = {};
    if (response.predictions) {
      response.predictions.forEach((prediction) => {
        const wardId = prediction.ward_code;

        // Determine risk level based on Random Forest flood_risk_level
        let risk_level = "Low";
        const floodRiskLevel = prediction.random_forest?.flood_risk_level || 0;
        switch (floodRiskLevel) {
          case 2:
            risk_level = "High";
            break;
          case 1:
            risk_level = "Medium";
            break;
          case 0:
          default:
            risk_level = "Low";
            break;
        }

        ward_predictions[wardId] = {
          risk_level: risk_level,
          flood_probability: prediction.bayesian_probability || 0,
          ward_id: wardId,
          ward_name: prediction.ward_name,
        };
      });
    }

    return {
      ward_predictions,
      summary: response.summary || {},
      timestamp: response.timestamp || new Date().toISOString(),
    };
  } catch (error) {
    console.error("Error predicting multiple wards:", error);
    throw error;
  }
};

export const getAvailableWards = async () => {
  try {
    const response = await getWardClusters();
    const wards =
      response.ward_clusters?.map((cluster) => ({
        ward_id: cluster.ward_code,
        ward_name: WARD_NAMES[cluster.ward_code] || cluster.ward_code,
        zone: cluster.risk_zone,
      })) || [];

    return { wards };
  } catch (error) {
    console.error("Error fetching available wards:", error);
    throw new Error("Unable to load ward data from backend");
  }
};

export const getWardCurrentStatus = async (wardId) => {
  try {
    const prediction = await getWardFloodPrediction(wardId);

    return {
      status: "current",
      risk_level: prediction.combined_assessment?.high_risk ? "High" : "Low",
      current_flood_probability: prediction.bayesian_probability || 0,
      last_updated: prediction.timestamp || new Date().toISOString(),
      data_freshness: "current",
      recommendation: prediction.combined_assessment?.high_risk
        ? "High flood risk detected. Monitor conditions closely."
        : "Low flood risk. Normal conditions expected.",
    };
  } catch (error) {
    console.error("Error fetching ward current status:", error);
    throw new Error("Unable to fetch current ward status from backend");
  }
};
