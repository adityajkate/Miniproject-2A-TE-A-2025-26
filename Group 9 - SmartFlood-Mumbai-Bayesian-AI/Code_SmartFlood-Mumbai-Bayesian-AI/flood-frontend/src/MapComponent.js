import React, { useState, useEffect, useRef } from "react";
import { MapContainer, TileLayer, GeoJSON } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "./MapComponent.css";
import Legend from "./Legend";
import LoadingOverlay from "./LoadingOverlay";
import Notification from "./Notification";
import {
  predictSingleWard,
  getCurrentWeather,
  getCurrentSeason,
  checkServerHealth,
} from "./api";
// Mumbai wards data will be fetched from public folder

const MapComponent = ({
  predictions,
  setPredictions,
  selectedWard,
  setSelectedWard,
  loading,
  setLoading,
  onWardCoordinatesChange,
}) => {
  const [wardsData, setWardsData] = useState(null);
  const [notification, setNotification] = useState("");
  const geoJsonRef = useRef(null);

  // Load GeoJSON data
  useEffect(() => {
    fetch("/mumbai-wards-cleaned.geojson")
      .then((response) => response.json())
      .then((data) => setWardsData(data))
      .catch((error) => console.error("Error loading wards data:", error));
  }, []);

  // Check server health on component mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const health = await checkServerHealth();
        console.log("Server health check passed:", health);
      } catch (error) {
        console.warn("Server health check failed:", error.message);
        // You could show a notification here if needed
      }
    };

    checkHealth();
  }, []);

  // Get color based on risk level
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
        return "#4285f4";
    }
  };

  // Style function for GeoJSON features
  const getFeatureStyle = (feature) => {
    const wardName =
      feature.properties.ward_name_full ||
      feature.properties.name ||
      feature.properties.NAME ||
      "Unknown";
    const wardCode =
      feature.properties.ward_code ||
      feature.properties.ward ||
      feature.properties.WARD ||
      null;
    const prediction =
      (wardCode && predictions[wardCode]) || predictions[wardName];
    const isSelected = selectedWard === wardName;

    return {
      fillColor: prediction ? getRiskColor(prediction.risk_level) : "#f8f9fa", // Light gray for unpredicted wards
      weight: isSelected ? 4 : 2, // Thicker border for selected ward
      opacity: 1,
      color: isSelected ? "#007bff" : "#495057", // Blue border for selected, dark gray for others
      dashArray: prediction ? "" : "5, 5", // Dashed border for unpredicted wards
      fillOpacity: prediction ? 0.8 : 0.3, // Higher opacity for predicted wards
    };
  };

  // Calculate centroid of a polygon
  const calculateCentroid = (coordinates) => {
    if (!coordinates || !coordinates[0]) return null;

    const polygon = coordinates[0]; // Get the outer ring
    let x = 0,
      y = 0;

    for (let i = 0; i < polygon.length; i++) {
      x += polygon[i][0];
      y += polygon[i][1];
    }

    return {
      longitude: x / polygon.length,
      latitude: y / polygon.length,
    };
  };

  // Handle ward click
  const onEachFeature = (feature, layer) => {
    const wardName = feature.properties.ward_name_full || "Unknown Ward";
    const wardCode = feature.properties.ward_code || "N/A";
    const prediction = predictions[wardName];

    // Add tooltip with ward information
    const tooltipContent = `
      <div class="tooltip-container">
        <div class="tooltip-title">${wardName}</div>
        <div class="tooltip-ward-code">Ward Code: ${wardCode}</div>
        ${
          prediction
            ? (() => {
                const riskLevelText =
                  (prediction.risk_level || "low")
                    .toString()
                    .charAt(0)
                    .toUpperCase() +
                  (prediction.risk_level || "low").toString().slice(1);
                const prob = Number(prediction.flood_risk_probability);
                const probText = isFinite(prob)
                  ? (prob * 100).toFixed(1) + "%"
                  : "N/A";
                return `<div class="tooltip-prediction-details">
                  <div class="tooltip-risk-level risk-level-${(
                    prediction.risk_level || "low"
                  )
                    .toString()
                    .toLowerCase()}">
                    ${riskLevelText} Risk (${probText})
                  </div>
                </div>`;
              })()
            : `<div class="tooltip-no-prediction">Click for ML prediction</div>`
        }
      </div>
    `;

    layer.bindTooltip(tooltipContent, {
      permanent: false,
      direction: "top",
      offset: [0, -10],
      className: "custom-tooltip",
    });

    layer.on({
      click: async (e) => {
        const properties = feature.properties;
        const wardName = properties.ward_name_full || "Unknown Ward";
        const wardCode =
          properties.ward_code || properties.ward || properties.WARD || "N/A";
        const coordinates = calculateCentroid(feature.geometry.coordinates);

        console.log("Ward clicked:", wardName);
        console.log("Ward code:", wardCode);
        console.log("Coordinates:", coordinates);
        console.log("All properties:", properties);

        setSelectedWard(wardName);
        setLoading(true);

        // Pass coordinates to parent component for weather data
        if (onWardCoordinatesChange) {
          onWardCoordinatesChange({
            latitude: coordinates.latitude,
            longitude: coordinates.longitude,
            wardName: wardName,
            wardCode: wardCode,
          });
        }

        try {
          // Get ward-specific weather data first, then make prediction
          const weatherData = await getCurrentWeather(wardCode);
          console.log("Real weather data for ward:", weatherData);

          // Extract rainfall from weather data or throw error if not available
          const currentRainfall = weatherData.Rainfall_mm;
          if (currentRainfall === undefined || currentRainfall === null) {
            throw new Error("Rainfall data not available from weather service");
          }

          // Get current season and tide data from weather service
          const currentSeason = getCurrentSeason();
          const tideHeight = weatherData.Tide_Level_m || 2.0; // Use weather service tide data

          // Call the ward-specific API endpoint with ward code
          const prediction = await predictSingleWard(
            wardCode,
            currentRainfall,
            tideHeight,
            currentSeason
          );
          console.log("Ward-specific ML prediction:", prediction);
          console.log("Bayesian probability:", prediction.bayesian_probability);
          console.log("Random forest data:", prediction.random_forest);
          console.log("Combined assessment:", prediction.combined_assessment);

          // Transform the backend response to match frontend expectations
          const enhancedPrediction = {
            flood_risk_probability: prediction.bayesian_probability,
            risk_level:
              prediction.random_forest?.flood_risk_level === 2
                ? "high"
                : prediction.random_forest?.flood_risk_level === 1
                ? "medium"
                : "low",
            confidence: prediction.combined_assessment?.confidence || "Medium",
            will_flood: prediction.random_forest?.will_flood || false,
            ward_risk_zone: prediction.ward_risk_zone,
            weather_data: prediction.weather_data,
            timestamp: prediction.timestamp,
            // Add the structure that WardDetails expects
            overall_assessment: {
              combined_risk_level: prediction.combined_assessment?.high_risk
                ? "High"
                : "Low",
              combined_flood_probability: prediction.bayesian_probability || 0,
              models_used: "Random Forest + Bayesian Network",
              recommendation: prediction.combined_assessment?.high_risk
                ? "High flood risk detected. Avoid travel to this area if possible."
                : "Low flood risk. Normal conditions expected.",
            },
            random_forest: {
              flood_probability: prediction.random_forest?.risk_probabilities
                ? Math.max(
                    ...Object.values(
                      prediction.random_forest.risk_probabilities
                    )
                  )
                : 0,
              risk_level:
                prediction.random_forest?.flood_risk_level === 2
                  ? "High"
                  : prediction.random_forest?.flood_risk_level === 1
                  ? "Medium"
                  : "Low",
              confidence:
                prediction.combined_assessment?.confidence || "Medium",
            },
            bayesian_network: {
              flood_probability: prediction.bayesian_probability || 0,
              risk_level:
                (prediction.bayesian_probability || 0) > 0.7
                  ? "High"
                  : (prediction.bayesian_probability || 0) > 0.3
                  ? "Medium"
                  : "Low",
            },
            cluster_analysis: {
              cluster: prediction.ward_risk_zone,
              cluster_risk_level: prediction.ward_risk_zone,
            },
            wardDetails: {
              name: wardName,
              code: wardCode,
              coordinates: coordinates,
              area: properties.Shape__Area
                ? parseFloat(properties.Shape__Area).toFixed(2)
                : "N/A",
              perimeter: properties.Shape__Length
                ? parseFloat(properties.Shape__Length).toFixed(2)
                : "N/A",
              objectId: properties.OBJECTID || "N/A",
            },
          };

          console.log("Enhanced prediction being stored:", enhancedPrediction);

          // Store the enhanced prediction
          setPredictions((prev) => ({
            ...prev,
            [wardName]: enhancedPrediction,
          }));
        } catch (error) {
          console.error("Error getting prediction:", error);
          setNotification(
            `Error connecting to backend: ${error.message}. Make sure the server is running.`
          );
        } finally {
          setLoading(false);
        }
      },
      mouseover: (e) => {
        const layer = e.target;
        const wardName = feature.properties.ward_name_full || "Unknown Ward";
        const wardCode =
          feature.properties.ward_code ||
          feature.properties.ward ||
          feature.properties.WARD ||
          null;
        const prediction =
          (wardCode && predictions[wardCode]) || predictions[wardName];

        layer.setStyle({
          weight: 4,
          color: "#007bff",
          dashArray: "",
          fillOpacity: prediction ? 0.9 : 0.6,
          fillColor: prediction
            ? getRiskColor(prediction.risk_level)
            : "#e3f2fd",
        });

        if (!L.Browser.ie && !L.Browser.opera && !L.Browser.edge) {
          layer.bringToFront();
        }
      },
      mouseout: (e) => {
        if (geoJsonRef.current) {
          geoJsonRef.current.resetStyle(e.target);
        }
      },
    });
  };

  return (
    <div className="map-component-container">
      <MapContainer
        center={[19.076, 72.8777]} // Mumbai coordinates
        zoom={11}
        style={{ height: "100%", width: "100%" }} // It's common to keep this for Leaflet
      >
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
          subdomains="abcd"
          maxZoom={19}
        />

        {wardsData && (
          <GeoJSON
            ref={geoJsonRef}
            data={wardsData}
            style={getFeatureStyle}
            onEachFeature={onEachFeature}
            key={JSON.stringify(predictions)} // Force re-render when predictions change
          />
        )}
      </MapContainer>

      <Legend />

      <LoadingOverlay loading={loading} />
      <Notification
        message={notification}
        onClose={() => setNotification("")}
      />
    </div>
  );
};

export default MapComponent;
