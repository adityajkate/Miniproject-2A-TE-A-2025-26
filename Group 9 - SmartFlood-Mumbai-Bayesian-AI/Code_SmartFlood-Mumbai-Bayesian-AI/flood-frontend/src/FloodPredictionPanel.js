import React, { useState, useEffect } from 'react';
import {
    getWardFloodPrediction,
    getWardCurrentWeather,
    formatFloodPrediction,
    getFloodRiskColor,
    getRiskZoneColor,
    WARD_NAMES
} from './api';
import './FloodPredictionPanel.css';

const FloodPredictionPanel = ({ selectedWard, onPredictionUpdate }) => {
    const [prediction, setPrediction] = useState(null);
    const [weather, setWeather] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (selectedWard) {
            loadWardPrediction(selectedWard);
        }
    }, [selectedWard]);

    const loadWardPrediction = async (wardCode) => {
        try {
            setLoading(true);
            setError(null);

            // Load both prediction and weather data
            const [predictionData, weatherData] = await Promise.all([
                getWardFloodPrediction(wardCode),
                getWardCurrentWeather(wardCode)
            ]);

            const formattedPrediction = formatFloodPrediction(predictionData);
            setPrediction(formattedPrediction);
            setWeather(weatherData);

            // Notify parent component
            if (onPredictionUpdate) {
                onPredictionUpdate(wardCode, formattedPrediction);
            }

        } catch (error) {
            setError(error.message);
            console.error('Error loading ward prediction:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleRefresh = () => {
        if (selectedWard) {
            loadWardPrediction(selectedWard);
        }
    };

    if (!selectedWard) {
        return (
            <div className="flood-prediction-panel">
                <div className="no-selection">
                    <div className="no-selection-icon">🗺️</div>
                    <h3>Select a Ward</h3>
                    <p>Click on a ward in the map to view flood prediction details</p>
                </div>
            </div>
        );
    }

    if (loading) {
        return (
            <div className="flood-prediction-panel">
                <div className="loading-state">
                    <div className="loading-spinner"></div>
                    <p>Loading flood prediction for Ward {selectedWard}...</p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="flood-prediction-panel">
                <div className="error-state">
                    <div className="error-icon">⚠️</div>
                    <h3>Error Loading Prediction</h3>
                    <p>{error}</p>
                    <button onClick={handleRefresh} className="retry-button">
                        Try Again
                    </button>
                </div>
            </div>
        );
    }

    if (!prediction) {
        return (
            <div className="flood-prediction-panel">
                <div className="no-data">
                    <p>No prediction data available for Ward {selectedWard}</p>
                    <button onClick={handleRefresh} className="retry-button">
                        Load Data
                    </button>
                </div>
            </div>
        );
    }

    const getRiskLevelText = (level) => {
        switch (level) {
            case 2: return 'High Risk';
            case 1: return 'Medium Risk';
            case 0: return 'Low Risk';
            default: return 'Unknown';
        }
    };

    const getConfidenceText = (confidence) => {
        if (confidence >= 0.8) return 'Very High';
        if (confidence >= 0.6) return 'High';
        if (confidence >= 0.4) return 'Medium';
        return 'Low';
    };

    return (
        <div className="flood-prediction-panel">
            <div className="panel-header">
                <div className="ward-info">
                    <h2>Ward {selectedWard}</h2>
                    <p className="ward-name">{WARD_NAMES[selectedWard] || 'Mumbai Ward'}</p>
                </div>
                <button onClick={handleRefresh} className="refresh-button" title="Refresh Data">
                    🔄
                </button>
            </div>

            {/* Risk Zone Classification */}
            <div className="risk-zone-card">
                <div className="card-header">
                    <h3>Ward Risk Classification</h3>
                    <span className="model-tag">K-means Clustering</span>
                </div>
                <div
                    className="risk-zone-badge"
                    style={{ backgroundColor: getRiskZoneColor(prediction.riskZone) }}
                >
                    {prediction.riskZone}
                </div>
            </div>

            {/* Random Forest Prediction */}
            <div className="prediction-card">
                <div className="card-header">
                    <h3>Flood Risk Prediction</h3>
                    <span className="model-tag">Random Forest</span>
                </div>

                <div className="prediction-result">
                    <div
                        className="risk-level-badge"
                        style={{ backgroundColor: getFloodRiskColor(prediction.riskLevel) }}
                    >
                        {getRiskLevelText(prediction.riskLevel)}
                    </div>

                    <div className="flood-status">
                        <span className={`flood-indicator ${prediction.willFlood ? 'flood-yes' : 'flood-no'}`}>
                            {prediction.willFlood ? '🌊 Flood Expected' : '✅ No Flood Expected'}
                        </span>
                    </div>
                </div>

                <div className="risk-probabilities">
                    <h4>Risk Probabilities</h4>
                    <div className="probability-bars">
                        <div className="probability-item">
                            <span>Low Risk</span>
                            <div className="probability-bar">
                                <div
                                    className="probability-fill low"
                                    style={{ width: `${(prediction.riskProbabilities.low * 100)}%` }}
                                ></div>
                            </div>
                            <span>{(prediction.riskProbabilities.low * 100).toFixed(1)}%</span>
                        </div>
                        <div className="probability-item">
                            <span>Medium Risk</span>
                            <div className="probability-bar">
                                <div
                                    className="probability-fill medium"
                                    style={{ width: `${(prediction.riskProbabilities.medium * 100)}%` }}
                                ></div>
                            </div>
                            <span>{(prediction.riskProbabilities.medium * 100).toFixed(1)}%</span>
                        </div>
                        <div className="probability-item">
                            <span>High Risk</span>
                            <div className="probability-bar">
                                <div
                                    className="probability-fill high"
                                    style={{ width: `${(prediction.riskProbabilities.high * 100)}%` }}
                                ></div>
                            </div>
                            <span>{(prediction.riskProbabilities.high * 100).toFixed(1)}%</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Bayesian Network Prediction */}
            <div className="bayesian-card">
                <div className="card-header">
                    <h3>Probabilistic Analysis</h3>
                    <span className="model-tag">Bayesian Network</span>
                </div>

                <div className="bayesian-result">
                    <div className="probability-circle">
                        <svg width="80" height="80" viewBox="0 0 80 80">
                            <circle
                                cx="40"
                                cy="40"
                                r="35"
                                fill="none"
                                stroke="#e0e0e0"
                                strokeWidth="6"
                            />
                            <circle
                                cx="40"
                                cy="40"
                                r="35"
                                fill="none"
                                stroke={prediction.bayesianProbability > 0.5 ? '#ff6b6b' : '#51cf66'}
                                strokeWidth="6"
                                strokeDasharray={`${prediction.bayesianProbability * 220} 220`}
                                strokeDashoffset="55"
                                transform="rotate(-90 40 40)"
                            />
                            <text x="40" y="45" textAnchor="middle" className="probability-text">
                                {(prediction.bayesianProbability * 100).toFixed(1)}%
                            </text>
                        </svg>
                    </div>
                    <p>Flood Probability</p>
                </div>
            </div>

            {/* Combined Assessment */}
            <div className="assessment-card">
                <div className="card-header">
                    <h3>Combined Assessment</h3>
                    <span className="model-tag">All Models</span>
                </div>

                <div className="assessment-result">
                    <div className={`overall-risk ${prediction.highRisk ? 'high-risk' : 'low-risk'}`}>
                        {prediction.highRisk ? '⚠️ High Risk Detected' : '✅ Low Risk'}
                    </div>

                    <div className="confidence-meter">
                        <span>Confidence: {getConfidenceText(prediction.confidence)}</span>
                        <div className="confidence-bar">
                            <div
                                className="confidence-fill"
                                style={{ width: `${prediction.confidence * 100}%` }}
                            ></div>
                        </div>
                        <span>{(prediction.confidence * 100).toFixed(1)}%</span>
                    </div>
                </div>
            </div>

            {/* Weather Data */}
            {weather && (
                <div className="weather-card">
                    <div className="card-header">
                        <h3>Current Weather</h3>
                        <span className="model-tag">OpenWeatherMap</span>
                    </div>

                    <div className="weather-grid">
                        <div className="weather-item">
                            <span className="weather-label">Rainfall</span>
                            <span className="weather-value">{weather.Rainfall_mm} mm</span>
                        </div>
                        <div className="weather-item">
                            <span className="weather-label">24hr Rainfall</span>
                            <span className="weather-value">{weather.Rainfall_24hr} mm</span>
                        </div>
                        <div className="weather-item">
                            <span className="weather-label">Temperature</span>
                            <span className="weather-value">{weather.Temperature_C}°C</span>
                        </div>
                        <div className="weather-item">
                            <span className="weather-label">Humidity</span>
                            <span className="weather-value">{weather['Humidity_%']}%</span>
                        </div>
                        <div className="weather-item">
                            <span className="weather-label">Wind Speed</span>
                            <span className="weather-value">{weather.Wind_Speed_kmh} km/h</span>
                        </div>
                        <div className="weather-item">
                            <span className="weather-label">Tide Level</span>
                            <span className="weather-value">{weather.Tide_Level_m} m</span>
                        </div>
                    </div>
                </div>
            )}

            <div className="timestamp">
                Last updated: {new Date(prediction.timestamp).toLocaleString()}
            </div>
        </div>
    );
};

export default FloodPredictionPanel;