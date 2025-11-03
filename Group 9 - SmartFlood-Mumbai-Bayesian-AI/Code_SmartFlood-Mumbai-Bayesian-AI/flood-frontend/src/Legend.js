import React from 'react';
import './Legend.css';

const Legend = () => {
  return (
    <div className="map-legend">
      <h4>🌊 Flood Risk Levels</h4>
      <div className="legend-item">
        <div className="legend-color unpredicted"></div>
        <span>Not Analyzed</span>
      </div>
      <div className="legend-item">
        <div className="legend-color low"></div>
        <span>Low Risk</span>
      </div>
      <div className="legend-item">
        <div className="legend-color medium"></div>
        <span>Medium Risk</span>
      </div>
      <div className="legend-item">
        <div className="legend-color high"></div>
        <span>High Risk</span>
      </div>
      <div className="legend-item">
        <div className="legend-color critical"></div>
        <span>Critical Risk</span>
      </div>
      <div className="legend-footer">
        🤖 Click wards for ML predictions
      </div>
    </div>
  );
};

export default Legend;
