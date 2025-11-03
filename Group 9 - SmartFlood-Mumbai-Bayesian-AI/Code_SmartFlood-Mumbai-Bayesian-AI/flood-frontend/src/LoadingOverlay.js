import React from 'react';
import './LoadingOverlay.css';

const LoadingOverlay = ({ loading, message }) => {
  if (!loading) return null;

  return (
    <div className="map-loading-overlay">
      <div className="loading-content">
        <div className="map-loading-spinner"></div>
        {message && <div className="loading-message">{message}</div>}
      </div>
    </div>
  );
};

export default LoadingOverlay;
