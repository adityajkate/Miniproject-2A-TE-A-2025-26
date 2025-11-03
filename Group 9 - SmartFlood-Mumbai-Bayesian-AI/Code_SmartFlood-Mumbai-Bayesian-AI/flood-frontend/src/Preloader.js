import React from "react";
import "./Preloader.css";

/*
  Preloader (Modern / Minimal)
  - Full-screen glassmorphic overlay
  - Animated gradient ring with a water droplet icon
  - Subtle status text with animated dots
*/
const Preloader = ({ show, statusText = "Loading" }) => {
  if (!show) return null;

  return (
    <div className="preloader-overlay" role="status" aria-live="polite">
      <div className="preloader-card">
        <div className="ring-wrap" aria-hidden="true">
          <div className="ring" />
          <div className="ring-glow" />
          <svg
            className="droplet"
            width="38"
            height="38"
            viewBox="0 0 24 24"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            {/* Water droplet */}
            <defs>
              <linearGradient id="dropGrad" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stopColor="#60a5fa" />
                <stop offset="100%" stopColor="#06b6d4" />
              </linearGradient>
            </defs>
            <path
              d="M12 2c3.5 4.6 6 8.3 6 11.1A6 6 0 1 1 6 13.1C6 10.3 8.5 6.6 12 2Z"
              fill="url(#dropGrad)"
            />
            <circle cx="9.5" cy="10.5" r="1.2" fill="#e0f2fe" opacity="0.9" />
          </svg>
        </div>

        <h2 className="preloader-title">Mumbai Flood Prediction</h2>
        <p className="preloader-sub">AI-driven Monsoon Insights</p>

        <p className="preloader-status">
          <span>{statusText}</span>
          <span className="dots">
            <i></i>
            <i></i>
            <i></i>
          </span>
        </p>
      </div>
    </div>
  );
};

export default Preloader;
