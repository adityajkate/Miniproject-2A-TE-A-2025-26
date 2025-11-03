import React from "react";
import "./StatsCards.css";

const StatsCards = ({ predictions }) => {
  // Calculate statistics from predictions
  const calculateStats = () => {
    console.log("StatsCards - predictions received:", predictions);
    console.log("StatsCards - predictions keys:", Object.keys(predictions));
    console.log("StatsCards - predictions values:", Object.values(predictions));

    const totalWards = Object.keys(predictions).length;
    let lowRisk = 0;
    let mediumRisk = 0;
    let highRisk = 0;
    let floodExpected = 0;
    let avgBayesianProb = 0;

    Object.values(predictions).forEach((prediction, index) => {
      console.log(`StatsCards - prediction ${index}:`, prediction);
      console.log(
        `StatsCards - prediction ${index} riskLevel:`,
        prediction.riskLevel
      );
      console.log(
        `StatsCards - prediction ${index} willFlood:`,
        prediction.willFlood
      );
      console.log(
        `StatsCards - prediction ${index} bayesianProbability:`,
        prediction.bayesianProbability
      );

      // Count by Random Forest risk level (0=Low, 1=Medium, 2=High)
      switch (prediction.riskLevel) {
        case 0:
          lowRisk++;
          break;
        case 1:
          mediumRisk++;
          break;
        case 2:
          highRisk++;
          break;
        default:
          console.log(
            `StatsCards - Unknown risk level: ${prediction.riskLevel}`
          );
          break;
      }

      // Count flood expectations
      if (prediction.willFlood) {
        floodExpected++;
      }

      // Sum Bayesian probabilities for average
      avgBayesianProb += prediction.bayesianProbability || 0;
    });

    // Calculate average Bayesian probability
    avgBayesianProb = totalWards > 0 ? avgBayesianProb / totalWards : 0;

    return {
      totalWards,
      lowRisk,
      mediumRisk,
      highRisk,
      floodExpected,
      avgBayesianProb: (avgBayesianProb * 100).toFixed(1), // Convert to percentage
    };
  };

  const stats = calculateStats();

  return (
    <div className="stats-cards">
      <div className="stats-card total">
        <div className="card-content">
          <div className="card-icon">
            <svg
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M21 16V8A2 2 0 0 0 19 6H5A2 2 0 0 0 3 8V16A2 2 0 0 0 5 18H19A2 2 0 0 0 21 16Z"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <polyline
                points="7,10 12,15 17,10"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </div>
          <div className="card-info">
            <h3>ML Analyzed</h3>
            <div className="card-number">{stats.totalWards}</div>
            <div className="card-subtitle">Real-time predictions</div>
          </div>
        </div>
      </div>

      <div className="stats-card low-risk">
        <div className="card-content">
          <div className="card-icon">
            <div className="risk-indicator low"></div>
          </div>
          <div className="card-info">
            <h3>Low Risk</h3>
            <div className="card-number">{stats.lowRisk}</div>
          </div>
        </div>
      </div>

      <div className="stats-card medium-risk">
        <div className="card-content">
          <div className="card-icon">
            <div className="risk-indicator medium"></div>
          </div>
          <div className="card-info">
            <h3>Medium Risk</h3>
            <div className="card-number">{stats.mediumRisk}</div>
          </div>
        </div>
      </div>

      <div className="stats-card high-risk">
        <div className="card-content">
          <div className="card-icon">
            <div className="risk-indicator high"></div>
          </div>
          <div className="card-info">
            <h3>High Risk</h3>
            <div className="card-number">{stats.highRisk}</div>
          </div>
        </div>
      </div>

      <div className="stats-card flood-expected">
        <div className="card-content">
          <div className="card-icon">
            <svg
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M12 2.69L13.5 7.5H18.5L14.5 10.5L16 15.31L12 12.31L8 15.31L9.5 10.5L5.5 7.5H10.5L12 2.69Z"
                fill="currentColor"
              />
            </svg>
          </div>
          <div className="card-info">
            <h3>Flood Expected</h3>
            <div className="card-number">{stats.floodExpected}</div>
            <div className="card-subtitle">Random Forest</div>
          </div>
        </div>
      </div>

      <div className="stats-card bayesian-avg">
        <div className="card-content">
          <div className="card-icon">
            <svg
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <circle
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="2"
              />
              <path
                d="M8 12L11 15L16 9"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </div>
          <div className="card-info">
            <h3>Avg Probability</h3>
            <div className="card-number">{stats.avgBayesianProb}%</div>
            <div className="card-subtitle">Bayesian Network</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StatsCards;
