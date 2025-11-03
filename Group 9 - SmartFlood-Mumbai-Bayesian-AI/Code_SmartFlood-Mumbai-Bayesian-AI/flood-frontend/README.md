# Mumbai Flood Risk Prediction Frontend

This React application provides an interactive map interface for predicting flood risks in Mumbai wards.

## Features

- Interactive map of Mumbai with ward boundaries
- Click on any ward to get flood risk prediction
- Real-time API communication with the backend
- Color-coded risk levels (Low: Green, Medium: Yellow, High: Orange, Critical: Red)
- Information panel showing prediction details

## Prerequisites

Make sure your backend Flask server is running on `http://127.0.0.1:8000`

## Installation

```bash
npm install
```

## Running the Application

```bash
npm start
```

The application will open in your browser at `http://localhost:3000`

## How to Use

1. The map will load showing Mumbai with all wards in blue
2. Click on any ward to get a flood risk prediction
3. The ward will change color based on the risk level
4. View detailed information in the info panel on the right

## Libraries Used

- **React**: Frontend framework
- **Leaflet**: Interactive mapping library
- **React-Leaflet**: React components for Leaflet
- **Axios**: HTTP client for API requests