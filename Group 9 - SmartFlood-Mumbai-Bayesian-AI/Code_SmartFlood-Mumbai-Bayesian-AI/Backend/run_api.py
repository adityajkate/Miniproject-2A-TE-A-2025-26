"""
API Server Runner for Mumbai Flood Prediction System
"""

import uvicorn
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

if __name__ == "__main__":
    print("Starting Mumbai Flood Prediction API...")
    print("API will be available at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")

    # Tip: For routing to avoid self-call deadlocks, prefer >=2 workers (no reload)
    workers_str = os.environ.get("API_WORKERS", "2").strip()
    try:
        workers = int(workers_str)
    except Exception:
        workers = 2

    # Allow toggling auto-reload via env; default is OFF to avoid WatchFiles issues on Windows
    reload_env = os.environ.get("API_RELOAD", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    # Uvicorn does not support reload with multiple workers
    use_reload = bool(reload_env and workers == 1)

    if use_reload:
        print("Uvicorn reload: ENABLED (development mode, workers forced to 1)")
    else:
        print(
            "Uvicorn reload: DISABLED (set API_RELOAD=1 to enable in dev with API_WORKERS=1)"
        )

    uvicorn.run(
        "api.flood_prediction_api:app",
        host="0.0.0.0",
        port=8000,
        reload=use_reload,
        log_level="info",
        workers=(1 if use_reload else workers),
    )
