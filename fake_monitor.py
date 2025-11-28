"""
Fake monitoring API for testing purposes.
This API serves as a mock endpoint that returns metrics from monitor_test.json.
It's used for local development and testing of the monitoring system.
This endpoint is particularly useful for testing the monitoring pipeline without requiring a real cluster.
The API is designed to be a drop-in replacement for the real monitoring endpoint during development.
"""

from typing import Dict, Any
import json
from fastapi import FastAPI
import uvicorn


METRICS_FILENAME = "monitor_test_menorzin.json"
PORT = 8082
RELOAD = True

app = FastAPI(
    title="Metrics API",
    description=f"Fake implementation for testing (local development) - Returns data from {METRICS_FILENAME} for monitoring pipeline testing",
    version="1.0.1",
)


@app.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """
    Returns fake monitoring metrics from {METRICS_FILENAME}.
    This endpoint is used by the monitoring system to fetch test metrics.
    It provides a consistent interface for testing the monitoring pipeline
    and simulates the behavior of the real monitoring endpoint.

    Returns:
        Dict[str, Any]: Monitoring metrics or error information

    Raises:
        FileNotFoundError: If {METRICS_FILENAME} file is not found
        json.JSONDecodeError: If the JSON file is invalid
        Exception: For any other unexpected errors

    """
    try:
        with open(METRICS_FILENAME, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": f"{METRICS_FILENAME} not found"}
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON in {METRICS_FILENAME}: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


if __name__ == "__main__":
    uvicorn.run("fake_monitor:app", host="0.0.0.0", port=PORT, reload=RELOAD)
