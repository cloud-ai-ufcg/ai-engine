"""
Fake monitoring API for testing purposes.
This API serves as a mock endpoint that returns metrics from monitor_test.json.
It's used for local development and testing of the monitoring system.
This endpoint is particularly useful for testing the monitoring pipeline without requiring a real cluster.
The API is designed to be a drop-in replacement for the real monitoring endpoint during development.
"""

from fastapi import FastAPI
import uvicorn
from typing import Dict, Any
import json

app = FastAPI(
    title="Metrics API",
    description="Fake implementation for testing (local development) - Returns data from monitor_test.json for monitoring pipeline testing",
    version="1.0.1",
)


@app.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """
    Returns fake monitoring metrics from monitor_test.json.
    This endpoint is used by the monitoring system to fetch test metrics.
    It provides a consistent interface for testing the monitoring pipeline
    and simulates the behavior of the real monitoring endpoint.

    Returns:
        Dict[str, Any]: Monitoring metrics or error information

    Raises:
        FileNotFoundError: If monitor_test.json file is not found
        json.JSONDecodeError: If the JSON file is invalid
        Exception: For any other unexpected errors

    """
    try:
        with open("monitor_test.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": "monitor_test.json not found"}
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON in monitor_test.json: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


if __name__ == "__main__":
    uvicorn.run("fake_monitor:app", host="0.0.0.0", port=8082, reload=True)
