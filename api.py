from fastapi import FastAPI
from typing import Optional, List, Union, Dict, Any, Tuple
from contextlib import asynccontextmanager
import uvicorn
import json
from pydantic import BaseModel, Field

from engine.main import (
    process_monitoring_data,
    shard_and_analyze_workloads,
    save_and_log_explanations,
)

from engine.data_types import *

from engine.util import (
    get_logger,
    load_config,
    format_message,
)

import asyncio
import aiohttp
import json
import schedule
import threading


# Global state variables to control the recommendation loop
config = load_config()

SCHEDULER_INTERVAL_DEFAULT = 60 * 5 # 5 minutes
SCHEDULER_INTERVAL: int = int(config["ai"]["scheduler_interval"]) or int(SCHEDULER_INTERVAL_DEFAULT)  # seconds between recommendation cycles

running: bool = False
stop_event: threading.Event = threading.Event()
# Background thread that runs the scheduler; populated when `/start` is called
scheduler_thread: Optional[threading.Thread] = None

logger = get_logger("api")

class AppState:
    def __init__(self):
        self.models = {}
        self.config = None


app_state = AppState()


# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load configuration and models at startup
    app_state.config = load_config()
    yield


app = FastAPI(
    title="AI Engine API",
    description="API for performing workload analysis and generating migration recommendations",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "message": "AI Engine API is running"}


async def apply_recommendations(result_df):
    """
    Sends a POST request to the Actuator service endpoint with the recommendations

    Args:
        result_df: DataFrame containing the recommendations
    Returns:
        None
    """
    try:
        # Apply recommendations
        recommendations_json = json.dumps(
            result_df.to_dict(orient="records"), ensure_ascii=False
        )
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{app_state.config['actuator']['host']}:{app_state.config['actuator']['port']}/{app_state.config['actuator']['route']}",
                data=recommendations_json,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status == 200:
                    logger.info(
                        format_message(
                            "Successfully applied recommendations",
                            icon="✅",
                            color="GREEN",
                        )
                    )
                else:
                    logger.error(f"Failed to apply recommendations: {response.status}")

    except Exception as e:
        logger.error(f"Error applying recommendations: {e}")
        # raise HTTPException(status_code=500, detail=str(e))


@app.post("/start")
async def start():
    """Start the AI Engine"""
    global running, stop_event

    # Mark the engine as running and clear any previous stop signal
    running = True
    if stop_event.is_set():
        stop_event.clear()

    # Function to fetch metrics from monitor
    async def fetch_metrics():
        if not running:
            return

        try:
            async with aiohttp.ClientSession() as session:
                url = f"http://{app_state.config['monitor']['host']}:{app_state.config['monitor']['port']}/{app_state.config['monitor']['route']}"
                json={"interval": app_state.config['monitor']['interval']}
                
                logger.debug(f"Fetching metrics from MONITOR: {url}")
                logger.debug(f"Interval: {app_state.config['monitor']['interval']}")

                async with session.get(url, json=json) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(
                            format_message(
                                "Successfully fetched metrics from MONITOR",
                                icon="📊",
                                color="GREEN",
                            )
                        )
                    else:
                        logger.error(
                            f"Failed to fetch metrics from MONITOR: {response.status}"
                        )
        except Exception as e:
            logger.error(f"Error fetching metrics from MONITOR: {e}")

        workloads = process_monitoring_data(data)
        if workloads:
            result_df, explanations = shard_and_analyze_workloads(workloads, app_state.config)
            save_and_log_explanations(result_df, explanations)

            await apply_recommendations(result_df)

    # Function to run the scheduler in a separate thread
    def run_scheduler():
        while not stop_event.is_set():
            schedule.run_pending()

    # Run fetch_metrics immediately if configured
    if app_state.config["ai"]["fetch_metrics_immediately"]:
        # We're already in an async context (FastAPI request handler), so just await
        await fetch_metrics()

    # Start the scheduler to run every SCHEDULER_INTERVAL seconds after the first execution
    global scheduler_thread
    schedule.every(SCHEDULER_INTERVAL).seconds.do(lambda: asyncio.run(fetch_metrics()))
    scheduler_thread = threading.Thread(
        target=run_scheduler, name="scheduler-thread", daemon=True
    )
    scheduler_thread.start()

    logger.info(format_message("Recommendations are running", icon="🚀", color="GREEN"))
    return {"status": "Recommendations are running"}

@app.post("/stop")
async def stop():
    """Stop the AI Engine"""
    global running, stop_event, scheduler_thread
    running = False
    stop_event.set()
    # Wait for the background thread to finish in a non-blocking way
    if scheduler_thread is not None and scheduler_thread.is_alive():
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, scheduler_thread.join)
    logger.info(format_message("Recommendations stopped", icon="🛑", color="RED"))
    return {"status": "Recommendations stopped"}

class AnalyzeRequest(BaseModel):
    """Schema for /analyze endpoint"""
    input_json: Optional[str] = Field(default=None, description="Path to JSON file containing workloads")
    workloads: Optional[list] = Field(default=None, description="List of workload dictionaries")

@app.post("/analyze")
async def analyze_workloads_direct(request: AnalyzeRequest):
    """
    Analyze workloads directly from POST request data
    
    Args:
        workloads: List of workload data in JSON format
    Returns:
        Dictionary with analysis results and status
    """
    try:
        # Determine source of workload data
        workloads: Optional[list] = None
        if request.input_json:
            try:
                with open(request.input_json, "r") as f:
                    workloads = json.load(f)
            except Exception as e:
                logger.error(f"Failed to read input_json file: {e}")
                return {"status": "error", "message": f"Failed to read input_json file: {e}"}
        else:
            workloads = request.workloads

        if not workloads:
            return {"status": "error", "message": "No workload data provided"}

        result_df, explanations = analyze_workloads(workloads, app_state.config)
        save_and_log_explanations(result_df, explanations)
        
        return {
            "status": "success",
            "message": "Workloads analyzed successfully",
            "recommendations": result_df.to_dict(orient="records")
        }
    except Exception as e:
        logger.error(f"Error analyzing workloads: {e}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    # When running this module directly (e.g. `python api.py` inside Docker), the
    # FastAPI lifespan event has **not** executed yet, so the configuration has
    # not been loaded.  Ensure we load it manually before starting uvicorn so
    # that the server settings are available.
    if app_state.config is None:
        app_state.config = load_config()

    uvicorn.run(
        "api:app",
        host=app_state.config["server"]["host"],
        port=app_state.config["server"]["port"],
        reload=app_state.config["server"].get("reload", False),
    )
