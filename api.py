"""
AI Engine API

This module provides a FastAPI-based REST API for workload analysis and migration recommendations.

Endpoints:
- GET /: Health check
- POST /start: Start the AI Engine
- POST /stop: Stop the AI Engine
- POST /analyze: Analyze workloads and generate recommendations
"""

import asyncio
import threading
import json
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List

import aiohttp
import schedule
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from engine.main import (
    process_monitoring_data,
    save_and_log_explanations,
    analyze_workloads,
)


from engine.data_types import *  # pylint: disable=wildcard-import, unused-wildcard-import

from engine.util import (
    get_logger,
    load_config,
    format_message,
)
from engine.util import build_workload_recommendations

logger = get_logger("api")


class AppState:
    """
    Application state to hold configuration and runtime variables.
    """

    def __init__(self):
        self.models = {}
        self.config = None
        self.running: bool = False
        self.stop_event: threading.Event = threading.Event()
        self.scheduler_thread: Optional[threading.Thread] = None
        self.current_batch_id: int = 0
        self.scheduler_interval: int = 300


app_state = AppState()


# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Loads configuration and models at startup.
    """
    # Load configuration and models at startup
    app_state.config = load_config()

    # Initialize scheduler interval
    default_interval = 60 * 5
    interval_config = app_state.config.get("ai", {}).get("scheduler_interval")
    app_state.scheduler_interval = (
        int(interval_config) if interval_config else default_interval
    )

    yield

    # Clean up
    if app_state.scheduler_thread and app_state.scheduler_thread.is_alive():
        app_state.running = False
        app_state.stop_event.set()
        app_state.scheduler_thread.join(timeout=5)


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


async def apply_recommendations(
    recommendations: List[WorkloadRecommendation] | List[Dict] | Any,
):
    """
    Sends a POST request to the Actuator service endpoint with the recommendations

    Args:
        recommendations: List of WorkloadRecommendation objects or dicts.
    Returns:
        None
    """
    try:
        # Apply recommendations
        payload = []
        if isinstance(recommendations, list):
            payload = [
                (
                    r.model_dump()
                    if hasattr(r, "model_dump")
                    else (r.dict() if hasattr(r, "dict") else r)
                )
                for r in recommendations
            ]
        elif hasattr(recommendations, "to_dict"):
            payload = recommendations.to_dict(orient="records")

        recommendations_json = json.dumps(payload, ensure_ascii=False)

        actuator_config = app_state.config.get("actuator", {})
        host = actuator_config.get("host", "localhost")
        port = actuator_config.get("port", 8080)
        route = actuator_config.get("route", "actuate")

        url = f"http://{host}:{port}/{route}"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
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
                    logger.error("Failed to apply recommendations: %s", response.status)

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error applying recommendations: %s", e)


@app.post("/start")
async def start():
    """Start the AI Engine"""
    # Mark the engine as running and clear any previous stop signal
    app_state.running = True
    if app_state.stop_event.is_set():
        app_state.stop_event.clear()

    # Function to fetch metrics from monitor
    async def fetch_metrics():
        if not app_state.running:
            return

        try:
            monitor_config = app_state.config.get("monitor", {})
            host = monitor_config.get("host", "localhost")
            port = monitor_config.get("port", 8080)
            route = monitor_config.get("route", "metrics")
            interval = monitor_config.get("interval", "5m")

            url = f"http://{host}:{port}/{route}"
            json_payload = {"interval": interval}

            logger.debug("Fetching metrics from MONITOR: %s", url)
            async with aiohttp.ClientSession() as session:
                async with session.get(url, json=json_payload) as response:
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
                            "Failed to fetch metrics from MONITOR: %s", response.status
                        )
                        return
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Error fetching metrics from MONITOR: %s", e)
            return

        processed_data = process_monitoring_data(data)

        if not processed_data:
            return

        workloads = processed_data.get("workloads", [])
        cluster_info = processed_data.get("cluster_info", [])
        interval_duration = processed_data.get("interval_duration", "unknown")

        if workloads:

            # Start new history batch if feature enabled
            history_config = app_state.config.get('recommendation_history', {})
            if history_config.get('enabled', False):
                from engine.history_batch_manager import get_history_batch_manager
                from engine.recommendation_history_db import save_history_batch_recommendations
                
                history_batch_mgr = get_history_batch_manager(app_state.config)
                current_history_batch_id = history_batch_mgr.start_new_batch()
                logger.info(f"Started history batch {current_history_batch_id}")
            
            result_df, explanations = analyze_workloads(
                workloads,
                app_state.config,
                cluster_info=cluster_info,
                interval_duration=interval_duration,
            )

            save_and_log_explanations(result_df, explanations, workloads)

            # Transform into WorkloadRecommendation-shaped list[dict]
            app_state.current_batch_id += 1
            recommendations = build_workload_recommendations(
                result_df, explanations, workloads, app_state.current_batch_id
            )
            
            # Save to history database if feature enabled
            if history_config.get('enabled', False):
                try:
                    save_history_batch_recommendations(
                        history_config.get('mongodb'),
                        current_history_batch_id,
                        recommendations
                    )
                    logger.info(f"Saved recommendations to history batch {current_history_batch_id}")
                except Exception as e:
                    logger.error(f"Failed to save history batch recommendations: {e}")

            await apply_recommendations(recommendations)

    # run the scheduler in a separate thread
    def run_scheduler():
        while not app_state.stop_event.is_set():
            schedule.run_pending()

    # Run fetch_metrics immediately if configured
    if app_state.config["ai"].get("fetch_metrics_immediately", False):
        await fetch_metrics()

    # Start the scheduler to run every SCHEDULER_INTERVAL seconds after the first execution
    schedule.every(app_state.scheduler_interval).seconds.do(
        lambda: asyncio.run(fetch_metrics())
    )
    app_state.scheduler_thread = threading.Thread(
        target=run_scheduler, name="scheduler-thread", daemon=True
    )
    app_state.scheduler_thread.start()

    logger.info(format_message("Recommendations are running", icon="🚀", color="GREEN"))
    return {"status": "Recommendations are running"}


@app.post("/stop")
async def stop():
    """Stop the AI Engine"""
    app_state.running = False
    app_state.stop_event.set()

    # Wait for the background thread to finish in a non-blocking way
    if app_state.scheduler_thread is not None and app_state.scheduler_thread.is_alive():
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, app_state.scheduler_thread.join)

    logger.info(format_message("Recommendations stopped", icon="🛑", color="RED"))
    return {"status": "Recommendations stopped"}


class AnalyzeRequest(BaseModel):
    """Schema for /analyze endpoint"""

    input_json: Optional[str] = Field(
        default=None, description="Path to JSON file containing workloads"
    )
    workloads: Optional[list] = Field(
        default=None, description="List of workload dictionaries"
    )


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
                with open(request.input_json, "r", encoding="utf-8") as f:
                    workloads = json.load(f)
            except Exception as e:
                logger.error(f"Failed to read input_json file: {e}")
                return {
                    "status": "error",
                    "message": f"Failed to read input_json file: {e}",
                }
        else:
            workloads = request.workloads

        if not workloads:
            return {"status": "error", "message": "No workload data provided"}

        # Start new history batch if feature enabled
        history_config = app_state.config.get('recommendation_history', {})
        if history_config.get('enabled', False):
            from engine.history_batch_manager import get_history_batch_manager
            from engine.recommendation_history_db import save_history_batch_recommendations
            
            history_batch_mgr = get_history_batch_manager(app_state.config)
            current_history_batch_id = history_batch_mgr.start_new_batch()
            logger.info(f"Started history batch {current_history_batch_id}")

        result_df, explanations = analyze_workloads(workloads, app_state.config)
        save_and_log_explanations(result_df, explanations, workloads)
        
        # Save to history database if feature enabled
        if history_config.get('enabled', False):
            try:
                recommendations = _build_workload_recommendations(
                    result_df, explanations, workloads
                )
                save_history_batch_recommendations(
                    history_config.get('mongodb'),
                    current_history_batch_id,
                    recommendations
                )
                logger.info(f"Saved recommendations to history batch {current_history_batch_id}")
            except Exception as e:
                logger.error(f"Failed to save history batch recommendations: {e}")

        return {
            "status": "success",
            "message": "Workloads analyzed successfully",
            "recommendations": result_df.to_dict(orient="records"),
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
