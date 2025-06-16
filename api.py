from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    Form,
    BackgroundTasks,
)
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Any, Optional
import pandas as pd
import os
import tempfile
import shutil
from contextlib import asynccontextmanager
import uvicorn

from engine.main import (
    load_models,
    predict_with_models,
    predict_with_machine_learning,
    process_monitoring_data,
    write_recommendations,
    load_monitoring_data,
    analyze_workloads,
    save_and_log_explanations,
)

from engine.data_types import *

from engine.util import (
    get_logger,
    load_config,
    format_message,
    COLORS,
    MODELS_DIR,
    OUTPUT_DIR,
    ENGINE_LOG_DIR,
)

import asyncio
import aiohttp
import schedule
import threading
import time

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
    # app_state.models = load_models()
    # logger.info(f"Loaded {len(app_state.models)} models")
    yield
    # Clean up resources at shutdown
    app_state.models = {}


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


@app.get("/models")
async def get_models():
    """List all available models"""
    return {"models": list(app_state.models.keys())}


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
        recommendations_dict = result_df.to_dict(orient="records")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{app_state.config['actuator']['host']}:{app_state.config['actuator']['port']}/{app_state.config['actuator']['route']}",
                json={"recommendations": recommendations_dict},
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

    # Global variable to track if the engine is running
    running = True
    stop_event = threading.Event()

    # Function to fetch metrics from monitor
    async def fetch_metrics():
        if not running:
            return

        try:
            async with aiohttp.ClientSession() as session:
                url = f"http://{app_state.config['monitor']['host']}:{app_state.config['monitor']['port']}/{app_state.config['monitor']['route']}"
                logger.info(f"Fetching metrics from MONITOR: {url}")
                async with session.get(url, json={}) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(
                            format_message(
                                "Successfully fetched metrics from MONITOR",
                                icon="📊",
                                color="GREEN",
                            )
                        )
                        # Process the metrics data

                    else:
                        logger.error(
                            f"Failed to fetch metrics from MONITOR: {response.status}"
                        )
        except Exception as e:
            logger.error(f"Error fetching metrics from MONITOR: {e}")

        workloads = process_monitoring_data(data)
        if workloads:
            result_df, explanations = analyze_workloads(workloads, app_state.config)
            save_and_log_explanations(result_df, explanations)
            # Ensure output directory exists
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            write_recommendations(result_df)
            await apply_recommendations(result_df)

    # Function to run the scheduler in a separate thread
    def run_scheduler():
        while not stop_event.is_set():
            schedule.run_pending()

    # Run fetch_metrics immediately
    asyncio.create_task(fetch_metrics())

    # Start the scheduler to run every 30 seconds after the first execution
    schedule.every(30).seconds.do(lambda: asyncio.run(fetch_metrics()))
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.start()

    return {"status": "Recommendations are running"}


@app.post("/stop")
async def stop():
    """Stop the AI Engine"""
    global running
    running = False
    stop_event.set()
    return {"status": "Recommendations stopped"}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8083, reload=False)
