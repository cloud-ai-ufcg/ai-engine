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

from main import (
    load_models,
    predict_with_models,
    predict_with_machine_learning,
    process_monitoring_data,
    write_recommendations,
    load_monitoring_data,
    analyze_workloads,
    save_and_log_explanations,
)


from data_types import *

from util import (
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


# Create app state to store models
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


# Initialize FastAPI app
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


@app.post("/predict/models")
async def predict_models(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    output_dir: str = Form(OUTPUT_DIR),
):
    """Run predictions using all available models"""
    # Create a temporary file to store the uploaded CSV
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    try:
        # Save uploaded file to temporary location
        with temp_file:
            shutil.copyfileobj(file.file, temp_file)

        # Run predictions (in background to avoid blocking)
        background_tasks.add_task(
            predict_with_models, temp_file.name, app_state.models, output_dir
        )

        return {
            "status": "processing",
            "message": f"Predictions are being processed using {len(app_state.models)} models",
            "output_dir": output_dir,
        }
    except Exception as e:
        logger.error(f"Error processing CSV file: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temp file
        os.unlink(temp_file.name)


@app.post("/analyze")
async def analyze(workload_data: WorkloadInput):
    """Analyze workloads and generate recommendations"""
    try:
        # Process the input workloads
        workloads = process_monitoring_data(workload_data.workloads)

        if not workloads:
            raise HTTPException(
                status_code=400, detail="No valid workloads found in input data"
            )

        # Analyze the workloads
        config = app_state.config
        result_df, explanations = analyze_workloads(workloads, config)

        # Save explanations
        save_and_log_explanations(result_df, explanations)

        # Write recommendations
        output_csv = write_recommendations(result_df)

        # Prepare the response
        recommendations = result_df.to_dict(orient="records")

        return AnalysisResponse(
            recommendations=recommendations,
            explanation=explanations.get("explanation", "No explanation provided"),
            csv_path=output_csv,
        )
    except Exception as e:
        logger.error(f"Error analyzing workloads: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
                async with session.get(
                    f"http://{app_state.config['monitor']['host']}:{app_state.config['monitor']['port']}/{app_state.config['monitor']['route']}",
                    json={},
                ) as response:
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
                        if "workloads" in data:
                            workloads = process_monitoring_data(data["workloads"])
                            if workloads:
                                result_df, explanations = analyze_workloads(
                                    workloads, app_state.config
                                )
                                save_and_log_explanations(result_df, explanations)
                                write_recommendations(result_df)
                    else:
                        logger.error(
                            f"Failed to fetch metrics from MONITOR: {response.status}"
                        )
        except Exception as e:
            logger.error(f"Error fetching metrics from MONITOR: {e}")

    # Function to run the scheduler in a separate thread
    def run_scheduler():
        while not stop_event.is_set():
            schedule.run_pending()

    # Start the scheduler
    schedule.every(30).seconds.do(lambda: asyncio.run(fetch_metrics()))
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.start()

    return {"status": "Recommendations are running"}


@app.post("/stop")
async def stop():
    """Stop the AI Engine"""
    stop_event.set()
    return {"status": "Recommendations stopped"}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8082, reload=True)
