from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    Form,
    Depends,
    BackgroundTasks,
)
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Tuple, Union, Optional
import pandas as pd
import os
import json
import joblib
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
    app_state.models = load_models()
    logger.info(f"Loaded {len(app_state.models)} models")
    yield
    # Clean up resources at shutdown
    app_state.models = {}


# Initialize FastAPI app
api = FastAPI(
    title="AI Engine API",
    description="API for performing workload analysis and generating migration recommendations",
    version="1.0.0",
    lifespan=lifespan,
)


@api.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "message": "AI Engine API is running"}


@api.get("/models")
async def get_models():
    """List all available models"""
    return {"models": list(app_state.models.keys())}


@api.post("/predict/models")
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


@api.post("/predict/ml")
async def predict_ml(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    output_csv: str = Form(os.path.join(OUTPUT_DIR, "recommendations.csv")),
):
    """Predict migration recommendations using machine learning model"""
    # Create a temporary file to store the uploaded JSON
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    try:
        # Save uploaded file to temporary location
        with temp_file:
            shutil.copyfileobj(file.file, temp_file)

        # Select model if specified
        model = None
        if model_name and model_name in app_state.models:
            model = app_state.models[model_name]

        # Run prediction
        predict_with_machine_learning(temp_file.name, model, output_csv)

        # Try to read the results
        result_df = pd.read_csv(output_csv)
        recommendations = result_df.to_dict(orient="records")

        return {
            "status": "completed",
            "recommendations": recommendations,
            "output_csv": output_csv,
        }
    except Exception as e:
        logger.error(f"Error processing JSON file: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temp file
        os.unlink(temp_file.name)


@api.post("/analyze")
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


@api.get("/results/{filename}")
async def get_results(filename: str):
    """Get result file from actuator directory"""
    file_path = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File {filename} not found")

    return FileResponse(file_path)


@api.post("/monitoring-data")
async def process_data(
    file: UploadFile = File(...), timestamp_lookback_seconds: int = Form(30)
):
    """Process monitoring data from uploaded JSON file"""
    # Create a temporary file to store the uploaded JSON
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    try:
        # Save uploaded file to temporary location
        with temp_file:
            shutil.copyfileobj(file.file, temp_file)

        # Load and process data
        with open(temp_file.name, "r") as f:
            data = json.load(f)

        workloads = process_monitoring_data(data, timestamp_lookback_seconds)

        return {
            "status": "success",
            "workload_count": len(workloads),
            "workloads": workloads,
        }
    except Exception as e:
        logger.error(f"Error processing monitoring data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temp file
        os.unlink(temp_file.name)


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8082, reload=True)
