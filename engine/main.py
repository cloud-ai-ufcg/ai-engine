import os
import joblib
import json
import pandas as pd
from typing import Dict, List, Any, Tuple, Union
import datetime

from .util import (
    get_logger,
    load_config,
    format_message,
    COLORS,
    MODELS_DIR,
    OUTPUT_DIR,
    ENGINE_LOG_DIR,
)
from .agents import label_workloads_with_gemini, label_workloads_with_crewai

logger = get_logger("main")


def load_models(models_dir=MODELS_DIR):
    """
    Loads all sklearn models from the given directory.
    Returns a dict of {model_filename: model_object}
    """
    models = {}
    for fname in os.listdir(models_dir):
        fpath = os.path.join(models_dir, fname)
        if os.path.isfile(fpath):
            try:
                model = joblib.load(fpath)
                models[fname] = model
            except Exception as e:
                logger.warning(f"Skipping {fname}: {e}")
    return models


def predict_with_models(input_csv, models=None, output_dir=OUTPUT_DIR):
    """
    Loads input data from CSV, runs predictions for each model, and writes CSV outputs.
    Each output is named <model_filename>_predictions.csv in the actuator directory.
    """
    if models is None:
        models = load_models()
    df = pd.read_csv(input_csv)
    for model_name, model in models.items():
        try:
            preds = model.predict(df)
            out_df = df.copy()
            out_df["prediction"] = preds
            out_name = f"{model_name}_predictions.csv"
            out_path = os.path.join(output_dir, out_name)
            out_df.to_csv(out_path, index=False)
            logger.info(f"Predictions written to {out_path}")
        except Exception as e:
            logger.error(f"Prediction failed for {model_name}: {e}")


def predict_with_machine_learning(
    json_path, model=None, output_csv=os.path.join(OUTPUT_DIR, "recommendations.csv")
):
    """
    Reads workload info from JSON, prepares features, uses model to predict migration,
    and writes (workload_id, kind, label) to output_csv.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    df = pd.json_normalize(data)
    # Feature engineering: select and convert relevant columns
    feature_cols = [
        "resources.cpu",
        "resources.memory",
        "pods_total",
        "pods_pending",
        "percent_pending",
        "timestamp",
        "cluster_load",
        "cluster_label",
        "cluster_cpu_capacity",
        "cluster_memory_capacity",
    ]
    X = df[feature_cols].copy()

    # Convert cpu/memory to numeric (e.g., '500m' -> 0.5, '1024Mi' -> 1024)
    def cpu_to_float(cpu):
        if isinstance(cpu, str) and cpu.endswith("m"):
            return float(cpu[:-1]) / 1000.0
        return float(cpu)

    def mem_to_float(mem):
        if isinstance(mem, str) and mem.endswith("Mi"):
            return float(mem[:-2])
        return float(mem)

    X["resources.cpu"] = X["resources.cpu"].apply(cpu_to_float)
    X["resources.memory"] = X["resources.memory"].apply(mem_to_float)
    X["cluster_cpu_capacity"] = X["cluster_cpu_capacity"].apply(cpu_to_float)
    X["cluster_memory_capacity"] = X["cluster_memory_capacity"].apply(mem_to_float)
    # Encode cluster_label
    X["cluster_label"] = X["cluster_label"].map({"private": 0, "public": 1})
    # If model is not provided, load the first available model
    if model is None:
        models = load_models()
        if not models:
            raise RuntimeError("No models found in models directory")
        model = list(models.values())[0]
    # Predict
    y_pred = model.predict(X)
    # Output (workload_id, kind, label)
    result = df[["workload_id", "kind"]].copy()
    result["label"] = y_pred
    result.to_csv(output_csv, index=False)
    logger.info(f"Recommendations written to {output_csv}")


def process_monitoring_data(data, timestamp_lookback_seconds=30):
    """
    Process monitoring data from different formats and extract workloads.
    Processes data from the last timestamp_lookback_seconds seconds of timestamps.

    Args:
        data: The loaded JSON data which can be in different formats
        timestamp_lookback_seconds: Number of seconds to look back in time

    Returns:
        List of workload objects with relevant cluster information
    """
    # Handle different data structures
    if isinstance(data, dict) and all(
        isinstance(key, str) and key.isdigit() for key in data.keys()
    ):
        # Get all timestamps and sort them
        timestamps = sorted([int(ts) for ts in data.keys()], reverse=True)

        if not timestamps:
            logger.error("No valid timestamps found in data")
            return []

        latest_timestamp = str(timestamps[0])
        logger.info(
            format_message(f"Latest timestamp: {latest_timestamp}", color="CYAN")
        )

        # Select timestamps from the last timestamp_lookback_seconds seconds
        cutoff_timestamp = timestamps[0] - timestamp_lookback_seconds
        recent_timestamps = [str(ts) for ts in timestamps if ts >= cutoff_timestamp]

        logger.info(
            format_message(
                f" Processing data from {len(recent_timestamps)} timestamps in the last {timestamp_lookback_seconds} seconds",
                icon="⏱️",
                color="MAGENTA",
            )
        )

        # Get cluster info from the latest timestamp (assuming it doesn't change much)
        latest_data = data[latest_timestamp]
        cluster_info = latest_data.get("cluster_info", [])

        # Create a dictionary of cluster information for easy lookup
        cluster_data = {}
        for cluster in cluster_info:
            if "cluster_label" in cluster:
                cluster_data[cluster["cluster_label"]] = {
                    "cpu_load": cluster.get("cluster_load", {}).get("cpu", 0),
                    "memory_load": cluster.get("cluster_load", {}).get("memory", 0),
                    "cpu_capacity": cluster.get("cluster_cpu_capacity", "8000m"),
                    "memory_capacity": cluster.get(
                        "cluster_memory_capacity", "16384Mi"
                    ),
                }

        logger.info(
            format_message(
                f"Found {len(cluster_data)} clusters: {', '.join(cluster_data.keys())}",
                icon="🔍",
                color="CYAN",
            )
        )

        # Collect workloads from all recent timestamps
        all_workloads = []
        for ts in recent_timestamps:
            timestamp_data = data[ts]
            raw_workloads = timestamp_data.get("workloads", [])

            # Add timestamp to each workload
            for w in raw_workloads:
                w["timestamp"] = int(ts)
                all_workloads.append(w)

        # Create a dictionary to store the latest state of each workload
        latest_workloads = {}
        for w in all_workloads:
            workload_id = w.get("workload_id")
            if workload_id:
                # If this workload is already in the dictionary, only replace it if this one is newer
                if (
                    workload_id not in latest_workloads
                    or w["timestamp"] > latest_workloads[workload_id]["timestamp"]
                ):
                    latest_workloads[workload_id] = w

        # Filter out workloads with zero resources
        workloads = []
        for workload_id, w in latest_workloads.items():
            # Check if resources are zero
            cpu = w.get("resources", {}).get("cpu", "0")
            memory = w.get("resources", {}).get("memory", "0")

            # Skip workloads with zero resources
            if cpu == "0m" and memory == "0Mi":
                logger.debug(f"Skipping workload {workload_id} with zero resources")
                continue

            # Add cluster information to the workload
            cluster_label = w.get("cluster_label")
            if cluster_label and cluster_label in cluster_data:
                w["cluster_load"] = cluster_data[cluster_label]["cpu_load"]
                w["cluster_cpu_capacity"] = cluster_data[cluster_label]["cpu_capacity"]
                w["cluster_memory_capacity"] = cluster_data[cluster_label][
                    "memory_capacity"
                ]

            workloads.append(w)

        logger.info(
            format_message(
                f"Filtered to {len(workloads)} unique workloads with non-zero resources from the last {timestamp_lookback_seconds} seconds",
                icon="🔄",
                color="CYAN",
            )
        )
    else:
        # Assume it's the old format (array of workloads)
        logger.info("Processing data in legacy format")
        workloads = data

    return workloads


def write_recommendations(result_df, output_dir=OUTPUT_DIR):
    # Log detailed information about workload migrations
    migrated_workloads = result_df[result_df["label"] == 1]
    non_migrated_workloads = result_df[result_df["label"] == 0]

    # Log summary statistics
    total_workloads = len(result_df)
    migrated_count = len(migrated_workloads)
    non_migrated_count = len(non_migrated_workloads)

    # Use formatted messages with colors and icons
    logger.info(
        format_message(
            f"Total workloads processed: {total_workloads}",
            icon="📊",
            color="CYAN",
            bold=True,
        )
    )
    logger.info(
        format_message(
            f" Workloads to be migrated to public cluster: {migrated_count} ({migrated_count/total_workloads*100:.1f}%)",
            icon="☁️",
            color="BLUE",
            bold=True,
        )
    )
    logger.info(
        format_message(
            f"Workloads remaining in private cluster: {non_migrated_count} ({non_migrated_count/total_workloads*100:.1f}%)",
            icon="🔁",
            color="GREEN",
            bold=True,
        )
    )

    # Log detailed information about each workload
    if not migrated_workloads.empty:
        logger.info(
            format_message(
                " Workloads to be migrated to public cluster:",
                icon="☁️",
                color="BLUE",
                bold=True,
            )
        )
        for _, row in migrated_workloads.iterrows():
            logger.info(
                format_message(
                    f"Workload ID: {row['workload_id']}, Kind: {row['kind']}",
                    color="BLUE",
                )
            )

    if not non_migrated_workloads.empty:
        logger.info(
            format_message(
                "Workloads remaining in private cluster:",
                icon="🔁",
                color="GREEN",
                bold=True,
            )
        )
        for _, row in non_migrated_workloads.iterrows():
            logger.info(
                format_message(
                    f"Workload ID: {row['workload_id']}, Kind: {row['kind']}",
                    color="GREEN",
                )
            )

    # Write recommendations to CSV
    try:
        output_csv = os.path.join(output_dir, "recommendations.csv")
        result_df.to_csv(output_csv, index=False)
        logger.info(
            format_message(
                f"Recommendations written to {output_csv}", icon="📝", color="MAGENTA"
            )
        )
    except Exception as e:
        logger.error(f"Error writing recommendations to CSV: {e}")
    
    return output_csv


def load_monitoring_data(config):
    """
    Load monitoring data from the configured input file.

    Args:
        config: Configuration dictionary

    Returns:
        Processed workload data or None if input file is not specified
    """
    json_input = config.get("data", {}).get("input_json")

    if not json_input:
        logger.error("❌ No input JSON file specified in the configuration")
        return None

    # Make sure the path is absolute or relative to the current directory
    if not os.path.isabs(json_input):
        json_input = os.path.join(os.path.dirname(__file__), json_input)

    try:
        with open(json_input, "r") as f:
            data = json.load(f)

        workloads = process_monitoring_data(data)
        return workloads
    except Exception as e:
        logger.error(f"❌ Error loading monitoring data: {str(e)}")
        return None


def analyze_workloads(workloads, config):
    """
    Analyze workloads using the configured AI model.

    Args:
        workloads: List of workload objects to analyze
        config: Configuration dictionary

    Returns:
        Tuple containing (DataFrame with results, explanations dictionary)
    """
    # Get AI model to use from config (default to crewai if not specified)
    model_type = config.get("ai", {}).get("model", "crewai").lower()

    # Analyze workloads with the selected model
    if model_type == "gemini":
        logger.info(
            format_message(
                "Using Gemini model for workload analysis",
                icon="🧠",
                color="MAGENTA",
                bold=True,
            )
        )
        # Unpack the tuple returned by label_workloads_with_gemini
        labels, explanations = label_workloads_with_gemini(workloads)
    else:
        logger.info(
            format_message(
                "Using CrewAI for workload analysis",
                icon="🧠",
                color="MAGENTA",
                bold=True,
            )
        )
        # Unpack the tuple returned by label_workloads_with_crewai
        labels, explanations = label_workloads_with_crewai(workloads)

    # Create result DataFrame
    df = pd.DataFrame(workloads)
    result = df[["workload_id", "kind"]].copy()
    result["label"] = labels

    return result, explanations


def save_and_log_explanations(result_df, explanations):
    """
    Save explanations to a file and log them.

    Args:
        result_df: DataFrame with workload results
        explanations: Dictionary with explanations
    """
    # Save explanations to a JSON file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(ENGINE_LOG_DIR):
        os.makedirs(ENGINE_LOG_DIR, exist_ok=True)
    explanations_file = os.path.join(
        ENGINE_LOG_DIR, f"recommendations_explanations_{timestamp}.json"
    )

    with open(explanations_file, "w") as f:
        json.dump(explanations, f, indent=2)
    logger.info(f"Explanations written to {explanations_file}")

    # Log the explanations
    logger.info(
        format_message(
            f"Overall explanation: {explanations.get('explanation', 'No overall explanation provided')}",
            icon="💡",
            color="YELLOW",
            bold=True,
        )
    )
    logger.info(format_message("Detailed explanations for each workload:", bold=True))
    for idx, explanation in enumerate(explanations.get("workload_explanations", [])):
        if idx < len(result_df):
            workload_id = result_df.iloc[idx]["workload_id"]
            kind = result_df.iloc[idx]["kind"]
            label = result_df.iloc[idx]["label"]
            cluster = "public" if label == 1 else "private"
            # Use different colors based on the cluster
            color = "BLUE" if label == 1 else "GREEN"
            logger.info(
                format_message(
                    f"Workload {workload_id} ({kind}) → {cluster}: {explanation}",
                    icon="💡",
                    color=color,
                )
            )
        else:
            logger.info(
                format_message(
                    f"Additional explanation {idx}: {explanation}",
                    icon="💡",
                    color="YELLOW",
                )
            )

    # If there are any additional explanation fields, log them too
    for key, value in explanations.items():
        if key not in ["explanation", "workload_explanations"]:
            logger.info(f"💡 {key}: {value}")



