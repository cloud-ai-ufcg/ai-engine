import os
import json
import pandas as pd
import threading
from typing import Dict, List, Any
from datetime import datetime, timedelta
from .data_types import WorkloadRecommendation

from .util import (
    get_logger,
    format_message,
    OUTPUT_DIR,
    ENGINE_LOG_DIR,
)
from .agents import label_workloads
from .data_processor import process_monitoring_data

logger = get_logger("main")
CURRENT_BATCH_ID = 1

def _filter_and_fill_workloads(
    latest_workloads: Dict[str, Any],
    cluster_data: Dict[str, Any],
    timestamp_lookback_seconds: int,
) -> List[Dict[str, Any]]:
    """Filter workloads with non-zero resources and fills them with cluster
    information.

    Args:
        latest_workloads: Mapping of workload_id to latest workload dict.
        cluster_data: Mapping of cluster labels to load/capacity data.
        timestamp_lookback_seconds: Window size for logging context.

    Returns:
        List of filled workload dicts.
    """
    workloads: List[Dict[str, Any]] = []

    # Backwards-compat: if a list of snapshots was passed, convert it into
    # a mapping workload_id -> latest snapshot (by timestamp) so the body of
    # this function can remain unchanged and operate on a dict as before.
    if isinstance(latest_workloads, list):
        tmp: Dict[str, Any] = {}
        for w in latest_workloads:
            wid = w.get("workload_id") or w.get("id")
            if not wid:
                logger.debug("Skipping workload without workload_id")
                continue
            existing = tmp.get(wid)
            if existing is None or w.get("timestamp", 0) > existing.get("timestamp", 0):
                tmp[wid] = w
        latest_workloads = tmp

    for workload_id, w in latest_workloads.items():
        # Skip workloads with zero requested resources
        cpu = w.get("resources", {}).get("cpu", "0")
        memory = w.get("resources", {}).get("memory", "0")
        if cpu == "0m" and memory == "0Mi":
            logger.debug(f"Skipping workload {workload_id} with zero resources")
            continue

        # Fill cluster details if available
        cluster_label = w.get("cluster_label")
        if cluster_label and cluster_label in cluster_data:
            w["cluster_load"] = cluster_data[cluster_label]["cpu_load"]
            w["cluster_cpu_capacity"] = cluster_data[cluster_label]["cpu_capacity"]
            w["cluster_memory_capacity"] = cluster_data[cluster_label][
                "memory_capacity"
            ]

        workloads.append(w)

    # Summary log
    logger.info(
        format_message(
            f"Filtered to {len(workloads)} unique workloads with non-zero resources \n"
            f"from the last {timestamp_lookback_seconds} seconds",
            icon="🔄",
            color="GREEN",
        )
    )
    return workloads

def process_monitoring_data(data, timestamp_lookback_seconds=None):
    """
    Process monitoring data from different formats and extract workloads.
    Processes data from the last timestamp_lookback_seconds seconds of timestamps.

    Parses timestamps in "%Y-%m-%d %H:%M:%S" format into datetime objects.

    Args:
        data: JSON monitoring data in one of the supported formats
        timestamp_lookback_seconds: Number of seconds to look back
            (defaults to config value)

    Returns:
        dict with:
            - "workloads": list of workload dicts enriched with cluster info
            - "cluster_info": latest cluster information extracted from data
    """
    config = load_config()
    if timestamp_lookback_seconds is None:
        timestamp_lookback_seconds = config.get("ai", {}).get(
            "timestamp_lookback_seconds", 30
        )

    preserve_history = bool(config.get("ai", {}).get("preserve_history", False))

    if isinstance(data, dict) and all(
        isinstance(value, dict) and "workloads" in value for value in data.values()
    ):
        try:
            timestamps = sorted(
                data.keys(),
                key=lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"),
                reverse=True,
            )
        except Exception as e:
            logger.error(f"Error parsing timestamps: {e}")
            return {"workloads": [], "cluster_info": []}

        if not timestamps:
            return {"workloads": [], "cluster_info": []}

        latest_timestamp = timestamps[0]
        latest_dt = datetime.strptime(latest_timestamp, "%Y-%m-%d %H:%M:%S")
        cutoff_dt = latest_dt - timedelta(seconds=timestamp_lookback_seconds)
        cutoff_ts = cutoff_dt.strftime("%Y-%m-%d %H:%M:%S")

        recent_timestamps = [ts for ts in timestamps if ts >= cutoff_ts]

        latest_data = data[latest_timestamp]
        cluster_info = latest_data.get("cluster_info", [])

        # Prepare cluster mapping
        cluster_data = {}
        for cluster in cluster_info:
            label = cluster.get("cluster_label")
            if not label:
                continue
            cluster_data[label] = {
                "cpu_load": cluster.get("cluster_load", {}).get("cpu", 0),
                "memory_load": cluster.get("cluster_load", {}).get("memory", 0),
                "cpu_capacity": cluster.get("cluster_cpu_capacity", "8000m"),
                "memory_capacity": cluster.get("cluster_memory_capacity", "16384Mi"),
            }

        # Build list of snapshots
        all_workloads = []
        for ts in recent_timestamps:
            ts_data = data[ts]
            for w in ts_data.get("workloads", []):
                w["timestamp"] = datetime.strptime(
                    ts, "%Y-%m-%d %H:%M:%S"
                ).timestamp()
                all_workloads.append(w)

        latest_workloads = {}
        for w in all_workloads:
            wid = w.get("workload_id") or w.get("id")
            if not wid:
                continue
            prev = latest_workloads.get(wid)
            if prev is None or w["timestamp"] > prev["timestamp"]:
                latest_workloads[wid] = w

        filled = _filter_and_fill_workloads(
            latest_workloads, cluster_data, timestamp_lookback_seconds
        )

        return {
            "workloads": filled,
            "cluster_info": cluster_info,
        }

    return {
        "workloads": data,
        "cluster_info": [],
    }

def write_recommendations(result_df, output_dir=OUTPUT_DIR):
    """
    Write recommendations summary and CSV output.

    Accepts either:
      - A pandas DataFrame with columns [workload_id, kind, label] (legacy), or
      - A list of dicts shaped like WorkloadRecommendation with fields
        [workload_id, kind, origin_cluster, destination_cluster, reason].

    Behavior remains backward compatible (CSV with workload_id, kind, label).
    """
    structured_input = False
    df = result_df
    # If recommendations come in the new structured list[dict] form, convert to DataFrame
    if not hasattr(result_df, "to_dict") and isinstance(result_df, list):
        structured_input = True
        try:
            df = pd.DataFrame(result_df)
            # Map destination_cluster -> label for legacy-compatible reporting
            if "label" not in df.columns and "destination_cluster" in df.columns:
                df["label"] = df["destination_cluster"].astype(int)
        except Exception as e:
            logger.error(f"Failed to convert structured recommendations to DataFrame: {e}")
            return None

    migrated_workloads = df[df["label"] == 1]
    non_migrated_workloads = df[df["label"] == 0]
    invalid_workloads = df[df["label"] == -1]

    total_workloads = len(df)
    migrated_count = len(migrated_workloads)
    non_migrated_count = len(non_migrated_workloads)
    invalid_count = len(invalid_workloads)

    logger.info(
        format_message(
            f"Total workloads processed: {total_workloads}",
            icon="📊",
            color="CYAN",
            bold=True,
        )
    )
    if invalid_count > 0:
        logger.warning(
            format_message(
                f"Workloads ignored due to invalid LLM response: {invalid_count} ({invalid_count/total_workloads*100:.1f}%)",
                icon="❌",
                color="RED",
                bold=True,
            )
        )
    logger.info(
        format_message(
            f" Workloads to be migrated to public cluster: {migrated_count}",
            icon="☁️",
            color="BLUE",
            bold=True,
        )
    )
    pct = (non_migrated_count / total_workloads * 100) if total_workloads else 0.0

    logger.info(
        format_message(
            f"Workloads remaining in private cluster: {non_migrated_count} ({pct:.1f}%)",
            icon="🔁",
            color="GREEN",
            bold=True,
        )
    )

    try:
        output_csv = os.path.join(output_dir, "recommendations.csv")

        df_to_save = pd.concat([migrated_workloads, non_migrated_workloads])
        # Persist legacy CSV columns
        columns_to_save = [
            c
            for c in [
                "workload_id",
                "kind",
                "origin_cluster",
                "destination_cluster",
                "reason",
            ]
            if c in df.columns
        ]
        if columns_to_save:
            # df[columns_to_save].to_csv(output_csv, index=False)
            df_to_save[columns_to_save].to_csv(output_csv, index=False)
        else:
            # Fallback: write everything
            # df.to_csv(output_csv, index=False)
            df_to_save.to_csv(output_csv, index=False)
        logger.info(
            format_message(
                f"Recommendations written to {output_csv}", icon="📝", color="MAGENTA"
            )
        )
    except Exception as e:
        logger.error(f"Error writing recommendations to CSV: {e}")

    # If we received structured recommendations, also write a JSON artifact
    if structured_input:
        try:
            output_json = os.path.join(output_dir, "recommendations_structured.json")
            with open(output_json, "w") as f:
                json.dump(result_df, f, indent=2)
            logger.info(
                format_message(
                    f"Structured recommendations written to {output_json}",
                    icon="📝",
                    color="MAGENTA",
                )
            )
        except Exception as e:
            logger.error(f"Error writing structured recommendations JSON: {e}")

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

    if not os.path.isabs(json_input):
        json_input = os.path.join(os.path.dirname(__file__), json_input)

    try:
        with open(json_input, "r") as f:
            data = json.load(f)

        processed_data = process_monitoring_data(data)
        return processed_data
    except Exception as e:
        logger.error(f"❌ Error loading monitoring data: {e}")
        return None


def analyze_workloads(workloads, config, cluster_info=None, interval_duration=None):
    """
    Analyze workloads using the configured AI model.

    Args:
        workloads: List of workload objects to analyze
        config: Configuration dictionary
        cluster_info: List of cluster information dictionaries (optional)

    Returns:
        Tuple containing (DataFrame with results, explanations dictionary)
    """
    provider = (
        config.get("ai", {}).get("default_config", {}).get("provider", "openrouter")
    )

    logger.info(
        format_message(
            f"Using {provider} platform for workload analysis",
            icon="🧠",
            color="MAGENTA",
            bold=True,
        )
    )
    # Don't pass multiagent parameter - let label_workloads use config.mode instead
    labels, explanations = label_workloads(
        workloads,
        cluster_info=cluster_info,
        interval_duration=interval_duration,
        provider=provider,
    )

    df = pd.DataFrame(workloads)
    result = df[["workload_id", "kind"]].copy()

    if not result.empty:
        label_series = pd.Series(labels)

        numeric_labels = pd.to_numeric(label_series, errors='coerce')

        result["label"] = numeric_labels.fillna(-1).astype(int)

    else:

        result["label"] = pd.Series(dtype=int)
        logger.warning("No workloads processed; 'label' column initialized empty.")

    # Safely attach reasons column
    if (
        isinstance(explanations, dict)
        and "workload_explanations" in explanations
        and explanations["workload_explanations"]
    ):
        expl_list = explanations["workload_explanations"]
        if len(expl_list) == len(result):
            result["reason"] = expl_list
        else:
            # Pad / truncate to match length
            padded = (expl_list + ["No explanation."] * len(result))[: len(result)]
            result["reason"] = padded
    else:
        result["reason"] = "No explanation provided"

    return result, explanations


def save_and_log_explanations(
    result_df, explanations, workloads=None, batch_id: int | None = None
):
    """
    Save recommendations log using the WorkloadRecommendation schema and log them.

    Args:
        result_df: DataFrame with workload results
        explanations: Dictionary with explanations
        workloads: Optional original workloads list to infer origin_cluster
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Ensure we have a batch_id; if not provided, increment global counter

    global CURRENT_BATCH_ID

    if batch_id is None:
        CURRENT_BATCH_ID += 1
        batch_id = CURRENT_BATCH_ID

    if not os.path.exists(ENGINE_LOG_DIR):
        os.makedirs(ENGINE_LOG_DIR, exist_ok=True)
    explanations_file = os.path.join(
        ENGINE_LOG_DIR, f"recommendations_explanations_{timestamp}.json"
    )

    # Build WorkloadRecommendation-shaped list[dict]
    origin_by_id = {}
    if workloads:
        for w in workloads:
            wid = w.get("workload_id")
            if wid is not None:
                origin_by_id[wid] = w.get("cluster_label", "private")

    explanations_list = (explanations or {}).get("workload_explanations", [])

    recs_payload = []

    for idx, row in result_df.iterrows():
        wid = row.get("workload_id")
        kind = row.get("kind")
        destination_cluster = row.get("label", 0)

        destination_cluster = int(destination_cluster)

        origin_label = origin_by_id.get(wid, "private")
        origin_cluster = 0 if origin_label == "private" else 1

        reason = (
            explanations_list[idx]
            if idx < len(explanations_list)
            else (
                f"Recommended to {'public' if destination_cluster == 1 else 'private'} "
                f"cluster based on resource analysis"
            )
        )

        recs_payload.append(
            WorkloadRecommendation(
                batch_id=batch_id,
                workload_id=wid,
                kind=kind,
                origin_cluster=origin_cluster,
                destination_cluster=destination_cluster,
                reason=reason,
            )
        )

    # Serialize Pydantic models to plain dicts for JSON output
    recs_payload_serialized = [
        (
            r.model_dump()
            if hasattr(r, "model_dump")
            else (r.dict() if hasattr(r, "dict") else r)
        )
        for r in recs_payload
    ]

    with open(explanations_file, "w") as f:
        json.dump(recs_payload_serialized, f, indent=2)
    logger.info(f"Explanations written to {explanations_file}")

    logger.info(format_message("Detailed explanations for each workload:", bold=True))

    for idx, explanation in enumerate(explanations.get("workload_explanations", [])):
        if idx < len(result_df):
            workload_id = result_df.iloc[idx]["workload_id"]
            label = result_df.iloc[idx]["label"]

            if str(label) == "-1" or label == -1:
                continue

            kind = result_df.iloc[idx]["kind"]
            cluster = "public" if label == 1 else "private"
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

    for key, value in explanations.items():
        if key not in ["explanation", "workload_explanations"]:
            logger.info(f"💡 {key}: {value}")
