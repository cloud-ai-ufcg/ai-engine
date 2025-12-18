import os
import json
import pandas as pd
from datetime import datetime, timedelta
from .cluster_config import get_cluster_manager
from typing import Dict, List, Any


from .util import (
    get_logger,
    format_message,
    OUTPUT_DIR,
    ENGINE_LOG_DIR,
    load_config,
)
from .ai_config import WorkloadRecommendation
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
                w["timestamp"] = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").timestamp()
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
      - A pandas DataFrame with columns [workload_id, kind, label] where label is now a cluster ID (string), or
      - A list of dicts shaped like WorkloadRecommendation with fields
        [workload_id, kind, origin_cluster, destination_cluster, reason].

    Behavior remains backward compatible (CSV with workload_id, kind, destination_cluster).
    """
    structured_input = False
    df = result_df
    
    # If recommendations come in the new structured list[dict] form, convert to DataFrame
    if not hasattr(result_df, "to_dict") and isinstance(result_df, list):
        structured_input = True
        try:
            df = pd.DataFrame(result_df)
            # Map destination_cluster -> label for reporting
            if "label" not in df.columns and "destination_cluster" in df.columns:
                df["label"] = df["destination_cluster"].astype(str)
        except Exception as e:
            logger.error(
                f"Failed to convert structured recommendations to DataFrame: {e}"
            )
            return None

    # Count migrations (workloads that changed clusters)
    migrations_list = []
    non_migrations_list = []
    
    for idx, row in df.iterrows():
        origin = str(row.get("origin_cluster", "unknown"))
        destination = str(row.get("destination_cluster", row.get("label", "unknown")))
        
        if origin != destination:
            migrations_list.append(row)
        else:
            non_migrations_list.append(row)
    
    migrated_workloads = pd.DataFrame(migrations_list) if migrations_list else pd.DataFrame()
    non_migrated_workloads = pd.DataFrame(non_migrations_list) if non_migrations_list else pd.DataFrame()

    total_workloads = len(df)
    migrated_count = len(migrated_workloads)
    non_migrated_count = len(non_migrated_workloads)

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
            f"Workloads recommended for migration: {migrated_count} ({migrated_count/total_workloads*100:.1f}%)",
            icon="🔄",
            color="BLUE",
            bold=True,
        )
    )
    pct = (non_migrated_count / total_workloads * 100) if total_workloads else 0.0

    logger.info(
        format_message(
            f"Workloads staying in current cluster: {non_migrated_count} ({non_migrated_count/total_workloads*100:.1f}%)",
            icon="✅",
            color="GREEN",
            bold=True,
        )
    )

    if not migrated_workloads.empty:
        logger.info(
            format_message(
                "Workloads recommended for migration:",
                icon="🔄",
                color="BLUE",
                bold=True,
            )
        )
        for _, row in migrated_workloads.iterrows():
            origin = str(row.get("origin_cluster", "unknown"))
            destination = str(row.get("destination_cluster", row.get("label", "unknown")))
            logger.info(
                format_message(
                    f"Workload ID: {row['workload_id']}, Kind: {row['kind']}, {origin} → {destination}, Reason: {row.get('reason', 'N/A')}",
                    color="BLUE",
                )
            )

    if not non_migrated_workloads.empty:
        logger.info(
            format_message(
                "Workloads staying in current cluster:",
                icon="✅",
                color="GREEN",
                bold=True,
            )
        )
        for _, row in non_migrated_workloads.iterrows():
            cluster = str(row.get("origin_cluster", row.get("label", "unknown")))
            logger.info(
                format_message(
                    f"Workload ID: {row['workload_id']}, Kind: {row['kind']}, Cluster: {cluster}, Reason: {row.get('reason', 'N/A')}",
                    color="GREEN",
                )
            )

    try:
        output_csv = os.path.join(output_dir, "recommendations.csv")
        # Persist columns relevant to multi-cluster
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
            df[columns_to_save].to_csv(output_csv, index=False)
        else:
            # Fallback: write everything
            # df.to_csv(output_csv, index=False)
            df.to_csv(output_csv, index=False)
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
    provider = config.get("ai", {}).get("default_config", {}).get("provider", "openrouter")

    logger.info(
        format_message(
            f"Using {provider} platform for workload analysis",
            icon="🧠",
            color="MAGENTA",
            bold=True,
        )
    )
    
    labels, explanations = label_workloads(
        workloads, cluster_info=cluster_info, interval_duration=interval_duration, provider=provider
    )

    df = pd.DataFrame(workloads)
    result = df[["workload_id", "kind"]].copy()
    result["origin_cluster"] = df["cluster_id"]
    
    # Normalize labels and explanations to match result length
    def _normalize_list(items, target_len, default_value):
        """Normalize list length by truncating or padding."""
        if len(items) == target_len:
            return items
        if len(items) > target_len:
            logger.warning(f"Truncating {len(items)} items to {target_len}")
            return items[:target_len]
        missing = target_len - len(items)
        logger.warning(f"Padding with {missing} default values")
        return items + [default_value] * missing
    
    result["destination_cluster"] = _normalize_list(labels, len(result), df.iloc[0]["cluster_id"] if len(df) > 0 else "unknown")
    
    expl_list = explanations.get("workload_explanations", []) if isinstance(explanations, dict) else []
    result["reason"] = _normalize_list(expl_list, len(result), "No explanation provided")

    return result, explanations


def save_and_log_explanations(
    result_df, explanations, workloads=None, batch_id: int | None = None
):
    """
    Save recommendations log using the WorkloadRecommendation schema and log them.

    Args:
        result_df: DataFrame with workload results (contains origin_cluster and destination_cluster as cluster IDs)
        explanations: Dictionary with explanations
        workloads: Optional original workloads list to infer origin_cluster
        batch_id: Optional batch ID; incremented if not provided
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    global CURRENT_BATCH_ID

    if batch_id is None:
        CURRENT_BATCH_ID += 1
        batch_id = CURRENT_BATCH_ID

    os.makedirs(ENGINE_LOG_DIR, exist_ok=True)
    explanations_file = os.path.join(
        ENGINE_LOG_DIR, f"recommendations_explanations_{timestamp}.json"
    )

    cluster_manager = get_cluster_manager()
    
    # Map workload_id -> origin cluster from original workloads
    origin_by_id = {w.get("workload_id"): w.get("cluster_id") for w in (workloads or []) if w.get("workload_id")}
    explanations_list = (explanations or {}).get("workload_explanations", [])

    def _get_cluster_profile(cluster_id):
        """Helper to get cluster profile, checking by ID first then by label."""
        config = cluster_manager.get_cluster_by_id(cluster_id) or cluster_manager.get_cluster_by_label(cluster_id)
        return config.cluster_profile if config else None

    def _serialize_model(obj):
        """Convert Pydantic model to dict if needed."""
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        elif hasattr(obj, "dict"):
            return obj.dict()
        return obj

    recs_payload = []

    for idx, row in result_df.iterrows():
        wid = row.get("workload_id")
        kind = row.get("kind")
        origin_cluster = str(origin_by_id.get(wid, row.get("origin_cluster")))
        destination_cluster = str(row.get("destination_cluster", row.get("label")))
        
        reason = (
            explanations_list[idx]
            if idx < len(explanations_list)
            else f"Recommended to {destination_cluster} cluster based on resource analysis"
        )

        recs_payload.append(
            WorkloadRecommendation(
                batch_id=batch_id,
                workload_id=wid,
                kind=kind,
                origin_cluster=origin_cluster,
                destination_cluster=destination_cluster,
                reason=reason,
                origin_cluster_profile=_get_cluster_profile(origin_cluster),
                destination_cluster_profile=_get_cluster_profile(destination_cluster),
            )
        )

    # Serialize and save recommendations
    with open(explanations_file, "w") as f:
        json.dump([_serialize_model(r) for r in recs_payload], f, indent=2)
    logger.info(f"Explanations written to {explanations_file}")

    logger.info(format_message("Detailed explanations for each workload:", bold=True))

    for idx, explanation in enumerate(explanations.get("workload_explanations", [])):
        if idx < len(result_df):
            workload_id = result_df.iloc[idx]["workload_id"]
            kind = result_df.iloc[idx]["kind"]
            origin = str(result_df.iloc[idx].get("origin_cluster"))
            destination = str(result_df.iloc[idx].get("destination_cluster", result_df.iloc[idx].get("label")))
            
            icon, color = ("🔄", "BLUE") if origin != destination else ("✅", "GREEN")
            
            logger.info(
                format_message(
                    f"Workload {workload_id} ({kind}) {origin} → {destination}: {explanation}",
                    icon=icon,
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
