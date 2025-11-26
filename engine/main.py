import os
import json
import pandas as pd
import datetime
import concurrent.futures
from .ai_config import WorkloadRecommendation
from .cluster_config import get_cluster_manager

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
            df[columns_to_save].to_csv(output_csv, index=False)
        else:
            # Fallback: write everything
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
        logger.error(f"❌ Error loading monitoring data: {str(e)}")
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
    cluster_manager = get_cluster_manager()
    
    # labels now contain cluster IDs (strings) instead of binary decisions
    labels, explanations = label_workloads(
        workloads, cluster_info=cluster_info, interval_duration=interval_duration, provider=provider
    )

    df = pd.DataFrame(workloads)
    result = df[["workload_id", "kind"]].copy()
    
    result["origin_cluster"] = df["cluster_label"].apply(
        lambda label: cluster_manager.resolve_cluster_label_to_id(label) or label
    )
    
    result["destination_cluster"] = labels

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


def shard_and_analyze_workloads(workloads, config):
    """Analyze workloads in shards using threads.

    Args:
        workloads: List of workload items to analyze.
        config: Configuration dictionary loaded from YAML.

    Returns:
        Tuple[pd.DataFrame, dict]: Combined DataFrame with all shard results and aggregated explanations.
    """
    # Retrieve shard size from config; fallback to processing all at once
    shard_size = int(
        config.get('ai', {}).get('workloads_shard_size', len(workloads))
    ) or len(workloads)
    if shard_size <= 0:
        shard_size = len(workloads)

    # Split workloads into shards
    shards = [
        workloads[i : i + shard_size] for i in range(0, len(workloads), shard_size)
    ]

    results = []
    combined_explanations = {"workload_explanations": []}

    def _analyze(shard_idx, shard):
        """Analyze a shard, logging its thread and position."""
        import threading  # local import avoids adding a new top-level import

        logger.info(
            format_message(
                f"🔢 Starting shard {shard_idx + 1}/{len(shards)} "
                f"({len(shard)} workloads) on thread {threading.current_thread().name}",
                color="CYAN",
            )
        )
        df, expl = analyze_workloads(shard, config)
        logger.info(
            format_message(
                f"✅ Finished shard {shard_idx + 1}/{len(shards)}",
                color="GREEN",
            )
        )
        return df, expl

    # Create a thread for each shard
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(shards)) as executor:
        future_to_shard = {
            executor.submit(_analyze, idx, shard): idx
            for idx, shard in enumerate(shards)
        }
        for future in concurrent.futures.as_completed(future_to_shard):
            try:
                shard_df, shard_expl = future.result()
                results.append(shard_df)
                if shard_expl:
                    combined_explanations["workload_explanations"].extend(
                        shard_expl.get("workload_explanations", [])
                    )
            except Exception as exc:
                logger.error(f"Error analyzing shard: {exc}")

    combined_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    return combined_df, combined_explanations


def save_and_log_explanations(
    result_df, explanations, workloads=None, batch_id: int | None = None
):
    """
    Save recommendations log using the WorkloadRecommendation schema and log them.

    Args:
        result_df: DataFrame with workload results (now contains origin_cluster and destination_cluster as strings)
        explanations: Dictionary with explanations
        workloads: Optional original workloads list to infer origin_cluster
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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

    cluster_manager = get_cluster_manager()
    
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
        destination_cluster = str(row.get("destination_cluster", row.get("label", "unknown")))
        
        origin_label = origin_by_id.get(wid, row.get("origin_cluster", "private"))
        if isinstance(origin_label, str) and origin_label in ["0", "1"]:
            # Handle legacy binary cluster labels
            origin_cluster = "private" if origin_label == "0" else "public"
        else:
            origin_cluster = str(origin_label)

        reason = (
            explanations_list[idx]
            if idx < len(explanations_list)
            else (
                f"Recommended to {destination_cluster} cluster based on resource analysis"
            )
        )
        
        # Get profile information if available
        origin_profile = None
        destination_profile = None
        origin_config = cluster_manager.get_cluster_by_id(origin_cluster) or cluster_manager.get_cluster_by_label(origin_cluster)
        dest_config = cluster_manager.get_cluster_by_id(destination_cluster) or cluster_manager.get_cluster_by_label(destination_cluster)
        
        if origin_config:
            origin_profile = origin_config.cluster_profile
        if dest_config:
            destination_profile = dest_config.cluster_profile

        recs_payload.append(
            WorkloadRecommendation(
                batch_id=batch_id,
                workload_id=wid,
                kind=kind,
                origin_cluster=origin_cluster,
                destination_cluster=destination_cluster,
                reason=reason,
                origin_cluster_profile=origin_profile,
                destination_cluster_profile=destination_profile,
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
            kind = result_df.iloc[idx]["kind"]
            origin = str(result_df.iloc[idx].get("origin_cluster", "unknown"))
            destination = str(result_df.iloc[idx].get("destination_cluster", result_df.iloc[idx].get("label", "unknown")))
            
            if origin != destination:
                icon = "🔄"
                color = "BLUE"
            else:
                icon = "✅"
                color = "GREEN"
            
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
