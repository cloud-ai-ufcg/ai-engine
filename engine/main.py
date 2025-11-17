import os
import json
import pandas as pd
import datetime
from .ai_config import WorkloadRecommendation

from .util import (
    get_logger,
    load_config,
    format_message,
    COLORS,
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
            logger.error(
                f"Failed to convert structured recommendations to DataFrame: {e}"
            )
            return None

    migrated_workloads = df[df["label"] == 1]
    non_migrated_workloads = df[df["label"] == 0]

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
                    f"Workload ID: {row['workload_id']}, Kind: {row['kind']}, Reason: {row.get('reason', 'N/A')}",
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
                    f"Workload ID: {row['workload_id']}, Kind: {row['kind']}, Reason: {row.get('reason', 'N/A')}",
                    color="GREEN",
                )
            )

    try:
        output_csv = os.path.join(output_dir, "recommendations.csv")
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
        workloads, provider=provider
    )

    df = pd.DataFrame(workloads)
    result = df[["workload_id", "kind"]].copy()
    result["label"] = pd.Series(labels)
    result["label"] = result["label"].fillna(0).astype(int)

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
        destination_cluster = int(row.get("label", 0))
        origin_label = origin_by_id.get(wid, "private")
        origin_cluster = 0 if origin_label == "private" else 1

        reason = (
            explanations_list[idx]
            if idx < len(explanations_list)
            else (
                f"Recommended to {'public' if destination_cluster == 1 else 'private'} cluster based on resource analysis"
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
            kind = result_df.iloc[idx]["kind"]
            label = result_df.iloc[idx]["label"]
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
