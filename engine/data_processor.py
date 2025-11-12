"""
Data processing utilities for monitoring data and workload filtering.

This module handles the transformation of raw monitoring data into normalized
workload structures, independent of AI agent functionality.
"""

import os
import json
from typing import Dict, List, Any
from .ai_config import build_cluster_selection_from_config

from .util import (
    get_logger,
    load_config,
    format_message,
)

logger = get_logger("data_processor")


def _filter_and_fill_workloads(
    latest_workloads: Dict[str, Any],
    cluster_data: Dict[str, Any],
    timestamp_lookback_seconds: int,
) -> List[Dict[str, Any]]:
    """Filter workloads with non-zero resources and fills them with cluster information.

    Args:
        latest_workloads: Mapping of workload_id to latest workload dict.
        cluster_data: Mapping of cluster labels to load/capacity data.
        timestamp_lookback_seconds: Window size for logging context.

    Returns:
        List of filled workload dicts.
    """
    workloads: List[Dict[str, Any]] = []

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
            f"Filtered to {len(workloads)} unique workloads with non-zero resources from the last {timestamp_lookback_seconds} seconds",
            icon="🔄",
            color="GREEN",
        )
    )
    return workloads


def process_monitoring_data(data, timestamp_lookback_seconds=None):
    """
    Process monitoring data from different formats and extract workloads.
    Processes data from the last timestamp_lookback_seconds seconds of timestamps.

    Args:
        data: The loaded JSON data which can be in different formats
        timestamp_lookback_seconds: Number of seconds to look back in time (defaults to config value)

    Returns:
        List of workload objects with relevant cluster information
    """
    if timestamp_lookback_seconds is None:
        config = load_config()
        timestamp_lookback_seconds = config.get("ai", {}).get(
            "timestamp_lookback_seconds", 30
        )

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
            format_message(f"Latest timestamp: {latest_timestamp}", color="GREEN")
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

        # Get the cluster selection from config
        listed_clusters = build_cluster_selection_from_config()

        if listed_clusters:
            logger.info(
                format_message(
                    f"Using listed clusters from config: {', '.join(listed_clusters)}",
                    icon="🔍",
                    color="CYAN",
                )
            )

        # Get cluster info from the latest timestamp (assuming it doesn't change much)
        latest_data = data[latest_timestamp]
        cluster_info = latest_data.get("cluster_info", [])

        # Create a dictionary of cluster information for easy lookup
        cluster_data = {}
        for cluster in cluster_info:
            if "cluster_label" in cluster:
                cluster_label = cluster["cluster_label"]

                if listed_clusters and (cluster_label not in listed_clusters):
                    # Skip clusters that are not in the listed clusters
                    continue

                cluster_data[cluster_label] = {
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
                if not listed_clusters or (w.get("cluster_label") in listed_clusters):
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

        # Filter out zero-resource workloads and fill with cluster info
        workloads = _filter_and_fill_workloads(
            latest_workloads, cluster_data, timestamp_lookback_seconds
        )
    else:
        # Assume it's the old format (array of workloads)
        logger.warning("Processing data in legacy format")
        workloads = data

    return workloads
