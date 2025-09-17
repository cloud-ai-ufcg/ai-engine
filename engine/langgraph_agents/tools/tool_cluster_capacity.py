from typing import List, Dict, Any
from langchain.tools import tool

@tool("calculate_cluster_capacity", return_direct=False)
def calculate_cluster_capacity(data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Calculates the total CPU and memory allocation per cluster (public/private) for each timestamp.

    Args:
        data: Mapping of timestamp (str) -> {
            "workloads": [ ... ],
            "cluster_info": [ ... ]  # ignored here
        }

    Returns:
        Dict mapping timestamp -> dict by cluster_label -> capacity info
        Example:
        {
            "1756491164": {
                "private": {
                    "cpu_allocated": "18000m",
                    "memory_allocated": "20480Mi"
                },
                "public": {
                    "cpu_allocated": "9000m",
                    "memory_allocated": "73728Mi"
                }
            }
        }
    """

    if not data:
        return {"error": "No data provided"}

    if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, dict) for k, v in data.items()):
        return {"error": "Expected dict of timestamps -> {'workloads': [...]}."}

    results_by_ts: Dict[str, Dict[str, Dict[str, str]]] = {}

    for ts, ts_payload in data.items():
        workloads = ts_payload.get("workloads", []) or []
        cluster_data: Dict[str, Dict[str, int]] = {}

        for workload in workloads:
            cluster_label = workload.get("cluster_label")
            if not cluster_label:
                continue

            if cluster_label not in cluster_data:
                cluster_data[cluster_label] = {
                    "cpu_allocated": 0,
                    "memory_allocated": 0,
                }

            resources = workload.get("resources", {})
            cpu_str = resources.get("cpu", "0m")
            memory_str = resources.get("memory", "0Mi")

            cpu_allocated = int(cpu_str.replace("m", "")) if isinstance(cpu_str, str) and cpu_str.endswith("m") else 0
            memory_allocated = int(memory_str.replace("Mi", "")) if isinstance(memory_str, str) and memory_str.endswith("Mi") else 0

            total_cpu = cpu_allocated
            total_memory = memory_allocated

            cluster_data[cluster_label]["cpu_allocated"] += total_cpu
            cluster_data[cluster_label]["memory_allocated"] += total_memory

        formatted: Dict[str, Dict[str, str]] = {}
        for cluster_label, agg in cluster_data.items():
            formatted[cluster_label] = {
                "cpu_allocated": f"{agg['cpu_allocated']}m",
                "memory_allocated": f"{agg['memory_allocated']}Mi",
            }

        results_by_ts[ts] = formatted

    return results_by_ts