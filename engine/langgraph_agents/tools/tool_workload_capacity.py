from typing import List, Dict, Any
from langchain.tools import tool

@tool("calculate_workload_capacity", return_direct=False)
def calculate_workload_capacity(data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Calculates allocated (required) and used CPU/memory per workload for each timestamp.
    Rule: used = required * (active_pods / total_pods), active_pods = pods_total - pods_pending.

    Args:
        data: Mapping of timestamp (str) -> {
            "workloads": [ ... ],
            "cluster_info": [ ... ]
        }

    Returns:
        Dict mapping timestamp -> dict by workload_id -> capacity info
        Example:
        {
            "1756491164": {
                "default/1": {
                    "cpu_allocated": "7000m",
                    "memory_allocated": "12288Mi",
                    "cpu_used": "7000m",
                    "memory_used": "12288Mi"
                },
                "default/2": {
                    "cpu_allocated": "4000m", 
                    "memory_allocated": "2048Mi",
                    "cpu_used": "0m",
                    "memory_used": "0Mi"
                }
            }
        }
    """

    if not data:
        return {"error": "No data provided"}

    if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, dict) for k, v in data.items()):
        return {"error": "Expected dict of timestamps -> {'workloads': [...], 'cluster_info': [...]}."}

    results_by_ts: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for ts, ts_payload in data.items():
        workloads = ts_payload.get("workloads", []) or []
        
        workload_data: Dict[str, Dict[str, Any]] = {}

        for workload in workloads:
            workload_id = workload.get("workload_id")
            if not workload_id:
                continue

            resources = workload.get("resources", {})
            cpu_str = resources.get("cpu", "0m")
            memory_str = resources.get("memory", "0Mi")
            pods_total = workload.get("pods_total", 0) or 0
            pods_pending = workload.get("pods_pending", 0) or 0
            active_pods = max(pods_total - pods_pending, 0)

            # Parse allocated values
            cpu_alloc_m = int(cpu_str.replace("m", "")) if isinstance(cpu_str, str) and cpu_str.endswith("m") else 0
            mem_alloc_mi = int(memory_str.replace("Mi", "")) if isinstance(memory_str, str) and memory_str.endswith("Mi") else 0

            # Compute used proportionally to active pods
            if pods_total > 0:
                cpu_used = int(cpu_alloc_m * (active_pods / pods_total))
                memory_used = int(mem_alloc_mi * (active_pods / pods_total))
            else:
                cpu_used = 0
                memory_used = 0

            workload_data[workload_id] = {
                "cpu_allocated": cpu_str,
                "memory_allocated": memory_str,
                "cpu_used": f"{cpu_used}m",
                "memory_used": f"{memory_used}Mi",
            }

        results_by_ts[ts] = workload_data

    return results_by_ts