from typing import Dict, Any
from langchain_core.tools import tool
from .utils import *

@tool("workload_pricing", return_direct=False)
def workload_pricing(data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    LangGraph tool to estimate workload-level costs based on resource usage and AWS pricing.

    This tool calculates cost per workload over multiple timestamps by:
      - Estimating cluster infrastructure cost using AWS instance pricing (from pricing_data)
      - Distributing that cost proportionally among workloads based on their CPU and memory requests
      - Multiplying by the number of replicas (pods_total)
      - Adjusting for pending pods (percent_pending)
      - Summing across timestamps for cumulative totals

    Args:
        data: Dict with timestamps as keys and monitoring data as values.
              Each timestamp entry must include:
              {
                "interval_duration": "30s",
                "workloads": [...],
                "cluster_info": [...]
              }

              Workloads must contain CPU, memory, pods_total, percent_pending, and cluster_label.

    Returns:
        Dict mapping workload_id to estimated cost components:
        {
          "workload_id": {
            "interval_cost_total": <float>,   # Total estimated cost (all pods)
            "interval_cost_running": <float>  # Effective cost after adjusting for pending pods
          }
        }

    Notes:
        - CPU and memory are converted from Kubernetes-style units (e.g., "8000m", "32768Mi")
        - AWS pricing reference comes from pricing_data.py
        - Costs are per interval, aggregated across timestamps
    """
    if not data:
        return {}

    totals: Dict[str, Dict[str, float]] = {}

    for ts, ts_data in data.items():
        interval_str = ts_data.get("interval_duration", "30s")
        try:
            interval_seconds = int(str(interval_str).rstrip("s"))
        except Exception:
            interval_seconds = 30

        workloads = ts_data.get("workloads", []) or []
        cluster_info = ts_data.get("cluster_info", []) or []

        infra_costs: Dict[str, float] = {}
        for c in cluster_info:
            label = c.get("cluster_label")
            node_info = c.get("node_info", {}) or {}
            node_vcpus = node_info.get("cpu", 0)
            node_mem_gb = node_info.get("memory", 0)
            quantity = int(node_info.get("quantity", 1) or 1)

            inst = find_minimum_viable_instance(node_vcpus, node_mem_gb)
            if not inst:
                infra_costs[label] = 0.0
                continue

            hourly_price = float(inst["price_usd_per_hour"])
            hourly_total = hourly_price * max(1, quantity)
            interval_cost = (hourly_total * interval_seconds) / 3600.0
            infra_costs[label] = round(interval_cost, 10)

        cluster_caps: Dict[str, Dict[str, float]] = {}
        for c in cluster_info:
            label = c.get("cluster_label")
            cpu_cap = parse_millicores_to_cores(c.get("cluster_cpu_capacity", "0m"))
            mem_cap = parse_mebibytes_to_gb(c.get("cluster_memory_capacity", "0Mi"))
            cluster_caps[label] = {"cpu": cpu_cap, "memory": mem_cap}

        for w in workloads:
            wid = w.get("workload_id")
            if not wid:
                continue

            label = w.get("cluster_label")
            if label is None:
                continue

            caps = cluster_caps.get(label, {"cpu": 0.0, "memory": 0.0})
            infra_cost = float(infra_costs.get(label, 0.0))

            cpu_req = parse_millicores_to_cores(w.get("resources", {}).get("cpu", "0m"))
            mem_req = parse_mebibytes_to_gb(w.get("resources", {}).get("memory", "0Mi"))
            pods_total = int(w.get("pods_total", 1) or 1)
            percent_pending = float(w.get("percent_pending", 0) or 0)

            cpu_share = (cpu_req / caps["cpu"]) if caps["cpu"] > 0 else 0.0
            mem_share = (mem_req / caps["memory"]) if caps["memory"] > 0 else 0.0

            workload_share = (cpu_share + mem_share) / 2.0

            interval_cost_total = infra_cost * workload_share * pods_total
            interval_cost_running = interval_cost_total * (1.0 - percent_pending / 100.0)

            if wid not in totals:
                totals[wid] = {"interval_cost_total": 0.0, "interval_cost_running": 0.0}

            totals[wid]["interval_cost_total"] = round(
                totals[wid]["interval_cost_total"] + interval_cost_total, 10
            )
            totals[wid]["interval_cost_running"] = round(
                totals[wid]["interval_cost_running"] + interval_cost_running, 10
            )

    return totals