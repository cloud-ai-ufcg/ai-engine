import json
from typing import Dict, List

def load_workloads_from_monitor(path: str) -> Dict[str, List[Dict]]:
    """
    Reads monitor_outputs.json with timestamp structure as key
    and returns a dictionary with:
      - workloads: list of workloads
      - cluster_info: list of clusters

    Useful for use as initial state in LangGraph.
    """
    with open(path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Expected a dictionary with timestamps as keys.")

    # Pega o primeiro snapshot disponível
    first_snapshot = next(iter(data.values()))

    workloads = first_snapshot.get("workloads", [])
    cluster_info = first_snapshot.get("cluster_info", [])

    return {
        "workloads": workloads,
        "cluster_info": cluster_info,
    }