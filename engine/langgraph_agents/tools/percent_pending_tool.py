from langchain.tools import tool

@tool("pending_percentage", return_direct=False)
def pending_percentage(data: dict) -> dict:
    """
    Calculate the percentage of pending pods for each workload from the latest timestamp.
    Input:
      - dict with timestamps as keys and workload data as values
    Output:
      - dict {workload_id: "XX%"} for each workload
    """
    if not data:
        return {"error": "no data provided"}

    try:
        latest_ts = max(data.keys(), key=lambda x: int(x))
    except Exception:
        return {"error": "timestamps not in expected format"}

    workloads = data[latest_ts].get("workloads", [])
    if not workloads:
        return {"error": "not found workloads"}

    result = {}
    for wl in workloads:
        wl_id = wl.get("workload_id", "unknown")
        percent = wl.get("percent_pending", 0)
        result[wl_id] = f"{percent}%"

    return result