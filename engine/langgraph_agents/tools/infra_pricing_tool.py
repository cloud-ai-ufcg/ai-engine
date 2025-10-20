from typing import Dict, Any
from langchain_core.tools import tool
from .pricing_data import aws_instance_types


@tool("infra_pricing", return_direct=False)
def infra_pricing(data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    LangGraph tool to calculate infrastructure costs for clusters across all timestamps.
    
    This function processes monitoring data with timestamp-based cluster information
    and calculates the infrastructure cost for each cluster based on AWS instance pricing.
    
    The tool:
    - Processes all timestamps in the data
    - For each cluster, finds the minimum viable AWS instance type that can accommodate the node size
    - Calculates the cost per interval based on the hourly price and interval duration
    - Returns the infrastructure cost for each cluster at each timestamp
    
    Args:
        data: Dict with timestamp as key containing cluster_info and interval_duration.
        Example: {
            "1760540497": {
                "interval_duration": "30s",
                "cluster_info": [
                    {
                        "cluster_label": "private",
                        "node_info": {"cpu": 8, "memory": 16, "quantity": 19}
                    },
                    ...
                ]
            }
        }
        
    Returns:
        Dict mapping timestamp to dict of cluster_label to infrastructure cost (float).
        Example: {
            "1760540497": {"private": 1.234, "public": 0.987},
            "1760540467": {"private": 1.234, "public": 0.987}
        }
    """
    if not data:
        return {"error": "No data provided"}
    
    if not isinstance(data, dict):
        return {"error": "Data must be a dictionary"}
    
    # Result will store infrastructure costs for each timestamp
    result = {}
    
    # Process each timestamp
    for timestamp, timestamp_data in data.items():
        # Extract interval duration and cluster info
        interval_duration_str = timestamp_data.get("interval_duration", "")
        cluster_info_list = timestamp_data.get("cluster_info", [])
        
        if not cluster_info_list:
            result[timestamp] = {"error": "No cluster_info found"}
            continue
        
        # Parse interval duration (format: "30s" -> 30 seconds)
        try:
            interval_seconds = int(interval_duration_str.rstrip('s'))
        except (ValueError, AttributeError):
            result[timestamp] = {"error": f"Invalid interval_duration format: {interval_duration_str}"}
            continue
        
        # Calculate infrastructure cost for each cluster at this timestamp
        timestamp_costs = {}
        
        for cluster in cluster_info_list:
            cluster_label = cluster.get("cluster_label")
            if not cluster_label:
                continue
            
            node_info = cluster.get("node_info", {})
            node_cpu = node_info.get("cpu", 0)
            node_memory = node_info.get("memory", 0)
            node_quantity = node_info.get("quantity", 0)
            
            if node_cpu == 0 or node_memory == 0 or node_quantity == 0:
                timestamp_costs[cluster_label] = 0.0
                continue
            
            # Find minimum viable instance type
            min_instance = find_minimum_viable_instance(node_cpu, node_memory)
            
            if not min_instance:
                timestamp_costs[cluster_label] = 0.0
                continue
            
            # Calculate cost per interval
            hourly_price = min_instance["price_usd_per_hour"]
            interval_price = (hourly_price * interval_seconds) / 3600  # Convert hourly to interval
            total_cost = interval_price * node_quantity
            
            timestamp_costs[cluster_label] = round(total_cost, 6)
        
        result[timestamp] = timestamp_costs
    
    return result


def find_minimum_viable_instance(required_cpu: int, required_memory: int) -> Dict[str, Any]:
    """
    Find the minimum viable AWS instance type that meets or exceeds the required CPU and memory.
    
    Iterates through aws_instance_types in order (which is already sorted by size and price)
    and returns the first instance that meets the requirements.
    
    Args:
        required_cpu: Required number of vCPUs
        required_memory: Required memory in GB
        
    Returns:
        Dict containing instance information, or None if no suitable instance found
    """
    # Iterate through instances in order and return the first viable one
    # The pricing_data.py list is already ordered by size/price (ascending)
    for instance in aws_instance_types:
        if instance["vcpus"] >= required_cpu and instance["memory_gb"] >= required_memory:
            return instance
    
    # No viable instance found
    return None
