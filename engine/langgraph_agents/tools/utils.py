from typing import Dict, Any
from .pricing_data import aws_instance_types

def parse_mebibytes_to_gb(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).lower()
    try:
        if s.endswith("mi"):
            return float(s[:-2]) / 1024.0
        if s.endswith("gi"):
            return float(s[:-2])
        return float(s)
    except Exception:
        return 0.0
    
def parse_millicores_to_cores(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value)
    if s.endswith("m"):
        try:
            return float(s[:-1]) / 1000.0
        except Exception:
            return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0

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
