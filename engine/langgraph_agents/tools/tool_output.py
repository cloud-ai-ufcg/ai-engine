from langchain.tools import tool
from typing import List, Dict, Any

@tool("final_recommendations", return_direct=True)
def final_recommendations(decisions: List[Dict[str, Any]], overall_explanation: str) -> Dict[str, Any]:
    """
    Formats the final migration decisions and explanations into a structured JSON.

    Args:
        decisions: A list of workload decisions (workload_id, decision, reason).
        overall_explanation: A high-level summary of the recommendations.
    """
    return {"decisions": decisions, "overall_explanation": overall_explanation}