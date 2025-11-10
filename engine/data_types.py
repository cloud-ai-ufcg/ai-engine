from pydantic import BaseModel
from typing import Dict, List, Any, TypedDict


# Defines the shared state between nodes
class WorkflowState(TypedDict):
    
    workloads: List[Dict] 
    cpu_votes: List[int]
    mem_votes: List[int]
    pending_votes: List[int]
    final_decisions: List[int]
    explanations: Dict
    historical_context: str  # Historical recommendations context for LLM
    
class WorkloadBase(BaseModel):
    workload_id: str
    kind: str
    cluster_label: str
    resources: Dict[str, str]
    pods_total: int
    pods_pending: int
    percent_pending: float
    timestamp: int


class WorkloadInput(BaseModel):
    workloads: List[Dict[str, Any]]


class RecommendationResponse(BaseModel):
    workload_id: str
    kind: str
    label: int


class AnalysisResponse(BaseModel):
    recommendations: List[RecommendationResponse]
    explanation: str
    csv_path: str
