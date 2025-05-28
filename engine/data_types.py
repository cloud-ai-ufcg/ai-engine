from pydantic import BaseModel
from typing import Dict, List, Any


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
