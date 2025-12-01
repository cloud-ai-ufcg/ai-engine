from pydantic import BaseModel, Field
from typing import Dict, List, Any, TypedDict


# Defines the shared state between nodes
class WorkflowState(TypedDict):

    workloads: List[Dict]
    cpu_votes: List[int]
    mem_votes: List[int]
    pending_votes: List[int]
    final_decisions: List[int]
    explanations: Dict


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


class WorkloadLabelOutput(BaseModel):
    """Structured output for workload labeling"""

    decisions: List[int] = Field(
        description="Array of decisions where 0=private, 1=public for each workload"
    )
    explanations: List[str] = Field(
        description="Array of explanations for each decision"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkloadLabelOutput":
        """Create instance from dictionary"""
        return cls(**data)

    def validate_output(self) -> bool:
        """Validate that decisions and explanations have matching lengths"""
        return len(self.decisions) == len(self.explanations)


class WorkloadRecommendation(BaseModel):
    """Structured output for workload recommendations"""

    batch_id: int | None = None
    workload_id: str
    kind: str
    origin_cluster: int
    destination_cluster: int
    reason: str  # Explanation for the decision
