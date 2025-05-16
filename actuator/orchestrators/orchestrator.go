package orchestrators

import "actuator/models"

type Orchestrator interface {
	ApplyDecision(w models.Workload) error
}
