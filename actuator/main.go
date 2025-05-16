package main

import (
	"fmt"
	"log"

	"actuator/engine"
	"actuator/orchestrators"
)

func main() {
	workloads, err := engine.LoadRecommendations("recommendations.csv")
	if err != nil {
		log.Fatalf("Error reading CSV: %v", err)
	}

	orc, err := orchestrators.NewKarmadaOrchestrator()
	if err != nil {
		log.Fatalf("Error creating orchestrator: %v", err)
	}

	for _, w := range workloads {
		if err := orc.ApplyDecision(w); err != nil {
			fmt.Printf("Error applying workload %s: %v\n", w.ID, err)
		}
	}
}
