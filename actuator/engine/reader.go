package engine

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
	"strings"

	"actuator/models"
)

func LoadRecommendations(csvPath string) ([]models.Workload, error) {
	file, err := os.Open(csvPath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	header, err := reader.Read()
	if err != nil {
		return nil, fmt.Errorf("failed to read header from CSV: %w", err)
	}

	// Map header names to their column indices
	colIndex := make(map[string]int)
	for i, name := range header {
		colIndex[strings.TrimSpace(name)] = i
	}

	// Verify required columns are present
	idColIdx, kindColIdx, labelColIdx := -1, -1, -1

	if idx, ok := colIndex["workload_id"]; ok {
		idColIdx = idx
	} else {
		return nil, fmt.Errorf("missing required column 'workload_id' in CSV header")
	}

	if idx, ok := colIndex["kind"]; ok {
		kindColIdx = idx
	} else {
		return nil, fmt.Errorf("missing required column 'kind' in CSV header")
	}

	if idx, ok := colIndex["label"]; ok {
		labelColIdx = idx
	} else {
		return nil, fmt.Errorf("missing required column 'label' in CSV header")
	}

	var results []models.Workload
	for {
		record, err := reader.Read()
		if err != nil {
			if err.Error() == "EOF" { // Correctly check for EOF
				break
			}
			return nil, fmt.Errorf("error reading CSV record: %w", err)
		}

		if len(record) <= idColIdx || len(record) <= kindColIdx || len(record) <= labelColIdx {
			return nil, fmt.Errorf("CSV record has fewer columns than expected based on header. Record: %v", record)
		}

		workloadID := strings.TrimSpace(record[idColIdx])
		kind := strings.TrimSpace(record[kindColIdx])
		labelStr := strings.TrimSpace(record[labelColIdx])

		label, err := strconv.Atoi(labelStr)
		if err != nil {
			return nil, fmt.Errorf("failed to convert label '%s' to int for workload_id '%s': %w", labelStr, workloadID, err)
		}

		results = append(results, models.Workload{
			ID:    workloadID,
			Label: label,
			Kind:  kind,
		})
	}
	return results, nil
}
