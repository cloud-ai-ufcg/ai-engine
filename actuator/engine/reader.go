package engine

import (
	"encoding/csv"
	"os"
	"strconv"

	"actuator/models"
)

func LoadRecommendations(csvPath string) ([]models.Workload, error) {
	file, err := os.Open(csvPath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	_, _ = reader.Read()

	var results []models.Workload
	for {
		record, err := reader.Read()
		if err != nil {
			break
		}
		label, _ := strconv.Atoi(record[1])
		results = append(results, models.Workload{
			ID:    record[0],
			Label: label,
			Kind:  record[2],
		})
	}
	return results, nil
}
