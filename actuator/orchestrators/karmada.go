package orchestrators

import (
	"context"
	"fmt"
	"log"
	"strings"

	"actuator/models"

	"golang.org/x/text/cases"
	"golang.org/x/text/language"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
)

type KarmadaOrchestrator struct {
	clientset *kubernetes.Clientset
}

func NewKarmadaOrchestrator() (*KarmadaOrchestrator, error) {
	config, err := clientcmd.BuildConfigFromFlags("", clientcmd.RecommendedHomeFile)
	if err != nil {
		return nil, err
	}

	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		return nil, err
	}

	return &KarmadaOrchestrator{
		clientset: clientset,
	}, nil
}

func parseNamespaceAndName(id string) (string, string, error) {
	parts := strings.Split(id, "/")
	if len(parts) != 2 {
		return "", "", fmt.Errorf("invalid workload ID format: %s (expected namespace/name)", id)
	}
	return parts[0], parts[1], nil
}


func (k *KarmadaOrchestrator) ApplyDecision(w models.Workload) error {
	labelValue := strings.TrimSpace(fmt.Sprintf("%d", w.Label))
	var labelName string

	switch labelValue {
	case "0":
		labelName = "private"
	case "1":
		labelName = "public"
	default:
		return fmt.Errorf("invalid label: %s", w.Label)
	}

	namespace, name, err := parseNamespaceAndName(w.ID)
	if err != nil {
		return err
	}

	switch strings.ToLower(w.Kind) {
	case "deployment":
		return k.updateDeploymentLabel(namespace, name, labelName, w.Kind)
	case "job":
		return k.updateJobLabel(namespace, name, labelName, w.Kind)
	default:
		return fmt.Errorf("unsupported kind: %s", w.Kind)
	}
}

func (k *KarmadaOrchestrator) updateDeploymentLabel(namespace, name, newLabel, kind string) error {
	deploy, err := k.clientset.AppsV1().Deployments(namespace).Get(context.TODO(), name, metav1.GetOptions{})
	if err != nil {
		return err
	}

	currentLabel := deploy.Labels["cloud"]
	if currentLabel == newLabel {
		log.Printf("⚠️  %s %s/%s already labeled %s", cases.Title(language.Und).String(strings.ToLower(kind)), namespace, name, newLabel)
		return nil
	}

	if deploy.Labels == nil {
		deploy.Labels = make(map[string]string)
	}
	deploy.Labels["cloud"] = newLabel

	if deploy.Annotations == nil {
		deploy.Annotations = make(map[string]string)
	}
	deploy.Annotations["clusterpropagationpolicy.karmada.io/name"] = "deploy-" + newLabel

	log.Printf("🔄 %s %s/%s updated to %s", cases.Title(language.Und).String(strings.ToLower(kind)), namespace, name, newLabel)

	_, err = k.clientset.AppsV1().Deployments(namespace).Update(context.TODO(), deploy, metav1.UpdateOptions{})
	return err
}

func (k *KarmadaOrchestrator) updateJobLabel(namespace, name, newLabel, kind string) error {
	job, err := k.clientset.BatchV1().Jobs(namespace).Get(context.TODO(), name, metav1.GetOptions{})
	if err != nil {
		return err
	}

	currentLabel := job.Labels["cloud"]
	if currentLabel == newLabel {
		log.Printf("⚠️  %s %s/%s already labeled %s", cases.Title(language.Und).String(strings.ToLower(kind)), namespace, name, newLabel)
		return nil
	}

	if job.Labels == nil {
		job.Labels = make(map[string]string)
	}
	job.Labels["cloud"] = newLabel

	if job.Annotations == nil {
		job.Annotations = make(map[string]string)
	}
	job.Annotations["clusterpropagationpolicy.karmada.io/name"] = "deploy-" + newLabel

	log.Printf("🔄 %s %s/%s updated to %s", cases.Title(language.Und).String(strings.ToLower(kind)), namespace, name, newLabel)

	_, err = k.clientset.BatchV1().Jobs(namespace).Update(context.TODO(), job, metav1.UpdateOptions{})
	return err
}
