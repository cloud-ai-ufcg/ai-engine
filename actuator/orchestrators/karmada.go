package orchestrators

import (
	"context"
	"fmt"
	"log"
	"strings"

	"actuator/models"

	"golang.org/x/text/cases"
	"golang.org/x/text/language"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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

	switch strings.ToLower(w.Kind) {
	case "deployment":
		return k.updateDeploymentLabel(w.ID, labelName, w.Kind)
	case "job":
		return k.updateJobLabel(w.ID, labelName, w.Kind)
	default:
		return fmt.Errorf("unsupported kind: %s", w.Kind)
	}
}

func (k *KarmadaOrchestrator) updateDeploymentLabel(name string, newLabel string, kind string) error {
	deploy, err := k.clientset.AppsV1().Deployments("default").Get(context.TODO(), name, metav1.GetOptions{})
	if err != nil {
		return err
	}

	currentLabel := deploy.Labels["cloud"]
	if currentLabel == newLabel {
		log.Printf("⚠️  %s %s already labeled %s", cases.Title(language.Und).String(strings.ToLower(kind)), name, newLabel)
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

	log.Printf("🔄 %s %s updated to %s", cases.Title(language.Und).String(strings.ToLower(kind)), name, newLabel)

	_, err = k.clientset.AppsV1().Deployments("default").Update(context.TODO(), deploy, metav1.UpdateOptions{})
	return err
}

func (k *KarmadaOrchestrator) updateJobLabel(name string, newLabel string, kind string) error {
	job, err := k.clientset.BatchV1().Jobs("default").Get(context.TODO(), name, metav1.GetOptions{})
	if err != nil {
		return err
	}

	currentLabel := job.Labels["cloud"]
	if currentLabel == newLabel {
		log.Printf("⚠️  %s %s already labeled %s", cases.Title(language.Und).String(strings.ToLower(kind)), name, newLabel)
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

	log.Printf("🔄 %s %s updated to %s", cases.Title(language.Und).String(strings.ToLower(kind)), name, newLabel)

	_, err = k.clientset.BatchV1().Jobs("default").Update(context.TODO(), job, metav1.UpdateOptions{})
	return err
}
