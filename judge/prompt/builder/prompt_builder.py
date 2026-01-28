import json
import yaml # type: ignore
from datetime import datetime
import re
class PromptBuilder:
    def __init__(self,recomendations:list,metrics:dict,cluster_info:list) -> list:
        self.recomendations = recomendations
        self.metrics = metrics
        self.cluster_info = cluster_info

    
    def build_prompt(self) -> str:
        
        formated_recomendations = []
        current_metrics = self.metrics
        for recomendation in self.recomendations:
            recomendation_formated = self.format_recomendation(recomendation,current_metrics)
            formated_recomendations.append(recomendation_formated)

        cluster_state = self.cluster_info
        
        current_metrics = self.clean_metrics(current_metrics)
        current_metrics = self.to_yaml(current_metrics)
        cluster_state = self.to_yaml(cluster_state)
        formated_recomendations = self.to_yaml(formated_recomendations)

        prompt = self.make_prompt(current_metrics,cluster_state,formated_recomendations)

        return prompt
    
    def clean_metrics(self,metrics:dict) -> dict:
        workloads = metrics
        for i in range(len(workloads)):
            workloads[i].pop("kind")
            workloads[i].pop("resources")
            workloads[i].pop("pods_total")
            workloads[i].pop("percent_pending")
            workloads[i].pop("timestamp")
            workloads[i].pop("cluster_load")
            workloads[i].pop("cluster_cpu_capacity")
            workloads[i].pop("cluster_memory_capacity")




        default_configs = {
            "workloads_configs_default":{
                "kind": "deployment",
                "resources": {
                    "cpu": "2000m",
                    "memory": "4096Mi"
                },
                "pods_total": 2
            }
        }
        workloads.insert(0,default_configs)
        metrics = workloads
        return metrics
    
    def format_recomendation(self,workload:dict,metrics:dict) -> dict:
        if workload["label"] == -1: return {}
        labels = {"public":"private","private":"public"}
        workload_aux = workload.copy()
        del workload_aux["kind"]
        workload_aux.pop("reason")

        for workload_metrics in metrics:
            if workload_metrics["workload_id"] != workload_aux["workload_id"]:
                continue

            workload_aux["migrate_to"] = labels[workload_metrics["cluster_label"]]
        
        del workload_aux["label"]

        return workload_aux 

    def to_yaml(self,recomendations:list) -> str:
        return yaml.dump(recomendations,sort_keys=False,indent=4)
        

    def transform_str_to_date(self, date:str):
        format_date = "%Y-%m-%d %H:%M:%S"
        return datetime.strptime(date,format_date)
         
    def make_prompt(self,workloads:str,cluster_state:str,migrations:str) -> str:
        WORKLOADS_YAML = workloads
        CLUSTER_STATE_YAML = cluster_state
        MIGRATIONS_YAML = migrations

        prompt = f"""
       
        You are a Senior Kubernetes Reliability Engineer (SRE) specializing in cmulti-cluster workload migration.
        Your task is to audit and validate a set of migration decisions made by another system. You must determine if each proposed migration is viable based on analysis of resources such as CPU/memory of the clusters.
        I will provide the following data:

        <workloads_data>
            {WORKLOADS_YAML}
        </workloads_data>

        <cluster_state>
            {CLUSTER_STATE_YAML}
        </cluster_state>

        <proposed_migrations>
            {MIGRATIONS_YAML}
        </proposed_migrations>

        <evaluation_criteria>
            You should prioritize cost-effectiveness between performance and migration cost. Do not approve migrations that worsen cluster performance even if they offer good value for money
        </evaluation_criteria>

        # INSTRUCTIONS
        1. Analyze the resource requirements (CPU/Memory) of each workload in `<workloads_data>`.
        2. Compare them against the available capacity in the destination cluster defined in `<cluster_state>`.
        3. Apply the rules found in `<evaluation_criteria>`.
        4. Validate if the move suggested in `<proposed_migrations>` is good or bad migration.

        You must return **ONLY** a valid JSON object.
        Do not include markdown formatting (like ```json) or conversational text. The JSON must follow this exact schema:

        {{
        "validated_recommendations": [
            {{
            "workload_id": "string (The ID of the workload)",
            "destination_cluster": "cluster to which the recommendation wanna transfer workload will be transferred. int (return 0 if it will be transfered to private cluster, return 0 if it will be transfered to public cluster)",
            "origin_cluster":"cluster where the workload is located int (0 = private, 1 = public)"
            "llm_status": "Status of recomendation, int (0 = REJECTED, 1 = APPROVED)",
            "llm_reasoning": "string (Concise technical explanation of why the decision is good or bad, citing specific resource metrics,remember, you must be concise on your answer)",
            }}
        ]
        }}

        "The recommendations must be listed by priority, with the most important migration appearing first."
        """
        return prompt
