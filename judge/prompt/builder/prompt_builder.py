from .log_parser import LogParser
import json
import yaml # type: ignore
from datetime import datetime
import re
class PromptBuilder:
    def __init__(self):
        self.log_parser = LogParser()
        self.prompt = ""

    
    def build_prompt(self) -> str:

        metrics = self.open_metrics()
        recomendations = self.open_recomendations()
        all_prompts = []
        for key in recomendations.keys():

            current_recomendation = recomendations[key]
            current_recomendation = self.recomendations_to_yaml(current_recomendation)
            
            nearest_key = self.get_nearest_key(key,metrics.keys())
            current_metrics = metrics[nearest_key]

            cluster_state = {
                             "state_private":current_metrics["cluster_info"][0],
                             "state_public":current_metrics["cluster_info"][1]
                             }
            del current_metrics["cluster_info"]

            current_metrics = self.clean_metrics(current_metrics)
            current_metrics = self.dict_to_yaml(current_metrics)
            cluster_state = self.dict_to_yaml(cluster_state)

            criteria_evaluation = "You must prioritize the cost-benefit ratio between performance and migration costs."
            all_prompts.append(self.make_prompt(current_metrics,cluster_state,current_recomendation,criteria_evaluation))

        return all_prompts

    def get_nearest_key(self,key:str,metric_keys:list) -> str:
        copy_metric_keys = list(metric_keys)
        for i in range(len(copy_metric_keys)):
            copy_metric_keys[i] = self.transform_str_to_date(copy_metric_keys[i])
        
        copy_metric_keys.sort()
        key =  self.transform_str_to_date(key)
        for i in range(len(copy_metric_keys)):
            if copy_metric_keys[i] > key:
                return f"{copy_metric_keys[i - 1]}"
    
    def clean_metrics(self,metrics:dict) -> dict:
        metrics.pop("interval_duration")
        workloads = metrics["workloads"]
        for i in range(len(workloads)):
            workloads[i].pop("kind")
            workloads[i].pop("resources")
            workloads[i].pop("pods_total")

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
        metrics["workloads"] = workloads
        return metrics
            
    def recomendations_to_yaml(self,recomendations:list) -> str:
        formated_recomendations = {
            "recomendations":[]
        }
        for recomendation in recomendations:
            dict_recomendation = self.recomendation_to_dict(recomendation)
            formated_recomendations["recomendations"].append(dict_recomendation)
        
        return yaml.dump(formated_recomendations,sort_keys=False)
        
    
    def recomendation_to_dict(self,recomendation:str) -> dict:
        id_workload = int(re.sub(r'\D',"",recomendation.split("deployment")[0]))
        start_private = recomendation.find("private")
        start_public = recomendation.find("public")
        cluster_destination = "public"

        if start_public == -1:
            cluster_destination = "private"
        elif start_public > 0 and start_private > 0:
            dict_aux = {start_public:"public",start_private:"private"}
            cluster_destination = dict_aux[min(start_private,start_public)]

        return {"id_workload":id_workload,"migrate_to":cluster_destination}

        


    def transform_str_to_date(self, date:str):
        format_date = "%Y-%m-%d %H:%M:%S"
        return datetime.strptime(date,format_date)
        
    def dict_to_yaml(self,metrics:dict) -> str:
        return yaml.dump(metrics,sort_keys=False)

    def open_metrics(self) -> dict:
        dados = {}
        with open("./prompt/builder/metrics.json","r",encoding="utf-8") as data:
            dados = json.load(data)
        return dados
    
    def open_recomendations(self) -> dict:
        dados = {}
        with open("./prompt/builder/recomendations.json","r",encoding="utf-8") as data:
            dados = json.load(data)
        return dados
    
    def make_prompt(self,workloads:str,cluster_state:str,migrations:str,criteria:str) -> str:
        WORKLOADS_YAML = workloads
        CLUSTER_STATE_YAML = cluster_state
        MIGRATIONS_YAML_OR_TEXT = migrations
        CRITERIA_TEXT = criteria

        prompt = f"""
        # ROLE
        You are a Senior Kubernetes Reliability Engineer (SRE) specializing in capacity planning and multi-cluster workload migration.
        # OBJECTIVE
        Your task is to audit and validate a set of migration decisions made by another system. You must determine if each proposed migration is viable and efficient based on resource constraints (CPU/Memory) and specific evaluation criteria.
        # INPUT DATA
        I will provide the following data in YAML format for efficiency:

        <workloads_data>
        {WORKLOADS_YAML}
        </workloads_data>

        <cluster_state>
        {CLUSTER_STATE_YAML}
        </cluster_state>

        <proposed_migrations>
        {MIGRATIONS_YAML_OR_TEXT}
        </proposed_migrations>

        <evaluation_criteria>
        {CRITERIA_TEXT}
        </evaluation_criteria>

        # INSTRUCTIONS
        1. Analyze the resource requirements (CPU/Memory) of each workload in `<workloads_data>`.
        2. Compare them against the available capacity in the destination cluster defined in `<cluster_state>`.
        3. Apply the rules found in `<evaluation_criteria>`.
        4. Validate if the move suggested in `<proposed_migrations>` is a "GO" or "NO-GO".
        5. You must priorize the benefit-cost between performance and cost of migration

        # OUTPUT FORMAT
        Return **ONLY** a valid JSON object. Do not include markdown formatting (like ```json) or conversational text. The JSON must follow this exact schema:

        {{
        "validated_recommendations": [
            {{
            "workload_id": "string (The ID of the workload)",
            "destination_cluster": "string (The target cluster name)",
            "llm_score": number (0.0 to 1.0, where 1.0 is a perfect decision),
            "llm_status": "string ('APPROVED' or 'REJECTED')",
            "llm_reasoning": "string (Concise technical explanation of why the decision is good or bad, citing specific resource metrics)",
            }}
        ]
        }}

        You should not truncate the output!
        """
        return prompt