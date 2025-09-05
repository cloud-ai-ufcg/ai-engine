import os
from dotenv import load_dotenv
from langsmith import Client

# carrega o .env
load_dotenv()

# inicializa o cliente
client = Client(
    api_key=os.getenv("LANGCHAIN_API_KEY"),
    api_url=os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
)

runs = client.list_runs(project_name="MultiAgent-engine")

for run in runs:
    print("Run ID:", run.id)
    print("Name:", run.name)
    print("Start:", run.start_time)
    print("End:", run.end_time)
    print("Error:", run.error)
    print("Status:", run.status)
    print("Extra info:", run.extra)
    print("Reference example:", run.reference_example_id)
    print(run.inputs)     
    print(run.outputs)    
    print(run.metadata)   

    print("-" * 40)
