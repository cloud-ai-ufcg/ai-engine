import os
import sys
from pathlib import Path
from openai import OpenAI
import dotenv
from fastapi import FastAPI
from fastapi import Body
 

from prompt.builder.prompt_builder import PromptBuilder 

dotenv.load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

app = FastAPI()        

def call_llm(prompt:str) -> str:
    try:
        response = client.chat.completions.create(model="gpt-4o",messages=[
            {"role":"user","content":prompt}])
        
        return response.choices[0].message.content
    except Exception as e:
        return print(e)

def send_prompt() -> str:
    prompts = PromptBuilder().build_prompt()
    for i in range(len(prompts)):
        output = call_llm(prompts[i])
        with open(f"output_llm/output{i + 1}.json","w") as f:
            f.write(output)
        with open(f"input_llm/input{i + 1}.txt","w") as f:
            f.write(prompts[i])
            
send_prompt()
 