import subprocess
import json
import re

class LogParser:
    def __init__(self):
        self.save_cleaned_recomendations()
    
    def organize_recomendations(self) -> dict:
        output = self.get_recomendations()
        result = {}
        for recomendation in output:
            cleaned_recomendation = self.clean_recomendation(recomendation)
            key = cleaned_recomendation[1:20]
            cleaned_recomendation = cleaned_recomendation.split(" -  ")[1]
            if not result.get(key,False):
                result[key] = []
                result[key].append(cleaned_recomendation)
            else:
                result[key].append(cleaned_recomendation)

        return json.dumps(result)
    
    def get_recomendations(self) -> str:

        out = subprocess.run(["./prompt/builder/filter_recomendations.sh"],capture_output=True,text=True)
        list_recomendations = out.stdout.split(sep="\n")
        return list_recomendations[0:len(list_recomendations) - 1]
    
    def remove_ansii_code(self,line:str) -> str:
        regex_ansi = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return regex_ansi.sub("",line)
    
    def remove_emojis(self,line:str) -> str:
        line = line.encode("ascii","ignore").decode("ascii")
        return line
    
    def clean_recomendation(self,line:str) -> str:
        recomendation = self.remove_ansii_code(line)
        recomendation = self.remove_emojis(recomendation)
        return recomendation
    
    def save_cleaned_recomendations(self):
        archive = self.organize_recomendations()
        with open("./prompt/builder/recomendations.json","w") as arch:
            arch.write(archive)
