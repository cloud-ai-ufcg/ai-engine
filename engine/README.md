# AI Engine

## Requisitos

- Python 3.12
- crewai
- google-generativeai

## Instalação

```bash
pip install -r requirements.txt
```

## Uso

```bash
python main.py --json-input monitor_outputs_fake.json
```

## Arquivos

- `main.py`: Script principal que lê o arquivo JSON e faz as predições.
- `agents.py`: Contém a função `label_workloads_with_gemini` que usa o CrewAI + Gemini para decidir os labels dos workloads.
- `monitor_outputs_fake.json`: Arquivo de exemplo com dados de monitoramento de workloads.
