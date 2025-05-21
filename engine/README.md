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

```bash
make run
```

## Arquivos

- `main.py`: Script principal que lê o arquivo JSON e faz as predições.
- `agents.py`: Contém a função `label_workloads_with_gemini` que usa o CrewAI + Gemini para decidir os labels dos workloads.
- `model-pipeline/`: Diretório com os modelos de machine learning.
- `util.py`: Contém funções utilitárias para leitura e escrita de arquivos.
- `monitor_outputs_fake.json`: Arquivo de exemplo com dados de monitoramento de workloads.
- `config.yaml`: Arquivo de configuração com as variáveis de ambiente.
- `requirements.txt`: Arquivo com as dependências do projeto.
- `README.md`: Arquivo com as instruções de uso do projeto.
- `Makefile`: Arquivo com as instruções de build e deploy do projeto.


