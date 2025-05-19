"""
Agente CrewAI + Gemini para decisão de labels de workloads.
Requer as libs:
  pip install crewai google-generativeai
E a variável de ambiente GOOGLE_API_KEY configurada.
"""

from typing import List, Union
import os
import pandas as pd
from crewai import Crew, Task, Agent
import re
from dotenv import load_dotenv
from util import get_logger

logger = get_logger("agents")

try:
    import google.generativeai as genai

    HAS_GENAI = True
except ImportError:
    genai = None
    HAS_GENAI = False


def label_workloads_with_gemini(workloads: Union[list, 'pd.DataFrame']) -> List[int]:
    """
    Usa Gemini API para decidir labels dos workloads.
    Cada label: 0 = private, 1 = public.
    Args:
        workloads: lista de dicts ou DataFrame com os campos do workload.
    Returns:
        Lista de labels (0 ou 1) na mesma ordem.
    """
    # Verifica se o Gemini API está disponível
    if not HAS_GENAI:
        logger.warning(
            "Modelo Gemini não disponível. Usando modelo tradicional como fallback."
        )
        return _label_workloads_with_ml(workloads)

    # Verifica se a API key está configurada

    # Carrega as variáveis do arquivo .env
    load_dotenv()

    # Obtém a API key do ambiente
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        logger.warning(
            "API Key para Gemini não encontrada. Usando modelo tradicional como fallback."
        )
        return _label_workloads_with_ml(workloads)

    # Converte para DataFrame se necessário
    if isinstance(workloads, list):
        df = pd.DataFrame(workloads)
    else:
        df = workloads

    # Prepare prompt para o Gemini
    prompt = (
        "Você é um orquestrador de clusters Kubernetes. Para cada workload, decida se deve ficar no cluster 'private' (0) ou migrar para 'public' (1).\n"
        "Regras de decisão:\n"
        "- Workloads com alta demanda (alto use de CPU ou memória) devem ir para o cluster mais forte (public).\n"
        "- Se o percent_pending é alto, considere mover para o cluster público.\n"
        "- Considere o cluster_load para evitar sobrecarregar o cluster destino.\n"
        "Responda APENAS com uma linha contendo 0s e 1s sem separação, onde cada digito representa o cluster recomendado para um workload (0=private, 1=public).\n\n"
        "Workloads: " + df.to_json(orient='records', indent=2)
    )

    try:
        # Configure e use o Gemini diretamente
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-001')
        response = model.generate_content(prompt)

        # Extrair a sequência de 0s e 1s da resposta
        text_response = response.text
        # Procura por sequências de 0s e 1s na resposta
        pattern = r'[01]+'
        matches = re.findall(pattern, text_response)

        # Considere a sequência mais longa como a resposta
        if matches:
            longest_match = max(matches, key=len)
            if len(longest_match) == len(df):
                return [int(digit) for digit in longest_match]

        # Se não conseguiu extrair corretamente, tenta uma abordagem mais permissiva
        all_digits = re.findall(r'[01]', text_response)
        if len(all_digits) >= len(df):
            logger.info(f"Extraindo labels da resposta do Gemini: {text_response}")
            return [int(digit) for digit in all_digits[: len(df)]]

        logger.warning(
            f"Não foi possível extrair labels da resposta do Gemini: {text_response}"
        )
        return _label_workloads_with_ml(workloads)
    except Exception as e:
        logger.warning(f"Erro ao usar Gemini API: {e}")
        return _label_workloads_with_ml(workloads)


def _label_workloads_with_ml(workloads: Union[list, 'pd.DataFrame']) -> List[int]:
    """
    Fallback: usa heurísticas simples para decidir labels quando IA não está disponível.
    """

    # Converte para DataFrame se for lista
    if isinstance(workloads, list):
        df = pd.DataFrame(workloads)
    else:
        df = workloads.copy()

    # Função para converter valores de CPU/memória para float
    def cpu_to_float(cpu):
        if isinstance(cpu, str) and cpu.endswith('m'):
            return float(cpu[:-1]) / 1000.0
        return float(cpu) if cpu else 0.0

    def mem_to_float(mem):
        if isinstance(mem, str) and mem.endswith('Mi'):
            return float(mem[:-2])
        return float(mem) if mem else 0.0

    # Extrair e converter recursos
    try:
        cpu_values = df['resources'].apply(
            lambda x: cpu_to_float(x.get('cpu', 0)) if isinstance(x, dict) else 0.0
        )
        mem_values = df['resources'].apply(
            lambda x: mem_to_float(x.get('memory', 0)) if isinstance(x, dict) else 0.0
        )
    except:
        # Alternativa se o formato for diferente
        cpu_values = df.get('resources.cpu', df.get('cpu', 0)).apply(cpu_to_float)
        mem_values = df.get('resources.memory', df.get('memory', 0)).apply(mem_to_float)

    # Regras heurísticas simples:
    # 1. Se CPU > 0.5 ou memória > 1024Mi: move para public
    # 2. Se percent_pending > 20%: move para public
    try:
        percent_pending = df['percent_pending'].fillna(0)
    except:
        percent_pending = pd.Series([0] * len(df))

    # Combina regras para decidir: 0=private, 1=public
    labels = ((cpu_values > 0.5) | (mem_values > 1024) | (percent_pending > 50)).astype(
        int
    )

    return labels.tolist()
