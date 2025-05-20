from typing import List, Union
import os
import pandas as pd
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
    if not HAS_GENAI:
        logger.warning(
            "Modelo Gemini não disponível. Usando modelo tradicional como fallback."
        )
        return _label_workloads_with_heuristics(workloads)

    load_dotenv()

    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        logger.warning(
            "API Key para Gemini não encontrada. Usando modelo tradicional como fallback."
        )
        return _label_workloads_with_heuristics(workloads)

    if isinstance(workloads, list):
        df = pd.DataFrame(workloads)
    else:
        df = workloads

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
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-001')
        response = model.generate_content(prompt)

        text_response = response.text
        pattern = r'[01]+'
        matches = re.findall(pattern, text_response)

        if matches:
            longest_match = max(matches, key=len)
            if len(longest_match) == len(df):
                return [int(digit) for digit in longest_match]

        all_digits = re.findall(r'[01]', text_response)
        if len(all_digits) >= len(df):
            logger.info(f"Extraindo labels da resposta do Gemini: {text_response}")
            return [int(digit) for digit in all_digits[: len(df)]]

        logger.warning(
            f"Não foi possível extrair labels da resposta do Gemini: {text_response}"
        )
        return _label_workloads_with_heuristics(workloads)
    except Exception as e:
        logger.warning(f"Erro ao usar Gemini API: {e}")
        return _label_workloads_with_heuristics(workloads)


def _label_workloads_with_heuristics(
    workloads: Union[list, 'pd.DataFrame']
) -> List[int]:
    """
    Fallback: uses simple heuristics to decide labels when AI is not available.
    """
    if isinstance(workloads, list):
        df = pd.DataFrame(workloads)
    else:
        df = workloads.copy()

    def cpu_to_float(cpu):
        if isinstance(cpu, str) and cpu.endswith('m'):
            return float(cpu[:-1]) / 1000.0
        return float(cpu) if cpu else 0.0

    def mem_to_float(mem):
        if isinstance(mem, str) and mem.endswith('Mi'):
            return float(mem[:-2])
        return float(mem) if mem else 0.0

    # Extract and convert resources
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

    # Rules heuristics:
    # 1. If CPU > 0.5 or memory > 1024Mi: move to public
    # 2. If percent_pending > 20%: move to public
    try:
        percent_pending = df['percent_pending'].fillna(0)
    except:
        percent_pending = pd.Series([0] * len(df))

    # Combine rules to decide: 0=private, 1=public
    labels = ((cpu_values > 0.5) | (mem_values > 1024) | (percent_pending > 50)).astype(
        int
    )

    return labels.tolist()
