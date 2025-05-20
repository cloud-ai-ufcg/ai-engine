from typing import List, Union
import os
import pandas as pd
import re
from dotenv import load_dotenv
from util import get_logger, load_config

logger = get_logger("agents")

try:
    import google.generativeai as genai

    HAS_GENAI = True
except ImportError:
    genai = None
    HAS_GENAI = False


def label_workloads_with_gemini(workloads: Union[list, 'pd.DataFrame']) -> List[int]:
    """
    Uses Gemini API to decide workload labels.
    Each label: 0 = private, 1 = public.
    Args:
        workloads: list of dicts or DataFrame with workload fields.
    Returns:
        List of labels (0 or 1) in the same order.
    """
    logger.info("Starting workload analysis for migration decision")
    
    if not HAS_GENAI:
        logger.warning(
            "Gemini model not available. Using traditional model as fallback."
        )
        return _label_workloads_with_heuristics(workloads)

    load_dotenv()

    config = load_config()
    api_key = os.environ.get('GOOGLE_API_KEY') or config.get('api-key', {}).get('google')
    if not api_key:
        logger.warning(
            "API Key for Gemini not found. Using traditional model as fallback."
        )
        return _label_workloads_with_heuristics(workloads)

    if isinstance(workloads, list):
        df = pd.DataFrame(workloads)
    else:
        df = workloads
        
    prompt = (
        "You are a Kubernetes cluster orchestrator. For each workload, decide whether it should stay in the 'private' cluster (0) or migrate to the 'public' cluster (1).\n"
        "Decision rules:\n"
        "- Workloads with high demand (high CPU or memory usage) should go to the stronger cluster (public).\n"
        "- If percent_pending is high, consider moving to the public cluster.\n"
        "- Consider the cluster_load to avoid overloading the destination cluster.\n"
        "ONLY respond with a single line containing 0s and 1s without separation, where each digit represents the recommended cluster for a workload (0=private, 1=public).\n\n"
        "Workloads: " + df.to_json(orient='records', indent=2)
    )

    logger.info("Sending request to Gemini model")
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-001')
        response = model.generate_content(prompt)

        text_response = response.text
        logger.debug(f"Complete Gemini response: {text_response}")
        
        pattern = r'[01]+'
        matches = re.findall(pattern, text_response)

        if matches:
            longest_match = max(matches, key=len)
            if len(longest_match) == len(df):
                logger.info(f"Migration pattern identified: {longest_match}")
                labels = [int(digit) for digit in longest_match]
                
                # Log the decision for each workload
                for idx, (label, workload) in enumerate(zip(labels, df.iterrows())):
                    workload_id = workload[1].get('workload_id', f'workload-{idx}')
                    kind = workload[1].get('kind', 'unknown')
                    destination = "public" if label == 1 else "private"
                    logger.info(f"Decision for {workload_id} ({kind}): Cluster {destination}")
                    
                return labels

        all_digits = re.findall(r'[01]', text_response)
        if len(all_digits) >= len(df):
            logger.info(f"Extracting labels from Gemini response: {text_response}")
            labels = [int(digit) for digit in all_digits[: len(df)]]
            
            # Log the decision for each workload
            for idx, (label, workload) in enumerate(zip(labels, df.iterrows())):
                workload_id = workload[1].get('workload_id', f'workload-{idx}')
                kind = workload[1].get('kind', 'unknown')
                destination = "public" if label == 1 else "private"
                logger.info(f"Decision for {workload_id} ({kind}): Cluster {destination}")
                
            return labels

        logger.warning(
            f"Could not extract labels from Gemini response: {text_response}"
        )
        return _label_workloads_with_heuristics(workloads)
    except Exception as e:
        logger.warning(f"Error using Gemini API: {e}")
        return _label_workloads_with_heuristics(workloads)


def _label_workloads_with_heuristics(
    workloads: Union[list, 'pd.DataFrame']
) -> List[int]:
    """
    Fallback: uses simple heuristics to decide labels when AI is not available.
    """
    logger.info("Using heuristics as fallback for migration decision")
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
        # Alternative if the format is different
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
