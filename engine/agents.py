from typing import List, Union, Dict, Any, Tuple, Optional
import os
import pandas as pd
import re
import json
from .ai_config import get_model_config, get_prompt, PROMPTS
from dotenv import load_dotenv
from .util import get_logger, load_config, log_token_usage
from .langgraph_agents.graph.migration_graph import build_graph


logger = get_logger("agents")

try:
    import google.generativeai as genai

    HAS_GENAI = True
except ImportError:
    genai = None
    HAS_GENAI = False

# Try to import Groq client for Llama models
try:
    from groq import Groq  # Groq Python client
    HAS_GROQ = True
except ImportError:
    Groq = None
    HAS_GROQ = False

# global variables
REQUEST_COUNTER = 0
TOKEN_TOTALS = {"input": 0, "output": 0, "total": 0}

# ---------------------------------------------------------------------------
# Gemini API Helper Functions
# ---------------------------------------------------------------------------

def _get_gemini_api_key(config: Dict[str, Any]) -> Optional[str]:
    """Retrieve Gemini API key from config or environment."""
    return config.get("api-key", {}).get("google") or os.environ.get("GOOGLE_API_KEY")


def _setup_gemini_model(api_key: str, config: Dict[str, Any]):
    """Configure and return Gemini model with config."""
    model_config = config.get("ai", {}).get("models", {}).get("gemini", {})
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_config["model_name"])
    return model, model_config


def _normalize_workloads_to_dataframe(workloads: Union[list, "pd.DataFrame"]) -> "pd.DataFrame":
    """Convert workloads input to DataFrame format."""
    return pd.DataFrame(workloads) if isinstance(workloads, list) else workloads


def _extract_json_from_response(text_response: str) -> Dict[str, Any]:
    """Extract and parse JSON from model response."""
    json_match = re.search(r"\{[\s\S]*\}", text_response)
    if not json_match:
        raise ValueError("No JSON object found in model response")
    
    json_str = json_match.group(0)
    return json.loads(json_str)


def _validate_and_extract_decisions(response_data: Dict[str, Any]) -> Tuple[List[int], List[str]]:
    """Validate response using schema and extract decisions/explanations."""
    prompt_config = PROMPTS.get("label_workloads", {})
    output_schema = prompt_config.get("output_schema")
    
    if output_schema:
        try:
            output = output_schema.from_dict(response_data)
            if not output.validate_output():
                logger.warning(
                    "Output validation failed: decisions and explanations have different lengths"
                )
            return output.decisions, output.explanations
        except Exception as e:
            logger.error(f"Failed to validate response with schema: {e}")
    
    # Fallback to direct extraction
    decisions = response_data.get("decisions", [])
    explanations = response_data.get("explanations", [])
    return decisions, explanations


def _create_explanation_output(labels: List[int], explanations: List[str], df: "pd.DataFrame") -> Dict[str, Any]:
    """Create structured explanation output and log decisions."""
    explanation_output = {
        "explanation": "Migration decisions based on AI analysis of workload characteristics",
        "workload_explanations": [],
    }
    
    for idx, (label, workload) in enumerate(zip(labels, df.iterrows())):
        workload_id = workload[1].get("workload_id", f"workload-{idx}")
        kind = workload[1].get("kind", "unknown")
        destination = "public" if label == 1 else "private"
        
        # Log the decision
        logger.info(f"Decision for {workload_id} ({kind}): Cluster {destination}")
        
        # Add explanation for this workload
        explanation = (
            explanations[idx]
            if idx < len(explanations)
            else f"Workload {workload_id} recommended for {destination} cluster based on resource requirements"
        )
        explanation_output["workload_explanations"].append(explanation)
    
    return explanation_output


def label_workloads_with_gemini(
    workloads: Union[list, "pd.DataFrame"],
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Uses Gemini API to decide workload labels with explanations.
    Each label: 0 = private, 1 = public.
    Args:
        workloads: list of dicts or DataFrame with workload fields.
    Returns:
        Tuple containing:
        - List of labels (0 or 1) in the same order
        - Dictionary with explanations for each workload
    """
    global REQUEST_COUNTER, TOKEN_TOTALS

    logger.info("Starting workload analysis for migration decision")

    # Dependency validation - early return if not available
    if not HAS_GENAI:
        logger.warning("Gemini model not available. Using traditional model as fallback.")
        return _label_workloads_with_heuristics(workloads)

    # API key setup - load config once
    load_dotenv()
    config = load_config()
    api_key = _get_gemini_api_key(config)
    if not api_key:
        logger.warning("API Key for Gemini not found. Using traditional model as fallback.")
        return _label_workloads_with_heuristics(workloads)

    # Data preparation
    df = _normalize_workloads_to_dataframe(workloads)
    prompt = get_prompt("label_workloads", workloads_json=df.to_json(orient="records", indent=2))
    
    logger.info("Sending request to Gemini model")

    try:
        # Model setup and API call
        model, model_config = _setup_gemini_model(api_key, config)
        logger.info(f"CONFIG: Using Gemini model: {model_config['model_name']}")
        logger.info(f"CONFIG: Using Gemini generation config: {model_config['generation_config']}")
        
        response = model.generate_content(prompt, generation_config=model_config["generation_config"])
        
        if hasattr(response, "prompt_feedback") and response.prompt_feedback:
            if getattr(response.prompt_feedback, "block_reason", None) == "MAX_TOKENS":
                logger.warning("Gemini model response was truncated due to max_output_tokens limit.")

        REQUEST_COUNTER += 1
        logger.info(f"Gemini API request count: {REQUEST_COUNTER}")

        input_tokens = model.count_tokens(prompt).total_tokens
        output_tokens = model.count_tokens(response.text).total_tokens

        TOKEN_TOTALS["input"] += input_tokens
        TOKEN_TOTALS["output"] += output_tokens
        TOKEN_TOTALS["total"] += input_tokens + output_tokens

        logger.info(
            f"Token Usage | "
            f"Input: {TOKEN_TOTALS['input']} | "
            f"Output: {TOKEN_TOTALS['output']} | "
            f"Total: {TOKEN_TOTALS['total']}"
        )

        # Response processing
        text_response = response.text
        logger.debug(f"Complete Gemini response: {text_response}")

        # Parse and validate response
        response_data = _extract_json_from_response(text_response)
        decisions, explanations = _validate_and_extract_decisions(response_data)

        # Validate decision count - handle mismatches gracefully
        if len(decisions) != len(df):
            logger.warning(f"Decision count mismatch: expected {len(df)}, got {len(decisions)}")
            
            # Handle mismatch by adjusting decisions list
            if len(decisions) > len(df):
                # Too many decisions - truncate
                decisions = decisions[:len(df)]
                explanations = explanations[:len(df)] if len(explanations) > len(df) else explanations
                logger.info(f"Truncated decisions to match {len(df)} workloads")
            else:
                # Too few decisions - pad with original cluster labels (no migration)
                missing_count = len(df) - len(decisions)
                
                # Get original cluster labels for missing decisions
                for i in range(len(decisions), len(df)):
                    workload_row = df.iloc[i]
                    original_cluster = workload_row.get('cluster_label', 'private')
                    # Convert cluster label to decision: private=0, public=1
                    original_decision = 0 if original_cluster == 'private' else 1
                    decisions.append(original_decision)
                    explanations.append(f"Maintaining original cluster ({original_cluster}) due to missing AI decision")
                
                logger.info(f"Padded {missing_count} missing decisions with original cluster assignments (no migration)")

        # Convert to integers and create output
        labels = [int(decision) for decision in decisions]
        logger.info("Migration decisions extracted from JSON response")
        
        explanation_output = _create_explanation_output(labels, explanations, df)
        return labels, explanation_output

    except Exception as e:
        logger.warning(f"Error parsing JSON response: {e}")

        # Fallback: try to extract just the decisions if JSON parsing failed
        pattern = r"[01]+"
        matches = re.findall(pattern, text_response)

        if matches:
            longest_match = max(matches, key=len)
            if len(longest_match) == len(df):
                logger.info(f"Migration pattern identified: {longest_match}")
                labels = [int(digit) for digit in longest_match]

                # Create a basic explanation output
                explanation_output = {
                    "explanation": "Migration decisions based on resource usage patterns",
                    "workload_explanations": [],
                }

                # Log the decision for each workload
                for idx, (label, workload) in enumerate(zip(labels, df.iterrows())):
                    workload_id = workload[1].get("workload_id", f"workload-{idx}")
                    kind = workload[1].get("kind", "unknown")
                    destination = "public" if label == 1 else "private"
                    logger.info(
                        f"Decision for {workload_id} ({kind}): Cluster {destination}"
                    )

                    # Add a generic explanation
                    if label == 1:
                        explanation = f"Workload {workload_id} ({kind}) recommended for public cluster due to high resource requirements"
                    else:
                        explanation = f"Workload {workload_id} ({kind}) recommended to stay in private cluster due to lower resource requirements"
                    explanation_output["workload_explanations"].append(explanation)

                return labels, explanation_output

        all_digits = re.findall(r"[01]", text_response)
        if len(all_digits) >= len(df):
            logger.info(f"Extracting labels from Gemini response: {text_response}")
            labels = [int(digit) for digit in all_digits[: len(df)]]

            # Create a basic explanation output
            explanation_output = {
                "explanation": "Migration decisions based on resource usage patterns",
                "workload_explanations": [],
            }

            # Log the decision for each workload
            for idx, (label, workload) in enumerate(zip(labels, df.iterrows())):
                workload_id = workload[1].get("workload_id", f"workload-{idx}")
                kind = workload[1].get("kind", "unknown")
                destination = "public" if label == 1 else "private"
                logger.info(
                    f"Decision for {workload_id} ({kind}): Cluster {destination}"
                )

                # Add a generic explanation
                if label == 1:
                    explanation = f"Workload {workload_id} ({kind}) recommended for public cluster due to high resource requirements"
                else:
                    explanation = f"Workload {workload_id} ({kind}) recommended to stay in private cluster due to lower resource requirements"
                explanation_output["workload_explanations"].append(explanation)

            return labels, explanation_output

        logger.warning(
            f"Could not extract labels from Gemini response: {text_response}"
        )
        labels = _label_workloads_with_heuristics(workloads)

        # Create a fallback explanation output
        explanation_output = {
            "explanation": "Migration decisions based on heuristic rules (fallback)",
            "workload_explanations": [],
        }

        # Add generic explanations
        for idx, (label, workload) in enumerate(zip(labels, df.iterrows())):
            workload_id = workload[1].get("workload_id", f"workload-{idx}")
            kind = workload[1].get("kind", "unknown")

            if label == 1:
                explanation = f"Workload {workload_id} ({kind}) recommended for public cluster due to high resource requirements"
            else:
                explanation = f"Workload {workload_id} ({kind}) recommended to stay in private cluster due to lower resource requirements"
            explanation_output["workload_explanations"].append(explanation)

        return labels, explanation_output
    except Exception as e:
        logger.warning(f"Error using Gemini API: {e}")
        labels = _label_workloads_with_heuristics(workloads)

        # Create a fallback explanation output
        explanation_output = {
            "explanation": f"Migration decisions based on heuristic rules due to API error: {str(e)}",
            "workload_explanations": [],
        }

        # Add generic explanations
        for idx, (label, workload) in enumerate(zip(labels, df.iterrows())):
            workload_id = workload[1].get("workload_id", f"workload-{idx}")
            kind = workload[1].get("kind", "unknown")

            if label == 1:
                explanation = f"Workload {workload_id} ({kind}) recommended for public cluster due to high resource requirements"
            else:
                explanation = f"Workload {workload_id} ({kind}) recommended to stay in private cluster due to lower resource requirements"
            explanation_output["workload_explanations"].append(explanation)

        return labels, explanation_output


# ---------------------------------------------------------------------------
# Groq / Llama implementation
# ---------------------------------------------------------------------------


def label_workloads_with_llama(
    workloads: Union[list, "pd.DataFrame"],
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Uses Groq's Llama model to decide workload labels with explanations.
    Each label: 0 = private, 1 = public.
    """
    logger.info("Starting workload analysis with Groq Llama model")

    if not HAS_GROQ:
        logger.warning("Groq client not available. Falling back to heuristics.")
        return _label_workloads_with_heuristics(workloads), {
            "explanation": "Used heuristic rules because Groq client is missing.",
            "workload_explanations": [],
        }

    load_dotenv()
    config = load_config()
    api_key = config.get("api-key", {}).get("groq") or os.environ.get("GROQ_API_KEY")

    if not api_key:
        logger.warning("GROQ_API_KEY missing. Falling back to heuristics.")
        return _label_workloads_with_heuristics(workloads), {
            "explanation": "Used heuristic rules due to missing Groq API key.",
            "workload_explanations": [],
        }

    # Ensure DataFrame
    df = pd.DataFrame(workloads) if isinstance(workloads, list) else workloads

    prompt = get_prompt(
        "label_workloads", workloads_json=df.to_json(orient="records", indent=2)
    )

    try:
        model_cfg = get_model_config("llama")
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=model_cfg.get("model_name", "llama-3.1-8b-instant"),
            messages=[{"role": "user", "content": prompt}],
            temperature=model_cfg.get("generation_config", {}).get("temperature", 0.1),
            max_tokens=model_cfg.get("generation_config", {}).get(
                "max_output_tokens", 8000
            ),
        )
        text_response = completion.choices[0].message.content  # type: ignore[attr-defined]
        logger.debug(f"Complete Groq response: {text_response}")

        json_match = re.search(r"\{[\s\S]*\}", text_response)
        if not json_match:
            raise ValueError("No JSON object found in Groq response")

        response_data = json.loads(json_match.group(0))
        prompt_cfg = PROMPTS.get("label_workloads", {})
        output_schema = prompt_cfg.get("output_schema")

        if output_schema:
            try:
                output = output_schema.from_dict(response_data)  # type: ignore[attr-defined]
                decisions = output.decisions
                explanations = output.explanations
            except Exception as e:
                logger.warning(f"Schema validation failed: {e}")
                decisions = response_data.get("decisions", [])
                explanations = response_data.get("explanations", [])
        else:
            decisions = response_data.get("decisions", [])
            explanations = response_data.get("explanations", [])

        if len(decisions) != len(df):
            raise ValueError("Mismatch between decisions and workloads length")

        labels = [int(d) for d in decisions]
        explanation_output = {
            "explanation": "Migration decisions based on Groq Llama AI model",
            "workload_explanations": [],
        }
        for idx, (label, workload) in enumerate(zip(labels, df.iterrows())):
            workload_id = workload[1].get("workload_id", f"workload-{idx}")
            kind = workload[1].get("kind", "unknown")
            destination = "public" if label == 1 else "private"
            exp = (
                explanations[idx]
                if idx < len(explanations)
                else f"Workload {workload_id} recommended for {destination} cluster based on resource requirements"
            )
            explanation_output["workload_explanations"].append(exp)
            logger.info(f"Decision for {workload_id} ({kind}): Cluster {destination}")

        return labels, explanation_output

    except Exception as e:
        logger.error(f"Error using Groq Llama model: {e}")
        return _label_workloads_with_heuristics(workloads), {
            "explanation": f"Used heuristic rules due to Groq error: {e}",
            "workload_explanations": [],
        }
# ---------------------------------------------------------------------------
# MultiAgent implementation
# ---------------------------------------------------------------------------

def label_workloads_multiagent(
    workloads: Union[list, "pd.DataFrame"],
) -> Tuple[List[int], Dict[str, Any]]:
    state = {
        "workloads": workloads,
        "cpu_votes": [],
        "mem_votes": [],
        "pending_votes": [],
        "final_decisions": [],
        "explanations": {}
    }

    graph = build_graph()
    final_state = graph.invoke(state)

    labels = final_state.get("final_decisions", [])

    explanations = final_state.get("explanations", {})

    return labels, explanations

# ---------------------------------------------------------------------------
# Generic wrapper
# ---------------------------------------------------------------------------


def label_workloads(
    workloads: Union[list, "pd.DataFrame"],
    provider: str | None = None,
    multiagent: bool | None = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """Public API to label workloads with the configured AI provider."""
    
    if provider is None or multiagent is None:
        cfg = load_config()
    
    if provider is None:
        provider = cfg.get("ai", {}).get("selected_model", "gemini")
    
    if multiagent is None:
        multiagent = cfg.get("ai", {}).get("multiagent", False)
    
    if multiagent:
        return label_workloads_multiagent(workloads)    
    
    provider = provider.lower()
    if provider in {"gemini", "google"}:
        return label_workloads_with_gemini(workloads)
    if provider in {"llama", "groq", "llama_groq"}:
        return label_workloads_with_llama(workloads)

    logger.warning(f"Unknown provider '{provider}'. Falling back to heuristics.")
    labels = _label_workloads_with_heuristics(workloads)
    return labels, {
        "explanation": "Used heuristic rules due to unknown provider.",
        "workload_explanations": [],
    }


def _label_workloads_with_heuristics(
    workloads: Union[list, "pd.DataFrame"],
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
        if isinstance(cpu, str) and cpu.endswith("m"):
            return float(cpu[:-1]) / 1000.0
        return float(cpu) if cpu else 0.0

    def mem_to_float(mem):
        if isinstance(mem, str) and mem.endswith("Mi"):
            return float(mem[:-2])
        return float(mem) if mem else 0.0

    # Extract and convert resources
    try:
        cpu_values = df["resources"].apply(
            lambda x: cpu_to_float(x.get("cpu", 0)) if isinstance(x, dict) else 0.0
        )
        mem_values = df["resources"].apply(
            lambda x: mem_to_float(x.get("memory", 0)) if isinstance(x, dict) else 0.0
        )
    except:
        # Alternative if the format is different
        cpu_values = df.get("resources.cpu", df.get("cpu", 0)).apply(cpu_to_float)
        mem_values = df.get("resources.memory", df.get("memory", 0)).apply(mem_to_float)

    # Rules heuristics:
    # 1. If CPU > 0.5 or memory > 1024Mi: move to public
    # 2. If percent_pending > 20%: move to public
    try:
        percent_pending = df["percent_pending"].fillna(0)
    except:
        percent_pending = pd.Series([0] * len(df))

    # Combine rules to decide: 0=private, 1=public
    labels = ((cpu_values > 0.5) | (mem_values > 1024) | (percent_pending > 50)).astype(
        int
    )

    return labels.tolist()

# Get metrics of token usage and requests
def get_usage_metrics() -> Dict[str, Any]:
    return {
        "total_requests": REQUEST_COUNTER,
        "total_tokens": TOKEN_TOTALS  
    }
