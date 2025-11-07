from typing import List, Union, Dict, Any, Tuple, Optional
import os
import pandas as pd
import re
import json
import uuid
from dotenv import load_dotenv
from .ai_config import get_model_config, get_prompt, PROMPTS
from .util import get_logger, load_config, log_token_usage
from .langgraph_agents.graph.migration_graph import create_migration_graph
from .client import OpenRouterClient

logger = get_logger("agents")

# Initialize OpenRouter client
try:
    openrouter_client = OpenRouterClient()
    HAS_OPENROUTER = True
except Exception as e:
    logger.warning(f"Failed to initialize OpenRouterClient: {e}")
    openrouter_client = None
    HAS_OPENROUTER = False

# Keep legacy imports for fallback
try:
    import google.generativeai as genai

    HAS_GENAI = True
except ImportError:
    genai = None
    HAS_GENAI = False


# global variables
REQUEST_COUNTER = 0
TOKEN_TOTALS = {"input": 0, "output": 0, "total": 0}


def _normalize_workloads_to_dataframe(
    workloads: Union[list, "pd.DataFrame"],
) -> "pd.DataFrame":
    """Convert workloads input to DataFrame format."""
    return pd.DataFrame(workloads) if isinstance(workloads, list) else workloads


def _extract_json_from_response(text_response: str) -> Dict[str, Any]:
    """Extract and parse JSON from model response."""
    json_match = re.search(r"\{[\s\S]*\}", text_response)
    if not json_match:
        raise ValueError("No JSON object found in model response")

    json_str = json_match.group(0)
    return json.loads(json_str)


def _validate_and_extract_decisions(
    response_data: Dict[str, Any],
) -> Tuple[List[int], List[str]]:
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


def _create_explanation_output(
    labels: List[int], explanations: List[str], df: "pd.DataFrame"
) -> Dict[str, Any]:
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


def label_workloads_with_llm(
    workloads: Union[list, "pd.DataFrame"],
    model: str = "google/gemini-2.0-flash-001",
    client: OpenRouterClient = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Uses LLM API to decide workload labels with explanations.
    Each label: 0 = private, 1 = public.
    Args:
        workloads: list of dicts or DataFrame with workload fields.
        model: Model to use via Provider (default: google/gemini-2.0-flash-001)
    Returns:
        Tuple containing:
        - List of labels (0 or 1) in the same order
        - Dictionary with explanations for each workload
    """
    global REQUEST_COUNTER, TOKEN_TOTALS

    logger.info("Starting workload analysis for migration decision using OpenRouter")

    # Dependency validation - early return if not available
    if not HAS_OPENROUTER or not openrouter_client:
        logger.error("OpenRouter client not available, skipping this cycle")
        return [], {}

    # Data preparation
    df = _normalize_workloads_to_dataframe(workloads)
    user_prompt = get_prompt(
        "label_workloads", workloads_json=df.to_json(orient="records", indent=2)
    )
    system_prompt = "You are an expert Kubernetes workload migration advisor. Analyze the provided workloads and make migration decisions."

    logger.info(f"Sending request to OpenRouter with model: {model}")

    try:
        config = load_config()
        model_config = config.get("ai", {}).get("models", {}).get("gemini", {})
        generation_config = model_config.get("generation_config", {})

        logger.info(f"CONFIG: Using OpenRouter model: {model}")
        logger.info(f"CONFIG: Using generation config: {generation_config}")

        text_response = openrouter_client.chat(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=generation_config.get("temperature", 0.1),
            max_tokens=generation_config.get("max_output_tokens", 8000),
        )

        REQUEST_COUNTER += 1
        logger.info(f"OpenRouter API request count: {REQUEST_COUNTER}")
        logger.debug(f"Complete OpenRouter response: {text_response}")

        # Parse and validate response
        response_data = _extract_json_from_response(text_response)
        decisions, explanations = _validate_and_extract_decisions(response_data)

        # Validate decision count - handle mismatches gracefully
        if len(decisions) != len(df):
            logger.warning(
                f"Decision count mismatch: expected {len(df)}, got {len(decisions)}"
            )

            # Handle mismatch by adjusting decisions list
            if len(decisions) > len(df):
                # Too many decisions - truncate
                decisions = decisions[: len(df)]
                explanations = (
                    explanations[: len(df)]
                    if len(explanations) > len(df)
                    else explanations
                )
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
                    explanations.append(
                        f"Maintaining original cluster ({original_cluster}) due to missing AI decision"
                    )

                logger.info(
                    f"Padded {missing_count} missing decisions with original cluster assignments (no migration)"
                )

        # Convert to integers and create output
        labels = [int(decision) for decision in decisions]
        logger.info("Migration decisions extracted from JSON response")

        explanation_output = _create_explanation_output(labels, explanations, df)
        return labels, explanation_output

    except Exception as e:
        logger.error(f"Error parsing JSON response: {e}")
        logger.error(f"Raw LLM response (first 500 chars): {text_response[:500]}")
        logger.error("LLM failed to generate recommendations, skipping this cycle")
        return [], {}
    except Exception as e:
        logger.error(f"Error using {client} API: {e}")
        logger.error("LLM failed to generate recommendations, skipping this cycle")
        return [], {}


def label_workloads_multiagent(workloads, provider="langgraph"):
    df = _normalize_workloads_to_dataframe(workloads)
    state = {
        "workloads": df.to_dict(orient="records"),
        "cpu_votes": [],
        "mem_votes": [],
        "pending_votes": [],
        "final_decisions": [],
        "explanations": {},
    }

    graph = create_migration_graph()

    thread_id = str(uuid.uuid4())

    final_state = graph.invoke(state, config={"configurable": {"thread_id": thread_id}})

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
        return label_workloads_with_llm(
            workloads, model="google/gemini-2.0-flash-001"
        )
    if provider in {"openrouter"}:
        return label_workloads_with_llm(workloads)

    logger.error(f"Unknown provider '{provider}', skipping this cycle")
    return [], {}


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
    return {"total_requests": REQUEST_COUNTER, "total_tokens": TOKEN_TOTALS}
