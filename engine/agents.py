from typing import List, Union, Dict, Any, Tuple
import pandas as pd
import re
import json
import uuid
from .ai_config import get_model_config, get_prompt, PROMPTS, build_system_prompt_from_config, get_agent_mode, get_agent_config
from .util import get_logger, load_config, log_token_usage

from .langgraph_agents.graph.tool_system_graph import create_tool_system_migration_graph
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
        logger.warning(
            "OpenRouter client not available. Using traditional model as fallback."
        )
        return _label_workloads_with_heuristics(workloads)

    # Data preparation
    df = _normalize_workloads_to_dataframe(workloads)
    user_prompt = get_prompt(
        "label_workloads", workloads_json=df.to_json(orient="records", indent=2)
    )


    # Initialize to avoid UnboundLocalError if exception raised before assignment
    text_response = ""

    try:
        config = load_config()
        model = config.get("ai", {}).get("selected_model", "google/gemini-2.0-flash-001")

        model_config = config.get("ai", {}).get("default_config", {})
        system_prompt = build_system_prompt_from_config()
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

        # Adicionar logs detalhados para depuração
        logger.debug(f"Número de workloads no DataFrame: {len(df)}")
        logger.debug(f"Número de rótulos gerados: {len(labels)}")

        # Garantir que o número de rótulos corresponda ao número de workloads
        if len(labels) != len(df):
            logger.warning(
                f"Mismatch entre número de rótulos ({len(labels)}) e workloads ({len(df)}). Ajustando..."
            )

            if len(labels) > len(df):
                # Truncar rótulos extras
                labels = labels[: len(df)]
                explanations = explanations[: len(df)]
            else:
                # Preencher rótulos ausentes com valores padrão (0)
                missing_count = len(df) - len(labels)
                labels.extend([0] * missing_count)
                explanations.extend([
                    "Rótulo padrão aplicado devido à ausência de decisão do modelo"
                ] * missing_count)

            logger.info(f"Rótulos ajustados para corresponder ao número de workloads: {len(labels)}")

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
            logger.info(f"Extracting labels from {model} response: {text_response}")
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

        logger.warning(f"Could not extract labels from LLM, response: {text_response}")
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
        logger.warning(f"Error using {client} API: {e}")
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
# MultiAgent implementation
# ---------------------------------------------------------------------------
def label_workloads_multiagent(
    workloads: List[dict], cluster_info: List[dict]
) -> Tuple[List[int], Dict[str, Any]]:
    df = _normalize_workloads_to_dataframe(workloads)
    state = {
        "workloads": df.to_dict(orient="records"),
        "cluster_info": cluster_info,
        "decisions": [],
        "explanations": {},
    }

    graph = create_tool_system_migration_graph()
    logger.info("Using LangGraph for workload recommendations")

    thread_id = str(uuid.uuid4())

    final_state = graph.invoke(state, config={"configurable": {"thread_id": thread_id}})
    logger.info("Final state: ", final_state)

    recommendations_dict = final_state.get("explanations", {})

    final_decisions = final_state.get("decisions", [])
    workload_explanations = recommendations_dict.get("workload_explanations", [])
    overall_explanation = recommendations_dict.get(
        "overall_explanation", "No overall explanation provided."
    )

    if not final_decisions or len(final_decisions) != len(workloads):
        logger.warning("Final decisions missing or length mismatch. Using default labels (0) for all workloads.")
        labels = final_decisions
    else:
        labels = final_decisions

    final_explanations = {
        "overall_explanation": overall_explanation,
        "workload_explanations": workload_explanations,
    }

    return labels, final_explanations


def label_workloads_multiagent_votes(workloads, provider="langgraph"):
    """
    Label workloads using a multi-agent with voting system.
    """
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
    """
    Public API to label workloads with the configured AI provider.
    
    Args:
        workloads: List of workloads or DataFrame
        provider: Override the configured provider (optional)
        multiagent: Override the configured mode (optional, legacy parameter)
    
    Returns:
        Tuple of (labels, explanations)
    """
    cfg = load_config()

    # Determine the agent mode
    if multiagent is not None:
        # Legacy parameter support
        mode = "multi_agent" if multiagent else "single_agent"
        logger.info(f"Using legacy multiagent parameter: mode={mode}")
    else:
        mode = get_agent_mode()
    
    # Get mode-specific configuration
    agent_config = get_agent_config(mode)
    
    # Determine provider
    if provider is None:
        provider = agent_config.get("provider", "openrouter")
    
    provider = provider.lower()
    
    try:
        model = cfg.get("ai", {}).get("selected_model", "google/gemini-2.0-flash-001")
        generation_config = agent_config.get("generation_config", {})

        logger.info(f"CONFIG: Using agent mode: {mode}")
        logger.info(f"CONFIG: Using provider: {provider}")
        logger.info(f"CONFIG: Using model: {model}")
        logger.info(f"CONFIG: Using generation config: {generation_config}")
    except Exception as e:
        logger.warning(f"Could not log model configuration: {e}")
        
    labels = []
    explanations = {}
    
    # Route to appropriate labeling function based on mode
    if mode == "multi_agent":
        cluster_info = cfg.get("cluster_info", [])
        labels, explanations = label_workloads_multiagent(workloads, cluster_info)
    elif provider in {"gemini", "google", "openrouter"}:
        labels, explanations = label_workloads_with_llm(workloads)
    else:
        logger.warning(f"Unknown provider '{provider}'. Falling back to heuristics.")
        labels = _label_workloads_with_heuristics(workloads)
        explanations = {
            "overall_explanation": "Used heuristic rules due to unknown provider.",
            "workload_explanations": [],
        }

    # Normalize explanations format
    if isinstance(explanations, list):
        workload_explanations = explanations
        explanations = {
            "overall_explanation": "Recommendations for each workload:",
            "workload_explanations": workload_explanations,
        }

    return labels, explanations


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
