from typing import List, Union, Dict, Any, Tuple
import pandas as pd
import re
import json
import uuid
from .ai_config import (
    get_prompt,
    PROMPTS,
    build_system_prompt_from_config,
    get_agent_mode,
    get_agent_config,
    build_cluster_selection_from_config,
)
from .cluster_config import get_cluster_manager
from engine.langgraph_agents.nodes.agents_tools import _convert_binary_decisions_to_cluster_ids
from .util import get_logger, load_config


from .langgraph_agents.graph.tool_system_graph import create_tool_system_migration_graph
from .langgraph_agents.graph.vote_system_graph import create_vote_system_migration_graph
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


def _get_cluster_list_for_prompt() -> str:
    """
    Build a formatted list of available clusters for use in prompts.
    
    Returns:
        A formatted string describing available clusters and their indices
    """
    
    cluster_manager = get_cluster_manager()
    all_clusters = cluster_manager.get_all_clusters()
    listed_clusters = build_cluster_selection_from_config()
    
    # Filter clusters based on listed_clusters config
    if listed_clusters:
        clusters = [
            c for c in all_clusters 
            if c.cluster_label in listed_clusters or c.cluster_id in listed_clusters
        ]
        logger.info(f"Filtered clusters for LLM prompt: {[c.cluster_label for c in clusters]}")
    else:
        clusters = all_clusters
        logger.info(f"Using all clusters for LLM prompt (no filter configured)")
    
    if not clusters:
        logger.warning("No clusters available after filtering. Using default private/public clusters.")
        return "(0) private cluster, (1) public cluster"
    
    cluster_descriptions = []
    for idx, cluster in enumerate(clusters):
        cluster_descriptions.append(
            f"({idx}) {cluster.cluster_id} [Profile: {cluster.cluster_profile}] - {cluster.cluster_description}"
        )
    
    return ", ".join(cluster_descriptions)


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
    labels: List[str], explanations: List[str], df: "pd.DataFrame", cluster_manager=None
) -> Dict[str, Any]:
    """Create structured explanation output and log decisions."""
    if cluster_manager is None:
        cluster_manager = get_cluster_manager()
    
    explanation_output = {
        "workload_explanations": [],
    }

    for idx, (label, workload) in enumerate(zip(labels, df.iterrows())):
        workload_id = workload[1].get("workload_id", f"workload-{idx}")
        kind = workload[1].get("kind", "unknown")
        current_cluster = workload[1].get("cluster_label", "unknown")
        destination_cluster = label

        dest_cluster_config = cluster_manager.get_cluster_by_id(destination_cluster)
        dest_cluster_label = dest_cluster_config.cluster_label if dest_cluster_config else destination_cluster
        
        is_staying = (
            current_cluster == destination_cluster or 
            current_cluster == dest_cluster_label
        )
        logger.info(
            f"Decision for {workload_id} ({kind}): {current_cluster} → {destination_cluster} "
            f"[Staying: {is_staying}]"
        )

        llm_explanation = (
            explanations[idx]
            if idx < len(explanations)
            else f"Workload {workload_id} recommended for {destination_cluster} cluster based on resource requirements"
        )
        
        # Check for potential hallucinations (explanation contradicts decision)
        if llm_explanation and is_staying:
            migration_keywords = ["migrate", "move", "transfer", "relocate", "shift", "should go"]
            if any(keyword in llm_explanation.lower() for keyword in migration_keywords):
                logger.warning(
                    f"⚠️  POTENTIAL LLM HALLUCINATION - {workload_id}: "
                    f"Explanation mentions migration but workload stays in {current_cluster}. "
                    f"Explanation: \"{llm_explanation}\""
                )
        
        explanation_output["workload_explanations"].append(llm_explanation)

    return explanation_output


def label_workloads_with_llm(
    workloads: Union[list, "pd.DataFrame"],
    model: str = "google/gemini-2.0-flash-001",
    client: OpenRouterClient = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Uses LLM API to decide workload labels with explanations.
    Each label is now a cluster ID (e.g., "private-cluster-1", "aws-3241").
    
    Args:
        workloads: list of dicts or DataFrame with workload fields.
        model: Model to use via Provider (default: google/gemini-2.0-flash-001)
    Returns:
        Tuple containing:
        - List of cluster IDs (strings) in the same order as workloads
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
    
    cluster_manager = get_cluster_manager()
    all_clusters = cluster_manager.get_all_clusters()
    listed_clusters = build_cluster_selection_from_config()
    
    if listed_clusters:
        clusters = [
            c for c in all_clusters 
            if c.cluster_label in listed_clusters or c.cluster_id in listed_clusters
        ]
    else:
        clusters = all_clusters
    
    cluster_list_str = _get_cluster_list_for_prompt()
    
    user_prompt = get_prompt(
        "label_workloads", 
        workloads_json=df.to_json(orient="records", indent=2),
        cluster_list=cluster_list_str,
        num_clusters=len(clusters)
    )

    # Initialize to avoid UnboundLocalError if exception raised before assignment
    text_response = ""

    try:
        config = load_config()
        model = config.get("ai", {}).get(
            "selected_model", "google/gemini-2.0-flash-001"
        )

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
                # Too few decisions - pad with binary 0 (stay in first cluster - no migration)
                missing_count = len(df) - len(decisions)
                for i in range(missing_count):
                    decisions.append(0)  # Binary 0 = stay (will be converted to first cluster)
                    explanations.append(
                        f"Maintaining current cluster due to missing AI decision"
                    )
                logger.info(
                    f"Padded {missing_count} missing decisions with 0 (no migration)"
                )

        # Convert binary indices (0, 1, 2, ...) to actual cluster IDs (strings)
        labels = _convert_binary_decisions_to_cluster_ids(decisions, workloads)
        logger.info(f"Converted binary decisions {decisions} to cluster IDs: {labels}")

        explanation_output = _create_explanation_output(labels, explanations, df, cluster_manager)
        return labels, explanation_output

    except Exception as e:
        logger.warning(f"Error parsing JSON response: {e}")

        # Fallback: try to extract just the cluster indices/IDs
        # Pattern to match cluster indices or IDs
        pattern = r"\d+"
        matches = re.findall(pattern, text_response)

        if matches and len(matches) >= len(df):
            logger.info(f"Cluster indices identified in response")
            # Convert indices to cluster IDs
            labels = []
            for match in matches[:len(df)]:
                idx = int(match)
                if 0 <= idx < len(clusters):
                    labels.append(clusters[idx].cluster_id)
                else:
                    # Fallback to original cluster
                    labels.append(df.iloc[len(labels)].get('cluster_label', 'private'))

            # Create a basic explanation output
            explanation_output = {
                "explanation": "Migration decisions based on resource usage patterns",
                "workload_explanations": [],
            }

            # Log the decision for each workload
            for idx, (label, workload) in enumerate(zip(labels, df.iterrows())):
                workload_id = workload[1].get("workload_id", f"workload-{idx}")
                kind = workload[1].get("kind", "unknown")
                logger.info(
                    f"Decision for {workload_id} ({kind}): Cluster {label}"
                )

                # Add a generic explanation
                explanation = f"Workload {workload_id} ({kind}) recommended for {label} cluster based on resource requirements"
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

            explanation = f"Workload {workload_id} ({kind}) recommended for {label} cluster based on resource analysis"
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

            explanation = f"Workload {workload_id} ({kind}) recommended for {label} cluster based on resource analysis"
            explanation_output["workload_explanations"].append(explanation)

        return labels, explanation_output
    

# ---------------------------------------------------------------------------
# MultiAgent implementation
# ---------------------------------------------------------------------------
def label_workloads_multiagent(
    workloads: List[dict], cluster_info: List[dict], interval_duration: str = None
) -> Tuple[List[str], Dict[str, Any]]:
    df = _normalize_workloads_to_dataframe(workloads)

    # Build initial state WITHOUT pre-populating 'decisions' or 'explanations'
    # so they only appear in the graph output (not in the input trace).
    state = {
        "workloads": df.to_dict(orient="records"),
        "cluster_info": cluster_info,
        "interval_duration": interval_duration,
    }

    graph = create_tool_system_migration_graph()
    logger.info("Using LangGraph for workload recommendations")

    thread_id = str(uuid.uuid4())

    final_state = graph.invoke(state, config={"configurable": {"thread_id": thread_id}})

    recommendations_dict = final_state.get("explanations", {})
    final_decisions = final_state.get("decisions", [])
    workload_explanations = recommendations_dict.get("workload_explanations", [])

    if not final_decisions or len(final_decisions) != len(workloads):
        logger.warning(
            f"Number of decisions ({len(final_decisions)}) does not match number of workloads ({len(workloads)}). "
            "Filling missing recommendations with -1."
        )

        corrected_decisions = []
        missing_workloads = []

        for i, wl in enumerate(workloads):
            if i < len(final_decisions):
                corrected_decisions.append(final_decisions[i])
            else:
                corrected_decisions.append(-1)
                missing_workloads.append(wl.get("workload_id", f"workload_{i}"))

        if missing_workloads:
            logger.warning(
                f"No response from LLM for workloads: {', '.join(missing_workloads)}"
            )

        labels = corrected_decisions
    else:
        labels = _convert_binary_decisions_to_cluster_ids(final_decisions, workloads)

    final_explanations = {
        "workload_explanations": workload_explanations,
    }

    return labels, final_explanations


def label_workloads_multiagent_votes(workloads, provider="langgraph"):
    """
    Label workloads using a multi-agent with voting system.
    """
    df = _normalize_workloads_to_dataframe(workloads)
    # Do not pre-populate 'final_decisions' or 'explanations' here either;
    # let the graph/nodes produce them as output so they don't show up in input traces.
    state = {
        "workloads": df.to_dict(orient="records"),
        "cpu_votes": [],
        "mem_votes": [],
        "pending_votes": [],
        "final_decisions": [],
        "explanations": {},
    }

    graph = create_vote_system_migration_graph()

    thread_id = str(uuid.uuid4())

    final_state = graph.invoke(state, config={"configurable": {"thread_id": thread_id}})

    final_decisions = final_state.get("final_decisions", [])
    
    labels = _convert_binary_decisions_to_cluster_ids(final_decisions, workloads)

    explanations = final_state.get("explanations", {})

    return labels, explanations


# ---------------------------------------------------------------------------
# Generic wrapper
# ---------------------------------------------------------------------------
def label_workloads(
    workloads: Union[list, "pd.DataFrame"],
    cluster_info: List[dict] = None,
    interval_duration: str = None,
    provider: str | None = None,
    multiagent: bool | None = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Public API to label workloads with the configured AI provider.
    Now returns cluster IDs (strings) instead of binary decisions.
    
    Args:
        workloads: List of workloads or DataFrame
        cluster_info: List of cluster information dictionaries (optional)
        provider: Override the configured provider (optional)
        multiagent: Override the configured mode (optional, legacy parameter)

    Returns:
        Tuple of (cluster_ids, explanations) where cluster_ids are now strings
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
        # Use provided cluster_info or default to empty list
        if cluster_info is None:
            cluster_info = []
            logger.warning(
                "No cluster_info provided to label_workloads, using empty list"
            )
        labels, explanations = label_workloads_multiagent(
            workloads, cluster_info, interval_duration
        )
    elif provider in {"gemini", "google", "openrouter"}:
        labels, explanations = label_workloads_with_llm(workloads)
    else:
        logger.warning(f"Unknown provider '{provider}'. skipping this cycle.")
        explanations = {
            "workload_explanations": [],
        }

    # Normalize explanations format
    if isinstance(explanations, list):
        workload_explanations = explanations
        explanations = {
            "workload_explanations": workload_explanations,
        }

    return labels, explanations
