from typing import List, Union, Dict, Any, Tuple
import os
import pandas as pd
import re
import json
from .ai_config import get_model_config, get_prompt, PROMPTS
from dotenv import load_dotenv
from .util import get_logger, load_config, log_token_usage


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
    logger.info("Starting workload analysis for migration decision")

    if not HAS_GENAI:
        logger.warning(
            "Gemini model not available. Using traditional model as fallback."
        )
        return _label_workloads_with_heuristics(workloads)

    load_dotenv()

    config = load_config()
    api_key = config.get("api-key", {}).get("google") or os.environ.get(
        "GOOGLE_API_KEY"
    )
    if not api_key:
        logger.warning(
            "API Key for Gemini not found. Using traditional model as fallback."
        )
        return _label_workloads_with_heuristics(workloads)

    if isinstance(workloads, list):
        df = pd.DataFrame(workloads)
    else:
        df = workloads

    # Get the prompt from the configuration system
    prompt = get_prompt(
        "label_workloads", workloads_json=df.to_json(orient="records", indent=2)
    )

    logger.info("Sending request to Gemini model")

    try:
        # Get model configuration from the configuration system
        model_config = get_model_config("gemini")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_config["model_name"])

        
        response = model.generate_content(prompt, generation_config=model_config["generation_config"])

        # Use generalized token counting (4 chars = 1 token)
        token_counts = log_token_usage(
            prompt=prompt,
            response=response.text,
            model_type="gemini"  # Still track model type for analytics
        )

        logger.info(
            f"Token Usage (4 chars = 1 token) | "
            f"Input: {token_counts['input_tokens']} | "
            f"Output: {token_counts['output_tokens']} | "
            f"Total: {token_counts['total_tokens']}"
        )

        text_response = response.text
        logger.debug(f"Complete Gemini response: {text_response}")

        # Try to parse the JSON response
        try:
            # Extract JSON from the response (in case there's any surrounding text)
            json_match = re.search(r"\{[\s\S]*\}", text_response)
            if json_match:
                json_str = json_match.group(0)
                response_data = json.loads(json_str)

                # Use the structured output model for validation
                prompt_config = PROMPTS.get("label_workloads", {})
                output_schema = prompt_config.get("output_schema")

                if output_schema:
                    try:
                        # Validate with Pydantic model
                        output = output_schema.from_dict(response_data)
                        if not output.validate_output():
                            logger.warning(
                                "Output validation failed: decisions and explanations have different lengths"
                            )

                        # Extract validated data
                        decisions = output.decisions
                        explanations = output.explanations
                    except Exception as e:
                        logger.error(f"Failed to validate response with schema: {e}")
                        # Fallback to direct extraction
                        decisions = response_data.get("decisions", [])
                        explanations = response_data.get("explanations", [])
                else:
                    # No schema defined, use direct extraction
                    decisions = response_data.get("decisions", [])
                    explanations = response_data.get("explanations", [])

                # Ensure we have the right number of decisions
                if len(decisions) == len(df):
                    logger.info(f"Migration decisions extracted from JSON response")
                    labels = [int(decision) for decision in decisions]

                    # Create explanation output
                    explanation_output = {
                        "explanation": "Migration decisions based on AI analysis of workload characteristics",
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

                        # Add explanation for this workload
                        explanation = (
                            explanations[idx]
                            if idx < len(explanations)
                            else f"Workload {workload_id} recommended for {destination} cluster based on resource requirements"
                        )
                        explanation_output["workload_explanations"].append(explanation)

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
            model=model_cfg.get("model_name", "llama3-70b-8192"),
            messages=[{"role": "user", "content": prompt}],
            temperature=model_cfg.get("generation_config", {}).get("temperature", 0.1),
            max_tokens=model_cfg.get("generation_config", {}).get(
                "max_output_tokens", 1024
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
# Generic wrapper
# ---------------------------------------------------------------------------


def label_workloads(
    workloads: Union[list, "pd.DataFrame"],
    provider: str | None = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """Public API to label workloads with the configured AI provider."""
    if provider is None:
        cfg = load_config()
        provider = cfg.get("ai", {}).get("selected_model", "gemini")

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
