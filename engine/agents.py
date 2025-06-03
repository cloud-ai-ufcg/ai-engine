from typing import List, Union, Dict, Any, Tuple, Callable
import os
import pandas as pd
import re
import json
from ai_config import get_model_config, get_prompt, PROMPTS
from dotenv import load_dotenv
from util import get_logger, load_config
from crewai import Agent, Task, Crew

logger = get_logger("agents")

try:
    import google.generativeai as genai

    HAS_GENAI = True
except ImportError:
    genai = None
    HAS_GENAI = False


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
    prompt = get_prompt("label_workloads", workloads_json=df.to_json(orient="records", indent=2))

    logger.info("Sending request to Gemini model")

    try:
        # Get model configuration from the configuration system
        model_config = get_model_config("gemini")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_config["model_name"])
        response = model.generate_content(prompt, generation_config=model_config["generation_config"])

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
                            logger.warning("Output validation failed: decisions and explanations have different lengths")
                        
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


# Tools for CrewAI agents
def analyze_workload_resources(workloads_json: str) -> str:
    """
    Analyzes workload resources and returns statistics.
    """
    try:
        workloads = json.loads(workloads_json)
        if not workloads:
            return "No workloads provided for analysis."

        # Convert resources to standard format
        for workload in workloads:
            if "resources" in workload and isinstance(workload["resources"], dict):
                cpu = workload["resources"].get("cpu", "0")
                memory = workload["resources"].get("memory", "0")

                # Convert CPU
                if isinstance(cpu, str) and cpu.endswith("m"):
                    cpu_value = float(cpu[:-1]) / 1000.0
                else:
                    cpu_value = float(cpu) if cpu else 0.0

                # Convert Memory
                if isinstance(memory, str) and memory.endswith("Mi"):
                    memory_value = float(memory[:-2])
                else:
                    memory_value = float(memory) if memory else 0.0

                workload["cpu_value"] = cpu_value
                workload["memory_value"] = memory_value

        # Calculate statistics
        total_workloads = len(workloads)
        avg_cpu = (
            sum(w.get("cpu_value", 0) for w in workloads) / total_workloads
            if total_workloads
            else 0
        )
        avg_memory = (
            sum(w.get("memory_value", 0) for w in workloads) / total_workloads
            if total_workloads
            else 0
        )

        # Count workload types
        workload_types = {}
        for w in workloads:
            kind = w.get("kind", "Unknown")
            workload_types[kind] = workload_types.get(kind, 0) + 1

        result = {
            "total_workloads": total_workloads,
            "average_cpu": round(avg_cpu, 3),
            "average_memory": round(avg_memory, 2),
            "workload_types": workload_types,
        }

        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error analyzing workloads: {str(e)}"


def label_workloads_with_crewai(
    workloads: Union[list, "pd.DataFrame"],
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Uses CrewAI to decide workload labels with explanations.
    Each label: 0 = private, 1 = public.

    Args:
        workloads: list of dicts or DataFrame with workload fields.

    Returns:
        Tuple containing:
        - List of labels (0 or 1) in the same order
        - Dictionary with explanations
    """
    logger.info("Starting workload analysis with CrewAI")

    # Convert workloads to DataFrame if needed
    if isinstance(workloads, list):
        df = pd.DataFrame(workloads)
    else:
        df = workloads.copy()

    # Convert DataFrame to JSON for the agents
    workloads_json = df.to_json(orient="records", indent=2)

    # Load environment variables and config
    load_dotenv()
    config = load_config()

    # Check for API key
    api_key = os.environ.get("GOOGLE_API_KEY") or config.get("api-key", {}).get(
        "google"
    )
    if not api_key:
        logger.warning("API Key not found. Using traditional model as fallback.")
        labels = _label_workloads_with_heuristics(workloads)
        return labels, {
            "explanation": "Used heuristic rules due to missing API key",
            "clusters": {},
        }

    try:
        # Define tools as simple functions - CrewAI 0.1.0 doesn't use Tool class
        # Instead, we'll pass these functions directly to the agents
        analyze_tool = analyze_workload_resources

        # Define agents - adapting for CrewAI 0.1.0
        analyst_agent = Agent(
            role="Kubernetes Resource Analyst",
            goal="Analyze workload resources and identify patterns",
            backstory="You are an expert in Kubernetes resource management who can analyze workload patterns and resource usage.",
            verbose=True,
        )

        decision_agent = Agent(
            role="Migration Decision Maker",
            goal="Decide which workloads should be migrated to the public cluster",
            backstory="You are a Kubernetes cluster orchestrator who decides which workloads should stay in the private cluster or move to the public cluster.",
            verbose=True,
        )

        explanation_agent = Agent(
            role="Migration Explainer",
            goal="Explain migration decisions and provide cluster overview",
            backstory="You are a technical communicator who explains complex technical decisions in clear, concise language.",
            verbose=True,
        )

        # Define tasks - adapting for CrewAI 0.1.0
        analysis_task = Task(
            description=f"""Analyze the workloads and identify resource patterns and clusters.
            Workloads: {workloads_json}
            
            Use the tools to analyze the workload resources and identify natural clusters.
            Provide a detailed analysis of resource usage patterns.
            """,
            agent=analyst_agent,
        )

        decision_task = Task(
            description="""Based on the analysis, decide which workloads should be migrated to the public cluster.
            Decision rules:
            - Workloads with high demand (high CPU or memory usage) should go to the stronger cluster (public).
            - If percent_pending is high, consider moving to the public cluster.
            - Consider the cluster_load to avoid overloading the destination cluster.
            
            Provide your decision as a JSON with a 'labels' key containing an array of 0s and 1s,
            where each digit represents the recommended cluster for a workload (0=private, 1=public).
            """,
            agent=decision_agent,
            expected_output="A JSON with migration decisions",
            context=[analysis_task],
        )

        explanation_task = Task(
            description="""Explain the migration decisions and provide a cluster overview.
            For each workload, explain why it was recommended for the private or public cluster.
            Also provide a short overview of the identified clusters and their characteristics.
            
            Format your response as a JSON with the following structure:
            {
                "explanation": "Overall explanation of the migration strategy",
                "workload_explanations": [List of explanations for each workload],
                "clusters": "Overview of the identified clusters"
            }
            """,
            agent=explanation_agent,
            context=[analysis_task, decision_task],
        )

        # Create and run the crew - adapting for CrewAI 0.1.0
        crew = Crew(
            agents=[analyst_agent, decision_agent, explanation_agent],
            tasks=[analysis_task, decision_task, explanation_task],
            verbose=True,
        )

        result = crew.kickoff()
        logger.info("CrewAI analysis completed")

        try:
            decision_output = json.loads(decision_task.output)
            labels = decision_output.get("labels", [])

            # If labels are not in the expected format, extract them
            if not labels:
                # Try to extract a pattern of 0s and 1s
                pattern = r"[01]+"
                matches = re.findall(pattern, decision_task.output)
                if matches:
                    longest_match = max(matches, key=len)
                    if len(longest_match) == len(df):
                        labels = [int(digit) for digit in longest_match]

            # Parse the explanation result
            try:
                explanation_output = json.loads(explanation_task.output)
            except:
                # If parsing fails, create a basic structure
                explanation_output = {
                    "explanation": "Migration decisions based on resource usage patterns",
                    "workload_explanations": [],
                    "clusters": "Clusters identified based on resource usage",
                }

            # If we still don't have labels, fall back to heuristics
            if not labels or len(labels) != len(df):
                logger.warning(
                    "Could not extract valid labels from CrewAI output. Using heuristics."
                )
                labels = _label_workloads_with_heuristics(workloads)
                explanation_output["explanation"] += " (fallback to heuristics)"

            # Log the decisions
            for idx, (label, workload) in enumerate(zip(labels, df.iterrows())):
                workload_id = workload[1].get("workload_id", f"workload-{idx}")
                kind = workload[1].get("kind", "unknown")
                destination = "public" if label == 1 else "private"
                logger.info(
                    f"Decision for {workload_id} ({kind}): Cluster {destination}"
                )

            return labels, explanation_output

        except Exception as e:
            logger.warning(f"Error parsing CrewAI output: {e}")
            labels = _label_workloads_with_heuristics(workloads)
            return labels, {
                "explanation": f"Used heuristic rules due to error: {str(e)}",
                "clusters": {},
            }

    except Exception as e:
        logger.warning(f"Error using CrewAI: {e}")
        labels = _label_workloads_with_heuristics(workloads)
        return labels, {
            "explanation": f"Used heuristic rules due to error: {str(e)}",
            "clusters": {},
        }
