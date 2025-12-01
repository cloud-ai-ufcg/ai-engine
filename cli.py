from engine.util import load_config, format_message, get_logger
from engine.main import (
    load_monitoring_data,
    analyze_workloads,
    save_and_log_explanations,
    write_recommendations,
)
from engine.util import build_workload_recommendations

logger = get_logger("cli")


# Maintain a local batch counter for CLI runs
# pylint: disable=invalid-name
current_batch_id = 0


def cli():
    """
    Analyze workloads in CLI mode
    """
    config = load_config()

    processed_data = load_monitoring_data(config)
    if not processed_data:
        logger.error("❌ No data found")
        return

    workloads = processed_data.get("workloads", [])
    cluster_info = processed_data.get("cluster_info", [])

    if not workloads:
        logger.error("❌ No workloads found")
        return

    result, explanations = analyze_workloads(
        workloads, config, cluster_info=cluster_info
    )

    save_and_log_explanations(result, explanations, workloads)

    global current_batch_id
    current_batch_id += 1

    # Build WorkloadRecommendation-shaped list[dict] (keeps backward compatibility)
    recommendations = build_workload_recommendations(
        result, explanations, workloads, current_batch_id
    )

    # Log a concise preview of recommendations
    logger.info(
        format_message(
            f"Prepared {len(recommendations)} recommendations (CLI)",
            icon="🧾",
            color="CYAN",
        )
    )

    # Keep existing CSV output for backward compatibility
    write_recommendations(result)


if __name__ == "__main__":
    cli()
