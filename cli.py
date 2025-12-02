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

    # Start new history batch if feature enabled
    history_config = config.get('recommendation_history', {})
    if history_config.get('enabled', False):
        from engine.history_batch_manager import get_history_batch_manager
        from engine.recommendation_history_db import save_history_batch_recommendations
        
        history_batch_mgr = get_history_batch_manager(config)
        current_history_batch_id = history_batch_mgr.start_new_batch()
        logger.info(f"Started history batch {current_history_batch_id}")

    result, explanations = analyze_workloads(workloads, config, cluster_info=cluster_info)

    metrics = get_usage_metrics()

    logger.info(
        format_message(
            f"Requisições totais: {metrics['total_requests']} | "
            f"Tokens usados: {metrics['total_tokens']}",
            icon="📝",
            color="MAGENTA",
            bold=True,
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
    
    # Save to history database if feature enabled
    if history_config.get('enabled', False):
        try:
            save_history_batch_recommendations(
                history_config.get('mongodb'),
                current_history_batch_id,
                recommendations
            )
            logger.info(f"Saved recommendations to history batch {current_history_batch_id}")
        except Exception as e:
            logger.error(f"Failed to save history batch recommendations: {e}")


    # Keep existing CSV output for backward compatibility
    write_recommendations(result)


if __name__ == "__main__":
    cli()
