from engine import *

def main():
    """
    Main function that orchestrates the workload analysis process.
    """
    config = load_config()

    workloads = load_monitoring_data(config)
    if not workloads:
        logger.error("❌ No workloads found")
        return

    result, explanations = analyze_workloads(workloads, config)
    result, explanations = analyze_workloads(workloads, config)

    metrics = get_usage_metrics()

    logger.info(
         format_message(
        f"Requisições totais: {metrics['total_requests']} | "
        f"Tokens usados: {metrics['total_tokens']}",
        icon="📝",
        color="MAGENTA",
        bold=True,    
        )
    )

    save_and_log_explanations(result, explanations)

    write_recommendations(result)


if __name__ == "__main__":
    main()