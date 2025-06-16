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

    save_and_log_explanations(result, explanations)

    write_recommendations(result)


if __name__ == "__main__":
    main()