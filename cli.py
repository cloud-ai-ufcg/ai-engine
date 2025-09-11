from engine.main import *

from engine.agents import get_usage_metrics

# Maintain a local batch counter for CLI runs
CURRENT_BATCH_ID = 0

def cli():
    """
    Analyze workloads in CLI mode
    """
    config = load_config()

    workloads = load_monitoring_data(config)
    if not workloads:
        logger.error("❌ No workloads found")
        return

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

    save_and_log_explanations(result, explanations, workloads)

    # Build WorkloadRecommendation-shaped list[dict] (keeps backward compatibility)
    recommendations = _build_workload_recommendations_cli(result, explanations, workloads)

    # Log a concise preview of recommendations
    logger.info(format_message(f"Prepared {len(recommendations)} recommendations (CLI)", icon="🧾", color="CYAN"))

    # Keep existing CSV output for backward compatibility
    write_recommendations(result)


def _build_workload_recommendations_cli(result_df, explanations, workloads):
    """
    CLI variant of the builder used in API to transform results into the
    WorkloadRecommendation-shaped dictionaries.

    Fields:
      - workload_id: str
      - kind: str
      - origin_cluster: int  (0=private, 1=public)
      - destination_cluster: int (0=private, 1=public) from result_df['label']
      - reason: str
    """
    # Map workload_id -> origin cluster label from original workloads
    origin_by_id = {}
    for w in workloads:
        wid = w.get("workload_id")
        if wid is not None:
            origin_by_id[wid] = w.get("cluster_label", "private")

    explanations_list = (explanations or {}).get("workload_explanations", [])

    # Increment batch id for this CLI build
    global CURRENT_BATCH_ID
    CURRENT_BATCH_ID += 1
    batch_id = CURRENT_BATCH_ID

    recs = []
    for idx, row in result_df.iterrows():
        wid = row.get("workload_id")
        kind = row.get("kind")
        label = int(row.get("label", 0))

        origin_label = origin_by_id.get(wid, "private")
        origin_cluster = 0 if origin_label == "private" else 1
        destination_cluster = label

        reason = (
            explanations_list[idx]
            if idx < len(explanations_list)
            else (
                f"Recommended to {'public' if destination_cluster == 1 else 'private'} cluster based on resource analysis"
            )
        )

        recs.append(
            WorkloadRecommendation(
                batch_id=batch_id,
                workload_id=wid,
                kind=kind,
                origin_cluster=origin_cluster,
                destination_cluster=destination_cluster,
                reason=reason,
            )
        )

    return recs


if __name__ == "__main__":
    cli()
