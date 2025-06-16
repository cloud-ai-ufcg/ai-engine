from fastapi import FastAPI
import uvicorn
from typing import Dict, Any
import json

app = FastAPI(
    title="Metrics API",
    description="API for providing monitoring metrics",
    version="1.0.0",
)


@app.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    metrics_data = {
        "1747936314": {
            "workloads": [
                {
                    "workload_id": "default/prometheus-kube-state-metrics",
                    "kind": "deployment",
                    "resources": {"cpu": "0m", "memory": "0Mi"},
                    "pods_total": 1,
                    "pods_pending": 0,
                    "percent_pending": 0,
                    "cluster_label": "private",
                },
                {
                    "workload_id": "default/prometheus-prometheus-pushgateway",
                    "kind": "deployment",
                    "resources": {"cpu": "0m", "memory": "0Mi"},
                    "pods_total": 1,
                    "pods_pending": 0,
                    "percent_pending": 0,
                    "cluster_label": "private",
                },
                {
                    "workload_id": "default/prometheus-server",
                    "kind": "deployment",
                    "resources": {"cpu": "0m", "memory": "0Mi"},
                    "pods_total": 1,
                    "pods_pending": 0,
                    "percent_pending": 0,
                    "cluster_label": "private",
                },
                {
                    "workload_id": "default/stress",
                    "kind": "deployment",
                    "resources": {"cpu": "60000m", "memory": "7680Mi"},
                    "pods_total": 30,
                    "pods_pending": 25,
                    "percent_pending": 83.33333333333334,
                    "cluster_label": "private",
                },
                {
                    "workload_id": "default/test-monitor-job",
                    "kind": "job",
                    "resources": {"cpu": "1500m", "memory": "1920Mi"},
                    "pods_total": 15,
                    "pods_pending": 6,
                    "percent_pending": 40,
                    "cluster_label": "private",
                },
                {
                    "workload_id": "default/autoscaler-kwok-kwok-cluster-autoscaler",
                    "kind": "deployment",
                    "resources": {"cpu": "0m", "memory": "0Mi"},
                    "pods_total": 1,
                    "pods_pending": 0,
                    "percent_pending": 0,
                    "cluster_label": "public",
                },
                {
                    "workload_id": "default/prometheus-kube-state-metrics",
                    "kind": "deployment",
                    "resources": {"cpu": "0m", "memory": "0Mi"},
                    "pods_total": 1,
                    "pods_pending": 0,
                    "percent_pending": 0,
                    "cluster_label": "public",
                },
                {
                    "workload_id": "default/prometheus-prometheus-pushgateway",
                    "kind": "deployment",
                    "resources": {"cpu": "0m", "memory": "0Mi"},
                    "pods_total": 1,
                    "pods_pending": 0,
                    "percent_pending": 0,
                    "cluster_label": "public",
                },
                {
                    "workload_id": "default/prometheus-server",
                    "kind": "deployment",
                    "resources": {"cpu": "0m", "memory": "0Mi"},
                    "pods_total": 1,
                    "pods_pending": 0,
                    "percent_pending": 0,
                    "cluster_label": "public",
                },
            ],
            "cluster_info": [
                {
                    "cluster_label": "private",
                    "cluster_load": {"cpu": 5.212, "memory": 0.317},
                    "cluster_cpu_capacity": "12000m",
                    "cluster_memory_capacity": "31782Mi",
                },
                {
                    "cluster_label": "public",
                    "cluster_load": {"cpu": 0.087, "memory": 0.015},
                    "cluster_cpu_capacity": "12000m",
                    "cluster_memory_capacity": "31782Mi",
                },
            ],
        },
        "1747936317": {
            "workloads": [
                {
                    "workload_id": "default/prometheus-kube-state-metrics",
                    "kind": "deployment",
                    "resources": {"cpu": "0m", "memory": "0Mi"},
                    "pods_total": 1,
                    "pods_pending": 0,
                    "percent_pending": 0,
                    "cluster_label": "private",
                },
                {
                    "workload_id": "default/prometheus-prometheus-pushgateway",
                    "kind": "deployment",
                    "resources": {"cpu": "0m", "memory": "0Mi"},
                    "pods_total": 1,
                    "pods_pending": 0,
                    "percent_pending": 0,
                    "cluster_label": "private",
                },
                {
                    "workload_id": "default/prometheus-server",
                    "kind": "deployment",
                    "resources": {"cpu": "0m", "memory": "0Mi"},
                    "pods_total": 1,
                    "pods_pending": 0,
                    "percent_pending": 0,
                    "cluster_label": "private",
                },
                {
                    "workload_id": "default/stress",
                    "kind": "deployment",
                    "resources": {"cpu": "60000m", "memory": "7680Mi"},
                    "pods_total": 30,
                    "pods_pending": 25,
                    "percent_pending": 83.33333333333334,
                    "cluster_label": "private",
                },
                {
                    "workload_id": "default/test-monitor-job",
                    "kind": "job",
                    "resources": {"cpu": "1500m", "memory": "1920Mi"},
                    "pods_total": 15,
                    "pods_pending": 6,
                    "percent_pending": 40,
                    "cluster_label": "private",
                },
                {
                    "workload_id": "default/autoscaler-kwok-kwok-cluster-autoscaler",
                    "kind": "deployment",
                    "resources": {"cpu": "0m", "memory": "0Mi"},
                    "pods_total": 1,
                    "pods_pending": 0,
                    "percent_pending": 0,
                    "cluster_label": "public",
                },
                {
                    "workload_id": "default/prometheus-kube-state-metrics",
                    "kind": "deployment",
                    "resources": {"cpu": "0m", "memory": "0Mi"},
                    "pods_total": 1,
                    "pods_pending": 0,
                    "percent_pending": 0,
                    "cluster_label": "public",
                },
                {
                    "workload_id": "default/prometheus-prometheus-pushgateway",
                    "kind": "deployment",
                    "resources": {"cpu": "0m", "memory": "0Mi"},
                    "pods_total": 1,
                    "pods_pending": 0,
                    "percent_pending": 0,
                    "cluster_label": "public",
                },
                {
                    "workload_id": "default/prometheus-server",
                    "kind": "deployment",
                    "resources": {"cpu": "0m", "memory": "0Mi"},
                    "pods_total": 1,
                    "pods_pending": 0,
                    "percent_pending": 0,
                    "cluster_label": "public",
                },
            ],
            "cluster_info": [
                {
                    "cluster_label": "private",
                    "cluster_load": {"cpu": 5.212, "memory": 0.317},
                    "cluster_cpu_capacity": "12000m",
                    "cluster_memory_capacity": "31782Mi",
                },
                {
                    "cluster_label": "public",
                    "cluster_load": {"cpu": 0.087, "memory": 0.015},
                    "cluster_cpu_capacity": "12000m",
                    "cluster_memory_capacity": "31782Mi",
                },
            ],
        },
    }

    return metrics_data


if __name__ == "__main__":
    uvicorn.run("fake_monitor:app", host="0.0.0.0", port=8082, reload=True)
