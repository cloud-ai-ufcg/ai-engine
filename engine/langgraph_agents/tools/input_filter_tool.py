from typing import List, Dict, Any, Union, Optional
from langchain_core.tools import tool


process_monitoring_data = None  # will be set on first call


@tool("input_filter", return_direct=False)
def input_filter(
    data: Union[Dict[str, Any], List[Dict[str, Any]]],
    timestamp_lookback_seconds: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Normalize monitoring *data* into a list of workload dicts.

    This tool converts *raw* monitoring data (potentially in several formats) into a
    normalized list of workload dictionaries, mirroring the behaviour of
    ``engine.data_processor.process_monitoring_data``.

    It is a very thin wrapper around that function so it can be used as a LangChain
    / LangGraph "@tool".  If the data is already a list of workload dictionaries it
    will simply filter out entries that do not contain a ``workload_id`` key –
    replicating the legacy behaviour.

    Parameters
    ----------
    data : dict | list
        Raw monitoring payload.  If it is a dict, it is assumed to follow the
        timestamp-keyed structure expected by ``process_monitoring_data``.  If
        it is already a list of workload dictionaries we keep backward
        compatibility and only remove entries missing ``workload_id``.
    timestamp_lookback_seconds : int, optional
        Time-window used when *data* is a timestamp-keyed dict.  If omitted the
        default from the YAML configuration will be applied (handled inside
        ``process_monitoring_data``).

    Returns
    -------
    list[dict]
        Workload dictionaries enriched with cluster information and without
        zero-resource entries – exactly the same structure produced by
        ``process_monitoring_data``.
    """
    # Fast-path for the legacy case where *data* is already a list of workloads
    if isinstance(data, list):
        return [w for w in data if w.get("workload_id") is not None]

    # Otherwise delegate to the canonical implementation in engine.data_processor
    global process_monitoring_data
    if process_monitoring_data is None:
        # Local import to avoid issues with import ordering
        from engine.data_processor import process_monitoring_data as _process_monitoring_data  # type: ignore
        process_monitoring_data = _process_monitoring_data

    try:
        result = process_monitoring_data(
            data, timestamp_lookback_seconds=timestamp_lookback_seconds
        )
        # process_monitoring_data now returns a dict with workloads, cluster_info, etc.
        # For this tool, we only return the workloads list for backward compatibility
        if isinstance(result, dict):
            return result.get("workloads", [])
        # Fallback for old behavior (shouldn't happen with new implementation)
        return result if isinstance(result, list) else []
    except Exception as exc:  # pragma: no cover – defensive fallback
        # If anything goes wrong fallback to returning an empty list so that the
        # graph does not crash catastrophically.
        from engine.util import get_logger, format_message  # late import

        logger = get_logger("input_filter")
        logger.error(format_message(f"input_filter failed: {exc}", icon="❌", color="RED"))
        return []