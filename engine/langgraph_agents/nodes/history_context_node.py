"""
History Context Node for LangGraph Pipeline

Fetches historical recommendations context and adds it to the graph state
before the recommendations node executes.
"""

import logging
from typing import Dict, Any
from engine.util import load_config
from engine.history_batch_manager import get_history_batch_manager
from engine.recommendation_history_db import get_historical_context_string

logger = logging.getLogger(__name__)


def fetch_history_context_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch historical recommendations context and add to state.
    This node runs BEFORE the recommendations node.

    Args:
        state: Current graph state

    Returns:
        Dict with historical_context key containing formatted string
    """
    try:
        # Load config
        config = load_config()

        # Check if feature is enabled
        history_config = config.get("recommendation_history", {})
        if not history_config.get("enabled", False):
            logger.debug("Recommendation history feature disabled")
            return {"historical_context": ""}

        # Get current history batch ID
        history_batch_mgr = get_history_batch_manager(config)
        current_history_batch_id = history_batch_mgr.get_current()

        # If this is the first history batch, no context available
        if current_history_batch_id <= 1:
            logger.info("No historical context available (first batch)")
            return {"historical_context": ""}

        # Retrieve historical context
        lookback_count = history_config.get("lookback_count", 3)
        historical_context = get_historical_context_string(
            config.get("recommendation_history", {}).get("mongodb"), current_history_batch_id, lookback_count
        )

        if historical_context:
            logger.info(
                f"Retrieved historical context for history batch {current_history_batch_id} (lookback: {lookback_count} batches)"
            )
        else:
            logger.info(
                f"No historical summaries found for history batch {current_history_batch_id}"
            )

        return {"historical_context": historical_context}

    except Exception as e:
        logger.error(f"Error fetching historical context: {e}", exc_info=True)
        # Return empty context on error to avoid breaking the pipeline
        return {"historical_context": ""}
