"""
History Batch Manager Module

Manages history batch IDs independently from the logging batch_id system.
This is used exclusively for the recommendation history feature.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class HistoryBatchManager:
    """
    Manages history batch IDs independently from the logging batch_id system.
    This is used exclusively for the recommendation history feature.
    """

    def __init__(self, mongodb_config: Optional[dict] = None):
        """
        Initialize the HistoryBatchManager.

        Args:
            mongodb_config: MongoDB configuration dictionary with host, port, database, collection
        """
        self._mongodb_config = mongodb_config
        self._current_history_batch_id = self._load_latest_batch_id()

    def _load_latest_batch_id(self) -> int:
        """
        Load the latest history_batch_id from database or return 0.

        Returns:
            The latest history_batch_id from the database, or 0 if no records exist
        """
        if not self._mongodb_config:
            logger.debug("No MongoDB configured, starting with batch ID 0")
            return 0

        try:
            from .recommendation_history_db import get_latest_history_batch_id

            latest_id = get_latest_history_batch_id(self._mongodb_config)
            logger.info(
                f"Loaded latest history batch ID from MongoDB: {latest_id}"
            )
            return latest_id
        except Exception as e:
            logger.warning(
                f"Failed to load latest batch ID from MongoDB: {e}. Starting with 0."
            )
            return 0

    def start_new_batch(self) -> int:
        """
        Increment and return new history batch ID.
        Also inserts a new record in MongoDB.

        Returns:
            The new history_batch_id
        """
        self._current_history_batch_id += 1
        new_batch_id = self._current_history_batch_id

        if self._mongodb_config:
            try:
                from .recommendation_history_db import insert_history_batch

                insert_history_batch(self._mongodb_config, new_batch_id)
                logger.info(f"Started new history batch: {new_batch_id}")
            except Exception as e:
                logger.error(
                    f"Failed to insert new history batch {new_batch_id}: {e}"
                )
                # Don't fail the entire operation if database write fails
        else:
            logger.debug(
                f"No MongoDB configured, incremented history batch ID to {new_batch_id}"
            )

        return new_batch_id

    def get_current(self) -> int:
        """
        Get current history batch ID (0 if no batch started).

        Returns:
            Current history_batch_id
        """
        return self._current_history_batch_id


# Singleton instance
_history_batch_manager: Optional[HistoryBatchManager] = None


def get_history_batch_manager(config: Optional[dict] = None) -> HistoryBatchManager:
    """
    Get or create the singleton HistoryBatchManager.

    Args:
        config: Configuration dictionary containing recommendation_history settings

    Returns:
        The singleton HistoryBatchManager instance
    """
    global _history_batch_manager

    if _history_batch_manager is None:
        mongodb_config = None
        if config:
            mongodb_config = config.get("recommendation_history", {}).get("mongodb")

        _history_batch_manager = HistoryBatchManager(mongodb_config)
        logger.debug("Initialized HistoryBatchManager singleton")

    return _history_batch_manager


def reset_history_batch_manager():
    """
    Reset the singleton instance (mainly for testing purposes).
    """
    global _history_batch_manager
    _history_batch_manager = None
    logger.debug("Reset HistoryBatchManager singleton")
