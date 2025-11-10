"""
Recommendation History Database Module

Handles MongoDB operations for storing and retrieving
historical recommendation summaries.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from pymongo import MongoClient, ASCENDING
from .ai_config import WorkloadRecommendation

logger = logging.getLogger(__name__)


def _get_mongo_client(mongodb_config: Dict[str, Any]) -> MongoClient:
    """
    Create and return a MongoDB client.

    Args:
        mongodb_config: Dictionary with host, port, database, collection

    Returns:
        MongoClient instance
    """
    host = mongodb_config.get("host", "localhost")
    port = mongodb_config.get("port", 27017)
    
    return MongoClient(host, port)


def init_history_database(mongodb_config: Dict[str, Any]) -> None:
    """
    Initialize MongoDB collection and create indexes if they don't exist.

    Args:
        mongodb_config: Dictionary with host, port, database, collection
    """
    try:
        client = _get_mongo_client(mongodb_config)
        db = client[mongodb_config["database"]]
        collection = db[mongodb_config["collection"]]

        # Create index on history_batch_id for efficient queries
        collection.create_index([("history_batch_id", ASCENDING)])
        
        client.close()
        logger.info(f"MongoDB collection initialized successfully: {mongodb_config['database']}.{mongodb_config['collection']}")

    except Exception as e:
        logger.error(f"Failed to initialize MongoDB collection: {e}")
        raise


def get_latest_history_batch_id(mongodb_config: Dict[str, Any]) -> int:
    """
    Get the latest history_batch_id from MongoDB.

    Args:
        mongodb_config: Dictionary with host, port, database, collection

    Returns:
        Latest history_batch_id, or 0 if no records exist
    """
    try:
        client = _get_mongo_client(mongodb_config)
        db = client[mongodb_config["database"]]
        collection = db[mongodb_config["collection"]]

        # Find document with highest history_batch_id
        result = collection.find_one(
            sort=[("history_batch_id", -1)]
        )

        client.close()

        if result and "history_batch_id" in result:
            return int(result["history_batch_id"])
        return 0

    except Exception as e:
        logger.error(f"Failed to get latest history batch ID: {e}")
        return 0


def insert_history_batch(mongodb_config: Dict[str, Any], history_batch_id: int) -> None:
    """
    Insert a new history batch record.

    Args:
        mongodb_config: Dictionary with host, port, database, collection
        history_batch_id: The history batch ID to insert
    """
    try:
        client = _get_mongo_client(mongodb_config)
        db = client[mongodb_config["database"]]
        collection = db[mongodb_config["collection"]]

        # Check if batch already exists
        existing = collection.find_one({"history_batch_id": history_batch_id, "type": "batch"})
        if existing:
            logger.warning(f"History batch {history_batch_id} already exists (duplicate insert)")
            client.close()
            return

        # Insert batch document
        batch_doc = {
            "history_batch_id": history_batch_id,
            "type": "batch",
            "timestamp": datetime.now()
        }
        collection.insert_one(batch_doc)
        client.close()

        logger.debug(f"Inserted history batch {history_batch_id}")

    except Exception as e:
        logger.error(f"Failed to insert history batch {history_batch_id}: {e}")
        raise


def _generate_summary_string(
    workload_ids: List[str], direction: str, history_batch_id: int
) -> str:
    """
    Generate compact summary string for token efficiency.

    Args:
        workload_ids: List of workload IDs that were migrated
        direction: Migration direction ('private_to_public' or 'public_to_private')
        history_batch_id: The history batch ID

    Returns:
        Formatted summary string

    Example output:
        "Workloads nginx-1, redis-2, mysql-3 were migrated from private to public in history batch 5"
    """
    if direction == "private_to_public":
        direction_text = "from private to public"
    else:
        direction_text = "from public to private"

    # Join workload IDs with commas
    workload_list = ", ".join(workload_ids)

    # Use singular or plural form
    workload_word = "Workload" if len(workload_ids) == 1 else "Workloads"

    return f"{workload_word} {workload_list} {'was' if len(workload_ids) == 1 else 'were'} migrated {direction_text} in history batch {history_batch_id}"


def save_history_batch_recommendations(
    mongodb_config: Dict[str, Any],
    history_batch_id: int,
    recommendations: List[WorkloadRecommendation],
) -> None:
    """
    Save recommendations to MongoDB.

    Only migrated workloads (where origin_cluster != destination_cluster) are saved.
    Workloads are grouped by migration direction and summarized.

    Args:
        mongodb_config: Dictionary with host, port, database, collection
        history_batch_id: The history batch ID
        recommendations: List of WorkloadRecommendation objects
    """
    try:
        # Group workloads by migration direction
        private_to_public = []
        public_to_private = []

        for rec in recommendations:
            # Only process migrated workloads
            if rec.origin_cluster != rec.destination_cluster:
                if rec.origin_cluster == 0 and rec.destination_cluster == 1:
                    # Private to Public
                    private_to_public.append(rec.workload_id)
                elif rec.origin_cluster == 1 and rec.destination_cluster == 0:
                    # Public to Private
                    public_to_private.append(rec.workload_id)

        # If no migrations occurred, optionally log or save a note
        if not private_to_public and not public_to_private:
            logger.info(
                f"No migrations in history batch {history_batch_id}, no summaries to save"
            )
            return

        # Save summaries to MongoDB
        client = _get_mongo_client(mongodb_config)
        db = client[mongodb_config["database"]]
        collection = db[mongodb_config["collection"]]

        summaries_saved = 0

        if private_to_public:
            summary = _generate_summary_string(
                private_to_public, "private_to_public", history_batch_id
            )
            summary_doc = {
                "history_batch_id": history_batch_id,
                "type": "summary",
                "migration_direction": "private_to_public",
                "summary_text": summary,
                "timestamp": datetime.now()
            }
            collection.insert_one(summary_doc)
            summaries_saved += 1
            logger.debug(
                f"Saved private_to_public summary for batch {history_batch_id}"
            )

        if public_to_private:
            summary = _generate_summary_string(
                public_to_private, "public_to_private", history_batch_id
            )
            summary_doc = {
                "history_batch_id": history_batch_id,
                "type": "summary",
                "migration_direction": "public_to_private",
                "summary_text": summary,
                "timestamp": datetime.now()
            }
            collection.insert_one(summary_doc)
            summaries_saved += 1
            logger.debug(
                f"Saved public_to_private summary for batch {history_batch_id}"
            )

        client.close()

        logger.info(
            f"Saved {summaries_saved} migration summaries to history batch {history_batch_id}"
        )

    except Exception as e:
        logger.error(
            f"Failed to save history batch recommendations for batch {history_batch_id}: {e}"
        )
        raise


def get_historical_context_string(
    mongodb_config: Dict[str, Any], current_history_batch_id: int, lookback_count: int
) -> str:
    """
    Retrieve and format historical summaries as a string for LLM consumption.

    Args:
        mongodb_config: Dictionary with host, port, database, collection
        current_history_batch_id: The current history batch ID
        lookback_count: Number of past batches to retrieve

    Returns:
        Formatted string with historical context, or empty string if no history

    Example output:
        Previous Migration History (last 3 batches):
        - History Batch 2: Workloads app-1, app-2 were migrated from private to public
        - History Batch 3: Workloads worker-7 was migrated from private to public
    """
    try:
        # Handle edge cases
        if current_history_batch_id <= 1:
            logger.debug("No historical context available (first batch or batch 0)")
            return ""

        # Calculate batch range: [max(1, current - lookback), current - 1]
        min_batch = max(1, current_history_batch_id - lookback_count)
        max_batch = current_history_batch_id - 1

        if max_batch < min_batch:
            logger.debug("Invalid batch range for historical context")
            return ""

        # Query MongoDB
        client = _get_mongo_client(mongodb_config)
        db = client[mongodb_config["database"]]
        collection = db[mongodb_config["collection"]]

        # Find summary documents in batch range, sorted by batch ID
        cursor = collection.find(
            {
                "type": "summary",
                "history_batch_id": {"$gte": min_batch, "$lte": max_batch}
            }
        ).sort("history_batch_id", ASCENDING)

        documents = list(cursor)
        client.close()

        if not documents:
            logger.debug(
                f"No historical summaries found for batches {min_batch}-{max_batch}"
            )
            return ""

        # Format as multi-line string
        lines = [f"Previous Migration History (last {lookback_count} batches):"]
        for doc in documents:
            batch_id = doc["history_batch_id"]
            summary_text = doc["summary_text"]
            lines.append(f"- History Batch {batch_id}: {summary_text}")

        formatted_context = "\n".join(lines)

        logger.debug(
            f"Retrieved {len(documents)} historical summaries for batches {min_batch}-{max_batch}"
        )

        return formatted_context

    except Exception as e:
        logger.error(f"Failed to get historical context string: {e}")
        return ""
