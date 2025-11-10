"""
Recommendation History Database Module

Handles SQLite database operations for storing and retrieving
historical recommendation summaries.
"""

import sqlite3
import logging
import os
from typing import List, Optional
from datetime import datetime
from .ai_config import WorkloadRecommendation

logger = logging.getLogger(__name__)


def init_history_database(db_path: str) -> None:
    """
    Initialize SQLite database and create tables if they don't exist.

    Args:
        db_path: Path to the SQLite database file
    """
    try:
        # Ensure directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            logger.info(f"Created directory for history database: {db_dir}")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create history_batches table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS history_batches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                history_batch_id INTEGER NOT NULL UNIQUE,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create history_summaries table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS history_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                history_batch_id INTEGER NOT NULL,
                migration_direction TEXT NOT NULL,
                summary_text TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (history_batch_id) REFERENCES history_batches(history_batch_id)
            )
        """
        )

        # Create index for efficient queries
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_history_batch_id 
            ON history_summaries(history_batch_id)
        """
        )

        conn.commit()
        conn.close()

        logger.info(f"History database initialized successfully: {db_path}")

    except Exception as e:
        logger.error(f"Failed to initialize history database: {e}")
        raise


def get_latest_history_batch_id(db_path: str) -> int:
    """
    Get the latest history_batch_id from the database.

    Args:
        db_path: Path to the SQLite database file

    Returns:
        Latest history_batch_id, or 0 if no records exist
    """
    try:
        # Initialize database if it doesn't exist
        if not os.path.exists(db_path):
            init_history_database(db_path)
            return 0

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT MAX(history_batch_id) FROM history_batches
        """
        )

        result = cursor.fetchone()
        conn.close()

        if result and result[0] is not None:
            return int(result[0])
        return 0

    except Exception as e:
        logger.error(f"Failed to get latest history batch ID: {e}")
        return 0


def insert_history_batch(db_path: str, history_batch_id: int) -> None:
    """
    Insert a new history batch record.

    Args:
        db_path: Path to the SQLite database file
        history_batch_id: The history batch ID to insert
    """
    try:
        # Initialize database if it doesn't exist
        if not os.path.exists(db_path):
            init_history_database(db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO history_batches (history_batch_id, timestamp)
            VALUES (?, ?)
        """,
            (history_batch_id, datetime.now()),
        )

        conn.commit()
        conn.close()

        logger.debug(f"Inserted history batch {history_batch_id}")

    except sqlite3.IntegrityError:
        logger.warning(
            f"History batch {history_batch_id} already exists (duplicate insert)"
        )
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
    db_path: str,
    history_batch_id: int,
    recommendations: List[WorkloadRecommendation],
) -> None:
    """
    Save recommendations to the history database.

    Only migrated workloads (where origin_cluster != destination_cluster) are saved.
    Workloads are grouped by migration direction and summarized.

    Args:
        db_path: Path to the SQLite database file
        history_batch_id: The history batch ID
        recommendations: List of WorkloadRecommendation objects
    """
    try:
        # Initialize database if it doesn't exist
        if not os.path.exists(db_path):
            init_history_database(db_path)

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

        # Save summaries to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        summaries_saved = 0

        if private_to_public:
            summary = _generate_summary_string(
                private_to_public, "private_to_public", history_batch_id
            )
            cursor.execute(
                """
                INSERT INTO history_summaries (history_batch_id, migration_direction, summary_text, timestamp)
                VALUES (?, ?, ?, ?)
            """,
                (history_batch_id, "private_to_public", summary, datetime.now()),
            )
            summaries_saved += 1
            logger.debug(
                f"Saved private_to_public summary for batch {history_batch_id}"
            )

        if public_to_private:
            summary = _generate_summary_string(
                public_to_private, "public_to_private", history_batch_id
            )
            cursor.execute(
                """
                INSERT INTO history_summaries (history_batch_id, migration_direction, summary_text, timestamp)
                VALUES (?, ?, ?, ?)
            """,
                (history_batch_id, "public_to_private", summary, datetime.now()),
            )
            summaries_saved += 1
            logger.debug(
                f"Saved public_to_private summary for batch {history_batch_id}"
            )

        conn.commit()
        conn.close()

        logger.info(
            f"Saved {summaries_saved} migration summaries to history batch {history_batch_id}"
        )

    except Exception as e:
        logger.error(
            f"Failed to save history batch recommendations for batch {history_batch_id}: {e}"
        )
        raise


def get_historical_context_string(
    db_path: str, current_history_batch_id: int, lookback_count: int
) -> str:
    """
    Retrieve and format historical summaries as a string for LLM consumption.

    Args:
        db_path: Path to the SQLite database file
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

        # Initialize database if it doesn't exist
        if not os.path.exists(db_path):
            logger.debug("History database does not exist, no context available")
            return ""

        # Calculate batch range: [max(1, current - lookback), current - 1]
        min_batch = max(1, current_history_batch_id - lookback_count)
        max_batch = current_history_batch_id - 1

        if max_batch < min_batch:
            logger.debug("Invalid batch range for historical context")
            return ""

        # Query database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT history_batch_id, summary_text
            FROM history_summaries
            WHERE history_batch_id >= ? AND history_batch_id <= ?
            ORDER BY history_batch_id ASC, id ASC
        """,
            (min_batch, max_batch),
        )

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            logger.debug(
                f"No historical summaries found for batches {min_batch}-{max_batch}"
            )
            return ""

        # Format as multi-line string
        lines = [f"Previous Migration History (last {lookback_count} batches):"]
        for batch_id, summary_text in rows:
            lines.append(f"- History Batch {batch_id}: {summary_text}")

        formatted_context = "\n".join(lines)

        logger.debug(
            f"Retrieved {len(rows)} historical summaries for batches {min_batch}-{max_batch}"
        )

        return formatted_context

    except Exception as e:
        logger.error(f"Failed to get historical context string: {e}")
        return ""
