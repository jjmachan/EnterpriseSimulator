"""World DB snapshot utility — create a self-contained copy for task execution."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path


def create_snapshot(source_db: Path, output_dir: Path) -> Path:
    """Copy world.db to output directory as a task-execution snapshot.

    The snapshot is a full copy of the simulation's world.db at the point
    tasks were mined. Trainee agents run against this snapshot so they see
    the same world state the rubrics were grounded on.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / "world_snapshot.db"

    # Use SQLite backup API for a consistent copy
    src_conn = sqlite3.connect(str(source_db))
    dst_conn = sqlite3.connect(str(dest))
    src_conn.backup(dst_conn)
    src_conn.close()
    dst_conn.close()

    return dest


def reset_snapshot_for_task(snapshot_db: Path, task_context: dict) -> None:
    """Reset a snapshot DB to the state needed for a specific task.

    Removes simulation-generated ticket messages beyond the initial customer
    message, so the trainee agent starts fresh with just the customer's opening.
    """
    ticket_id = task_context.get("ticket_id")
    if not ticket_id:
        return

    conn = sqlite3.connect(str(snapshot_db))
    conn.row_factory = sqlite3.Row

    # Keep only the first customer message for this ticket
    first_msg = conn.execute(
        "SELECT id FROM ticket_messages WHERE ticket_id = ? AND sender_role = 'customer' ORDER BY id LIMIT 1",
        (ticket_id,),
    ).fetchone()

    if first_msg:
        # Delete all messages after the first customer message
        conn.execute(
            "DELETE FROM ticket_messages WHERE ticket_id = ? AND id > ?",
            (ticket_id, first_msg["id"]),
        )
        # Reset ticket status to open
        conn.execute(
            "UPDATE tickets SET status = 'open', assigned_agent = NULL, resolved_at = NULL WHERE id = ?",
            (ticket_id,),
        )

    conn.commit()
    conn.close()
