"""Simulation quality analysis: ticket patterns, agent behavior, conversations."""

from __future__ import annotations

import json
from pathlib import Path

from enterprise_sim.orchestrator.world_db import get_connection

# Seed data baseline
SEED_TICKET_COUNT = 8
SEED_MESSAGE_COUNT = 11


def ticket_patterns(db_path: Path) -> dict:
    """Analyze ticket generation patterns from simulation."""
    conn = get_connection(db_path)
    try:
        # Check if sim_events table exists
        has_sim = _table_exists(conn, "sim_events")

        # Total tickets and new ones
        total = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
        new_count = total - SEED_TICKET_COUNT

        # Tickets per customer (all tickets)
        per_customer = conn.execute("""
            SELECT c.id, c.name, COUNT(t.id) as ticket_count
            FROM customers c
            LEFT JOIN tickets t ON c.id = t.customer_id
            GROUP BY c.id
            ORDER BY ticket_count DESC
        """).fetchall()
        per_customer_list = [
            {"id": r["id"], "name": r["name"], "tickets": r["ticket_count"]}
            for r in per_customer
        ]

        # Tickets by status
        by_status = {}
        rows = conn.execute(
            "SELECT status, COUNT(*) as cnt FROM tickets GROUP BY status"
        ).fetchall()
        for r in rows:
            by_status[r["status"]] = r["cnt"]

        # Tickets by priority
        by_priority = {}
        rows = conn.execute(
            "SELECT priority, COUNT(*) as cnt FROM tickets GROUP BY priority"
        ).fetchall()
        for r in rows:
            by_priority[r["priority"]] = r["cnt"]

        # Subject diversity (unique subjects)
        subjects = conn.execute("SELECT subject FROM tickets").fetchall()
        unique_subjects = len(set(r["subject"] for r in subjects))

        # Tickets per tick (from sim_events if available)
        per_tick = {}
        if has_sim:
            rows = conn.execute("""
                SELECT tick, COUNT(*) as cnt
                FROM sim_events
                WHERE event_type = 'ticket_created'
                GROUP BY tick
                ORDER BY tick
            """).fetchall()
            for r in rows:
                per_tick[r["tick"]] = r["cnt"]

        return {
            "total_tickets": total,
            "new_tickets": new_count,
            "per_customer": per_customer_list,
            "by_status": by_status,
            "by_priority": by_priority,
            "unique_subjects": unique_subjects,
            "per_tick": per_tick,
        }
    finally:
        conn.close()


def agent_behavior(db_path: Path) -> dict:
    """Analyze agent behavior from simulation traces."""
    conn = get_connection(db_path)
    try:
        if not _table_exists(conn, "sim_traces"):
            return {"available": False, "reason": "No sim_traces table (not a simulation output)"}

        total_traces = conn.execute("SELECT COUNT(*) FROM sim_traces").fetchone()[0]
        if total_traces == 0:
            return {"available": True, "total_traces": 0}

        # Per-agent stats
        per_agent = conn.execute("""
            SELECT agent_id, phase,
                   COUNT(*) as trace_count,
                   AVG(duration_ms) as avg_duration,
                   MIN(duration_ms) as min_duration,
                   MAX(duration_ms) as max_duration
            FROM sim_traces
            GROUP BY agent_id, phase
        """).fetchall()

        agents = {}
        for r in per_agent:
            aid = r["agent_id"]
            if aid not in agents:
                agents[aid] = {"phases": {}, "total_traces": 0, "total_duration_ms": 0}
            agents[aid]["phases"][r["phase"]] = {
                "count": r["trace_count"],
                "avg_duration_ms": round(r["avg_duration"] or 0),
                "min_duration_ms": r["min_duration"] or 0,
                "max_duration_ms": r["max_duration"] or 0,
            }
            agents[aid]["total_traces"] += r["trace_count"]

        # Tool usage across all traces
        tool_counts = {}
        rows = conn.execute("SELECT tool_calls FROM sim_traces WHERE tool_calls IS NOT NULL").fetchall()
        for r in rows:
            try:
                calls = json.loads(r["tool_calls"])
                if isinstance(calls, list):
                    for call in calls:
                        name = call.get("tool") or call.get("name") or "unknown"
                        tool_counts[name] = tool_counts.get(name, 0) + 1
            except (json.JSONDecodeError, TypeError):
                pass

        # By phase summary
        by_phase = conn.execute("""
            SELECT phase, COUNT(*) as cnt, AVG(duration_ms) as avg_dur
            FROM sim_traces
            GROUP BY phase
        """).fetchall()
        phase_summary = {
            r["phase"]: {"count": r["cnt"], "avg_duration_ms": round(r["avg_dur"] or 0)}
            for r in by_phase
        }

        return {
            "available": True,
            "total_traces": total_traces,
            "per_agent": agents,
            "tool_usage": tool_counts,
            "by_phase": phase_summary,
        }
    finally:
        conn.close()


def conversation_quality(db_path: Path) -> dict:
    """Analyze conversation quality: depth, length, turn patterns."""
    conn = get_connection(db_path)
    try:
        # Messages per ticket
        per_ticket = conn.execute("""
            SELECT t.id, t.subject, t.status, COUNT(tm.id) as msg_count,
                   AVG(LENGTH(tm.content)) as avg_msg_len
            FROM tickets t
            LEFT JOIN ticket_messages tm ON t.id = tm.ticket_id
            GROUP BY t.id
        """).fetchall()

        msg_counts = [r["msg_count"] for r in per_ticket]
        avg_lengths = [r["avg_msg_len"] for r in per_ticket if r["avg_msg_len"] is not None]

        # Categorize
        no_messages = sum(1 for c in msg_counts if c == 0)
        single_message = sum(1 for c in msg_counts if c == 1)
        multi_turn = sum(1 for c in msg_counts if c > 1)

        # Turn analysis: count alternating role switches per ticket
        turn_counts = []
        for row in per_ticket:
            tid = row["id"]
            messages = conn.execute(
                "SELECT sender_role FROM ticket_messages WHERE ticket_id = ? ORDER BY timestamp",
                (tid,)
            ).fetchall()
            if len(messages) <= 1:
                turn_counts.append(len(messages))
                continue
            turns = 1
            for i in range(1, len(messages)):
                if messages[i]["sender_role"] != messages[i - 1]["sender_role"]:
                    turns += 1
            turn_counts.append(turns)

        # Per-ticket detail
        per_ticket_detail = []
        for i, row in enumerate(per_ticket):
            per_ticket_detail.append({
                "ticket_id": row["id"],
                "subject": row["subject"],
                "status": row["status"],
                "message_count": row["msg_count"],
                "avg_message_length": round(row["avg_msg_len"] or 0),
                "turns": turn_counts[i] if i < len(turn_counts) else 0,
            })

        return {
            "total_messages": sum(msg_counts),
            "new_messages": max(0, sum(msg_counts) - SEED_MESSAGE_COUNT),
            "tickets_with_no_messages": no_messages,
            "tickets_with_single_message": single_message,
            "tickets_with_multi_turn": multi_turn,
            "avg_messages_per_ticket": round(sum(msg_counts) / len(msg_counts), 1) if msg_counts else 0,
            "avg_message_length": round(sum(avg_lengths) / len(avg_lengths)) if avg_lengths else 0,
            "avg_turns_per_ticket": round(sum(turn_counts) / len(turn_counts), 1) if turn_counts else 0,
            "per_ticket": per_ticket_detail,
        }
    finally:
        conn.close()


def resolution_metrics(db_path: Path) -> dict:
    """Analyze resolution and escalation rates."""
    conn = get_connection(db_path)
    try:
        total = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
        new_count = total - SEED_TICKET_COUNT

        # Status counts for new tickets only (id > 8)
        new_statuses = conn.execute("""
            SELECT status, COUNT(*) as cnt
            FROM tickets
            WHERE id > ?
            GROUP BY status
        """, (SEED_TICKET_COUNT,)).fetchall()
        new_by_status = {r["status"]: r["cnt"] for r in new_statuses}

        resolved_new = new_by_status.get("resolved", 0) + new_by_status.get("closed", 0)
        escalated_new = new_by_status.get("escalated", 0)
        open_new = new_by_status.get("open", 0) + new_by_status.get("in_progress", 0)

        # Time to resolution for resolved tickets
        resolved_times = conn.execute("""
            SELECT id,
                   julianday(resolved_at) - julianday(created_at) as days_to_resolve
            FROM tickets
            WHERE resolved_at IS NOT NULL AND id > ?
        """, (SEED_TICKET_COUNT,)).fetchall()

        resolution_days = [r["days_to_resolve"] for r in resolved_times if r["days_to_resolve"] is not None]

        # Assigned vs unassigned
        unassigned = conn.execute(
            "SELECT COUNT(*) FROM tickets WHERE assigned_agent IS NULL AND id > ?",
            (SEED_TICKET_COUNT,)
        ).fetchone()[0]

        return {
            "total_tickets": total,
            "new_tickets": new_count,
            "new_by_status": new_by_status,
            "resolution_rate": round(resolved_new / new_count * 100, 1) if new_count > 0 else 0,
            "escalation_rate": round(escalated_new / new_count * 100, 1) if new_count > 0 else 0,
            "open_rate": round(open_new / new_count * 100, 1) if new_count > 0 else 0,
            "unassigned_new": unassigned,
            "avg_days_to_resolve": round(sum(resolution_days) / len(resolution_days), 2) if resolution_days else None,
        }
    finally:
        conn.close()


def _table_exists(conn, table_name: str) -> bool:
    """Check if a table exists in the database."""
    row = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    ).fetchone()
    return row[0] > 0
