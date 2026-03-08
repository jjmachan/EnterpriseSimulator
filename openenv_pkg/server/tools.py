"""Direct DB tool functions for the customer support environment.

Ported from enterprise_sim/tools/employee_tools.py — same logic but as plain
functions (no Click, no subprocess) that take db_path and return JSON strings.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path


def get_connection(db_path: Path) -> sqlite3.Connection:
    """Get a SQLite connection with Row factory."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def lookup_customer(db_path: Path, customer_id: str = "", customer_name: str = "") -> str:
    """Look up a customer profile with order and ticket history summary."""
    if not customer_id and not customer_name:
        return json.dumps({"error": "Provide customer_id or customer_name"})

    conn = get_connection(db_path)
    try:
        if customer_id:
            row = conn.execute("SELECT * FROM customers WHERE id = ?", (customer_id,)).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM customers WHERE name LIKE ?", (f"%{customer_name}%",)
            ).fetchone()

        if not row:
            return json.dumps({"error": "Customer not found"})

        customer = dict(row)
        cid = customer["id"]

        order_count = conn.execute(
            "SELECT COUNT(*) FROM orders WHERE customer_id = ?", (cid,)
        ).fetchone()[0]
        ticket_count = conn.execute(
            "SELECT COUNT(*) FROM tickets WHERE customer_id = ?", (cid,)
        ).fetchone()[0]
        open_tickets = conn.execute(
            "SELECT COUNT(*) FROM tickets WHERE customer_id = ? AND status IN ('open', 'in_progress', 'escalated')",
            (cid,),
        ).fetchone()[0]

        customer["order_count"] = order_count
        customer["ticket_count"] = ticket_count
        customer["open_tickets"] = open_tickets
        return json.dumps(customer, indent=2, default=str)
    finally:
        conn.close()


def check_order(db_path: Path, order_id: str) -> str:
    """Get full order details including items, status, and shipping info."""
    conn = get_connection(db_path)
    try:
        order = conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
        if not order:
            return json.dumps({"error": f"Order {order_id} not found"})

        result = dict(order)

        items = conn.execute(
            """
            SELECT oi.quantity, oi.unit_price, p.name as product_name, p.id as product_id
            FROM order_items oi
            JOIN products p ON oi.product_id = p.id
            WHERE oi.order_id = ?
            """,
            (order_id,),
        ).fetchall()
        result["items"] = [dict(item) for item in items]

        customer = conn.execute(
            "SELECT name, email FROM customers WHERE id = ?", (result["customer_id"],)
        ).fetchone()
        if customer:
            result["customer_name"] = customer["name"]
            result["customer_email"] = customer["email"]

        return json.dumps(result, indent=2, default=str)
    finally:
        conn.close()


def send_reply(db_path: Path, ticket_id: int, message: str, agent_id: str = "student") -> str:
    """Send a reply to a customer on a ticket."""
    conn = get_connection(db_path)
    try:
        ticket = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        if not ticket:
            return json.dumps({"error": f"Ticket {ticket_id} not found"})

        conn.execute(
            "INSERT INTO ticket_messages (ticket_id, sender_id, sender_role, content) VALUES (?, ?, 'agent', ?)",
            (ticket_id, agent_id, message),
        )
        if ticket["status"] == "open":
            conn.execute(
                "UPDATE tickets SET status = 'in_progress' WHERE id = ?", (ticket_id,)
            )
        conn.commit()

        return json.dumps({"status": "sent", "ticket_id": ticket_id, "message_length": len(message)})
    finally:
        conn.close()


def update_ticket(db_path: Path, ticket_id: int, status: str = "", notes: str = "") -> str:
    """Update ticket status and/or add internal notes."""
    conn = get_connection(db_path)
    try:
        ticket = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        if not ticket:
            return json.dumps({"error": f"Ticket {ticket_id} not found"})

        if status:
            if status in ("resolved", "closed"):
                conn.execute(
                    "UPDATE tickets SET status = ?, resolved_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (status, ticket_id),
                )
            else:
                conn.execute(
                    "UPDATE tickets SET status = ? WHERE id = ?", (status, ticket_id)
                )

        if notes:
            conn.execute(
                "INSERT INTO ticket_messages (ticket_id, sender_id, sender_role, content) VALUES (?, ?, 'system', ?)",
                (ticket_id, "system", f"[Internal Note] {notes}"),
            )

        conn.commit()
        return json.dumps({
            "status": "updated",
            "ticket_id": ticket_id,
            "new_status": status or dict(ticket)["status"],
        })
    finally:
        conn.close()
