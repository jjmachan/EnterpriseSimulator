"""Employee CLI tools for interacting with the world database."""

import json

import click

from enterprise_sim.orchestrator.world_db import get_connection


def _json_output(data: dict | list) -> None:
    click.echo(json.dumps(data, indent=2, default=str))


@click.command("lookup-customer")
@click.option("--id", "customer_id", help="Customer ID")
@click.option("--name", "customer_name", help="Customer name (partial match)")
def lookup_customer(customer_id: str | None, customer_name: str | None):
    """Look up a customer profile with order and ticket history summary."""
    if not customer_id and not customer_name:
        _json_output({"error": "Provide --id or --name"})
        return

    conn = get_connection()
    try:
        if customer_id:
            row = conn.execute("SELECT * FROM customers WHERE id = ?", (customer_id,)).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM customers WHERE name LIKE ?", (f"%{customer_name}%",)
            ).fetchone()

        if not row:
            _json_output({"error": "Customer not found"})
            return

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
        _json_output(customer)
    finally:
        conn.close()


@click.command("check-order")
@click.option("--order-id", required=True, help="Order ID to look up")
def check_order(order_id: str):
    """Get full order details including items, status, and shipping info."""
    conn = get_connection()
    try:
        order = conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
        if not order:
            _json_output({"error": f"Order {order_id} not found"})
            return

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

        _json_output(result)
    finally:
        conn.close()


@click.command("send-reply")
@click.option("--ticket-id", required=True, type=int, help="Ticket ID")
@click.option("--message", required=True, help="Reply message content")
@click.option("--agent-id", default="employee_support_01", help="Agent sending the reply")
def send_reply(ticket_id: int, message: str, agent_id: str):
    """Send a reply to a customer on a ticket."""
    conn = get_connection()
    try:
        ticket = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        if not ticket:
            _json_output({"error": f"Ticket {ticket_id} not found"})
            return

        conn.execute(
            "INSERT INTO ticket_messages (ticket_id, sender_id, sender_role, content) VALUES (?, ?, 'agent', ?)",
            (ticket_id, agent_id, message),
        )
        # Auto-update ticket to in_progress if it was open
        if ticket["status"] == "open":
            conn.execute(
                "UPDATE tickets SET status = 'in_progress' WHERE id = ?", (ticket_id,)
            )
        conn.commit()

        _json_output({"status": "sent", "ticket_id": ticket_id, "message_length": len(message)})
    finally:
        conn.close()


@click.command("update-ticket")
@click.option("--ticket-id", required=True, type=int, help="Ticket ID")
@click.option(
    "--status",
    type=click.Choice(["open", "in_progress", "escalated", "resolved", "closed"]),
    help="New ticket status",
)
@click.option("--notes", help="Internal notes to add")
def update_ticket(ticket_id: int, status: str | None, notes: str | None):
    """Update ticket status and/or add internal notes."""
    conn = get_connection()
    try:
        ticket = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        if not ticket:
            _json_output({"error": f"Ticket {ticket_id} not found"})
            return

        if status:
            updates = {"status": status}
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
        _json_output({
            "status": "updated",
            "ticket_id": ticket_id,
            "new_status": status or ticket["status"],
        })
    finally:
        conn.close()
