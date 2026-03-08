"""World quality analysis: entity statistics, coherence checks, interconnectedness."""

from __future__ import annotations

import json
from pathlib import Path

from enterprise_sim.orchestrator.world_db import get_connection


# Known seed counts for baseline comparison
SEED_COUNTS = {
    "customers": 12,
    "products": 25,
    "orders": 40,
    "order_items": 40,
    "tickets": 8,
    "ticket_messages": 11,
    "knowledge_base": 18,
    "transactions": 18,
    "channels": 5,
    "channel_messages": 0,
    "callbacks": 0,
}

# Tables that may be added by the simulation engine (not in base schema)
SIM_TABLES = ["sim_clock", "sim_events", "sim_traces"]


def entity_statistics(db_path: Path) -> dict:
    """Count entities per table and compute relationship density."""
    conn = get_connection(db_path)
    try:
        # Count all core tables
        core_tables = [
            "customers", "products", "orders", "order_items",
            "tickets", "ticket_messages", "knowledge_base",
            "transactions", "callbacks", "channels", "channel_messages",
        ]
        counts = {}
        for table in core_tables:
            try:
                row = conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()
                counts[table] = row[0]
            except Exception:
                counts[table] = 0

        # Sim-specific tables (may not exist)
        sim_counts = {}
        for table in SIM_TABLES:
            try:
                row = conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()
                sim_counts[table] = row[0]
            except Exception:
                sim_counts[table] = 0

        total_entities = sum(counts.values())

        # New entities beyond seed
        new_entities = {}
        for table, count in counts.items():
            seed = SEED_COUNTS.get(table, 0)
            new_entities[table] = max(0, count - seed)

        # Relationship density
        density = {}

        # Orders per customer
        rows = conn.execute(
            "SELECT customer_id, COUNT(*) as cnt FROM orders GROUP BY customer_id"
        ).fetchall()
        if rows:
            vals = [r["cnt"] for r in rows]
            density["orders_per_customer"] = _stats(vals)

        # Items per order
        rows = conn.execute(
            "SELECT order_id, COUNT(*) as cnt FROM order_items GROUP BY order_id"
        ).fetchall()
        if rows:
            vals = [r["cnt"] for r in rows]
            density["items_per_order"] = _stats(vals)

        # Tickets per customer
        rows = conn.execute(
            "SELECT customer_id, COUNT(*) as cnt FROM tickets GROUP BY customer_id"
        ).fetchall()
        if rows:
            vals = [r["cnt"] for r in rows]
            density["tickets_per_customer"] = _stats(vals)

        # Messages per ticket
        rows = conn.execute(
            "SELECT ticket_id, COUNT(*) as cnt FROM ticket_messages GROUP BY ticket_id"
        ).fetchall()
        if rows:
            vals = [r["cnt"] for r in rows]
            density["messages_per_ticket"] = _stats(vals)

        return {
            "entity_counts": counts,
            "total_entities": total_entities,
            "new_entities": new_entities,
            "total_new": sum(new_entities.values()),
            "sim_counts": sim_counts,
            "relationship_density": density,
        }
    finally:
        conn.close()


def coherence_checks(db_path: Path) -> dict:
    """Check referential integrity, value ranges, and temporal coherence."""
    conn = get_connection(db_path)
    try:
        issues = []
        warnings = []

        # --- Referential integrity ---
        # Orders without valid customer
        orphan_orders = conn.execute(
            "SELECT o.id FROM orders o LEFT JOIN customers c ON o.customer_id = c.id WHERE c.id IS NULL"
        ).fetchall()
        if orphan_orders:
            issues.append(f"Orphan orders (no customer): {[r['id'] for r in orphan_orders]}")

        # Order items without valid order
        orphan_items = conn.execute(
            "SELECT oi.id FROM order_items oi LEFT JOIN orders o ON oi.order_id = o.id WHERE o.id IS NULL"
        ).fetchall()
        if orphan_items:
            issues.append(f"Orphan order_items (no order): {len(orphan_items)} records")

        # Order items without valid product
        orphan_prods = conn.execute(
            "SELECT oi.id FROM order_items oi LEFT JOIN products p ON oi.product_id = p.id WHERE p.id IS NULL"
        ).fetchall()
        if orphan_prods:
            issues.append(f"Orphan order_items (no product): {len(orphan_prods)} records")

        # Tickets without valid customer
        orphan_tickets = conn.execute(
            "SELECT t.id FROM tickets t LEFT JOIN customers c ON t.customer_id = c.id WHERE c.id IS NULL"
        ).fetchall()
        if orphan_tickets:
            issues.append(f"Orphan tickets (no customer): {[r['id'] for r in orphan_tickets]}")

        # Messages without valid ticket
        orphan_msgs = conn.execute(
            "SELECT tm.id FROM ticket_messages tm LEFT JOIN tickets t ON tm.ticket_id = t.id WHERE t.id IS NULL"
        ).fetchall()
        if orphan_msgs:
            issues.append(f"Orphan messages (no ticket): {len(orphan_msgs)} records")

        # --- Value ranges ---
        bad_prices = conn.execute(
            "SELECT id, price FROM products WHERE price <= 0"
        ).fetchall()
        if bad_prices:
            issues.append(f"Products with non-positive price: {[r['id'] for r in bad_prices]}")

        bad_stock = conn.execute(
            "SELECT id, stock_level FROM products WHERE stock_level < 0"
        ).fetchall()
        if bad_stock:
            issues.append(f"Products with negative stock: {[r['id'] for r in bad_stock]}")

        bad_satisfaction = conn.execute(
            "SELECT id, satisfaction_score FROM customers WHERE satisfaction_score < 0 OR satisfaction_score > 1"
        ).fetchall()
        if bad_satisfaction:
            issues.append(f"Customers with satisfaction out of [0,1]: {[r['id'] for r in bad_satisfaction]}")

        bad_patience = conn.execute(
            "SELECT id, patience_level FROM customers WHERE patience_level < 0 OR patience_level > 1"
        ).fetchall()
        if bad_patience:
            issues.append(f"Customers with patience out of [0,1]: {[r['id'] for r in bad_patience]}")

        # --- Temporal coherence ---
        # Tickets with order: ticket should be after order
        temporal_issues = conn.execute("""
            SELECT t.id as ticket_id, t.created_at as t_created, o.created_at as o_created
            FROM tickets t
            JOIN orders o ON t.order_id = o.id
            WHERE t.created_at < o.created_at
        """).fetchall()
        if temporal_issues:
            warnings.append(f"Tickets created before their order: {[r['ticket_id'] for r in temporal_issues]}")

        # Resolved tickets should have resolved_at
        missing_resolved = conn.execute(
            "SELECT id FROM tickets WHERE status = 'resolved' AND resolved_at IS NULL"
        ).fetchall()
        if missing_resolved:
            warnings.append(f"Resolved tickets without resolved_at: {[r['id'] for r in missing_resolved]}")

        # resolved_at should be after created_at
        bad_resolved = conn.execute(
            "SELECT id FROM tickets WHERE resolved_at IS NOT NULL AND resolved_at < created_at"
        ).fetchall()
        if bad_resolved:
            issues.append(f"Tickets resolved before creation: {[r['id'] for r in bad_resolved]}")

        # --- Status consistency ---
        # Returned orders should have refund transactions
        returned_no_refund = conn.execute("""
            SELECT o.id FROM orders o
            WHERE o.status = 'returned'
            AND NOT EXISTS (
                SELECT 1 FROM transactions tx WHERE tx.order_id = o.id AND tx.type = 'refund'
            )
        """).fetchall()
        if returned_no_refund:
            warnings.append(f"Returned orders without refund: {[r['id'] for r in returned_no_refund]}")

        return {
            "issues": issues,
            "warnings": warnings,
            "issue_count": len(issues),
            "warning_count": len(warnings),
            "passed": len(issues) == 0,
        }
    finally:
        conn.close()


def interconnectedness(db_path: Path) -> dict:
    """Measure entity interconnectedness per customer."""
    conn = get_connection(db_path)
    try:
        customers = conn.execute("SELECT id, name FROM customers").fetchall()
        per_customer = {}

        for cust in customers:
            cid = cust["id"]

            # Orders
            orders = conn.execute(
                "SELECT id FROM orders WHERE customer_id = ?", (cid,)
            ).fetchall()
            order_ids = [r["id"] for r in orders]

            # Products through orders
            products = set()
            for oid in order_ids:
                items = conn.execute(
                    "SELECT product_id FROM order_items WHERE order_id = ?", (oid,)
                ).fetchall()
                products.update(r["product_id"] for r in items)

            # Tickets
            tickets = conn.execute(
                "SELECT id FROM tickets WHERE customer_id = ?", (cid,)
            ).fetchall()
            ticket_ids = [r["id"] for r in tickets]

            # Messages
            msg_count = 0
            for tid in ticket_ids:
                cnt = conn.execute(
                    "SELECT COUNT(*) FROM ticket_messages WHERE ticket_id = ?", (tid,)
                ).fetchone()[0]
                msg_count += cnt

            # Transactions through orders
            txn_count = 0
            for oid in order_ids:
                cnt = conn.execute(
                    "SELECT COUNT(*) FROM transactions WHERE order_id = ?", (oid,)
                ).fetchone()[0]
                txn_count += cnt

            total_reach = len(order_ids) + len(products) + len(ticket_ids) + msg_count + txn_count

            per_customer[cid] = {
                "name": cust["name"],
                "orders": len(order_ids),
                "products": len(products),
                "tickets": len(ticket_ids),
                "messages": msg_count,
                "transactions": txn_count,
                "total_reach": total_reach,
            }

        reach_values = [v["total_reach"] for v in per_customer.values()]

        return {
            "per_customer": per_customer,
            "summary": _stats(reach_values) if reach_values else {},
            "total_unique_relationships": sum(reach_values),
        }
    finally:
        conn.close()


def _stats(vals: list[int | float]) -> dict:
    """Compute basic stats for a list of numbers."""
    if not vals:
        return {"mean": 0, "min": 0, "max": 0, "total": 0, "count": 0}
    return {
        "mean": round(sum(vals) / len(vals), 2),
        "min": min(vals),
        "max": max(vals),
        "total": sum(vals),
        "count": len(vals),
    }
