"""Tick processing logic for the simulation engine."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from random import Random

from enterprise_sim.orchestrator.agent_pool import AgentPool
from enterprise_sim.orchestrator.sim_config import TickSummary, WorldConfig
from enterprise_sim.orchestrator.world_db import get_connection


class TickProcessor:
    """Processes one simulation tick across all 4 phases.

    Phases 1 (customer) and 3 (employee) run agent LLM calls in parallel
    using ThreadPoolExecutor. DB writes are done sequentially in the main
    thread after all LLM calls complete (for customer phase) or handled
    by the agents themselves via CLI tools inside Docker (for employee phase,
    with busy_timeout handling lock contention).
    """

    def __init__(self, pool: AgentPool, db_path: Path, config: WorldConfig, rng: Random):
        self.pool = pool
        self.db_path = db_path
        self.config = config
        self.rng = rng
        self.max_workers = max(1, min(4, len(pool.customers) + len(pool.employees)))

    def process(self, tick: int, sim_time: datetime) -> TickSummary:
        summary = TickSummary(tick=tick, sim_time=sim_time.strftime("%I:%M %p"))

        # Phase 1: Customer responses to agent replies + new ticket generation
        self._customer_phase(tick, sim_time, summary)

        # Phase 2: Route unassigned tickets to employees
        self._routing_phase(tick, sim_time, summary)

        # Phase 3: Employee actions on assigned tickets
        self._employee_phase(tick, sim_time, summary)

        # Phase 4: Manager handles escalations
        self._manager_phase(tick, sim_time, summary)

        return summary

    # ------------------------------------------------------------------
    # Phase 1: Customer phase (parallel LLM calls, sequential DB writes)
    # ------------------------------------------------------------------

    def _customer_phase(self, tick: int, sim_time: datetime, summary: TickSummary) -> None:
        conn = get_connection(self.db_path)
        try:
            # --- Gather work items from DB ---
            respond_work = []  # (agent_id, agent, ticket_id, agent_message)
            file_work = []     # (agent_id, agent)

            for agent_id, agent in self.pool.customers.items():
                # Check for tickets needing customer response
                tickets = conn.execute(
                    "SELECT id FROM tickets WHERE customer_id = ? AND status IN ('open', 'in_progress')",
                    (agent_id,),
                ).fetchall()
                for ticket_row in tickets:
                    tid = ticket_row["id"]
                    last_msg = conn.execute(
                        "SELECT sender_role, content FROM ticket_messages WHERE ticket_id = ? ORDER BY id DESC LIMIT 1",
                        (tid,),
                    ).fetchone()
                    if last_msg and last_msg["sender_role"] == "agent":
                        respond_work.append((agent_id, agent, tid, last_msg["content"]))

                # Check if customer might file a new ticket
                open_count = conn.execute(
                    "SELECT COUNT(*) FROM tickets WHERE customer_id = ? AND status IN ('open', 'in_progress', 'escalated')",
                    (agent_id,),
                ).fetchone()[0]
                if open_count == 0 and self.rng.random() <= self.config.ticket_probability:
                    file_work.append((agent_id, agent))

            # --- Parallel LLM calls for customer responses ---
            respond_results = []
            if respond_work:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {}
                    for agent_id, agent, tid, msg in respond_work:
                        def _do_respond(a=agent, m=msg):
                            return a.respond("send_reply", {"message": m})
                        futures[executor.submit(_do_respond)] = (agent_id, tid)

                    for future in as_completed(futures):
                        agent_id, tid = futures[future]
                        try:
                            response = future.result()
                            respond_results.append((agent_id, tid, response))
                        except Exception as e:
                            print(f"[Tick {tick}] Customer {agent_id} respond error: {e}")
                            _log_event(conn, tick, "agent_error", agent_id, {"error": str(e), "phase": "customer_respond"})
                            conn.commit()
                            try:
                                self.pool.customers[agent_id].respawn()
                            except Exception:
                                pass

            # --- Write response results to DB sequentially ---
            for agent_id, tid, response in respond_results:
                if response.customer_message:
                    conn.execute(
                        "INSERT INTO ticket_messages (ticket_id, sender_id, sender_role, content, timestamp) VALUES (?, ?, 'customer', ?, ?)",
                        (tid, agent_id, response.customer_message, sim_time.isoformat()),
                    )
                    summary.customer_responses.append(tid)
                    _log_event(conn, tick, "customer_responded", agent_id, {"ticket_id": tid})
                if response.is_resolved:
                    conn.execute(
                        "UPDATE tickets SET status = 'resolved', resolved_at = ? WHERE id = ?",
                        (sim_time.isoformat(), tid),
                    )
                    summary.resolved_tickets.append(tid)
                    _log_event(conn, tick, "ticket_resolved", agent_id, {"ticket_id": tid})
                conn.commit()

            # --- Parallel LLM calls for new ticket filing ---
            file_results = []
            if file_work:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {}
                    for agent_id, agent in file_work:
                        def _do_file(a=agent):
                            raw = a.send_message(
                                "You're contacting customer support now. Describe your issue "
                                "as you would in a real support chat. Be natural and in character."
                            )
                            parsed = a._parse_response(raw)
                            return parsed.customer_message or raw.strip()
                        futures[executor.submit(_do_file)] = (agent_id,)

                    for future in as_completed(futures):
                        (agent_id,) = futures[future]
                        try:
                            message = future.result()
                            if message:
                                file_results.append((agent_id, message))
                        except Exception as e:
                            print(f"[Tick {tick}] Customer {agent_id} file-ticket error: {e}")
                            _log_event(conn, tick, "agent_error", agent_id, {"error": str(e), "phase": "customer_file"})
                            conn.commit()
                            try:
                                self.pool.customers[agent_id].respawn()
                            except Exception:
                                pass

            # --- Write new tickets to DB sequentially ---
            for agent_id, message in file_results:
                cursor = conn.execute(
                    "INSERT INTO tickets (customer_id, subject, status, priority, created_at) VALUES (?, ?, 'open', 'normal', ?)",
                    (agent_id, _extract_subject(message), sim_time.isoformat()),
                )
                tid = cursor.lastrowid
                conn.execute(
                    "INSERT INTO ticket_messages (ticket_id, sender_id, sender_role, content, timestamp) VALUES (?, ?, 'customer', ?, ?)",
                    (tid, agent_id, message, sim_time.isoformat()),
                )
                conn.commit()
                summary.new_tickets.append(tid)
                _log_event(conn, tick, "ticket_created", agent_id, {"ticket_id": tid, "subject": _extract_subject(message)})

        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Phase 2: Routing (no LLM calls, stays sequential)
    # ------------------------------------------------------------------

    def _routing_phase(self, tick: int, sim_time: datetime, summary: TickSummary) -> None:
        conn = get_connection(self.db_path)
        try:
            unassigned = conn.execute(
                "SELECT id FROM tickets WHERE status = 'open' AND assigned_agent IS NULL ORDER BY id"
            ).fetchall()

            if not unassigned:
                return

            employee_ids = list(self.pool.employees.keys())
            if not employee_ids:
                return

            for i, row in enumerate(unassigned):
                agent_id = employee_ids[i % len(employee_ids)]
                conn.execute(
                    "UPDATE tickets SET assigned_agent = ? WHERE id = ?",
                    (agent_id, row["id"]),
                )
                summary.assignments.append((row["id"], agent_id))
                _log_event(conn, tick, "ticket_assigned", agent_id, {"ticket_id": row["id"]})

            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Phase 3: Employee phase (parallel LLM calls)
    # ------------------------------------------------------------------

    def _employee_phase(self, tick: int, sim_time: datetime, summary: TickSummary) -> None:
        conn = get_connection(self.db_path)
        try:
            # Gather work — which employees have actionable tickets
            work = []  # (agent_id, agent, perception, ticket_ids)
            for agent_id, agent in self.pool.employees.items():
                if not agent.is_alive():
                    try:
                        agent.respawn()
                    except Exception as e:
                        print(f"[Tick {tick}] Employee {agent_id} respawn failed: {e}")
                        continue

                actionable = self._get_actionable_tickets(conn, agent_id)
                if not actionable:
                    continue
                perception = self._build_employee_perception(agent_id, actionable, sim_time)
                work.append((agent_id, agent, perception, [t["id"] for t in actionable]))

            # Parallel LLM calls — agents act autonomously via CLI tools in Docker
            # busy_timeout handles SQLite lock contention from concurrent container writes
            if work:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {}
                    for agent_id, agent, perception, ticket_ids in work:
                        def _do_act(a=agent, p=perception):
                            a.send_message(p)
                        futures[executor.submit(_do_act)] = (agent_id, ticket_ids)

                    for future in as_completed(futures):
                        agent_id, ticket_ids = futures[future]
                        try:
                            future.result()
                            summary.employee_actions += len(ticket_ids)
                            _log_event(conn, tick, "agent_acted", agent_id, {"tickets_handled": ticket_ids})
                        except Exception as e:
                            print(f"[Tick {tick}] Employee {agent_id} error: {e}")
                            _log_event(conn, tick, "agent_error", agent_id, {"error": str(e), "phase": "employee"})
                            try:
                                self.pool.employees[agent_id].respawn()
                            except Exception:
                                pass

            # Check if any tickets got escalated during this phase
            newly_escalated = conn.execute(
                "SELECT id FROM tickets WHERE status = 'escalated'"
            ).fetchall()
            for row in newly_escalated:
                if row["id"] not in summary.escalated_tickets:
                    summary.escalated_tickets.append(row["id"])

            conn.commit()
        finally:
            conn.close()

    def _get_actionable_tickets(self, conn, agent_id: str) -> list[dict]:
        """Find tickets assigned to this agent where the last message is from a customer."""
        tickets = conn.execute(
            """SELECT t.id, t.subject, t.status, t.customer_id
               FROM tickets t
               WHERE t.assigned_agent = ?
                 AND t.status IN ('open', 'in_progress')
               ORDER BY t.id""",
            (agent_id,),
        ).fetchall()

        actionable = []
        for t in tickets:
            last_msg = conn.execute(
                "SELECT sender_role, content FROM ticket_messages WHERE ticket_id = ? ORDER BY id DESC LIMIT 1",
                (t["id"],),
            ).fetchone()
            if last_msg and last_msg["sender_role"] == "customer":
                actionable.append({
                    "id": t["id"],
                    "subject": t["subject"],
                    "status": t["status"],
                    "customer_id": t["customer_id"],
                    "last_message": last_msg["content"],
                })

        return actionable

    def _build_employee_perception(self, agent_id: str, tickets: list[dict], sim_time: datetime) -> str:
        lines = [
            f"It is {sim_time.strftime('%I:%M %p')}. You have {len(tickets)} ticket(s) needing your attention:\n"
        ]
        for t in tickets:
            lines.append(f"--- Ticket #{t['id']} ---")
            lines.append(f"Customer: {t['customer_id']} | Subject: {t['subject']} | Status: {t['status']}")
            lines.append(f"Latest message from customer:\n{t['last_message']}")
            lines.append("")

        lines.append(
            "Handle each ticket. Use your tools (esim lookup-customer, esim check-order, "
            "esim send-reply, esim update-ticket) as needed. If a ticket requires escalation, "
            "use esim update-ticket to set status to escalated and esim send-msg to notify #escalations."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Phase 4: Manager phase (sequential — typically only 1 manager)
    # ------------------------------------------------------------------

    def _manager_phase(self, tick: int, sim_time: datetime, summary: TickSummary) -> None:
        conn = get_connection(self.db_path)
        try:
            escalated = conn.execute(
                "SELECT id, subject, customer_id, assigned_agent FROM tickets WHERE status = 'escalated'"
            ).fetchall()

            # Also check for new messages in #escalations
            escalation_msgs = conn.execute(
                "SELECT sender_id, content FROM channel_messages WHERE channel_id = '#escalations' ORDER BY id DESC LIMIT 5"
            ).fetchall()

            if not escalated and not escalation_msgs:
                return

            for agent_id, agent in self.pool.managers.items():
                try:
                    if not agent.is_alive():
                        agent.respawn()
                    perception = self._build_manager_perception(escalated, escalation_msgs, sim_time)
                    agent.send_message(perception)
                    summary.manager_actions += 1
                    _log_event(conn, tick, "manager_acted", agent_id, {
                        "escalated_tickets": [dict(r)["id"] for r in escalated],
                    })
                except Exception as e:
                    print(f"[Tick {tick}] Manager {agent_id} error: {e}")
                    _log_event(conn, tick, "agent_error", agent_id, {"error": str(e), "phase": "manager"})
                    conn.commit()
                    try:
                        agent.respawn()
                    except Exception:
                        pass

            conn.commit()
        finally:
            conn.close()

    def _build_manager_perception(self, escalated, escalation_msgs, sim_time: datetime) -> str:
        lines = [f"It is {sim_time.strftime('%I:%M %p')}. Manager check-in:\n"]

        if escalated:
            lines.append(f"ESCALATED TICKETS ({len(escalated)}):")
            for t in escalated:
                lines.append(f"  Ticket #{t['id']} | Customer: {t['customer_id']} | Subject: {t['subject']} | Assigned to: {t['assigned_agent']}")
            lines.append("")

        if escalation_msgs:
            lines.append("RECENT #ESCALATIONS MESSAGES:")
            for m in escalation_msgs:
                lines.append(f"  [{m['sender_id']}]: {m['content']}")
            lines.append("")

        lines.append(
            "Review the escalated tickets and messages. Use your tools to resolve issues: "
            "esim lookup-customer, esim check-order, esim send-reply, esim update-ticket, "
            "esim send-msg. You can approve refunds and override policies as needed."
        )
        return "\n".join(lines)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _extract_subject(message: str) -> str:
    """Extract a short subject from a customer message."""
    # Take the first sentence, truncated to 60 chars
    first_line = message.split("\n")[0].split(".")[0].strip()
    if len(first_line) > 60:
        return first_line[:57] + "..."
    return first_line or "Customer issue"


def _log_event(conn, tick: int, event_type: str, agent_id: str, details: dict) -> None:
    """Write a simulation event to the sim_events table."""
    conn.execute(
        "INSERT INTO sim_events (tick, event_type, agent_id, details) VALUES (?, ?, ?, ?)",
        (tick, event_type, agent_id, json.dumps(details)),
    )
