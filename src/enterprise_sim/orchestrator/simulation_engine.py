"""SimulationEngine — Smallville-style multi-agent world simulator."""

from __future__ import annotations

import shutil
from datetime import datetime, timedelta
from pathlib import Path
from random import Random

from enterprise_sim.orchestrator.agent_pool import AgentPool
from enterprise_sim.orchestrator.sim_config import TickSummary, WorldConfig
from enterprise_sim.orchestrator.tick_processor import TickProcessor
from enterprise_sim.orchestrator.world_db import get_connection, init_db, seed_db


class SimulationEngine:
    """Runs a multi-agent world simulation to produce a populated world.db.

    Spawns customer, support, and manager agents in Docker containers.
    Runs a tick-based simulation loop where agents interact organically
    through the shared database. The engine mediates all communication.
    """

    def __init__(self, config: WorldConfig):
        self.config = config
        self.rng = Random(config.seed)
        self.db_path: Path | None = None
        self.pool: AgentPool | None = None

    def run(self) -> Path:
        """Run the full simulation. Returns path to the output world.db."""
        self._setup()
        try:
            start_time = datetime(2026, 3, 9, 9, 0, 0)  # Simulated 9:00 AM
            for tick in range(self.config.num_ticks):
                sim_time = start_time + timedelta(minutes=tick * self.config.tick_duration_minutes)
                summary = self._process_tick(tick, sim_time)
                self._log_tick(summary)
        finally:
            self._teardown()

        self._print_summary()
        return self.db_path

    def _setup(self) -> None:
        """Initialize output DB and spawn all agent containers."""
        # 1. Create output directory
        if self.config.output_dir:
            output_dir = self.config.output_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("output") / f"sim_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = output_dir / "world.db"

        print(f"[Sim] Output: {self.db_path}")

        # 2. Initialize and seed the DB
        init_db(self.db_path)
        seed_db(self.db_path)

        # 3. Add simulation tracking tables
        conn = get_connection(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sim_clock (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                current_tick INTEGER DEFAULT 0,
                sim_time TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sim_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tick INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                agent_id TEXT,
                details TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.execute(
            "INSERT OR REPLACE INTO sim_clock (id, current_tick, sim_time) VALUES (1, 0, ?)",
            (datetime(2026, 3, 9, 9, 0, 0).isoformat(),),
        )
        conn.commit()
        conn.close()

        # 4. Spawn agents
        self.pool = AgentPool(self.db_path, self.config)
        nc = len(self.pool.customers)  # counted after discovery but before spawn
        print(f"[Sim] Starting simulation: {self.config.num_ticks} ticks, ticket_prob={self.config.ticket_probability}")
        print(f"[Sim] Spawning agents...")
        self.pool.spawn_all()

        nc = len(self.pool.customers)
        ne = len(self.pool.employees)
        nm = len(self.pool.managers)
        print(f"[Sim] Agents ready: {nc} customers, {ne} support, {nm} managers")

    def _process_tick(self, tick: int, sim_time: datetime) -> TickSummary:
        """Execute one simulation tick."""
        # Update sim_clock
        conn = get_connection(self.db_path)
        conn.execute(
            "UPDATE sim_clock SET current_tick = ?, sim_time = ? WHERE id = 1",
            (tick, sim_time.isoformat()),
        )
        conn.commit()
        conn.close()

        processor = TickProcessor(self.pool, self.db_path, self.config, self.rng)
        return processor.process(tick, sim_time)

    def _log_tick(self, summary: TickSummary) -> None:
        """Print tick progress."""
        parts = [f"[Tick {summary.tick + 1:02d}] {summary.sim_time}"]

        if summary.new_tickets:
            parts.append(f"new tickets: {summary.new_tickets}")
        if summary.customer_responses:
            parts.append(f"customer responses on: {summary.customer_responses}")
        if summary.assignments:
            assigns = [f"#{tid}→{aid}" for tid, aid in summary.assignments]
            parts.append(f"assigned: {', '.join(assigns)}")
        if summary.employee_actions:
            parts.append(f"employee handled {summary.employee_actions} ticket(s)")
        if summary.manager_actions:
            parts.append(f"manager acted")
        if summary.resolved_tickets:
            parts.append(f"RESOLVED: {summary.resolved_tickets}")
        if summary.escalated_tickets:
            parts.append(f"ESCALATED: {summary.escalated_tickets}")

        if len(parts) == 1:
            parts.append("(quiet)")

        print(" | ".join(parts))

    def _teardown(self) -> None:
        """Shutdown all agent containers."""
        if self.pool:
            print("[Sim] Shutting down agents...")
            self.pool.shutdown_all()

    def _print_summary(self) -> None:
        """Print final simulation statistics."""
        conn = get_connection(self.db_path)
        try:
            # Count new data beyond seed (seed has 8 tickets, 11 messages)
            total_tickets = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
            total_messages = conn.execute("SELECT COUNT(*) FROM ticket_messages").fetchone()[0]
            resolved = conn.execute("SELECT COUNT(*) FROM tickets WHERE status = 'resolved'").fetchone()[0]
            escalated = conn.execute("SELECT COUNT(*) FROM tickets WHERE status = 'escalated'").fetchone()[0]
            channel_msgs = conn.execute("SELECT COUNT(*) FROM channel_messages").fetchone()[0]
            events = conn.execute("SELECT COUNT(*) FROM sim_events").fetchone()[0]

            print(f"\n[Sim] Simulation complete!")
            print(f"  Tickets: {total_tickets} total ({total_tickets - 8} new)")
            print(f"  Messages: {total_messages} ticket msgs, {channel_msgs} channel msgs")
            print(f"  Resolved: {resolved} | Escalated: {escalated}")
            print(f"  Events logged: {events}")
            print(f"  Output: {self.db_path}")
        finally:
            conn.close()
