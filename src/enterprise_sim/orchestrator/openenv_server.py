"""CustomerSupportEnv — the OpenEnv environment for enterprise customer support."""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

from enterprise_sim.orchestrator.models import SupportAction, SupportObservation
from enterprise_sim.orchestrator.reward import SatisfactionTracker, compute_reward
from enterprise_sim.orchestrator.scenarios import (
    SCENARIOS,
    HardcodedScenario,
    ScenarioConfig,
)
from enterprise_sim.orchestrator.world_db import get_connection, get_db_path, init_db, seed_db


AVAILABLE_TOOLS = {"lookup_customer", "check_order", "send_reply", "update_ticket"}


class CustomerSupportEnv:
    """RL environment for customer support training.

    The Student (LLM being trained) calls reset() to start an episode,
    then repeatedly calls step() with SupportActions until done=True.
    """

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or get_db_path()
        init_db(self.db_path)
        seed_db(self.db_path)

        self.scenario: HardcodedScenario | None = None
        self.tracker: SatisfactionTracker | None = None
        self.current_ticket_id: int | None = None
        self.step_count = 0
        self.episode_id = 0
        self.done = False

    def reset(self, scenario_index: int | None = None) -> SupportObservation:
        """Start a new episode. Picks a scenario, creates a ticket, returns initial observation."""
        self.episode_id += 1
        self.step_count = 0
        self.done = False

        # Pick scenario
        if scenario_index is not None:
            config = SCENARIOS[scenario_index % len(SCENARIOS)]
        else:
            config = random.choice(SCENARIOS)

        self.scenario = HardcodedScenario(config)
        self.tracker = SatisfactionTracker(baseline=config.patience_level)

        # Create ticket in DB
        conn = get_connection(self.db_path)
        try:
            cursor = conn.execute(
                "INSERT INTO tickets (customer_id, order_id, subject, status, priority) VALUES (?, ?, ?, 'open', 'normal')",
                (config.customer_id, config.order_id, config.subject),
            )
            self.current_ticket_id = cursor.lastrowid
            conn.execute(
                "INSERT INTO ticket_messages (ticket_id, sender_id, sender_role, content) VALUES (?, ?, 'customer', ?)",
                (self.current_ticket_id, config.customer_id, config.opening_message),
            )
            conn.commit()
        finally:
            conn.close()

        ticket_context = (
            f"Ticket #{self.current_ticket_id} | Customer: {config.customer_name} "
            f"| Subject: {config.subject} | Status: open"
        )

        return SupportObservation(
            customer_message=config.opening_message,
            tool_result="",
            ticket_context=ticket_context,
            internal_messages="",
            reward=0.0,
            done=False,
            info={
                "episode_id": self.episode_id,
                "step_count": 0,
                "satisfaction": self.tracker.score,
                "customer_id": config.customer_id,
                "ticket_id": self.current_ticket_id,
            },
        )

    def step(self, action: SupportAction) -> SupportObservation:
        """Execute one step: run the tool, get customer response, compute reward."""
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self.step_count += 1
        tool_result = ""
        customer_message = ""

        # 1. Execute the tool
        if action.tool_name not in AVAILABLE_TOOLS:
            tool_result = json.dumps({"error": f"Unknown tool: {action.tool_name}. Available: {sorted(AVAILABLE_TOOLS)}"})
        else:
            tool_result = self._execute_tool(action.tool_name, action.tool_args)

        # 2. Get customer response from scenario
        response = self.scenario.respond(action.tool_name, action.tool_args)
        customer_message = response.customer_message
        self.tracker.update(response.satisfaction_delta)

        # 3. If customer replied, add to ticket thread
        if customer_message:
            conn = get_connection(self.db_path)
            try:
                conn.execute(
                    "INSERT INTO ticket_messages (ticket_id, sender_id, sender_role, content) VALUES (?, ?, 'customer', ?)",
                    (self.current_ticket_id, self.scenario.config.customer_id, customer_message),
                )
                conn.commit()
            finally:
                conn.close()

        # 4. Check done conditions
        resolved = response.is_resolved
        if resolved or self.tracker.abandoned or self.step_count >= self.scenario.max_steps:
            self.done = True

        # 5. Compute reward
        reward = compute_reward(resolved, self.tracker.score, self.step_count) if self.done else 0.0

        # 6. Build ticket context
        ticket_context = self._get_ticket_context()

        return SupportObservation(
            customer_message=customer_message,
            tool_result=tool_result,
            ticket_context=ticket_context,
            internal_messages="",
            reward=reward,
            done=self.done,
            info={
                "episode_id": self.episode_id,
                "step_count": self.step_count,
                "satisfaction": self.tracker.score,
                "resolved": resolved,
                "customer_id": self.scenario.config.customer_id,
                "ticket_id": self.current_ticket_id,
            },
        )

    def state(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "step_count": self.step_count,
            "customer_satisfaction": self.tracker.score if self.tracker else None,
            "ticket_id": self.current_ticket_id,
            "done": self.done,
        }

    def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        """Execute a CLI tool and return its output."""
        cmd_map = {
            "lookup_customer": "lookup-customer",
            "check_order": "check-order",
            "send_reply": "send-reply",
            "update_ticket": "update-ticket",
        }
        cli_name = cmd_map.get(tool_name, tool_name)

        args = [sys.executable, "-m", "enterprise_sim.tools.cli", cli_name]
        for key, value in tool_args.items():
            flag = f"--{key.replace('_', '-')}"
            args.extend([flag, str(value)])

        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(self.db_path.parent.parent.parent),
            )
            return result.stdout.strip() or result.stderr.strip()
        except subprocess.TimeoutExpired:
            return json.dumps({"error": "Tool execution timed out"})

    def _get_ticket_context(self) -> str:
        conn = get_connection(self.db_path)
        try:
            ticket = conn.execute(
                "SELECT * FROM tickets WHERE id = ?", (self.current_ticket_id,)
            ).fetchone()
            if not ticket:
                return ""
            return (
                f"Ticket #{ticket['id']} | Customer: {self.scenario.config.customer_name} "
                f"| Subject: {ticket['subject']} | Status: {ticket['status']}"
            )
        finally:
            conn.close()
