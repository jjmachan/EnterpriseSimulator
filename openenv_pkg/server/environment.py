"""CustomerSupportEnvironment — MCPEnvironment for enterprise customer support RL training."""

from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import CallToolAction
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field

from server.customer_agent import CustomerAgent
from server import tools as db_tools

# Paths relative to the openenv_pkg directory
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BASE_WORLD_DB = DATA_DIR / "world.db"
AGENTS_DIR = DATA_DIR / "agents"
TASKS_DIR = DATA_DIR / "tasks"

MAX_STEPS = 10


# --- Custom Observation with first-class fields ---
# (metadata is excluded by OpenEnv's serialize_observation, so we use real fields)

class SupportObservation(Observation):
    """Observation for the customer support environment."""
    customer_message: str = Field(default="", description="Customer's message or complaint")
    tool_result: str = Field(default="", description="Result from the last tool call")
    tool_name: str = Field(default="", description="Name of the last tool called")
    ticket_context: str = Field(default="", description="Current ticket info")
    ticket_id: int = Field(default=0, description="Active ticket ID")
    customer_id: str = Field(default="", description="Customer ID")
    satisfaction: float = Field(default=0.7, description="Current satisfaction score 0-1")
    satisfaction_delta: float = Field(default=0.0, description="Satisfaction change from last action")
    resolved: bool = Field(default=False, description="Whether the issue is resolved")
    step_count: int = Field(default=0, description="Steps taken in this episode")
    episode_id: str = Field(default="", description="Episode identifier")


# --- Reward computation (from reward.py) ---

class SatisfactionTracker:
    def __init__(self, baseline: float = 0.7):
        self.score = baseline
        self.baseline = baseline

    def update(self, delta: float) -> float:
        self.score = max(0.0, min(1.0, self.score + delta))
        return self.score

    @property
    def abandoned(self) -> bool:
        return self.score <= 0.0


def compute_reward(resolved: bool, satisfaction: float, steps: int) -> float:
    resolution = 1.0 if resolved else 0.0
    efficiency = max(0.0, 1.0 - 0.1 * max(0, steps - 5))
    return 0.55 * resolution + 0.30 * satisfaction + 0.15 * efficiency


# --- Environment ---

class CustomerSupportEnvironment(MCPEnvironment):
    """OpenEnv RL environment for customer support training.

    Exposes 4 MCP tools (lookup_customer, check_order, send_reply, update_ticket).
    Customer responses are generated via OpenAI API. Reward is computed from
    resolution, satisfaction, and efficiency.
    """

    def __init__(self):
        mcp = FastMCP("customer-support")

        # Register tools as FastMCP tools.
        # These closures capture `self` to access the episode DB path.
        @mcp.tool
        def lookup_customer(customer_id: str = "", customer_name: str = "") -> str:
            """Look up a customer profile with order and ticket history summary."""
            return db_tools.lookup_customer(self._episode_db, customer_id, customer_name)

        @mcp.tool
        def check_order(order_id: str) -> str:
            """Get full order details including items, status, and shipping info."""
            return db_tools.check_order(self._episode_db, order_id)

        @mcp.tool
        def send_reply(ticket_id: int, message: str) -> str:
            """Send a reply to a customer on a ticket. This will trigger a customer response."""
            return db_tools.send_reply(self._episode_db, ticket_id, message)

        @mcp.tool
        def update_ticket(ticket_id: int, status: str = "", notes: str = "") -> str:
            """Update ticket status and/or add internal notes."""
            return db_tools.update_ticket(self._episode_db, ticket_id, status, notes)

        super().__init__(mcp)

        # Episode state
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._episode_db: Path | None = None
        self._customer_agent: CustomerAgent | None = None
        self._tracker: SatisfactionTracker | None = None
        self._ticket_id: int | None = None
        self._done = False
        self._resolved = False
        self._last_tool_name: str | None = None
        self._last_tool_args: dict = {}

        # Discover customer agent directories
        self._agent_dirs = sorted(
            d for d in AGENTS_DIR.iterdir()
            if d.is_dir() and (d / "persona.json").exists()
        ) if AGENTS_DIR.exists() else []

        # Discover task files
        self._task_files = sorted(TASKS_DIR.glob("task_*.json")) if TASKS_DIR.exists() else []

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SupportObservation:
        """Start a new episode.

        Kwargs:
            task_id: Load a specific mined task (e.g., "task_001_order_tracking")
        """
        # Clean up previous episode
        if self._episode_db and self._episode_db.exists():
            self._episode_db.unlink(missing_ok=True)

        self._done = False
        self._resolved = False
        self._last_tool_name = None
        self._last_tool_args = {}

        ep_id = episode_id or str(uuid4())
        self._state = State(episode_id=ep_id, step_count=0)

        if seed is not None:
            random.seed(seed)

        # Copy base world.db to a temp file for this episode
        self._episode_db = Path(f"/tmp/esim_episode_{ep_id}.db")
        src_conn = sqlite3.connect(str(BASE_WORLD_DB))
        dst_conn = sqlite3.connect(str(self._episode_db))
        src_conn.backup(dst_conn)
        src_conn.close()
        dst_conn.close()

        # Load task or pick random customer
        task_id = kwargs.get("task_id")
        task_data = None
        if task_id:
            task_data = self._load_task(task_id)

        if task_data:
            customer_id = task_data["context"]["customer_id"]
            agent_dir = AGENTS_DIR / customer_id
            user_message = task_data["user_message"]
        else:
            agent_dir = random.choice(self._agent_dirs)
            customer_id = agent_dir.name
            user_message = None

        # Load persona and create customer agent
        persona_path = agent_dir / "persona.json"
        with open(persona_path) as f:
            persona = json.load(f)

        self._customer_agent = CustomerAgent(
            agent_id=customer_id,
            persona=persona,
            agent_dir=agent_dir,
        )

        # Get opening message
        if user_message:
            # Use the task's user_message directly but still init the agent
            self._customer_agent.init_episode()
            opening_message = user_message
        else:
            opening_message = self._customer_agent.init_episode()

        config = self._customer_agent.config
        self._tracker = SatisfactionTracker(
            baseline=config.patience_level if config else 0.7
        )

        # Create ticket in episode DB
        conn = db_tools.get_connection(self._episode_db)
        try:
            cursor = conn.execute(
                "INSERT INTO tickets (customer_id, order_id, subject, status, priority) VALUES (?, ?, ?, 'open', 'normal')",
                (customer_id, config.order_id if config else None, config.subject if config else "Customer issue"),
            )
            self._ticket_id = cursor.lastrowid
            conn.execute(
                "INSERT INTO ticket_messages (ticket_id, sender_id, sender_role, content) VALUES (?, ?, 'customer', ?)",
                (self._ticket_id, customer_id, opening_message),
            )
            conn.commit()
        finally:
            conn.close()

        ticket_context = (
            f"Ticket #{self._ticket_id} | Customer: {config.customer_name if config else customer_id} "
            f"| Subject: {config.subject if config else 'Customer issue'} | Status: open"
        )

        return SupportObservation(
            done=False,
            reward=0.0,
            customer_message=opening_message,
            ticket_context=ticket_context,
            ticket_id=self._ticket_id,
            customer_id=customer_id,
            satisfaction=self._tracker.score,
            episode_id=ep_id,
            step_count=0,
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Execute a step. MCP tool calls are dispatched by base class, then post-processed."""
        from openenv.core.env_server.mcp_types import ListToolsAction as _LTA

        # ListToolsAction is metadata-only, no episode logic
        if isinstance(action, _LTA):
            return super().step(action, timeout_s=timeout_s, **kwargs)

        if self._done:
            return SupportObservation(
                done=True,
                reward=0.0,
                customer_message="Episode is done. Call reset() to start a new episode.",
            )

        self._state.step_count += 1

        # Track which tool was called for post-processing
        if isinstance(action, CallToolAction):
            self._last_tool_name = action.tool_name
            self._last_tool_args = action.arguments

        # Let base class dispatch MCP tools
        obs = super().step(action, timeout_s=timeout_s, **kwargs)

        # Post-process: add customer response, satisfaction, reward
        return self._post_process(obs)

    def _post_process(self, obs: Observation) -> SupportObservation:
        """Enrich observation with customer response and reward after tool execution."""
        tool_name = self._last_tool_name or ""
        tool_args = self._last_tool_args
        customer_message = ""
        satisfaction_delta = 0.0

        if tool_name == "send_reply" and self._customer_agent:
            # Get customer response via LLM
            agent_message = tool_args.get("message", "")
            response = self._customer_agent.respond_to_reply(agent_message)
            customer_message = response.message
            satisfaction_delta = response.satisfaction_delta
            if response.is_resolved:
                self._resolved = True

            # Record customer response in DB
            if customer_message and self._ticket_id:
                conn = db_tools.get_connection(self._episode_db)
                try:
                    conn.execute(
                        "INSERT INTO ticket_messages (ticket_id, sender_id, sender_role, content) VALUES (?, ?, 'customer', ?)",
                        (self._ticket_id, self._customer_agent.agent_id, customer_message),
                    )
                    conn.commit()
                finally:
                    conn.close()

        elif tool_name == "update_ticket":
            status = tool_args.get("status", "")
            if status == "resolved" and self._customer_agent and not self._resolved:
                response = self._customer_agent.respond_to_resolve()
                customer_message = response.message
                satisfaction_delta = response.satisfaction_delta
                if response.is_resolved:
                    self._resolved = True
            elif status == "resolved":
                self._resolved = True

        elif tool_name in ("lookup_customer", "check_order"):
            # Info gathering — small positive signal
            satisfaction_delta = 0.1

        # Update satisfaction
        if self._tracker:
            self._tracker.update(satisfaction_delta)

        # Check done conditions
        if self._resolved or (self._tracker and self._tracker.abandoned) or self._state.step_count >= MAX_STEPS:
            self._done = True

        # Compute reward only at episode end
        reward = 0.0
        if self._done and self._tracker:
            reward = compute_reward(self._resolved, self._tracker.score, self._state.step_count)

        # Extract tool result from the MCP observation
        tool_result = ""
        if hasattr(obs, "result") and obs.result is not None:
            result = obs.result
            # Handle FastMCP CallToolResult
            if hasattr(result, "data"):
                tool_result = str(result.data)
            elif isinstance(result, dict) and "data" in result:
                tool_result = str(result["data"])
            else:
                tool_result = str(result)
        elif hasattr(obs, "metadata") and obs.metadata:
            tool_result = str(obs.metadata.get("result", ""))

        return SupportObservation(
            done=self._done,
            reward=reward,
            tool_name=tool_name,
            tool_result=tool_result,
            customer_message=customer_message,
            satisfaction=self._tracker.score if self._tracker else 0.0,
            satisfaction_delta=satisfaction_delta,
            resolved=self._resolved,
            step_count=self._state.step_count,
            episode_id=self._state.episode_id or "",
            ticket_id=self._ticket_id or 0,
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SupportObservation:
        """Handle non-MCP actions (should not be called in normal usage)."""
        return SupportObservation(
            done=False,
            reward=0.0,
            customer_message=f"Unknown action type: {type(action).__name__}. "
            "Use call_tool() with lookup_customer, check_order, send_reply, or update_ticket.",
        )

    def _load_task(self, task_id: str) -> dict | None:
        """Load a mined task JSON by ID."""
        for f in self._task_files:
            if task_id in f.name:
                with open(f) as fh:
                    return json.load(fh)
        return None

    @property
    def state(self) -> State:
        return self._state

    def close(self) -> None:
        """Clean up episode DB."""
        if self._episode_db and self._episode_db.exists():
            self._episode_db.unlink(missing_ok=True)
        super().close()
