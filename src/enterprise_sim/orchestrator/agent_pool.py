"""AgentPool — manages PiAgent lifecycle for simulation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from random import Random

from enterprise_sim.orchestrator.agent_manager import PiAgent
from enterprise_sim.orchestrator.sim_config import WorldConfig

AGENTS_DIR = Path(__file__).resolve().parent.parent / "agents"
SHARED_DIR = Path(__file__).resolve().parent.parent / "shared"


class AgentPool:
    """Spawns and manages all agent Docker containers for a simulation run."""

    def __init__(self, db_path: Path, config: WorldConfig):
        self.db_path = db_path
        self.config = config
        self.customers: dict[str, PiAgent] = {}
        self.employees: dict[str, PiAgent] = {}
        self.managers: dict[str, PiAgent] = {}

    def spawn_all(self) -> None:
        """Discover agents, create PiAgents, spawn Docker containers."""
        env = {}
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            env["OPENAI_API_KEY"] = api_key
        # Point agents at the simulation's output DB
        env["ENTERPRISE_SIM_DB_PATH"] = "/shared/world.db"

        for agent_dir in sorted(AGENTS_DIR.iterdir()):
            if not agent_dir.is_dir():
                continue

            detection = _detect_agent(agent_dir)
            if detection is None:
                continue

            config_data, agent_type = detection
            agent_id = agent_dir.name

            agent = PiAgent(
                agent_id,
                agent_dir,
                provider=self.config.provider,
                model=self.config.model,
                env=dict(env),
            )
            # Override the shared dir mount to point at the simulation's output DB directory
            agent._sim_db_dir = self.db_path.parent
            agent.timeout = self.config.agent_timeout_seconds

            if agent_type == "customer":
                self.customers[agent_id] = agent
            elif "manager" in config_data.get("role", "").lower():
                self.managers[agent_id] = agent
            else:
                self.employees[agent_id] = agent

        # Apply agent count limits
        if self.config.max_customers and len(self.customers) > self.config.max_customers:
            rng = Random(self.config.seed)
            selected = rng.sample(list(self.customers.keys()), self.config.max_customers)
            self.customers = {k: v for k, v in self.customers.items() if k in selected}

        if self.config.max_employees and len(self.employees) > self.config.max_employees:
            rng = Random(self.config.seed)
            selected = rng.sample(list(self.employees.keys()), self.config.max_employees)
            self.employees = {k: v for k, v in self.employees.items() if k in selected}

        # Spawn all containers
        for agent_id, agent in self.customers.items():
            print(f"  Spawning customer: {agent_id}")
            agent.spawn()

        for agent_id, agent in self.employees.items():
            print(f"  Spawning employee: {agent_id}")
            agent.spawn()

        for agent_id, agent in self.managers.items():
            print(f"  Spawning manager: {agent_id}")
            agent.spawn()

    def shutdown_all(self) -> None:
        """Gracefully shutdown all agent containers."""
        for agent in [*self.customers.values(), *self.employees.values(), *self.managers.values()]:
            agent.shutdown()


def _detect_agent(agent_dir: Path) -> tuple[dict, str] | None:
    """Load agent config and detect type. Returns (config, type) or None."""
    role_path = agent_dir / "role.json"
    persona_path = agent_dir / "persona.json"
    if role_path.exists():
        with open(role_path) as f:
            return json.load(f), "employee"
    if persona_path.exists():
        with open(persona_path) as f:
            return json.load(f), "customer"
    return None
