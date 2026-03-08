"""Configuration dataclasses for the simulation engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TicketPacing:
    """Time-aware, personality-driven ticket pacing configuration."""

    base_probability: float = 0.10
    rush_hour_multiplier: float = 2.5  # applied during 9-10 AM, 1-2 PM
    quiet_multiplier: float = 0.3  # applied outside business hours
    patience_modifier: bool = True  # impatient customers file more often
    max_active_tickets: int = 2  # allow concurrent tickets per customer


@dataclass
class WorldConfig:
    """Configuration for a simulation run."""

    num_ticks: int = 48
    tick_duration_minutes: int = 5
    ticket_probability: float = 0.15
    provider: str = "openai"
    model: str = "gpt-5-mini"
    seed: int | None = None
    output_dir: Path | None = None
    agent_timeout_seconds: int = 120
    pacing: TicketPacing = field(default_factory=TicketPacing)
    max_customers: int | None = None
    max_employees: int | None = None


@dataclass
class TickSummary:
    """Summary of what happened during one tick."""

    tick: int
    sim_time: str
    new_tickets: list[int] = field(default_factory=list)
    customer_responses: list[int] = field(default_factory=list)
    assignments: list[tuple[int, str]] = field(default_factory=list)
    employee_actions: int = 0
    manager_actions: int = 0
    resolved_tickets: list[int] = field(default_factory=list)
    escalated_tickets: list[int] = field(default_factory=list)
