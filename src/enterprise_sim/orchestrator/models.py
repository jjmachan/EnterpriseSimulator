"""Data models for the customer support environment."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SupportAction:
    tool_name: str
    tool_args: dict = field(default_factory=dict)


@dataclass
class SupportObservation:
    customer_message: str
    tool_result: str
    ticket_context: str
    internal_messages: str
    reward: float
    done: bool
    info: dict = field(default_factory=dict)
