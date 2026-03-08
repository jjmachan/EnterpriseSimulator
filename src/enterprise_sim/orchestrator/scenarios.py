"""Hardcoded customer scenarios for Phase 1 (before pi-mono integration)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enterprise_sim.orchestrator.reward import DELTAS


@dataclass
class ScenarioResponse:
    customer_message: str
    satisfaction_delta: float
    is_resolved: bool


@dataclass
class ScenarioConfig:
    customer_id: str
    customer_name: str
    order_id: str | None
    subject: str
    opening_message: str
    patience_level: float


class HardcodedScenario:
    """State machine-based scripted customer for testing.

    Tracks which phase the conversation is in and returns appropriate
    responses based on what tool the agent used.
    """

    def __init__(self, config: ScenarioConfig):
        self.config = config
        self.step_count = 0
        self.phase = "opening"  # opening -> info_provided -> resolution_offered -> resolved/abandoned
        self.max_steps = 10

    def respond(self, tool_name: str, tool_args: dict) -> ScenarioResponse:
        self.step_count += 1

        # Patience decay every step
        base_delta = DELTAS["patience_decay"]

        # Check for abandonment
        if self.step_count >= self.max_steps:
            return ScenarioResponse(
                customer_message="You know what, forget it. I'll just dispute the charge with my bank.",
                satisfaction_delta=-0.2,
                is_resolved=False,
            )

        if tool_name == "send_reply":
            return self._handle_reply(tool_args.get("message", ""), base_delta)

        if tool_name in ("lookup_customer", "check_order"):
            # Agent is gathering info — no customer message, small positive signal
            return ScenarioResponse(
                customer_message="",
                satisfaction_delta=base_delta + DELTAS["correct_tool"],
                is_resolved=False,
            )

        if tool_name == "update_ticket":
            status = tool_args.get("status", "")
            if status == "resolved":
                if self.phase == "resolution_offered":
                    return ScenarioResponse(
                        customer_message="",
                        satisfaction_delta=DELTAS["resolved"],
                        is_resolved=True,
                    )
            return ScenarioResponse(
                customer_message="",
                satisfaction_delta=base_delta,
                is_resolved=False,
            )

        # Unknown/wrong tool
        return ScenarioResponse(
            customer_message="",
            satisfaction_delta=base_delta + DELTAS["wrong_tool"],
            is_resolved=False,
        )

    def _handle_reply(self, message: str, base_delta: float) -> ScenarioResponse:
        msg_lower = message.lower()

        if self.phase == "opening":
            # Agent's first reply — did they acknowledge the issue?
            if any(w in msg_lower for w in ["sorry", "apologize", "understand", "look into"]):
                self.phase = "info_provided"
                return ScenarioResponse(
                    customer_message=f"Thank you. My order number is {self.config.order_id}. "
                    "I need this sorted out as soon as possible.",
                    satisfaction_delta=base_delta + DELTAS["acknowledged_frustration"],
                    is_resolved=False,
                )
            else:
                return ScenarioResponse(
                    customer_message="Are you even listening? I said I have a problem with my order.",
                    satisfaction_delta=base_delta + DELTAS["wrong_tool"],
                    is_resolved=False,
                )

        if self.phase == "info_provided":
            # Agent should have looked up the order and is now offering a solution
            if any(w in msg_lower for w in ["refund", "replacement", "discount", "expedite", "resolve"]):
                self.phase = "resolution_offered"
                return ScenarioResponse(
                    customer_message="Okay, that sounds reasonable. Please go ahead and process that.",
                    satisfaction_delta=base_delta + DELTAS["resolved"],
                    is_resolved=True,
                )
            elif any(w in msg_lower for w in ["check", "looking", "investigate"]):
                return ScenarioResponse(
                    customer_message="Alright, I'll wait. But please hurry.",
                    satisfaction_delta=base_delta,
                    is_resolved=False,
                )
            else:
                return ScenarioResponse(
                    customer_message="That doesn't help me. Can you actually fix this?",
                    satisfaction_delta=base_delta + DELTAS["wrong_tool"],
                    is_resolved=False,
                )

        if self.phase == "resolution_offered":
            return ScenarioResponse(
                customer_message="Thanks, I think we're good here.",
                satisfaction_delta=DELTAS["resolved"],
                is_resolved=True,
            )

        return ScenarioResponse(
            customer_message="I'm still waiting for a resolution.",
            satisfaction_delta=base_delta,
            is_resolved=False,
        )


# Pre-built scenarios
SCENARIOS = [
    ScenarioConfig(
        customer_id="customer_001",
        customer_name="Sarah Chen",
        order_id="ord_001",
        subject="Standing desk surface is scratched",
        opening_message="Hi, I received my ErgoDesk Pro last week and there's a large scratch across "
        "the entire surface. This was a $550 desk — I expect better quality. What are you going to do about this?",
        patience_level=0.4,
    ),
    ScenarioConfig(
        customer_id="customer_003",
        customer_name="Maria Lopez",
        order_id="ord_006",
        subject="Ergonomic chair hasn't arrived",
        opening_message="I ordered an ergonomic chair two weeks ago and it still hasn't arrived. "
        "The tracking hasn't updated in 5 days. I need this for my home office — I'm working from a dining chair.",
        patience_level=0.3,
    ),
    ScenarioConfig(
        customer_id="customer_010",
        customer_name="Tom Martinez",
        order_id="ord_025",
        subject="Wrong item received — again",
        opening_message="I ordered a single monitor arm and received a desk pad instead. This is the second time "
        "you've messed up my order. I want this fixed immediately and I want compensation for my time.",
        patience_level=0.25,
    ),
]
