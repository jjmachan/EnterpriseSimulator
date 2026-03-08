"""LLM-based customer agent using direct OpenAI API calls.

Replaces Docker/pi-mono PiAgent for the OpenEnv deployment. Reuses the same
system prompts and response parsing logic from agent_manager.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI


# Satisfaction delta constants (from reward.py)
PATIENCE_DECAY = -0.05


@dataclass
class CustomerResponse:
    message: str
    satisfaction_delta: float
    is_resolved: bool


@dataclass
class CustomerConfig:
    customer_id: str
    customer_name: str
    order_id: str | None
    subject: str
    opening_message: str
    patience_level: float


class CustomerAgent:
    """Generative customer agent powered by OpenAI chat completions.

    Maintains conversation history across the episode. Each call appends
    to the message list so the customer remembers prior interactions.
    """

    def __init__(
        self,
        agent_id: str,
        persona: dict,
        agent_dir: Path,
        model: str = "gpt-5-mini",
    ):
        self.agent_id = agent_id
        self.persona = persona
        self.agent_dir = agent_dir
        self.model = model
        self.client = OpenAI()
        self.messages: list[dict] = []
        self._is_resolved = False
        self.config: CustomerConfig | None = None

        # Build system prompt
        self._system_prompt = self._build_system_prompt()

    def _read_file(self, relative_path: str) -> str:
        path = self.agent_dir / relative_path
        if path.exists():
            return path.read_text().strip()
        return ""

    def _build_system_prompt(self) -> str:
        persona = self.persona

        about_me = self._read_file("life_context/about_me.md")
        current_issues = self._read_file("life_context/current_issues.md")
        recent_purchases = self._read_file("life_context/recent_purchases.md")

        traits = ", ".join(persona.get("personality_traits", []))
        style = persona.get("communication_style", "")
        patience = persona.get("patience_level", 0.5)

        return f"""You are {persona['name']}, a customer of an office furniture company chatting with customer support.

PERSONALITY: {traits}
COMMUNICATION STYLE: {style}
PATIENCE: {patience}/1.0 (lower means less patient)

BACKGROUND:
{about_me}

RECENT PURCHASES:
{recent_purchases}

CURRENT ISSUE:
{current_issues}

WHEN TO CONSIDER YOUR ISSUE RESOLVED:
- The agent has looked up your order/account and confirmed the details
- The agent has offered a concrete solution (replacement, refund, fix, discount, etc.)
- You feel the proposed solution is fair and addresses your concern
- You do NOT need to wait for the replacement to arrive or the refund to process — once the agent commits to a specific action, that counts as resolved
- If the agent only says vague things like "I'll look into it" without specifics, that is NOT resolved

If the support agent tries to resolve your issue without fully understanding it, without looking up your order details, or without properly investigating, express frustration and insist they investigate properly before jumping to conclusions.

RESPONSE FORMAT:
1. Write your reply (1-3 sentences, in character). React naturally — warm up if the agent is helpful, get frustrated if they are dismissive or slow.
2. At the END of every response, include exactly one XML tag:
   <satisfaction-delta>X</satisfaction-delta>
   where X is a number between -0.2 and +0.2 reflecting how this interaction step made you feel.
   Examples: +0.15 if genuinely helpful, +0.0 if neutral, -0.1 if unhelpful, -0.2 if terrible.
3. If the issue is resolved (agent committed to a concrete solution), ALSO include:
   <resolved>true</resolved>
   Don't wait multiple turns after they've offered a good solution.
4. Never break character. Never mention the satisfaction tag or resolved tag to the support agent."""

    def init_episode(self) -> str:
        """Initialize episode: get opening complaint from customer.

        Returns the opening message text.
        """
        self.messages = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": (
                    "You're contacting customer support now. Describe your issue "
                    "as you would in a real support chat. Be natural and in character."
                ),
            },
        ]

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=1.0,
        )

        raw = completion.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": raw})

        parsed = self._parse_response(raw)

        # Extract order_id and subject from current_issues.md
        issues_text = self._read_file("life_context/current_issues.md")
        order_match = re.search(r"Order:\s*(ord_\d+)", issues_text)
        order_id = order_match.group(1) if order_match else None
        subject_match = re.search(r"##\s+(.+)", issues_text)
        subject = subject_match.group(1).strip() if subject_match else "Customer issue"

        self.config = CustomerConfig(
            customer_id=self.agent_id,
            customer_name=self.persona["name"],
            order_id=order_id,
            subject=subject,
            opening_message=parsed.message,
            patience_level=self.persona.get("patience_level", 0.5),
        )

        return parsed.message

    def respond_to_reply(self, agent_message: str) -> CustomerResponse:
        """Customer responds to a support agent's reply."""
        prompt = f"""The support agent has replied to your ticket:
---
{agent_message}
---
Respond in character as the customer. Remember to include <satisfaction-delta>X</satisfaction-delta> at the end. If the agent has committed to a concrete solution (refund, replacement, fix, etc.) and you're satisfied with it, also include <resolved>true</resolved> — don't wait for the action to physically complete."""

        self.messages.append({"role": "user", "content": prompt})

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=1.0,
        )

        raw = completion.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": raw})

        return self._parse_response(raw)

    def respond_to_resolve(self) -> CustomerResponse:
        """Customer responds to their ticket being marked as resolved."""
        prompt = """The support agent has marked your ticket as resolved. How do you feel about the resolution? Were you satisfied with the outcome? Respond naturally in character. Include <satisfaction-delta>X</satisfaction-delta> and <resolved>true</resolved> if you agree it's resolved."""

        self.messages.append({"role": "user", "content": prompt})

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=1.0,
        )

        raw = completion.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": raw})

        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> CustomerResponse:
        """Extract satisfaction-delta and resolved XML tags from response."""
        # Extract satisfaction delta
        delta_match = re.search(
            r"<satisfaction-delta>([\+\-]?\d*\.?\d+)</satisfaction-delta>", raw
        )
        if delta_match:
            delta = float(delta_match.group(1))
            message = re.sub(
                r"\s*<satisfaction-delta>.*?</satisfaction-delta>\s*", "", raw
            ).strip()
        else:
            message = raw.strip()
            delta = self._heuristic_delta(message)

        # Extract resolved tag
        resolved_match = re.search(r"<resolved>true</resolved>", raw, re.IGNORECASE)
        if resolved_match:
            self._is_resolved = True
            message = re.sub(
                r"\s*<resolved>.*?</resolved>\s*", "", message
            ).strip()

        return CustomerResponse(
            message=message,
            satisfaction_delta=delta,
            is_resolved=self._is_resolved,
        )

    def _heuristic_delta(self, message: str) -> float:
        """Fallback keyword-based satisfaction inference."""
        lower = message.lower()
        if any(w in lower for w in ["thank", "great", "perfect", "appreciate"]):
            return 0.15
        if any(w in lower for w in ["okay", "alright", "fine", "i'll wait"]):
            return 0.0
        if any(w in lower for w in ["frustrated", "unacceptable", "ridiculous", "useless"]):
            return -0.15
        if any(w in lower for w in ["forget it", "dispute", "cancel", "done with"]):
            return -0.2
        return PATIENCE_DECAY
