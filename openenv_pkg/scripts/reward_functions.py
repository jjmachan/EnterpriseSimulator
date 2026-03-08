"""Reward functions for GRPO training.

These are used during GRPO training (not during data collection).
Each function scores model-generated completions against ground truth.

Signature: func(completions, answer, **kwargs) -> list[float]
  - completions: list of [{"role": "assistant", "content": "..."}]
  - answer: list of JSON strings with ground truth metadata
"""

import json
import re

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>", re.DOTALL
)
PARAM_RE = re.compile(r"<parameter=(\w+)>(.*?)</parameter>", re.DOTALL)

VALID_TOOLS = {"lookup_customer", "check_order", "send_reply", "update_ticket"}


def _get_text(completion):
    """Extract text from completion (handles both list and string formats)."""
    if isinstance(completion, list):
        return completion[0]["content"] if completion else ""
    return str(completion)


def _parse_answer(ans):
    """Parse the answer JSON string."""
    if isinstance(ans, str):
        try:
            return json.loads(ans)
        except json.JSONDecodeError:
            return {}
    return ans if isinstance(ans, dict) else {}


def format_reward(completions, answer, **kwargs) -> list[float]:
    """Does the completion contain a valid XML tool call?

    +1.0  valid <tool_call> XML with function and parameters
    +0.25 has some structure but not fully valid
    -1.0  empty or no tool call at all
    """
    rewards = []
    for completion in completions:
        text = _get_text(completion)
        if not text.strip():
            rewards.append(-1.0)
            continue
        match = TOOL_CALL_RE.search(text)
        if match:
            # Check if it has at least one parameter
            params = PARAM_RE.findall(match.group(2))
            if params:
                rewards.append(1.0)
            else:
                rewards.append(0.5)  # Tool call but no params
        elif "<tool_call>" in text or "<function=" in text:
            rewards.append(0.25)  # Partial format
        else:
            rewards.append(-1.0)
    return rewards


def tool_validity_reward(completions, answer, **kwargs) -> list[float]:
    """Is the tool name one of the valid tools?

    +1.0  valid tool name
    -0.5  parseable but invalid tool name
     0.0  no tool call found
    """
    rewards = []
    for completion, ans in zip(completions, answer):
        text = _get_text(completion)
        ans_data = _parse_answer(ans)
        valid = set(ans_data.get("valid_tools", VALID_TOOLS))

        match = TOOL_CALL_RE.search(text)
        if match:
            tool_name = match.group(1).strip()
            if tool_name in valid:
                rewards.append(1.0)
            else:
                rewards.append(-0.5)
        else:
            rewards.append(0.0)
    return rewards


def action_quality_reward(completions, answer, **kwargs) -> list[float]:
    """Heuristic quality score based on ground truths and reasoning.

    +0.3 per ground truth value referenced in the completion
    +0.2 for having reasoning text before the tool call
    Capped at 1.0
    """
    rewards = []
    for completion, ans in zip(completions, answer):
        text = _get_text(completion)
        ans_data = _parse_answer(ans)
        ground_truths = ans_data.get("ground_truth_values", [])

        score = 0.0

        # Check ground truth references
        for gt in ground_truths:
            if gt and str(gt).lower() in text.lower():
                score += 0.3

        # Check for reasoning before tool call
        tool_call_pos = text.find("<tool_call>")
        if tool_call_pos > 30:
            score += 0.2

        rewards.append(min(1.0, score))
    return rewards


def episode_reward_func(completions, answer, **kwargs) -> list[float]:
    """Pre-computed environment reward, mapped to [-1, 1].

    The episode reward is in [0, 1]. Map to [-1, 1] for GRPO.
    """
    rewards = []
    for ans in answer:
        ans_data = _parse_answer(ans)
        r = ans_data.get("episode_reward", 0.0)
        rewards.append(2.0 * r - 1.0)
    return rewards
