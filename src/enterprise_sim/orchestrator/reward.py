"""Reward function for the customer support environment."""


class SatisfactionTracker:
    """Tracks customer satisfaction over the course of an episode."""

    def __init__(self, baseline: float = 0.7):
        self.score = baseline
        self.baseline = baseline

    def update(self, delta: float) -> float:
        self.score = max(0.0, min(1.0, self.score + delta))
        return self.score

    @property
    def abandoned(self) -> bool:
        return self.score <= 0.0


# Per-step satisfaction deltas
DELTAS = {
    "correct_tool": 0.1,
    "acknowledged_frustration": 0.15,
    "resolved": 0.2,
    "wrong_tool": -0.1,
    "asked_repeat_info": -0.15,
    "ignored_escalation": -0.2,
    "patience_decay": -0.05,
}


def compute_reward(resolved: bool, satisfaction: float, steps: int) -> float:
    """Compute final episode reward.

    Components:
        - resolution (55%): binary, did the customer's issue get resolved?
        - satisfaction (30%): final satisfaction score (0-1)
        - efficiency (15%): 1.0 if resolved in <=5 steps, decays 0.1 per extra step
    """
    resolution = 1.0 if resolved else 0.0
    efficiency = max(0.0, 1.0 - 0.1 * max(0, steps - 5))

    return 0.55 * resolution + 0.30 * satisfaction + 0.15 * efficiency
