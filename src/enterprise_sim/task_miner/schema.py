"""Task schema definition and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class RubricCriterion:
    """A single rubric criterion for evaluating task completion."""
    criterion: str
    type: str  # "tool_use" | "correctness" | "constraint" | "format"
    weight: float
    ground_truth: str | None = None  # verifiable fact from DB


@dataclass
class Task:
    """A mined task ready for OpenEnv training."""
    id: str
    category: str  # "information_retrieval" | "communication" | "reasoning" | "multi_step"
    difficulty: str  # "easy" | "medium" | "hard"
    system_prompt: str
    user_message: str
    tools: list[str]
    rubric: list[RubricCriterion]
    context: dict = field(default_factory=dict)  # ticket_id, customer_id, order_id etc.
    reference_trajectory: dict | None = None  # from simulation traces

    def to_dict(self) -> dict:
        d = asdict(self)
        # Clean up None values in rubric ground_truth
        for r in d["rubric"]:
            if r["ground_truth"] is None:
                del r["ground_truth"]
        return d

    def save(self, output_dir: Path) -> Path:
        """Save task as JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{self.id}.json"
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: Path) -> Task:
        """Load task from JSON file."""
        with open(path) as f:
            data = json.load(f)
        rubric = [RubricCriterion(**r) for r in data.pop("rubric")]
        return cls(**data, rubric=rubric)


def validate_task(task: Task) -> list[str]:
    """Validate a task, returning a list of issues (empty = valid)."""
    issues = []

    if not task.id:
        issues.append("Missing task ID")
    if task.category not in ("information_retrieval", "communication", "reasoning", "multi_step"):
        issues.append(f"Invalid category: {task.category}")
    if task.difficulty not in ("easy", "medium", "hard"):
        issues.append(f"Invalid difficulty: {task.difficulty}")
    if not task.system_prompt:
        issues.append("Missing system prompt")
    if not task.user_message:
        issues.append("Missing user message")
    if not task.tools:
        issues.append("No tools specified")
    if not task.rubric:
        issues.append("No rubric criteria")

    total_weight = sum(r.weight for r in task.rubric)
    if abs(total_weight - 1.0) > 0.01:
        issues.append(f"Rubric weights sum to {total_weight:.2f}, expected 1.0")

    valid_types = {"tool_use", "correctness", "constraint", "format"}
    for r in task.rubric:
        if r.type not in valid_types:
            issues.append(f"Invalid rubric type: {r.type}")

    return issues
