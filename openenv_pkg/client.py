"""Client for the CustomerSupport OpenEnv environment."""

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation, ToolError
from openenv.core.env_server.types import Observation
from openenv.core.mcp_client import MCPToolClient


class SupportStepResult:
    """Rich step result that exposes all SupportObservation fields."""

    def __init__(self, raw: Dict[str, Any]):
        self._raw = raw
        obs = raw.get("observation", {})
        self.customer_message: str = obs.get("customer_message", "")
        self.tool_result: str = obs.get("tool_result", "")
        self.tool_name: str = obs.get("tool_name", "")
        self.ticket_context: str = obs.get("ticket_context", "")
        self.ticket_id: int = obs.get("ticket_id", 0)
        self.customer_id: str = obs.get("customer_id", "")
        self.satisfaction: float = obs.get("satisfaction", 0.0)
        self.satisfaction_delta: float = obs.get("satisfaction_delta", 0.0)
        self.resolved: bool = obs.get("resolved", False)
        self.step_count: int = obs.get("step_count", 0)
        self.episode_id: str = obs.get("episode_id", "")
        self.reward: float = raw.get("reward", 0.0)
        self.done: bool = raw.get("done", False)

    def __repr__(self) -> str:
        parts = [f"step={self.step_count}", f"satisfaction={self.satisfaction:.2f}"]
        if self.tool_name:
            parts.append(f"tool={self.tool_name}")
        if self.customer_message:
            msg = self.customer_message[:80] + ("..." if len(self.customer_message) > 80 else "")
            parts.append(f'customer="{msg}"')
        if self.done:
            parts.append(f"reward={self.reward:.3f}")
        return f"SupportStepResult({', '.join(parts)})"


class CustomerSupportEnv(MCPToolClient):
    """Client for interacting with the EnterpriseSim customer support environment.

    Tools available:
    - lookup_customer(customer_id, customer_name) — look up customer profile
    - check_order(order_id) — get full order details
    - send_reply(ticket_id, message) — send reply to customer (triggers LLM response)
    - update_ticket(ticket_id, status, notes) — update ticket status

    Example:
        >>> env = CustomerSupportEnv(base_url="http://localhost:8000")
        >>> with env:
        ...     obs = env.reset()
        ...     print(obs.customer_message)
        ...     tools = env.list_tools()
        ...     obs = env.call_tool("lookup_customer", customer_id="customer_002")
        ...     print(obs.tool_result)
        ...     obs = env.call_tool("send_reply", ticket_id=1, message="Hi!")
        ...     print(obs.customer_message)
    """

    def reset(self, **kwargs: Any) -> SupportStepResult:
        """Reset the environment and return the initial observation."""
        message = {"type": "reset", "data": kwargs}
        response = self._send_and_receive(message)
        return SupportStepResult(response.get("data", {}))

    def call_tool(self, name: str, **kwargs: Any) -> SupportStepResult:
        """Call a tool and return rich observation with customer response, satisfaction, etc."""
        action = CallToolAction(tool_name=name, arguments=kwargs)
        payload = self._step_payload(action)
        message = {"type": "step", "data": payload}
        response = self._send_and_receive(message)
        return SupportStepResult(response.get("data", {}))
