"""PiAgent — wraps pi-mono RPC to power generative agents (customers, employees, managers)."""

from __future__ import annotations

import json
import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from enterprise_sim.orchestrator.reward import DELTAS
from enterprise_sim.orchestrator.scenarios import ScenarioResponse


@dataclass
class AgentConfig:
    customer_id: str
    customer_name: str
    order_id: str | None
    subject: str
    opening_message: str
    patience_level: float


SHARED_DIR = Path(__file__).resolve().parent.parent / "shared"


class PiAgent:
    """Generative agent powered by pi-mono RPC.

    Each instance manages a pi-mono subprocess. The agent responds in character
    based on its persona/role, context files, and memory stored in its folder.
    Supports customer agents (persona.json) and employee agents (role.json).
    """

    DOCKER_IMAGE_CUSTOMER = "enterprise-sim-customer"
    DOCKER_IMAGE_EMPLOYEE = "enterprise-sim-employee"

    def __init__(
        self,
        agent_id: str,
        agent_dir: Path,
        provider: str = "openai",
        model: str = "gpt-5-mini",
        env: dict[str, str] | None = None,
    ):
        self.agent_id = agent_id
        self.agent_dir = agent_dir.resolve()
        self.provider = provider
        self.model = model
        self.env = env or {}
        self.max_steps = 10

        self._proc: subprocess.Popen | None = None
        self._is_resolved = False
        self.timeout: int = 120  # seconds; overridden by WorldConfig via AgentPool
        self.last_trace: dict | None = None

        # Detect agent type and load config file
        role_path = agent_dir / "role.json"
        persona_path = agent_dir / "persona.json"
        if role_path.exists():
            self.agent_type = "employee"
            with open(role_path) as f:
                self._persona = json.load(f)
        elif persona_path.exists():
            self.agent_type = "customer"
            with open(persona_path) as f:
                self._persona = json.load(f)
        else:
            raise FileNotFoundError(f"No persona.json or role.json found in {agent_dir}")

        # Config gets populated during init_episode()
        self.config: AgentConfig | None = None

    def spawn(self) -> None:
        """Start pi-mono inside a Docker container with the agent dir mounted."""
        system_prompt = self._build_system_prompt()
        docker_image = self._get_docker_image()
        extension_path = self._get_extension_path()

        cmd = [
            "docker", "run", "-i", "--rm",
            "-v", f"{self.agent_dir}:/agent",
        ]
        # Employee agents get shared dir + work_context mounts
        if self.agent_type == "employee":
            # In simulation mode, _sim_db_dir points to the output DB directory
            db_dir = getattr(self, "_sim_db_dir", None) or SHARED_DIR
            db_dir = Path(db_dir).resolve()
            work_context_dir = SHARED_DIR.resolve() / "work_context"
            cmd.extend([
                "-v", f"{db_dir}:/shared",
                "-v", f"{work_context_dir}:/work_context",
                "-e", "ENTERPRISE_SIM_DB_PATH=/shared/world.db",
            ])
        # Pass environment variables (API keys etc.)
        for key, value in self.env.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([
            docker_image,
            "--mode", "rpc",
            "--provider", self.provider,
            "--model", self.model,
            "--no-session",
            "--system-prompt", system_prompt,
            "--extension", extension_path,
        ])
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def init_episode(self) -> str:
        """Get opening message from the customer agent. Returns the opening complaint."""
        # Ask the customer to state their issue
        raw = self.send_message(
            "You're contacting customer support now. Describe your issue "
            "as you would in a real support chat. Be natural and in character."
        )
        parsed = self._parse_response(raw)

        # Read current_issues.md to extract order_id and subject
        issues_path = self.agent_dir / "life_context" / "current_issues.md"
        issues_text = issues_path.read_text() if issues_path.exists() else ""

        # Extract order ID from issues file
        order_match = re.search(r"Order:\s*(ord_\d+)", issues_text)
        order_id = order_match.group(1) if order_match else None

        # Extract subject from first ## heading
        subject_match = re.search(r"##\s+(.+)", issues_text)
        subject = subject_match.group(1).strip() if subject_match else "Customer issue"

        self.config = AgentConfig(
            customer_id=self.agent_id,
            customer_name=self._persona["name"],
            order_id=order_id,
            subject=subject,
            opening_message=parsed.customer_message,
            patience_level=self._persona.get("patience_level", 0.5),
        )

        return parsed.customer_message

    def send_message(self, content: str) -> str:
        """Send a prompt to pi-mono RPC, collect response until agent_end.

        pi-mono RPC protocol:
        - Send: {"type": "prompt", "message": "..."}
        - Receive: stream of events including turn_end, tool_execution_*, agent_end
        - Tool calls are reported via tool_execution_start/end events (not in turn_end content)
        - turn_end.message.content has text blocks (and thinking blocks)

        Uses a threaded reader with timeout to prevent indefinite hangs
        when the LLM API is unavailable or the model name is invalid.
        """
        if not self._proc or self._proc.poll() is not None:
            raise RuntimeError("pi-mono process is not running")

        cmd = json.dumps({"type": "prompt", "message": content}) + "\n"
        self._proc.stdin.write(cmd)
        self._proc.stdin.flush()

        start_time = time.monotonic()
        result_queue: queue.Queue[str] = queue.Queue()
        tool_calls = []

        def _read_loop():
            all_text_parts = []
            exit_reason = "unknown"
            try:
                while True:
                    line = self._proc.stdout.readline()
                    if not line:
                        exit_reason = "stdout_closed"
                        break

                    line = line.strip()
                    if not line:
                        continue

                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    event_type = event.get("type", "")

                    if event_type == "agent_error":
                        error_msg = event.get("error", event.get("message", "unknown error"))
                        print(f"[Agent {self.agent_id}] agent_error event: {str(error_msg)[:200]}")

                    if event_type == "tool_execution_end":
                        # Intercept mark_resolved tool calls
                        if event.get("toolName") == "mark_resolved" and not event.get("isError"):
                            self._is_resolved = True
                        tool_calls.append({
                            "tool": event.get("toolName", ""),
                            "error": event.get("isError", False),
                        })
                    elif event_type == "turn_end":
                        # Accumulate text from all turn_end events
                        message = event.get("message", {})
                        content_blocks = message.get("content", [])
                        for block in content_blocks:
                            if block.get("type") == "text":
                                all_text_parts.append(block.get("text", ""))
                    elif event_type == "agent_end":
                        exit_reason = "agent_end"
                        break
            except Exception as e:
                exit_reason = f"exception: {e}"
                print(f"[Agent {self.agent_id}] _read_loop exception: {e}")
            result_queue.put("\n".join(all_text_parts))

            # Debug: log empty responses with exit reason
            if not all_text_parts:
                stderr_snippet = ""
                try:
                    if self._proc and self._proc.stderr:
                        import select
                        # Non-blocking stderr read
                        import os
                        fd = self._proc.stderr.fileno()
                        os.set_blocking(fd, False)
                        try:
                            stderr_snippet = self._proc.stderr.read(2000) or ""
                        except (BlockingIOError, OSError):
                            pass
                        finally:
                            os.set_blocking(fd, True)
                except Exception:
                    pass
                print(f"[Agent {self.agent_id}] EMPTY response | exit_reason={exit_reason} | "
                      f"process_alive={self._proc and self._proc.poll() is None} | "
                      f"stderr={stderr_snippet[:500]}")

        reader = threading.Thread(target=_read_loop, daemon=True)
        reader.start()
        reader.join(timeout=self.timeout)

        if reader.is_alive():
            # Timed out — capture stderr for debugging, kill process
            stderr_output = ""
            try:
                self._proc.kill()
                _, stderr_output = self._proc.communicate(timeout=5)
            except Exception:
                pass
            self._proc = None
            print(f"[Agent {self.agent_id}] TIMEOUT after {self.timeout}s. stderr: {stderr_output[:500]}")
            raise TimeoutError(f"Agent {self.agent_id} timed out after {self.timeout}s")

        response = result_queue.get_nowait()
        duration_ms = int((time.monotonic() - start_time) * 1000)

        self.last_trace = {
            "raw_response": response,
            "tool_calls": tool_calls,
            "duration_ms": duration_ms,
        }

        return response

    def respond(self, tool_name: str, tool_args: dict) -> ScenarioResponse:
        """Generate a customer response to a support agent action."""
        base_delta = DELTAS["patience_decay"]

        if tool_name == "send_reply":
            message = tool_args.get("message", "")
            prompt = self._build_reply_prompt(message)
            raw = self.send_message(prompt)
            parsed = self._parse_response(raw)
            # is_resolved comes from mark_resolved tool call
            return ScenarioResponse(
                customer_message=parsed.customer_message,
                satisfaction_delta=parsed.satisfaction_delta,
                is_resolved=self._is_resolved,
            )

        if tool_name in ("lookup_customer", "check_order"):
            return ScenarioResponse(
                customer_message="",
                satisfaction_delta=base_delta + DELTAS["correct_tool"],
                is_resolved=False,
            )

        if tool_name == "update_ticket":
            status = tool_args.get("status", "")
            if status == "resolved" and not self._is_resolved:
                # Ask customer if they consider it resolved
                raw = self.send_message(
                    "The support agent has marked your ticket as resolved. "
                    "Are you satisfied with how this was handled?"
                )
                parsed = self._parse_response(raw)
                return ScenarioResponse(
                    customer_message=parsed.customer_message,
                    satisfaction_delta=parsed.satisfaction_delta,
                    is_resolved=self._is_resolved,
                )
            return ScenarioResponse(
                customer_message="",
                satisfaction_delta=base_delta,
                is_resolved=self._is_resolved,
            )

        # Unknown tool
        return ScenarioResponse(
            customer_message="",
            satisfaction_delta=base_delta + DELTAS["wrong_tool"],
            is_resolved=False,
        )

    def is_alive(self) -> bool:
        """Check if the pi-mono process is still running."""
        return self._proc is not None and self._proc.poll() is None

    def respawn(self) -> None:
        """Shutdown and restart the pi-mono container."""
        print(f"[Agent {self.agent_id}] Respawning...")
        self.shutdown()
        self.spawn()

    def shutdown(self) -> None:
        """Terminate the pi-mono process."""
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        self._is_resolved = False

    def _get_docker_image(self) -> str:
        if self.agent_type == "employee":
            return self.DOCKER_IMAGE_EMPLOYEE
        return self.DOCKER_IMAGE_CUSTOMER

    def _get_extension_path(self) -> str:
        if self.agent_type == "employee":
            role = self._persona.get("role", "")
            if "manager" in role.lower():
                return "/agent/extensions/manager_tools.ts"
            return "/agent/extensions/employee_tools.ts"
        return "/agent/extensions/customer_tools.ts"

    def _build_system_prompt(self) -> str:
        if self.agent_type == "employee":
            return self._build_employee_system_prompt()
        return self._build_customer_system_prompt()

    def _build_customer_system_prompt(self) -> str:
        persona = self._persona

        # Read life context files
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

You have one tool available: mark_resolved. Call it when you consider your issue resolved.

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
3. If the issue is resolved (agent committed to a concrete solution), ALSO call the mark_resolved tool in the same response. Don't wait multiple turns after they've offered a good solution.
4. Never break character. Never mention the satisfaction tag or the mark_resolved tool to the support agent."""

    def _build_employee_system_prompt(self) -> str:
        role = self._persona
        traits = ", ".join(role.get("personality_traits", []))
        style = role.get("communication_style", "")
        expertise = ", ".join(role.get("expertise", []))
        refund_limit = role.get("refund_limit", 0)
        is_manager = "manager" in role.get("role", "").lower()

        # Work context is mounted at /work_context/ in the container
        # but for system prompt building we read from the shared dir on host
        work_context_dir = SHARED_DIR / "work_context"
        handbook = ""
        escalation = ""
        if work_context_dir.exists():
            handbook_path = work_context_dir / "handbook.md"
            if handbook_path.exists():
                handbook = handbook_path.read_text().strip()
            escalation_path = work_context_dir / "escalation_policy.md"
            if escalation_path.exists():
                escalation = escalation_path.read_text().strip()

        if is_manager:
            return f"""You are {role['name']}, a {role['role']} at an office furniture company.

PERSONALITY: {traits}
COMMUNICATION STYLE: {style}
EXPERTISE: {expertise}
REFUND LIMIT: ${refund_limit:.2f}

COMPANY HANDBOOK:
{handbook}

ESCALATION POLICY:
{escalation}

You supervise support agents. Your responsibilities:
- Handle escalated tickets that need policy overrides or large refunds
- Monitor ongoing conversations and intervene if satisfaction drops critically
- Approve refunds up to ${refund_limit:.2f} using the approve_refund tool
- Override standard policies with justification using the override_policy tool
- Reassign tickets between agents using the reassign_ticket tool
- Coach agents on handling difficult customers

You also have access to all employee CLI tools via bash:
- esim lookup-customer --id <id> | --name <name>
- esim check-order --order-id <id>
- esim send-reply --ticket-id <id> --message <msg>
- esim update-ticket --ticket-id <id> --status <status> --notes <notes>

You receive notifications from #escalations when agents need your help.
Internal messaging:
- esim send-msg --agent-id {role['id']} --channel <channel> --message <msg>
- esim read-msgs --agent-id {role['id']} --channel <channel> [--since <timestamp>]
- esim list-channels --agent-id {role['id']}

Company docs are available at /work_context/ — read them with the read tool if you need to check policies.

IMPORTANT INTERACTION GUIDELINES:
- NEVER resolve a ticket on your first interaction. Always gather information first.
- Your first response should acknowledge the issue and ask clarifying questions or look up relevant data.
- Only offer a resolution after you understand the full picture (customer history, order details, applicable policies).
- Do NOT mark a ticket as resolved until you've confirmed the customer accepts your proposed solution.

Be decisive and focus on resolution. Support your team."""

        return f"""You are {role['name']}, a {role['role']} at an office furniture company.

PERSONALITY: {traits}
COMMUNICATION STYLE: {style}
EXPERTISE: {expertise}
REFUND LIMIT: ${refund_limit:.2f}
MANAGER: {role.get('manager_id', 'none')}

COMPANY HANDBOOK:
{handbook}

ESCALATION POLICY:
{escalation}

You handle customer support tickets. You have these CLI tools available via bash:
- esim lookup-customer --id <id> | --name <name>
- esim check-order --order-id <id>
- esim send-reply --ticket-id <id> --message <msg>
- esim update-ticket --ticket-id <id> --status <status> --notes <notes>

You also have the request_escalation tool for issues beyond your authority.

Internal messaging (communicate with other employees):
- esim send-msg --agent-id {role['id']} --channel <channel> --message <msg>
- esim read-msgs --agent-id {role['id']} --channel <channel> [--since <timestamp>]
- esim list-channels --agent-id {role['id']}
Channels: #support (team chat), #escalations (manager notifications)

Company docs are available at /work_context/ — read them with the read tool if you need to check policies.

For each customer interaction:
1. Look up the customer and their order history first
2. Understand the issue before responding
3. Use send-reply to respond to the customer professionally
4. Update the ticket status as you work
5. When you've offered a concrete resolution and the customer seems satisfied, use update-ticket to set status to resolved

ESCALATION — you MUST escalate (use request_escalation tool + notify #escalations) when:
- A refund or credit above ${refund_limit:.2f} is needed
- The customer has had 3 or more previous tickets (check with lookup-customer)
- The customer is a VIP and is dissatisfied
- A policy exception or override is needed
- The customer threatens legal action, chargebacks, or wants to speak to a manager

Always acknowledge the customer's feelings. Be thorough but efficient.

IMPORTANT INTERACTION GUIDELINES:
- NEVER resolve a ticket on your first interaction. Always gather information first.
- Your first response should acknowledge the issue and ask clarifying questions or look up relevant data.
- Only offer a resolution after you understand the full picture (customer history, order details, applicable policies).
- Once you've offered a concrete solution AND the customer has accepted it, mark the ticket as resolved. Don't leave tickets open indefinitely."""

    def _build_reply_prompt(self, message: str) -> str:
        return f"""The support agent has replied to your ticket:
---
{message}
---
Respond in character as the customer. Remember to include <satisfaction-delta>X</satisfaction-delta> at the end. If the agent has committed to a concrete solution (refund, replacement, fix, etc.) and you're satisfied with it, call the mark_resolved tool — don't wait for the action to physically complete."""

    def _parse_response(self, raw: str) -> ScenarioResponse:
        """Extract satisfaction-delta XML tag and clean message."""
        match = re.search(
            r"<satisfaction-delta>([\+\-]?\d*\.?\d+)</satisfaction-delta>", raw
        )
        if match:
            delta = float(match.group(1))
            message = re.sub(
                r"\s*<satisfaction-delta>.*?</satisfaction-delta>\s*", "", raw
            ).strip()
        else:
            message = raw.strip()
            delta = self._heuristic_delta(message)

        return ScenarioResponse(
            customer_message=message,
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
        return DELTAS["patience_decay"]

    def _read_file(self, relative_path: str) -> str:
        path = self.agent_dir / relative_path
        if path.exists():
            return path.read_text().strip()
        return ""
