---
title: EnterpriseSim Customer Support
emoji: 🎧
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# EnterpriseSim Customer Support Environment

An OpenEnv RL environment for training customer support agents. Built with generative customer personas powered by LLMs.

## Tools

| Tool | Description |
|------|-------------|
| `lookup_customer` | Look up customer profile with order and ticket history |
| `check_order` | Get full order details including items, status, shipping |
| `send_reply` | Send a reply to the customer (triggers LLM customer response) |
| `update_ticket` | Update ticket status and/or add internal notes |

## Quick Start

```python
from openenv.core.mcp_client import MCPToolClient

env = MCPToolClient(base_url="http://localhost:8000")
with env:
    env.reset()
    tools = env.list_tools()
    result = env.call_tool("lookup_customer", customer_id="customer_002")
    result = env.call_tool("send_reply", ticket_id=1, message="Hi, let me help!")
```

## Reward

Reward is computed at episode end from three components:
- **Resolution** (55%): Whether the customer's issue was resolved
- **Satisfaction** (30%): Customer satisfaction score (0-1)
- **Efficiency** (15%): Penalty for taking more than 5 steps
