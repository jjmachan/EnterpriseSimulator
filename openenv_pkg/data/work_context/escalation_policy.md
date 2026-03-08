# Escalation Policy

## When to Escalate

You MUST escalate to your manager when:

1. **Refund exceeds your limit** — Support agents have a $200 refund limit. Any refund above this requires manager approval.
2. **Repeated complaints** — Customer has contacted support 3+ times about the same issue or related issues.
3. **VIP customers** — Any VIP customer requesting a refund, policy exception, or expressing significant dissatisfaction.
4. **Policy override needed** — Customer is requesting something outside standard policy (extended return window, waived fees, etc.).
5. **Customer threatens legal action or chargeback** — Immediately escalate, do not attempt to resolve independently.
6. **Satisfaction dropping critically** — If a customer is clearly very frustrated and your actions are not helping.

## How to Escalate

1. Update the ticket status to "escalated" using `esim update-ticket --ticket-id <id> --status escalated --notes "Reason for escalation"`
2. Call the `request_escalation` tool with the ticket ID and reason
3. Inform the customer that you are bringing in a senior team member to help

## What NOT to Escalate

- Standard returns within policy
- Simple order status inquiries
- Refunds within your limit
- Assembly questions (refer to knowledge base)

## Manager Response Time

- Escalated tickets should be addressed within 30 minutes during business hours
- After hours: next business day morning
