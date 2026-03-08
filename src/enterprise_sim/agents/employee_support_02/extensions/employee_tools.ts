export default function (pi: any) {
  pi.registerTool({
    name: "request_escalation",
    label: "Request Escalation",
    description: "Request escalation to your manager when an issue exceeds your authority (e.g., refund above your limit, policy override needed, repeated complaints)",
    parameters: {
      type: "object",
      properties: {
        ticket_id: { type: "number", description: "The ticket ID to escalate" },
        reason: { type: "string", description: "Why this needs escalation" }
      },
      required: ["ticket_id", "reason"]
    },
    async execute(_toolCallId: string, params: any, _signal: any, _onUpdate: any, _ctx: any) {
      return {
        content: [{ type: "text", text: `Escalation requested for ticket #${params.ticket_id}. Manager has been notified.` }],
        details: {},
      };
    },
  });
}
