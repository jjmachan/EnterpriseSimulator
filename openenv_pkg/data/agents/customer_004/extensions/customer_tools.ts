export default function (pi: any) {
  pi.registerTool({
    name: "mark_resolved",
    label: "Mark Resolved",
    description: "Call this when your issue has been fully resolved to your satisfaction",
    parameters: { type: "object", properties: {}, required: [] },
    async execute(_toolCallId: string, _params: any, _signal: any, _onUpdate: any, _ctx: any) {
      return {
        content: [{ type: "text", text: "Issue marked as resolved." }],
        details: {},
      };
    },
  });

}
