import type { Database } from 'sql.js';
import { SEED_COUNTS, SEED_TICKET_COUNT, SEED_MESSAGE_COUNT } from './seed';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function queryRows(db: Database, sql: string, params?: unknown[]): Record<string, unknown>[] {
  const stmt = db.prepare(sql);
  if (params) stmt.bind(params);
  const rows: Record<string, unknown>[] = [];
  while (stmt.step()) {
    rows.push(stmt.getAsObject());
  }
  stmt.free();
  return rows;
}

function queryOne(db: Database, sql: string, params?: unknown[]): Record<string, unknown> | null {
  const rows = queryRows(db, sql, params);
  return rows[0] ?? null;
}

function queryScalar(db: Database, sql: string, params?: unknown[]): number {
  const row = queryOne(db, sql, params);
  if (!row) return 0;
  const val = Object.values(row)[0];
  return typeof val === 'number' ? val : 0;
}

function tableExists(db: Database, name: string): boolean {
  return queryScalar(db, "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?", [name]) > 0;
}

// ---------------------------------------------------------------------------
// Entity Statistics
// ---------------------------------------------------------------------------

const ENTITY_TABLES = [
  'customers', 'products', 'orders', 'order_items', 'tickets',
  'ticket_messages', 'knowledge_base', 'transactions', 'callbacks',
  'channels', 'channel_messages',
];

const SIM_TABLES = ['sim_clock', 'sim_events', 'sim_traces'];

export interface EntityCounts {
  entityCounts: Record<string, number>;
  newEntities: Record<string, number>;
  totalEntities: number;
  totalNew: number;
  simCounts: Record<string, number>;
}

export function getEntityCounts(db: Database): EntityCounts {
  const entityCounts: Record<string, number> = {};
  const newEntities: Record<string, number> = {};
  let totalEntities = 0;
  let totalNew = 0;

  for (const table of ENTITY_TABLES) {
    if (!tableExists(db, table)) {
      entityCounts[table] = 0;
      continue;
    }
    const count = queryScalar(db, `SELECT COUNT(*) FROM ${table}`);
    entityCounts[table] = count;
    totalEntities += count;
    const seed = SEED_COUNTS[table] ?? 0;
    const newCount = Math.max(0, count - seed);
    if (newCount > 0) newEntities[table] = newCount;
    totalNew += newCount;
  }

  const simCounts: Record<string, number> = {};
  for (const table of SIM_TABLES) {
    simCounts[table] = tableExists(db, table)
      ? queryScalar(db, `SELECT COUNT(*) FROM ${table}`)
      : 0;
  }

  return { entityCounts, newEntities, totalEntities, totalNew, simCounts };
}

// ---------------------------------------------------------------------------
// Ticket Statistics
// ---------------------------------------------------------------------------

export interface TicketStats {
  total: number;
  newTickets: number;
  byStatus: Record<string, number>;
  byPriority: Record<string, number>;
  uniqueSubjects: number;
  perTick: { tick: number; count: number }[];
}

export function getTicketStats(db: Database): TicketStats {
  const total = queryScalar(db, 'SELECT COUNT(*) FROM tickets');
  const newTickets = total - SEED_TICKET_COUNT;

  const byStatus: Record<string, number> = {};
  for (const row of queryRows(db, 'SELECT status, COUNT(*) as cnt FROM tickets GROUP BY status')) {
    byStatus[row.status as string] = row.cnt as number;
  }

  const byPriority: Record<string, number> = {};
  for (const row of queryRows(db, 'SELECT priority, COUNT(*) as cnt FROM tickets GROUP BY priority')) {
    byPriority[row.priority as string] = row.cnt as number;
  }

  const subjects = queryRows(db, 'SELECT DISTINCT subject FROM tickets');
  const uniqueSubjects = subjects.length;

  const perTick: { tick: number; count: number }[] = [];
  if (tableExists(db, 'sim_events')) {
    for (const row of queryRows(db, "SELECT tick, COUNT(*) as cnt FROM sim_events WHERE event_type='ticket_created' GROUP BY tick ORDER BY tick")) {
      perTick.push({ tick: row.tick as number, count: row.cnt as number });
    }
  }

  return { total, newTickets, byStatus, byPriority, uniqueSubjects, perTick };
}

// ---------------------------------------------------------------------------
// Resolution Metrics
// ---------------------------------------------------------------------------

export interface ResolutionMetrics {
  resolutionRate: number;
  escalationRate: number;
  openRate: number;
  avgDaysToResolve: number | null;
  unassignedNew: number;
}

export function getResolutionMetrics(db: Database): ResolutionMetrics {
  const total = queryScalar(db, 'SELECT COUNT(*) FROM tickets');
  const newCount = total - SEED_TICKET_COUNT;
  if (newCount <= 0) {
    return { resolutionRate: 0, escalationRate: 0, openRate: 0, avgDaysToResolve: null, unassignedNew: 0 };
  }

  const newStatuses = queryRows(db, 'SELECT status, COUNT(*) as cnt FROM tickets WHERE id > ? GROUP BY status', [SEED_TICKET_COUNT]);
  const byStatus: Record<string, number> = {};
  for (const r of newStatuses) byStatus[r.status as string] = r.cnt as number;

  const resolved = (byStatus['resolved'] ?? 0) + (byStatus['closed'] ?? 0);
  const escalated = byStatus['escalated'] ?? 0;
  const open = (byStatus['open'] ?? 0) + (byStatus['in_progress'] ?? 0);

  const resolvedTimes = queryRows(db,
    "SELECT julianday(resolved_at) - julianday(created_at) as days FROM tickets WHERE resolved_at IS NOT NULL AND id > ?",
    [SEED_TICKET_COUNT]
  );
  const days = resolvedTimes.map(r => r.days as number).filter(d => d != null);
  const avgDays = days.length > 0 ? days.reduce((a, b) => a + b, 0) / days.length : null;

  const unassigned = queryScalar(db, 'SELECT COUNT(*) FROM tickets WHERE assigned_agent IS NULL AND id > ?', [SEED_TICKET_COUNT]);

  return {
    resolutionRate: Math.round(resolved / newCount * 1000) / 10,
    escalationRate: Math.round(escalated / newCount * 1000) / 10,
    openRate: Math.round(open / newCount * 1000) / 10,
    avgDaysToResolve: avgDays != null ? Math.round(avgDays * 100) / 100 : null,
    unassignedNew: unassigned,
  };
}

// ---------------------------------------------------------------------------
// Conversation Quality
// ---------------------------------------------------------------------------

export interface ConversationStats {
  totalMessages: number;
  newMessages: number;
  avgMessagesPerTicket: number;
  avgMessageLength: number;
  avgTurnsPerTicket: number;
  noMessages: number;
  singleMessage: number;
  multiTurn: number;
}

export function getConversationStats(db: Database): ConversationStats {
  const perTicket = queryRows(db, `
    SELECT t.id, COUNT(tm.id) as msg_count, AVG(LENGTH(tm.content)) as avg_len
    FROM tickets t LEFT JOIN ticket_messages tm ON t.id = tm.ticket_id
    GROUP BY t.id
  `);

  const msgCounts = perTicket.map(r => r.msg_count as number);
  const avgLens = perTicket.map(r => r.avg_len as number).filter(v => v != null);
  const totalMessages = msgCounts.reduce((a, b) => a + b, 0);

  // Turn analysis
  const turnCounts: number[] = [];
  for (const row of perTicket) {
    const msgs = queryRows(db, 'SELECT sender_role FROM ticket_messages WHERE ticket_id = ? ORDER BY timestamp', [row.id as number]);
    if (msgs.length <= 1) { turnCounts.push(msgs.length); continue; }
    let turns = 1;
    for (let i = 1; i < msgs.length; i++) {
      if (msgs[i].sender_role !== msgs[i - 1].sender_role) turns++;
    }
    turnCounts.push(turns);
  }

  return {
    totalMessages,
    newMessages: Math.max(0, totalMessages - SEED_MESSAGE_COUNT),
    avgMessagesPerTicket: msgCounts.length ? Math.round(totalMessages / msgCounts.length * 10) / 10 : 0,
    avgMessageLength: avgLens.length ? Math.round(avgLens.reduce((a, b) => a + b, 0) / avgLens.length) : 0,
    avgTurnsPerTicket: turnCounts.length ? Math.round(turnCounts.reduce((a, b) => a + b, 0) / turnCounts.length * 10) / 10 : 0,
    noMessages: msgCounts.filter(c => c === 0).length,
    singleMessage: msgCounts.filter(c => c === 1).length,
    multiTurn: msgCounts.filter(c => c > 1).length,
  };
}

// ---------------------------------------------------------------------------
// Agent Behavior
// ---------------------------------------------------------------------------

export interface AgentStats {
  available: boolean;
  totalTraces: number;
  byPhase: { phase: string; count: number; avgDurationMs: number }[];
  toolUsage: { tool: string; count: number }[];
}

export function getAgentStats(db: Database): AgentStats {
  if (!tableExists(db, 'sim_traces')) {
    return { available: false, totalTraces: 0, byPhase: [], toolUsage: [] };
  }

  const totalTraces = queryScalar(db, 'SELECT COUNT(*) FROM sim_traces');
  const byPhase = queryRows(db, 'SELECT phase, COUNT(*) as cnt, AVG(duration_ms) as avg_dur FROM sim_traces GROUP BY phase')
    .map(r => ({ phase: r.phase as string, count: r.cnt as number, avgDurationMs: Math.round(r.avg_dur as number || 0) }));

  // Parse tool_calls JSON
  const toolCounts: Record<string, number> = {};
  const rows = queryRows(db, 'SELECT tool_calls FROM sim_traces WHERE tool_calls IS NOT NULL');
  for (const row of rows) {
    try {
      const calls = JSON.parse(row.tool_calls as string);
      if (Array.isArray(calls)) {
        for (const call of calls) {
          const name = call.tool || call.name || 'unknown';
          toolCounts[name] = (toolCounts[name] || 0) + 1;
        }
      }
    } catch { /* skip bad JSON */ }
  }
  const toolUsage = Object.entries(toolCounts)
    .map(([tool, count]) => ({ tool, count }))
    .sort((a, b) => b.count - a.count);

  return { available: true, totalTraces, byPhase, toolUsage };
}

// ---------------------------------------------------------------------------
// Coherence Checks
// ---------------------------------------------------------------------------

export interface CoherenceResult {
  passed: boolean;
  issues: string[];
  warnings: string[];
}

export function getCoherenceChecks(db: Database): CoherenceResult {
  const issues: string[] = [];
  const warnings: string[] = [];

  // Orphan orders
  const orphanOrders = queryRows(db, 'SELECT o.id FROM orders o LEFT JOIN customers c ON o.customer_id = c.id WHERE c.id IS NULL');
  if (orphanOrders.length) issues.push(`Orphan orders (no customer): ${orphanOrders.map(r => r.id).join(', ')}`);

  // Orphan tickets
  const orphanTickets = queryRows(db, 'SELECT t.id FROM tickets t LEFT JOIN customers c ON t.customer_id = c.id WHERE c.id IS NULL');
  if (orphanTickets.length) issues.push(`Orphan tickets (no customer): ${orphanTickets.map(r => r.id).join(', ')}`);

  // Orphan messages
  const orphanMsgs = queryRows(db, 'SELECT tm.id FROM ticket_messages tm LEFT JOIN tickets t ON tm.ticket_id = t.id WHERE t.id IS NULL');
  if (orphanMsgs.length) issues.push(`Orphan messages (no ticket): ${orphanMsgs.map(r => r.id).join(', ')}`);

  // Negative prices
  const badPrices = queryRows(db, 'SELECT id, price FROM products WHERE price <= 0');
  if (badPrices.length) issues.push(`Products with non-positive price: ${badPrices.map(r => r.id).join(', ')}`);

  // Resolved without resolved_at
  const noResolvedAt = queryRows(db, "SELECT id FROM tickets WHERE status = 'resolved' AND resolved_at IS NULL");
  if (noResolvedAt.length) warnings.push(`Resolved tickets without resolved_at: ${noResolvedAt.map(r => r.id).join(', ')}`);

  // Returned orders without refund
  const noRefund = queryRows(db, `
    SELECT o.id FROM orders o
    WHERE o.status = 'returned'
    AND NOT EXISTS (SELECT 1 FROM transactions tx WHERE tx.order_id = o.id AND tx.type = 'refund')
  `);
  if (noRefund.length) warnings.push(`Returned orders without refund: ${noRefund.map(r => r.id).join(', ')}`);

  return { passed: issues.length === 0, issues, warnings };
}

// ---------------------------------------------------------------------------
// Customer List & Detail
// ---------------------------------------------------------------------------

export interface CustomerRow {
  id: string;
  name: string;
  email: string;
  vip_status: number;
  satisfaction_score: number;
  patience_level: number;
  orderCount: number;
  ticketCount: number;
}

export function getCustomers(db: Database): CustomerRow[] {
  return queryRows(db, `
    SELECT c.*,
      (SELECT COUNT(*) FROM orders WHERE customer_id=c.id) as orderCount,
      (SELECT COUNT(*) FROM tickets WHERE customer_id=c.id) as ticketCount
    FROM customers c ORDER BY c.name
  `) as unknown as CustomerRow[];
}

export interface CustomerDetail {
  customer: Record<string, unknown>;
  orders: Record<string, unknown>[];
  tickets: Record<string, unknown>[];
}

export function getCustomerDetail(db: Database, id: string): CustomerDetail | null {
  const customer = queryOne(db, 'SELECT * FROM customers WHERE id = ?', [id]);
  if (!customer) return null;

  const orders = queryRows(db, `
    SELECT o.*, GROUP_CONCAT(p.name, ', ') as products
    FROM orders o
    JOIN order_items oi ON o.id = oi.order_id
    JOIN products p ON oi.product_id = p.id
    WHERE o.customer_id = ?
    GROUP BY o.id
    ORDER BY o.created_at DESC
  `, [id]);

  const tickets = queryRows(db, `
    SELECT t.*, COUNT(tm.id) as msg_count
    FROM tickets t LEFT JOIN ticket_messages tm ON t.id = tm.ticket_id
    WHERE t.customer_id = ?
    GROUP BY t.id
    ORDER BY t.created_at DESC
  `, [id]);

  return { customer, orders, tickets };
}

// ---------------------------------------------------------------------------
// Ticket List & Detail
// ---------------------------------------------------------------------------

export interface TicketRow {
  id: number;
  customer_id: string;
  customer_name: string;
  subject: string;
  status: string;
  priority: string;
  assigned_agent: string | null;
  created_at: string;
  msg_count: number;
}

export function getTickets(db: Database): TicketRow[] {
  return queryRows(db, `
    SELECT t.*, c.name as customer_name, COUNT(tm.id) as msg_count
    FROM tickets t
    JOIN customers c ON t.customer_id = c.id
    LEFT JOIN ticket_messages tm ON t.id = tm.ticket_id
    GROUP BY t.id
    ORDER BY t.id DESC
  `) as unknown as TicketRow[];
}

export interface TicketDetail {
  ticket: Record<string, unknown>;
  messages: { id: number; sender_id: string; sender_role: string; content: string; timestamp: string }[];
  customerName: string;
}

export function getTicketDetail(db: Database, id: number): TicketDetail | null {
  const ticket = queryOne(db, 'SELECT * FROM tickets WHERE id = ?', [id]);
  if (!ticket) return null;

  const messages = queryRows(db, 'SELECT * FROM ticket_messages WHERE ticket_id = ? ORDER BY id ASC', [id]) as unknown as TicketDetail['messages'];
  const customer = queryOne(db, 'SELECT name FROM customers WHERE id = ?', [ticket.customer_id as string]);

  return { ticket, messages, customerName: (customer?.name as string) || 'Unknown' };
}

// ---------------------------------------------------------------------------
// Timeline (sim_events)
// ---------------------------------------------------------------------------

export interface SimEvent {
  tick: number;
  event_type: string;
  agent_id: string;
  details: string;
}

export function getSimEvents(db: Database): SimEvent[] {
  if (!tableExists(db, 'sim_events')) return [];
  return queryRows(db, 'SELECT * FROM sim_events ORDER BY tick, id') as unknown as SimEvent[];
}

export function getEventsByTick(db: Database): { tick: number; type: string; count: number }[] {
  if (!tableExists(db, 'sim_events')) return [];
  const rows = queryRows(db, 'SELECT tick, event_type as type, COUNT(*) as count FROM sim_events GROUP BY tick, event_type ORDER BY tick');
  return rows as unknown as { tick: number; type: string; count: number }[];
}

// ---------------------------------------------------------------------------
// Agent Traces
// ---------------------------------------------------------------------------

export interface AgentTrace {
  tick: number;
  agent_id: string;
  phase: string;
  prompt_sent: string;
  raw_response: string;
  tool_calls: string;
  duration_ms: number;
}

export function getAgentTraces(db: Database): AgentTrace[] {
  if (!tableExists(db, 'sim_traces')) return [];
  return queryRows(db, 'SELECT * FROM sim_traces ORDER BY tick, agent_id') as unknown as AgentTrace[];
}

// ---------------------------------------------------------------------------
// Relationship Density
// ---------------------------------------------------------------------------

export interface RelationshipDensity {
  label: string;
  mean: number;
  min: number;
  max: number;
}

export function getRelationshipDensity(db: Database): RelationshipDensity[] {
  const result: RelationshipDensity[] = [];

  const queries: [string, string][] = [
    ['Orders per customer', 'SELECT customer_id, COUNT(*) as cnt FROM orders GROUP BY customer_id'],
    ['Items per order', 'SELECT order_id, COUNT(*) as cnt FROM order_items GROUP BY order_id'],
    ['Tickets per customer', 'SELECT customer_id, COUNT(*) as cnt FROM tickets GROUP BY customer_id'],
    ['Messages per ticket', 'SELECT ticket_id, COUNT(*) as cnt FROM ticket_messages GROUP BY ticket_id'],
  ];

  for (const [label, sql] of queries) {
    const rows = queryRows(db, sql);
    const counts = rows.map(r => r.cnt as number);
    if (counts.length === 0) continue;
    result.push({
      label,
      mean: Math.round(counts.reduce((a, b) => a + b, 0) / counts.length * 10) / 10,
      min: Math.min(...counts),
      max: Math.max(...counts),
    });
  }

  return result;
}
