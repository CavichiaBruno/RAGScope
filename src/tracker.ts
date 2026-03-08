// ─────────────────────────────────────────────
//  RAGScope — Tracker
//  Captures every step of a RAG query and stores
//  a lightweight trace in memory (later: SQLite)
// ─────────────────────────────────────────────

export interface ChunkTrace {
  id: string            // e.g. "file.ts:10:45"
  content: string
  similarityScore: number
  tokenCount: number
  isTruncated: boolean  // ends mid-sentence / mid-function?
}

export interface QueryTrace {
  id: string
  timestamp: Date
  query: string
  chunks: ChunkTrace[]
  answer: string
  durationMs: number
  issues: Issue[]
  scores?: EvalScores
}

export type IssueType =
  | "EMPTY_RETRIEVAL"
  | "LOW_RELEVANCE"
  | "TRUNCATED_CHUNK"
  | "CONTEXT_OVERFLOW"
  | "OBVIOUS_MISS"

export interface Issue {
  type: IssueType
  detail: string
}

export interface EvalScores {
  faithfulness: number      // is the answer grounded in the chunks?
  answerRelevancy: number   // does it actually answer the question?
  contextPrecision: number  // were the right chunks retrieved?
}

// In-memory store — swap for SQLite later
const traces: QueryTrace[] = []

export function saveTrace(trace: QueryTrace) {
  traces.push(trace)
}

export function getTraces(): QueryTrace[] {
  return traces
}

export function getSummary() {
  const total = traces.length
  if (total === 0) return null

  const failed = traces.filter(t => t.issues.length > 0).length
  const issueCounts: Record<string, number> = {}

  for (const trace of traces) {
    for (const issue of trace.issues) {
      issueCounts[issue.type] = (issueCounts[issue.type] ?? 0) + 1
    }
  }

  const avgScore = traces
    .filter(t => t.scores)
    .reduce((sum, t) => sum + (t.scores!.answerRelevancy), 0)
    / (traces.filter(t => t.scores).length || 1)

  return {
    total,
    passed: total - failed,
    failed,
    passRate: `${(((total - failed) / total) * 100).toFixed(1)}%`,
    issueCounts,
    avgAnswerRelevancy: avgScore.toFixed(2),
  }
}
