// ─────────────────────────────────────────────
//  RAGScope — Evaluator
//  Two layers:
//    1. Rule-based: fast, free, catches structural problems
//    2. LLM-as-judge: scores quality of the final answer
// ─────────────────────────────────────────────

import { Mistral } from "@mistralai/mistralai"
import type { ChunkTrace, Issue, EvalScores } from "./tracker.js"

const CONTEXT_LIMIT = 4096 // tokens
const IGNORE_TERMS = [
  "database", "config", "model", "view", "controller",
  "middleware", "migration", "seeder", "route", "schema",
  "table", "index", "store", "update", "delete", "create"
]

// ── Layer 1: Rule-based issue detection ──────

export function detectIssues(
  chunks: ChunkTrace[],
  query: string,
  allChunks: { id: string }[],
  maxTokens = CONTEXT_LIMIT
): Issue[] {
  const issues: Issue[] = []

  // No chunks came back at all
  if (chunks.length === 0) {
    issues.push({
      type: "EMPTY_RETRIEVAL",
      detail: "Vector search returned 0 chunks. Check your index.",
    })
    return issues
  }

  // 1. OBVIOUS_MISS: Did we miss a file explicitly mentioned in the query?
  const allFiles = [...new Set(allChunks.map(c => {
    // Extract everything before the first colon
    return c.id.split(":")[0];
  }))];

  // Group retrieved chunks by file for easy lookup
  const retrievedFileSet = new Set(chunks.map(c => c.id.split(":")[0].toLowerCase()));

  for (const filePath of allFiles) {
    // Get just the filename from path (handle both \ and /)
    const fileName = filePath.split(/[\\/]/).pop() || "";
    const baseName = fileName.split(".")[0];
    
    // Skip generic terms that are likely false positives
    if (IGNORE_TERMS.includes(baseName.toLowerCase())) continue;

    if (baseName.length > 3) {
      const regex = new RegExp(`\\b${baseName}\\b`, "i");
      if (regex.test(query)) {
        const wasRetrieved = retrievedFileSet.has(filePath.toLowerCase());
        
        if (!wasRetrieved) {
          issues.push({
            type: "OBVIOUS_MISS",
            detail: `Query mentions "${baseName}" but "${fileName}" wasn't retrieved in the top chunks.`,
          });
        }
      }
    }
  }

  // 2. Best chunk score is too low
  const maxScore = Math.max(...chunks.map(c => c.similarityScore))
  if (maxScore < 0.65) {
    issues.push({
      type: "LOW_RELEVANCE",
      detail: `Best similarity score was ${maxScore.toFixed(2)} (threshold: 0.65).`,
    })
  }

  // 3. Any chunk looks truncated
  const truncated = chunks.filter(c => c.isTruncated)
  if (truncated.length > 0) {
    issues.push({
      type: "TRUNCATED_CHUNK",
      detail: `${truncated.length} chunk(s) appear truncated mid-sentence or mid-function.`,
    })
  }

  // 4. Context Overflow
  const totalTokens = chunks.reduce((sum, c) => sum + c.tokenCount, 0)
  if (totalTokens > maxTokens * 0.85) {
    issues.push({
      type: "CONTEXT_OVERFLOW",
      detail: `${totalTokens} tokens sent (${((totalTokens / maxTokens) * 100).toFixed(0)}% of limit).`,
    })
  }

  return issues
}

// ── Layer 2: LLM-as-judge scoring ────────────

export async function scoreWithLLM(
  client: Mistral,
  query: string,
  chunks: ChunkTrace[],
  answer: string,
  model = "mistral-small-latest"
): Promise<EvalScores> {
  const context = chunks.map(c => c.content).join("\n\n---\n\n")

  const prompt = `You are an expert RAG evaluator. Score the following RAG output.
Return ONLY a valid JSON object with exactly these three keys, no explanation, no markdown.

Query: ${query}

Retrieved context:
${context}

Generated answer:
${answer}

Score each dimension from 0.0 to 1.0:
- faithfulness: Is every claim in the answer supported by the context? (1.0 = fully grounded, 0.0 = hallucinated)
- answerRelevancy: Does the answer actually address the query? (1.0 = directly answers, 0.0 = completely off-topic)
- contextPrecision: Were the retrieved chunks useful and relevant? (1.0 = perfect retrieval, 0.0 = irrelevant chunks)

JSON only:`

  const response = await client.chat.complete({
    model: model,
    messages: [{ role: "user", content: prompt }],
    temperature: 0,
  })

  const raw = response.choices?.[0]?.message?.content ?? "{}"
  const text = typeof raw === "string" ? raw : ""

  try {
    // Robust extraction: find the first { and last }
    const match = text.match(/\{[\s\S]*\}/);
    if (!match) throw new Error("No JSON found");
    
    const parsed = JSON.parse(match[0])
    return {
      faithfulness: Number(parsed.faithfulness ?? 0.5),
      answerRelevancy: Number(parsed.answerRelevancy ?? 0.5),
      contextPrecision: Number(parsed.contextPrecision ?? 0.5),
    }
  } catch (e) {
    // silent fail, return neutral
    return { faithfulness: 0.5, answerRelevancy: 0.5, contextPrecision: 0.5 }
  }
}
