// ─────────────────────────────────────────────
//  RAGScope — End-to-end example
//
//  What this does:
//    1. Takes a small in-memory "codebase" (fake chunks)
//    2. Embeds them with Mistral
//    3. For each query: retrieves relevant chunks, generates answer
//    4. Tracks every step, detects issues, scores quality
//    5. Prints a full report to console
//
//  Run: npm run example
// ─────────────────────────────────────────────

import "dotenv/config"
import { Mistral } from "@mistralai/mistralai"
import { detectIssues, scoreWithLLM } from "./evaluator.js"
import { saveTrace, getSummary, type ChunkTrace, type QueryTrace, type Issue } from "./tracker.js"
import { Indexer, type IndexedChunk } from "./indexer.js"
import { DatasetGenerator } from "./generator.js"
import crypto from "crypto"
import path from "path"
import fs from "fs"

const client = new Mistral({ apiKey: process.env.MISTRAL_API_KEY! })

async function buildIndex() {
  const indexer = new Indexer(process.env.MISTRAL_API_KEY!);
  
  // Get path from CLI: npx tsx src/example.ts /path/to/repo
  const targetPath = process.argv[2] || "./src";
  const srcDir = path.resolve(targetPath);
  
  if (!fs.existsSync(srcDir)) {
    console.log(`[ERROR] Directory not found: ${srcDir}`);
    console.log(`[INFO]  Usage: npx tsx src/example.ts <path_to_repo>`);
    process.exit(1);
  }

  // Use a local cache file unique to the target directory to avoid collisions
  const pathHash = crypto.createHash("md5").update(srcDir).digest("hex").slice(0, 8);
  const cachePath = path.join(process.cwd(), `.ragscope_cache_${pathHash}.json`);

  console.log(`[INFO] Indexing codebase at ${srcDir}...`);
  return await indexer.indexDirectory(srcDir, cachePath);
}

// ── Step 2: Hybrid Retrieval ─────────────────

interface KeywordResult {
  id: string;
  content: string;
  keywordScore: number;
}

function keywordSearch(query: string, index: IndexedChunk[], topK = 10): KeywordResult[] {
  const terms = query.toLowerCase().split(/\W+/).filter(t => t.length > 2);
  if (terms.length === 0) return [];

  const results: KeywordResult[] = index.map(chunk => {
    const content = (chunk.content + " " + chunk.id).toLowerCase();
    let score = 0;
    for (const term of terms) {
      if (content.includes(term)) {
        // Very basic TF-IDF: freq * idf_proxy
        const occurrences = (content.split(term).length - 1);
        score += occurrences;
      }
    }
    return { id: chunk.id, content: chunk.content, keywordScore: score };
  });

  return results
    .filter(r => r.keywordScore > 0)
    .sort((a, b) => b.keywordScore - a.keywordScore)
    .slice(0, topK);
}

function reciprocalRankFusion(
  vectorResults: { id: string }[],
  keywordResults: { id: string }[],
  topK = 3,
  k = 60
): string[] {
  const scores: Record<string, number> = {};

  vectorResults.forEach((res, rank) => {
    scores[res.id] = (scores[res.id] || 0) + 1 / (k + rank + 1);
  });

  keywordResults.forEach((res, rank) => {
    scores[res.id] = (scores[res.id] || 0) + 1 / (k + rank + 1);
  });

  return Object.keys(scores)
    .sort((a, b) => scores[b] - scores[a])
    .slice(0, topK);
}

async function expandQuery(query: string): Promise<string> {
  const response = await client.chat.complete({
    model: "mistral-small-latest",
    messages: [
      {
        role: "system",
        content: `You are a search query optimizer for a PHP/Laravel codebase.
Convert the user's natural language question into a space-separated list of technical keywords, class names, or method names likely to appear in the code.
DO NOT provide explanations. DO NOT hallucinate file paths or extensions.
Keep it under 10 words.

Example:
Input: "How are orders stored?"
Output: "Pedido migration model table schema database orders"`,
      },
      {
        role: "user",
        content: `Input: "${query}"`,
      },
    ],
  });

  const expanded = response.choices?.[0]?.message?.content ?? query;
  return typeof expanded === "string" ? expanded.trim().replace(/^output:\s*/i, "") : query;
}

function cosineSimilarity(a: number[], b: number[]): number {
  const dot = a.reduce((sum, val, i) => sum + val * b[i], 0)
  const normA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0))
  const normB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0))
  return dot / (normA * normB)
}

function looksLikeTruncated(content: string): boolean {
  const trimmed = content.trimEnd()
  return !/[.!?})\];>]$/.test(trimmed)
}

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4)
}

async function retrieve(
  query: string,
  index: IndexedChunk[],
  topK = 3
): Promise<ChunkTrace[]> {
  const queryEmbedding = await client.embeddings.create({
    model: "mistral-embed",
    inputs: [query],
  })

  const raw = queryEmbedding.data[0].embedding
  if (!raw) throw new Error("Mistral returned a null embedding for the query.")
  const queryVec: number[] = raw

  const scored = index
    .filter(chunk => !!chunk.embedding)
    .map(chunk => ({
      id: chunk.id,
      content: chunk.content,
      similarityScore: cosineSimilarity(queryVec, chunk.embedding!),
      tokenCount: estimateTokens(chunk.content),
      isTruncated: looksLikeTruncated(chunk.content),
    }))

  return scored
    .sort((a, b) => b.similarityScore - a.similarityScore)
    .slice(0, topK)
}

async function hybridRetrieve(
  query: string,
  index: IndexedChunk[],
  topK = 3
): Promise<ChunkTrace[]> {
  // 1. Expand query
  const expandedQuery = await expandQuery(query);
  if (process.env.DEBUG === "true") {
     console.log(`      Expanding: "${query}" -> "${expandedQuery}"`);
  }
  const [vResults, kResults] = await Promise.all([
    retrieve(expandedQuery, index, 10), // Vector
    Promise.resolve(keywordSearch(expandedQuery, index, 10)) // Keyword
  ]);

  // 3. Fusion
  const fusedIds = reciprocalRankFusion(vResults, kResults, topK);
  
  // 4. Reconstruct results (enrich with metadata)
  return fusedIds.map(id => {
    const vMatch = vResults.find(v => v.id === id);
    const originalChunk = index.find(c => c.id === id)!;

    return {
      id: id,
      content: originalChunk.content,
      similarityScore: vMatch?.similarityScore ?? 0,
      tokenCount: estimateTokens(originalChunk.content),
      isTruncated: looksLikeTruncated(originalChunk.content)
    };
  });
}

// ── Step 3: Generate answer ───────────────────

async function generate(
  query: string, 
  chunks: ChunkTrace[], 
  model = process.env.MISTRAL_MODEL || "mistral-small-latest"
): Promise<string> {
  const context = chunks.map(c => `// ${c.id}\n${c.content}`).join("\n\n---\n\n")

  const response = await client.chat.complete({
    model: model,
    messages: [
      {
        role: "system",
        content: "You are a helpful code assistant. Answer questions about the codebase using only the provided context. Be concise.",
      },
      {
        role: "user",
        content: `Context:\n${context}\n\nQuestion: ${query}`,
      },
    ],
  })

  const content = response.choices?.[0]?.message?.content ?? ""
  return typeof content === "string" ? content : ""
}

// ── Step 4: Track everything ──────────────────

async function ragWithTracking(
  query: string,
  index: IndexedChunk[]
) {
  const start = Date.now()
  const model = process.env.MISTRAL_MODEL || "mistral-small-latest"

  console.log(`\n[QUERY] "${query}"...`);
  const chunks = await hybridRetrieve(query, index)
  const answer = await generate(query, chunks, model)
  const issues = detectIssues(chunks, query, index)
  const scores = await scoreWithLLM(client, query, chunks, answer, model)

  const trace: QueryTrace = {
    id: crypto.randomUUID(),
    timestamp: new Date(),
    query,
    chunks,
    answer,
    durationMs: Date.now() - start,
    issues,
    scores,
  }

  saveTrace(trace)
  printTrace(trace)
}

// ── Console output ────────────────────────────

function printTrace(trace: QueryTrace) {
  const status = trace.issues.length === 0 ? "[PASS]" : "[FAIL]"
  const line = "─".repeat(60)

  console.log(`\n${line}`)
  console.log(`${status}  QUERY: "${trace.query}"`)
  console.log(`${line}`)

  console.log("\nCHUNKS RETRIEVED:")
  for (const chunk of trace.chunks) {
    const scoreColor = chunk.similarityScore >= 0.65 ? "+" : "-"
    const truncFlag = chunk.isTruncated ? " [TRUNCATED]" : ""
    console.log(`   ${scoreColor} [${chunk.similarityScore.toFixed(3)}] ${chunk.id}${truncFlag}`)
  }

  console.log(`\nANSWER:\n   ${trace.answer.trim().split("\n").join("\n   ")}`)

  if (trace.scores) {
    console.log("\nQUALITY SCORES (LLM-as-judge):")
    console.log(`   Faithfulness:      ${bar(trace.scores.faithfulness)} ${trace.scores.faithfulness.toFixed(2)}`)
    console.log(`   Answer Relevancy:  ${bar(trace.scores.answerRelevancy)} ${trace.scores.answerRelevancy.toFixed(2)}`)
    console.log(`   Context Precision: ${bar(trace.scores.contextPrecision)} ${trace.scores.contextPrecision.toFixed(2)}`)
  }

  if (trace.issues.length > 0) {
    console.log("\nISSUES DETECTED:")
    for (const issue of trace.issues) {
      console.log(`   [${issue.type}] ${issue.detail}`)
    }
  }

  console.log(`\n${trace.durationMs}ms`)
}

function bar(score: number): string {
  const filled = Math.round(score * 10)
  return "█".repeat(filled) + "░".repeat(10 - filled)
}

function printSummary() {
  const summary = getSummary()
  if (!summary) return

  console.log("\n" + "═".repeat(60))
  console.log("  RAGSCOPE REPORT")
  console.log("═".repeat(60))
  console.log(`  Total queries:     ${summary.total}`)
  console.log(`  Passed:            ${summary.passed}`)
  console.log(`  Failed:            ${summary.failed}`)
  console.log(`  Pass rate:         ${summary.passRate}`)
  console.log(`  Avg relevancy:     ${summary.avgAnswerRelevancy}`)

  if (Object.keys(summary.issueCounts).length > 0) {
    console.log("\n  Issue breakdown:")
    for (const [type, count] of Object.entries(summary.issueCounts)) {
      console.log(`    ${type}: ${count}`)
    }
  }
  console.log("═".repeat(60) + "\n")
}

// ── Main ──────────────────────────────────────

interface DatasetItem {
  query: string;
  expected_chunk_id: string;
}

async function runDatasetEvaluation(dataset: DatasetItem[], index: IndexedChunk[]) {
  console.log(`\n[INFO] Running evaluation against dataset (${dataset.length} queries)...`);
  let hits = 0;

  for (const item of dataset) {
    // Basic rate limit protection (3s because hybrid does 2 calls: expand + embed)
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    const retrieved = await hybridRetrieve(item.query, index, 3);
    const isHit = retrieved.some(c => c.id === item.expected_chunk_id);
    if (isHit) hits++;
    
    console.log(`   [${isHit ? "HIT" : "MISS"}] Query: "${item.query}"`);
    if (!isHit && retrieved.length > 0) {
       console.log(`      Expected:   ${item.expected_chunk_id}`);
       console.log(`      Best match: ${retrieved[0]?.id} (Score: ${retrieved[0]?.similarityScore.toFixed(3)})`);
    }
  }

  const hitRate = (hits / dataset.length) * 100;
  console.log("\n" + "═".repeat(60));
  console.log(`  EVALUATION RESULTS`);
  console.log("═".repeat(60));
  console.log(`  Total Queries: ${dataset.length}`);
  console.log(`  Hits:          ${hits}`);
  console.log(`  Hit Rate:      ${hitRate.toFixed(2)}%`);
  console.log("═".repeat(60) + "\n");
}

// ── ask command ───────────────────────────────

interface ConfidenceResult {
  label: string;
  score: string;
}

function calculateConfidence(chunks: ChunkTrace[], issues: Issue[]): ConfidenceResult {
  if (chunks.length === 0) return { label: "LOW", score: "0.00" };
  const avgScore = chunks.reduce((s, c) => s + c.similarityScore, 0) / chunks.length;
  if (issues.length > 0 || avgScore < 0.70) return { label: "LOW",    score: avgScore.toFixed(2) };
  if (avgScore < 0.80)                       return { label: "MEDIUM", score: avgScore.toFixed(2) };
  return                                            { label: "HIGH",   score: avgScore.toFixed(2) };
}

async function askCommand(index: IndexedChunk[], question: string): Promise<void> {
  const chunks  = await hybridRetrieve(question, index, 3);
  const answer  = await generate(question, chunks);
  const issues  = detectIssues(chunks, question, index);
  const conf    = calculateConfidence(chunks, issues);

  const sep = "─".repeat(60);

  console.log(`\n${sep}`);
  console.log(`\n  ${answer.trim().split("\n").join("\n  ")}`);
  console.log(`\n${sep}`);
  console.log(`\nConfidence : ${conf.label} (${conf.score})`);
  console.log(`Sources    :`);
  for (const c of chunks) {
    console.log(`   - ${c.id}  (score: ${c.similarityScore.toFixed(3)})`);
  }

  if (issues.length > 0) {
    console.log(`\nWarnings:`);
    for (const issue of issues) {
      console.log(`   [${issue.type}] ${issue.detail}`);
    }
  } else {
    console.log(`\nNo issues detected.`);
  }
}

async function main() {
  if (!process.env.MISTRAL_API_KEY) {
    console.error("[ERROR] Missing MISTRAL_API_KEY in .env file")
    process.exit(1)
  }

  const args = process.argv.slice(2);
  const isAsk            = args.includes("--ask");
  const isGenerateDataset = args.includes("--generate-dataset") || args.includes("--full");
  const isRunEval         = args.includes("--eval")             || args.includes("--full");
  const datasetPathArg    = args.find(a => a.startsWith("--dataset="))?.split("=")[1];
  const datasetPath       = datasetPathArg || "ragscope-dataset.json";

  // For --ask, the question is the argument after --ask
  const askIdx      = args.indexOf("--ask");
  const askQuestion = isAsk ? args[askIdx + 1] : null;

  if (isAsk && !askQuestion) {
    console.error("[ERROR] --ask requires a question.");
    console.error(`[INFO]  Usage: npx tsx src/example.ts <path> --ask "your question"`);
    process.exit(1);
  }

  console.log("[INFO] Starting RAGScope...\n");
  const index = await buildIndex();
  console.log(`[INFO] Indexed ${index.length} chunks\n`);

  // ── ask: single natural-language query ───────
  if (isAsk && askQuestion) {
    await askCommand(index, askQuestion);
    return;
  }

  // ── generate + eval ──────────────────────────
  if (isGenerateDataset) {
    const generator = new DatasetGenerator(process.env.MISTRAL_API_KEY!);
    const dataset = await generator.generate(index, 10);
    generator.saveDataset(dataset, datasetPath);
    console.log(`[INFO] Dataset saved to ${datasetPath}\n`);
    if (!isRunEval) return;
  }

  if (isRunEval) {
    if (!fs.existsSync(datasetPath)) {
      console.error(`[ERROR] Dataset not found: ${datasetPath}`);
      console.error(`[INFO]  Run with --generate-dataset first, or use --full to do both at once.`);
      process.exit(1);
    }
    const dataset = JSON.parse(fs.readFileSync(datasetPath, "utf-8")) as DatasetItem[];
    await runDatasetEvaluation(dataset, index);
    return;
  }

  // ── default: generic demo queries ────────────
  const queries = [
    "What functions or methods are defined in this codebase?",
    "How is authentication handled?",
    "What database models or schemas are defined?",
  ];

  for (const query of queries) {
    await ragWithTracking(query, index);
  }

  printSummary();
}

main().catch(console.error);
