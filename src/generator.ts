import fs from "fs";
import { Mistral } from "@mistralai/mistralai";
import { IndexedChunk } from "./indexer.js";

export interface DatasetItem {
  query: string;
  expected_chunk_id: string;
}

export class DatasetGenerator {
  private client: Mistral;

  constructor(apiKey: string) {
    this.client = new Mistral({ apiKey });
  }

  async generate(chunks: IndexedChunk[], count = 20): Promise<DatasetItem[]> {
    if (chunks.length === 0) return [];

    // Sample chunks: select 'count' chunks evenly distributed
    const step = Math.max(1, Math.floor(chunks.length / count));
    const sampledChunks = [];
    for (let i = 0; i < chunks.length && sampledChunks.length < count; i += step) {
      sampledChunks.push(chunks[i]);
    }

    const dataset: DatasetItem[] = [];
    console.log(`[INFO] Generating synthetic queries for ${sampledChunks.length} chunks...`);

    for (const chunk of sampledChunks) {
      try {
        const queries = await this.generateQueriesForChunk(chunk);
        for (const query of queries) {
          dataset.push({
            query,
            expected_chunk_id: chunk.id
          });
        }
        console.log(`   [OK]    Generated ${queries.length} queries for ${chunk.id}`);
      } catch (err) {
        console.error(`   [ERROR] Failed to generate queries for ${chunk.id}:`, err);
      }
    }

    return dataset;
  }

  private async generateQueriesForChunk(chunk: IndexedChunk): Promise<string[]> {
    const prompt = `You are generating evaluation queries for a RAG system over a codebase.

Given this code chunk:
${chunk.content}
From file: ${chunk.metadata.file} (lines ${chunk.metadata.lines})

Generate 2 evaluation queries following these strict rules:

RULES:
1. Always reference the specific file name, class name, or method name from the chunk
2. The query must be answerable ONLY by reading this specific chunk
3. Never use vague references like "this method", "this context", "this code", "the function"
4. Reference something unique and specific to this chunk: a variable name, a return value, a specific number, a class name, a specific behavior
5. The query should be useful to someone exploring an unfamiliar codebase

BAD (too generic, works for any codebase):
- "What does the findById() method do in this context?"
- "How many records does paginate() return?"
- "What does this method return?"

GOOD (specific to this chunk):
- "What exception does UserController's show() method throw when a user is not found?"
- "How many items per page does ProductController's index() paginate to?"
- "What columns does the create_orders_table migration define as nullable?"

The query must make sense without seeing the code. Someone reading only the query should know exactly which part of the codebase it refers to.

Return ONLY a valid JSON array with exactly 2 strings. No markdown, no explanation.
["query 1", "query 2"]`;

    const response = await this.client.chat.complete({
      model: "mistral-small-latest",
      messages: [{ role: "user", content: prompt }],
      temperature: 0.7,
    });

    const content = response.choices?.[0]?.message?.content ?? "";
    const text = typeof content === "string" ? content : "";

    // Strip markdown code fences if the model wraps its response (e.g. ```json ... ```)
    const cleaned = text.trim().replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "").trim();

    try {
      const parsed = JSON.parse(cleaned);
      if (Array.isArray(parsed)) {
        return parsed.filter((q): q is string => typeof q === "string" && q.length > 5).slice(0, 3);
      }
    } catch {
      // Fallback: extract quoted strings from the response
      const matches = cleaned.match(/"([^"]{10,})"/g);
      if (matches) {
        return matches.map(m => m.slice(1, -1)).filter(q => q.length > 5).slice(0, 3);
      }
    }

    return [];
  }

  saveDataset(dataset: DatasetItem[], filePath: string) {
    fs.writeFileSync(filePath, JSON.stringify(dataset, null, 2));
    console.log(`[INFO] Dataset saved to ${filePath}`);
  }
}
