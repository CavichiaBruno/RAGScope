# RAGScope

RAGScope is a lightweight, open-source observability and debugging library for RAG (Retrieval-Augmented Generation) pipelines. It helps you identify exactly why your RAG system is failing — whether it's due to truncated chunks, low relevance, or context overflow.

## Features

- **Tracker** — Captures every step of the RAG process (retrieval, generation, evaluation).
- **Evaluator**
  - **Rule-based**: Instantly flags `TRUNCATED_CHUNK`, `LOW_RELEVANCE`, `CONTEXT_OVERFLOW`, and `OBVIOUS_MISS`.
  - **LLM-as-judge**: Automated scoring for Faithfulness, Relevancy, and Precision via Mistral AI.
- **Indexer** — Intelligent function-level code chunking (TS/JS/PHP) with local vector cache.
- **Hybrid Retrieval** — Combines vector similarity + keyword search via Reciprocal Rank Fusion.

## Quick Start

### 1. Install

```bash
npm install
```

### 2. Configure

Create a `.env` file at the root:

```env
MISTRAL_API_KEY=your_api_key_here
MISTRAL_MODEL=mistral-small-latest   # optional, defaults to mistral-small-latest
```

### 3. Ask a question (main command)

```bash
npx tsx src/example.ts /path/to/your/repo --ask "Where is scroll handled in Hero?"
```

Output:
```
────────────────────────────────────────────────────────────

  Scroll is handled in Hero.tsx's handleScroll method (line 32).
  It uses an easing function to calculate logo transformations
  and sets a 150ms timeout to detect when scrolling stops.

────────────────────────────────────────────────────────────

Confidence : HIGH (0.89)
Sources    :
   - src/app/components/Hero.tsx:32:42  (score: 0.891)
   - src/app/components/Hero.tsx:71:95  (score: 0.743)

No issues detected.
```

**Confidence levels:**
- `HIGH` — avg similarity > 0.80, no issues detected
- `MEDIUM` — avg similarity 0.70–0.80
- `LOW` — avg similarity < 0.70 or issues found (results may be unreliable)

### 4. Evaluation pipeline (internal / CI)

**Default mode** — runs a set of generic queries and prints a quality report:
```bash
npx tsx src/example.ts /path/to/your/repo
```

**Generate a synthetic evaluation dataset** from the codebase:
```bash
npx tsx src/example.ts /path/to/your/repo --generate-dataset
```

**Run evaluation** against an existing dataset:
```bash
npx tsx src/example.ts /path/to/your/repo --eval
```

**Full pipeline** — generate dataset and immediately evaluate in one shot:
```bash
npx tsx src/example.ts /path/to/your/repo --full
```

**Custom dataset path** (optional, default is `ragscope-dataset.json`):
```bash
npx tsx src/example.ts /path/to/your/repo --full --dataset=my-eval.json
```

## How It Works

### Intelligent Chunking

RAGScope's Indexer parses code structure (via tree-sitter WASM for TypeScript and PHP, with a brace-counting regex fallback) to keep functions and classes intact — never splitting mid-function.

### Hybrid Retrieval

Each query goes through:
1. **Query expansion** — Mistral rewrites the query into technical keywords.
2. **Vector search** — cosine similarity over Mistral embeddings.
3. **Keyword search** — basic TF-IDF over chunk content.
4. **RRF fusion** — Reciprocal Rank Fusion merges both result lists.

### OBVIOUS_MISS Detection

If a query explicitly mentions a file or class that exists in the index but wasn't retrieved in the top chunks, RAGScope flags it as `OBVIOUS_MISS` — helping you spot gaps in your embedding strategy.

## Project Structure

| File | Purpose |
|------|---------|
| `src/indexer.ts` | Codebase crawling, chunking, embedding, and cache |
| `src/evaluator.ts` | Rule-based issue detection and LLM-as-judge scoring |
| `src/tracker.ts` | Trace storage and summary generation |
| `src/generator.ts` | Synthetic query generation for dataset evaluation |
| `src/example.ts` | End-to-end demonstration |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `MISTRAL_API_KEY` | Yes | Your Mistral AI API key |
| `MISTRAL_MODEL` | No | Model to use (default: `mistral-small-latest`) |
| `DEBUG` | No | Set to `true` to log query expansion output |

## License

MIT
