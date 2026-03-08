# RAGScope 🔍

RAGScope is a lightweight, open-source observability and debugging library for RAG (Retrieval-Augmented Generation) pipelines. It helps you identify exactly why your RAG system is failing—whether it's due to truncated chunks, low relevance, or context overflow.

## Features

- **Tracker**: Captures every step of the RAG process (retrieval, generation, evaluation).
- **Evaluator**: 
  - **Rule-based Detection**: Instantly flags `TRUNCATED_CHUNK`, `LOW_RELEVANCE`, `CONTEXT_OVERFLOW`, and `OBVIOUS_MISS`.
  - **LLM-as-judge**: Automated scoring for Faithfulness, Relevancy, and Precision (powered by Mistral AI).
- **Indexer**: Intelligent code chunking (TS/JS/PHP/Blade) with local vector persistence.
- **Mistral AI Integration**: Native support for Mistral embeddings and models.

## Quick Start

### 1. Install Dependencies
```bash
npm install
```

### 2. Configure Environment
Create a `.env` file at the root:
```env
MISTRAL_API_KEY=your_api_key_here
MISTRAL_MODEL=mistral-small-latest
```

### 3. Run the Example
Point RAGScope to any local repository to evaluate its RAG performance:
```bash
npx tsx src/example.ts C:/path/to/your/project
```

## How it works

### Intelligent Chunking
Unlike naive character-based splitters, RAGScope's Indexer understands code structure. It uses brace-counting to ensure functions and classes are kept together, providing better context for the LLM.

### OBVIOUS_MISS Detection
RAGScope flags an `OBVIOUS_MISS` if your query explicitly mentions a file or entity (e.g., "CocheController") that exists in the index but wasn't retrieved in the top chunks. This helps you identify gaps in your embedding strategy.

## Project Structure

- `src/tracker.ts`: Trace storage and summary generation.
- `src/evaluator.ts`: Logic for issue detection and LLM scoring.
- `src/indexer.ts`: Codebase crawling and embedding.
- `src/example.ts`: End-to-end demonstration.

## License

MIT
