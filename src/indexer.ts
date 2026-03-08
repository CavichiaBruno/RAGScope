import fs from "fs";
import path from "path";
import { Mistral } from "@mistralai/mistralai";
import { CodeParser } from "./parser.js";

export interface IndexedChunk {
  id: string; // filename:startLine:endLine
  content: string;
  metadata: {
    file: string;
    class: string | null;
    function?: string | null;
    lines: string;
    language?: string;
  };
  embedding?: number[];
  tokenCount: number;
  isTruncated: boolean;
}

export class Indexer {
  private client: Mistral;
  private parser: CodeParser;

  constructor(apiKey: string) {
    this.client = new Mistral({ apiKey });
    this.parser = new CodeParser();
  }

  /**
   * Recursively crawls a directory and returns an array of chunks
   */
  async indexDirectory(dirPath: string, cachePath?: string): Promise<IndexedChunk[]> {
    const absoluteDirPath = path.resolve(dirPath);
    
    if (cachePath && fs.existsSync(cachePath)) {
      const cacheData = JSON.parse(fs.readFileSync(cachePath, "utf-8"));
      if (cacheData.dirPath === absoluteDirPath) {
        console.log(`[INFO] Loading index from cache for: ${absoluteDirPath}`);
        return cacheData.chunks;
      } else {
        console.log(`[INFO] Cache exists but for a different path (${cacheData.dirPath}). Re-indexing...`);
      }
    }

    // Initialize parser
    await this.parser.init();

    const files = this.getAllFiles(dirPath);
    console.log(`[INFO] Found ${files.length} candidate files in ${dirPath}`);

    const allChunks: IndexedChunk[] = [];
    for (const file of files) {
      const relativePath = path.relative(dirPath, file);
      const content = fs.readFileSync(file, "utf-8");
      const chunks = await this.chunkFile(relativePath, content);
      allChunks.push(...chunks);
      console.log(`   [file] [${relativePath}] -> ${chunks.length} chunks`);
    }

    console.log(`[INFO] Created ${allChunks.length} chunks. Starting embedding...`);

    // Embed in batches to avoid rate limits / large payloads
    const batchSize = 16;
    for (let i = 0; i < allChunks.length; i += batchSize) {
      const batch = allChunks.slice(i, i + batchSize);
      const response = await this.client.embeddings.create({
        model: "mistral-embed",
        inputs: batch.map((c) => c.content),
      });

      for (let j = 0; j < batch.length; j++) {
        batch[j].embedding = response.data[j].embedding;
      }
      console.log(`   [embed] [${Math.min(i + batchSize, allChunks.length)}/${allChunks.length}] chunks embedded...`);
    }

    if (cachePath) {
      this.saveIndex(dirPath, allChunks, cachePath);
    }

    return allChunks;
  }

  saveIndex(dirPath: string, chunks: IndexedChunk[], filePath: string) {
    const data = {
      dirPath: path.resolve(dirPath),
      chunks
    };
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
    console.log(`[INFO] Index saved to ${filePath}`);
  }

  private getAllFiles(dir: string, fileList: string[] = []): string[] {
    const files = fs.readdirSync(dir);
    for (const file of files) {
      const name = path.join(dir, file);
      if (fs.statSync(name).isDirectory()) {
        const ignoreList = ["node_modules", ".git", "dist", "vendor", "storage", "bootstrap/cache", "public"];
        if (ignoreList.includes(file)) continue;
        this.getAllFiles(name, fileList);
      } else {
        const allowedExtensions = [".ts", ".js", ".tsx", ".jsx", ".php"];
        if (allowedExtensions.some(ext => file.endsWith(ext))) {
          fileList.push(name);
        }
      }
    }
    return fileList;
  }

  /**
   * Splits a file into function-level chunks using tree-sitter (WASM)
   * Fallback to regex-based chunking if tree-sitter is not available for the language
   */
  private async chunkFile(filename: string, content: string): Promise<IndexedChunk[]> {
    const ext = path.extname(filename).toLowerCase();
    
    // Attempt tree-sitter parsing for supported languages
    if (ext === ".ts" || ext === ".tsx" || ext === ".php") {
      try {
        const lang = (ext === ".php") ? "php" : "typescript";
        const treeChunks = await this.parser.parse(content, lang);
        
        if (treeChunks.length > 0) {
          return treeChunks.map(tc => ({
            id: `${filename}:${tc.startLine}:${tc.endLine}`,
            content: tc.content,
            metadata: {
              file: filename,
              class: tc.className,
              function: tc.functionName,
              lines: `${tc.startLine}:${tc.endLine}`,
              language: tc.language
            },
            tokenCount: Math.ceil(tc.content.length / 4),
            isTruncated: false
          }));
        }
      } catch (err) {
        console.warn(`   [WARN] Tree-sitter failed for ${filename}, falling back to regex.`, err);
      }
    }

    // Fallback to regex chunking (original logic)
    return this.regexChunkFile(filename, content);
  }

  private regexChunkFile(filename: string, content: string): IndexedChunk[] {
    const lines = content.split("\n");
    const chunks: IndexedChunk[] = [];
    let currentClass: string | null = null;
    
    let i = 0;
    while (i < lines.length) {
      const line = lines[i].trim();
      
      const classMatch = line.match(/class\s+(\w+)/);
      if (classMatch) {
        currentClass = classMatch[1];
      }

      const funcMatch = line.match(/(?:(?:export\s+)?(?:async\s+)?(?:public|private|protected|static\s+)?function\s+(\w+)|(?:public|private|protected|static\s+)?(?:async\s+)?(\w+)\s*\([^)]*\)\s*(?::\s*[^\{]+)?\s*\{|const\s+(\w+)\s*=\s*(?:async\s*)?\(.*?\)\s*=>)/);
      
      if (funcMatch && line.includes("{")) {
        const startLine = i + 1;
        const functionName = funcMatch[1] || funcMatch[2] || funcMatch[3];
        
        let braceCount = 0;
        let j = i;
        let functionContent = "";
        
        while (j < lines.length) {
          const l = lines[j];
          functionContent += l + "\n";
          braceCount += (l.match(/\{/g) || []).length;
          braceCount -= (l.match(/\}/g) || []).length;
          if (braceCount === 0 && j >= i) break;
          j++;
        }
        
        const endLine = j + 1;
        chunks.push({
          id: `${filename}:${startLine}:${endLine}`,
          content: functionContent.trim(),
          metadata: {
            file: filename,
            class: currentClass,
            function: functionName,
            lines: `${startLine}:${endLine}`
          },
          tokenCount: Math.ceil(functionContent.length / 4),
          isTruncated: false
        });
        
        i = j + 1;
        continue;
      }
      i++;
    }

    if (chunks.length === 0 && content.trim().length > 0) {
      chunks.push({
        id: `${filename}:1:${lines.length}`,
        content: content.trim(),
        metadata: {
          file: filename,
          class: null,
          lines: `1:${lines.length}`
        },
        tokenCount: Math.ceil(content.length / 4),
        isTruncated: false
      });
    }

    return chunks;
  }
}
