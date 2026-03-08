import { Parser, Language, Query } from "web-tree-sitter";
import path from "path";
import fs from "fs";

export interface CodeChunk {
  content: string;
  startLine: number;
  endLine: number;
  functionName: string | null;
  className: string | null;
  language: string;
}

export class CodeParser {
  private parser: Parser | null = null;
  private languages: Record<string, Language> = {};

  async init() {
    try {
      console.log("   Initializing web-tree-sitter runtime...");
      const wasmPath = path.resolve(process.cwd(), "bin", "web-tree-sitter.wasm");
      const wasmBuffer = fs.readFileSync(wasmPath);
      
      await Parser.init({
        wasmBinary: wasmBuffer
      });
      this.parser = new Parser();
    } catch (err) {
      console.error("   ❌ Failed to initialize web-tree-sitter:", err);
      throw err;
    }
  }

  async loadLanguage(lang: "typescript" | "php") {
    if (this.languages[lang]) return this.languages[lang];

    try {
      const wasmPath = path.resolve(process.cwd(), "bin", `${lang}.wasm`);
      console.log(`   Loading WASM buffer for ${lang}: ${wasmPath}`);
      const wasmBuffer = fs.readFileSync(wasmPath);
      const language = await Language.load(wasmBuffer);
      this.languages[lang] = language;
      return language;
    } catch (err) {
      console.error(`   ❌ Failed to load WASM buffer for ${lang}:`, err);
      throw err;
    }
  }

  async parse(code: string, lang: "typescript" | "php"): Promise<CodeChunk[]> {
    if (!this.parser) await this.init();
    
    const language = await this.loadLanguage(lang);
    this.parser!.setLanguage(language);

    const tree = this.parser!.parse(code);
    if (!tree) return [];

    const chunks: CodeChunk[] = [];

    const query = this.getQuery(lang, language);
    const matches = query.matches(tree.rootNode);

    for (const match of matches) {
      const nodeMatch = match.captures.find((c: any) => c.name === "function")?.node;
      if (!nodeMatch) continue;

      const functionName = match.captures.find((c: any) => c.name === "name")?.node.text || null;
      const className = this.findParentClass(nodeMatch, lang);

      chunks.push({
        content: nodeMatch.text,
        startLine: nodeMatch.startPosition.row + 1,
        endLine: nodeMatch.endPosition.row + 1,
        functionName,
        className,
        language: lang
      });
    }

    return chunks;
  }

  private findParentClass(node: any, lang: string): string | null {
    let current = node.parent;
    
    while (current) {
      if (current.type === "class_declaration" || current.type === "class_definition") {
        const nameNode = current.childForFieldName("name") || current.children.find((c: any) => c.type === "type_identifier" || c.type === "name");
        return nameNode?.text || "AnonymousClass";
      }
      current = current.parent;
    }
    return null;
  }

  private getQuery(lang: string, language: Language): Query {
    if (lang === "typescript") {
      return new Query(language, `
        (function_declaration name: (identifier) @name) @function
        (method_definition name: (property_identifier) @name) @function
        (arrow_function) @function
        (function_expression) @function
      `);
    } else {
      // PHP
      return new Query(language, `
        (function_definition name: (name) @name) @function
        (method_declaration name: (name) @name) @function
      `);
    }
  }
}
