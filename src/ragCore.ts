import { createHash } from "node:crypto";

export type RagSource = { id?: string; dominio?: string; url?: string; activo?: boolean; referencia?: string };
export type RagChunk = { sourceId: string; source: string; norm: string; article?: string; jurisdiction?: string; domain?: string; title?: string; text: string; status?: string; metadata?: Record<string, unknown> };

export function activeSources(manifest: { fuentes?: RagSource[] }): RagSource[] {
  return (manifest.fuentes || []).filter((source) => source.activo === true && Boolean(source.url));
}

export function normalizeNorm(value: string): string {
  return value.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase().replace(/\b(?:nro?\.?|numero)\s*/g, "").replace(/\s+/g, " ").replace(/(\d)\.(\d{3})\b/g, "$1$2").replace(/\/(\d{2})\b/g, (_m, yy: string) => `/20${yy}`).trim();
}

export function sourceFingerprint(manifest: unknown, documents: unknown): string {
  return createHash("sha256").update(JSON.stringify({ manifest, documents })).digest("hex");
}

export function recordsFromDocument(data: unknown): Record<string, any>[] {
  if (Array.isArray(data)) return data.filter((row) => row && typeof row === "object");
  if (!data || typeof data !== "object") return [];
  const obj = data as Record<string, unknown>;
  for (const key of ["registros", "servicios"]) if (Array.isArray(obj[key])) return (obj[key] as unknown[]).filter((row) => row && typeof row === "object") as Record<string, any>[];
  return [obj as Record<string, any>];
}

export function classifyNomenclator(question: string): string | null {
  const q = normalizeNorm(question);
  if (/1480\s*\/\s*2019|jardines? maternales?|pasitos cuidados?/.test(q)) return "decreto_1480_2019";
  if (/2453\s*\/\s*2019|centros educativos?/.test(q)) return "decreto_2453_2019";
  if (/2431\s*\/\s*2016|nomenclador general/.test(q)) return "decreto_2431_2016";
  return null;
}

export function rankNormative(chunks: RagChunk[], question: string): RagChunk[] {
  const q = normalizeNorm(question);
  const normMatch = q.match(/\b(ley|decreto(?:[- ]ley| acuerdo| ec?t)?|resolucion)\s*(\d+)\s*\/?\s*(\d{2,4})?/);
  const requestedNorm = normMatch ? { kind: normMatch[1].startsWith("decreto") ? "decreto" : normMatch[1], number: normMatch[2], year: normMatch[3] ? (normMatch[3].length === 2 ? `20${normMatch[3]}` : normMatch[3]) : "" } : null;
  const article = q.match(/\b(?:art(?:iculo)?\.?|articulo)\s*(\d+)/)?.[1];
  const nomenclator = classifyNomenclator(question);
  return chunks.map((chunk) => {
    const norm = normalizeNorm(`${chunk.norm} ${chunk.source}`);
    let score = 0;
    if (requestedNorm) {
      const exactIdentity = norm.includes(requestedNorm.kind) && norm.includes(requestedNorm.number) && (!requestedNorm.year || norm.includes(requestedNorm.year));
      score += exactIdentity ? 1000 : -1000;
    }
    const foundArticle = chunk.article?.match(/\d+/)?.[0];
    if (article) score += foundArticle === article ? 500 : requestedNorm ? -2000 : -100;
    if (nomenclator) score += norm.includes(nomenclator.replaceAll("_", " ")) || normalizeNorm(chunk.sourceId).includes(nomenclator) ? 700 : -500;
    for (const token of q.split(/\W+/).filter((x) => x.length > 3)) if (normalizeNorm(`${chunk.title || ""} ${chunk.text}`).includes(token)) score += 2;
    return { chunk, score };
  }).filter((x) => !requestedNorm || x.score > 0).sort((a, b) => b.score - a.score).map((x) => x.chunk);
}

export function attachDocumentControl(chunks: RagChunk[], control: Record<string, unknown>): RagChunk[] {
  return chunks.map((chunk) => ({ ...chunk, metadata: { ...chunk.metadata, ...control } }));
}
