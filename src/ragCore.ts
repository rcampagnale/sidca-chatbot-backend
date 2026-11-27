import { createHash } from "node:crypto";

export type RagSource = { id?: string; dominio?: string; url?: string; activo?: boolean; referencia?: string };
export type RagChunk = { sourceId: string; source: string; norm: string; article?: string; jurisdiction?: string; domain?: string; title?: string; text: string; status?: string; metadata?: Record<string, unknown> };
export type NormativeReference = { kind: string; number: string; year?: string; qualifier?: string; jurisdiction?: string };

export function activeSources(manifest: { fuentes?: RagSource[] }): RagSource[] {
  return (manifest.fuentes || []).filter((source) => source.activo === true && Boolean(source.url));
}

export function normalizeNorm(value: string): string {
  return value.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase().replace(/\b(?:nro?\.?|numero)\s*/g, "").replace(/\s+/g, " ").replace(/(\d)\.(\d{3})\b/g, "$1$2").replace(/\/(\d{2})\b/g, (_m, yy: string) => `/20${yy}`).trim();
}

export function parseNormativeReferences(value: string): NormativeReference[] {
  const references: NormativeReference[] = [];
  const pattern = /\b(ley(?:\s+(?:nacional|provincial))?|decreto(?:\s*[-–]\s*ley|\s+acuerdo|\s+ecyt)?|dcto\.?\s*acdo\.?|resoluci[oó]n(?:\s+(?:nacional|provincial))?)\s*(?:(?:n(?:ro)?|numero)\s*(?:[º°ª]\.?|\.\s*[º°ª]?|)?\s*)?(\d[\d.]*(?:\s*\/\s*\d{2,4})?)/gi;
  for (const match of value.matchAll(pattern)) {
    const label = normalizeNorm(match[1]).replace(/\s+/g, " ");
    const rawNumber = match[2].replace(/\s/g, "");
    const [numberPart, yearPart] = rawNumber.split("/");
    const digits = numberPart.replace(/\D/g, "");
    if (!digits) continue;
    let kind = "";
    let qualifier: string | undefined;
    if (label.startsWith("ley")) {
      kind = "ley";
      qualifier = label.includes("nacional") ? "nacional" : label.includes("provincial") ? "provincial" : undefined;
    } else if (label.startsWith("resolucion")) {
      kind = "resolucion";
      qualifier = label.includes("nacional") ? "nacional" : label.includes("provincial") ? "provincial" : undefined;
    } else if (label.includes("ley")) {
      kind = "decreto-ley";
    } else {
      kind = "decreto";
      qualifier = label.includes("acuerdo") || label.startsWith("dcto") ? "acuerdo" : label.includes("ecyt") ? "ecyt" : undefined;
    }
    references.push({ kind, number: digits, ...(yearPart ? { year: yearPart.length === 2 ? `20${yearPart}` : yearPart } : {}), ...(qualifier ? { qualifier } : {}),
      ...(qualifier === "nacional" || qualifier === "provincial" ? { jurisdiction: qualifier } : {}) });
  }
  return references;
}

export function normativeReferenceMatches(requested: NormativeReference, candidates: NormativeReference[], jurisdiction = ""): boolean {
  const requestedJurisdiction = normalizeNorm(`${requested.jurisdiction || ""}`);
  return candidates.some((candidate) => {
    if (candidate.kind !== requested.kind || candidate.number !== requested.number) return false;
    if (requested.year && candidate.year !== requested.year) return false;
    if (requested.kind === "decreto" && requested.qualifier && candidate.qualifier !== requested.qualifier) return false;
    if (requested.qualifier && candidate.qualifier && candidate.qualifier !== requested.qualifier) return false;
    if (requestedJurisdiction) {
      const candidateJurisdiction = normalizeNorm(`${candidate.jurisdiction || jurisdiction}`);
      if (requestedJurisdiction === "provincial" && candidateJurisdiction && candidateJurisdiction !== "provincial" && candidateJurisdiction.includes("nacional")) return false;
      else if (requestedJurisdiction !== "provincial" && candidateJurisdiction && !candidateJurisdiction.includes(requestedJurisdiction)) return false;
      if (!candidateJurisdiction && !requested.qualifier) return false;
    }
    return true;
  });
}

export function chunkMatchesNormativeReference(chunk: RagChunk, requested: NormativeReference): boolean {
  const candidates = [chunk.norm, chunk.source, chunk.title || "", String(chunk.metadata?.documento || ""), chunk.sourceId]
    .flatMap((identity) => parseNormativeReferences(identity));
  return normativeReferenceMatches(requested, candidates, chunk.jurisdiction || String(chunk.metadata?.jurisdiccion || ""));
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
  const requestedNorm = parseNormativeReferences(question)[0] || null;
  const article = q.match(/\b(?:art(?:iculo)?\.?|articulo)\s*(\d+)/)?.[1];
  const nomenclator = classifyNomenclator(question);
  return chunks.map((chunk) => {
    const norm = normalizeNorm(`${chunk.norm} ${chunk.source}`);
    let score = 0;
    if (requestedNorm) {
      const exactIdentity = chunkMatchesNormativeReference(chunk, requestedNorm);
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
