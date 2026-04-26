import OpenAI from "openai";

export type DominioBackend =
  | "licencias"
  | "estatuto"
  | "general"
  | "coberturas";

export type ChatbotQueryInput = {
  pregunta: string;
  dominio?: DominioBackend;
  maxResults?: number;
};

export type SearchHit = {
  filename?: string;
  score?: number;
  text?: string;
};

export type BackendArticulo = {
  capitulo?: string;
  titulo_capitulo?: string;
  articulo?: number | string;
  titulo_articulo?: string;
  texto?: string;
  resumen?: string | null;
  fuente?: string;
  [key: string]: unknown;
};

export type ChatbotWorkflowOutput = {
  ok: boolean;
  tipo: string;
  dominio: DominioBackend;
  origen: string;
  consulta: string;
  consultaNormalizada: string;
  respuesta: string;
  articulos: BackendArticulo[];
  referencias: string[];
  busqueda: SearchHit[];
  conversationId: string | null;
  error?: string;
};

type SourceConfig = {
  dominio: DominioBackend;
  source: string;
  filename: string;
  referencia: string;
};

type LocalChunk = {
  id: string;
  dominio: DominioBackend;
  filename: string;
  source: string;
  referencia: string;
  articulo?: string;
  titulo?: string;
  text: string;
  raw?: unknown;
};

const NO_ENCONTRADO =
  "No encontré fragmentos suficientes en la base cargada para responder con precisión a esta consulta.";

const SERVICIO_NO_DISPONIBLE = "Servicio no disponible por el momento.";

const DEFAULT_LICENCIAS_URL =
  "https://raw.githubusercontent.com/rcampagnale/sidca-chatbot-docs/main/regimen_licencias_docentes.json";

const DEFAULT_ESTATUTO_URL =
  "https://raw.githubusercontent.com/rcampagnale/sidca-chatbot-docs/main/texto_provincial_para_firestore.json";

const DEFAULT_GENERAL_URL =
  "https://raw.githubusercontent.com/rcampagnale/sidca-chatbot-docs/main/consultas_generales.json";

let cachedChunks: LocalChunk[] | null = null;
let cachedAt = 0;

function getEnv(name: string): string {
  return process.env[name]?.trim() || "";
}

function dominioSeguro(dominio?: string): DominioBackend {
  if (
    dominio === "licencias" ||
    dominio === "estatuto" ||
    dominio === "general" ||
    dominio === "coberturas"
  ) {
    return dominio;
  }

  return "general";
}

function normalizeText(str: string): string {
  return (str || "")
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

const STOPWORDS = new Set([
  "para",
  "como",
  "cuando",
  "donde",
  "desde",
  "hasta",
  "sobre",
  "tengo",
  "quiero",
  "necesito",
  "consulta",
  "docente",
  "docentes",
  "licencia",
  "licencias",
  "articulo",
  "articulos",
  "catamarca",
  "provincia",
  "provincial",
  "sidca",
  "cual",
  "cuanto",
  "cuantos",
  "cuanta",
  "cuantas",
  "debo",
  "puedo",
  "hacer",
  "tiene",
  "esta",
  "este",
  "estos",
  "estas",
  "ella",
  "ellos",
  "ellas",
  "usted",
  "ustedes",
  "ante",
  "bajo",
  "entre",
  "pero",
  "porque",
]);

function tokenize(text: string): string[] {
  const normalized = normalizeText(text);

  return Array.from(
    new Set(
      normalized
        .split(" ")
        .map((w) => w.trim())
        .filter(Boolean)
        .filter((w) => w.length >= 3)
        .filter((w) => !STOPWORDS.has(w))
    )
  );
}

function classifyDomain(
  pregunta: string,
  dominio?: DominioBackend
): DominioBackend {
  if (dominio) return dominio;

  const text = normalizeText(pregunta);

  const rules: Record<DominioBackend, string[]> = {
    licencias: [
      "licencia",
      "licencias",
      "inasistencia",
      "franquicia",
      "maternidad",
      "embarazo",
      "lactancia",
      "enfermedad",
      "tratamiento",
      "corto",
      "largo",
      "accidente",
      "familiar",
      "duelo",
      "matrimonio",
      "estudio",
      "estudios",
      "examen",
      "examenes",
      "rendir",
      "capacitacion",
      "perfeccionamiento",
      "violencia",
      "genero",
      "adopcion",
      "donacion",
      "sangre",
      "aislamiento",
    ],
    estatuto: [
      "estatuto",
      "ley 3122",
      "ingreso",
      "docencia",
      "titular",
      "titularidad",
      "estabilidad",
      "traslado",
      "permuta",
      "ascenso",
      "junta",
      "clasificacion",
      "disciplina",
      "sancion",
      "remuneracion",
      "deberes",
      "derechos",
    ],
    coberturas: [
      "cobertura",
      "coberturas",
      "asamblea",
      "cargo",
      "cargos",
      "horas",
      "catedra",
      "interino",
      "interinato",
      "suplente",
      "suplencia",
      "cabecera",
      "lom",
      "listado",
      "vacante",
      "vacantes",
      "fua",
      "opcion",
      "destino",
      "renuncia",
    ],
    general: [
      "afiliacion",
      "afiliado",
      "afiliados",
      "oficina",
      "horario",
      "contacto",
      "telefono",
      "whatsapp",
      "certificado",
      "formulario",
      "beneficio",
      "tramite",
      "sindicato",
      "gestion",
    ],
  };

  let bestDomain: DominioBackend = "general";
  let bestScore = 0;

  for (const domain of Object.keys(rules) as DominioBackend[]) {
    const score = rules[domain].reduce((acc, keyword) => {
      return text.includes(normalizeText(keyword)) ? acc + 1 : acc;
    }, 0);

    if (score > bestScore) {
      bestScore = score;
      bestDomain = domain;
    }
  }

  return bestDomain;
}

function getSourceConfigs(): SourceConfig[] {
  const licenciasUrl =
    getEnv("SIDCA_DOCS_LICENCIAS_URL") || DEFAULT_LICENCIAS_URL;

  const estatutoUrl =
    getEnv("SIDCA_DOCS_ESTATUTO_URL") || DEFAULT_ESTATUTO_URL;

  const generalUrl = getEnv("SIDCA_DOCS_GENERAL_URL") || DEFAULT_GENERAL_URL;

  const coberturasUrl = getEnv("SIDCA_DOCS_COBERTURAS_URL");

  const sources: SourceConfig[] = [
    {
      dominio: "licencias",
      source: licenciasUrl,
      filename: "regimen_licencias_docentes.json",
      referencia: "Decreto Acuerdo Nº 1092/2015 – Régimen de Licencias Docentes",
    },
    {
      dominio: "estatuto",
      source: estatutoUrl,
      filename: "texto_provincial_para_firestore.json",
      referencia: "Ley Nº 3122 – Estatuto del Docente Provincial",
    },
    {
      dominio: "general",
      source: generalUrl,
      filename: "consultas_generales.json",
      referencia: "Consultas generales SiDCa",
    },
  ];

  if (coberturasUrl) {
    sources.push({
      dominio: "coberturas",
      source: coberturasUrl,
      filename: "coberturas_docentes.json",
      referencia: "Normativa de coberturas docentes",
    });
  }

  return sources.filter((s) => Boolean(s.source));
}

async function fetchJson(url: string): Promise<unknown> {
  const controller = new AbortController();
  const timeoutId = globalThis.setTimeout(() => controller.abort(), 15000);

  try {
    const response = await fetch(url, {
      signal: controller.signal,
      headers: {
        accept: "application/json,text/plain,*/*",
      },
    });

    if (!response.ok) {
      throw new Error(`No se pudo cargar ${url}. Estado HTTP ${response.status}`);
    }

    const text = await response.text();
    return JSON.parse(text.replace(/^\uFEFF/, ""));
  } finally {
    globalThis.clearTimeout(timeoutId);
  }
}

function primitiveToText(value: unknown): string {
  if (value === null || value === undefined) return "";

  if (typeof value === "string") return value;

  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }

  return "";
}

function objectToText(value: unknown, depth = 0): string {
  if (depth > 6) return "";

  const primitive = primitiveToText(value);
  if (primitive) return primitive;

  if (Array.isArray(value)) {
    return value
      .map((item) => objectToText(item, depth + 1))
      .filter(Boolean)
      .join("\n");
  }

  if (typeof value === "object" && value !== null) {
    return Object.entries(value as Record<string, unknown>)
      .filter(([key]) => {
        const normalizedKey = normalizeText(key);

        return ![
          "embedding",
          "vector",
          "idvector",
          "createdat",
          "updatedat",
        ].includes(normalizedKey);
      })
      .map(([key, item]) => {
        const text = objectToText(item, depth + 1);
        if (!text) return "";

        const normalizedKey = normalizeText(key);

        if (
          [
            "contenido",
            "texto",
            "respuesta",
            "descripcion",
            "titulo",
            "titulo_articulo",
            "pregunta",
            "articulo",
            "fuente",
            "keywords",
          ].includes(normalizedKey)
        ) {
          return text;
        }

        return `${key}: ${text}`;
      })
      .filter(Boolean)
      .join("\n");
  }

  return "";
}

function getObjectField(item: unknown, keys: string[]): string | undefined {
  if (typeof item !== "object" || item === null) return undefined;

  const obj = item as Record<string, unknown>;

  for (const key of keys) {
    const value = obj[key];

    if (value !== null && value !== undefined) {
      const text = primitiveToText(value).trim();
      if (text) return text;
    }
  }

  return undefined;
}

function extractArticle(text: string, item: unknown): string | undefined {
  const direct = getObjectField(item, ["articulo", "artículo", "article"]);

  if (direct) return direct;

  const match = text.match(/art(?:i|í)culo\s+(\d+)/i);
  if (match?.[1]) return match[1];

  const bracketMatch = text.match(/\[art(?:i|í)culo\s+(\d+)/i);
  if (bracketMatch?.[1]) return bracketMatch[1];

  return undefined;
}

function extractTitle(item: unknown): string | undefined {
  return getObjectField(item, [
    "titulo_articulo",
    "título_articulo",
    "titulo",
    "título",
    "descripcion",
    "descripción",
    "pregunta",
    "tema",
    "nombre",
  ]);
}

function makeChunk(
  item: unknown,
  source: SourceConfig,
  index: number,
  prefix = ""
): LocalChunk | null {
  const text = objectToText(item).trim();

  if (!text) return null;

  const titulo = extractTitle(item);
  const articulo = extractArticle(text, item);

  return {
    id: `${source.dominio}-${index}-${articulo || "sin-articulo"}`,
    dominio: source.dominio,
    filename: source.filename,
    source: source.source,
    referencia: source.referencia,
    articulo,
    titulo: prefix && titulo ? `${prefix} - ${titulo}` : titulo || prefix || undefined,
    text,
    raw: item,
  };
}

function extractChunksFromJson(data: unknown, source: SourceConfig): LocalChunk[] {
  if (Array.isArray(data)) {
    return data
      .map((item, index) => makeChunk(item, source, index))
      .filter((item): item is LocalChunk => Boolean(item));
  }

  if (typeof data === "object" && data !== null) {
    const obj = data as Record<string, unknown>;
    const chunks: LocalChunk[] = [];

    for (const [key, value] of Object.entries(obj)) {
      if (Array.isArray(value)) {
        value.forEach((item, index) => {
          const chunk = makeChunk(item, source, chunks.length + index, key);

          if (chunk) chunks.push(chunk);
        });
      }
    }

    if (chunks.length) return chunks;

    const singleChunk = makeChunk(data, source, 0);
    return singleChunk ? [singleChunk] : [];
  }

  const singleChunk = makeChunk(data, source, 0);
  return singleChunk ? [singleChunk] : [];
}

async function loadLocalChunks(): Promise<LocalChunk[]> {
  const ttl = Number(getEnv("LOCAL_RAG_CACHE_TTL_MS") || 5 * 60 * 1000);
  const now = Date.now();

  if (cachedChunks && now - cachedAt < ttl) {
    return cachedChunks;
  }

  const sources = getSourceConfigs();
  const chunks: LocalChunk[] = [];

  for (const source of sources) {
    try {
      const json = await fetchJson(source.source);
      chunks.push(...extractChunksFromJson(json, source));
    } catch (error: any) {
      console.warn(
        `[groqWorkflow] No se pudo cargar fuente ${source.filename}:`,
        error?.message || error
      );
    }
  }

  cachedChunks = chunks;
  cachedAt = now;

  return chunks;
}

function expandQuestion(pregunta: string, dominio: DominioBackend): string {
  const normalized = normalizeText(pregunta);
  const extras: string[] = [];

  if (dominio === "licencias") {
    extras.push("licencia justificacion franquicia docente regimen");
  }

  if (dominio === "estatuto") {
    extras.push("estatuto docente ley 3122 derechos deberes ingreso traslado");
  }

  if (dominio === "coberturas") {
    extras.push(
      "asamblea publica cobertura cargos horas catedra interinato suplencia"
    );
  }

  if (dominio === "general") {
    extras.push("consulta general sindicato afiliado tramite formulario beneficio");
  }

  if (normalized.includes("matern")) {
    extras.push("embarazo parto prenatal postnatal nacimiento lactancia");
  }

  if (normalized.includes("familiar")) {
    extras.push("grupo familiar conyuge hijo padre madre hermano nieto atencion");
  }

  if (normalized.includes("largo")) {
    extras.push("afecciones largo tratamiento junta medica haberes");
  }

  if (normalized.includes("corto")) {
    extras.push("afecciones corto tratamiento enfermedad comun");
  }

  if (
    normalized.includes("estudio") ||
    normalized.includes("estudios") ||
    normalized.includes("examen") ||
    normalized.includes("examenes") ||
    normalized.includes("rendir")
  ) {
    extras.push(
      "estudios examenes practicas rendir examen finales nivel superior universitario postgrado licencia remunerada treinta dias anuales cinco dias por examen articulo 35"
    );
  }

  if (normalized.includes("violencia")) {
    extras.push("violencia genero reduccion jornada reordenacion tiempo trabajo");
  }

  return `${pregunta} ${extras.join(" ")}`.trim();
}

function extractArticleNumbers(pregunta: string): string[] {
  const normalized = normalizeText(pregunta);
  const results = new Set<string>();

  const regexes = [
    /articulo\s+(\d+)/g,
    /art\s+(\d+)/g,
    /art\s*(\d+)/g,
  ];

  for (const regex of regexes) {
    let match: RegExpExecArray | null;

    while ((match = regex.exec(normalized)) !== null) {
      if (match[1]) results.add(match[1]);
    }
  }

  return Array.from(results);
}

function isResumenChunk(chunk: LocalChunk): boolean {
  const titulo = normalizeText(chunk.titulo || "");
  const text = normalizeText(chunk.text);

  return (
    titulo.includes("resumen") ||
    text.startsWith("resumen de licencia") ||
    text.includes("tipo tabla")
  );
}

function scoreChunk(
  chunk: LocalChunk,
  pregunta: string,
  dominio: DominioBackend
): number {
  if (chunk.dominio !== dominio) return -1000;

  const expanded = expandQuestion(pregunta, dominio);
  const queryTokens = tokenize(expanded);
  const chunkText = normalizeText(
    `${chunk.filename}\n${chunk.titulo || ""}\n${chunk.articulo || ""}\n${chunk.text}`
  );

  const normalizedPregunta = normalizeText(pregunta);
  const normalizedChunk = normalizeText(chunk.text);
  const normalizedTitle = normalizeText(chunk.titulo || "");

  let score = 0;

  for (const token of queryTokens) {
    if (chunkText.includes(token)) {
      score += token.length >= 8 ? 4 : token.length >= 5 ? 3 : 1;
    }
  }

  const exactQuestion = normalizeText(pregunta);

  if (exactQuestion.length >= 8 && chunkText.includes(exactQuestion)) {
    score += 20;
  }

  const articleNumbers = extractArticleNumbers(pregunta);

  if (articleNumbers.length && chunk.articulo) {
    const normalizedArticle = normalizeText(String(chunk.articulo));

    if (
      articleNumbers.some(
        (n) => normalizedArticle === n || normalizedArticle.includes(n)
      )
    ) {
      score += 60;
    }
  }

  if (chunk.titulo) {
    const title = normalizeText(chunk.titulo);

    for (const token of tokenize(pregunta)) {
      if (title.includes(token)) score += 5;
    }
  }

  const preguntaExamen =
    normalizedPregunta.includes("examen") ||
    normalizedPregunta.includes("examenes") ||
    normalizedPregunta.includes("rendir");

  if (preguntaExamen && chunk.articulo === "35") {
    score += 100;
  }

  if (preguntaExamen && normalizedChunk.includes("para rendir examen")) {
    score += 80;
  }

  if (preguntaExamen && normalizedChunk.includes("treinta 30 dias anuales")) {
    score += 40;
  }

  if (preguntaExamen && normalizedChunk.includes("cinco 5 dias")) {
    score += 30;
  }

  if (
    preguntaExamen &&
    normalizedChunk.includes("no se consideraran examenes las pruebas")
  ) {
    score += 25;
  }

  if (
    preguntaExamen &&
    (normalizedTitle.includes("estudios") ||
      normalizedTitle.includes("examenes") ||
      normalizedTitle.includes("practicas"))
  ) {
    score += 30;
  }

  if (preguntaExamen && chunk.articulo === "34") {
    score += 15;
  }

  if (preguntaExamen && chunk.articulo === "36") {
    score += 10;
  }

  if (isResumenChunk(chunk) && !normalizedPregunta.includes("resumen")) {
    score -= 35;
  }

  if (
    isResumenChunk(chunk) &&
    preguntaExamen &&
    normalizedChunk.includes("35 y 36")
  ) {
    score += 10;
  }

  return score;
}

function truncateText(text: string, maxLength: number): string {
  const clean = text.replace(/\s+/g, " ").trim();

  if (clean.length <= maxLength) return clean;

  return `${clean.slice(0, maxLength).trim()}...`;
}

async function searchLocalFragments(
  input: ChatbotQueryInput,
  dominio: DominioBackend
): Promise<LocalChunk[]> {
  const chunks = await loadLocalChunks();
  const maxResults = Math.min(Math.max(input.maxResults ?? 5, 1), 8);

  const scored = chunks
    .map((chunk) => ({
      chunk,
      score: scoreChunk(chunk, input.pregunta, dominio),
    }))
    .filter((item) => item.score > 0)
    .sort((a, b) => b.score - a.score);

  return scored.slice(0, maxResults).map((item) => ({
    ...item.chunk,
    text: item.chunk.text,
    raw: item.chunk.raw,
  }));
}

function buildContext(chunks: LocalChunk[]): string {
  return chunks
    .map((chunk, index) => {
      const parts = [
        `### Fragmento ${index + 1}`,
        `Archivo: ${chunk.filename}`,
        `Fuente: ${chunk.referencia}`,
        chunk.articulo ? `Artículo: ${chunk.articulo}` : "",
        chunk.titulo ? `Título: ${chunk.titulo}` : "",
        `Contenido:\n${truncateText(chunk.text, 2200)}`,
      ];

      return parts.filter(Boolean).join("\n");
    })
    .join("\n\n");
}

function buildLocalFallbackAnswer(chunks: LocalChunk[]): string {
  if (!chunks.length) return NO_ENCONTRADO;

  const main = chunks
    .slice(0, 3)
    .map((chunk) => {
      const title = chunk.articulo
        ? `Artículo ${chunk.articulo}${chunk.titulo ? ` – ${chunk.titulo}` : ""}`
        : chunk.titulo || "Fragmento encontrado";

      return `${title}:\n${truncateText(chunk.text, 1200)}`;
    })
    .join("\n\n");

  const referencias = Array.from(new Set(chunks.map((chunk) => chunk.referencia)));

  return [
    "No pude generar la respuesta con Groq en este momento, pero encontré información local relacionada:",
    "",
    main,
    "",
    `Fuente consultada: ${referencias.join(" | ")}`,
  ].join("\n");
}

function toSearchHits(
  chunks: LocalChunk[],
  pregunta: string,
  dominio: DominioBackend
): SearchHit[] {
  return chunks.map((chunk) => ({
    filename: chunk.filename,
    score: scoreChunk(chunk, pregunta, dominio),
    text: truncateText(chunk.text, 1200),
  }));
}

function toArticulos(chunks: LocalChunk[]): BackendArticulo[] {
  return chunks.map((chunk) => ({
    articulo: chunk.articulo,
    titulo_articulo: chunk.titulo,
    texto: truncateText(chunk.text, 1800),
    fuente: chunk.referencia,
  }));
}

function getGroqClient(): OpenAI | null {
  const apiKey = getEnv("GROQ_API_KEY");

  if (!apiKey) return null;

  return new OpenAI({
    apiKey,
    baseURL: getEnv("GROQ_BASE_URL") || "https://api.groq.com/openai/v1",
  });
}

function buildGroqSystemPrompt(): string {
  return [
    "Sos el Asistente Virtual de SiDCa.",
    "Respondé únicamente con la información incluida en los fragmentos normativos proporcionados.",
    "No inventes artículos, plazos, requisitos, autoridades ni procedimientos.",
    "Si la información no está en los fragmentos, indicá que no se encontró información suficiente.",
    "Usá lenguaje claro, formal y útil para docentes afiliados.",
    "Cuando corresponda, mencioná artículos o fuentes presentes en los fragmentos.",
    "No digas que algo no existe si en los fragmentos aparece información relacionada y suficiente.",
    "Al final incluí una línea con el formato: Fuente consultada: nombre de la fuente.",
  ].join(" ");
}

async function askGroq(
  pregunta: string,
  dominio: DominioBackend,
  chunks: LocalChunk[]
): Promise<string> {
  const client = getGroqClient();

  if (!client) {
    throw new Error("Falta GROQ_API_KEY.");
  }

  const model = getEnv("GROQ_MODEL") || "llama-3.1-8b-instant";
  const maxTokens = Number(getEnv("GROQ_MAX_TOKENS") || 900);
  const context = buildContext(chunks);

  const referencias = Array.from(new Set(chunks.map((chunk) => chunk.referencia)));

  const response: any = await client.chat.completions.create({
    model,
    temperature: 0.2,
    max_tokens: maxTokens,
    messages: [
      {
        role: "system",
        content: buildGroqSystemPrompt(),
      },
      {
        role: "user",
        content:
          `Dominio detectado: ${dominio}\n\n` +
          `Consulta del docente:\n${pregunta}\n\n` +
          `Fragmentos recuperados:\n\n${context}\n\n` +
          `Fuentes disponibles para citar al final: ${referencias.join(" | ")}\n\n` +
          "Redactá una respuesta clara, precisa y directa. No uses información externa a los fragmentos.",
      },
    ],
  });

  const content = response?.choices?.[0]?.message?.content;

  if (typeof content !== "string" || !content.trim()) {
    throw new Error("Groq no devolvió contenido utilizable.");
  }

  return content.trim();
}

function isRateLimitError(error: any): boolean {
  const status = Number(error?.status || error?.response?.status || error?.code);
  const message = String(error?.message || "").toLowerCase();

  return status === 429 || message.includes("rate limit") || message.includes("429");
}

function buildOutput(params: {
  ok: boolean;
  tipo: string;
  origen: string;
  dominio: DominioBackend;
  consulta: string;
  respuesta: string;
  chunks: LocalChunk[];
  error?: string;
}): ChatbotWorkflowOutput {
  const referencias = Array.from(
    new Set(params.chunks.map((chunk) => chunk.referencia).filter(Boolean))
  );

  return {
    ok: params.ok,
    tipo: params.tipo,
    dominio: params.dominio,
    origen: params.origen,
    consulta: params.consulta,
    consultaNormalizada: params.consulta,
    respuesta: params.respuesta,
    articulos: toArticulos(params.chunks),
    referencias,
    busqueda: toSearchHits(params.chunks, params.consulta, params.dominio),
    conversationId: null,
    error: params.error,
  };
}

export async function runGroqWorkflow(
  input: ChatbotQueryInput
): Promise<ChatbotWorkflowOutput> {
  const dominio = classifyDomain(input.pregunta, input.dominio);

  try {
    const chunks = await searchLocalFragments(input, dominio);

    if (!chunks.length) {
      return buildOutput({
        ok: true,
        tipo: "sin_resultados",
        origen: "local_rag",
        dominio,
        consulta: input.pregunta,
        respuesta: NO_ENCONTRADO,
        chunks: [],
      });
    }

    try {
      const respuestaGroq = await askGroq(input.pregunta, dominio, chunks);

      return buildOutput({
        ok: true,
        tipo: "respuesta_ia",
        origen: "groq_rag",
        dominio,
        consulta: input.pregunta,
        respuesta: respuestaGroq,
        chunks,
      });
    } catch (groqError: any) {
      const localAnswer = buildLocalFallbackAnswer(chunks);
      const rateLimit = isRateLimitError(groqError);

      console.warn(
        "[groqWorkflow] Groq no respondió. Se usa fallback local:",
        groqError?.message || groqError
      );

      return buildOutput({
        ok: true,
        tipo: "respuesta_local",
        origen: rateLimit ? "local_rag_fallback_429" : "local_rag_fallback",
        dominio,
        consulta: input.pregunta,
        respuesta: localAnswer,
        chunks,
        error: groqError?.message || "Error al consultar Groq",
      });
    }
  } catch (error: any) {
    console.error("[groqWorkflow] Error general:", error);

    return buildOutput({
      ok: false,
      tipo: "error",
      origen: "groq_rag",
      dominio,
      consulta: input.pregunta,
      respuesta: SERVICIO_NO_DISPONIBLE,
      chunks: [],
      error: error?.message || "Error interno",
    });
  }
}