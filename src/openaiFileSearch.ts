import "dotenv/config";
import OpenAI from "openai";

export type Dominio = "licencias" | "estatuto" | "general";

export type ChatbotQueryInput = {
  pregunta: string;
  dominio?: Dominio;
  maxResults?: number;
};

export type ChatbotQueryOutput = {
  ok: boolean;
  respuesta: string;
  referencias: string[];
  busqueda: Array<{
    filename?: string;
    score?: number;
    text?: string;
  }>;
  error?: string;
};

type SearchResultItem = {
  filename?: string;
  score?: number;
  text?: string;
};

function getRequiredEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) {
    throw new Error(
      `Falta ${name}. Completa el archivo .env local o configura la variable en Cloud Run.`
    );
  }
  return value;
}

function getOpenAIClient(): OpenAI {
  const apiKey = getRequiredEnv("OPENAI_API_KEY");
  return new OpenAI({ apiKey });
}

function getVectorStoreIdByDominio(dominio?: Dominio): string {
  if (dominio === "licencias") {
    return getRequiredEnv("OPENAI_VECTOR_STORE_ID_LICENCIAS");
  }

  if (dominio === "estatuto") {
    return getRequiredEnv("OPENAI_VECTOR_STORE_ID_ESTATUTO");
  }

  return getRequiredEnv("OPENAI_VECTOR_STORE_ID_GENERAL");
}

function normalizeText(str: string): string {
  return (str || "")
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .trim();
}

function extractKeywords(text: string): string[] {
  return normalizeText(text)
    .split(/[^a-z0-9ñ]+/)
    .filter(Boolean)
    .filter((w) => w.length >= 4);
}

function buildExpandedQuestion(
  pregunta: string,
  dominio?: Dominio
): string {
  const base = pregunta.trim();
  const norm = normalizeText(base);

  const extras: string[] = [];

  if (dominio === "licencias") {
    extras.push("Buscar en régimen de licencias docentes provincial.");
  }

  if (dominio === "estatuto") {
    extras.push("Buscar en estatuto docente provincial.");
  }

  if (dominio === "general") {
    extras.push(
      "Buscar en consultas generales docentes provinciales no relacionadas directamente con licencias ni con estatuto, salvo que el texto recuperado lo mencione."
    );
  }

  if (norm.includes("capacit")) {
    extras.push(
      "Relacionar con licencia para estudiar, licencia por estudio, cursos, perfeccionamiento, formación docente, exámenes."
    );
  }

  if (norm.includes("familiar")) {
    extras.push(
      "Relacionar con enfermedad de familiar directo, atención de familiar enfermo, razones familiares."
    );
  }

  if (norm.includes("violencia")) {
    extras.push(
      "Relacionar con violencia de género, licencias especiales y razones familiares si existieran en la normativa."
    );
  }

  if (norm.includes("matern")) {
    extras.push(
      "Relacionar con licencia por maternidad, embarazo, prenatal y postnatal."
    );
  }

  if (norm.includes("accidente")) {
    extras.push(
      "Relacionar con accidente de trabajo, enfermedades profesionales y artículo correspondiente."
    );
  }

  return extras.length ? `${base}\n\nPistas de búsqueda: ${extras.join(" ")}` : base;
}

function getFilePriority(
  filename: string | undefined,
  dominio?: Dominio
): number {
  const name = (filename || "").toLowerCase();

  if (dominio === "licencias") {
    if (name.includes("regimen_licencias_docentes")) return 100;
    if (name.includes("consultas_generales")) return 20;
    if (name.includes("texto_provincial_para_firestore")) return 10;
    return 10;
  }

  if (dominio === "estatuto") {
    if (name.includes("texto_provincial_para_firestore")) return 100;
    if (name.includes("consultas_generales")) return 20;
    if (name.includes("regimen_licencias_docentes")) return 10;
    return 10;
  }

  if (name.includes("consultas_generales")) return 100;
  if (name.includes("regimen_licencias_docentes")) return 20;
  if (name.includes("texto_provincial_para_firestore")) return 20;
  return 10;
}

function scoreResult(
  result: SearchResultItem,
  pregunta: string,
  dominio?: Dominio
): number {
  const keywords = extractKeywords(pregunta);
  const hayKeywords = keywords.length > 0;

  const text = normalizeText(
    `${result.filename || ""}\n${result.text || ""}`
  );

  let keywordMatches = 0;
  for (const kw of keywords) {
    if (text.includes(kw)) keywordMatches++;
  }

  const keywordRatio = hayKeywords ? keywordMatches / keywords.length : 0;
  const filePriority = getFilePriority(result.filename, dominio);
  const retrievalQuality =
    typeof result.score === "number" ? Math.max(0, 1 - result.score) : 0.3;

  return filePriority + keywordRatio * 50 + retrievalQuality * 20;
}

function dedupeAndRankResults(
  results: SearchResultItem[],
  pregunta: string,
  dominio?: Dominio
): SearchResultItem[] {
  const seen = new Set<string>();
  const unique: SearchResultItem[] = [];

  for (const item of results) {
    const key = `${item.filename || ""}::${(item.text || "").slice(0, 200)}`;
    if (seen.has(key)) continue;
    seen.add(key);
    unique.push(item);
  }

  return unique.sort((a, b) => {
    const sb = scoreResult(b, pregunta, dominio);
    const sa = scoreResult(a, pregunta, dominio);
    return sb - sa;
  });
}

function buildContextFromResults(results: SearchResultItem[]): string {
  if (!results.length) return "";

  return results
    .map((r, idx) => {
      const text = (r.text || "").trim();
      const snippet =
        text.length > 1800 ? `${text.slice(0, 1800)}...` : text;

      return [
        `### Fragmento ${idx + 1}`,
        `Archivo: ${r.filename || "sin_nombre"}`,
        typeof r.score === "number" ? `Score: ${r.score}` : "",
        `Contenido:\n${snippet}`,
      ]
        .filter(Boolean)
        .join("\n");
    })
    .join("\n\n");
}

function buildSystemPrompt(dominio?: Dominio): string {
  if (dominio === "licencias") {
    return [
      "Eres un asistente del SIDCA.",
      "Responde únicamente con base en los fragmentos recuperados del régimen de licencias docentes.",
      "Si el fragmento incluye artículo, menciónalo expresamente.",
      "No inventes artículos, plazos, autoridades ni procedimientos.",
      "Si la base normativa es insuficiente, indícalo claramente.",
      "Responde en español claro, breve y útil para docentes.",
    ].join(" ");
  }

  if (dominio === "estatuto") {
    return [
      "Eres un asistente del SIDCA.",
      "Responde únicamente con base en los fragmentos recuperados del estatuto docente provincial.",
      "Si el fragmento incluye ley, título, capítulo o artículo, menciónalo expresamente.",
      "No inventes artículos, requisitos, derechos ni procedimientos.",
      "Si la evidencia es insuficiente, indícalo claramente.",
      "Responde en español claro, breve y útil para docentes.",
    ].join(" ");
  }

  return [
    "Eres un asistente del SIDCA.",
    "Responde únicamente con base en los fragmentos recuperados del dominio de consultas generales.",
    "No mezcles normativa no presente en los fragmentos.",
    "No inventes datos, artículos, plazos ni procedimientos.",
    "Si la evidencia es insuficiente o ambigua, indícalo claramente.",
    "Responde en español claro, breve y útil para docentes.",
  ].join(" ");
}

function extractTextFromSearchResult(result: any): string | undefined {
  if (typeof result?.text === "string" && result.text.trim()) {
    return result.text.trim();
  }

  if (Array.isArray(result?.content)) {
    const joined = result.content
      .map((c: any) => {
        if (typeof c?.text === "string") return c.text;
        if (typeof c?.value === "string") return c.value;
        if (typeof c?.content === "string") return c.content;
        return "";
      })
      .filter(Boolean)
      .join("\n")
      .trim();

    if (joined) return joined;
  }

  if (typeof result?.chunk?.text === "string" && result.chunk.text.trim()) {
    return result.chunk.text.trim();
  }

  return undefined;
}

function extractSearchResults(searchResponse: any): SearchResultItem[] {
  const candidates = Array.isArray(searchResponse?.data)
    ? searchResponse.data
    : Array.isArray(searchResponse?.results)
    ? searchResponse.results
    : Array.isArray(searchResponse?.search_results)
    ? searchResponse.search_results
    : [];

  return candidates.map((result: any) => ({
    filename:
      typeof result?.filename === "string"
        ? result.filename
        : typeof result?.file?.filename === "string"
        ? result.file.filename
        : typeof result?.attributes?.filename === "string"
        ? result.attributes.filename
        : undefined,
    score:
      typeof result?.score === "number"
        ? result.score
        : typeof result?.ranking_score === "number"
        ? result.ranking_score
        : undefined,
    text: extractTextFromSearchResult(result),
  }));
}

export async function runChatbotFileSearch(
  input: ChatbotQueryInput
): Promise<ChatbotQueryOutput> {
  const client = getOpenAIClient();
  const vectorStoreId = getVectorStoreIdByDominio(input.dominio);
  const model = process.env.OPENAI_MODEL?.trim() || "gpt-4.1";

  const desiredSearchResults = Math.min(Math.max(input.maxResults ?? 8, 4), 8);
  const desiredAnswerResults = Math.min(Math.max(input.maxResults ?? 5, 3), 8);

  const expandedQuestion = buildExpandedQuestion(
    input.pregunta,
    input.dominio
  );

  // 1) Recuperación explícita en el vector store correcto
  const searchResponse: any = await (client.vectorStores as any).search(
    vectorStoreId,
    {
      query: expandedQuestion,
      max_num_results: desiredSearchResults,
      filters: {
        type: "eq",
        key: "ambito",
        value: "provincial",
      },
    }
  );

  const rawResults = extractSearchResults(searchResponse);
  const rawReferencias = new Set<string>();

  for (const item of rawResults) {
    if (item.filename) {
      rawReferencias.add(item.filename);
    }
  }

  const rankedResults = dedupeAndRankResults(
    rawResults,
    input.pregunta,
    input.dominio
  );

  const selectedResults = rankedResults.slice(0, desiredAnswerResults);
  const context = buildContextFromResults(selectedResults);

  if (!context.trim()) {
    return {
      ok: true,
      respuesta:
        "No encontré fragmentos suficientes en la base cargada para responder con precisión a esta consulta.",
      referencias: Array.from(rawReferencias),
      busqueda: rankedResults,
    };
  }

  // 2) Generación de respuesta usando solo los fragmentos seleccionados
  const answerResponse: any = await client.responses.create({
    model,
    input: [
      {
        role: "system",
        content: [
          {
            type: "input_text",
            text: buildSystemPrompt(input.dominio),
          },
        ],
      },
      {
        role: "user",
        content: [
          {
            type: "input_text",
            text:
              `Consulta del docente:\n${input.pregunta}\n\n` +
              `Fragmentos recuperados:\n\n${context}\n\n` +
              "Redacta una respuesta breve y precisa. Si la base documental es insuficiente, indícalo claramente.",
          },
        ],
      },
    ],
  });

  const outputText =
    typeof answerResponse.output_text === "string" &&
    answerResponse.output_text.trim()
      ? answerResponse.output_text.trim()
      : "No encontré una respuesta suficientemente clara en la base cargada.";

  const referencias = Array.from(
    new Set(
      selectedResults
        .map((r) => r.filename)
        .filter((x): x is string => Boolean(x))
    )
  );

  return {
    ok: true,
    respuesta: outputText,
    referencias,
    busqueda: selectedResults,
  };
}