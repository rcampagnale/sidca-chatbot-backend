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
    if (name.includes("texto_provincial_para_firestore")) return 70;
    if (name.includes("consultas_generales")) return 20;
    return 10;
  }

  if (dominio === "estatuto") {
    if (name.includes("texto_provincial_para_firestore")) return 100;
    if (name.includes("regimen_licencias_docentes")) return 60;
    if (name.includes("consultas_generales")) return 20;
    return 10;
  }

  // general
  if (name.includes("regimen_licencias_docentes")) return 100;
  if (name.includes("texto_provincial_para_firestore")) return 95;
  if (name.includes("consultas_generales")) return 25;
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
  const dominioTexto =
    dominio === "licencias"
      ? "régimen de licencias docentes"
      : dominio === "estatuto"
      ? "estatuto docente provincial"
      : "normativa docente provincial";

  return [
    "Eres un asistente del SIDCA.",
    `Responde únicamente con base en los fragmentos recuperados de ${dominioTexto}.`,
    "Da prioridad a fragmentos normativos de regimen_licencias_docentes.json y texto_provincial_para_firestore.json.",
    "Usa consultas_generales.json solo como apoyo orientativo y nunca para contradecir o reemplazar normativa.",
    "Si no hay base normativa suficiente en los fragmentos, indícalo claramente.",
    "No inventes artículos, plazos, autoridades ni procedimientos.",
    "Responde en español claro y útil para docentes.",
    "Si la evidencia es insuficiente, dilo explícitamente.",
  ].join(" ");
}

export async function runChatbotFileSearch(
  input: ChatbotQueryInput
): Promise<ChatbotQueryOutput> {
  const client = getOpenAIClient();
  const vectorStoreId = getRequiredEnv("OPENAI_VECTOR_STORE_ID");
  const model = process.env.OPENAI_MODEL?.trim() || "gpt-4.1";

  const expandedQuestion = buildExpandedQuestion(
    input.pregunta,
    input.dominio
  );

  const tools: any[] = [
    {
      type: "file_search",
      vector_store_ids: [vectorStoreId],
      max_num_results: Math.min(Math.max(input.maxResults ?? 8, 4), 8),
      filters: input.dominio && input.dominio !== "general"
    ? {
        type: "and",
        filters: [
          {
            type: "eq",
            key: "ambito",
            value: "provincial",
          },
          {
            type: "eq",
            key: "dominio",
            value: input.dominio,
          },
        ],
      }
    : {
        type: "eq",
        key: "ambito",
        value: "provincial",
      },
    },
  ];

  // 1) Recuperación documental
  const retrievalResponse: any = await client.responses.create({
    model,
    input: [
      {
        role: "system",
        content: [
          {
            type: "input_text",
            text:
              "Recupera los fragmentos documentales más útiles para responder una consulta normativa docente provincial. Prioriza normativa específica por sobre FAQs generales.",
          },
        ],
      },
      {
        role: "user",
        content: [
          {
            type: "input_text",
            text: expandedQuestion,
          },
        ],
      },
    ],
    tools,
    include: ["file_search_call.results"],
  });

  const rawResults: SearchResultItem[] = [];
  const rawReferencias = new Set<string>();

  if (Array.isArray(retrievalResponse.output)) {
    for (const item of retrievalResponse.output) {
      if (item?.type === "file_search_call" && Array.isArray(item?.results)) {
        for (const result of item.results) {
          const text =
            typeof result?.text === "string"
              ? result.text
              : Array.isArray(result?.content)
              ? result.content
                  .map((c: any) =>
                    typeof c?.text === "string" ? c.text : ""
                  )
                  .filter(Boolean)
                  .join("\n")
              : undefined;

          rawResults.push({
            filename: result?.filename,
            score: typeof result?.score === "number" ? result.score : undefined,
            text,
          });

          if (result?.filename) {
            rawReferencias.add(String(result.filename));
          }
        }
      }
    }
  }

  const rankedResults = dedupeAndRankResults(
    rawResults,
    input.pregunta,
    input.dominio
  );

  const selectedResults = rankedResults.slice(0, input.maxResults ?? 5);
  const context = buildContextFromResults(selectedResults);

  if (!context.trim()) {
    return {
      ok: true,
      respuesta:
        "No encontré fragmentos suficientes en la normativa cargada para responder con precisión a esta consulta.",
      referencias: Array.from(rawReferencias),
      busqueda: rankedResults,
    };
  }

  // 2) Generación de respuesta usando SOLO los fragmentos seleccionados
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
              "Redacta una respuesta breve y precisa. Si la base normativa es insuficiente, indícalo claramente.",
          },
        ],
      },
    ],
  });

  const outputText =
    typeof answerResponse.output_text === "string" &&
    answerResponse.output_text.trim()
      ? answerResponse.output_text.trim()
      : "No encontré una respuesta suficientemente clara en la normativa cargada.";

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