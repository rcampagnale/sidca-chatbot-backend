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

export async function runChatbotFileSearch(
  input: ChatbotQueryInput
): Promise<ChatbotQueryOutput> {
  const client = getOpenAIClient();
  const vectorStoreId = getRequiredEnv("OPENAI_VECTOR_STORE_ID");
  const model = process.env.OPENAI_MODEL?.trim() || "gpt-4.1";

  const tools: any[] = [
    {
      type: "file_search",
      vector_store_ids: [vectorStoreId],
      max_num_results: input.maxResults ?? 4,
      filters: input.dominio
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

  const response: any = await client.responses.create({
    model,
    input: [
      {
        role: "system",
        content: [
          {
            type: "input_text",
            text:
              "Eres un asistente del SIDCA. Responde únicamente con base en la normativa docente provincial recuperada desde el buscador documental. Si la información no está en los documentos recuperados, indícalo claramente. No inventes artículos, plazos, autoridades ni procedimientos. Responde en español claro y útil para docentes.",
          },
        ],
      },
      {
        role: "user",
        content: [
          {
            type: "input_text",
            text: input.pregunta,
          },
        ],
      },
    ],
    tools,
    include: ["file_search_call.results"],
  });

  const outputText =
    typeof response.output_text === "string" && response.output_text.trim()
      ? response.output_text.trim()
      : "No encontré una respuesta suficientemente clara en la normativa cargada.";

  const referencias = new Set<string>();
  const busqueda: Array<{ filename?: string; score?: number; text?: string }> = [];

  if (Array.isArray(response.output)) {
    for (const item of response.output) {
      if (item?.type === "message" && Array.isArray(item?.content)) {
        for (const contentItem of item.content) {
          if (Array.isArray(contentItem?.annotations)) {
            for (const ann of contentItem.annotations) {
              if (ann?.filename) referencias.add(String(ann.filename));
            }
          }
        }
      }

      if (item?.type === "file_search_call" && Array.isArray(item?.results)) {
        for (const result of item.results) {
          busqueda.push({
            filename: result?.filename,
            score: typeof result?.score === "number" ? result.score : undefined,
            text:
              typeof result?.text === "string"
                ? result.text
                : Array.isArray(result?.content)
                ? result.content
                    .map((c: any) => (typeof c?.text === "string" ? c.text : ""))
                    .filter(Boolean)
                    .join("\n")
                : undefined,
          });

          if (result?.filename) referencias.add(String(result.filename));
        }
      }
    }
  }

  return {
    ok: true,
    respuesta: outputText,
    referencias: Array.from(referencias),
    busqueda,
  };
}