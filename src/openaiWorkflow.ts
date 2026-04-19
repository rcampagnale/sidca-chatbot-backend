import { runWorkflow } from "./sidcaAgentWorkflow.js";

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

type SearchHit = {
  filename?: string;
  score?: number;
  text?: string;
};

type BackendArticulo = {
  capitulo?: string;
  titulo_capitulo?: string;
  articulo?: number | string;
  titulo_articulo?: string;
  texto?: string;
  resumen?: string | null;
  [key: string]: any;
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

function extraerRespuestaDesdeOutput(result: any): string {
  if (typeof result?.output_parsed?.respuesta === "string") {
    return result.output_parsed.respuesta.trim();
  }

  if (typeof result?.output_text === "string" && result.output_text.trim()) {
    try {
      const parsed = JSON.parse(result.output_text);
      if (typeof parsed?.respuesta === "string") {
        return parsed.respuesta.trim();
      }
    } catch {
      return result.output_text.trim();
    }
    return result.output_text.trim();
  }

  return "";
}

export async function runChatbotWorkflow(
  input: ChatbotQueryInput
): Promise<ChatbotWorkflowOutput> {
  try {
    const result = await runWorkflow({
      input_as_text: input.pregunta,
    });

    const respuesta =
      extraerRespuestaDesdeOutput(result) ||
      "No encontré fragmentos suficientes en la base cargada para responder con precisión a esta consulta.";

    return {
      ok: true,
      tipo: "respuesta_ia",
      dominio: dominioSeguro(input.dominio),
      origen: "agent_builder_sdk",
      consulta: input.pregunta,
      consultaNormalizada: input.pregunta,
      respuesta,
      articulos: [],
      referencias: [],
      busqueda: [],
      conversationId: null,
    };
  } catch (error: any) {
    console.error("[sidca-agent-sdk] Error:", error);

    return {
      ok: false,
      tipo: "error",
      dominio: dominioSeguro(input.dominio),
      origen: "agent_builder_sdk",
      consulta: input.pregunta,
      consultaNormalizada: input.pregunta,
      respuesta: "Hubo un problema al consultar el agente publicado.",
      articulos: [],
      referencias: [],
      busqueda: [],
      conversationId: null,
      error: error?.message || "Error interno",
    };
  }
}