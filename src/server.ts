import "dotenv/config";
import express from "express";
import cors from "cors";
import multer from "multer";
import OpenAI, { toFile } from "openai";
import { z } from "zod";
import { runChatbotWorkflow } from "./openaiWorkflow.js";

const app = express();
const PORT = Number(process.env.PORT || 8080);

const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 25 * 1024 * 1024,
  },
});

const bodySchema = z.object({
  pregunta: z.string().trim().min(2, "La pregunta es obligatoria"),
  dominio: z
    .enum(["licencias", "estatuto", "general", "coberturas"])
    .optional(),
  maxResults: z.number().int().min(1).max(8).optional(),
});

const mercadoPagoPreferenceSchema = z.object({
  pagoId: z.string().trim().min(1, "El ID del pago es obligatorio"),
  dni: z.string().trim().min(5, "El DNI es obligatorio"),
  afiliadoNombre: z.string().trim().min(2, "El afiliado es obligatorio"),
  concepto: z.string().trim().min(2, "El concepto es obligatorio"),
  detalle: z.string().trim().optional(),
  importe: z.union([z.number(), z.string()]),
});

function getOpenAITranscriptionClient(): OpenAI {
  const apiKey = process.env.OPENAI_API_KEY?.trim();

  if (!apiKey) {
    throw new Error(
      "Falta OPENAI_API_KEY. La consulta del chatbot usa Groq, pero la transcripción de audio todavía requiere OpenAI."
    );
  }

  return new OpenAI({ apiKey });
}

function getMercadoPagoAccessToken(): string {
  const accessToken =
    process.env.MP_ACCESS_TOKEN_TEST?.trim() ||
    process.env.MP_ACCESS_TOKEN?.trim();

  if (!accessToken) {
    throw new Error(
      "Falta configurar MP_ACCESS_TOKEN_TEST o MP_ACCESS_TOKEN en el backend."
    );
  }

  return accessToken;
}

function normalizarImporte(valor: number | string): number {
  if (typeof valor === "number") {
    return valor;
  }

  const limpio = valor
    .replace(/\$/g, "")
    .replace(/\s/g, "")
    .replace(/\./g, "")
    .replace(",", ".");

  return Number(limpio);
}

function getMercadoPagoBackUrls(): {
  success: string;
  pending: string;
  failure: string;
} {
  const success = process.env.MP_BACK_URL_SUCCESS?.trim();
  const pending = process.env.MP_BACK_URL_PENDING?.trim();
  const failure = process.env.MP_BACK_URL_FAILURE?.trim();
  const faltantes = [
    !success ? "MP_BACK_URL_SUCCESS" : null,
    !pending ? "MP_BACK_URL_PENDING" : null,
    !failure ? "MP_BACK_URL_FAILURE" : null,
  ].filter(Boolean);

  if (faltantes.length > 0) {
    throw new Error(
      `Faltan configurar las URL de retorno de Mercado Pago: ${faltantes.join(
        ", "
      )}.`
    );
  }

  return {
    success: success as string,
    pending: pending as string,
    failure: failure as string,
  };
}

app.use(cors());
app.use(express.json({ limit: "1mb" }));

app.get("/health", (_req, res) => {
  res.status(200).json({
    ok: true,
    service: "sidca-chatbot-backend",
    chatbot: "groq_rag",
    endpoint: "/api/chatbot/query",
  });
});

app.post("/api/chatbot/query", async (req, res) => {
  try {
    const input = bodySchema.parse(req.body);
    const result = await runChatbotWorkflow(input);

    res.status(200).json(result);
  } catch (error: any) {
    if (error instanceof z.ZodError) {
      res.status(400).json({
        ok: false,
        tipo: "error",
        dominio: req.body?.dominio || "general",
        origen: "backend",
        consulta: req.body?.pregunta || "",
        consultaNormalizada: req.body?.pregunta || "",
        respuesta: "Solicitud inválida.",
        articulos: [],
        referencias: [],
        busqueda: [],
        conversationId: null,
        error: error.issues.map((i: any) => i.message).join(" | "),
      });
      return;
    }

    console.error("[sidca-chatbot-backend] Error:", error);

    res.status(500).json({
      ok: false,
      tipo: "error",
      dominio: req.body?.dominio || "general",
      origen: "backend",
      consulta: req.body?.pregunta || "",
      consultaNormalizada: req.body?.pregunta || "",
      respuesta: "Servicio no disponible por el momento.",
      articulos: [],
      referencias: [],
      busqueda: [],
      conversationId: null,
      error: error?.message || "Error interno",
    });
  }
});

app.post(["/preference-pago", "/api/pagos/mercadopago/preference"], async (req, res) => {
  try {
    const input = mercadoPagoPreferenceSchema.parse(req.body);
    const importe = normalizarImporte(input.importe);
    const backUrls = getMercadoPagoBackUrls();

    if (!Number.isFinite(importe) || importe <= 0) {
      res.status(400).json({
        ok: false,
        error: "El importe debe ser un número mayor a cero.",
      });
      return;
    }

    const preferenceBody = {
      items: [
        {
          id: input.pagoId,
          title: input.concepto,
          description: input.detalle || input.concepto,
          quantity: 1,
          currency_id: "ARS",
          unit_price: importe,
        },
      ],
      payer: {
        name: input.afiliadoNombre,
        identification: {
          type: "DNI",
          number: input.dni,
        },
      },
      external_reference: input.pagoId,
      metadata: {
        pagoId: input.pagoId,
        dni: input.dni,
        afiliadoNombre: input.afiliadoNombre,
        concepto: input.concepto,
      },
      back_urls: backUrls,
      auto_return: "approved",
      notification_url: process.env.MP_WEBHOOK_URL?.trim() || undefined,
      statement_descriptor: "SIDCA",
    };

    const mpResponse = await fetch(
      "https://api.mercadopago.com/checkout/preferences",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${getMercadoPagoAccessToken()}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(preferenceBody),
      }
    );

    const data = await mpResponse.json();

    if (!mpResponse.ok) {
      console.error("[sidca-chatbot-backend] Mercado Pago error:", data);
      res.status(mpResponse.status).json({
        ok: false,
        error: "No se pudo crear la preferencia de Mercado Pago.",
        detalle: data?.message || data?.error || "Error de Mercado Pago",
      });
      return;
    }

    res.status(200).json({
      ok: true,
      preferenceId: data.id,
      init_point: data.init_point,
      sandbox_init_point: data.sandbox_init_point,
    });
  } catch (error: any) {
    if (error instanceof z.ZodError) {
      res.status(400).json({
        ok: false,
        error: error.issues.map((i: any) => i.message).join(" | "),
      });
      return;
    }

    console.error("[sidca-chatbot-backend] Error MP preference:", error);
    res.status(500).json({
      ok: false,
      error: error?.message || "No se pudo preparar el pago.",
    });
  }
});

app.post(
  "/api/chatbot/transcribe",
  upload.single("audio"),
  async (req, res) => {
    try {
      if (!req.file) {
        res.status(400).json({
          ok: false,
          error: "No se recibió ningún archivo de audio.",
        });
        return;
      }

      const openai = getOpenAITranscriptionClient();

      const audioFile = await toFile(
        req.file.buffer,
        req.file.originalname || "audio.m4a",
        {
          type: req.file.mimetype || "audio/m4a",
        }
      );

      const transcription = await openai.audio.transcriptions.create({
        file: audioFile,
        model: process.env.OPENAI_TRANSCRIBE_MODEL || "gpt-4o-mini-transcribe",
        language: "es",
      });

      res.status(200).json({
        ok: true,
        texto: transcription.text?.trim() || "",
      });
    } catch (error: any) {
      console.error("[sidca-chatbot-backend] Error STT:", error);

      res.status(500).json({
        ok: false,
        error: error?.message || "No se pudo transcribir el audio.",
      });
    }
  }
);

app.listen(PORT, "0.0.0.0", () => {
  console.log(`[sidca-chatbot-backend] running on http://0.0.0.0:${PORT}`);
  console.log("[sidca-chatbot-backend] chatbot mode: groq_rag");
});
