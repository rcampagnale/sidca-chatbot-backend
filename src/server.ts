import express from "express";
import cors from "cors";
import { z } from "zod";
import { runChatbotFileSearch } from "./openaiFileSearch.js";

const app = express();
const PORT = Number(process.env.PORT || 8080);

const bodySchema = z.object({
  pregunta: z.string().trim().min(2, "La pregunta es obligatoria"),
  dominio: z.enum(["licencias", "estatuto", "general"]).optional(),
  maxResults: z.number().int().min(1).max(8).optional(),
});

app.use(cors());
app.use(express.json({ limit: "1mb" }));

app.get("/health", (_req, res) => {
  res.status(200).json({
    ok: true,
    service: "sidca-chatbot-backend",
  });
});

app.post("/api/chatbot/query", async (req, res) => {
  try {
    const input = bodySchema.parse(req.body);
    const result = await runChatbotFileSearch(input);
    res.status(200).json(result);
  } catch (error: any) {
    if (error instanceof z.ZodError) {
      res.status(400).json({
        ok: false,
        respuesta: "Solicitud inválida.",
        referencias: [],
        busqueda: [],
        error: error.issues.map((i) => i.message).join(" | "),
      });
      return;
    }

    console.error("[sidca-chatbot-backend] Error:", error);

    res.status(500).json({
      ok: false,
      respuesta: "Hubo un problema al consultar la IA.",
      referencias: [],
      busqueda: [],
      error: error?.message || "Error interno",
    });
  }
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`[sidca-chatbot-backend] running on http://0.0.0.0:${PORT}`);
});