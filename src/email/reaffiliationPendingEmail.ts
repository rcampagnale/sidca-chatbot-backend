import crypto from "node:crypto";
import { Resend } from "resend";
import { buildInstitutionalEmail, type InstitutionalEmail } from "./institutionalEmail.js";

export type ReaffiliationPendingEmailInput = {
  dni: string;
  nombre: string;
  email: string;
  fechaSolicitud: string;
  nroAfiliacion?: string | number | null;
  solicitudId: string;
};

export const REAFFILIATION_PENDING_EMAIL_SUBJECT = "SiDCA | Solicitud de reafiliación recibida";

export function buildReaffiliationPendingEmail(input: Omit<ReaffiliationPendingEmailInput, "email" | "solicitudId">): InstitutionalEmail {
  return buildInstitutionalEmail({
    subject: REAFFILIATION_PENDING_EMAIL_SUBJECT,
    preheader: "Recibimos tu solicitud de reafiliación y está en revisión.",
    title: "Recibimos tu solicitud de reafiliación",
    state: "pending",
    greetingName: input.nombre,
    paragraphs: [
      "Tu solicitud de reafiliación a SiDCA quedó registrada correctamente.",
      "Está en proceso de revisión por la Comisión del Sindicato. Abajo te dejamos los datos con los que la recibimos y cómo sigue el trámite.",
    ],
    details: [
      { label: "DNI", value: input.dni },
      { label: "Fecha de solicitud", value: input.fechaSolicitud },
    ],
    // Este correo es de trámite en curso: no corresponde listar beneficios
    // todavía, porque la afiliación aún no está aprobada. Lo que el afiliado
    // necesita saber acá es qué pasa después y que no tiene que rehacer nada.
    sections: [
      {
        titulo: "Cómo sigue el trámite",
        items: [
          {
            icono: "📋",
            nombre: "Revisión de la solicitud",
            detalle:
              "La Comisión del Sindicato verifica los datos que enviaste junto con tu registro histórico.",
          },
          {
            icono: "✉️",
            nombre: "Aviso del resultado",
            detalle:
              "Cuando la solicitud se resuelva te escribimos a esta misma dirección de correo electrónico.",
          },
          {
            icono: "✅",
            nombre: "No tenés que hacer nada más",
            detalle:
              "No hace falta volver a completar el formulario ni enviar la solicitud nuevamente.",
          },
        ],
      },
    ],
    closing: {
      titulo: "¿Tenés una consulta sobre tu trámite?",
      texto:
        "Podés comunicarte con SiDCA por los canales oficiales o acercarte a la Sede Central.",
    },
  });
}

export function buildReaffiliationPendingEmailHtml(
  input: Omit<ReaffiliationPendingEmailInput, "email" | "solicitudId">,
): string {
  return buildReaffiliationPendingEmail(input).html;
}

function requireEmailConfig(): { apiKey: string; from: string; replyTo?: string } {
  const apiKey = String(process.env.RESEND_API_KEY || "").trim();
  const from = String(process.env.SIDCA_EMAIL_FROM || "").trim();
  const replyTo = String(process.env.SIDCA_EMAIL_REPLY_TO || "").trim();
  if (!apiKey) throw new Error("Falta configurar RESEND_API_KEY.");
  if (!from) throw new Error("Falta configurar SIDCA_EMAIL_FROM.");
  return { apiKey, from, ...(replyTo ? { replyTo } : {}) };
}

export async function sendReaffiliationPendingEmail(input: ReaffiliationPendingEmailInput): Promise<string> {
  const config = requireEmailConfig();
  const resend = new Resend(config.apiKey);
  const email = buildReaffiliationPendingEmail(input);
  const solicitudHash = crypto.createHash("sha256").update(input.solicitudId).digest("hex").slice(0, 24);
  const response = await resend.emails.send(
    {
      from: config.from,
      to: input.email,
      subject: REAFFILIATION_PENDING_EMAIL_SUBJECT,
      ...(config.replyTo ? { replyTo: config.replyTo } : {}),
      html: email.html,
      text: email.text,
      attachments: email.attachments,
    },
    { idempotencyKey: `sidca-reafiliacion-pendiente-${input.dni}-${solicitudHash}` },
  );

  if (response.error || !response.data?.id) {
    throw new Error(response.error?.message || "Resend no devolvió un identificador de envío.");
  }

  return response.data.id;
}
