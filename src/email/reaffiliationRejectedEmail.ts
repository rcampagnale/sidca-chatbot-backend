import crypto from "node:crypto";
import { Resend } from "resend";
import { buildInstitutionalEmail, type InstitutionalEmail } from "./institutionalEmail.js";

export type ReaffiliationRejectedEmailInput = {
  dni: string;
  nombre: string;
  email: string;
  fechaResolucion: string;
  nroAfiliacion?: string | number | null;
  motivo?: string | null;
  solicitudId: string;
};

export const REAFFILIATION_REJECTED_EMAIL_SUBJECT = "SiDCA | Actualización de tu reafiliación";

export function buildReaffiliationRejectedEmail(input: Omit<ReaffiliationRejectedEmailInput, "email" | "solicitudId">): InstitutionalEmail {
  return buildInstitutionalEmail({
    subject: REAFFILIATION_REJECTED_EMAIL_SUBJECT,
    preheader: "Tenemos una actualización sobre tu solicitud de reafiliación.",
    title: "Tu solicitud de reafiliación no fue aprobada",
    state: "rejected",
    greetingName: input.nombre,
    paragraphs: [
      // "Comisión del Sindicato" es quien revisa la solicitud según el correo
      // de trámite en curso; nombrarla también acá cierra el mismo relato.
      "Luego de la revisión correspondiente, la Comisión del Sindicato no aprobó tu solicitud de reafiliación.",
      "Tu afiliación continúa inactiva.",
    ],
    // El motivo se muestra antes que los datos administrativos: es lo que la
    // persona necesita leer primero.
    observation: input.motivo,
    details: [
      { label: "DNI", value: input.dni },
      { label: "Fecha de resolución", value: input.fechaResolucion },
    ],
    closing: {
      titulo: "¿Necesitás hacer una consulta?",
      texto:
        "Podés comunicarte con SiDCA por los canales oficiales o acercarte a la Sede Central para aclarar tu situación.",
    },
  });
}

export function buildReaffiliationRejectedEmailHtml(
  input: Omit<ReaffiliationRejectedEmailInput, "email" | "solicitudId">,
): string {
  return buildReaffiliationRejectedEmail(input).html;
}

function requireEmailConfig(): { apiKey: string; from: string; replyTo?: string } {
  const apiKey = String(process.env.RESEND_API_KEY || "").trim();
  const from = String(process.env.SIDCA_EMAIL_FROM || "").trim();
  const replyTo = String(process.env.SIDCA_EMAIL_REPLY_TO || "").trim();
  if (!apiKey) throw new Error("Falta configurar RESEND_API_KEY.");
  if (!from) throw new Error("Falta configurar SIDCA_EMAIL_FROM.");
  return { apiKey, from, ...(replyTo ? { replyTo } : {}) };
}

export async function sendReaffiliationRejectedEmail(input: ReaffiliationRejectedEmailInput): Promise<string> {
  const config = requireEmailConfig();
  const resend = new Resend(config.apiKey);
  const email = buildReaffiliationRejectedEmail(input);
  const solicitudHash = crypto.createHash("sha256").update(input.solicitudId).digest("hex").slice(0, 24);
  const response = await resend.emails.send(
    {
      from: config.from,
      to: input.email,
      subject: REAFFILIATION_REJECTED_EMAIL_SUBJECT,
      ...(config.replyTo ? { replyTo: config.replyTo } : {}),
      html: email.html,
      text: email.text,
      attachments: email.attachments,
    },
    { idempotencyKey: `sidca-reafiliacion-rechazada-${input.dni}-${solicitudHash}` },
  );

  if (response.error || !response.data?.id) {
    throw new Error(response.error?.message || "Resend no devolvió un identificador de envío.");
  }

  return response.data.id;
}
