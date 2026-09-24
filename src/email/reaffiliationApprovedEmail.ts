import crypto from "node:crypto";
import { Resend } from "resend";
import {
  FRASE_CIERRE,
  HIGHLIGHTS_APP,
  SECCIONES_SERVICIOS,
} from "./contenidoInstitucional.js";
import { buildInstitutionalEmail, type InstitutionalEmail } from "./institutionalEmail.js";

export type ReaffiliationApprovedEmailInput = {
  dni: string;
  nombre: string;
  email: string;
  fechaAprobacion: string;
  nroAfiliacion?: string | number | null;
  solicitudId: string;
};

export const REAFFILIATION_APPROVED_EMAIL_SUBJECT = "SiDCA | Reafiliación aprobada";

export function buildReaffiliationApprovedEmail(input: Omit<ReaffiliationApprovedEmailInput, "email" | "solicitudId">): InstitutionalEmail {
  return buildInstitutionalEmail({
    subject: REAFFILIATION_APPROVED_EMAIL_SUBJECT,
    preheader: "Tu reafiliación fue aprobada: tu afiliación está activa nuevamente.",
    title: "Tu afiliación a SiDCA está activa nuevamente",
    state: "approved",
    // El título ya anuncia la novedad; la etiqueta informa el estado del
    // trámite sin repetir la misma frase dos veces seguidas.
    stateLabel: "Reafiliación aprobada",
    greetingName: input.nombre,
    paragraphs: [
      "Tu solicitud de reafiliación a SiDCA fue aprobada.",
      "Tu afiliación vuelve a estar activa y ya tenés disponibles todos los servicios, beneficios y herramientas para afiliados y afiliadas.",
    ],
    details: [
      { label: "DNI", value: input.dni },
      { label: "Fecha de aprobación", value: input.fechaAprobacion },
    ],
    appAccess: { dni: input.dni },
    // La afiliación está activa otra vez, así que acá sí corresponde detallar
    // las prestaciones que el párrafo anterior anuncia. Mismo contenido que
    // la bienvenida, tomado del módulo compartido.
    sections: SECCIONES_SERVICIOS,
    highlights: HIGHLIGHTS_APP,
    closing: {
      titulo: "Gracias por volver a formar parte de SiDCa.",
      texto: FRASE_CIERRE,
    },
  });
}

export function buildReaffiliationApprovedEmailHtml(
  input: Omit<ReaffiliationApprovedEmailInput, "email" | "solicitudId">,
): string {
  return buildReaffiliationApprovedEmail(input).html;
}

function requireEmailConfig(): { apiKey: string; from: string; replyTo?: string } {
  const apiKey = String(process.env.RESEND_API_KEY || "").trim();
  const from = String(process.env.SIDCA_EMAIL_FROM || "").trim();
  const replyTo = String(process.env.SIDCA_EMAIL_REPLY_TO || "").trim();
  if (!apiKey) throw new Error("Falta configurar RESEND_API_KEY.");
  if (!from) throw new Error("Falta configurar SIDCA_EMAIL_FROM.");
  return { apiKey, from, ...(replyTo ? { replyTo } : {}) };
}

export async function sendReaffiliationApprovedEmail(input: ReaffiliationApprovedEmailInput): Promise<string> {
  const config = requireEmailConfig();
  const resend = new Resend(config.apiKey);
  const email = buildReaffiliationApprovedEmail(input);
  const solicitudHash = crypto.createHash("sha256").update(input.solicitudId).digest("hex").slice(0, 24);
  const response = await resend.emails.send(
    {
      from: config.from,
      to: input.email,
      subject: REAFFILIATION_APPROVED_EMAIL_SUBJECT,
      ...(config.replyTo ? { replyTo: config.replyTo } : {}),
      html: email.html,
      text: email.text,
      attachments: email.attachments,
    },
    { idempotencyKey: `sidca-reafiliacion-aprobada-${input.dni}-${solicitudHash}` },
  );

  if (response.error || !response.data?.id) {
    throw new Error(response.error?.message || "Resend no devolvió un identificador de envío.");
  }

  return response.data.id;
}
