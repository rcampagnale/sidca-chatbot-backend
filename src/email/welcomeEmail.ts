import { Resend } from "resend";
import {
  FRASE_CIERRE,
  HIGHLIGHTS_APP,
  SECCIONES_SERVICIOS,
} from "./contenidoInstitucional.js";
import {
  buildInstitutionalEmail,
  type InstitutionalEmail,
} from "./institutionalEmail.js";

export type WelcomeEmailInput = {
  usuarioId: string;
  email: string;
  nombre: string;
  dni: string;
};

function buildWelcomeEmail(input: Pick<WelcomeEmailInput, "nombre" | "dni">): InstitutionalEmail {
  return buildInstitutionalEmail({
    subject: "SiDCA | Te damos la bienvenida",
    preheader: "Tu afiliación fue aprobada: ya sos parte de SiDCA.",
    title: "Te damos la bienvenida al Sindicato de Docentes de Catamarca",
    state: "approved",
    // La bienvenida pasó al título, así que la etiqueta vuelve a informar el
    // estado del trámite en vez de repetir la misma frase dos veces.
    stateLabel: "Afiliación aprobada",
    greetingName: input.nombre,
    paragraphs: [
      "Tu afiliación a SiDCA fue registrada correctamente.",
      "Desde hoy formás parte de una organización que representa, acompaña y trabaja junto a las y los docentes de Catamarca.",
    ],
    // Sin caja "Datos de la solicitud": el DNI ya aparece donde es útil, en
    // el bloque de acceso a la APP.
    appAccess: { dni: input.dni },
    // Servicios y prestaciones: viven en contenidoInstitucional.ts para que
    // este correo y el de reafiliación aprobada nunca queden desalineados.
    sections: SECCIONES_SERVICIOS,
    highlights: HIGHLIGHTS_APP,
    closing: {
      titulo: "Gracias por formar parte de SiDCa.",
      texto: FRASE_CIERRE,
    },
  });
}

export function buildWelcomeEmailHtml(input: Omit<WelcomeEmailInput, "usuarioId">): string {
  return buildWelcomeEmail(input).html;
}

function requireEmailConfig(): { apiKey: string; from: string; replyTo?: string } {
  const apiKey = String(process.env.RESEND_API_KEY || "").trim();
  const from = String(process.env.SIDCA_EMAIL_FROM || "").trim();
  const replyTo = String(process.env.SIDCA_EMAIL_REPLY_TO || "").trim();
  if (!apiKey) throw new Error("Falta configurar RESEND_API_KEY.");
  if (!from) throw new Error("Falta configurar SIDCA_EMAIL_FROM.");
  return { apiKey, from, ...(replyTo ? { replyTo } : {}) };
}

export async function sendWelcomeEmail(input: WelcomeEmailInput): Promise<string> {
  const config = requireEmailConfig();
  const resend = new Resend(config.apiKey);
  const email = buildWelcomeEmail(input);
  const response = await resend.emails.send(
    {
      from: config.from,
      to: input.email,
      subject: "SiDCA | Te damos la bienvenida",
      ...(config.replyTo ? { replyTo: config.replyTo } : {}),
      html: email.html,
      text: email.text,
      attachments: email.attachments,
    },
    { idempotencyKey: `sidca-welcome-${input.usuarioId}` },
  );

  if (response.error || !response.data?.id) {
    throw new Error(response.error?.message || "Resend no devolvió un identificador de envío.");
  }

  return response.data.id;
}
