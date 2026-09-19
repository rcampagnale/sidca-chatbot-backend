import { Resend } from "resend";
import crypto from "node:crypto";

export type ReaffiliationPendingEmailInput = {
  dni: string;
  nombre: string;
  email: string;
  fechaSolicitud: string;
  nroAfiliacion?: string | number | null;
  solicitudId: string;
};

function escapeHtml(value: unknown): string {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function optionalUrl(name: string): string | null {
  const value = String(process.env[name] || "").trim();
  return /^https:\/\//i.test(value) ? value : null;
}

function link(url: string, label: string): string {
  return `<a href="${escapeHtml(url)}" style="color:#005CFE;text-decoration:none;">${escapeHtml(label)}</a>`;
}

function renderLogoRow(): string {
  const logos = [
    ["SIDCA_LOGO_URL", "SIDCA"],
    ["CEA_LOGO_URL", "CEA"],
    ["IE_LOGO_URL", "IE"],
    ["CGT_LOGO_URL", "CGT"],
  ]
    .map(([envName, alt]) => {
      const url = optionalUrl(envName);
      return url
        ? `<td style="padding:0 8px;text-align:center;"><img src="${escapeHtml(url)}" alt="${alt}" style="max-width:120px;max-height:55px;height:auto;border:0;" /></td>`
        : "";
    })
    .join("");

  return logos
    ? `<table role="presentation" align="center" cellpadding="0" cellspacing="0" border="0"><tr>${logos}</tr></table>`
    : "";
}

function stepCard(number: string, title: string, body: string): string {
  return `<tr><td style="padding:7px 0;font-family:Arial,sans-serif;color:#09232B;font-size:16px;line-height:1.5;"><table role="presentation" cellpadding="0" cellspacing="0" border="0"><tr><td valign="top" style="width:32px;height:32px;border-radius:50%;background:#FFAA00;color:#09232B;text-align:center;font-weight:700;font-size:16px;line-height:32px;">${escapeHtml(number)}</td><td style="padding-left:12px;"><strong>${escapeHtml(title)}</strong><br />${escapeHtml(body)}</td></tr></table></td></tr>`;
}

export function buildReaffiliationPendingEmailHtml(
  input: Omit<ReaffiliationPendingEmailInput, "email" | "solicitudId">,
): string {
  const nombre = escapeHtml(input.nombre);
  const dni = escapeHtml(input.dni);
  const fechaSolicitud = escapeHtml(input.fechaSolicitud);
  const nroAfiliacion = String(input.nroAfiliacion ?? "").trim();
  const nroBlock = nroAfiliacion
    ? `<tr><td style="padding-top:8px;"><strong>N.º de afiliación histórico:</strong> ${escapeHtml(nroAfiliacion)}</td></tr>`
    : "";
  const logoRow = renderLogoRow();

  return `<!doctype html><html lang="es"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width,initial-scale=1" /></head>
<body style="margin:0;padding:0;background:#09232B;"><table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background:#09232B;"><tr><td align="center" style="padding:22px 10px;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="max-width:700px;background:#FFFFFF;border-radius:10px;overflow:hidden;">
<tr><td style="padding:28px 26px;background:#FFAA00;color:#09232B;text-align:center;font-family:Arial,sans-serif;">${logoRow}<div style="font-size:15px;font-weight:700;letter-spacing:1px;margin-top:${logoRow ? "16px" : "0"};">SIDCA · TU SINDICATO</div><div style="font-size:28px;font-weight:700;margin-top:10px;">Solicitud de reafiliación recibida</div><p style="margin:14px auto 0;max-width:560px;font-size:17px;line-height:1.5;">Recibimos correctamente tu solicitud para volver a formar parte de SIDCA. Ahora inicia la etapa de revisión y aprobación.</p></td></tr>
<tr><td style="padding:28px 26px 10px;font-family:Arial,sans-serif;color:#09232B;font-size:17px;line-height:1.55;"><p style="margin:0 0 12px;">Hola, <strong>${nombre}</strong>:</p><p style="margin:0;">Tu solicitud de reafiliación fue registrada correctamente.</p></td></tr>
<tr><td style="padding:18px 26px;font-family:Arial,sans-serif;"><table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background:#FFF4D6;border-left:5px solid #FFAA00;border-radius:6px;"><tr><td style="padding:18px 20px;color:#09232B;font-size:17px;line-height:1.55;"><strong style="font-size:20px;">Pendiente de aprobación</strong><br />Tu solicitud se encuentra en revisión. En esta etapa todavía no implica una aprobación ni la reactivación automática de tu afiliación.<table role="presentation" cellpadding="0" cellspacing="0" border="0" style="margin-top:14px;"><tr><td><strong>Fecha de solicitud:</strong> ${fechaSolicitud}</td></tr><tr><td style="padding-top:8px;"><strong>DNI:</strong> ${dni}</td></tr>${nroBlock}</table></td></tr></table></td></tr>
<tr><td style="padding:0 26px 18px;font-family:Arial,sans-serif;color:#09232B;font-size:16px;line-height:1.55;"><strong>Importante:</strong> mientras la solicitud esté pendiente, tu afiliación continúa inactiva. Los servicios y accesos correspondientes a una afiliación activa se habilitarán únicamente en caso de aprobación.</td></tr>
<tr><td style="padding:4px 26px 12px;font-family:Arial,sans-serif;color:#09232B;"><h2 style="margin:0 0 8px;text-align:center;font-size:22px;">Qué sucede ahora</h2><table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">${stepCard("1", "Revisamos tu solicitud", "SIDCA verifica los datos presentados y el antecedente de afiliación correspondiente.")}${stepCard("2", "Se emite una resolución", "La solicitud podrá ser aprobada o rechazada de acuerdo con la revisión realizada.")}${stepCard("3", "Te avisamos por correo", "Recibirás un nuevo email con el resultado de tu solicitud. No es necesario iniciar otra gestión mientras esté pendiente.")}</table></td></tr>
<tr><td style="padding:18px 26px;font-family:Arial,sans-serif;"><table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background:#EAF2FF;border-left:5px solid #005CFE;border-radius:6px;"><tr><td style="padding:18px 20px;color:#09232B;font-size:16px;line-height:1.55;"><strong style="font-size:19px;">No es necesario volver a solicitar la reafiliación</strong><br />Tu solicitud ya quedó registrada. Conservaremos esta presentación hasta que finalice la revisión. Te enviaremos un nuevo correo cuando la solicitud sea aprobada o rechazada.</td></tr></table></td></tr>
<tr><td style="padding:10px 26px 22px;font-family:Arial,sans-serif;color:#09232B;text-align:center;font-size:16px;line-height:1.7;"><strong>SIDCA Gestión / Adherentes</strong><br />${link("https://wa.me/5493834230813", "WhatsApp: 3834 230813")}<br /><span>Atención por WhatsApp · sólo mensajes.</span></td></tr>
<tr><td style="padding:8px 26px 24px;font-family:Arial,sans-serif;color:#09232B;text-align:center;font-size:15px;line-height:1.7;"><strong>Seguinos y conocé más</strong><br />${link("https://sidcagremio.com/", "Sitio web")} · ${link("https://www.facebook.com/sidca.catamarca.7", "Facebook")} · ${link("https://www.youtube.com/@RadioSiDCa", "YouTube")} · ${link("https://www.instagram.com/sidcagremio/", "Instagram")}<p style="margin:22px 0 0;">Gracias por volver a elegir SIDCA. Una vez finalizada la revisión, recibirás la resolución de tu solicitud por este mismo medio.</p><p style="margin:18px 0 0;font-size:13px;">SIDCA · Sindicato Docente Catamarca<br />Ayacucho 227 · 1.º piso · San Fernando del Valle de Catamarca<br />Inscripción Gremial 2902</p></td></tr>
</table></td></tr></table></body></html>`;
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
  const solicitudHash = crypto.createHash("sha256").update(input.solicitudId).digest("hex").slice(0, 24);
  const response = await resend.emails.send(
    {
      from: config.from,
      to: input.email,
      subject: "Recibimos tu solicitud de reafiliación a SIDCA",
      ...(config.replyTo ? { replyTo: config.replyTo } : {}),
      html: buildReaffiliationPendingEmailHtml(input),
    },
    { idempotencyKey: `sidca-reafiliacion-pendiente-${input.dni}-${solicitudHash}` },
  );

  if (response.error || !response.data?.id) {
    throw new Error(response.error?.message || "Resend no devolvió un identificador de envío.");
  }

  return response.data.id;
}
