import crypto from "node:crypto";
import { Resend } from "resend";

export type ReaffiliationApprovedEmailInput = {
  dni: string;
  nombre: string;
  email: string;
  fechaAprobacion: string;
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

function bullet(title: string, body: string): string {
  return `<tr><td valign="top" style="width:14px;color:#005CFE;font-size:22px;line-height:1.25;">•</td><td style="padding:0 0 10px 8px;"><strong>${escapeHtml(title)}</strong>${body ? `<br />${escapeHtml(body)}` : ""}</td></tr>`;
}

export function buildReaffiliationApprovedEmailHtml(
  input: Omit<ReaffiliationApprovedEmailInput, "email" | "solicitudId">,
): string {
  const nombre = escapeHtml(input.nombre);
  const dni = escapeHtml(input.dni);
  const fechaAprobacion = escapeHtml(input.fechaAprobacion);
  const nroAfiliacion = String(input.nroAfiliacion ?? "").trim();
  const nroBlock = nroAfiliacion
    ? `<tr><td style="padding-top:8px;"><strong>N.º de afiliación:</strong> ${escapeHtml(nroAfiliacion)}</td></tr>`
    : "";
  const logoRow = renderLogoRow();

  return `<!doctype html><html lang="es"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width,initial-scale=1" /></head>
<body style="margin:0;padding:0;background:#09232B;"><table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background:#09232B;"><tr><td align="center" style="padding:22px 10px;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="max-width:700px;background:#FFFFFF;border-radius:10px;overflow:hidden;">
<tr><td style="padding:28px 26px;background:#FFAA00;color:#09232B;text-align:center;font-family:Arial,sans-serif;">${logoRow}<div style="font-size:15px;font-weight:700;letter-spacing:1px;margin-top:${logoRow ? "16px" : "0"};">SIDCA · TU SINDICATO</div><div style="font-size:28px;font-weight:700;margin-top:10px;">Reafiliación aprobada</div><p style="margin:14px auto 0;max-width:560px;font-size:17px;line-height:1.5;">Tu solicitud de reafiliación a SIDCA – Sindicato Docente Catamarca fue aprobada.</p></td></tr>
<tr><td style="padding:28px 26px 10px;font-family:Arial,sans-serif;color:#09232B;font-size:17px;line-height:1.55;"><p style="margin:0 0 12px;">Hola, <strong>${nombre}</strong>:</p><p style="margin:0 0 12px;">A partir de este momento, tu afiliación vuelve a encontrarse activa.</p><p style="margin:0;">Podés volver a utilizar los servicios, beneficios y herramientas disponibles para afiliados.</p></td></tr>
<tr><td style="padding:18px 26px;font-family:Arial,sans-serif;"><table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background:#E8F8ED;border-left:5px solid #1A8F55;border-radius:6px;"><tr><td style="padding:18px 20px;color:#09232B;font-size:17px;line-height:1.55;"><strong style="font-size:20px;">ESTADO DE TU AFILIACIÓN</strong><br /><strong style="display:inline-block;margin-top:8px;color:#1A8F55;font-size:22px;">ACTIVA</strong><table role="presentation" cellpadding="0" cellspacing="0" border="0" style="margin-top:14px;"><tr><td><strong>Fecha de aprobación:</strong> ${fechaAprobacion}</td></tr><tr><td style="padding-top:8px;"><strong>DNI:</strong> ${dni}</td></tr>${nroBlock}</table></td></tr></table></td></tr>
<tr><td style="padding:4px 26px 12px;font-family:Arial,sans-serif;color:#09232B;"><h2 style="margin:0 0 10px;text-align:center;font-size:22px;">Ya podés volver a utilizar la APP SIDCA</h2><table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="font-family:Arial,sans-serif;color:#09232B;font-size:16px;line-height:1.5;">${bullet("Credencial digital", "")}${bullet("Convenios y descuentos", "")}${bullet("Cursos y capacitaciones", "")}${bullet("Certificados y constancias", "")}${bullet("Registro de asistencia", "")}${bullet("Aula Virtual SIDCA", "")}</table></td></tr>
<tr><td style="padding:18px 26px 10px;font-family:Arial,sans-serif;color:#09232B;"><h2 style="margin:0 0 10px;text-align:center;font-size:22px;">Servicios y beneficios</h2><table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="font-size:16px;line-height:1.5;">${bullet("Capacitaciones", "")}${bullet("Turismo y viajes", "")}${bullet("Convenios con empresas", "")}${bullet("Convenios hoteleros", "")}${bullet("Casa del Docente", "")}${bullet("Médica gremial", "")}${bullet("Simulador de sueldo", "")}${bullet("Asesoramiento Gremial", "")}${bullet("Departamento Jurídico", "")}</table></td></tr>
<tr><td style="padding:14px 26px 22px;font-family:Arial,sans-serif;color:#09232B;text-align:center;font-size:15px;line-height:1.7;"><strong>Contactos SIDCA</strong><br />${link("https://wa.me/5493834051983", "Asesoramiento Gremial · 3834 051983")}<br />${link("https://wa.me/5493834397239", "Departamento Jurídico · 3834 397239")}<br />${link("https://wa.me/5493834230813", "SIDCA Gestión / Adherentes · 3834 230813")}<br />${link("https://wa.me/5493834283151", "SIDCA Turismo · 3834 283151")}<br />${link("https://wa.me/5493834250139", "Casa del Docente · 3834 250139")}<br />${link("https://wa.me/5493832437803", "Soporte Técnico APP / Aula Virtual · 3832 437803")}<br />${link("https://wa.me/5493834023970", "Médica Gremial · 3834 023970")}<br />${link("https://wa.me/5493834782864", "Radio SIDCA · 3834 782864")}</td></tr>
<tr><td style="padding:8px 26px 24px;font-family:Arial,sans-serif;color:#09232B;text-align:center;font-size:15px;line-height:1.7;"><strong>Seguinos y conocé más</strong><br />${link("https://sidcagremio.com/", "Sitio web")} · ${link("https://www.facebook.com/sidca.catamarca.7", "Facebook")} · ${link("https://www.youtube.com/@RadioSiDCa", "YouTube")} · ${link("https://www.instagram.com/sidcagremio/", "Instagram")}<p style="margin:22px 0 0;">Nos alegra volver a contar con vos como afiliado/a de SIDCA.</p><p style="margin:18px 0 0;font-size:13px;">SIDCA · Sindicato Docente Catamarca<br />Ayacucho 227 · 1.º piso · San Fernando del Valle de Catamarca<br />Inscripción Gremial 2902</p></td></tr>
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

export async function sendReaffiliationApprovedEmail(input: ReaffiliationApprovedEmailInput): Promise<string> {
  const config = requireEmailConfig();
  const resend = new Resend(config.apiKey);
  const solicitudHash = crypto.createHash("sha256").update(input.solicitudId).digest("hex").slice(0, 24);
  const response = await resend.emails.send(
    {
      from: config.from,
      to: input.email,
      subject: "Tu reafiliación a SIDCA fue aprobada",
      ...(config.replyTo ? { replyTo: config.replyTo } : {}),
      html: buildReaffiliationApprovedEmailHtml(input),
    },
    { idempotencyKey: `sidca-reafiliacion-aprobada-${input.dni}-${solicitudHash}` },
  );

  if (response.error || !response.data?.id) {
    throw new Error(response.error?.message || "Resend no devolvió un identificador de envío.");
  }

  return response.data.id;
}
