import { Resend } from "resend";

export type WelcomeEmailInput = {
  usuarioId: string;
  email: string;
  nombre: string;
  dni: string;
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

function whatsappLink(number: string): string {
  return `https://wa.me/${number.replace(/\D/g, "")}`;
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

function featureCard(title: string, body: string, emphasized = false): string {
  const background = emphasized ? "#FFF4D6" : "#F3F4F6";
  const border = emphasized ? "#FFAA00" : "#D9DEE7";
  return `<td width="${emphasized ? "50%" : "33%"}" valign="top" style="padding:6px;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background:${background};border:1px solid ${border};border-radius:8px;">
      <tr><td style="padding:14px 12px;font-family:Arial,sans-serif;color:#09232B;font-size:15px;line-height:1.45;">
        <strong>${escapeHtml(title)}</strong><br />${escapeHtml(body)}
      </td></tr>
    </table>
  </td>`;
}

export function buildWelcomeEmailHtml(input: Omit<WelcomeEmailInput, "usuarioId">): string {
  const nombre = escapeHtml(input.nombre || "Afiliado/a SIDCA");
  const dni = escapeHtml(input.dni);
  const logoRow = renderLogoRow();

  return `<!doctype html><html lang="es"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width,initial-scale=1" /></head>
<body style="margin:0;padding:0;background:#09232B;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background:#09232B;">
    <tr><td align="center" style="padding:22px 10px;">
      <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="max-width:700px;background:#FFFFFF;border-radius:10px;overflow:hidden;">
        <tr><td style="padding:28px 26px;background:#FFAA00;color:#09232B;text-align:center;font-family:Arial,sans-serif;">
          ${logoRow}
          <div style="font-size:15px;font-weight:700;letter-spacing:1px;margin-top:${logoRow ? "16px" : "0"};">SIDCA · TU SINDICATO</div>
          <div style="font-size:28px;font-weight:700;margin-top:10px;">¡Bienvenido/a a SiDCa!</div>
          <p style="margin:14px auto 0;max-width:560px;font-size:17px;line-height:1.5;">Desde hoy formás parte de una organización que representa, acompaña y trabaja junto a las y los docentes de Catamarca.</p>
        </td></tr>
        <tr><td style="padding:28px 26px 8px;font-family:Arial,sans-serif;color:#09232B;font-size:17px;line-height:1.55;">
          <p style="margin:0 0 12px;">Hola, <strong>${nombre}</strong>:</p>
          <p style="margin:0 0 12px;">Tu afiliación fue registrada correctamente.</p>
          <p style="margin:0;">Queremos que desde el primer día sepas dónde encontrar ayuda, beneficios y herramientas para tu vida docente.</p>
        </td></tr>
        <tr><td style="padding:18px 26px;font-family:Arial,sans-serif;">
          <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background:#EAF2FF;border-left:5px solid #005CFE;border-radius:6px;"><tr><td style="padding:18px 20px;color:#09232B;font-size:17px;line-height:1.55;">
            <strong style="font-size:20px;">Tu primer paso: ingresá a la APP SIDCA con tu DNI</strong><br />
            Tu número de DNI es tu usuario de acceso.<br /><strong>DNI: ${dni}</strong><br /><br />
            Desde la APP podés consultar tu credencial, cursos, certificados, beneficios, servicios y novedades sindicales.
          </td></tr></table>
        </td></tr>
        <tr><td style="padding:10px 20px 4px;font-family:Arial,sans-serif;color:#09232B;"><h2 style="margin:0;text-align:center;font-size:22px;">Acompañamiento principal</h2></td></tr>
        <tr><td style="padding:6px 20px 14px;"><table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"><tr>
          ${featureCard("Asesoramiento gremial", "Orientación sobre derechos laborales, reclamos, titularización, trámites y gestiones sindicales.", true)}
          ${featureCard("Departamento Jurídico", "Acompañamiento legal ante situaciones vinculadas con la actividad docente y laboral.", true)}
        </tr></table></td></tr>
        <tr><td style="padding:4px 20px;font-family:Arial,sans-serif;color:#09232B;"><h2 style="margin:0;text-align:center;font-size:22px;">Beneficios y servicios</h2></td></tr>
        <tr><td style="padding:6px 20px 18px;"><table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"><tr>
          ${featureCard("Capacitaciones", "Propuestas presenciales y virtuales, cursos, congresos y formación continua para el desarrollo profesional docente.")}
          ${featureCard("Turismo y viajes", "Viajes, propuestas recreativas y beneficios turísticos gestionados por SIDCA.")}
          ${featureCard("Convenios con empresas", "Descuentos y beneficios en empresas y comercios adheridos utilizando tu credencial.")}
        </tr><tr>
          ${featureCard("Convenios hoteleros", "Beneficios de alojamiento y acuerdos especiales con hoteles en diferentes provincias y localidades.")}
          ${featureCard("Casa del Docente", "Alojamiento temporal y servicios destinados a afiliados y afiliadas.")}
          ${featureCard("Médica gremial", "Atención exclusiva para afiliados y afiliadas, con consultas presenciales y virtuales.")}
        </tr><tr>
          ${featureCard("Simulador de sueldo", "Herramienta disponible en la APP SIDCA para estimar y comprender mejor tu liquidación salarial.")}
          <td colspan="2"></td>
        </tr></table></td></tr>
        <tr><td style="padding:18px 26px;background:#09232B;color:#FFFFFF;font-family:Arial,sans-serif;text-align:center;">
          <h2 style="margin:0 0 12px;font-size:22px;">Todo en tu APP SIDCA</h2>
          <p style="margin:0;line-height:1.8;">Credencial digital · Convenios y descuentos · Cursos y capacitaciones<br />Certificados y constancias · Registro de asistencia · Aula Virtual SIDCA</p>
        </td></tr>
        <tr><td style="padding:24px 26px 10px;font-family:Arial,sans-serif;color:#09232B;"><h2 style="margin:0 0 12px;text-align:center;font-size:22px;">Contactos</h2>
          <p style="margin:0;line-height:1.75;text-align:center;">${link(whatsappLink("3834051983"), "Asesoramiento Gremial · WhatsApp 3834 051983")}<br />${link(whatsappLink("3834397239"), "Departamento Jurídico · WhatsApp 3834 397239")}<br />${link(whatsappLink("3834230813"), "SIDCA Gestión / Adherentes · WhatsApp 3834 230813")}<br />${link(whatsappLink("3834283151"), "SIDCA Turismo · WhatsApp 3834 283151")}<br />${link(whatsappLink("3834250139"), "Casa del Docente · WhatsApp 3834 250139")}<br />${link(whatsappLink("3832437803"), "Soporte Técnico APP / Aula Virtual · WhatsApp 3832 437803")}<br />${link(whatsappLink("3834023970"), "Médica Gremial · WhatsApp 3834 023970")}<br />${link(whatsappLink("3834782864"), "Radio SIDCA · WhatsApp 3834 782864")}</p>
        </td></tr>
        <tr><td style="padding:8px 26px 24px;font-family:Arial,sans-serif;color:#09232B;text-align:center;font-size:15px;line-height:1.7;">
          <strong>Seguinos y conocé más</strong><br />${link("https://sidcagremio.com.ar/site/", "Sitio web")} · ${link("https://www.facebook.com/sidca.catamarca.7", "Facebook")} · ${link("https://www.youtube.com/@RadioSiDCa", "YouTube")} · ${link("https://www.instagram.com/sidcagremio/", "Instagram")}
          <p style="margin:22px 0 0;">Sede Central: Ayacucho 227, 1.º piso · San Fernando del Valle de Catamarca · Catamarca CP 4700<br />Correo: ${link("mailto:sidcagremio2023@gmail.com", "sidcagremio2023@gmail.com")}</p>
          <p style="margin:18px 0 0;">Gracias por formar parte de SiDCa.<br />Seguimos construyendo un sindicato presente, cercano y comprometido con quienes educan.</p>
          <p style="margin:18px 0 0;font-size:13px;">SIDCA · Sindicato Docente Catamarca<br />Inscripción Gremial 2902.</p>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body></html>`;
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
  const response = await resend.emails.send(
    {
      from: config.from,
      to: input.email,
      subject: "¡Bienvenido/a a SIDCA!",
      ...(config.replyTo ? { replyTo: config.replyTo } : {}),
      html: buildWelcomeEmailHtml(input),
    },
    { idempotencyKey: `sidca-welcome-${input.usuarioId}` }
  );

  if (response.error || !response.data?.id) {
    throw new Error(response.error?.message || "Resend no devolvió un identificador de envío.");
  }

  return response.data.id;
}
