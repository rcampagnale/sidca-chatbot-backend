import fs from "node:fs";
import path from "node:path";
import type { Attachment } from "resend";

export type InstitutionalEmailState = "pending" | "approved" | "rejected";

/** Grupo de servicios: un título y una lista de prestaciones con detalle. */
export type InstitutionalEmailSection = {
  titulo: string;
  items: Array<{ icono?: string; nombre: string; detalle: string }>;
};

export type InstitutionalEmailOptions = {
  subject: string;
  preheader: string;
  title: string;
  state: InstitutionalEmailState;
  greetingName?: string;
  paragraphs: string[];
  details?: Array<{ label: string; value?: unknown }>;
  appAccess?: { dni: string };
  observation?: string | null;

  /**
   * Reemplaza la etiqueta del estado conservando su color.
   * Lo usa el correo de bienvenida para decir "TE DAMOS LA BIENVENIDA" en
   * lugar de "SOLICITUD APROBADA", sin perder el verde de aprobado.
   */
  stateLabel?: string;

  /** Bloques de servicios y prestaciones. */
  sections?: InstitutionalEmailSection[];

  /** Franja destacada con prestaciones breves separadas por punto medio. */
  highlights?: { titulo: string; items: string[] };

  /** Cierre institucional antes del pie. */
  closing?: { titulo?: string; texto: string };
};

export type InstitutionalEmail = {
  html: string;
  text: string;
  attachments: Attachment[];
};

/**
 * Identidad institucional.
 *
 * Nota: el arte del logo usa "SiDCa" (a final minúscula) mientras que estos
 * textos usan "SiDCA" para conservar una única forma visible de la marca.
 */
const MARCA = {
  nombre: "SiDCA",
  razon: "Sindicato Docente de Catamarca",
  area: "Gestión de Afiliaciones",
  sede: "Ayacucho 227, 1.º piso · San Fernando del Valle de Catamarca · Catamarca CP 4700",
  pieNombre: "SiDCA · Sindicato Docente de Catamarca",
  inscripcion: "Inscripción Gremial 2902",
};

/**
 * Canales oficiales del pie.
 *
 * Sólo se muestran los que tienen URL cargada: no se inventan enlaces.
 *
 * Los iconos son PNG propios generados para el correo (no se enlazan desde
 * servidores de terceros, que suelen quedar bloqueados por los clientes de
 * correo) y viajan adjuntos por CID como los logos institucionales.
 */
const REDES: Array<{
  label: string;
  url: string;
  archivo: string;
  contentId: string;
}> = [
  {
    label: "Sitio web",
    url: "https://sidcagremio.com.ar/site/",
    archivo: "red-web.png",
    contentId: "red-web",
  },
  {
    label: "Facebook",
    url: "https://www.facebook.com/sidca.catamarca.7",
    archivo: "red-facebook.png",
    contentId: "red-facebook",
  },
  {
    label: "YouTube",
    url: "https://www.youtube.com/@RadioSiDCa",
    archivo: "red-youtube.png",
    contentId: "red-youtube",
  },
  {
    label: "Instagram",
    url: "https://www.instagram.com/sidcagremio/",
    archivo: "red-instagram.png",
    contentId: "red-instagram",
  },
];

/** Paleta institucional, nombrada en un único lugar. */
const COLOR = {
  tinta: "#09232B", // verde azulado profundo, base de la identidad
  tintaSuave: "#52606D",
  acento: "#FFAA00", // ámbar institucional
  papel: "#FFFFFF",
  borde: "#DCE3EA",
  fondoSuave: "#F5F7FA",
  appFondo: "#EEF5FF",
  appBarra: "#005CFE",
  appTexto: "#003FAD",
};

/**
 * Tipografías del sistema con Arial como último recurso.
 * Outlook de escritorio ignora esta pila y cae en Arial por la regla mso
 * declarada en el <head>, que es exactamente el comportamiento buscado.
 */
const FUENTE =
  "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif";

/**
 * Presentación de cada estado.
 *
 * El icono dejó de ser un emoji (⌛ / 📱 / ⚠): esos caracteres se dibujan con
 * la fuente de emoji del sistema operativo, así que cambiaban de color y de
 * forma según el cliente y en algunos se veían como un recuadro. Ahora son
 * glifos tipográficos que heredan el color del texto.
 *
 * El estado nunca se comunica sólo por color: siempre hay etiqueta en texto.
 */
const STATE_PRESENTATION: Record<InstitutionalEmailState, {
  label: string;
  icon: string;
  background: string;
  border: string;
  color: string;
}> = {
  pending: {
    label: "SOLICITUD EN REVISIÓN",
    icon: "&#9679;", // ● círculo lleno
    background: "#FFF8E6",
    border: "#E5A900",
    color: "#805B00",
  },
  approved: {
    label: "SOLICITUD APROBADA",
    icon: "&#10003;", // ✓
    background: "#EAF8F0",
    border: "#188A50",
    color: "#12663B",
  },
  rejected: {
    label: "ACTUALIZACIÓN DE TU SOLICITUD",
    icon: "&#33;", // !
    background: "#FFF3F3",
    border: "#A13B47",
    color: "#812D38",
  },
};

const ASSET_DIRECTORY = path.resolve(process.cwd(), "src/email/assets");

function crearAttachmentInline(options: {
  filename: string;
  assetPath: string;
  contentId: string;
}): Attachment {
  if (!fs.existsSync(options.assetPath)) {
    throw new Error(`No existe el asset de email: ${options.assetPath}`);
  }

  const content = fs.readFileSync(options.assetPath);
  if (content.length === 0) {
    throw new Error(`El asset de email está vacío: ${options.assetPath}`);
  }

  return {
    filename: options.filename,
    content,
    contentType: "image/png",
    contentId: options.contentId,
  };
}

export function escapeHtml(value: unknown): string {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

export function primerNombre(nombre: unknown): string {
  const limpio = String(nombre ?? "").trim();
  const despuesDeComa = limpio.includes(",") ? limpio.split(",")[1].trim() : limpio;
  return despuesDeComa.split(/\s+/)[0] || "Afiliado/a";
}

export function institutionalEmailAttachments(): Attachment[] {
  return [
    crearAttachmentInline({
      filename: "sidca.png",
      assetPath: path.join(ASSET_DIRECTORY, "sidca.png"),
      contentId: "sidca-logo",
    }),
    crearAttachmentInline({
      filename: "cea.png",
      assetPath: path.join(ASSET_DIRECTORY, "cea.png"),
      contentId: "cea-logo",
    }),
    crearAttachmentInline({
      filename: "internacional-educacion.png",
      assetPath: path.join(ASSET_DIRECTORY, "internacional-educacion.png"),
      contentId: "ie-logo",
    }),
    crearAttachmentInline({
      filename: "cgt.png",
      assetPath: path.join(ASSET_DIRECTORY, "cgt.png"),
      contentId: "cgt-logo",
    }),
    // Iconos de los canales oficiales. Pesan ~1-2 KB cada uno.
    ...REDES.filter((red) => red.url.trim()).map((red) => ({
      ...crearAttachmentInline({
        filename: red.archivo,
        assetPath: path.join(ASSET_DIRECTORY, red.archivo),
        contentId: red.contentId,
      }),
    })),
  ];
}

/**
 * Encabezado institucional: los cuatro logos en una sola línea.
 *
 * Los anchos NO son arbitrarios ni iguales. Cada archivo tiene una relación
 * de aspecto distinta y, además, distinto relleno transparente alrededor del
 * arte: el de SiDCA ocupa sólo ~60 % del alto de su PNG y el de Internacional
 * de la Educación ~67 %. Estos valores están calculados para que el CONTENIDO
 * VISIBLE de los cuatro quede a la misma altura óptica.
 *
 * Regla que no hay que romper: se fija únicamente el ancho, con height:auto.
 * Antes convivían width y max-height, y esa combinación es la que deformaba
 * los logos institucionales.
 */
const LOGOS_ENCABEZADO = [
  {
    contentId: "sidca-logo",
    alt: `${MARCA.nombre} - ${MARCA.razon}`,
    clase: "logo-sidca",
    ancho: 132, // remitente: se muestra algo mayor que las adherentes
  },
  {
    contentId: "cea-logo",
    alt: "Confederación de Educadores Argentinos",
    clase: "logo-cea",
    ancho: 80,
  },
  {
    contentId: "ie-logo",
    alt: "Internacional de la Educación",
    clase: "logo-ie",
    ancho: 60,
  },
  {
    contentId: "cgt-logo",
    alt: "Confederación General del Trabajo",
    clase: "logo-cgt",
    ancho: 40,
  },
] as const;

function renderLogoRow(): string {
  const celdas = LOGOS_ENCABEZADO.map(
    (logo) =>
      `<td align="center" valign="middle" style="padding:0 9px;"><img src="cid:${logo.contentId}" alt="${escapeHtml(logo.alt)}" width="${logo.ancho}" class="${logo.clase}" style="display:block;width:${logo.ancho}px;height:auto;margin:0 auto;border:0;" /></td>`
  ).join("");

  // La tabla se centra y las celdas se ajustan al contenido: ninguna celda
  // con ancho porcentual fuerza a un logo a estirarse.
  return `<table role="presentation" align="center" cellpadding="0" cellspacing="0" border="0" style="margin:0 auto;"><tr>${celdas}</tr></table>`;
}

/**
 * Filas de datos separadas por línea fina.
 * La etiqueta puede partirse en dos líneas; el valor no, para que una fecha
 * como "21/09/2026 10:30 hs" no deje el "hs" huérfano alineado a la derecha.
 */
function renderDetails(details: InstitutionalEmailOptions["details"]): string {
  const visibles = (details || []).filter((detail) => String(detail.value ?? "").trim());

  const rows = visibles
    .map((detail, indice) => {
      const separador = indice === 0 ? "" : `border-top:1px solid ${COLOR.borde};`;
      return `<tr><td width="45%" style="${separador}padding:9px 0;font-family:${FUENTE};color:${COLOR.tintaSuave};font-size:13px;line-height:1.4;">${escapeHtml(detail.label)}</td><td align="right" style="${separador}padding:9px 0 9px 10px;font-family:${FUENTE};color:${COLOR.tinta};font-size:15px;line-height:1.4;font-weight:700;white-space:nowrap;">${escapeHtml(detail.value)}</td></tr>`;
    })
    .join("");

  return rows
    ? `<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">${rows}</table>`
    : "";
}

/** Bloque destacado reutilizable (datos, acceso a la APP, observación). */
function renderCallout(options: {
  titulo: string;
  tituloColor: string;
  cuerpo: string;
  fondo: string;
  barra: string;
}): string {
  return `<tr><td class="sidca-pad" style="padding:0 28px 20px;"><table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background:${options.fondo};border-left:4px solid ${options.barra};border-radius:0 6px 6px 0;"><tr><td style="padding:16px 18px;font-family:${FUENTE};color:${COLOR.tinta};font-size:15px;line-height:1.6;"><div style="margin:0 0 6px;font-size:12px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:${options.tituloColor};">${options.titulo}</div>${options.cuerpo}</td></tr></table></td></tr>`;
}

function renderAppAccess(appAccess?: InstitutionalEmailOptions["appAccess"]): string {
  if (!appAccess?.dni) return "";
  const dni = escapeHtml(appAccess.dni);

  return renderCallout({
    titulo: `Acceso a la APP ${MARCA.nombre}`,
    tituloColor: COLOR.appTexto,
    fondo: COLOR.appFondo,
    barra: COLOR.appBarra,
    cuerpo: `Ingresá a la APP ${MARCA.nombre} con tu DNI sin puntos: <strong>${dni}</strong><br />La APP funciona con acceso por DNI; no hace falta registrarse ni crear una contraseña.`,
  });
}

function renderObservation(observation?: string | null): string {
  const value = String(observation ?? "").trim();
  if (!value) return "";

  return renderCallout({
    titulo: "Observación",
    tituloColor: "#812D38",
    fondo: "#FFF7F7",
    barra: "#A13B47",
    cuerpo: escapeHtml(value),
  });
}

/**
 * Grupos de servicios.
 *
 * Cada prestación es una fila de tabla de dos celdas (icono + texto), no una
 * lista con viñetas: así el sangrado se ve igual en Outlook, que maneja <ul>
 * de forma inconsistente.
 *
 * Los iconos son decorativos y van marcados con aria-hidden: el significado
 * lo lleva siempre el nombre en texto, nunca el símbolo.
 */
function renderSections(sections?: InstitutionalEmailSection[]): string {
  if (!sections?.length) return "";

  return sections
    .map((section) => {
      const items = section.items
        .map((item) => {
          const icono = item.icono
            ? `<td width="38" valign="top" style="padding:0 10px 16px 0;font-size:20px;line-height:1.3;" aria-hidden="true">${escapeHtml(item.icono)}</td>`
            : "";

          return `<tr>${icono}<td valign="top" style="padding:0 0 16px;font-family:${FUENTE};"><div style="color:${COLOR.tinta};font-size:15px;font-weight:700;line-height:1.4;">${escapeHtml(item.nombre)}</div><div style="margin-top:3px;color:${COLOR.tintaSuave};font-size:14px;line-height:1.55;">${escapeHtml(item.detalle)}</div></td></tr>`;
        })
        .join("");

      // Título de grupo resaltado: banda ámbar clara con barra lateral, en
      // lugar de la línea fina anterior que se perdía entre las prestaciones.
      const titulo = `<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background:#FFF8E6;border-left:4px solid ${COLOR.acento};border-radius:0 6px 6px 0;margin-bottom:16px;"><tr><td style="padding:11px 16px;font-family:${FUENTE};color:${COLOR.tinta};font-size:14px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;">${escapeHtml(section.titulo)}</td></tr></table>`;

      return `<tr><td class="sidca-pad" style="padding:0 28px 10px;font-family:${FUENTE};">${titulo}<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">${items}</table></td></tr>`;
    })
    .join("");
}

/**
 * Franja de prestaciones de la APP.
 *
 * Va sobre el verde azulado institucional con el título en ámbar: es el
 * único bloque oscuro dentro del cuerpo, así que corta la lectura y queda
 * como el remate destacado de la lista de servicios.
 */
function renderHighlights(highlights?: InstitutionalEmailOptions["highlights"]): string {
  if (!highlights?.items?.length) return "";

  const texto = highlights.items
    .map((item) => escapeHtml(item))
    .join(`<span style="color:${COLOR.acento};"> &middot; </span>`);

  return `<tr><td class="sidca-pad" style="padding:10px 28px 24px;"><table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background:${COLOR.tinta};border-radius:12px;"><tr><td align="center" style="padding:22px 20px;font-family:${FUENTE};"><div style="margin:0 0 10px;color:${COLOR.acento};font-size:14px;font-weight:700;letter-spacing:1.4px;text-transform:uppercase;">${escapeHtml(highlights.titulo)}</div><div style="color:#FFFFFF;font-size:14px;line-height:1.8;">${texto}</div></td></tr></table></td></tr>`;
}

/** Cierre institucional sobre fondo claro, antes del pie oscuro. */
function renderClosing(closing?: InstitutionalEmailOptions["closing"]): string {
  if (!closing?.texto) return "";

  const titulo = closing.titulo
    ? `<div style="margin:0 0 6px;color:${COLOR.tinta};font-size:16px;font-weight:700;line-height:1.4;">${escapeHtml(closing.titulo)}</div>`
    : "";

  return `<tr><td class="sidca-pad" style="padding:4px 28px 24px;font-family:${FUENTE};text-align:center;">${titulo}<div style="color:${COLOR.tintaSuave};font-size:14px;line-height:1.65;">${escapeHtml(closing.texto)}</div></td></tr>`;
}

/**
 * Canales oficiales configurados: icono sobre la etiqueta.
 *
 * Se arma con una tabla centrada en vez de inline-block porque Outlook no
 * respeta el centrado de elementos inline dentro del pie. El texto acompaña
 * siempre al icono: quien tenga las imágenes bloqueadas igual lee el canal.
 */
function renderRedes(): string {
  const activos = REDES.filter((red) => red.url.trim());
  if (!activos.length) return "";

  const celdas = activos
    .map(
      (red) =>
        `<td align="center" valign="top" style="padding:0 10px;"><a href="${red.url}" style="text-decoration:none;color:${COLOR.acento};"><img src="cid:${red.contentId}" alt="" width="30" style="display:block;width:30px;height:auto;margin:0 auto 5px;border:0;" /><span style="font-family:${FUENTE};font-size:11px;font-weight:700;color:${COLOR.acento};">${escapeHtml(red.label)}</span></a></td>`
    )
    .join("");

  return `<div style="margin-top:16px;color:#9FB4BC;font-family:${FUENTE};font-size:11px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;">Seguinos y conocé más</div><table role="presentation" align="center" cellpadding="0" cellspacing="0" border="0" style="margin:10px auto 0;"><tr>${celdas}</tr></table>`;
}

function buildPlainText(options: InstitutionalEmailOptions): string {
  const state = STATE_PRESENTATION[options.state];
  const details = (options.details || [])
    .filter((detail) => String(detail.value ?? "").trim())
    .map((detail) => `${detail.label}: ${String(detail.value)}`);

  const lines = [
    `${MARCA.nombre} - ${MARCA.razon}`,
    MARCA.area,
    "",
    options.stateLabel || state.label,
    options.title,
    "",
    `Hola, ${primerNombre(options.greetingName)}:`,
    ...options.paragraphs,
  ];

  // La observación va antes que los datos administrativos: en los correos de
  // rechazo es el motivo de la resolución, o sea lo que la persona necesita
  // leer primero. Mismo orden que en el HTML.
  if (String(options.observation ?? "").trim()) {
    lines.push("", "OBSERVACIÓN", String(options.observation).trim());
  }

  lines.push(...details);

  if (options.appAccess?.dni) {
    lines.push(
      "",
      `ACCESO A LA APP ${MARCA.nombre.toUpperCase()}`,
      `Usuario de acceso: DNI sin puntos (${options.appAccess.dni})`,
      `La APP funciona con acceso por DNI; no hace falta registrarse ni crear una contraseña.`
    );
  }

  for (const section of options.sections || []) {
    lines.push("", section.titulo.toUpperCase());
    for (const item of section.items) lines.push(`- ${item.nombre}: ${item.detalle}`);
  }

  if (options.highlights?.items?.length) {
    lines.push("", options.highlights.titulo.toUpperCase(), options.highlights.items.join(" · "));
  }

  if (options.closing?.texto) {
    lines.push("", ...(options.closing.titulo ? [options.closing.titulo] : []), options.closing.texto);
  }

  const redes = REDES.filter((red) => red.url.trim());
  if (redes.length) {
    lines.push("", "Seguinos y conocé más", redes.map((red) => `${red.label}: ${red.url}`).join("\n"));
  }

  lines.push(
    "",
    MARCA.pieNombre,
    `${MARCA.inscripcion}.`,
    MARCA.sede,
    "",
    "Comunicación automática del sistema de afiliaciones.",
    "No es necesario responder este correo."
  );

  return lines.join("\n");
}

export function buildInstitutionalEmail(options: InstitutionalEmailOptions): InstitutionalEmail {
  const state = STATE_PRESENTATION[options.state];
  const stateLabel = escapeHtml(options.stateLabel || state.label);
  const safeTitle = escapeHtml(options.title);
  const safePreheader = escapeHtml(options.preheader);

  const paragraphs = options.paragraphs
    .map((paragraph) => `<p style="margin:0 0 14px;">${escapeHtml(paragraph)}</p>`)
    .join("");

  const details = renderDetails(options.details);
  const detailsBlock = details
    ? `<tr><td class="sidca-pad" style="padding:0 28px 20px;"><table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background:${COLOR.fondoSuave};border:1px solid ${COLOR.borde};border-radius:8px;"><tr><td style="padding:16px 18px;font-family:${FUENTE};"><div style="margin:0 0 4px;font-size:12px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:${COLOR.tintaSuave};">Datos de la solicitud</div>${details}</td></tr></table></td></tr>`
    : "";

  const html = `<!doctype html>
<html lang="es" xmlns:v="urn:schemas-microsoft-com:vml" xmlns:o="urn:schemas-microsoft-com:office:office">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<meta name="x-apple-disable-message-reformatting" />
<!-- Se declara el esquema de color soportado para que Gmail y Apple Mail en
     modo oscuro no inviertan la paleta institucional por su cuenta. -->
<meta name="color-scheme" content="light" />
<meta name="supported-color-schemes" content="light" />
<title>${safeTitle}</title>
<!--[if mso]>
<xml><o:OfficeDocumentSettings><o:PixelsPerInch>96</o:PixelsPerInch></o:OfficeDocumentSettings></xml>
<style>body,table,td,p,div,span{font-family:Arial,Helvetica,sans-serif !important;}</style>
<![endif]-->
<style>
  /* Los clientes que soportan <style> afinan el respiro en pantallas chicas.
     Todo lo esencial ya está en estilos en línea, así que quien lo ignore
     igual ve el correo correctamente. */
  @media only screen and (max-width:600px) {
    .sidca-pad { padding-left:18px !important; padding-right:18px !important; }
    .sidca-pad-top { padding-top:20px !important; }
    .sidca-titulo { font-size:19px !important; }
    /* El encabezado achica su padding para que los cuatro logos conserven
       el mayor ancho posible dentro de la misma línea. */
    .sidca-header { padding-left:8px !important; padding-right:8px !important; }
    /* Reducción proporcional: cada logo baja ~25 % manteniendo su relación
       de aspecto, porque sólo se toca el ancho y el alto sigue en auto. */
    .logo-sidca { width:100px !important; }
    .logo-cea   { width:60px !important; }
    .logo-ie    { width:45px !important; }
    .logo-cgt   { width:30px !important; }
  }
</style>
</head>
<body style="margin:0;padding:0;width:100%;background:${COLOR.tinta};-webkit-text-size-adjust:100%;-ms-text-size-adjust:100%;">
<!-- Preheader: texto de vista previa. Los espacios finos evitan que Gmail
     complete la línea con el comienzo del cuerpo. -->
<div style="display:none;max-height:0;overflow:hidden;mso-hide:all;font-size:1px;line-height:1px;color:${COLOR.tinta};opacity:0;">${safePreheader}&#8203;&#847;&#847;&#847;&#847;&#847;&#847;&#847;&#847;&#847;&#847;&#847;&#847;&#847;&#847;&#847;&#847;&#847;&#847;&#847;&#847;&#847;&#847;&#847;&#847;&#847;&#847;&#847;&#847;&#847;&#847;</div>

<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background:${COLOR.tinta};">
<tr><td align="center" style="padding:24px 10px;">

<!--[if mso]><table role="presentation" width="600" align="center" cellpadding="0" cellspacing="0" border="0"><tr><td><![endif]-->
<!-- Outlook de escritorio no respeta max-width, por eso la tabla fantasma de
     arriba: sin ella el correo se estiraba a todo el ancho de la ventana. -->
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="max-width:600px;background:${COLOR.papel};border-radius:12px;overflow:hidden;">

<!-- Encabezado: SiDCA, CEA, Internacional de la Educación y CGT en línea -->
<tr><td class="sidca-header" style="padding:24px 20px 20px;background:${COLOR.papel};border-bottom:4px solid ${COLOR.acento};text-align:center;font-family:${FUENTE};">
${renderLogoRow()}
<div style="margin-top:18px;color:${COLOR.tintaSuave};font-size:12px;font-weight:700;letter-spacing:1.4px;text-transform:uppercase;">${escapeHtml(MARCA.area)}</div>
</td></tr>

<!-- Estado / bienvenida -->
<tr><td class="sidca-pad sidca-pad-top" style="padding:26px 28px 14px;font-family:${FUENTE};color:${COLOR.tinta};">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background:${state.background};border-left:4px solid ${state.border};border-radius:0 8px 8px 0;">
<tr><td style="padding:16px 18px;font-family:${FUENTE};">
<div style="color:${state.color};font-size:12px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;">${state.icon} ${stateLabel}</div>
<div class="sidca-titulo" style="margin-top:6px;color:${COLOR.tinta};font-size:21px;font-weight:700;line-height:1.3;">${safeTitle}</div>
</td></tr></table>
</td></tr>

<!-- Cuerpo -->
<tr><td class="sidca-pad" style="padding:8px 28px 18px;font-family:${FUENTE};color:${COLOR.tinta};font-size:16px;line-height:1.65;">
<p style="margin:0 0 14px;">Hola, <strong>${escapeHtml(primerNombre(options.greetingName))}</strong>:</p>${paragraphs}
</td></tr>

${renderObservation(options.observation)}${detailsBlock}${renderAppAccess(options.appAccess)}${renderSections(options.sections)}${renderHighlights(options.highlights)}${renderClosing(options.closing)}

<!-- Pie -->
<tr><td class="sidca-pad" style="padding:22px 28px;background:${COLOR.tinta};color:${COLOR.papel};font-family:${FUENTE};text-align:center;font-size:13px;line-height:1.65;">
<div style="font-weight:700;letter-spacing:.4px;">${escapeHtml(MARCA.pieNombre)}</div>
<div style="margin-top:4px;color:#BFD0D6;font-size:12px;">${escapeHtml(MARCA.inscripcion)}.</div>
<div style="margin-top:10px;color:#BFD0D6;font-size:12px;line-height:1.6;">Sede Central: ${escapeHtml(MARCA.sede)}</div>
${renderRedes()}
<div style="margin-top:14px;padding-top:12px;border-top:1px solid rgba(255,255,255,.16);color:#9FB4BC;font-size:11px;line-height:1.6;">Comunicación automática del sistema de afiliaciones.<br />No es necesario responder este correo.</div>
</td></tr>

</table>
<!--[if mso]></td></tr></table><![endif]-->

</td></tr></table>
</body></html>`;

  return { html, text: buildPlainText(options), attachments: institutionalEmailAttachments() };
}
