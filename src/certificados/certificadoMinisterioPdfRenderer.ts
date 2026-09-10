import { Storage } from "@google-cloud/storage";
import PDFDocument from "pdfkit";
import QRCode from "qrcode";
import fs from "node:fs/promises";
import path from "node:path";
import sharp from "sharp";

const PAGE = { width: 842, height: 595 };
const QR = { x: 683, y: 431, width: 90, height: 90 };
const TITULO_BOX = { left: 0.17, top: 0.355, width: 0.66, height: 0.052 };
const NIVELES_ORDENADOS = ["INICIAL", "PRIMARIO", "SECUNDARIO", "SUPERIOR"];

let marcoBuffer: Buffer | null = null;
let logoBuffer: Buffer | null = null;
let storage: Storage | null = null;

// La clave incluye bucket y ruta histórica del snapshot. Se guarda la promesa
// para que varias páginas no descarguen la misma firma en paralelo.
const firmasCache = new Map<string, Promise<Buffer>>();

export type RecursosPdfMinisterio = {
  cargarFirmaHistorica?: (firmante: any, emision: any) => Promise<Buffer>;
  incluirQr?: boolean;
};

const texto = (valor: unknown) => String(valor ?? "").replace(/\s+/g, " ").trim();
const x = (doc: PDFKit.PDFDocument, valor: number) => (doc.page.width * valor) / PAGE.width;
const y = (doc: PDFKit.PDFDocument, valor: number) => (doc.page.height * valor) / PAGE.height;
const s = (doc: PDFKit.PDFDocument, valor: number) => (doc.page.width * valor) / PAGE.width;

const referenciaCertificado = (emision: any) => {
  const valor = texto(emision?.certificadoId || emision?.token);
  if (!valor) return "sin identificador";
  if (valor.length <= 8) return valor;
  return `${valor.slice(0, 4)}...${valor.slice(-4)}`;
};

const leerAssets = async () => {
  if (!marcoBuffer) {
    marcoBuffer = await fs.readFile(
      path.resolve(process.cwd(), "src/assets/certificados/ministerio/marco_ministerio.png")
    );
  }
  if (!logoBuffer) {
    logoBuffer = await fs.readFile(
      path.resolve(process.cwd(), "src/assets/certificados/ministerio/logo_ministerio.png")
    );
  }
  return { marcoBuffer, logoBuffer };
};

const obtenerStorage = () => {
  if (!storage) storage = new Storage();
  return storage;
};

// Las firmas históricas pertenecen al bucket principal de Firebase. El bucket
// de PDFs es exclusivamente de salida y nunca debe usarse como fallback.
const bucketFirmas = () => texto(process.env.FIREBASE_STORAGE_BUCKET);

const cargarFirmaHistorica = async (firmante: any, emision: any): Promise<Buffer> => {
  const storagePath = texto(firmante?.imagenStoragePath);
  const firmanteId = texto(firmante?.id) || "sin-id";
  const referencia = referenciaCertificado(emision);

  if (!storagePath) {
    throw new Error(
      `Certificado ${referencia}: el firmante ${firmanteId} no tiene imagen histórica.`
    );
  }
  if (storagePath.includes("..") || storagePath.startsWith("gs://")) {
    throw new Error(
      `Certificado ${referencia}: la ruta histórica del firmante ${firmanteId} no es válida.`
    );
  }

  const bucket = bucketFirmas();
  if (!bucket) {
    throw new Error(
      `Certificado ${referencia}: falta configurar FIREBASE_STORAGE_BUCKET para la firma ${firmanteId}.`
    );
  }

  const claveCache = `${bucket}/${storagePath}`;
  const cacheada = firmasCache.get(claveCache);
  if (cacheada) {
    console.log(`[certificados-pdf] firma cache hit firmante=${firmanteId}`);
    return cacheada;
  }

  const descarga = (async () => {
    try {
      const [buffer] = await obtenerStorage().bucket(bucket).file(storagePath).download();
      const metadata = await sharp(buffer).metadata();
      if (!metadata.format || !metadata.width || !metadata.height) {
        throw new Error("La imagen no tiene dimensiones válidas.");
      }

      // PDFKit recibe un PNG normalizado y validado. Así una imagen corrupta
      // no termina convertida en una firma oficial incompleta.
      return sharp(buffer).png().toBuffer();
    } catch (error: any) {
      throw new Error(
        `Certificado ${referencia}: no se pudo cargar la firma histórica ${firmanteId} (${String(
          error?.message || "imagen inválida"
        ).slice(0, 120)}).`
      );
    }
  })();

  firmasCache.set(claveCache, descarga);
  try {
    return await descarga;
  } catch (error) {
    firmasCache.delete(claveCache);
    throw error;
  }
};

type TextoAjustado = {
  lines: string[];
  fontSize: number;
  lineHeight: number;
  height: number;
};

const partirTexto = (
  doc: PDFKit.PDFDocument,
  contenido: string,
  ancho: number
) => {
  const palabras = contenido.split(" ").filter(Boolean);
  const lineas: string[] = [];
  let linea = "";

  for (const palabra of palabras) {
    const candidata = linea ? `${linea} ${palabra}` : palabra;
    if (linea && doc.widthOfString(candidata) > ancho) {
      lineas.push(linea);
      linea = palabra;
    } else {
      linea = candidata;
    }
  }
  if (linea || !lineas.length) lineas.push(linea);
  return lineas;
};

const ajustarTexto = (
  doc: PDFKit.PDFDocument,
  contenido: string,
  opciones: {
    ancho: number;
    inicial: number;
    minimo: number;
    maxLineas: number;
    interlineado: number;
    bold?: boolean;
    cortePreferido?: string;
  }
): TextoAjustado => {
  const valor = texto(contenido);
  const fuente = opciones.bold ? "Helvetica-Bold" : "Helvetica";

  for (let cuerpo = opciones.inicial; cuerpo >= opciones.minimo; cuerpo -= 0.25) {
    doc.font(fuente).fontSize(cuerpo);
    const preferidas = opciones.cortePreferido
      ? (() => {
          const indice = valor.indexOf(opciones.cortePreferido!);
          return indice > 0
            ? [valor.slice(0, indice).trim(), valor.slice(indice).trim()]
            : null;
        })()
      : null;
    const lineas =
      preferidas && preferidas.every((linea) => doc.widthOfString(linea) <= opciones.ancho)
        ? preferidas
        : partirTexto(doc, valor, opciones.ancho);
    const lineHeight = cuerpo * opciones.interlineado;

    if (lineas.length <= opciones.maxLineas) {
      return {
        lines: lineas,
        fontSize: cuerpo,
        lineHeight,
        height: lineas.length * lineHeight,
      };
    }
  }

  throw new Error("El texto no entra en la caja oficial del certificado Ministerio.");
};

const dibujarTexto = (
  doc: PDFKit.PDFDocument,
  ajustado: TextoAjustado,
  izquierda: number,
  superior: number,
  ancho: number,
  bold = false
) => {
  doc.font(bold ? "Helvetica-Bold" : "Helvetica").fontSize(ajustado.fontSize).fillColor("#000000");
  ajustado.lines.forEach((linea, indice) => {
    doc.text(linea, izquierda, superior + indice * ajustado.lineHeight, {
      width: ancho,
      align: "center",
      lineBreak: false,
    });
  });
};

const fechaCorta = (valor: unknown) => {
  const fecha = texto(valor);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(fecha)) return "";
  const [anio, mes, dia] = fecha.split("-");
  return `${dia}/${mes}/${anio.slice(-2)}`;
};

const fechaExpedicion = (valor: unknown) => {
  const fecha = texto(valor);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(fecha)) return "";
  const [anio, mes, dia] = fecha.split("-").map(Number);
  const meses = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
  ];
  return `El presente certificado se expide en San Fernando del Valle de Catamarca a ${dia} días del mes de ${meses[mes - 1]} de ${anio}.`;
};

const nivelesInstitucionales = (niveles: unknown) => {
  const seleccionados = new Set(
    (Array.isArray(niveles) ? niveles : []).map((nivel) => texto(nivel).toUpperCase())
  );
  const ordenados = NIVELES_ORDENADOS.filter((nivel) => seleccionados.has(nivel));
  if (!ordenados.length) return "";
  if (ordenados.length === 1) return `Para el nivel ${ordenados[0]}.`;
  return `Para los niveles ${
    ordenados.length === 2
      ? ordenados.join(" y ")
      : `${ordenados.slice(0, -1).join(", ")} y ${ordenados.at(-1)}`
  }.`;
};

const cargaHorariaPresentacion = (valor: unknown) => {
  const carga = texto(valor);
  if (!carga) return "";
  return /(?:horas?|hs\.?)\s*c[aá]tedras?/i.test(carga)
    ? carga
    : `${carga} horas cátedra`;
};

const normalizarNumeroResolucion = (valor: unknown) =>
  texto(valor).replace(/^resoluci[oó]n\b[\s:.-]*/i, "");

const ETIQUETA_RESOLUCION_MINISTERIO = "S.I.E.(MECYT) N°";

const normalizarTextoAuspicio = (valor: unknown) =>
  texto(valor)
    .replace(
      /resoluci[oó]n\s+S\.I\.C\.E\.\(MET\)\s*N[°º]\s*$/i,
      `Resolución ${ETIQUETA_RESOLUCION_MINISTERIO}`
    )
    .replace(
      /resoluci[oó]n\s*$/i,
      `Resolución ${ETIQUETA_RESOLUCION_MINISTERIO}`
    );

const textoAuspicioConResolucion = (auspicio: unknown, resolucion: unknown) => {
  const base = normalizarTextoAuspicio(auspicio);
  const numero = normalizarNumeroResolucion(resolucion);
  const conPunto = (valor: string) =>
    valor && !/[.!?]$/.test(valor) ? `${valor}.` : valor;
  const resultado = !base
    ? numero
      ? `Resolución ${ETIQUETA_RESOLUCION_MINISTERIO} ${numero}.`
      : ""
    : !numero
      ? conPunto(base)
      : /N[°º]\s*$/i.test(base)
        ? `${base} ${numero}.`
        : `${conPunto(base)} Resolución ${ETIQUETA_RESOLUCION_MINISTERIO} ${numero}.`;
  return resultado
    .replace(/(N[°º])\.\s*/gi, "$1 ")
    .replace(/(N[°º]\.?\s*)resoluci[oó]n\b\s*/i, "$1");
};

const textoMinisterio = (certificado: any) => {
  const localidad = texto(certificado?.localidad);
  const departamento = texto(certificado?.departamento);
  const ubicacion = localidad
    ? `, en la Localidad de ${localidad}${departamento ? ` -DPTO ${departamento}` : ""}`
    : departamento
      ? `, -DPTO ${departamento}`
      : "";
  const evaluacion = texto(certificado?.textoEvaluacion);
  const carga = cargaHorariaPresentacion(certificado?.cargaHoraria);

  return {
    actividad: `Participó y aprobó el ${texto(certificado?.tipoActividad)} denominado`,
    dictado: `Dictado de manera ${texto(certificado?.modalidad)}, desde ${fechaCorta(
      certificado?.fechaInicio
    )} al ${fechaCorta(certificado?.fechaFin)}${ubicacion}.`,
    duracion: `Con una duración de ${carga}${evaluacion ? `, ${evaluacion}` : "."}`,
    niveles: nivelesInstitucionales(certificado?.niveles),
    auspicio: textoAuspicioConResolucion(certificado?.textoAuspicio, certificado?.resolucion),
    expedicion: fechaExpedicion(certificado?.fecha),
  };
};

const dibujarFirmantes = async (
  doc: PDFKit.PDFDocument,
  emision: any,
  recursos: RecursosPdfMinisterio
) => {
  const firmantes = (Array.isArray(emision?.certificado?.firmantesMinisterio)
    ? emision.certificado.firmantesMinisterio
    : []
  )
    .filter((firmante: any) => firmante?.activo !== false)
    .sort((a: any, b: any) => Number(a?.orden || 0) - Number(b?.orden || 0));

  if (!firmantes.length) {
    throw new Error(
      `Certificado ${referenciaCertificado(emision)}: no hay firmas históricas activas del modelo Ministerio.`
    );
  }

  const posiciones = [63, 228, 483];
  const anchosImagen = [124, 124, 99.1184];
  const anchosColumna = [170, 250, 150];

  for (let indice = 0; indice < firmantes.length; indice += 1) {
    const firmante = firmantes[indice];
    const columna = Math.min(indice, posiciones.length - 1);
    const anchoColumna = s(doc, anchosColumna[columna]);
    const anchoImagen = s(doc, anchosImagen[columna]);
    const izquierda = x(doc, posiciones[columna]);
    const firma = await (recursos.cargarFirmaHistorica
      ? recursos.cargarFirmaHistorica(firmante, emision)
      : cargarFirmaHistorica(firmante, emision));

    doc.image(firma, izquierda + (anchoColumna - anchoImagen) / 2, y(doc, 420), {
      fit: [anchoImagen, y(doc, 62)],
      align: "center",
      valign: "center",
    });

    let cursor = y(doc, 486);
    const textosFirmante = [
      { valor: texto(firmante?.nombre), size: s(doc, 6.5), bold: true },
      { valor: texto(firmante?.cargo), size: s(doc, 5.5), bold: false },
      ...texto(firmante?.organismo)
        .replace(/\.\s+/g, ".\n")
        .split("\n")
        .filter(Boolean)
        .map((valor) => ({ valor, size: s(doc, 5.5), bold: false })),
    ];

    for (const renglon of textosFirmante) {
      const ajustado = ajustarTexto(doc, renglon.valor, {
        ancho: anchoColumna,
        inicial: renglon.size,
        minimo: Math.max(s(doc, 4.2), renglon.size * 0.72),
        maxLineas: 1,
        interlineado: 1.12,
        bold: renglon.bold,
      });
      dibujarTexto(doc, ajustado, izquierda, cursor, anchoColumna, renglon.bold);
      cursor += ajustado.height + s(doc, 1.2);
    }
  }
};

const dibujarQr = async (doc: PDFKit.PDFDocument, emision: any) => {
  const cursoId = texto(emision?.cursoId);
  const token = texto(emision?.token || emision?.certificadoId);
  if (!cursoId || !token) {
    throw new Error(
      `Certificado ${referenciaCertificado(emision)}: faltan datos para generar el QR público.`
    );
  }
  const url = `https://sidcagremio.com/validar-certificado/${encodeURIComponent(
    cursoId
  )}/${encodeURIComponent(token)}`;
  const qr = await QRCode.toBuffer(url, { margin: 2, width: 360, errorCorrectionLevel: "M" });
  doc.image(qr, x(doc, QR.x), y(doc, QR.y), {
    width: s(doc, QR.width),
    height: s(doc, QR.height),
  });
};

export async function renderCertificadoMinisterioPdfPage(
  doc: PDFKit.PDFDocument,
  emision: any,
  recursos: RecursosPdfMinisterio = {}
) {
  const certificado = emision?.certificado || {};
  if (
    certificado.institucionCertificado !== "ministerio" ||
    certificado.layoutVersion !== "ministerio-v1"
  ) {
    throw new Error("La emisión no corresponde al layout Ministerio v1.");
  }

  const participante = emision?.participante || {};
  const nombre = texto(participante.apellidoNombre || participante.nombre);
  const dni = texto(participante.dni);
  if (!nombre || !dni) {
    throw new Error(
      `Certificado ${referenciaCertificado(emision)}: faltan datos históricos del participante.`
    );
  }

  const { marcoBuffer: marco, logoBuffer: logo } = await leerAssets();
  doc.addPage({ size: "A4", layout: "landscape", margin: 0 });
  doc.image(marco, 0, 0, { width: doc.page.width, height: doc.page.height });
  doc.image(logo, x(doc, 326.7), y(doc, 42), { width: s(doc, 49.4), height: y(doc, 58) });

  const cabecera = (valor: string, izquierda: number, superior: number, ancho: number, cuerpo: number, bold = false, align: "center" | "left" = "center") => {
    doc.font(bold ? "Helvetica-Bold" : "Helvetica").fontSize(s(doc, cuerpo)).fillColor("#000000");
    doc.text(valor, x(doc, izquierda), y(doc, superior), {
      width: s(doc, ancho),
      align,
      lineGap: 0,
    });
  };

  cabecera("Ministerio de\nEducación y Trabajo", 385, 49, 190, 13, true, "left");
  cabecera("Catamarca Gobierno", 385, 84, 190, 11, false, "left");
  cabecera("SECRETARIA DE INNOVACION Y CALIDAD EDUCATIVA", 274.3, 111.3, 293.1, 11, true);
  cabecera("CERTIFICA", 351.7, 136, 138.5, 26, true);

  const participanteTexto = `Que ${nombre} D.N.I. N° ${dni}`;
  const participanteAjustado = ajustarTexto(doc, participanteTexto, {
    ancho: s(doc, 640),
    inicial: s(doc, 13),
    minimo: s(doc, 11),
    maxLineas: 2,
    interlineado: 1.15,
  });
  dibujarTexto(
    doc,
    participanteAjustado,
    x(doc, 101),
    y(doc, 172.5) - participanteAjustado.height / 2,
    s(doc, 640)
  );

  const textos = textoMinisterio(certificado);
  const actividad = ajustarTexto(doc, textos.actividad, {
    ancho: s(doc, 672),
    inicial: s(doc, 11.5),
    minimo: s(doc, 10.5),
    maxLineas: 1,
    interlineado: 1.2,
  });
  dibujarTexto(doc, actividad, x(doc, 85), y(doc, 196.4) - actividad.height / 2, s(doc, 672));

  // Las comillas también ocupan ancho. Se incluyen antes de medir para que un
  // título de dos líneas nunca se desborde al agregarlas al momento de pintar.
  const titulo = ajustarTexto(doc, `“${texto(certificado.titulo)}”`, {
    ancho: doc.page.width * TITULO_BOX.width,
    inicial: s(doc, 14.5),
    minimo: s(doc, 11.5),
    maxLineas: 2,
    interlineado: 1.16,
    bold: true,
  });
  // El cuerpo conserva su flujo actual. La caja visual del título reproduce el
  // preview: al crecer a dos líneas se desplaza media línea hacia abajo, para
  // mantener el centro de la banda entre la actividad y el dictado.
  const limiteSuperiorDictado = y(doc, 223) + titulo.height + y(doc, 14);
  const desplazamientoTitulo = Math.max(0, (titulo.height - titulo.lineHeight) / 2);
  const tituloBoxTop = doc.page.height * TITULO_BOX.top;
  const tituloBoxHeight = doc.page.height * TITULO_BOX.height;
  const tituloTop =
    tituloBoxTop + (tituloBoxHeight - titulo.height) / 2 + desplazamientoTitulo;
  const tituloX = doc.page.width * TITULO_BOX.left;
  const tituloWidth = doc.page.width * TITULO_BOX.width;
  dibujarTexto(doc, titulo, tituloX, tituloTop, tituloWidth, true);

  const bloques = [
    { contenido: textos.dictado, maxLineas: 1 },
    { contenido: textos.duracion, maxLineas: 1 },
    { contenido: textos.niveles, maxLineas: 1 },
    {
      contenido: textos.auspicio,
      maxLineas: 2,
      cortePreferido: "el Ministerio de Educación y Trabajo",
    },
    { contenido: textos.expedicion, maxLineas: 1 },
  ].filter((bloque) => Boolean(texto(bloque.contenido)));

  let cursor = limiteSuperiorDictado;
  for (const bloque of bloques) {
    const ajustado = ajustarTexto(doc, bloque.contenido, {
      ancho: s(doc, 672),
      inicial: s(doc, 12),
      minimo: s(doc, 10.5),
      maxLineas: bloque.maxLineas,
      interlineado: 1.2,
      cortePreferido: "cortePreferido" in bloque ? bloque.cortePreferido : undefined,
    });
    dibujarTexto(doc, ajustado, x(doc, 85), cursor, s(doc, 672));
    cursor += ajustado.height + y(doc, 12);
  }

  if (cursor - y(doc, 12) >= y(doc, 408)) {
    throw new Error(
      `Certificado ${referenciaCertificado(emision)}: el texto Ministerio invade la zona de firmas.`
    );
  }

  await dibujarFirmantes(doc, emision, recursos);
  if (recursos.incluirQr === false) {
    cabecera("QR", QR.x, QR.y + 30, QR.width, 18, true);
  } else {
    await dibujarQr(doc, emision);
  }
}
