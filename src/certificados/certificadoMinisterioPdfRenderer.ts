import { Storage } from "@google-cloud/storage";
import PDFDocument from "pdfkit";
import QRCode from "qrcode";
import fs from "node:fs/promises";
import path from "node:path";
import sharp from "sharp";
import { nombreInstitucionOrganizadora } from "./institucionOrganizadora.js";
import {
  textoAuspicioConfiguracion,
  textoAuspicioConResolucion,
} from "./ministerioAuspicio.js";

const PAGE = { width: 842, height: 595 };
const QR = { x: 683, y: 431, width: 90, height: 90 };
const TITULO_BOX = { left: 0.17, top: 0.355, width: 0.66, height: 0.052 };
const COLOR_MINISTERIO = "#243b6b";
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

export type TextoAjustado = {
  lines: string[];
  fontSize: number;
  lineHeight: number;
  height: number;
};

// Tipografía oficial del bloque inferior. Se mantiene en unidades base del
// layout A4 horizontal para que el PDF y el preview compartan la misma
// jerarquía visual sin comprimir horizontalmente los textos.
export const FIRMANTES_MINISTERIO_TIPOGRAFIA = {
  nombre: {
    inicial: 8.2,
    minimo: 7.8,
    maxLineas: 1,
    interlineado: 1.15,
  },
  detalle: {
    inicial: 6.2,
    minimo: 5.8,
    maxLineas: 1,
    interlineado: 1.18,
  },
  separacion: 2.2,
  margenHorizontal: 2.5,
  escalaHorizontal: 1,
} as const;

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

export const ajustarTexto = (
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

export const fechaExpedicionMinisterio = (valor: unknown) => {
  const fecha = texto(valor);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(fecha)) return "";
  const [anio, mes, dia] = fecha.split("-").map(Number);
  const meses = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
  ];
  return `El presente certificado se expide en San Fernando Del Valle de Catamarca a ${dia} días del mes de ${meses[mes - 1]} de ${anio}.`;
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

export const formatearTituloMinisterio = (valor: unknown) => {
  const contenido = texto(valor)
    .replace(/^["“”]+/, "")
    .replace(/["“”]+$/, "")
    .trim();
  return contenido ? `"${contenido}"` : "";
};

export const construirTextoMinisterio = (certificado: any, participante: any = {}) => {
  const modalidad = texto(certificado?.modalidad) || "[MODALIDAD]";
  const inicio = fechaCorta(certificado?.fechaInicio) || "[FECHA INICIO]";
  const fin = fechaCorta(certificado?.fechaFin) || "[FECHA FIN]";
  const localidad = texto(certificado?.localidad);
  const departamento = texto(certificado?.departamento);
  const ubicacion = localidad
    ? `, en la Localidad de ${localidad}${departamento ? ` -DPTO ${departamento}` : ""}`
    : departamento
      ? `, -DPTO ${departamento}`
      : "";
  const evaluacion = texto(certificado?.textoEvaluacion);
  const carga = cargaHorariaPresentacion(certificado?.cargaHoraria);
  const nombre = texto(participante?.apellidoNombre || participante?.nombre);
  const dni = texto(participante?.dni);
  const auspicio = textoAuspicioConResolucion(
    textoAuspicioConfiguracion(certificado),
    certificado?.resolucion
  );
  const expedicion = fechaExpedicionMinisterio(certificado?.fecha);

  return {
    institucionOrganizadora: nombreInstitucionOrganizadora(certificado),
    modalidad: texto(certificado?.modalidad),
    textoEvaluacion: evaluacion,
    firmantes: Array.isArray(certificado?.firmantesMinisterio)
      ? certificado.firmantesMinisterio
      : [],
    titulo: formatearTituloMinisterio(certificado?.titulo),
    participante: `Que ${nombre || "[APELLIDO Y NOMBRE]"} D.N.I. N° ${dni || "[DNI]"}`,
    actividad: `Participó y aprobó el ${texto(certificado?.tipoActividad) || "[TIPO DE ACTIVIDAD]"} denominado`,
    dictado: `Dictado de manera ${modalidad}, desde ${inicio} al ${fin}${ubicacion}`,
    duracion: `con una duración de ${carga || "[CARGA HORARIA]"}${evaluacion ? `, ${evaluacion}` : "."}`,
    niveles: nivelesInstitucionales(certificado?.niveles),
    auspicio,
    textoAuspicioFinal: auspicio,
    expedicion,
    fechaExpedicion: expedicion,
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
    const margenTexto = s(doc, FIRMANTES_MINISTERIO_TIPOGRAFIA.margenHorizontal);
    const izquierdaTexto = izquierda + margenTexto;
    const anchoTexto = anchoColumna - margenTexto * 2;
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
      {
        valor: texto(firmante?.nombre),
        tipografia: FIRMANTES_MINISTERIO_TIPOGRAFIA.nombre,
      },
      {
        valor: texto(firmante?.cargo),
        tipografia: FIRMANTES_MINISTERIO_TIPOGRAFIA.detalle,
      },
      ...texto(firmante?.organismo)
        .replace(/\.\s+/g, ".\n")
        .split("\n")
        .filter(Boolean)
        .map((valor) => ({
          valor,
          tipografia: FIRMANTES_MINISTERIO_TIPOGRAFIA.detalle,
        })),
    ];

    for (const renglon of textosFirmante) {
      const ajustado = ajustarTexto(doc, renglon.valor, {
        ancho: anchoTexto,
        inicial: s(doc, renglon.tipografia.inicial),
        minimo: s(doc, renglon.tipografia.minimo),
        maxLineas: renglon.tipografia.maxLineas,
        interlineado: renglon.tipografia.interlineado,
        bold: false,
      });
      dibujarTexto(doc, ajustado, izquierdaTexto, cursor, anchoTexto, false);
      cursor += ajustado.height + s(doc, FIRMANTES_MINISTERIO_TIPOGRAFIA.separacion);
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

  const cabecera = (
    valor: string,
    izquierda: number,
    superior: number,
    ancho: number,
    cuerpo: number,
    bold = false,
    align: "center" | "left" = "center",
    color = "#000000",
    italic = false
  ) => {
    const fuente = italic
      ? bold
        ? "Helvetica-BoldOblique"
        : "Helvetica-Oblique"
      : bold
      ? "Helvetica-Bold"
      : "Helvetica";
    doc.font(fuente).fontSize(s(doc, cuerpo)).fillColor(color);
    doc.text(valor, x(doc, izquierda), y(doc, superior), {
      width: s(doc, ancho),
      align,
      lineGap: 0,
    });
  };

  cabecera("Ministerio de\nEducación y Trabajo", 385, 49, 190, 13, true, "left", COLOR_MINISTERIO);
  cabecera("Catamarca Gobierno", 385, 79, 190, 11, false, "left");
  cabecera("SECRETARIA DE INNOVACION Y CALIDAD EDUCATIVA", 274.3, 105, 293.1, 10.5, false);
  const nombreInstitucion = nombreInstitucionOrganizadora(certificado);
  const mostrarInstitucion = Boolean(nombreInstitucion);
  if (mostrarInstitucion) {
    cabecera(nombreInstitucion, 245, 122.5, 352, 12.5, true);
  }
  const coordenadas = mostrarInstitucion
    ? {
        certifica: 147.5,
        participante: 184,
        actividad: 207.9,
        titulo: 234.5,
        gapAfterTitle: 8,
        gapBetweenBlocks: 10,
      }
    : {
        certifica: 136,
        participante: 172.5,
        actividad: 196.4,
        titulo: 223,
        gapAfterTitle: 14,
        gapBetweenBlocks: 12,
      };
  cabecera("CERTIFICA", 351.7, coordenadas.certifica, 138.5, 26, true, "center", "#000000", true);

  const textos = construirTextoMinisterio(certificado, participante);
  const participanteTexto = textos.participante;
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
    y(doc, coordenadas.participante) - participanteAjustado.height / 2,
    s(doc, 640)
  );

  const actividad = ajustarTexto(doc, textos.actividad, {
    ancho: s(doc, 672),
    inicial: s(doc, 11.5),
    minimo: s(doc, 10.5),
    maxLineas: 1,
    interlineado: 1.2,
  });
  dibujarTexto(doc, actividad, x(doc, 85), y(doc, coordenadas.actividad) - actividad.height / 2, s(doc, 672));

  // Las comillas también ocupan ancho. Se incluyen antes de medir para que un
  // título de dos líneas nunca se desborde al agregarlas al momento de pintar.
  const titulo = ajustarTexto(doc, textos.titulo || '"[TÍTULO DE LA CAPACITACIÓN]"', {
    ancho: doc.page.width * TITULO_BOX.width,
    inicial: s(doc, 14.5),
    minimo: s(doc, 11.5),
    maxLineas: 2,
    interlineado: 1.1,
    bold: true,
  });
  // El título se centra dinámicamente entre la actividad y el primer bloque
  // del cuerpo. Su altura real depende de si entra en una o dos líneas.
  const limiteSuperiorDictado =
    y(doc, coordenadas.titulo) + titulo.height + y(doc, coordenadas.gapAfterTitle);
  const limiteInferiorTitulo = y(doc, coordenadas.actividad + 11.5);
  const tituloTop =
    limiteInferiorTitulo +
    (limiteSuperiorDictado - limiteInferiorTitulo - titulo.height) / 2;
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
    cursor += ajustado.height + y(doc, coordenadas.gapBetweenBlocks);
  }

  if (cursor - y(doc, coordenadas.gapBetweenBlocks) >= y(doc, 408)) {
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
