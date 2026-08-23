import QRCode from "qrcode";
import sharp from "sharp";
import fs from "node:fs/promises";
import path from "node:path";

let sidcaBuffer: Buffer | null = null;
let itmBuffer: Buffer | null = null;

const readTemplate = async (name: string) => {
  const file = path.resolve(process.cwd(), "src/assets/certificados", name);
  const source = await fs.readFile(file);
  // Las plantillas backend están almacenadas en portrait; el certificado PDF
  // es landscape. Se transforma una sola vez y queda cacheada por proceso.
  return sharp(source).rotate(-90).png().toBuffer();
};

const templates = async () => {
  if (!sidcaBuffer) sidcaBuffer = await readTemplate("certificadocursosidca.png");
  if (!itmBuffer) itmBuffer = await readTemplate("certificadoITM.png");
  return { sidcaBuffer, itmBuffer };
};

const text = (value: unknown) => String(value ?? "").trim();
const money = (value: unknown) => text(value);
const formatDni = (value: unknown) => text(value).replace(/\B(?=(\d{3})+(?!\d))/g, ".");
const LAYOUT = {
  nombre: [32, 36, 36.5], dni: [78, 35.9, 15], titulo: [17, 44.7, 66],
  modalidad: [18, 53, 16.5], dias: [51.8, 53.2, 22], carga: [52.5, 56.2, 23.5],
  resolucion: [72.6, 65.1, 15], fecha: [69.5, 69.3, 21], qr: [77.2, 71.4, 14.3],
};
const pct = (page: number, value: number) => page * value / 100;

/**
 * Un cqw del frontend, en puntos PDF.
 *
 * El preview expresa TODOS sus cuerpos en cqw —1% del ancho del certificado—
 * y acá el certificado ocupa la página entera, así que la equivalencia es
 * directa. En A4 landscape (841.89 pt) un cqw son ~8.42 pt.
 *
 * Es la única fuente de verdad tipográfica: los números sueltos en puntos que
 * había antes dejaban el PDF visiblemente más chico que la pantalla.
 */
const cqw = (doc: PDFKit.PDFDocument, valor: number) => (doc.page.width * valor) / 100;

/**
 * Banda reservada para el TÍTULO del curso.
 *
 * Medida sobre las dos plantillas: la leyenda "Participó y aprobó el curso
 * denominado" termina en 42,57% del alto (SIDCA) / 41,80% (ITM) y la línea
 * "Modalidad…" empieza en 49,08% / 48,94%. La intersección que sirve para
 * ambas va de 42,6% a 48,9%; descontando un resguardo quedan 84 px de 1414,
 * centrados en 45,76%.
 *
 * El bloque crece simétrico desde ese centro, así que una, dos o tres líneas
 * quedan siempre equilibradas dentro de la banda y nunca tocan las leyendas.
 *
 * Estas constantes son las MISMAS que usa la vista previa del frontend: si se
 * cambian acá, hay que cambiarlas allá o el PDF dejará de reproducir lo que se
 * ve en pantalla.
 */
const TITULO = {
  cuerpoMax: 2.3, // cqw
  cuerpoMin: 1.05, // cqw: piso legible para títulos largos
  interlinea: 1.1,
  lineasMax: 3,
  altoMax: 8.2, // % de alto de página reservado dinámicamente
  centroY: 45.6, // % del alto de la página
};

/**
 * Cuerpo más grande que hace entrar el texto en el ancho de su caja.
 *
 * widthOfString() mide con la fuente y el cuerpo ACTIVOS del documento: no
 * acepta un cuerpo por opciones. Por eso cada intento fija primero
 * doc.fontSize(size) y recién después mide. Pasar { size } no sólo no
 * compila —TextOptions no lo declara—, además mediría siempre con el cuerpo
 * anterior.
 *
 * La fuente se activa una vez al entrar, y es la misma con la que el llamador
 * va a dibujar el campo: medir con otra daría un ancho que no corresponde.
 */
const fontSizeQueEntra = (doc: PDFKit.PDFDocument, value: string, width: number, objetivo: number, minimo: number, font = "Helvetica-Bold") => {
  const texto = String(value || "");
  if (!texto) return objetivo;

  doc.font(font);
  let size = objetivo;

  while (size > minimo) {
    doc.fontSize(size);
    if (doc.widthOfString(texto) <= width) return Number(size.toFixed(2));
    size -= 0.25;
  }

  return minimo;
};

export async function renderCertificadoPdfPage(
  doc: PDFKit.PDFDocument,
  emision: any
) {
  const { sidcaBuffer, itmBuffer } = await templates();
  const certificado = emision?.certificado || {};
  const participante = emision?.participante || {};
  const itm = certificado.institucionCertificado === "itm";
  const template = itm ? itmBuffer : sidcaBuffer;
  doc.addPage({ size: "A4", layout: "landscape", margin: 0 });
  doc.image(template, 0, 0, { width: doc.page.width, height: doc.page.height });

  const nombre = text(participante.apellidoNombre || participante.nombre);
  const dni = text(participante.dni);
  const autoridades = Array.isArray(certificado.autoridades) ? certificado.autoridades : [];
  const draw = (value: unknown, x: number, y: number, width: number, size: number, opts: any = {}) => {
    const content = text(value);
    if (!content) return;
    const font = opts.bold ? "Helvetica-Bold" : "Helvetica";
    const fitted = fontSizeQueEntra(doc, content, width, size, opts.minimo || size * 0.72, font);
    doc.fontSize(fitted).fillColor("#111827").font(font);
    doc.text(content, x, y, { width, align: "center", lineGap: 0, ...opts });
  };

  /**
   * Dato de UNA línea que conserva la altura de letra.
   *
   * Es la estrategia del preview y la razón por la que el PDF se veía chico:
   * cuando un texto no entraba, el backend bajaba el cuerpo —o sea, achicaba
   * la letra— mientras el frontend mantiene la altura y sólo comprime el
   * ancho. Un DNI y un rango de fechas tienen que leerse igual de grandes
   * aunque el segundo tenga el doble de caracteres.
   *
   * `compresionMinima` es hasta dónde se puede achatar antes de que la letra
   * empiece a deformarse de más. Recién ahí, y sólo ahí, se baja el cuerpo lo
   * justo para que entre con esa compresión. Nunca se recorta ni se abrevia.
   */
  const dibujarUnaLinea = (
    value: unknown,
    x: number,
    y: number,
    width: number,
    size: number,
    opts: { compresionMinima?: number; bold?: boolean } = {}
  ) => {
    const content = text(value);
    if (!content) return;

    const font = opts.bold === false ? "Helvetica" : "Helvetica-Bold";
    const compresionMinima = opts.compresionMinima ?? 0.55;

    doc.font(font).fontSize(size);
    const anchoReal = doc.widthOfString(content);

    let escalaX = anchoReal > width ? width / anchoReal : 1;
    let cuerpo = size;

    if (escalaX < compresionMinima) {
      cuerpo = size * (escalaX / compresionMinima);
      escalaX = compresionMinima;
      doc.fontSize(cuerpo);
    }

    doc.fillColor("#111827");

    // Se traslada al borde izquierdo de la caja y se comprime desde ahí: el
    // ancho local pasa a ser width/escalaX, así el centrado sigue cayendo en
    // el centro real de la caja y la geometría no se mueve.
    doc.save();
    doc.translate(x, 0);
    doc.scale(escalaX, 1);
    doc.text(content, 0, y, {
      width: width / escalaX,
      align: "center",
      lineGap: 0,
      lineBreak: false,
    });
    doc.restore();
  };
  // Cuerpos tomados del preview y convertidos a puntos. Nombre, DNI, días y
  // resolución comparten 2.2cqw: en el certificado son datos del mismo rango
  // y tienen que leerse con la misma altura de letra, sin importar que uno
  // tenga siete caracteres y otro veinticuatro.
  const FS_DATO = cqw(doc, 2.2);
  const FS_SECUNDARIO = cqw(doc, 2.1);
  const FS_MODALIDAD = cqw(doc, 2.35);
  const FS_CARGA = cqw(doc, 2.35);
  // Los días llevan cuerpo propio, más chico que el resto de los datos: un
  // rango completo son veinticuatro caracteres y a 2.2cqw ocupaba el 100% de
  // su caja, quedando pegado a la leyenda impresa "…durante los días,". Con
  // 1.65cqw el texto entra sin comprimir y le queda aire a los dos lados.
  const FS_DIAS = cqw(doc, 1.65);

  const [nx, ny, nw] = LAYOUT.nombre; dibujarUnaLinea(nombre, pct(doc.page.width, nx), pct(doc.page.height, ny) - 8, pct(doc.page.width, nw), FS_DATO, { bold: true, compresionMinima: 0.55 });
  const [dx, dy, dw] = LAYOUT.dni; dibujarUnaLinea(formatDni(dni), pct(doc.page.width, dx), pct(doc.page.height, dy) - 7, pct(doc.page.width, dw), FS_DATO, { bold: true, compresionMinima: 0.7 });
  // El título es el único campo que puede envolver, así que no se comprime
  // horizontalmente —a lo ancho de su caja del 66% quedaría deformado— sino
  // que se busca el cuerpo más grande que entre en la banda disponible.
  //
  // La medición es real: heightOfString devuelve el alto que PDFKit va a
  // ocupar de verdad con ese ancho y ese interlineado, contando el
  // envolvimiento por palabras. Contar caracteres no serviría: una "M" y una
  // "i" no miden lo mismo.
  const [tx, , tw] = LAYOUT.titulo;
  const tituloTexto = text(certificado.titulo);

  if (tituloTexto) {
    const anchoTitulo = pct(doc.page.width, tw);
    const altoMaximo = pct(doc.page.height, TITULO.altoMax);
    const cuerpoMinimo = cqw(doc, TITULO.cuerpoMin);

    doc.font("Helvetica-Bold");

    // Interlineado compacto. PDFKit no expone line-height: se compensa con
    // lineGap. El alto natural del renglón se MIDE —midiendo una sola letra
    // con lineGap 0— en vez de deducirlo de currentLineHeight(), que devuelve
    // un valor distinto del que heightOfString termina aplicando: por eso el
    // renglón salía a 1,365 × cuerpo y dos líneas no entraban nunca.
    const medir = (cuerpo: number) => {
      doc.fontSize(cuerpo);
      const alturaNatural = doc.heightOfString("M", {
        width: anchoTitulo,
        lineGap: 0,
      });
      const lineGap = cuerpo * TITULO.interlinea - alturaNatural;
      const alto = doc.heightOfString(tituloTexto, {
        width: anchoTitulo,
        align: "center",
        lineGap,
      });
      return {
        lineGap,
        alto,
        lineas: Math.max(1, Math.round(alto / (cuerpo * TITULO.interlinea))),
      };
    };

    let cuerpo = cqw(doc, TITULO.cuerpoMax);
    let medida = medir(cuerpo);

    // Se baja de a poco hasta que entra en alto Y no pasa de tres líneas. El
    // piso evita que un título disparatado quede ilegible: llegado ahí se
    // dibuja igual, nunca se recorta ni se abrevia.
    while (
      cuerpo > cuerpoMinimo &&
      (medida.alto > altoMaximo || medida.lineas > TITULO.lineasMax)
    ) {
      cuerpo = Math.max(cuerpoMinimo, cuerpo - 0.25);
      medida = medir(cuerpo);
    }

    // Centrado vertical del BLOQUE completo dentro de la banda: con una, dos
    // o tres líneas el conjunto queda siempre equilibrado.
    doc.fontSize(cuerpo).fillColor("#111827");
    doc.text(
      tituloTexto,
      pct(doc.page.width, tx),
      pct(doc.page.height, TITULO.centroY) - medida.alto / 2,
      { width: anchoTitulo, align: "center", lineGap: medida.lineGap }
    );

    const [mx, my, mw] = LAYOUT.modalidad;
    dibujarUnaLinea(
      text(certificado.modalidad).toUpperCase(),
      pct(doc.page.width, mx),
      pct(doc.page.height, my) - 6,
      pct(doc.page.width, mw),
      FS_MODALIDAD,
      { bold: true, compresionMinima: 0.8 }
    );
    const [daysX, daysY, daysW] = LAYOUT.dias;
    dibujarUnaLinea(
      text(certificado.dias).toUpperCase(),
      pct(doc.page.width, daysX),
      pct(doc.page.height, daysY) - 6,
      pct(doc.page.width, daysW),
      FS_DIAS,
      { bold: true, compresionMinima: 0.85 }
    );
    const [cx, cy, cw] = LAYOUT.carga;
    dibujarUnaLinea(
      text(certificado.cargaHoraria).toUpperCase(),
      pct(doc.page.width, cx),
      pct(doc.page.height, cy) - 6,
      pct(doc.page.width, cw),
      FS_CARGA,
      { bold: true, compresionMinima: 0.8 }
    );
  } else {
    const [mx, my, mw] = LAYOUT.modalidad;
    dibujarUnaLinea(text(certificado.modalidad).toUpperCase(), pct(doc.page.width, mx), pct(doc.page.height, my) - 6, pct(doc.page.width, mw), FS_MODALIDAD, { bold: true, compresionMinima: 0.8 });
    const [daysX, daysY, daysW] = LAYOUT.dias;
    dibujarUnaLinea(text(certificado.dias).toUpperCase(), pct(doc.page.width, daysX), pct(doc.page.height, daysY) - 6, pct(doc.page.width, daysW), FS_DIAS, { bold: true, compresionMinima: 0.85 });
    const [cx, cy, cw] = LAYOUT.carga;
    dibujarUnaLinea(text(certificado.cargaHoraria).toUpperCase(), pct(doc.page.width, cx), pct(doc.page.height, cy) - 6, pct(doc.page.width, cw), FS_CARGA, { bold: true, compresionMinima: 0.8 });
  }
  // La resolución NO comparte cuerpo con el DNI: su caja es del 15% y a
  // 2.2cqw el texto se partía e invadía la línea de fecha. Acá el orden se
  // invierte respecto de los demás datos —primero se busca el cuerpo que
  // entra, entre 12.5 y 10.5 pt, y sólo si ni el mínimo alcanza se comprime,
  // y muy poco (8% como mucho)—. Siempre una línea: lineBreak va en false
  // dentro de dibujarUnaLinea.
  const [rx, ry, rw] = LAYOUT.resolucion;
  const anchoResolucion = pct(doc.page.width, rw);
  const fsResolucion = fontSizeQueEntra(doc, text(certificado.resolucion), anchoResolucion, 12.5, 10.5, "Helvetica-Bold");
  dibujarUnaLinea(certificado.resolucion, pct(doc.page.width, rx), pct(doc.page.height, ry) - 5, anchoResolucion, fsResolucion, { bold: true, compresionMinima: 0.92 });
  const [fx, fy, fw] = LAYOUT.fecha; dibujarUnaLinea(text(certificado.fecha).toUpperCase(), pct(doc.page.width, fx), pct(doc.page.height, fy) - 6, pct(doc.page.width, fw), FS_SECUNDARIO, { bold: true, compresionMinima: 0.8 });
  // Autoridades: cuatro renglones con jerarquía propia. Cada línea se mide y
  // se ajusta por separado, así un organismo largo no achica el nombre.
  // El avance vertical sale de la altura REAL de cada renglón —heightOfString
  // ya contempla el envolvimiento—, de modo que las líneas nunca se pisan y
  // un campo vacío no deja hueco: simplemente no consume alto.
  const RENGLONES_AUTORIDAD = [
    { campo: "nombre", cuerpo: cqw(doc, 1.74) },
    { campo: "cargo", cuerpo: cqw(doc, 1.52) },
    { campo: "organismo", cuerpo: cqw(doc, 1.38) },
    { campo: "referencia", cuerpo: cqw(doc, 1.36) },
  ];

  // Interlineado del bloque de firma. Helvetica trae ~1.15 de fábrica, que
  // separa de más para cuatro renglones que se leen como una unidad; el
  // lineGap negativo lo cierra a 1.08 sin llegar a pisar los ascendentes.
  const INTERLINEA_AUTORIDAD = 1.08;

  // Hasta cuánto se achata una línea con tal de no partirla en dos. Es un
  // margen chico a propósito: sólo rescata los casos al filo, como una
  // referencia que se pasa por unos pocos puntos.
  const COMPRESION_AUTORIDAD = 0.92;

  autoridades.slice(0, 2).forEach((autoridad: any, index: number) => {
    const x = pct(doc.page.width, index === 0 ? 20 : 48);
    const w = pct(doc.page.width, 24);
    let cursor = pct(doc.page.height, 75);

    RENGLONES_AUTORIDAD.forEach((renglon) => {
      const contenido = text(autoridad[renglon.campo]);
      if (!contenido) return;

      doc.font("Helvetica-Bold").fillColor("#111827");

      let cuerpo = renglon.cuerpo;
      doc.fontSize(cuerpo);

      // Una línea, comprimida apenas, antes que dos: es más legible una
      // referencia levemente achatada que la misma partida al medio.
      const anchoNatural = doc.widthOfString(contenido);
      if (anchoNatural > w && anchoNatural <= w / COMPRESION_AUTORIDAD) {
        dibujarUnaLinea(contenido, x, cursor, w, cuerpo, { bold: true, compresionMinima: COMPRESION_AUTORIDAD });
        cursor += cuerpo * INTERLINEA_AUTORIDAD;
        return;
      }

      const gap = () => cuerpo * INTERLINEA_AUTORIDAD - doc.currentLineHeight(false);
      const alto = () => doc.heightOfString(contenido, { width: w, align: "center", lineGap: gap() });

      // Máximo dos renglones. Si hicieran falta tres se baja el cuerpo de
      // ESTA línea solamente, nunca el del bloque entero.
      const minimo = renglon.cuerpo * 0.75;
      while (cuerpo > minimo && alto() > cuerpo * INTERLINEA_AUTORIDAD * 2.4) {
        cuerpo -= 0.25;
        doc.fontSize(cuerpo);
      }

      const altoFinal = alto();
      doc.text(contenido, x, cursor, { width: w, align: "center", lineGap: gap() });
      cursor += altoFinal;
    });
  });
  if (emision.urlValidacion) {
    const qr = await QRCode.toBuffer(String(emision.urlValidacion), { margin: 2, width: 220 });
    doc.image(qr, pct(doc.page.width, 77.2), pct(doc.page.height, 71.4), { width: pct(doc.page.width, 14.3), height: pct(doc.page.width, 14.3) });
  }
}
