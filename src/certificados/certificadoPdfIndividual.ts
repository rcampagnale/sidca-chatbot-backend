import PDFDocument from "pdfkit";
import { PassThrough } from "node:stream";
import { renderCertificadoPdfPage } from "./certificadoPdfRenderer.js";

export type OpcionesCertificadoPdfIndividual = {
  marcaAgua?: boolean;
  incluirQr?: boolean;
};

/**
 * Genera una sola página a partir de una emisión ya guardada.
 *
 * Esta función es deliberadamente sólo de renderizado: no consulta ni escribe
 * Firestore, no genera tokens y no modifica el objeto de emisión recibido.
 */
export async function generarCertificadoPdfIndividual(
  emision: Record<string, any>,
  opciones: OpcionesCertificadoPdfIndividual = {}
): Promise<Buffer> {
  if (!emision || typeof emision !== "object") {
    throw new Error("La emisión del certificado no es válida.");
  }

  const salida = new PassThrough();
  const partes: Buffer[] = [];

  salida.on("data", (parte: Buffer | Uint8Array) => {
    partes.push(Buffer.isBuffer(parte) ? parte : Buffer.from(parte));
  });

  const finalizada = new Promise<void>((resolve, reject) => {
    salida.once("finish", resolve);
    salida.once("error", reject);
  });

  const documento = new PDFDocument({
    autoFirstPage: false,
    size: "A4",
    layout: "landscape",
    margin: 0,
  });

  documento.pipe(salida);
  await renderCertificadoPdfPage(documento, emision, {
    incluirQr: opciones.incluirQr !== false,
  });

  if (opciones.marcaAgua === true) {
    const ancho = documento.page.width;
    const alto = documento.page.height;

    documento.save();
    documento.opacity(0.18);
    documento.fillColor("#6b7280");
    documento.font("Helvetica-Bold").fontSize(Math.min(ancho * 0.075, 58));
    documento.rotate(-32, { origin: [ancho / 2, alto / 2] });
    documento.text("DOCUMENTO NO VÁLIDO", 0, alto / 2 - 30, {
      width: ancho,
      align: "center",
      lineBreak: false,
    });
    documento.restore();
  }

  documento.end();
  await finalizada;
  return Buffer.concat(partes);
}
