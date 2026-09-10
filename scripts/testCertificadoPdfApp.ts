import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import { generarCertificadoPdfIndividual } from "../src/certificados/certificadoPdfIndividual.js";

const emision = {
  certificadoId: "0123456789abcdef0123456789abcdef0123456789abcdef",
  token: "0123456789abcdef0123456789abcdef0123456789abcdef",
  cursoId: "prueba-sidca",
  usuarioDocId: "usuario-prueba-1",
  estado: "vigente",
  participante: { apellidoNombre: "PRUEBA, USUARIO SIDCA", dni: "12345678" },
  certificado: {
    institucionCertificado: "sidca",
    titulo: "CURSO DE PRUEBA SIDCA",
    modalidad: "Virtual",
    dias: "01/08/2026 al 15/08/2026",
    cargaHoraria: "40 Hs cátedra",
    resolucion: "RES-2026-10-E-CAT",
    fecha: "19 de agosto de 2026",
    autoridades: [{
      nombre: "Prof. Sergio Guillamondegui",
      cargo: "Secretario General",
      organismo: "Sindicato de Docentes de Catamarca",
      referencia: "Inscripción Gremial N° 2902",
      orden: 1,
    }],
  },
  urlValidacion:
    "https://sidcagremio.com/validar-certificado/prueba-sidca/0123456789abcdef0123456789abcdef0123456789abcdef",
};

// La prueba escribe sólo artefactos derivados en el workspace compartido; no
// toca los videos ni ningún dato de producción del backend.
const salida = path.resolve("D:/SiDCaapp2024/docs/tutorial/backend-test-output");
fs.mkdirSync(salida, { recursive: true });

function textoStreamsPdf(buffer: Buffer): string {
  const texto = buffer.toString("latin1");
  const partes = [texto];
  const expresion = /stream\r?\n([\s\S]*?)\r?\nendstream/g;
  let coincidencia: RegExpExecArray | null;

  while ((coincidencia = expresion.exec(texto))) {
    try {
      partes.push(zlib.inflateSync(Buffer.from(coincidencia[1], "latin1")).toString("latin1"));
    } catch {
      // No todos los streams PDF están comprimidos con FlateDecode.
    }
  }

  return partes.join("\n");
}

function exigir(condicion: unknown, mensaje: string): asserts condicion {
  if (!condicion) throw new Error(mensaje);
}

const run = async () => {
  const snapshotAntes = JSON.stringify(emision);
  const preview = await generarCertificadoPdfIndividual(emision, { marcaAgua: true });
  const snapshotDespuesPreview = JSON.stringify(emision);
  const oficial = await generarCertificadoPdfIndividual(emision, { marcaAgua: false });
  const snapshotDespuesOficial = JSON.stringify(emision);

  const previewPath = path.join(salida, "certificado-app-preview.pdf");
  const oficialPath = path.join(salida, "certificado-app-oficial.pdf");
  fs.writeFileSync(previewPath, preview);
  fs.writeFileSync(oficialPath, oficial);

  exigir(preview.length > 0 && oficial.length > 0, "No se generaron los PDFs.");
  exigir(snapshotAntes === snapshotDespuesPreview, "El preview mutó la emisión.");
  exigir(snapshotAntes === snapshotDespuesOficial, "El PDF oficial mutó la emisión.");
  exigir(emision.token === "0123456789abcdef0123456789abcdef0123456789abcdef", "Cambió el token.");
  exigir(emision.urlValidacion.includes(emision.token), "Cambió urlValidacion.");
  exigir(emision.cursoId === "prueba-sidca", "Cambió cursoId.");

  const textoPreview = textoStreamsPdf(preview);
  const textoOficial = textoStreamsPdf(oficial);
  // PDFKit codifica los glifos como cadenas hexadecimales dentro del stream.
  // Se verifica el prefijo inequívoco de "DOCUMENTO" y la secuencia final de
  // "VÁLIDO" sin depender de un extractor PDF externo.
  exigir(textoPreview.includes("444f43554d454e54") && textoPreview.includes("c14c49444f"), "El preview no contiene la marca de agua.");
  exigir(!textoOficial.includes("444f43554d454e54") && !textoOficial.includes("c14c49444f"), "El oficial contiene marca de agua.");

  const ocultar = (valor: string) => `${valor.slice(0, 4)}...${valor.slice(-4)}`;
  console.log(JSON.stringify({
    ok: true,
    cursoId: emision.cursoId,
    certificadoId: ocultar(emision.certificadoId),
    token: ocultar(emision.token),
    urlValidacion: `${emision.urlValidacion.slice(0, 42)}...${emision.urlValidacion.slice(-8)}`,
    previewBytes: preview.length,
    oficialBytes: oficial.length,
    previewPath,
    oficialPath,
  }, null, 2));
};

run().catch((error) => {
  console.error("testCertificadoPdfApp falló:", error?.message || error);
  process.exitCode = 1;
});
