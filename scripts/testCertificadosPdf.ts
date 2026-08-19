import fs from "node:fs";
import path from "node:path";
import PDFDocument from "pdfkit";
import { renderCertificadoPdfPage } from "../src/certificados/certificadoPdfRenderer.js";

const output = path.resolve(process.cwd(), "tmp", "certificados-prueba.pdf");
fs.mkdirSync(path.dirname(output), { recursive: true });

const emisionSidca = {
  cursoId: "prueba-sidca", usuarioDocId: "usuario-prueba-1", estado: "vigente",
  participante: { apellidoNombre: "PRUEBA, USUARIO SIDCA", dni: "12345678" },
  certificado: { institucionCertificado: "sidca", titulo: "CURSO DE PRUEBA SIDCA", modalidad: "Virtual", dias: "01/08/2026 al 15/08/2026", cargaHoraria: "40 Hs cátedra", resolucion: "RES-2026-10-E-CAT", fecha: "19 de agosto de 2026", autoridades: [{ nombre: "Prof. Sergio Guillamondegui", cargo: "Secretario General", organismo: "Sindicato de Docentes de Catamarca", referencia: "Inscripción Gremial N° 2902", orden: 1 }] },
  urlValidacion: "https://sidcagremio.com/validar-certificado/prueba-sidca/0123456789abcdef0123456789abcdef0123456789abcdef",
};
const emisionItm = {
  cursoId: "prueba-itm", usuarioDocId: "usuario-prueba-2", estado: "vigente",
  participante: { apellidoNombre: "PRUEBA, USUARIO ITM", dni: "87654321" },
  certificado: { institucionCertificado: "itm", titulo: "CURSO DE PRUEBA ITM", modalidad: "Presencial", dias: "05/08/2026 al 20/08/2026", cargaHoraria: "30 Hs cátedra", resolucion: "RES-2026-20-E-CAT", fecha: "19 de agosto de 2026", autoridades: [{ nombre: "Prof. Sergio Adrián Nicolas Endrizzi", cargo: "Rector", organismo: "Instituto Tecnológico Municipal", referencia: "", orden: 1 }, { nombre: "Carlos Ortiz", cargo: "Secretario", organismo: "Secretaría de Innovación y Calidad Educativa", referencia: "Ministerio de Educación y Trabajo", orden: 2 }] },
  urlValidacion: "https://sidcagremio.com/validar-certificado/prueba-itm/abcdef0123456789abcdef0123456789abcdef0123456789",
};

const run = async () => {
  const stream = fs.createWriteStream(output);
  const pdf = new PDFDocument({ autoFirstPage: false, size: "A4", layout: "landscape", margin: 0 });
  pdf.pipe(stream);
  await renderCertificadoPdfPage(pdf, emisionSidca);
  await renderCertificadoPdfPage(pdf, emisionItm);
  pdf.end();
  await new Promise<void>((resolve, reject) => { stream.once("finish", resolve); stream.once("error", reject); });
  console.log(`PDF generado:\n${output}\nPáginas: 2\nTamaño: ${fs.statSync(output).size} bytes`);
};

run().catch((error) => { console.error(error); process.exitCode = 1; });
