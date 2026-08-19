import fs from "node:fs";
import path from "node:path";
import PDFDocument from "pdfkit";
import { renderCertificadoPdfPage } from "../src/certificados/certificadoPdfRenderer.js";
import { listarEmisionesVigentesCurso } from "../src/certificados/emisionesCurso.js";

const cursoId = String(process.env.CERTIFICADOS_PDF_CURSO_ID || "").trim();
if (!cursoId) throw new Error("Definí CERTIFICADOS_PDF_CURSO_ID.");
const project = process.env.FIREBASE_PROJECT_ID || process.env.GCLOUD_PROJECT || "";
const firestore = `https://firestore.googleapis.com/v1/projects/${project}/databases/(default)/documents`;
const accessToken = async () => {
  const tokenLocal = String(process.env.GOOGLE_OAUTH_ACCESS_TOKEN || "").trim();
  if (tokenLocal) return tokenLocal;
  const response = await fetch("http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token", { headers: { "Metadata-Flavor": "Google" } });
  if (!response.ok) throw new Error(`No se pudo obtener access token desde Metadata Server: ${response.status}`);
  const data = await response.json();
  return data.access_token;
};
const fromValue = (v: any): any => v?.stringValue ?? v?.integerValue ?? v?.doubleValue ?? v?.booleanValue ?? v?.timestampValue ?? (v?.mapValue ? Object.fromEntries(Object.entries(v.mapValue.fields || {}).map(([k, x]) => [k, fromValue(x)])) : v?.arrayValue ? (v.arrayValue.values || []).map(fromValue) : null);
const fromDoc = (d: any) => Object.fromEntries(Object.entries(d.fields || {}).map(([k, v]) => [k, fromValue(v)]));
const run = async () => {
  const emisiones = await listarEmisionesVigentesCurso(cursoId, await accessToken());
  emisiones.sort((a: any, b: any) => String(a.participante?.apellidoNombre || "").localeCompare(String(b.participante?.apellidoNombre || ""), "es", { sensitivity: "base" }));
  const output = path.resolve(process.cwd(), "tmp", "certificados-curso-prueba.pdf"); fs.mkdirSync(path.dirname(output), { recursive: true }); const stream = fs.createWriteStream(output); const pdf = new PDFDocument({ autoFirstPage: false, size: "A4", layout: "landscape", margin: 0 }); pdf.pipe(stream);
  for (const emision of emisiones) await renderCertificadoPdfPage(pdf, emision); pdf.end(); await new Promise<void>((resolve, reject) => { stream.once("finish", resolve); stream.once("error", reject); });
  console.log(`Curso: ${cursoId}\nEmisiones vigentes: ${emisiones.length}\nPáginas generadas: ${emisiones.length}\nPDF: ${output}\nTamaño: ${fs.statSync(output).size} bytes`); emisiones.forEach((e: any, i: number) => console.log(`${i + 1}. ${e.participante?.apellidoNombre || "—"} · DNI ${e.participante?.dni || "—"}`));
};
run().catch((error) => { console.error(error); process.exitCode = 1; });
