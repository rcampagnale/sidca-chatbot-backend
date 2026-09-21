import fs from "node:fs";
import path from "node:path";
import {
  buildInstitutionalEmail,
  type InstitutionalEmail,
} from "../../src/email/institutionalEmail.js";
import { buildWelcomeEmailHtml } from "../../src/email/welcomeEmail.js";
import { buildReaffiliationPendingEmailHtml } from "../../src/email/reaffiliationPendingEmail.js";
import { buildReaffiliationApprovedEmailHtml } from "../../src/email/reaffiliationApprovedEmail.js";
import { buildReaffiliationRejectedEmailHtml } from "../../src/email/reaffiliationRejectedEmail.js";

const outputDirectory = path.resolve(process.cwd(), "scripts/email-previews/generated");
const assetDirectory = path.resolve(process.cwd(), "src/email/assets");

function inlineImages(html: string): string {
  const assets: Record<string, [string, string]> = {
    "sidca-logo": ["sidca.png", "image/png"],
    "cea-logo": ["cea.png", "image/png"],
    "ie-logo": ["internacional-educacion.png", "image/png"],
    "cgt-logo": ["cgt.png", "image/png"],
    "red-web": ["red-web.png", "image/png"],
    "red-facebook": ["red-facebook.png", "image/png"],
    "red-youtube": ["red-youtube.png", "image/png"],
    "red-instagram": ["red-instagram.png", "image/png"],
  };

  return Object.entries(assets).reduce((result, [contentId, [filename, mime]]) => {
    const content = fs.readFileSync(path.join(assetDirectory, filename)).toString("base64");
    return result.replaceAll(`cid:${contentId}`, `data:${mime};base64,${content}`);
  }, html);
}

function previewFromHtml(filename: string, html: string): void {
  fs.writeFileSync(path.join(outputDirectory, filename), inlineImages(html), "utf8");
}

const fictional = {
  nombre: "Pérez, Juan",
  dni: "12345678",
  fecha: "21/09/2026 10:30 hs",
};

const nuevaAfiliacionPendiente = buildInstitutionalEmail({
  subject: "SiDCA | Solicitud de afiliación recibida",
  preheader: "Recibimos correctamente tu solicitud de afiliación.",
  title: "Solicitud de afiliación recibida",
  state: "pending",
  greetingName: fictional.nombre,
  paragraphs: [
    "Recibimos correctamente tu solicitud de afiliación a SiDCA.",
    "La misma se encuentra en proceso de revisión. Te notificaremos por este mismo medio cuando se actualice su estado.",
  ],
  details: [{ label: "DNI", value: fictional.dni }, { label: "Fecha de solicitud", value: fictional.fecha }],
});

const nuevaAfiliacionRechazada = buildInstitutionalEmail({
  subject: "SiDCA | Actualización de tu solicitud de afiliación",
  preheader: "Tenemos una actualización sobre tu solicitud de afiliación.",
  title: "Solicitud de afiliación no aprobada",
  state: "rejected",
  greetingName: fictional.nombre,
  paragraphs: [
    "Luego de la revisión correspondiente, tu solicitud de afiliación no fue aprobada.",
    "Si necesitás realizar una consulta o aclarar tu situación, podés comunicarte con SiDCA por sus canales oficiales.",
  ],
  details: [{ label: "DNI", value: fictional.dni }, { label: "Fecha de resolución", value: fictional.fecha }],
  observation: "Observación ficticia para revisar el diseño del correo.",
});

const previews: Array<[string, string | InstitutionalEmail]> = [
  ["01-nueva-afiliacion-pendiente.html", nuevaAfiliacionPendiente],
  ["02-nueva-afiliacion-aprobada.html", buildWelcomeEmailHtml({ nombre: fictional.nombre, dni: fictional.dni, email: "juan.perez@example.test" })],
  ["03-nueva-afiliacion-rechazada.html", nuevaAfiliacionRechazada],
  ["04-reafiliacion-pendiente.html", buildReaffiliationPendingEmailHtml({ nombre: fictional.nombre, dni: fictional.dni, fechaSolicitud: fictional.fecha, nroAfiliacion: "2902-1234" })],
  ["05-reafiliacion-aprobada.html", buildReaffiliationApprovedEmailHtml({ nombre: fictional.nombre, dni: fictional.dni, fechaAprobacion: fictional.fecha, nroAfiliacion: "2902-1234" })],
  ["06-reafiliacion-rechazada.html", buildReaffiliationRejectedEmailHtml({ nombre: fictional.nombre, dni: fictional.dni, fechaResolucion: fictional.fecha, nroAfiliacion: "2902-1234", motivo: "Observación ficticia para revisar el diseño del correo." })],
];

fs.mkdirSync(outputDirectory, { recursive: true });
for (const [filename, value] of previews) {
  previewFromHtml(filename, typeof value === "string" ? value : value.html);
}

console.log(`Previews generados en ${outputDirectory}`);
