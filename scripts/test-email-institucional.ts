import "dotenv/config";
import { Resend } from "resend";
import {
  buildWelcomeEmail,
  WELCOME_EMAIL_SUBJECT,
} from "../src/email/welcomeEmail.js";
import {
  buildReaffiliationPendingEmail,
  REAFFILIATION_PENDING_EMAIL_SUBJECT,
} from "../src/email/reaffiliationPendingEmail.js";
import {
  buildReaffiliationApprovedEmail,
  REAFFILIATION_APPROVED_EMAIL_SUBJECT,
} from "../src/email/reaffiliationApprovedEmail.js";
import {
  buildReaffiliationRejectedEmail,
  REAFFILIATION_REJECTED_EMAIL_SUBJECT,
} from "../src/email/reaffiliationRejectedEmail.js";

type TipoPrueba =
  | "welcome"
  | "reaffiliation-pending"
  | "reaffiliation-approved"
  | "reaffiliation-rejected";

type PruebaConstruida = {
  tipo: TipoPrueba;
  asunto: string;
  html: string;
  text: string;
  attachments: ReturnType<typeof buildWelcomeEmail>["attachments"];
};

const TIPOS_VALIDOS: readonly TipoPrueba[] = [
  "welcome",
  "reaffiliation-pending",
  "reaffiliation-approved",
  "reaffiliation-rejected",
];

function argumento(nombre: string): string | undefined {
  const prefijo = `--${nombre}=`;
  const valor = process.argv.slice(2).find((item) => item.startsWith(prefijo));
  return valor?.slice(prefijo.length).trim() || undefined;
}

function fail(message: string): never {
  console.error(`Error: ${message}`);
  process.exit(1);
}

function validarDestinatario(to: string): void {
  if (to.includes(",") || to.includes(";") || /\s/.test(to)) {
    fail("--to debe contener una sola dirección de correo, sin listas.");
  }

  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(to)) {
    fail("--to no tiene un formato de correo válido.");
  }
}

function construirPrueba(tipo: TipoPrueba): PruebaConstruida {
  const datosReafiliacion = {
    dni: "12345678",
    nombre: "Juan Pérez",
    fechaSolicitud: "21/09/2026",
    fechaAprobacion: "21/09/2026",
    fechaResolucion: "21/09/2026",
    motivo: "Prueba controlada del nuevo diseño institucional.",
  };

  switch (tipo) {
    case "welcome": {
      const email = buildWelcomeEmail({ nombre: "María González", dni: "23456789" });
      return { tipo, asunto: WELCOME_EMAIL_SUBJECT, ...email };
    }
    case "reaffiliation-pending": {
      const email = buildReaffiliationPendingEmail({
        dni: datosReafiliacion.dni,
        nombre: datosReafiliacion.nombre,
        fechaSolicitud: datosReafiliacion.fechaSolicitud,
        nroAfiliacion: undefined,
      });
      return { tipo, asunto: REAFFILIATION_PENDING_EMAIL_SUBJECT, ...email };
    }
    case "reaffiliation-approved": {
      const email = buildReaffiliationApprovedEmail({
        dni: datosReafiliacion.dni,
        nombre: datosReafiliacion.nombre,
        fechaAprobacion: datosReafiliacion.fechaAprobacion,
        nroAfiliacion: undefined,
      });
      return { tipo, asunto: REAFFILIATION_APPROVED_EMAIL_SUBJECT, ...email };
    }
    case "reaffiliation-rejected": {
      const email = buildReaffiliationRejectedEmail({
        dni: datosReafiliacion.dni,
        nombre: datosReafiliacion.nombre,
        fechaResolucion: datosReafiliacion.fechaResolucion,
        motivo: datosReafiliacion.motivo,
        nroAfiliacion: undefined,
      });
      return { tipo, asunto: REAFFILIATION_REJECTED_EMAIL_SUBJECT, ...email };
    }
  }
}

async function main(): Promise<void> {
  const tipo = argumento("tipo") as TipoPrueba | undefined;
  const to = argumento("to");

  if (!tipo || !TIPOS_VALIDOS.includes(tipo)) {
    fail(`--tipo es obligatorio y debe ser uno de: ${TIPOS_VALIDOS.join(", ")}.`);
  }
  if (!to) fail("--to es obligatorio.");
  validarDestinatario(to);

  const apiKey = String(process.env.RESEND_API_KEY || "").trim();
  const from = String(process.env.SIDCA_EMAIL_FROM || "").trim();
  const replyTo = String(process.env.SIDCA_EMAIL_REPLY_TO || "").trim();
  if (!apiKey) fail("Falta configurar RESEND_API_KEY.");
  if (!from) fail("Falta configurar SIDCA_EMAIL_FROM.");

  const prueba = construirPrueba(tipo);
  const asunto = `[PRUEBA] ${prueba.asunto}`;
  const resend = new Resend(apiKey);
  const response = await resend.emails.send({
    from,
    to,
    subject: asunto,
    ...(replyTo ? { replyTo } : {}),
    html: prueba.html,
    text: prueba.text,
    attachments: prueba.attachments,
  });

  if (response.error || !response.data?.id) {
    console.log(`Tipo: ${tipo}`);
    console.log(`Destinatario: ${to}`);
    console.log(`Asunto: ${asunto}`);
    console.log("Resultado: ERROR");
    throw new Error(response.error?.message || "Resend no devolvió un identificador de envío.");
  }

  console.log(`Tipo: ${tipo}`);
  console.log(`Destinatario: ${to}`);
  console.log(`Asunto: ${asunto}`);
  console.log("Resultado: OK");
  console.log(`Resend ID: ${response.data.id}`);
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : "Error desconocido al enviar la prueba.");
  process.exitCode = 1;
});
