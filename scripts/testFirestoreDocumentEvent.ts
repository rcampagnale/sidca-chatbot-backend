import assert from "node:assert/strict";
import {
  codificarDocumentEventParaPrueba,
  decodificarDniDesdeCounterEvent,
  decodificarUsuarioIdDesdeDocumentEvent,
  extraerUsuarioIdDesdeNombreDocumento,
  FIRESTORE_CREATED_EVENT_TYPE,
  FIRESTORE_UPDATED_EVENT_TYPE,
  validarTipoCloudEvent,
  validarTipoCloudEventEsperado,
} from "../src/events/firestoreDocumentEvent.js";
import {
  decidirCorreoAprobada,
  decidirCorreoPendiente,
  decidirCorreoRechazada,
  emailValidoReafiliacion,
  esSolicitudAprobadaValida,
  esSolicitudPendienteValida,
  esSolicitudRechazadaValida,
} from "../src/email/reaffiliationPendingSupport.js";
import { buildReaffiliationPendingEmailHtml } from "../src/email/reaffiliationPendingEmail.js";
import { buildReaffiliationApprovedEmailHtml } from "../src/email/reaffiliationApprovedEmail.js";
import { buildReaffiliationRejectedEmailHtml } from "../src/email/reaffiliationRejectedEmail.js";
import { resolverNumeroAfiliacionHistorico } from "../src/reafiliacion/historicAffiliationNumber.js";

const validName = "projects/sidca-a33f0/databases/(default)/documents/usuarios/ABC123";

validarTipoCloudEvent(FIRESTORE_CREATED_EVENT_TYPE);
assert.equal(
  decodificarUsuarioIdDesdeDocumentEvent(codificarDocumentEventParaPrueba(validName)),
  "ABC123",
);

assert.throws(
  () => validarTipoCloudEvent("google.cloud.firestore.document.v1.updated"),
  (error: any) => error?.statusCode === 400,
);

assert.throws(
  () => decodificarUsuarioIdDesdeDocumentEvent(
    codificarDocumentEventParaPrueba(`${validName}/cursos/curso-1`),
  ),
  (error: any) => error?.statusCode === 400,
);

assert.throws(
  () => extraerUsuarioIdDesdeNombreDocumento(
    "projects/sidca-a33f0/databases/(default)/documents/afiliados/ABC123",
  ),
  (error: any) => error?.statusCode === 400,
);

assert.throws(
  () => decodificarUsuarioIdDesdeDocumentEvent(Buffer.from([0xff, 0x00])),
  (error: any) => error?.statusCode === 400,
);

const counterName = "projects/x/databases/(default)/documents/nuevoAfiliado_counters/1234567";
const pending = {
  estadoReafiliacion: "pendiente",
  requiereRevisionComision: true,
  fechaSolicitudReafiliacion: "2026-09-17T12:00:00.000Z",
  usuarioIdHistorico: "ABC123",
};
assert.equal(esSolicitudPendienteValida(pending), true);
assert.equal(decidirCorreoPendiente(pending, pending.fechaSolicitudReafiliacion), "procesar");
assert.equal(decidirCorreoPendiente({ ...pending, estadoReafiliacion: "aprobada" }, pending.fechaSolicitudReafiliacion), "omitido");
assert.equal(decidirCorreoPendiente({ ...pending, estadoReafiliacion: "rechazada" }, pending.fechaSolicitudReafiliacion), "omitido");
assert.equal(decidirCorreoPendiente({ ...pending, requiereRevisionComision: false }, pending.fechaSolicitudReafiliacion), "omitido");
assert.equal(
  decidirCorreoPendiente(
    { ...pending, correoReafiliacionPendienteEstado: "enviado", correoReafiliacionPendienteSolicitudId: pending.fechaSolicitudReafiliacion },
    pending.fechaSolicitudReafiliacion,
  ),
  "ya_enviado",
);
assert.equal(
  decidirCorreoPendiente(
    { ...pending, correoReafiliacionPendienteEstado: "procesando", correoReafiliacionPendienteSolicitudId: pending.fechaSolicitudReafiliacion, correoReafiliacionPendienteProcesandoAt: "2026-09-17T11:55:00.000Z" },
    pending.fechaSolicitudReafiliacion,
    Date.parse("2026-09-17T12:00:00.000Z"),
  ),
  "procesando",
);
assert.equal(
  decidirCorreoPendiente(
    { ...pending, correoReafiliacionPendienteEstado: "enviado", correoReafiliacionPendienteSolicitudId: pending.fechaSolicitudReafiliacion },
    "2026-10-17T12:00:00.000Z",
  ),
  "procesar",
);
assert.equal(emailValidoReafiliacion("persona@example.com"), true);
assert.equal(emailValidoReafiliacion("no-es-un-email"), false);
const safeHtml = buildReaffiliationPendingEmailHtml({
  dni: "1234567",
  nombre: "<script>alert(1)</script>",
  fechaSolicitud: "17/09/2026 09:00 hs",
  nroAfiliacion: "A-1",
});
assert.equal(safeHtml.includes("&lt;script&gt;"), true);
assert.equal(safeHtml.includes("<script>alert(1)</script>"), false);

const approved = {
  ...pending,
  estadoReafiliacion: "aprobada",
  fechaResolucionReafiliacion: "2026-09-17T13:00:00.000Z",
};
assert.equal(esSolicitudAprobadaValida(approved), true);
assert.equal(decidirCorreoAprobada(approved, pending.fechaSolicitudReafiliacion), "procesar");
assert.equal(decidirCorreoAprobada({ ...approved, estadoReafiliacion: "pendiente" }, pending.fechaSolicitudReafiliacion), "omitido");
assert.equal(decidirCorreoAprobada({ ...approved, estadoReafiliacion: "rechazada" }, pending.fechaSolicitudReafiliacion), "omitido");
assert.equal(
  decidirCorreoAprobada(
    { ...approved, correoReafiliacionAprobadaEstado: "enviado", correoReafiliacionAprobadaSolicitudId: pending.fechaSolicitudReafiliacion },
    pending.fechaSolicitudReafiliacion,
  ),
  "ya_enviado",
);
assert.equal(
  decidirCorreoAprobada(
    { ...approved, correoReafiliacionAprobadaEstado: "procesando", correoReafiliacionAprobadaSolicitudId: pending.fechaSolicitudReafiliacion, correoReafiliacionAprobadaProcesandoAt: "2026-09-17T12:55:00.000Z" },
    pending.fechaSolicitudReafiliacion,
    Date.parse("2026-09-17T13:00:00.000Z"),
  ),
  "procesando",
);
assert.equal(
  decidirCorreoAprobada(
    { ...approved, correoReafiliacionAprobadaEstado: "enviado", correoReafiliacionAprobadaSolicitudId: pending.fechaSolicitudReafiliacion },
    "2026-10-17T12:00:00.000Z",
  ),
  "procesar",
);
const approvedHtml = buildReaffiliationApprovedEmailHtml({
  dni: "1234567",
  nombre: "<script>alert(1)</script>",
  fechaAprobacion: "17/09/2026 10:00 hs",
  nroAfiliacion: null,
});
assert.equal(approvedHtml.includes("&lt;script&gt;"), true);
assert.equal(approvedHtml.includes("<script>alert(1)</script>"), false);

const rejected = {
  ...pending,
  estadoReafiliacion: "rechazada",
  requiereRevisionComision: false,
  fechaResolucionReafiliacion: "2026-09-17T14:00:00.000Z",
  observacionResolucion: "Falta documentación <script>alert(1)</script>",
};
assert.equal(esSolicitudRechazadaValida(rejected), true);
assert.equal(decidirCorreoRechazada(rejected, pending.fechaSolicitudReafiliacion), "procesar");
assert.equal(decidirCorreoRechazada({ ...rejected, estadoReafiliacion: "pendiente" }, pending.fechaSolicitudReafiliacion), "omitido");
assert.equal(decidirCorreoRechazada({ ...rejected, estadoReafiliacion: "aprobada" }, pending.fechaSolicitudReafiliacion), "omitido");
assert.equal(
  decidirCorreoRechazada(
    { ...rejected, correoReafiliacionRechazadaEstado: "enviado", correoReafiliacionRechazadaSolicitudId: pending.fechaSolicitudReafiliacion },
    pending.fechaSolicitudReafiliacion,
  ),
  "ya_enviado",
);
assert.equal(
  decidirCorreoRechazada(
    { ...rejected, correoReafiliacionRechazadaEstado: "procesando", correoReafiliacionRechazadaSolicitudId: pending.fechaSolicitudReafiliacion, correoReafiliacionRechazadaProcesandoAt: "2026-09-17T13:55:00.000Z" },
    pending.fechaSolicitudReafiliacion,
    Date.parse("2026-09-17T14:00:00.000Z"),
  ),
  "procesando",
);
assert.equal(
  decidirCorreoRechazada(
    { ...rejected, correoReafiliacionRechazadaEstado: "enviado", correoReafiliacionRechazadaSolicitudId: pending.fechaSolicitudReafiliacion },
    "2026-10-17T12:00:00.000Z",
  ),
  "procesar",
);
const rejectedHtml = buildReaffiliationRejectedEmailHtml({
  dni: "1234567",
  nombre: "<script>alert(1)</script>",
  fechaResolucion: "17/09/2026 11:00 hs",
  nroAfiliacion: null,
  motivo: "Falta <b>documentación</b>",
});
assert.equal(rejectedHtml.includes("&lt;script&gt;"), true);
assert.equal(rejectedHtml.includes("&lt;b&gt;documentación&lt;/b&gt;"), true);
const rejectedWithoutReasonHtml = buildReaffiliationRejectedEmailHtml({
  dni: "1234567",
  nombre: "Persona",
  fechaResolucion: "17/09/2026 11:00 hs",
});
assert.equal(rejectedWithoutReasonHtml.includes("MOTIVO INFORMADO"), false);

validarTipoCloudEventEsperado(FIRESTORE_UPDATED_EVENT_TYPE, FIRESTORE_UPDATED_EVENT_TYPE);
assert.throws(
  () => validarTipoCloudEventEsperado(FIRESTORE_CREATED_EVENT_TYPE, FIRESTORE_UPDATED_EVENT_TYPE),
  (error: any) => error?.statusCode === 400,
);
assert.equal(decodificarDniDesdeCounterEvent(codificarDocumentEventParaPrueba(counterName)), "1234567");
assert.throws(
  () => decodificarDniDesdeCounterEvent(codificarDocumentEventParaPrueba(`${counterName}/subcoleccion/ABC`)),
  (error: any) => error?.statusCode === 400,
);
assert.throws(
  () => decodificarDniDesdeCounterEvent(codificarDocumentEventParaPrueba("projects/x/databases/(default)/documents/otra/1234567")),
  (error: any) => error?.statusCode === 400,
);

const historicoBase = {
  usuario: { dni: "012345678" },
  counter: {},
  usuarioIdHistorico: "historico-1",
  dni: "012345678",
};
assert.equal(
  resolverNumeroAfiliacionHistorico({
    ...historicoBase,
    nuevoAfiliadosPorUsuarioId: [{ id: "evento-1", usuarioId: "historico-1", dni: "012345678", nroAfiliacion: 1 }],
    nuevoAfiliadosPorDni: [],
  }).nroAfiliacion,
  1,
);
assert.equal(
  resolverNumeroAfiliacionHistorico({
    ...historicoBase,
    nuevoAfiliadosPorUsuarioId: [],
    nuevoAfiliadosPorDni: [{ id: "evento-dni", usuarioId: "historico-1", dni: "012345678", nroAfiliacion: "001" }],
  }).nroAfiliacion,
  1,
);
assert.equal(
  resolverNumeroAfiliacionHistorico({
    ...historicoBase,
    nuevoAfiliadosPorUsuarioId: [
      { id: "evento-1", usuarioId: "historico-1", dni: "012345678", nroAfiliacion: 1 },
      { id: "evento-2", usuarioId: "historico-1", dni: "012345678", nroAfiliacion: "1" },
    ],
    nuevoAfiliadosPorDni: [],
  }).nroAfiliacion,
  1,
);
assert.throws(
  () => resolverNumeroAfiliacionHistorico({
    ...historicoBase,
    nuevoAfiliadosPorUsuarioId: [
      { id: "evento-1", usuarioId: "historico-1", dni: "012345678", nroAfiliacion: 1 },
      { id: "evento-2", usuarioId: "historico-1", dni: "012345678", nroAfiliacion: 2 },
    ],
    nuevoAfiliadosPorDni: [],
  }),
  (error: any) => error?.statusCode === 409,
);
assert.throws(
  () => resolverNumeroAfiliacionHistorico({
    ...historicoBase,
    nuevoAfiliadosPorUsuarioId: [],
    nuevoAfiliadosPorDni: [],
  }),
  (error: any) => error?.statusCode === 409,
);
assert.equal(
  resolverNumeroAfiliacionHistorico({
    ...historicoBase,
    counter: { last: 1 },
    nuevoAfiliadosPorUsuarioId: [],
    nuevoAfiliadosPorDni: [],
  }).source,
  "counter_last",
);
assert.equal(
  resolverNumeroAfiliacionHistorico({
    ...historicoBase,
    counter: { last: 1 },
    usuario: { dni: "012345678", nroAfiliacion: 5 },
    nuevoAfiliadosPorUsuarioId: [],
    nuevoAfiliadosPorDni: [],
  }).nroAfiliacion,
  5,
);
assert.equal(
  resolverNumeroAfiliacionHistorico({
    ...historicoBase,
    counter: { nroAfiliacionHistorico: 4, last: 1 },
    nuevoAfiliadosPorUsuarioId: [],
    nuevoAfiliadosPorDni: [],
  }).nroAfiliacion,
  4,
);
for (const last of [0, null]) {
  assert.throws(
    () => resolverNumeroAfiliacionHistorico({
      ...historicoBase,
      counter: { last },
      nuevoAfiliadosPorUsuarioId: [],
      nuevoAfiliadosPorDni: [],
    }),
    (error: any) => error?.statusCode === 409,
  );
}

console.log("Firestore DocumentEventData protobuf tests: OK");
