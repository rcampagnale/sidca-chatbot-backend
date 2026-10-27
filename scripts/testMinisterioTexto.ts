import assert from "node:assert/strict";
import PDFDocument from "pdfkit";
import {
  construirTextoMinisterio,
  ajustarTexto,
  FIRMANTES_MINISTERIO_TIPOGRAFIA,
  formatearTituloMinisterio,
} from "../src/certificados/certificadoMinisterioPdfRenderer.js";
import { nombreInstitucionOrganizadora } from "../src/certificados/institucionOrganizadora.js";

const certificado = {
  institucionCertificado: "ministerio",
  institucionOrganizadora: "sidca",
  tipoActividad: "CURSO",
  titulo: "POR UNA PEDAGOGÍA DEL CUIDADO II: SUICIDIO Y PROBLEMÁTICAS EMERGENTES.",
  modalidad: "HÍBRIDO",
  fechaInicio: "2026-08-07",
  fechaFin: "2026-08-15",
  localidad: "SAN FERNANDO DEL VALLE",
  departamento: "CAPITAL",
  cargaHoraria: "36",
  niveles: ["SUPERIOR", "INICIAL", "SECUNDARIO", "PRIMARIO"],
  textoEvaluacion: "y el correspondiente trabajo de evaluación.",
  textoAuspicio:
    "Este evento de capacitación, fortalecimiento y actualización docente fue auspiciado por el Ministerio de Educación y Trabajo, a través de Resolución",
  resolucion: "RESOL-2026-35-E-CAT-SICE#MET",
  fecha: "2026-10-14",
  firmantesMinisterio: [
    {
      nombre: "Lic. Marcelo Luis Altamirano.",
      cargo: "SECRETARIA DE INNOVACION Y CALIDAD EDUCATIVA.",
      organismo: "MINISTERIO DE EDUCACION Y TRABAJO.",
    },
    {
      nombre: "Mgter. David Sánchez S.",
      cargo: "DIR. PROV. DE MONITOREO DE TRAYECTORIAS Y DESARROLLO PROFESIONAL.",
      organismo: "SECRETARIA DE INNOVACION Y CALIDAD EDUCATIVA. MINISTERIO DE EDUCACION Y TRABAJO.",
    },
    {
      nombre: "Dr. Sergio Guillamondegui.",
      cargo: "SECRETARIO GENERAL.",
      organismo: "SINDICATO DE DOCENTES DE CATAMARCA.",
    },
  ],
};

const participante = {
  apellidoNombre: "PRUEBA UNO, USUARIO UNO",
  dni: "123456789",
};

const individual = construirTextoMinisterio(certificado, participante);
const bulk = construirTextoMinisterio(certificado, participante);

assert.equal(nombreInstitucionOrganizadora(certificado), "SINDICATO DE DOCENTES DE CATAMARCA - SI.D.CA.");
assert.equal(
  nombreInstitucionOrganizadora({ ...certificado, institucionOrganizadora: "itm" }),
  "INSTITUTO TECNOLOGICO MUNICIPAL (I.T.M)"
);
assert.equal(
  individual.titulo,
  '"POR UNA PEDAGOGÍA DEL CUIDADO II: SUICIDIO Y PROBLEMÁTICAS EMERGENTES."'
);
assert.equal(individual.dictado, "Dictado de manera HÍBRIDO, desde 07/08/26 al 15/08/26, en la Localidad de SAN FERNANDO DEL VALLE -DPTO CAPITAL");
assert.equal(individual.duracion, "con una duración de 36 horas cátedra, y el correspondiente trabajo de evaluación.");
assert.equal(individual.auspicio, "Este evento de capacitación, fortalecimiento y actualización docente fue auspiciado por el Ministerio de Educación y Trabajo, a través de Resolución RESOL-2026-35-E-CAT-SICE#MET.");
assert.equal(individual.expedicion, "El presente certificado se expide en San Fernando Del Valle de Catamarca a 14 días del mes de octubre de 2026.");
assert.equal(individual.textoAuspicioFinal, individual.auspicio);
assert.equal(individual.fechaExpedicion, individual.expedicion);
assert.deepEqual(individual.firmantes, certificado.firmantesMinisterio);
assert.equal(individual.titulo, bulk.titulo);
assert.equal(individual.dictado, bulk.dictado);
assert.equal(individual.duracion, bulk.duracion);
assert.equal(individual.auspicio, bulk.auspicio);
assert.equal(individual.expedicion, bulk.expedicion);
assert.deepEqual(individual.firmantes, bulk.firmantes);
assert.equal(formatearTituloMinisterio('“POR UNA PEDAGOGÍA DEL CUIDADO II: SUICIDIO Y PROBLEMÁTICAS EMERGENTES.”'), individual.titulo);

const pdf = new PDFDocument({ size: "A4", layout: "landscape", margin: 0 });
const nombreAjustado = ajustarTexto(pdf, certificado.firmantesMinisterio[1].nombre, {
  ancho: 250,
  inicial: FIRMANTES_MINISTERIO_TIPOGRAFIA.nombre.inicial,
  minimo: FIRMANTES_MINISTERIO_TIPOGRAFIA.nombre.minimo,
  maxLineas: FIRMANTES_MINISTERIO_TIPOGRAFIA.nombre.maxLineas,
  interlineado: FIRMANTES_MINISTERIO_TIPOGRAFIA.nombre.interlineado,
});
assert.equal(nombreAjustado.fontSize, 8.2);
assert.equal(nombreAjustado.lines.length, 1);

const cargoLargoAjustado = ajustarTexto(
  pdf,
  certificado.firmantesMinisterio[1].cargo,
  {
    ancho: 245,
    inicial: FIRMANTES_MINISTERIO_TIPOGRAFIA.detalle.inicial,
    minimo: FIRMANTES_MINISTERIO_TIPOGRAFIA.detalle.minimo,
    maxLineas: FIRMANTES_MINISTERIO_TIPOGRAFIA.detalle.maxLineas,
    interlineado: FIRMANTES_MINISTERIO_TIPOGRAFIA.detalle.interlineado,
  }
);
assert.equal(cargoLargoAjustado.fontSize, 6.2);
assert.equal(cargoLargoAjustado.lines.length, 1);
assert.equal(FIRMANTES_MINISTERIO_TIPOGRAFIA.margenHorizontal, 2.5);
assert.equal(FIRMANTES_MINISTERIO_TIPOGRAFIA.escalaHorizontal, 1);
assert.equal(individual.firmantes[1].nombre, "Mgter. David Sánchez S.");

console.log("OK: texto canónico Ministerio y paridad individual/masivo");
console.log("OK: tipografía de firmantes, wrap controlado y escala horizontal 100%");
