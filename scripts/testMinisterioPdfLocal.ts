import fs from "node:fs";
import path from "node:path";
import PDFDocument from "pdfkit";
import { renderCertificadoMinisterioPdfPage } from "../src/certificados/certificadoMinisterioPdfRenderer.js";

const output = path.resolve(process.cwd(), "tmp", "ministerio-render-test.pdf");
const firmasDir = path.resolve(process.cwd(), "scripts", "fixtures", "ministerio");

const firmasLocales: Record<string, string> = {
  firmante_1: path.join(firmasDir, "firma_carlos_ortiz.png"),
  firmante_2: path.join(firmasDir, "firma_david_sanchez.png"),
  firmante_3: path.join(firmasDir, "firma_sergio_guillamondegui.png"),
};

const firmantesMinisterio = [
  {
    id: "firmante_1",
    orden: 1,
    nombre: "Lic. Carlos Alejandro Ortiz",
    cargo: "SECRETARIO DE INNOVACION Y CALIDAD EDUCATIVA",
    organismo: "MINISTERIO DE EDUCACION Y TRABAJO",
    activo: true,
    imagenStoragePath: "fixture/firma_carlos_ortiz.png",
    imagenVersion: 1,
    imagenSha256: "a".repeat(64),
  },
  {
    id: "firmante_2",
    orden: 2,
    nombre: "David Sánchez S.",
    cargo: "DIRECTOR PROVINCIAL DE MONITOREO DE TRAYECTORIAS Y DESARROLLO PROFESIONAL.",
    organismo: "SECRETARIA DE INNOVACION Y CALIDAD EDUCATIVA. MINISTERIO DE EDUCACION Y TRABAJO",
    activo: true,
    imagenStoragePath: "fixture/firma_david_sanchez.png",
    imagenVersion: 1,
    imagenSha256: "b".repeat(64),
  },
  {
    id: "firmante_3",
    orden: 3,
    nombre: "Dr. Sergio Guillamondegui",
    cargo: "SECRETARIO GENERAL",
    organismo: "SINDICATO DE DOCENTES DE CATAMARCA",
    activo: true,
    imagenStoragePath: "fixture/firma_sergio_guillamondegui.png",
    imagenVersion: 1,
    imagenSha256: "c".repeat(64),
  },
];

const crearEmision = ({ titulo, cargaHoraria, token }: {
  titulo: string;
  cargaHoraria: string;
  token: string;
}) => ({
  cursoId: "TEST-MINISTERIO",
  token,
  certificadoId: token,
  estado: "vigente",
  participante: {
    apellidoNombre: "PRUEBA UNO, USUARIO UNO",
    dni: "123456789",
  },
  certificado: {
    institucionCertificado: "ministerio",
    layoutVersion: "ministerio-v1",
    tipoActividad: "CURSO",
    titulo,
    modalidad: "PRESENCIAL",
    fechaInicio: "2026-05-22",
    fechaFin: "2026-05-31",
    localidad: "SAN FERNANDO DEL VALLE",
    departamento: "CAPITAL",
    cargaHoraria,
    niveles: ["SUPERIOR", "INICIAL", "SECUNDARIO", "PRIMARIO"],
    textoEvaluacion: "y el correspondiente trabajo de evaluación.",
    textoAuspicio:
      "Este evento de capacitación, fortalecimiento y actualización docente fue auspiciado por el Ministerio de Educación y Trabajo, a través de Resolución S.I.E.(MECYT) N°",
    resolucion: "03/2026",
    fecha: "2026-08-31",
    firmantesMinisterio,
  },
});

const contarPaginas = (buffer: Buffer) =>
  (buffer.toString("latin1").match(/\/Type\s*\/Page\b/g) || []).length;

const run = async () => {
  fs.mkdirSync(path.dirname(output), { recursive: true });

  const emisiones = [
    crearEmision({
      titulo: "CURSO DE PRUEBA 2024",
      cargaHoraria: "40",
      token: "token-prueba-ministerio-001",
    }),
    crearEmision({
      titulo:
        "I CONGRESO INTERNACIONAL POR UNA PEDAGOGÍA DEL CUIDADO: EDUCACIÓN, INCLUSIÓN Y PROBLEMÁTICAS EMERGENTES EN LOS NUEVOS ESCENARIOS EDUCATIVOS",
      cargaHoraria: "40 HS CÁTEDRAS",
      token: "token-prueba-ministerio-002",
    }),
  ];

  const stream = fs.createWriteStream(output);
  const pdf = new PDFDocument({
    autoFirstPage: false,
    size: "A4",
    layout: "landscape",
    margin: 0,
  });
  pdf.pipe(stream);

  const cargarFirmaHistorica = async (firmante: any) => {
    const archivo = firmasLocales[String(firmante?.id || "")];
    if (!archivo) throw new Error(`No existe fixture para ${firmante?.id || "firmante"}.`);
    return fs.promises.readFile(archivo);
  };

  for (const emision of emisiones) {
    await renderCertificadoMinisterioPdfPage(pdf, emision, {
      cargarFirmaHistorica,
    });
  }

  pdf.end();
  await new Promise<void>((resolve, reject) => {
    stream.once("finish", resolve);
    stream.once("error", reject);
  });

  const buffer = fs.readFileSync(output);
  const paginas = contarPaginas(buffer);
  if (!buffer.length) throw new Error("El PDF local quedó vacío.");
  if (paginas !== 2) throw new Error(`Se esperaban 2 páginas y se generaron ${paginas}.`);

  console.log(`PDF generado: ${output}`);
  console.log(`Páginas: ${paginas}`);
  console.log(`Tamaño: ${buffer.length} bytes`);
  console.log("QR página 1: https://sidcagremio.com/validar-certificado/TEST-MINISTERIO/token-prueba-ministerio-001");
};

run().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
