import fs from "node:fs";
import path from "node:path";
import PDFDocument from "pdfkit";
import { renderCertificadoMinisterioPdfPage } from "../src/certificados/certificadoMinisterioPdfRenderer.js";

const output = path.resolve(
  process.env.MINISTERIO_PDF_TEST_OUTPUT || path.resolve(process.cwd(), "tmp", "ministerio-render-test.pdf")
);
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
    nombre: "Lic. Marcelo Luis Altamirano.",
    cargo: "SECRETARIA DE INNOVACION Y CALIDAD EDUCATIVA.",
    organismo: "MINISTERIO DE EDUCACION Y TRABAJO.",
    activo: true,
    imagenStoragePath: "fixture/firma_carlos_ortiz.png",
    imagenVersion: 1,
    imagenSha256: "a".repeat(64),
  },
  {
    id: "firmante_2",
    orden: 2,
    nombre: "Mgter. David Sánchez S.",
    cargo: "DIR. PROV. DE MONITOREO DE TRAYECTORIAS Y DESARROLLO PROFESIONAL.",
    organismo: "SECRETARIA DE INNOVACION Y CALIDAD EDUCATIVA. MINISTERIO DE EDUCACION Y TRABAJO.",
    activo: true,
    imagenStoragePath: "fixture/firma_david_sanchez.png",
    imagenVersion: 1,
    imagenSha256: "b".repeat(64),
  },
  {
    id: "firmante_3",
    orden: 3,
    nombre: "Dr. Sergio Guillamondegui.",
    cargo: "SECRETARIO GENERAL.",
    organismo: "SINDICATO DE DOCENTES DE CATAMARCA.",
    activo: true,
    imagenStoragePath: "fixture/firma_sergio_guillamondegui.png",
    imagenVersion: 1,
    imagenSha256: "c".repeat(64),
  },
];

const crearEmision = ({ titulo, cargaHoraria, token, institucionOrganizadora, textoAuspicio }: {
  titulo: string;
  cargaHoraria: string;
  token: string;
  institucionOrganizadora?: "sidca" | "itm";
  textoAuspicio?: string;
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
    ...(institucionOrganizadora ? { institucionOrganizadora } : {}),
    layoutVersion: "ministerio-v1",
    tipoActividad: "CURSO",
    titulo,
    modalidad: "HÍBRIDO",
    fechaInicio: "2026-08-07",
    fechaFin: "2026-08-15",
    localidad: "SAN FERNANDO DEL VALLE",
    departamento: "CAPITAL",
    cargaHoraria,
    niveles: ["SUPERIOR", "INICIAL", "SECUNDARIO", "PRIMARIO"],
    textoEvaluacion: "y el correspondiente trabajo de evaluación.",
    ...(textoAuspicio !== undefined ? { textoAuspicio } : {}),
    resolucion: "RESOL-2026-35-E-CAT-SICE#MET",
    fecha: "2026-10-14",
    firmantesMinisterio,
  },
});

const contarPaginas = (buffer: Buffer) =>
  (buffer.toString("latin1").match(/\/Type\s*\/Page\b/g) || []).length;

const run = async () => {
  fs.mkdirSync(path.dirname(output), { recursive: true });

  const emisiones = [
    crearEmision({
      titulo: "POR UNA PEDAGOGÍA DEL CUIDADO II: SUICIDIO Y PROBLEMÁTICAS EMERGENTES.",
      cargaHoraria: "36",
      token: "token-prueba-ministerio-001",
      institucionOrganizadora: "sidca",
      textoAuspicio:
        "Este evento de capacitación, fortalecimiento y actualización docente fue auspiciado por el Ministerio de Educación y Trabajo, a través de Resolución",
    }),
    crearEmision({
      titulo:
        "I CONGRESO INTERNACIONAL POR UNA PEDAGOGÍA DEL CUIDADO: EDUCACIÓN, INCLUSIÓN Y PROBLEMÁTICAS EMERGENTES EN LOS NUEVOS ESCENARIOS EDUCATIVOS",
      cargaHoraria: "40 HS CÁTEDRAS",
      token: "token-prueba-ministerio-002",
      institucionOrganizadora: "itm",
    }),
    crearEmision({
      titulo: "CURSO LEGACY SIN INSTITUCIÓN",
      cargaHoraria: "20",
      token: "token-prueba-ministerio-legacy",
    }),
  ];

  const stream = fs.createWriteStream(output);
  const pdf = new PDFDocument({
    autoFirstPage: false,
    size: "A4",
    layout: "landscape",
    margin: 0,
    compress: false,
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
  if (paginas !== 3) throw new Error(`Se esperaban 3 páginas y se generaron ${paginas}.`);

  console.log(`PDF generado: ${output}`);
  console.log(`Páginas: ${paginas}`);
  console.log(`Tamaño: ${buffer.length} bytes`);
  console.log("Instituciones renderizadas: SIDCA + ITM + legacy sin institución");
  console.log("Texto de auspicio personalizado: página 1");
  console.log("QR página 1: https://sidcagremio.com/validar-certificado/TEST-MINISTERIO/token-prueba-ministerio-001");
};

run().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
