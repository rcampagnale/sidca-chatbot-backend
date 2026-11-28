import OpenAI from "openai";
import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { chunkMatchesNormativeReference, parseNormativeReferences, rankNormative, type RagChunk } from "./ragCore.js";

export type DominioBackend =
  | "licencias"
  | "estatuto"
  | "general"
  | "coberturas"
  | "servicios"
  | "nomenclador"
  | "sumarios"
  | "educacion_nacional"
  | "educacion_provincial"
  | "educacion_privada"
  | "educacion_tecnico_profesional";

export type ChatbotQueryInput = {
  pregunta: string;
  dominio?: DominioBackend;
  maxResults?: number;
  contextoConversacional?: string;
};

export type SearchHit = {
  filename?: string;
  score?: number;
  text?: string;
};

export type BackendArticulo = {
  capitulo?: string;
  titulo_capitulo?: string;
  articulo?: number | string;
  titulo_articulo?: string;
  texto?: string;
  resumen?: string | null;
  fuente?: string;
  otorgante?: string;
  interviniente?: string;
  seccion?: string;
  nivel?: string;
  tipo?: string;
  [key: string]: unknown;
};

export type BackendServiceContact = {
  area?: string;
  whatsapp?: string;
  url?: string;
};

export type ChatbotWorkflowOutput = {
  ok: boolean;
  tipo: string;
  dominio: DominioBackend;
  origen: string;
  consulta: string;
  consultaNormalizada: string;
  respuesta: string;
  articulos: BackendArticulo[];
  referencias: string[];
  busqueda: SearchHit[];
  conversationId: string | null;
  asesoramientoGremial?: BackendServiceContact;
  error?: string;
};

type SourceConfig = {
  id?: string;
  dominio: DominioBackend;
  source: string;
  filename: string;
  referencia: string;
  norma?: string;
};

type LocalChunkKind = "articulo" | "resumen" | "documento";

type LocalChunk = {
  id: string;
  kind: LocalChunkKind;
  dominio: DominioBackend;
  filename: string;
  source: string;
  referencia: string;
  articulo?: string;
  articleRefs: string[];
  titulo?: string;
  descripcion?: string;
  otorgante?: string;
  interviniente?: string;
  seccion?: string;
  nivel?: string;
  tipo?: string;
  norma?: string;
  sourceId?: string;
  metadata?: Record<string, unknown>;
  text: string;
  raw?: unknown;
};

type QueryExpansion = {
  dominio?: DominioBackend;
  triggers: string[];
  terms: string[];
  preferredArticles?: string[];
};

const NO_ENCONTRADO =
  "No encontré fragmentos suficientes en la base cargada para responder con precisión a esta consulta.";

const SERVICIO_NO_DISPONIBLE = "Servicio no disponible por el momento.";

const DEFAULT_LICENCIAS_URL =
  "https://raw.githubusercontent.com/rcampagnale/sidca-chatbot-docs/main/regimen_licencias_docentes.json";

const DEFAULT_ESTATUTO_URL =
  "https://raw.githubusercontent.com/rcampagnale/sidca-chatbot-docs/main/texto_provincial_para_firestore.json";

const DEFAULT_GENERAL_URL =
  "https://raw.githubusercontent.com/rcampagnale/sidca-chatbot-docs/main/consultas_generales.json";

const DEFAULT_COBERTURAS_URL =
  "https://raw.githubusercontent.com/rcampagnale/sidca-chatbot-docs/main/decreto_636_coberturas_actualizado_res3335.json";

const DEFAULT_SERVICIOS_URL =
  "https://raw.githubusercontent.com/rcampagnale/sidca-chatbot-docs/main/servicios_sidca_app.json";

const DEFAULT_MANIFEST_URL =
  "https://raw.githubusercontent.com/rcampagnale/sidca-chatbot-docs/main/fuentes_chatbot.json";

type ManifestFuente = {
  id?: string;
  dominio?: string;
  nombre?: string;
  referencia?: string;
  url?: string;
  activo?: boolean;
};

type SourcesManifest = {
  version?: string;
  actualizado?: string;
  descripcion?: string;
  fuentes?: ManifestFuente[];
};

let cachedChunks: LocalChunk[] | null = null;
let cachedAt = 0;
let cachedChunksComplete: LocalChunk[] | null = null;
let cachedChunksVersion = "";
let loadingChunksPromise: Promise<LocalChunk[]> | null = null;
let cachedSourceConfigs: SourceConfig[] | null = null;
let cachedInactiveSourceLabels: string[] = [];
const SOURCE_LOAD_CONCURRENCY = 4;
// El enfriamiento y la cola viven en memoria por instancia de Cloud Run; no representan una cuota global.
const GROQ_MAX_QUEUED_REQUESTS = 8;
const GROQ_QUEUE_WAIT_TIMEOUT_MS = 1500;
// Presupuesto estimado local: 1.000 tokens de margen sobre el TPM observado de 8.000.
const GROQ_ESTIMATED_TPM_BUDGET = 7000;
const RESPONSE_CACHE_TTL_MS = 24 * 60 * 60 * 1000;
const RESPONSE_CACHE_MAX_ENTRIES = 500;

type CachedChatbotResponse = {
  expiresAt: number;
  response: ChatbotWorkflowOutput;
};

const responseCache = new Map<string, CachedChatbotResponse>();
type WorkflowTestDependencies = { fetchJson?: (url: string) => Promise<unknown>; now?: () => number; groqClient?: OpenAI | null };
let workflowTestDependencies: WorkflowTestDependencies = {};
let groqCooldownUntil = 0;
let groqCooldownReason = "";
let groqQueue: Promise<void> = Promise.resolve();
let groqQueueDepth = 0;
export function setWorkflowTestDependencies(dependencies: WorkflowTestDependencies = {}): void {
  workflowTestDependencies = dependencies;
  cachedChunks = null; cachedChunksComplete = null; cachedAt = 0; cachedChunksVersion = ""; loadingChunksPromise = null; cachedSourceConfigs = null; cachedInactiveSourceLabels = []; responseCache.clear();
  groqCooldownUntil = 0; groqCooldownReason = ""; groqQueue = Promise.resolve(); groqQueueDepth = 0;
}
export function updateWorkflowTestDependencies(dependencies: WorkflowTestDependencies = {}): void {
  workflowTestDependencies = { ...workflowTestDependencies, ...dependencies };
}
const workflowNow = () => workflowTestDependencies.now?.() ?? Date.now();

const STOPWORDS = new Set([
  "para",
  "como",
  "cuando",
  "donde",
  "desde",
  "hasta",
  "sobre",
  "tengo",
  "quiero",
  "necesito",
  "consulta",
  "docente",
  "docentes",
  "licencia",
  "licencias",
  "justificacion",
  "justificaciones",
  "franquicia",
  "franquicias",
  "articulo",
  "articulos",
  "catamarca",
  "provincia",
  "provincial",
  "sidca",
  "cual",
  "cuanto",
  "cuantos",
  "cuanta",
  "cuantas",
  "debo",
  "puedo",
  "hacer",
  "tiene",
  "esta",
  "este",
  "estos",
  "estas",
  "ella",
  "ellos",
  "ellas",
  "usted",
  "ustedes",
  "ante",
  "bajo",
  "entre",
  "pero",
  "porque",
  "saber",
  "dice",
  "segun",
  "corresponde",
]);

const QUERY_EXPANSIONS: QueryExpansion[] = [
  {
    dominio: "licencias",
    triggers: ["licencia anual", "descanso anual", "vacaciones", "receso"],
    terms: ["licencia anual ordinaria descanso anual receso calendario escolar"],
    preferredArticles: ["12", "13", "14"],
  },
  {
    dominio: "licencias",
    triggers: [
      "corto tratamiento",
      "lesiones de corto tratamiento",
      "afecciones de corto tratamiento",
      "licencia corto tratamiento",
      "licencia por corto tratamiento",
      "enfermedad comun",
      "enfermedad común",
      "afeccion comun",
      "afección común",
      "gripe",
      "resfrio",
      "resfrío",
      "fiebre",
      "febricula",
      "febrícula",
      "enfermedad",
      "enfermo",
      "enferma",
      "afeccion",
      "afección",
      "lesion",
      "lesión",
      "medico",
      "médico",
      "reposo medico",
      "reposo médico",
      "certificado medico",
      "certificado médico",
      "operacion menor",
      "operación menor",
    ],
    terms: [
      "afecciones de corto tratamiento enfermedad comun gripe resfrio fiebre reposo medico certificado medico operaciones quirurgicas menores veinticinco dias",
    ],
    preferredArticles: ["16", "17"],
  },
  {
    dominio: "licencias",
    triggers: [
      "enfermedad hora",
      "me retire enfermo",
      "retirarme enfermo",
      "me enferme",
      "me enfermo",
      "retirarme por enfermedad",
      "en hora de labor",
      "mientras estaba trabajando",
      "same",
      "guardia",
    ],
    terms: ["enfermedades en hora de labor certificado medico guardia same"],
    preferredArticles: ["18"],
  },
  {
    dominio: "licencias",
    triggers: [
      "accidente de trabajo",
      "art",
      "enfermedad profesional",
      "riesgo de trabajo",
    ],
    terms: [
      "accidente de trabajo enfermedades profesionales ley riesgo de trabajo autoseguro",
    ],
    preferredArticles: ["19"],
  },
  {
    dominio: "licencias",
    triggers: [
      "largo tratamiento",
      "enfermedad de largo tratamiento",
      "tratamiento prolongado",
      "enfermedad prolongada",
      "afeccion prolongada",
      "reposo prolongado",
      "varios meses",
      "enfermedad grave",
      "junta medica",
      "junta médica",
      "complejo diagnostico",
      "complejo diagnóstico",
    ],
    terms: [
      "afecciones lesiones largo tratamiento tratamiento prolongado enfermedad prolongada reposo prolongado varios meses junta medica un año seis meses haberes",
    ],
    preferredArticles: ["20", "21", "22"],
  },
  {
    dominio: "licencias",
    triggers: [
      "incapacidad",
      "jubilacion por invalidez",
      "incapacidad permanente",
    ],
    terms: [
      "incapacidad total permanente parcial transitoria jubilacion por invalidez",
    ],
    preferredArticles: ["23", "24"],
  },
  {
    dominio: "licencias",
    triggers: [
      "cambio de funciones",
      "cambio de tareas",
      "cft",
      "cfd",
      "reduccion aptitud",
      "reducción aptitud",
    ],
    terms: [
      "cambio de funciones temporal definitivo reduccion aptitud laboral junta medica",
    ],
    preferredArticles: ["25"],
  },
  {
    dominio: "licencias",
    triggers: ["violencia de genero", "violencia género", "violencia"],
    terms: ["violencia de genero reduccion jornada reordenacion tiempo trabajo"],
    preferredArticles: ["26"],
  },
  {
    dominio: "licencias",
    triggers: [
      "maternidad",
      "embarazo",
      "parto",
      "prenatal",
      "postnatal",
      "prematuro",
      "mama",
      "mamá",
      "mam",
      "soy mama",
      "soy mamá",
      "soy mam",
      "madre",
      "voy a ser mama",
      "voy a ser mamá",
      "voy a ser mam",
      "estoy embarazada",
      "licencia para madre",
      "licencia por ser mama",
      "licencia por ser mamá",
    ],
    terms: [
      "maternidad embarazo parto nacimiento multiple prematuro defuncion fetal ciento veinte dias articulo 27",
    ],
    preferredArticles: ["27"],
  },
  {
    dominio: "licencias",
    triggers: [
      "hijos menores",
      "hijo menor",
      "conyuge fallezca",
      "conviviente fallezca",
    ],
    terms: ["atencion hijos menores conyuge conviviente fallezca diez años"],
    preferredArticles: ["28"],
  },
  {
    dominio: "licencias",
    triggers: ["adopcion", "adopción", "tenencia", "guarda"],
    terms: ["tenencia fines de adopcion certificado tenencia provisoria sentencia"],
    preferredArticles: ["29"],
  },
  {
    dominio: "licencias",
    triggers: [
      "grupo familiar",
      "familiar enfermo",
      "familiar accidentado",
      "atencion familiar",
      "atención familiar",
      "madre enferma",
      "padre enfermo",
      "hijo enfermo",
    ],
    terms: [
      "atencion grupo familiar cuidado familiar asistencia personal madre padre hijo enfermo accidentado quince dias cuarenta cinco dias con goce haberes",
    ],
    preferredArticles: ["30"],
  },
  {
    dominio: "licencias",
    triggers: ["matrimonio", "casamiento", "casarme", "enlace"],
    terms: ["matrimonio docente quince dias corridos partida registro civil"],
    preferredArticles: ["31"],
  },
  {
    dominio: "licencias",
    triggers: ["aislamiento", "epidemia", "pandemia"],
    terms: ["aislamiento epidemia pandemia cuarenta dias acto administrativo"],
    preferredArticles: ["32"],
  },
  {
    dominio: "licencias",
    triggers: [
      "persona disfuncionada",
      "discapacidad",
      "discapacitada",
      "persona con discapacidad",
    ],
    terms: ["atencion persona disfuncionada ley 4793 normas complementarias"],
    preferredArticles: ["33"],
  },
  {
    dominio: "licencias",
    triggers: [
      "examen",
      "examenes",
      "exámenes",
      "rendir",
      "mesa examinadora",
    ],
    terms: [
      "estudios examenes practicas rendir examen finales nivel superior universitario postgrado treinta dias cinco dias articulo 35",
    ],
    preferredArticles: ["35"],
  },
  {
    dominio: "licencias",
    triggers: [
      "practica",
      "práctica",
      "practicas",
      "prácticas",
      "practica obligatoria",
    ],
    terms: [
      "practicas obligatorias planes de estudios institutos superiores quince dias cinco dias superposicion horaria",
    ],
    preferredArticles: ["36"],
  },
  {
    dominio: "licencias",
    triggers: [
      "postgrado",
      "post-grado",
      "investigacion",
      "investigación",
      "estudios en el extranjero",
      "beca",
    ],
    terms: [
      "estudios investigaciones post grado trabajos cientificos tecnicos un año prorroga",
    ],
    preferredArticles: ["37"],
  },
  {
    dominio: "licencias",
    triggers: [
      "actividad educativa",
      "cientifica",
      "científica",
      "cultural",
      "tecnologica",
      "tecnológica",
    ],
    terms: [
      "actividades interes educativo cientifico cultural tecnologico representen provincia pais oficial",
    ],
    preferredArticles: ["38"],
  },
  {
    dominio: "licencias",
    triggers: ["deporte", "deportiva", "deportivas", "actividad deportiva"],
    terms: ["actividades deportivas no rentadas representacion deportiva"],
    preferredArticles: ["39"],
  },
  {
    dominio: "licencias",
    triggers: [
      "representacion gremial",
      "representación gremial",
      "mandato gremial",
      "gremio",
      "sindicato",
    ],
    terms: [
      "representacion gremial entidad personeria gremial mandato funcion representacion",
    ],
    preferredArticles: ["40"],
  },
  {
    dominio: "licencias",
    triggers: [
      "capacitacion",
      "capacitación",
      "capacitaciones",
      "capacitaci",
      "capacitac",
      "perfeccionamiento",
      "perfeccion",
      "curso",
      "cursos",
      "congreso",
      "congresos",
      "taller",
      "talleres",
      "jornada",
      "jornadas",
      "conferencia",
      "conferencias",
      "simposio",
      "simposios",
      "actualizacion profesional",
      "actualización profesional",
    ],
    terms: [
      "capacitacion perfeccionamiento docente talleres congresos cursos conferencias simposios actividad cultural auspicio oficial diez dias habiles cinco dias anticipacion articulo 41",
    ],
    preferredArticles: ["41"],
  },
  {
    dominio: "licencias",
    triggers: ["candidato", "candidatura", "elecciones", "cargo electivo"],
    terms: ["candidatura cargos electivos oficializacion candidatura acto comicial"],
    preferredArticles: ["42"],
  },
  {
    dominio: "licencias",
    triggers: ["cargo electivo", "funciones superiores", "gobierno"],
    terms: ["ejercicio cargos electivos funciones superiores gobierno reintegrarse cinco dias"],
    preferredArticles: ["43"],
  },
  {
    dominio: "licencias",
    triggers: [
      "mayor jerarquia",
      "mayor jerarquía",
      "cargo mayor",
      "incompatibilidad",
    ],
    terms: ["cargo mayor jerarquia incompatibilidad sin goce sueldo quince dias cese"],
    preferredArticles: ["44"],
  },
  {
    dominio: "licencias",
    triggers: ["asuntos particulares", "sin goce", "particular"],
    terms: ["asuntos particulares comun especial seis meses decenio sin goce haberes"],
    preferredArticles: ["45"],
  },
  {
    dominio: "licencias",
    triggers: [
      "acompañar conyuge",
      "acompañar cónyuge",
      "conyuge",
      "cónyuge",
      "concubino",
      "conviviente",
      "mision oficial",
      "misión oficial",
    ],
    terms: [
      "acompañar conyuge concubino conviviente mision oficial extranjero cien kilometros",
    ],
    preferredArticles: ["46"],
  },
  {
    dominio: "licencias",
    triggers: [
      "fallecimiento",
      "duelo",
      "fallecio",
      "murio",
      "murió",
      "muerte",
      "sepelio",
    ],
    terms: [
      "fallecimiento duelo conyuge padres hijos hermanos abuelos nietos suegros sepelio",
    ],
    preferredArticles: ["48"],
  },
  {
    dominio: "licencias",
    triggers: [
      "climatica",
      "climaticas",
      "climática",
      "climáticas",
      "climtica",
      "climticas",
      "clim",
      "clima",
      "lluvia",
      "temporal",
      "tormenta",
      "meteorologico",
      "meteorologicos",
      "meteorológica",
      "meteorológicas",
      "caso fortuito",
      "inundacion",
      "inundación",
      "crecida",
      "corte de camino",
      "imposibilidad material",
      "acceso interrumpido",
      "fenomeno meteorologico",
      "fenómeno meteorológico",
      "fuerza mayor",
      "razones climaticas",
      "razones climáticas",
    ],
    terms: [
      "razones extraordinarias fenomenos meteorologicos fuerza mayor caso fortuito inundacion crecida corte de camino imposibilidad material acceso interrumpido inasistencia justificacion articulo 49",
    ],
    preferredArticles: ["49"],
  },
  {
    dominio: "licencias",
    triggers: [
      "donacion de sangre",
      "donación de sangre",
      "donar sangre",
      "sangre",
    ],
    terms: ["donacion de sangre un dia goce haberes comprobante"],
    preferredArticles: ["50"],
  },
  {
    dominio: "licencias",
    triggers: [
      "donacion de organos",
      "donación de órganos",
      "donar organos",
      "órganos",
      "organos",
    ],
    terms: ["donacion de organos junta medica goce integro haberes"],
    preferredArticles: ["51"],
  },
  {
    dominio: "licencias",
    triggers: [
      "razones particulares",
      "razones especiales",
      "particulares",
      "dia particular",
    ],
    terms: [
      "razones especiales particulares veinticuatro horas dos dias mes cinco dias año",
    ],
    preferredArticles: ["52"],
  },
  {
    dominio: "licencias",
    triggers: [
      "superposicion horaria",
      "superposición horaria",
      "superposicin horaria",
      "superposici horaria",
      "superposicion de horarios",
      "superposición de horarios",
      "superposicin de horarios",
      "superposici de horarios",
      "licencia por superposicion",
      "licencia por superposición",
      "licencia por superposicin",
      "licencia por superposici",
      "existe alguna licencia por superposicion",
      "existe alguna licencia por superposición",
      "existe alguna licencia por superposicin",
      "existe alguna licencia por superposici",
      "dos escuelas",
    ],
    terms: [
      "superposicion horarios actos clases tribunales examinadores reuniones orden prelacion",
    ],
    preferredArticles: ["53", "54"],
  },
  {
    dominio: "licencias",
    triggers: ["festividad religiosa", "religiosa", "religiosas", "credo"],
    terms: [
      "festividades religiosas credos no catolicos cinco dias certificado autoridad religiosa",
    ],
    preferredArticles: ["55"],
  },
  {
    dominio: "licencias",
    triggers: [
      "comision de servicios",
      "comisión de servicios",
      "salida didactica",
      "salida didáctica",
      "excursion",
      "excursión",
      "viaje alumnos",
    ],
    terms: [
      "comision servicios salidas didacticas excursiones paseos viajes alumnos mision oficial",
    ],
    preferredArticles: ["56"],
  },
  {
    dominio: "licencias",
    triggers: [
      "horario estudiante",
      "horarios para estudiantes",
      "estudiante",
      "estudiar",
    ],
    terms: [
      "horario para estudiantes establecimiento oficial reposicion horaria reduccion dos horas",
    ],
    preferredArticles: ["58"],
  },
  {
    dominio: "licencias",
    triggers: ["lactancia", "lactante", "amamantar", "madre lactante"],
    terms: [
      "reduccion horaria docentes madres lactantes dos descansos media hora una hora diaria 360 dias",
    ],
    preferredArticles: ["59"],
  },
  {
    dominio: "licencias",
    triggers: [
      "citacion",
      "citación",
      "citaciones",
      "tramite personal",
      "trámite personal",
      "tribunal",
    ],
    terms: [
      "citaciones tramites personales obligatorios tribunales organismos constancia dos dias",
    ],
    preferredArticles: ["60"],
  },
  {
    dominio: "licencias",
    triggers: [
      "delegado gremial",
      "delegados gremiales",
      "delegado escolar",
      "congresal",
    ],
    terms: [
      "delegados gremiales franquicia ocho horas dos dias mensuales congresales",
    ],
    preferredArticles: ["61"],
  },
  {
    dominio: "licencias",
    triggers: [
      "llegue tarde",
      "llego tarde",
      "tardanza",
      "puntualidad",
      "falta de puntualidad",
    ],
    terms: [
      "falta de puntualidad tardanza diez minutos sancion apercibimiento suspension",
    ],
    preferredArticles: ["78", "83"],
  },
  {
    dominio: "licencias",
    triggers: [
      "falto 6 dias",
      "falto 6 días",
      "falte 6 dias",
      "falté 6 días",
      "6 dias sin justificar",
      "6 días sin justificar",
      "seis dias sin justificar",
      "seis días sin justificar",
      "no justifico",
      "no justifique",
      "no justifiqué",
      "sin justificar",
      "inasistencias sin justificar",
      "faltas sin justificar",
      "inasistencias injustificadas",
      "abandono de servicio",
    ],
    terms: [
      "abandono de servicio cinco dias laborales consecutivos diez dias año calendario ausencia injustificada sanciones descuento remuneracion articulo 80 articulo 84 articulo 85 articulo 86",
    ],
    preferredArticles: ["85", "86", "80", "84"],
  },
  {
    dominio: "licencias",
    triggers: [
      "inasistencia injustificada",
      "ausencia injustificada",
      "falta injustificada",
      "descuento",
    ],
    terms: [
      "ausencia injustificada descuento remuneracion sanciones apercibimiento suspension",
    ],
    preferredArticles: ["80", "84"],
  },
  {
    dominio: "licencias",
    triggers: ["abandono de servicio", "abandono", "no fui a trabajar"],
    terms: [
      "abandono de servicio cinco dias laborales consecutivos diez dias año calendario",
    ],
    preferredArticles: ["85", "86"],
  },
  {
    dominio: "licencias",
    triggers: ["libreta medica", "libreta médica"],
    terms: ["libreta medica docente obligatoria reconocimiento medico licencia"],
    preferredArticles: ["70"],
  },
  {
    dominio: "licencias",
    triggers: [
      "solicitud",
      "presentar licencia",
      "plazo",
      "plazos",
      "pedir licencia",
    ],
    terms: [
      "presentacion solicitud licencia plazos veinticuatro horas tres dias laborables",
    ],
    preferredArticles: ["71"],
  },
  {
    dominio: "licencias",
    triggers: [
      "certificado medico",
      "certificado médico",
      "reconocimiento medico",
      "alta medica",
    ],
    terms: [
      "certificacion estado sanitario certificado medico direccion control reconocimiento medicos",
    ],
    preferredArticles: ["72", "73", "74", "75", "76"],
  },

  {
    dominio: "coberturas",
    triggers: [
      "cobertura",
      "coberturas",
      "asamblea",
      "asambleas",
      "quiero saber de las asambleas",
      "sistema de asamblea",
      "asamblea publica",
      "asamblea pública",
    ],
    terms: [
      "sistema asamblea publica coberturas cargos horas catedras interinos suplentes decreto 636",
    ],
    preferredArticles: ["Decreto Artículo 1", "Decreto Artículo 2", "ANEXO I Punto 1", "ANEXO II Punto 1"],
  },
  {
    dominio: "coberturas",
    triggers: [
      "procedimiento de asamblea",
      "procedimiento asamblea",
      "como es el procedimiento de asamblea",
      "cómo es el procedimiento de asamblea",
      "procedimiento para cobertura",
      "procedimiento de cobertura",
      "cobertura de cargos y horas catedra",
      "cobertura de cargos y horas cátedra",
      "asamblea ordinaria procedimiento",
      "solicitud de cobertura",
      "pedido de cobertura",
    ],
    terms: [
      "asamblea ordinaria solicitud cobertura vacantes suplencias publicacion escuelas cabecera formulario unico asamblea cargos horas catedras directivos direcciones nivel",
    ],
    preferredArticles: ["ANEXO II Punto 9", "ANEXO I Punto 11", "ANEXO II Punto 1", "ANEXO I Punto 1"],
  },
  {
    dominio: "coberturas",
    triggers: [
      "cabecera cero",
      "cabecera 0",
      "que es cabecera cero",
      "qué es cabecera cero",
      "cargo desierto",
      "cargos desiertos",
      "asamblea desierta",
      "asambleas desiertas",
    ],
    terms: [
      "cabecera cero cargos quedaron desiertos despues de dos asambleas ordinarias orden titulo docente titulo habilitante titulo supletorio estudiantes avanzados",
    ],
    preferredArticles: ["ANEXO I Punto 12", "ANEXO II Punto 10"],
  },
  {
    dominio: "coberturas",
    triggers: [
      "cabecera cero secundaria",
      "cabecera 0 secundaria",
      "cabecera cero nivel secundario",
      "cabecera cero en secundaria",
      "horas catedra cabecera cero",
      "horas cátedra cabecera cero",
    ],
    terms: [
      "anexo ii punto 10 cabecera cero secundaria cargos horas catedra desiertos dos asambleas ordinarias",
    ],
    preferredArticles: ["ANEXO II Punto 10"],
  },
  {
    dominio: "coberturas",
    triggers: [
      "cabecera cero primaria",
      "cabecera cero inicial",
      "cabecera cero nivel primario",
      "cabecera cero nivel inicial",
    ],
    terms: [
      "anexo i punto 12 cabecera cero inicial primaria cargos desiertos dos asambleas ordinarias",
    ],
    preferredArticles: ["ANEXO I Punto 12"],
  },
  {
    dominio: "coberturas",
    triggers: [
      "documentacion para tomar cargo",
      "documentación para tomar cargo",
      "documentacion necesito",
      "documentación necesito",
      "que documentacion necesito",
      "qué documentación necesito",
      "fua",
      "formulario unico de alta",
      "formulario único de alta",
      "declaracion jurada de cargos",
      "declaración jurada de cargos",
    ],
    terms: [
      "documentacion alta formulario unico alta fua declaracion jurada cargos documento identidad legajo electronico cese licencia",
    ],
    preferredArticles: ["ANEXO I Punto 20", "ANEXO II Punto 15"],
  },
  {
    dominio: "coberturas",
    triggers: [
      "no me presento a destino",
      "no presentarse a destino",
      "no se presento",
      "no se presentó",
      "despues de tomar un cargo",
      "después de tomar un cargo",
      "inhabilitacion",
      "inhabilitación",
      "perder derecho a optar",
    ],
    terms: [
      "no presentarse destino pierde derecho optar resto ciclo lectivo inhabilitacion sancion",
    ],
    preferredArticles: ["ANEXO I Punto 15", "ANEXO II Punto 13"],
  },
  {
    dominio: "coberturas",
    triggers: [
      "como se cubren horas catedra",
      "cómo se cubren horas cátedra",
      "horas catedra secundaria",
      "horas cátedra secundaria",
      "cobertura horas catedra",
      "cobertura horas cátedra",
    ],
    terms: [
      "anexo ii secundaria cobertura cargos horas catedras interino suplente lom titulo docente habilitante",
    ],
    preferredArticles: ["ANEXO II Punto 1", "ANEXO II Punto 8", "ANEXO II Punto 9"],
  },

  {
    dominio: "estatuto",
    triggers: ["estabilidad", "estable", "titularidad"],
    terms: [
      "estabilidad cargo categoria jerarquia ubicacion buena conducta capacidad psicofisica domicilio real",
    ],
    preferredArticles: ["6", "21", "173"],
  },
  {
    dominio: "estatuto",
    triggers: ["derechos", "derecho docente"],
    terms: [
      "derechos docente estabilidad remuneracion ascenso traslado defensa agremiacion licencia",
    ],
    preferredArticles: ["6"],
  },
  {
    dominio: "estatuto",
    triggers: ["deberes", "obligaciones"],
    terms: [
      "deberes personal docente dignidad eficiencia lealtad conducta jurisdiccion tecnica administrativa",
    ],
    preferredArticles: ["5"],
  },
  {
    dominio: "estatuto",
    triggers: ["ingreso", "ingresar", "condiciones", "titulo", "título"],
    terms: [
      "ingreso docencia condiciones titulo docente capacidad fisica moral domicilio real concurso",
    ],
    preferredArticles: ["12", "13", "14", "18"],
  },
  {
    dominio: "estatuto",
    triggers: [
      "junta",
      "clasificacion",
      "clasificación",
      "orden de merito",
      "orden de mérito",
    ],
    terms: [
      "juntas clasificacion orden merito antecedentes aspirantes ingreso interinatos suplencias",
    ],
    preferredArticles: ["9", "10", "11"],
  },
  {
    dominio: "estatuto",
    triggers: ["ascenso", "ascensos", "concurso"],
    terms: [
      "ascensos concurso titulos antecedentes oposicion jerarquia categoria ubicacion",
    ],
    preferredArticles: [
      "26",
      "27",
      "28",
      "29",
      "30",
      "31",
      "32",
      "33",
      "34",
      "35",
    ],
  },
  {
    dominio: "estatuto",
    triggers: ["permuta", "traslado", "traslados"],
    terms: [
      "permutas traslados cambio destino razones salud nucleo familiar disponibilidad vacantes",
    ],
    preferredArticles: ["37", "38", "39", "40", "41", "45"],
  },
  {
    dominio: "estatuto",
    triggers: [
      "remuneracion",
      "remuneración",
      "sueldo",
      "bonificacion",
      "bonificación",
      "antiguedad",
      "antigüedad",
    ],
    terms: [
      "remuneraciones asignacion cargo bonificacion antiguedad ubicacion funcion diferenciada",
    ],
    preferredArticles: ["46", "47", "48", "49", "50", "51", "52"],
  },
  {
    dominio: "estatuto",
    triggers: [
      "disciplina",
      "sancion",
      "sanción",
      "sumario",
      "cesantia",
      "cesantía",
      "exoneracion",
    ],
    terms: [
      "disciplina sanciones sumario amonestacion apercibimiento suspension cesantia exoneracion",
    ],
    preferredArticles: ["61", "62", "63", "64", "65", "66", "67", "68", "69"],
  },
  {
    dominio: "estatuto",
    triggers: ["tribunal de disciplina", "tribunales de disciplina"],
    terms: [
      "tribunal disciplina miembros docentes actividad votacion directa secreta requisitos",
    ],
    preferredArticles: ["70", "71"],
  },
  {
    dominio: "estatuto",
    triggers: [
      "interino",
      "interinos",
      "suplente",
      "suplentes",
      "suplencia",
      "interinato",
    ],
    terms: [
      "interinatos suplencias designacion orden merito aspirantes lista junta clasificacion",
    ],
    preferredArticles: [
      "94",
      "95",
      "96",
      "97",
      "98",
      "99",
      "114",
      "115",
      "116",
      "117",
    ],
  },
];

function getEnv(name: string): string {
  return process.env[name]?.trim() || "";
}

function normalizeText(str: string): string {
  return (str || "")
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/�/g, "")
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function tokenize(text: string): string[] {
  const normalized = normalizeText(text);

  return Array.from(
    new Set(
      normalized
        .split(" ")
        .map((w) => w.trim())
        .filter(Boolean)
        .filter((w) => w.length >= 3)
        .filter((w) => !STOPWORDS.has(w))
    )
  );
}

function tokenLooksClose(queryToken: string, triggerToken: string): boolean {
  if (!queryToken || !triggerToken) return false;

  if (queryToken === triggerToken) return true;

  if (queryToken.length < 6 || triggerToken.length < 6) return false;

  if (Math.abs(queryToken.length - triggerToken.length) > 3) return false;

  return queryToken.startsWith(triggerToken) || triggerToken.startsWith(queryToken);
}

function triggerMatchesQuery(
  trigger: string,
  normalizedPregunta: string,
  preguntaTokens: string[]
): boolean {
  const normalizedTrigger = normalizeText(trigger);

  if (!normalizedTrigger) return false;

  const triggerTokens = tokenize(normalizedTrigger);

  if (!triggerTokens.length) return false;

  if (triggerTokens.length === 1) {
    const triggerToken = triggerTokens[0];

    return preguntaTokens.some((preguntaToken) =>
      tokenLooksClose(preguntaToken, triggerToken)
    );
  }

  if (normalizedPregunta.includes(normalizedTrigger)) {
    return true;
  }

  return triggerTokens.every((triggerToken) =>
    preguntaTokens.some((preguntaToken) =>
      tokenLooksClose(preguntaToken, triggerToken)
    )
  );
}

function getMatchedExpansions(
  pregunta: string,
  dominio: DominioBackend
): QueryExpansion[] {
  const normalizedPregunta = normalizeText(pregunta);
  const preguntaTokens = tokenize(normalizedPregunta);

  return QUERY_EXPANSIONS.filter((item) => {
    if (item.dominio && item.dominio !== dominio) return false;

    return item.triggers.some((trigger) =>
      triggerMatchesQuery(trigger, normalizedPregunta, preguntaTokens)
    );
  });
}


function isGremialRepresentationLicenseQuestion(pregunta: string): boolean {
  const text = normalizeText(pregunta);

  const mentionsGremial =
    text.includes("gremial") ||
    text.includes("representacion gremial") ||
    text.includes("representación gremial") ||
    text.includes("delegado gremial") ||
    text.includes("delegados gremiales");

  if (!mentionsGremial) return false;

  const mentionsLicenseOrArticle =
    text.includes("licencia") ||
    text.includes("licencias") ||
    text.includes("franquicia") ||
    text.includes("franquicias") ||
    text.includes("articulo 40") ||
    text.includes("artículo 40") ||
    text.includes("art 40") ||
    text.includes("articulo 61") ||
    text.includes("artículo 61") ||
    text.includes("art 61") ||
    text.includes("representacion gremial") ||
    text.includes("representación gremial") ||
    text.includes("delegado gremial") ||
    text.includes("delegados gremiales");

  const asksForContactOrAdvice =
    text.includes("asesoramiento") ||
    text.includes("contacto") ||
    text.includes("whatsapp") ||
    text.includes("telefono") ||
    text.includes("teléfono") ||
    text.includes("numero") ||
    text.includes("número") ||
    text.includes("a que numero") ||
    text.includes("a qué número") ||
    text.includes("con quien") ||
    text.includes("con quién") ||
    text.includes("comunicarme");

  return mentionsLicenseOrArticle && !asksForContactOrAdvice;
}

function isGremialAdviceServiceQuestion(pregunta: string): boolean {
  const text = normalizeText(pregunta);

  if (isGremialRepresentationLicenseQuestion(pregunta)) return false;

  const mentionsGremial =
    text.includes("asesoramiento gremial") ||
    text.includes("contacto gremial") ||
    text.includes("consulta gremial") ||
    text.includes("numero gremial") ||
    text.includes("número gremial") ||
    text.includes("area gremial") ||
    text.includes("área gremial") ||
    text.includes("gremial");

  if (!mentionsGremial) return false;

  const serviceIntent =
    text.includes("asesoramiento") ||
    text.includes("contacto") ||
    text.includes("whatsapp") ||
    text.includes("telefono") ||
    text.includes("teléfono") ||
    text.includes("numero") ||
    text.includes("número") ||
    text.includes("a que numero") ||
    text.includes("a qué número") ||
    text.includes("con quien") ||
    text.includes("con quién") ||
    text.includes("comunicarme") ||
    text.includes("sindicato") ||
    text.includes("sidca") ||
    text.includes("necesito");

  return serviceIntent;
}

function isAppNavigationServicesQuestion(pregunta: string): boolean {
  const text = normalizeText(pregunta);

  const navigationIntent =
    text.includes("en que parte de la app") ||
    text.includes("en qué parte de la app") ||
    text.includes("en que parte") ||
    text.includes("en qué parte") ||
    text.includes("donde ingreso") ||
    text.includes("dónde ingreso") ||
    text.includes("donde debo ingresar") ||
    text.includes("dónde debo ingresar") ||
    text.includes("donde entrar") ||
    text.includes("dónde entrar") ||
    text.includes("donde buscar") ||
    text.includes("dónde buscar") ||
    text.includes("buscar en la app") ||
    text.includes("ingresar en la app") ||
    text.includes("entrar en la app") ||
    (text.includes("app") &&
      (text.includes("ingresar") ||
        text.includes("entrar") ||
        text.includes("buscar") ||
        text.includes("donde") ||
        text.includes("dónde") ||
        text.includes("parte") ||
        text.includes("seccion") ||
        text.includes("sección")));

  if (!navigationIntent) return false;

  return (
    text.includes("decreto") ||
    text.includes("decretos") ||
    text.includes("resolucion") ||
    text.includes("resolución") ||
    text.includes("ley") ||
    text.includes("leyes") ||
    text.includes("normativa") ||
    text.includes("legal") ||
    text.includes("titularizacion") ||
    text.includes("titularización") ||
    text.includes("titularizaci") ||
    text.includes("oficina") ||
    text.includes("tramite") ||
    text.includes("trámite") ||
    text.includes("tramites") ||
    text.includes("trámites") ||
    text.includes("documentacion") ||
    text.includes("documentación") ||
    text.includes("formulario") ||
    text.includes("formularios") ||
    text.includes("certificado") ||
    text.includes("certificados") ||
    text.includes("convenio") ||
    text.includes("convenios") ||
    text.includes("capacitacion") ||
    text.includes("capacitación") ||
    text.includes("curso") ||
    text.includes("cursos")
  );
}

function classifyDomain(
  pregunta: string,
  dominio?: DominioBackend
): DominioBackend {
  if (
    dominio === "licencias" ||
    dominio === "estatuto" ||
    dominio === "coberturas" ||
    dominio === "servicios" || (dominio !== undefined && ["nomenclador", "sumarios", "educacion_nacional", "educacion_provincial", "educacion_privada", "educacion_tecnico_profesional"].includes(dominio))
  ) {
    return dominio;
  }

  const text = normalizeText(pregunta);
  const hasExplicitNormReference = parseNormativeReferences(pregunta).length > 0;

  if (/\b533\s*(?:\/\s*)?(?:21|2021)\b/.test(text)) return "coberturas";
  if (/\b636\s*(?:\/\s*)?2021\b/.test(text)) return "coberturas";
  if (/\b(?:2431\s*(?:\/\s*)?2016|2453\s*(?:\/\s*)?2019|1480\s*(?:\/\s*)?2019)\b/.test(text) || (!hasExplicitNormReference && text.includes("nomenclador"))) return "nomenclador";
  if (!hasExplicitNormReference && (text.includes("centro educativo") || text.includes("centros educativos") || text.includes("jardin maternal") || text.includes("jardines maternales") || text.includes("pasitos cuidados"))) return "nomenclador";
  if (/\b26\s*\.?\s*058\b/.test(text)) return "educacion_tecnico_profesional";
  if (/\b26\s*\.?\s*206\b/.test(text)) return "educacion_nacional";
  if (/\b5381\b/.test(text)) return "educacion_provincial";
  if (/\b3122\b/.test(text) || text.includes("estatuto")) return "estatuto";
  if (/\b3387\b/.test(text) || text.includes("educacion privada")) return "educacion_privada";
  if (hasExplicitNormReference) {
    const explicit = parseNormativeReferences(pregunta)[0];
    const normalizedNumber = explicit.number.replace(/\D/g, "");
    if (explicit.kind === "ley" && explicit.qualifier === "nacional") return "educacion_nacional";
    if (explicit.kind === "ley" && explicit.qualifier === "provincial") return "educacion_provincial";
    if (explicit.kind === "decreto-ley" && normalizedNumber === "3387") return "educacion_privada";
    if (explicit.kind === "decreto" && normalizedNumber === "1092") return "licencias";
    if (/\b(?:licencia|licencias|inasistencia|maternidad|tratamiento|franquicia)\b/.test(text)) return "licencias";
    return "general";
  }
  const nomenclatorMetrics = /\b(?:horas? (?:reloj|catedra)|minutos? reloj|carga horaria|punt(?:os?|aje)|incompatibilidad|jornada)\b/.test(text);
  const nomenclatorComparison = /\b(?:comparar|comparacion|compara|compar[aá]|diferencias?|versus|vs|contra)\b/.test(text);
  const teacherRoleAliases = ["director", "directora", "vicedirector", "vicedirectora", "secretario", "secretaria", "prosecretario", "prosecretaria", "maestro", "maestra", "profesor", "profesora", "preceptor", "preceptora", "bibliotecario", "bibliotecaria", "supervisor", "supervisora", "rector", "rectora", "vicerrector", "vicerrectora", "regente", "regenta", "bedel", "tutor", "tutora", "coordinador", "coordinadora", "ayudante", "maestro de sala", "maestro especial", "cargo docente"];
  const teachingRole = teacherRoleAliases.some((alias) => new RegExp(`(?:^|\\s)${alias.replace(/\s+/g, "\\s+")}(?:$|\\s)`).test(text));
  const coverageIntent = /\b(?:cobertura|asamblea|designaci[oó]n|vacante|suplencia|interinato|cabecera|lom|optar|tomar el cargo)\b/.test(text);
  const licenseIntent = /\b(?:licencia|licencias|rendir|examen|curso|capacitacion|tratamiento|asuntos? particulares|razones? particulares|familiar enfermo)\b/.test(text);
  if ((nomenclatorMetrics || nomenclatorComparison) && teachingRole && !coverageIntent && !licenseIntent) return "nomenclador";

  if (isGremialAdviceServiceQuestion(pregunta) || isAppNavigationServicesQuestion(pregunta)) {
    return "servicios";
  }

  if (
    isClimateQuestion(pregunta) ||
    isUnjustifiedAbsenceQuestion(pregunta) ||
    isMaternityQuestion(pregunta) ||
    isSuperpositionQuestion(pregunta)
  ) {
    return "licencias";
  }

  if (isServicesQuestion(pregunta)) {
    return "servicios";
  }

  if (text.includes("estatuto") || text.includes("ley 3122")) {
    return "estatuto";
  }

  if (
    text.includes("cobertura") ||
    text.includes("coberturas") ||
    text.includes("asamblea") ||
    text.includes("asambleas") ||
    text.includes("cabecera") ||
    text.includes("cabecera cero") ||
    text.includes("cabecera 0") ||
    text.includes("vacante") ||
    text.includes("vacantes") ||
    text.includes("lom") ||
    text.includes("fua") ||
    text.includes("formulario unico") ||
    text.includes("cargo desierto") ||
    text.includes("cargos desiertos") ||
    text.includes("horas catedra") ||
    text.includes("hora catedra") ||
    text.includes("interino") ||
    text.includes("interinos") ||
    text.includes("suplente") ||
    text.includes("suplentes")
  ) {
    return "coberturas";
  }

  if (
    text.includes("licencia") ||
    text.includes("licencias") ||
    text.includes("inasistencia") ||
    text.includes("justificacion") ||
    text.includes("razones particulares") ||
    text.includes("razones especiales") ||
    text.includes("franquicia") ||
    text.includes("maternidad") ||
    text.includes("enfermedad") ||
    text.includes("familiar") ||
    text.includes("examen") ||
    text.includes("capacitacion") ||
    text.includes("capacitaci") ||
    text.includes("capacitac") ||
    text.includes("perfeccionamiento") ||
    text.includes("perfeccion") ||
    text.includes("curso") ||
    text.includes("taller") ||
    text.includes("congreso")
  ) {
    return "licencias";
  }

  const rules: Record<DominioBackend, string[]> = {
    nomenclador: ["nomenclador", "centro educativo", "centros educativos", "jardin maternal", "jardines maternales"],
    sumarios: ["sumario", "sumarios", "procedimiento disciplinario", "instructor sumariante"],
    educacion_nacional: ["educacion nacional", "educación nacional", "ley nacional"],
    educacion_provincial: ["educacion provincial", "educación provincial", "estatuto docente provincial"],
    educacion_privada: ["educacion privada", "educación privada", "escuela privada"],
    educacion_tecnico_profesional: ["tecnico profesional", "técnico profesional", "educacion tecnica", "educación técnica"],
    licencias: [
      "licencia",
      "licencias",
      "inasistencia",
      "franquicia",
      "maternidad",
      "embarazo",
      "lactancia",
      "enfermedad",
      "tratamiento",
      "corto",
      "largo",
      "accidente",
      "familiar",
      "duelo",
      "matrimonio",
      "estudio",
      "estudios",
      "examen",
      "examenes",
      "rendir",
      "capacitacion",
      "capacitaci",
      "perfeccionamiento",
      "perfeccion",
      "violencia",
      "genero",
      "adopcion",
      "donacion",
      "sangre",
      "aislamiento",
      "climatica",
      "climaticas",
      "meteorologico",
      "fuerza mayor",
    ],
    estatuto: [
      "estatuto",
      "ley 3122",
      "ingreso",
      "docencia",
      "titular",
      "titularidad",
      "estabilidad",
      "traslado",
      "permuta",
      "ascenso",
      "junta",
      "clasificacion",
      "disciplina",
      "sancion",
      "sumario",
      "remuneracion",
      "deberes",
      "derechos",
    ],
    coberturas: [
      "cobertura",
      "coberturas",
      "asamblea",
      "asambleas",
      "cargo",
      "cargos",
      "horas",
      "catedra",
      "interino",
      "interinato",
      "suplente",
      "suplencia",
      "cabecera",
      "cabecera cero",
      "cabecera 0",
      "lom",
      "listado",
      "vacante",
      "vacantes",
      "fua",
      "opcion",
      "destino",
      "renuncia",
      "secundaria",
      "primaria",
      "inicial",
    ],
    servicios: [
      "servicio",
      "servicios",
      "contacto",
      "whatsapp",
      "telefono",
      "sede",
      "direccion",
      "turismo",
      "viajes",
      "reserva",
      "hoteleria",
      "hotel",
      "hoteles",
      "convenio",
      "convenios",
      "comercios",
      "casa del docente",
      "predio",
      "capacitaciones",
      "cursos disponibles",
      "aula virtual",
      "certificados",
      "entrega de certificados",
      "oficina de gestion",
      "oficina gesti",
      "oficina de gesti",
      "gestion expediente",
      "gesti expediente",
      "soporte tecnico",
      "afiliado adherente",
      "sidca radio",
      "sala de reuniones",
      "mi catamarca",
      "enlaces utiles",
      "simulador de sueldo",
    ],
    general: [
      "afiliacion",
      "afiliado",
      "afiliados",
      "oficina",
      "horario",
      "contacto",
      "telefono",
      "whatsapp",
      "certificado",
      "formulario",
      "beneficio",
      "tramite",
      "sindicato",
      "gestion",
    ],
  };

  let bestDomain: DominioBackend = "general";
  let bestScore = 0;

  for (const domain of Object.keys(rules) as DominioBackend[]) {
    const score = rules[domain].reduce((acc, keyword) => {
      return text.includes(normalizeText(keyword)) ? acc + 1 : acc;
    }, 0);

    if (score > bestScore) {
      bestScore = score;
      bestDomain = domain;
    }
  }

  return bestDomain;
}

function isDominioBackend(value: unknown): value is DominioBackend {
  return ["licencias", "estatuto", "general", "coberturas", "servicios", "nomenclador", "sumarios", "educacion_nacional", "educacion_provincial", "educacion_privada", "educacion_tecnico_profesional"].includes(String(value));
}

function getFilenameFromUrl(url: string, fallback: string): string {
  try {
    const parsed = new URL(url);
    const parts = parsed.pathname.split("/").filter(Boolean);
    return parts[parts.length - 1] || fallback;
  } catch {
    return fallback;
  }
}

function getFallbackSourceConfigs(): SourceConfig[] {
  const licenciasUrl =
    getEnv("SIDCA_DOCS_LICENCIAS_URL") || DEFAULT_LICENCIAS_URL;

  const estatutoUrl =
    getEnv("SIDCA_DOCS_ESTATUTO_URL") || DEFAULT_ESTATUTO_URL;

  const generalUrl = getEnv("SIDCA_DOCS_GENERAL_URL") || DEFAULT_GENERAL_URL;

  const coberturasUrl =
    getEnv("SIDCA_DOCS_COBERTURAS_URL") || DEFAULT_COBERTURAS_URL;

  const serviciosUrl =
    getEnv("SIDCA_DOCS_SERVICIOS_URL") || DEFAULT_SERVICIOS_URL;

  const sources: SourceConfig[] = [
    {
      dominio: "licencias",
      source: licenciasUrl,
      filename: "regimen_licencias_docentes.json",
      referencia: "Decreto Acuerdo Nº 1092/2015 – Régimen de Licencias Docentes",
    },
    {
      dominio: "estatuto",
      source: estatutoUrl,
      filename: "texto_provincial_para_firestore.json",
      referencia: "Ley Nº 3122 – Estatuto del Docente Provincial",
    },
    {
      dominio: "general",
      source: generalUrl,
      filename: "consultas_generales.json",
      referencia: "Consultas generales SiDCa",
    },
    {
      dominio: "coberturas",
      source: coberturasUrl,
      filename: "decreto_636_coberturas_docentes.json",
      referencia:
        "Dcto. Acdo. Nº 636/2021 – Sistema de Asamblea Pública de Coberturas",
    },
    {
      dominio: "servicios",
      source: serviciosUrl,
      filename: "servicios_sidca_app.json",
      referencia: "Información institucional disponible en la App SiDCa",
    },
  ];

  return sources.filter((source) => Boolean(source.source));
}

async function getSourceConfigs(): Promise<SourceConfig[]> {
  const manifestUrl =
    getEnv("SIDCA_DOCS_MANIFEST_URL") || DEFAULT_MANIFEST_URL;

  if (manifestUrl) {
    try {
      const manifest = (await loadJsonFromSource(manifestUrl)) as SourcesManifest;
      cachedInactiveSourceLabels = (manifest.fuentes || []).filter((fuente) => fuente.activo === false)
        .map((fuente) => [fuente.id, fuente.nombre, fuente.referencia].filter(Boolean).join(" "));

      const inferDomainFromSourceId = (sourceId: string): DominioBackend | undefined => {
        if (sourceId.includes("nomenclador")) return "nomenclador";
        if (sourceId.startsWith("ley_26058")) return "educacion_tecnico_profesional";
        if (sourceId.startsWith("ley_26206")) return "educacion_nacional";
        if (sourceId.startsWith("ley_5381")) return "educacion_provincial";
        if (sourceId.includes("educacion_privada")) return "educacion_privada";
        if (sourceId.includes("sumarios")) return "sumarios";
        if (sourceId.includes("licencias")) return "licencias";
        if (sourceId.includes("estatuto")) return "estatuto";
        if (sourceId.includes("consultas_generales")) return "general";
        if (sourceId.includes("coberturas") || sourceId.includes("movilidad")) return "coberturas";
        if (sourceId.includes("servicios")) return "servicios";
        return undefined;
      };

      const manifestSources = (manifest.fuentes || [])
        .filter((fuente) => fuente.activo === true)
        .filter((fuente) => fuente.url && (isDominioBackend(fuente.dominio) || inferDomainFromSourceId(String(fuente.id || ""))))
        .map((fuente) => {
          const source = String(fuente.url);
          const sourceId = String(fuente.id || "");
          const dominio = (sourceId.includes("nomenclador") ? "nomenclador"
            : sourceId.startsWith("ley_26058") ? "educacion_tecnico_profesional"
            : sourceId.startsWith("ley_26206") ? "educacion_nacional"
            : sourceId.startsWith("ley_5381") ? "educacion_provincial"
            : sourceId.includes("educacion_privada") ? "educacion_privada"
            : sourceId.includes("sumarios") ? "sumarios"
            : (isDominioBackend(fuente.dominio) ? fuente.dominio : inferDomainFromSourceId(sourceId))) as DominioBackend;

          return {
            id: fuente.id,
            dominio,
            source,
            filename: getFilenameFromUrl(source, `${fuente.id || dominio}.json`),
            referencia:
              fuente.referencia ||
              fuente.nombre ||
              fuente.id ||
              getFilenameFromUrl(source, `${dominio}.json`),
          };
        });

      if (manifestSources.length > 0) {
        cachedSourceConfigs = manifestSources;
        console.log(
          `[groqWorkflow] Fuentes cargadas desde índice: ${manifestSources.length}`
        );

        return manifestSources;
      }

      console.warn("[groqWorkflow] manifest_has_no_active_sources");
      cachedSourceConfigs = [];
      return [];
    } catch (error: any) {
      console.warn(
        "[groqWorkflow] manifest_load_failed",
        { code: "manifest_load_failed" },
      );
    }
  }

  if (cachedSourceConfigs) {
    console.warn("[groqWorkflow] manifest_stale_configuration_used");
    return cachedSourceConfigs;
  }
  throw new Error("document_manifest_unavailable");
}

async function loadJsonFromSource(source: string): Promise<unknown> {
  if (workflowTestDependencies.fetchJson) return workflowTestDependencies.fetchJson(source);
  if (/^https?:\/\//i.test(source)) {
    const controller = new AbortController();
    const timeoutId = globalThis.setTimeout(() => controller.abort(), 15000);

    try {
      const response = await fetch(source, {
        signal: controller.signal,
        headers: {
          accept: "application/json,text/plain,*/*",
        },
      });

      if (!response.ok) {
        throw new Error(
          `No se pudo cargar ${source}. Estado HTTP ${response.status}`
        );
      }

      const text = await response.text();
      return JSON.parse(text.replace(/^\uFEFF/, ""));
    } finally {
      globalThis.clearTimeout(timeoutId);
    }
  }

  const text = await readFile(source, "utf-8");
  return JSON.parse(text.replace(/^\uFEFF/, ""));
}

function primitiveToText(value: unknown): string {
  if (value === null || value === undefined) return "";

  if (typeof value === "string") return value;

  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }

  return "";
}

function objectToText(value: unknown, depth = 0): string {
  if (depth > 6) return "";

  const primitive = primitiveToText(value);
  if (primitive) return primitive;

  if (Array.isArray(value)) {
    return value
      .map((item) => objectToText(item, depth + 1))
      .filter(Boolean)
      .join("\n");
  }

  if (typeof value === "object" && value !== null) {
    return Object.entries(value as Record<string, unknown>)
      .filter(([key]) => {
        const normalizedKey = normalizeText(key);

        return ![
          "embedding",
          "vector",
          "idvector",
          "createdat",
          "updatedat",
        ].includes(normalizedKey);
      })
      .map(([key, item]) => {
        const text = objectToText(item, depth + 1);
        if (!text) return "";

        const normalizedKey = normalizeText(key);

        if (
          [
            "contenido",
            "texto",
            "respuesta",
            "descripcion",
            "titulo",
            "titulo_articulo",
            "pregunta",
            "articulo",
            "fuente",
            "keywords",
            "otorgante",
            "interviniente",
            "seccion",
            "nivel",
            "tipo",
            "categoria",
            "contenido",
            "respuesta_sugerida",
            "ubicacion_app",
            "preguntas_relacionadas",
            "palabras_clave",
            "whatsapp",
            "url",
            "direccion",
            "aula_virtual",
          ].includes(normalizedKey)
        ) {
          return text;
        }

        return `${key}: ${text}`;
      })
      .filter(Boolean)
      .join("\n");
  }

  return "";
}

function getObjectField(item: unknown, keys: string[]): string | undefined {
  if (typeof item !== "object" || item === null) return undefined;

  const obj = item as Record<string, unknown>;

  for (const key of keys) {
    const value = obj[key];

    if (value !== null && value !== undefined) {
      const text = primitiveToText(value).trim();
      if (text) return text;
    }
  }

  return undefined;
}

function getAsesoramientoGremialContact(
  chunks: LocalChunk[]
): BackendServiceContact | undefined {
  const chunk = chunks.find((item) => {
    if (item.dominio !== "servicios") return false;

    const id = getServiceItemId(item);
    if (id === "sidca_asesoramiento_gremial") return true;

    const title =
      item.titulo ||
      item.descripcion ||
      getObjectField(item.raw, ["categoria", "titulo", "nombre"]) ||
      "";
    return normalizeText(title).includes("asesoramiento gremial");
  });

  if (!chunk || typeof chunk.raw !== "object" || chunk.raw === null) {
    return undefined;
  }

  const raw = chunk.raw as Record<string, unknown>;
  let contact: Record<string, unknown> | undefined;

  const rawContact = raw.contact ?? raw.contacto;

  if (typeof rawContact === "object" && rawContact !== null) {
    contact = rawContact as Record<string, unknown>;
  } else if (Array.isArray(raw.contactos)) {
    const match = raw.contactos.find((item) => {
      if (typeof item !== "object" || item === null) return false;
      const area = getObjectField(item, ["area", "nombre"]) || "";
      return normalizeText(area).includes("asesoramiento");
    });
    if (typeof match === "object" && match !== null) {
      contact = match as Record<string, unknown>;
    }
  }

  const whatsapp = getObjectField(contact, ["whatsapp", "telefono", "celular"]);
  const url = getObjectField(contact, ["url"]);
  const area =
    getObjectField(contact, ["area", "nombre"]) || "Asesoramiento Gremial";

  if (!whatsapp && !url) return undefined;

  return { area, whatsapp, url };
}

function extractArticleRefs(value?: string): string[] {
  if (!value) return [];

  const matches = normalizeText(value).match(/\d+/g) || [];

  return Array.from(new Set(matches));
}

function extractArticle(text: string, item: unknown): string | undefined {
  const direct = getObjectField(item, ["articulo", "artículo", "article"]);

  if (direct) return direct;

  const match = text.match(/art(?:i|í)culo\s+(\d+)/i);
  if (match?.[1]) return match[1];

  const bracketMatch = text.match(/\[art(?:i|í)culo\s+(\d+)/i);
  if (bracketMatch?.[1]) return bracketMatch[1];

  return undefined;
}

function extractTitle(item: unknown): string | undefined {
  return getObjectField(item, [
    "titulo_articulo",
    "título_articulo",
    "titulo",
    "título",
    "descripcion",
    "descripción",
    "pregunta",
    "tema",
    "nombre",
  ]);
}

function isResumenContainer(item: unknown): boolean {
  if (typeof item !== "object" || item === null) return false;

  const titulo = normalizeText(getObjectField(item, ["titulo", "título"]) || "");
  const tipo = normalizeText(getObjectField(item, ["tipo"]) || "");
  const contenido = (item as Record<string, unknown>).contenido;

  return (
    Array.isArray(contenido) &&
    (titulo.includes("resumen") || tipo.includes("tabla"))
  );
}

function cleanArticleTextByArticle(
  articulo: string | undefined,
  text: string
): string {
  let clean = text;
  const article = String(articulo || "").trim();

  if (article === "35") {
    clean = clean.replace(/________________________________________[\s\S]*$/i, "").trim();
  }

  if (article === "36") {
    clean = clean
      .replace(/________________________________________\s*PARA REALIZAR ESTUDIOS[\s\S]*$/i, "")
      .trim();
  }

  if (article === "41") {
    clean = clean
      .replace(/LICENCIAS EXTRAORDINARIAS SIN GOCE DE HABERES[\s\S]*$/i, "")
      .trim();
  }

  if (article === "49") {
    clean = clean
      .replace(/________________________________________\s*CAPITULO VI\s*-\s*JUSTIFICACI[ÓO]N DE INASISTENCIAS[\s\S]*$/i, "")
      .trim();
  }

  if (article === "52") {
    clean = clean
      .replace(/________________________________________\s*SUPERPOSICI[ÓO]N DE HORARIOS[\s\S]*$/i, "")
      .trim();
  }

  if (article === "60") {
    clean = clean
      .replace(/________________________________________\s*DELEGADOS GREMIALES[\s\S]*$/i, "")
      .trim();
  }

  return clean;
}

function makeArticleChunk(
  item: unknown,
  source: SourceConfig,
  index: number,
  prefix = ""
): LocalChunk | null {
  const rawText = objectToText(item).trim();

  if (!rawText) return null;

  const titulo = extractTitle(item);
  const articulo = extractArticle(rawText, item);
  const text = cleanArticleTextByArticle(articulo, rawText);
  const articleRefs = extractArticleRefs(articulo);

  return {
    id: `${source.dominio}-${source.filename}-${index}-${articulo || "sin-articulo"}`,
    sourceId: source.id,
    kind: articulo ? "articulo" : "documento",
    dominio: source.dominio,
    filename: source.filename,
    source: source.source,
    referencia: source.referencia,
    articulo,
    articleRefs,
    titulo: prefix && titulo ? `${prefix} - ${titulo}` : titulo || prefix || undefined,
    descripcion: getObjectField(item, ["descripcion", "descripción"]),
    otorgante: getObjectField(item, ["otorgante"]),
    interviniente: getObjectField(item, ["interviniente"]),
    seccion: getObjectField(item, ["seccion", "sección"]),
    nivel: getObjectField(item, ["nivel"]),
    tipo: getObjectField(item, ["tipo"]),
    norma: getObjectField(item, ["norma", "numero_norma", "número_norma", "documento", "ley", "decreto", "resolucion", "resolución"]) || source.referencia,
    metadata: typeof item === "object" && item !== null ? item as Record<string, unknown> : undefined,
    text,
    raw: item,
  };
}

function makeResumenChunk(
  row: unknown,
  source: SourceConfig,
  index: number,
  prefix = "Resumen de Licencia"
): LocalChunk | null {
  if (typeof row !== "object" || row === null) return null;

  const articulo = getObjectField(row, ["articulo", "artículo"]);
  const descripcion = getObjectField(row, ["descripcion", "descripción"]);
  const otorgante = getObjectField(row, ["otorgante"]);
  const interviniente = getObjectField(row, ["interviniente"]);
  const seccion = getObjectField(row, ["seccion", "sección"]);
  const nivel = getObjectField(row, ["nivel"]);
  const tipo = getObjectField(row, ["tipo"]);

  const text = [
    prefix,
    articulo ? `Artículo/s: ${articulo}` : "",
    descripcion ? `Descripción: ${descripcion}` : "",
    seccion ? `Sección: ${seccion}` : "",
    nivel ? `Nivel: ${nivel}` : "",
    tipo ? `Tipo: ${tipo}` : "",
    otorgante ? `Funcionario otorgante: ${otorgante}` : "",
    interviniente ? `Funcionario interviniente: ${interviniente}` : "",
  ]
    .filter(Boolean)
    .join("\n");

  if (!text.trim()) return null;

  return {
    id: `${source.dominio}-${source.filename}-resumen-${index}-${articulo || "sin-articulo"}`,
    kind: "resumen",
    dominio: source.dominio,
    filename: source.filename,
    source: source.source,
    referencia: source.referencia,
    articulo,
    articleRefs: extractArticleRefs(articulo),
    titulo: prefix,
    descripcion,
    otorgante,
    interviniente,
    seccion,
    nivel,
    tipo,
    text,
    raw: row,
  };
}

function extractChunksFromJson(data: unknown, source: SourceConfig): LocalChunk[] {
  if (typeof data === "object" && data !== null && !Array.isArray(data)) {
    const envelope = data as Record<string, unknown>;
    const nested = Array.isArray(envelope.registros)
      ? envelope.registros
      : Array.isArray(envelope.servicios)
      ? envelope.servicios
      : null;
    if (nested) {
      const derivedDomain: DominioBackend = source.id?.includes("nomenclador")
        ? "nomenclador"
        : source.id?.startsWith("ley_26058") ? "educacion_tecnico_profesional"
        : source.id?.startsWith("ley_26206") ? "educacion_nacional"
        : source.id?.startsWith("ley_5381") ? "educacion_provincial"
        : source.id?.includes("educacion_privada") ? "educacion_privada"
        : source.id?.includes("sumarios") ? "sumarios"
        : source.dominio;
      const derivedSource = { ...source, dominio: derivedDomain, referencia: String(envelope.fuente && typeof envelope.fuente === "object" ? (envelope.fuente as any).norma || source.referencia : source.referencia) };
      const chunks = nested.map((item, index) => makeArticleChunk(item, derivedSource, index)).filter((chunk): chunk is LocalChunk => Boolean(chunk));
      const metadata = ["estado_vigencia", "actualizacion", "instruccion_chatbot", "instrucciones_chatbot", "observaciones", "notas", "tipo", "control_documental", "articulos_faltantes", "articulos_repetidos", "url_texto_actualizado"]
        .map((key) => `${key}: ${objectToText(envelope[key])}`).filter((value) => !value.endsWith(": ")).join("\n");
      if (metadata) for (const chunk of chunks) chunk.metadata = { ...chunk.metadata, ...(envelope as Record<string, unknown>) };
      return chunks;
    }
  }
  if (Array.isArray(data)) {
    const chunks: LocalChunk[] = [];
    const controls = data.filter((item: any) => item && typeof item === "object" && normalizeText(String(item.tipo || "")) === "control documental");
    const normativaRows = data.filter((item: any) => !(item && typeof item === "object" && normalizeText(String(item.tipo || "")) === "control documental"));

    normativaRows.forEach((originalItem: any, index) => {
      const matchingControl = controls.find((control: any) => {
        const controlDocument = normalizeText(String(control.documento || ""));
        const rowDocument = normalizeText(String(originalItem?.documento || originalItem?.documento_normativo || ""));
        return controlDocument && rowDocument && (controlDocument === rowDocument || rowDocument.includes(controlDocument));
      });
      const item = matchingControl ? { ...originalItem, control_documental: matchingControl } : originalItem;
      if (isResumenContainer(item)) {
        const obj = item as Record<string, unknown>;
        const titulo =
          getObjectField(item, ["titulo", "título"]) || "Resumen de Licencia";
        const contenido = obj.contenido;

        if (Array.isArray(contenido)) {
          contenido.forEach((row, rowIndex) => {
            const chunk = makeResumenChunk(row, source, rowIndex, titulo);
            if (chunk) chunks.push(chunk);
          });
        }

        return;
      }

      const chunk = makeArticleChunk(item, source, index);
      if (chunk) chunks.push(chunk);
    });

    return chunks;
  }

  if (typeof data === "object" && data !== null) {
    if (isResumenContainer(data)) {
      const obj = data as Record<string, unknown>;
      const titulo =
        getObjectField(data, ["titulo", "título"]) || "Resumen de Licencia";
      const contenido = obj.contenido;

      if (Array.isArray(contenido)) {
        return contenido
          .map((row, index) => makeResumenChunk(row, source, index, titulo))
          .filter((chunk): chunk is LocalChunk => Boolean(chunk));
      }
    }

    const obj = data as Record<string, unknown>;
    const chunks: LocalChunk[] = [];

    for (const [key, value] of Object.entries(obj)) {
      if (Array.isArray(value)) {
        value.forEach((item, index) => {
          if (isResumenContainer(item)) {
            const inner = item as Record<string, unknown>;
            const titulo =
              getObjectField(item, ["titulo", "título"]) ||
              key ||
              "Resumen de Licencia";

            if (Array.isArray(inner.contenido)) {
              inner.contenido.forEach((row, rowIndex) => {
                const chunk = makeResumenChunk(row, source, rowIndex, titulo);
                if (chunk) chunks.push(chunk);
              });
            }

            return;
          }

          const chunk = makeArticleChunk(item, source, chunks.length + index, key);
          if (chunk) chunks.push(chunk);
        });
      }
    }

    if (chunks.length) return chunks;

    const singleChunk = makeArticleChunk(data, source, 0);
    return singleChunk ? [singleChunk] : [];
  }

  const singleChunk = makeArticleChunk(data, source, 0);
  return singleChunk ? [singleChunk] : [];
}

async function mapConcurrenciaLimitada<T, R>(
  items: T[],
  concurrency: number,
  worker: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  const results = new Array<R>(items.length);
  let nextIndex = 0;

  const runWorker = async () => {
    while (true) {
      const index = nextIndex++;
      if (index >= items.length) return;
      results[index] = await worker(items[index], index);
    }
  };

  await Promise.all(
    Array.from(
      { length: Math.min(Math.max(concurrency, 1), items.length || 1) },
      () => runWorker(),
    ),
  );

  return results;
}

async function loadLocalChunks(): Promise<LocalChunk[]> {
  const ttl = Number(getEnv("LOCAL_RAG_CACHE_TTL_MS") || 5 * 60 * 1000);
  const now = workflowNow();

  if (cachedChunks && now - cachedAt < ttl) {
    console.log("[groqWorkflow] documents_cache_hit", {
      fragmentos: cachedChunks.length,
    });
    return cachedChunks;
  }

  if (loadingChunksPromise) {
    console.log("[groqWorkflow] documents_load_shared");
    return loadingChunksPromise;
  }

  loadingChunksPromise = (async () => {
    const startedAt = Date.now();
    const sources = await getSourceConfigs();
    console.log("[groqWorkflow] documents_load_start", {
      fuentes: sources.length,
    });

    const resultados = await mapConcurrenciaLimitada(
      sources,
      SOURCE_LOAD_CONCURRENCY,
      async (source) => {
        try {
          const json = await loadJsonFromSource(source.source);
          return { ok: true, chunks: extractChunksFromJson(json, source) };
        } catch {
          console.warn(
            `[groqWorkflow] source_load_failed ${source.filename}`,
            { code: "source_load_failed" },
          );
          return { ok: false, chunks: [] as LocalChunk[] };
        }
      },
    );

    const chunks = resultados.flatMap((resultado) => resultado.chunks);
    const cargaCompleta = resultados.length === sources.length &&
      resultados.every((resultado) => resultado.ok);

    if (cargaCompleta && chunks.length > 0) {
      const nextVersion = createHash("sha256").update(JSON.stringify({ sources, chunks })).digest("hex");
      const corpusChanged = cachedChunksVersion !== nextVersion;
      cachedChunks = chunks;
      cachedChunksComplete = chunks;
      cachedAt = workflowNow();
      cachedChunksVersion = nextVersion;
      if (corpusChanged) responseCache.clear();
    } else if (cachedChunksComplete) {
      const activeSourceUrls = new Set(sources.map((source) => source.source));
      const safeStaleChunks = cachedChunksComplete.filter((chunk) => activeSourceUrls.has(chunk.source));
      const nextVersion = createHash("sha256").update(JSON.stringify({ sources, chunks: safeStaleChunks })).digest("hex");
      const corpusChanged = cachedChunksVersion !== nextVersion;
      cachedChunks = safeStaleChunks;
      cachedChunksVersion = nextVersion;
      if (corpusChanged) responseCache.clear();
      console.warn("[groqWorkflow] documents_stale_cache_used", {
        fuentesFallidas: resultados.filter((resultado) => !resultado.ok).length,
        fragmentos: safeStaleChunks.length,
      });
      if (!safeStaleChunks.length) throw new Error("no_safe_documents_after_partial_failure");
      return safeStaleChunks;
    } else {
      throw new Error("initial_documents_unavailable");
    }

    console.log("[groqWorkflow] documents_loaded", {
      fuentes: sources.length,
      fragmentos: chunks.length,
      cargaCompleta,
      duracionMs: Date.now() - startedAt,
    });

    return chunks;
  })();

  try {
    return await loadingChunksPromise;
  } finally {
    loadingChunksPromise = null;
  }
}

function expandQuestion(pregunta: string, dominio: DominioBackend): string {
  const expansions = getMatchedExpansions(pregunta, dominio);
  const expandedTerms = expansions.flatMap((item) => item.terms).join(" ");

  const domainBase =
    dominio === "licencias"
      ? "licencia justificacion franquicia docente regimen articulo"
      : dominio === "estatuto"
      ? "estatuto docente ley 3122 derechos deberes carrera docente articulo"
      : dominio === "coberturas"
      ? "asamblea publica cobertura cargos horas catedra interinato suplencia cabecera cero lom fua decreto 636"
      : dominio === "servicios"
      ? "servicios app sidca contacto whatsapp beneficios turismo viajes casa del docente convenios capacitaciones certificados oficina gestion asesoramiento"
      : "consulta general sindicato afiliado tramite formulario beneficio";

  return `${pregunta} ${domainBase} ${expandedTerms}`.trim();
}

function extractArticleNumbersFromQuery(pregunta: string): string[] {
  const normalized = normalizeText(pregunta);
  const results = new Set<string>();
  const references = /\b(?:articulos?|arts?)\.?\s*(\d+(?:\s*(?:,|y|e)\s*\d+)*)/g;
  let match: RegExpExecArray | null;
  while ((match = references.exec(normalized)) !== null) {
    for (const number of match[1].match(/\d+/g) || []) results.add(number);
  }

  return Array.from(results);
}

function articleMatches(chunk: LocalChunk, articleNumber: string): boolean {
  if (!articleNumber) return false;

  const article = normalizeText(chunk.articulo || "");
  const target = normalizeText(articleNumber);
  const refs = chunk.articleRefs.map((ref) => normalizeText(ref));

  return article === target || refs.includes(target);
}

function isGenericLicenseArticle(chunk: LocalChunk): boolean {
  if (chunk.dominio !== "licencias") return false;

  return [
    "1",
    "2",
    "3",
    "4",
    "11",
    "15",
    "34",
    "47",
    "57",
    "62",
    "63",
    "64",
    "65",
  ].includes(String(chunk.articulo || ""));
}

function isGenericStatuteArticle(chunk: LocalChunk): boolean {
  if (chunk.dominio !== "estatuto") return false;

  return ["1", "2", "3", "4"].includes(String(chunk.articulo || ""));
}

function queryIsGeneral(pregunta: string): boolean {
  const text = normalizeText(pregunta);

  return (
    text.includes("alcance") ||
    text.includes("regimen") ||
    text.includes("general") ||
    text.includes("titular") ||
    text.includes("interino") ||
    text.includes("suplente") ||
    text.includes("derecho a las licencias") ||
    text.includes("quienes tienen derecho")
  );
}

function isClimateQuestion(pregunta: string): boolean {
  const text = normalizeText(pregunta);
  const compact = text.replace(/\s+/g, "");

  return (
    text.includes("razones climaticas") ||
    text.includes("razon climatica") ||
    text.includes("fenomeno meteorologico") ||
    text.includes("fenomenos meteorologicos") ||
    text.includes("meteorolog") ||
    text.includes("fuerza mayor") ||
    text.includes("caso fortuito") ||
    text.includes("inundacion") ||
    text.includes("crecida") ||
    text.includes("corte de camino") ||
    text.includes("imposibilidad material") ||
    text.includes("acceso interrumpido") ||
    text.includes("lluvia") ||
    text.includes("temporal") ||
    text.includes("tormenta") ||
    compact.includes("climatic") ||
    compact.includes("climtic") ||
    compact.includes("clim")
  );
}

function isIllnessDuringWorkQuestion(pregunta: string): boolean {
  const text = normalizeText(pregunta);
  const mentionsIllness =
    text.includes("enferme") ||
    text.includes("enfermo") ||
    text.includes("enfermedad") ||
    text.includes("fiebre") ||
    text.includes("gripe") ||
    text.includes("resfrio");
  const mentionsWorkContext =
    text.includes("trabaj") ||
    text.includes("escuela") ||
    text.includes("hora de labor") ||
    text.includes("durante la jornada");
  const mentionsLeaving =
    text.includes("retir") ||
    text.includes("me fui") ||
    text.includes("tuve que salir") ||
    text.includes("salir del trabajo");

  return mentionsIllness && mentionsWorkContext && mentionsLeaving;
}

function isUnjustifiedAbsenceQuestion(pregunta: string): boolean {
  const text = normalizeText(pregunta);

  return (
    text.includes("abandono de servicio") ||
    text.includes("sin justificar") ||
    text.includes("no justific") ||
    text.includes("inasistencia injustificada") ||
    text.includes("inasistencias injustificadas") ||
    text.includes("ausencia injustificada") ||
    text.includes("ausencias injustificadas") ||
    text.includes("falta injustificada") ||
    text.includes("faltas injustificadas") ||
    text.includes("falto 6") ||
    text.includes("falte 6") ||
    text.includes("falt 6") ||
    text.includes("seis dias") ||
    text.includes("seis dias laborales") ||
    text.includes("6 dias") ||
    text.includes("6 dias laborales")
  );
}

function isMaternityQuestion(pregunta: string): boolean {
  const text = normalizeText(pregunta);
  const compact = text.replace(/\s+/g, "");

  const isFamilyCareContext =
    (text.includes("madre") ||
      text.includes("padre") ||
      text.includes("hijo") ||
      text.includes("familiar") ||
      text.includes("grupo familiar")) &&
    (text.includes("enferm") ||
      text.includes("accident") ||
      text.includes("cuid") ||
      text.includes("atencion") ||
      text.includes("asistencia"));

  if (isFamilyCareContext) return false;

  return (
    text.includes("maternidad") ||
    text.includes("embarazo") ||
    text.includes("embarazada") ||
    text.includes("parto") ||
    text.includes("prenatal") ||
    text.includes("postnatal") ||
    text.includes("prematuro") ||
    text.includes("soy mama") ||
    text.includes("soy mam") ||
    text.includes("mama") ||
    text.includes("mam") ||
    text.includes("madre") ||
    text.includes("voy a ser mama") ||
    text.includes("voy a ser mam") ||
    text.includes("licencia para madre") ||
    text.includes("licencia por ser mama") ||
    compact.includes("soymam") ||
    compact.includes("licenciaparamam")
  );
}

function isFamilyCareQuestion(pregunta: string): boolean {
  const text = normalizeText(pregunta);
  const mentionsFamily =
    text.includes("grupo familiar") ||
    text.includes("familiar enfermo") ||
    text.includes("familiar accidentado") ||
    text.includes("atencion familiar") ||
    text.includes("cuidar a mi madre") ||
    text.includes("cuidar mi madre") ||
    text.includes("cuidar a mi padre") ||
    text.includes("cuidar mi padre") ||
    text.includes("cuidar a mi hijo") ||
    text.includes("cuidar mi hijo");

  if (mentionsFamily) return true;

  const mentionsRelative =
    text.includes("madre") || text.includes("padre") || text.includes("hijo");
  const mentionsCare =
    text.includes("enferm") ||
    text.includes("accident") ||
    text.includes("cuid") ||
    text.includes("atencion") ||
    text.includes("asistencia");

  return mentionsRelative && mentionsCare;
}

function isBereavementQuestion(pregunta: string): boolean {
  const text = normalizeText(pregunta);

  return (
    text.includes("fallec") ||
    text.includes("murio") ||
    text.includes("muerte") ||
    text.includes("duelo") ||
    text.includes("sepelio")
  );
}

function isLongTreatmentQuestion(pregunta: string): boolean {
  const text = normalizeText(pregunta);

  return (
    text.includes("largo tratamiento") ||
    text.includes("tratamiento prolongado") ||
    text.includes("enfermedad prolongada") ||
    text.includes("afeccion prolongada") ||
    text.includes("reposo prolongado") ||
    text.includes("varios meses") ||
    (text.includes("enfermedad") && text.includes("meses"))
  );
}

function hasShortTreatmentHealthSignal(pregunta: string): boolean {
  const text = normalizeText(pregunta);

  return (
    text.includes("gripe") ||
    text.includes("fiebre") ||
    text.includes("resfrio") ||
    text.includes("enfermedad") ||
    text.includes("enfermo") ||
    text.includes("enferma") ||
    text.includes("certificado medico") ||
    text.includes("reposo medico") ||
    text.includes("afeccion") ||
    text.includes("lesion") ||
    text.includes("medico")
  );
}

function isShortTreatmentQuestion(pregunta: string): boolean {
  const text = normalizeText(pregunta);
  const hasDuration =
    text.includes("pocos dias") ||
    text.includes("un dia") ||
    text.includes("dos dias") ||
    text.includes("tres dias") ||
    text.includes("1 dia") ||
    text.includes("2 dias") ||
    text.includes("3 dias");

  if (!hasShortTreatmentHealthSignal(pregunta)) return false;
  if (isLongTreatmentQuestion(pregunta)) return false;

  return Boolean(
    hasDuration ||
      text.includes("corto tratamiento") ||
      text.includes("afeccion comun") ||
      hasShortTreatmentHealthSignal(pregunta)
  );
}

function isExamLicenseQuestion(pregunta: string): boolean {
  const text = normalizeText(pregunta);

  return (
    text.includes("examen") ||
    text.includes("rendir") ||
    text.includes("mesa examinadora")
  );
}

function isBloodDonationQuestion(pregunta: string): boolean {
  const text = normalizeText(pregunta);

  return (
    text.includes("donacion de sangre") ||
    text.includes("donar sangre") ||
    text.includes("done sangre") ||
    text.includes("doné sangre") ||
    text.includes("donante de sangre")
  );
}

function isTrainingLicenseQuestion(pregunta: string): boolean {
  const text = normalizeText(pregunta);

  if (isExamLicenseQuestion(pregunta) || isBloodDonationQuestion(pregunta)) {
    return false;
  }

  return (
    text.includes("capacitacion") ||
    text.includes("perfeccionamiento") ||
    text.includes("congreso") ||
    text.includes("curso") ||
    text.includes("taller") ||
    text.includes("conferencia") ||
    text.includes("simposio") ||
    text.includes("actualizacion profesional")
  );
}

function getSpecificLicenseArticle(pregunta: string): string | null {
  const requestedArticles = extractArticleNumbersFromQuery(pregunta);
  if (requestedArticles.length > 1) return null;
  const text = normalizeText(pregunta);
  if (/\brazones? particulares\b/.test(text) && /\basuntos? particulares\b/.test(text)) return null;
  const explicitArticle = pregunta.match(/\b(?:art(?:i|í)culo|art)\.?\s*(\d+)\b/i)?.[1];
  if (explicitArticle) return explicitArticle;
  if (/\brazones? particulares\b/.test(text)) return "52";
  if (/\basuntos? particulares\b/.test(text)) return "45";
  if (isLongTreatmentQuestion(pregunta)) return "20";
  if (isShortTreatmentQuestion(pregunta)) return "16";
  if (isFamilyCareQuestion(pregunta)) return "30";
  if (isExamLicenseQuestion(pregunta)) return "35";
  if (isTrainingLicenseQuestion(pregunta)) return "41";
  if (isGremialRepresentationLicenseQuestion(pregunta)) return "40";
  if (isClimateQuestion(pregunta)) return "49";
  if (/\b(?:festividad(?:es)? religiosa(?:s)?|credo no catolico|religiones? no catolicas?)\b/.test(text)) return "55";
  return null;
}

function isAmbiguousParticularLicenseQuestion(pregunta: string): boolean {
  const text = normalizeText(pregunta);
  if (!/\bparticular(?:es)?\b/.test(text)) return false;
  if (/\brazones? particulares\b/.test(text) && /\basuntos? particulares\b/.test(text)) return false;
  if (/\brazones? particulares\b|\basuntos? particulares\b/.test(text)) return false;
  const article = pregunta.match(/\b(?:art(?:i|í)culo|art)\.?\s*(\d+)\b/i)?.[1];
  return article !== "45" && article !== "52";
}

function isSuperpositionQuestion(pregunta: string): boolean {
  const text = normalizeText(pregunta);
  const compact = text.replace(/\s+/g, "");

  return (
    text.includes("superposicion") ||
    text.includes("superposicin") ||
    text.includes("superposici") ||
    text.includes("superpuesto") ||
    text.includes("superpuesta") ||
    compact.includes("superposicion") ||
    compact.includes("superposicin") ||
    compact.includes("superposici")
  );
}

function isGenericLicenseQuestion(pregunta: string): boolean {
  const text = normalizeText(pregunta);
  const tokens = tokenize(text);

  const genericPatterns = [
    "quiero saber sobre licencia",
    "quiero saber de licencia",
    "quiero saber sobre tema de licencia",
    "tema de licencia",
    "sobre tema de licencia",
    "licencia de la",
    "licencia del",
    "licencia para",
    "sobre licencia",
  ];

  const hasGenericPattern = genericPatterns.some((pattern) =>
    text.includes(pattern)
  );

  const hasOnlyGenericTokens =
    tokens.includes("licencia") &&
    tokens.length <= 7 &&
    !tokens.some((token) =>
      [
        "maternidad",
        "embarazo",
        "embarazada",
        "mama",
        "mam",
        "madre",
        "enfermedad",
        "corto",
        "largo",
        "tratamiento",
        "capacitacion",
        "climatica",
        "climaticas",
        "clima",
        "superposicion",
        "superposicin",
        "superposici",
        "familiar",
        "duelo",
        "fallecimiento",
        "examen",
        "particulares",
        "maternidad",
        "lactancia",
      ].includes(token)
    );

  const hasSpecificKeyword =
    isMaternityQuestion(pregunta) ||
    isClimateQuestion(pregunta) ||
    isUnjustifiedAbsenceQuestion(pregunta) ||
    text.includes("enfermedad") ||
    text.includes("corto tratamiento") ||
    text.includes("largo tratamiento") ||
    text.includes("capacitacion") ||
    text.includes("capacitaci") ||
    text.includes("perfeccionamiento") ||
    isSuperpositionQuestion(pregunta) ||
    text.includes("familiar") ||
    text.includes("duelo") ||
    text.includes("fallecimiento") ||
    text.includes("examen") ||
    text.includes("razones particulares") ||
    text.includes("grupo familiar") ||
    text.includes("lactancia") ||
    text.includes("matrimonio");

  return (hasGenericPattern || hasOnlyGenericTokens) && !hasSpecificKeyword;
}


function isGenericTeacherQuestion(pregunta: string): boolean {
  const text = normalizeText(pregunta);
  const tokens = tokenize(text);
  const compact = text.replace(/\s+/g, "");

  const genericPatterns = [
    "docente",
    "docentes",
    "cargo docente",
    "cargos docentes",
    "quiero saber sobre docente",
    "quiero saber de docente",
    "tema docente",
    "sobre docente",
  ];

  const hasGenericPattern = genericPatterns.some((pattern) => text === pattern);

  const hasOnlyGenericTokens =
    tokens.length > 0 &&
    tokens.length <= 3 &&
    tokens.every((token) =>
      [
        "docente",
        "docentes",
        "cargo",
        "cargos",
        "profesor",
        "profesora",
        "maestro",
        "maestra",
      ].includes(token)
    );

  const hasSpecificKeyword =
    text.includes("interino") ||
    text.includes("interinos") ||
    text.includes("suplente") ||
    text.includes("suplentes") ||
    text.includes("titular") ||
    text.includes("titulares") ||
    text.includes("licencia") ||
    text.includes("licencias") ||
    text.includes("estatuto") ||
    text.includes("cobertura") ||
    text.includes("asamblea") ||
    text.includes("cabecera") ||
    text.includes("funcion") ||
    text.includes("funciones") ||
    text.includes("director") ||
    text.includes("secretario") ||
    compact.includes("maestrogrado");

  return (hasGenericPattern || hasOnlyGenericTokens) && !hasSpecificKeyword;
}

function isStandaloneGreeting(pregunta: string): boolean {
  return /^(hola|buen dia|buenos dias|buenas tardes|buenas noches|saludos|hola sidca)$/.test(normalizeText(pregunta));
}

function isIncompleteTeacherComparisonQuestion(pregunta: string): boolean {
  const text = normalizeText(pregunta);

  const asksDifference =
    text.includes("diferencia") ||
    text.includes("diferencias") ||
    text.includes("distingue") ||
    text.includes("comparar");

  if (!asksDifference) return false;

  const mentionsTeacher =
    text.includes("docente") ||
    text.includes("maestro") ||
    text.includes("profesor");

  const mentionsOneStatus =
    text.includes("suplente") ||
    text.includes("interino") ||
    text.includes("titular");

  const statusCount = ["suplente", "interino", "titular"].reduce(
    (acc, status) => acc + (text.includes(status) ? 1 : 0),
    0
  );

  const endsWithGenericTeacher =
    text.endsWith("y un docente") ||
    text.endsWith("y una docente") ||
    text.endsWith("y docente") ||
    text.endsWith("con un docente") ||
    text.endsWith("con una docente");

  return mentionsTeacher && mentionsOneStatus && (statusCount < 2 || endsWithGenericTeacher);
}

function isMedicalLicenseAfterAppointmentQuestion(pregunta: string): boolean {
  const text = normalizeText(pregunta);

  const compact = text.replace(/\s+/g, "");

  const mentionsMedicalLicense =
    text.includes("licencia medica") ||
    text.includes("licencia mdica") ||
    text.includes("licencia por enfermedad") ||
    text.includes("licencia de salud") ||
    text.includes("certificado medico") ||
    text.includes("certificado mdico") ||
    text.includes("carpeta medica") ||
    text.includes("carpeta mdica") ||
    (text.includes("licencia") &&
      (text.includes("medica") ||
        text.includes("mdica") ||
        text.includes("mdico") ||
        text.includes("medic") ||
        text.includes("mdic") ||
        compact.includes("licenciamdica") ||
        compact.includes("licenciamedica") ||
        text.includes("salud") ||
        text.includes("enfermedad") ||
        text.includes("enfermo") ||
        text.includes("enferma")));

  const mentionsAppointmentOrPosition =
    text.includes("optado") ||
    text.includes("opte") ||
    text.includes("optar") ||
    text.includes("opcion") ||
    text.includes("tomar cargo") ||
    text.includes("tome cargo") ||
    text.includes("tomado un cargo") ||
    text.includes("haber tomado") ||
    text.includes("designado") ||
    text.includes("designacion") ||
    text.includes("alta") ||
    text.includes("cargo docente") ||
    text.includes("cargo");

  const asksMinimumTime =
    text.includes("a partir") ||
    text.includes("cuantos dias") ||
    text.includes("cuanto dias") ||
    text.includes("desde cuando") ||
    text.includes("desde que dia") ||
    text.includes("despues de") ||
    text.includes("luego de") ||
    text.includes("plazo") ||
    text.includes("antiguedad");

  return mentionsMedicalLicense && mentionsAppointmentOrPosition && asksMinimumTime;
}


type ServiceExpansion = {
  triggers: string[];
  terms: string[];
  preferredIds: string[];
};

const SERVICE_EXPANSIONS: ServiceExpansion[] = [
  {
    triggers: ["servicios de la app", "servicios tiene la app", "qué servicios tiene la app", "secciones principales de la app", "secciones principales app", "qué puedo hacer en la app"],
    terms: ["app sidca secciones nosotros informacion asesoramiento beneficios servicios institucionales"],
    preferredIds: ["sidca_servicios_home_resumen"],
  },
  {
    triggers: ["contacto", "whatsapp", "telefono", "numero", "número", "sede", "direccion", "dirección", "donde queda sidca"],
    terms: ["medios de contacto whatsapp telefono sede central ayacucho 227"],
    preferredIds: ["sidca_contactos_institucionales"],
  },
  {
    triggers: ["turismo", "viajes", "viaje", "paseo", "paseos", "paquete", "paquetes", "reserva de turismo", "consulto por viajes"],
    terms: ["turismo viajes paquetes reservas sidca turismo whatsapp"],
    preferredIds: ["sidca_turismo_viajes", "sidca_contactos_institucionales"],
  },
  {
    triggers: ["casa del docente", "casa docente", "hospedaje", "alojamiento", "lavalle 815"],
    terms: ["casa del docente hospedaje lavalle 815 reserva whatsapp"],
    preferredIds: ["sidca_casa_docente", "sidca_contactos_institucionales"],
  },
  {
    triggers: ["predio", "predio recreativo", "predio deportivo", "banda de varela", "cancha", "voley", "vóley", "futbol", "fútbol"],
    terms: ["predio recreativo deportivo cultural banda de varela cancha futbol voley"],
    preferredIds: ["sidca_predio_recreativo"],
  },
  {
    triggers: ["convenio", "convenios", "red de convenios", "comercios", "comercios adheridos", "descuentos", "credencial para convenio"],
    terms: ["red de convenios comercios adheridos descuentos credencial digital"],
    preferredIds: ["sidca_red_convenios", "sidca_credencial_afiliado", "sidca_contactos_institucionales"],
  },
  {
    triggers: ["hotel", "hoteles", "hoteleria", "hotelería", "hoteleria interprovincial", "hoteles interprovinciales"],
    terms: ["convenio interprovincial hoteleros hoteleria interprovincial whatsapp"],
    preferredIds: ["sidca_red_convenios", "sidca_contactos_institucionales"],
  },
  {
    triggers: ["capacitaciones", "capacitacion", "capacitación", "cursos disponibles", "curso disponible", "inscribo a un curso", "inscripcion a curso", "inscripción a curso", "aula virtual", "registro de asistencia", "secretaria de capacitacion", "secretaría de capacitación"],
    terms: ["capacitaciones cursos disponibles aula virtual secretaria de capacitacion whatsapp inscripcion registro asistencia"],
    preferredIds: ["sidca_capacitaciones", "sidca_contactos_institucionales"],
  },
  {
    triggers: ["certificado", "certificados", "entrega de certificados", "retiro de certificados", "curso aprobado", "imprimir certificado"],
    terms: ["certificados cursos aprobados entrega de certificados whatsapp"],
    preferredIds: ["sidca_certificados", "sidca_contactos_institucionales"],
  },
  {
    triggers: ["enlaces", "enlaces utiles", "enlaces útiles", "mi catamarca", "directorio", "contactos docentes", "simulador de sueldo", "junta", "medicina laboral", "recursos humanos"],
    terms: ["red de contactos informacion docente mi catamarca directorio simulador sueldo junta medicina laboral recursos humanos"],
    preferredIds: ["sidca_enlaces_utilidad"],
  },
  {
    triggers: ["asesoramiento gremial", "gremial", "paritarias", "escala salarial", "reclamo administrativo", "reclamos administrativos"],
    terms: ["asesoramiento gremial paritarias escala salarial reclamos presentaciones administrativas whatsapp"],
    preferredIds: ["sidca_asesoramiento_gremial", "sidca_contactos_institucionales"],
  },
  {
    triggers: ["asesoramiento legal", "legal", "juridico", "jurídico", "departamento juridico", "departamento jurídico", "leyes", "decretos", "resoluciones", "normativa"],
    terms: ["asesoramiento legal juridico normativas leyes decretos resoluciones departamento juridico whatsapp"],
    preferredIds: ["sidca_asesoramiento_legal", "sidca_contactos_institucionales"],
  },
  {
    triggers: [
      "en que parte de la app",
      "en qué parte de la app",
      "donde ingreso en la app",
      "dónde ingreso en la app",
      "donde debo ingresar",
      "dónde debo ingresar",
      "donde buscar en la app",
      "dónde buscar en la app",
      "buscar decretos",
      "buscar los decretos",
      "decretos de titularizacion",
      "decretos de titularización",
      "decretos de titularizaci",
      "normativa de titularizacion",
      "normativa de titularización",
    ],
    terms: [
      "app asesoramiento legal decretos resoluciones normativa titularizacion oficina gestion tramites documentacion",
      "legal decretos resoluciones otras disposiciones oficina gestion titularizacion",
    ],
    preferredIds: ["sidca_asesoramiento_legal", "sidca_oficina_gestion", "sidca_contactos_institucionales"],
  },
  {
    triggers: [
      "oficina de gestion",
      "oficina gestión",
      "oficina de gestión",
      "oficina gestion",
      "oficina gesti",
      "oficina de gesti",
      "oficina de gesti n",
      "gesti n",
      "gestion expediente",
      "gestión expediente",
      "gesti expediente",
      "gesti n expediente",
      "gestion de expedientes",
      "gestión de expedientes",
      "gesti de expedientes",
      "gesti n de expedientes",
      "formularios institucionales",
      "formulario institucional",
      "presentar documentacion",
      "presentar documentación",
      "presentar documentaci",
      "presentar documentaci n",
      "tramites del sindicato",
      "trámites del sindicato",
      "tr mites del sindicato",
      "consulta dni",
      "titularizacion",
      "titularización",
      "titularizaci",
      "titularizaci n",
    ],
    terms: [
      "oficina gestion formularios institucionales documentacion tramites gestion expedientes consulta dni titularizacion whatsapp",
      "oficina gesti formularios documentaci tramites gesti expedientes consulta dni titularizaci whatsapp",
    ],
    preferredIds: ["sidca_oficina_gestion", "sidca_contactos_institucionales"],
  },
  {
    triggers: ["credencial", "credencial digital", "credencial de afiliado"],
    terms: ["credencial digital afiliado beneficios convenios"],
    preferredIds: ["sidca_credencial_afiliado", "sidca_red_convenios"],
  },
  {
    triggers: ["soporte", "soporte tecnico", "soporte técnico", "problema con la app", "app no funciona"],
    terms: ["soporte tecnico whatsapp app"],
    preferredIds: ["sidca_contactos_institucionales"],
  },
  {
    triggers: ["afiliado adherente", "adherente", "cuota adherente", "cuotas adherentes"],
    terms: ["afiliado adherente whatsapp cuotas adherentes"],
    preferredIds: ["sidca_contactos_institucionales"],
  },
  {
    triggers: ["radio", "sidca radio"],
    terms: ["sidca radio whatsapp"],
    preferredIds: ["sidca_contactos_institucionales"],
  },
  {
    triggers: ["sala de reuniones", "reuniones", "meet"],
    terms: ["sala de reuniones app informacion"],
    preferredIds: ["sidca_sala_reuniones", "sidca_contactos_institucionales"],
  },
];

function getServiceItemId(chunk: LocalChunk): string {
  return normalizeText(getObjectField(chunk.raw, ["id"]) || "");
}

function getMatchedServiceExpansions(pregunta: string): ServiceExpansion[] {
  const normalizedPregunta = normalizeText(pregunta);
  const preguntaTokens = tokenize(normalizedPregunta);

  return SERVICE_EXPANSIONS.filter((item) =>
    item.triggers.some((trigger) =>
      triggerMatchesQuery(trigger, normalizedPregunta, preguntaTokens)
    )
  );
}

function isServicesQuestion(pregunta: string): boolean {
  const text = normalizeText(pregunta);

  const mentionsLicense =
    text.includes("licencia") ||
    text.includes("licencias") ||
    text.includes("regimen de licencias");

  const isCertificateMedical =
    text.includes("certificado medico") ||
    text.includes("certificado mdico") ||
    text.includes("carpeta medica") ||
    text.includes("carpeta mdica");

  const matched = getMatchedServiceExpansions(pregunta);

  if (isGremialAdviceServiceQuestion(pregunta) || isAppNavigationServicesQuestion(pregunta)) {
    return true;
  }

  const isBrokenOfficeManagementQuery =
    text.includes("oficina") &&
    (text.includes("gestion") ||
      text.includes("gesti") ||
      text.includes("formulario") ||
      text.includes("documentacion") ||
      text.includes("documentaci") ||
      text.includes("tramite") ||
      text.includes("tramites") ||
      text.includes("consulta dni") ||
      text.includes("titularizacion") ||
      text.includes("titularizaci"));

  if (!matched.length && !isBrokenOfficeManagementQuery) return false;

  const serviceIntent =
    text.includes("contacto") ||
    text.includes("whatsapp") ||
    text.includes("telefono") ||
    text.includes("numero") ||
    text.includes("número") ||
    text.includes("comunicarme") ||
    text.includes("asesoramiento") ||
    text.includes("gremial") ||
    text.includes("sindicato") ||
    text.includes("sidca") ||
    text.includes("donde") ||
    text.includes("a quien") ||
    text.includes("con quien") ||
    text.includes("consulto") ||
    text.includes("consulta") ||
    text.includes("inscrib") ||
    text.includes("reserva") ||
    text.includes("ver") ||
    text.includes("ubicacion") ||
    text.includes("ubicación") ||
    text.includes("oficina") ||
    text.includes("servicio") ||
    text.includes("beneficio") ||
    text.includes("app") ||
    text.includes("sede") ||
    text.includes("direccion") ||
    text.includes("dirección") ||
    text.includes("aula virtual") ||
    text.includes("mi catamarca") ||
    text.includes("simulador") ||
    text.includes("entrega de certificados") ||
    text.includes("retiro de certificados") ||
    text.includes("casa del docente") ||
    text.includes("turismo") ||
    text.includes("viajes") ||
    text.includes("hotel") ||
    text.includes("convenio") ||
    text.includes("predio") ||
    text.includes("soporte");

  if (isCertificateMedical) return false;

  if (mentionsLicense && !serviceIntent) return false;

  return true;
}

function collectPreferredServiceChunks(
  chunks: LocalChunk[],
  pregunta: string,
  maxResults: number
): LocalChunk[] {
  const matched = getMatchedServiceExpansions(pregunta);
  const preferredIds = Array.from(
    new Set(matched.flatMap((item) => item.preferredIds).map(normalizeText))
  );
  const isCertificateRequest = /\bcertificados?\b/.test(normalizeText(pregunta)) && !/certificado medico/.test(normalizeText(pregunta));
  if (isCertificateRequest) {
    const certificateId = normalizeText("sidca_certificados");
    const contactId = normalizeText("sidca_contactos_institucionales");
    preferredIds.splice(0, preferredIds.length, ...[certificateId, contactId].filter((id) => preferredIds.includes(id)));
  }

  if (!preferredIds.length) return [];

  const selected: LocalChunk[] = [];

  for (const preferredId of preferredIds) {
    const chunk = chunks.find(
      (item) => item.dominio === "servicios" && getServiceItemId(item) === preferredId
    );

    if (chunk) uniquePushChunk(selected, chunk);
    if (selected.length >= maxResults) break;
  }

  return selected.slice(0, maxResults);
}

function scoreChunk(
  chunk: LocalChunk,
  pregunta: string,
  dominio: DominioBackend
): number {
  if (chunk.dominio !== dominio) return -1000;

  const comparativeNorms = /\b(compar|diferencia|entre|versus|vs\.?|ambos|diferentes|modific|reemplaz|relacion|relaci[oó]n)\b/i.test(pregunta);
  const normalizedQuestion = normalizeText(pregunta);
  const explicitCenter = /centros? educativos?/.test(normalizedQuestion);
  const explicitGarden = /jardines? maternales?|pasitos cuidados?/.test(normalizedQuestion);
  const sourceIdNormalized = normalizeText(`${chunk.sourceId || ""} ${chunk.filename}`);
  const explicitCommonSchool = /escuela comun|escuela primaria|escuela secundaria/.test(normalizedQuestion);
  if (explicitCenter && !explicitGarden && !sourceIdNormalized.includes("centros educativos")) return -100000;
  if (explicitGarden && !explicitCenter && !sourceIdNormalized.includes("jardines maternales")) return -100000;
  if (explicitCommonSchool && sourceIdNormalized.includes("centros educativos")) return -100000;
  if (explicitCommonSchool && sourceIdNormalized.includes("jardines maternales")) return -100000;
  const normativeQuery = !comparativeNorms && /\b(?:ley|decreto(?:[- ]ley)?|resoluci[oó]n)\s*(?:n(?:ro)?[º°.]?\s*)?\d[\d.]*(?:\s*\/\s*\d{2,4})?/i.test(pregunta);
  const requestedArticle = extractArticleNumbersFromQuery(pregunta).length > 0;
  if (normativeQuery || requestedArticle) {
    const normativeHit = rankNormative([{
      sourceId: chunk.sourceId || chunk.filename,
      source: chunk.referencia,
      norm: chunk.norma || chunk.referencia,
      article: chunk.articulo,
      jurisdiction: String(chunk.metadata?.jurisdiccion || ""),
      domain: chunk.dominio,
      title: chunk.titulo,
      text: chunk.text,
      status: String(chunk.metadata?.estado_vigencia || ""),
      metadata: chunk.metadata,
    } as RagChunk], pregunta);
    if (!normativeHit.length) return -100000;
  }

  const expanded = expandQuestion(pregunta, dominio);
  const queryTokens = tokenize(expanded);
  const preguntaTokens = tokenize(pregunta);
  const expansions = getMatchedExpansions(pregunta, dominio);

  const normalizedPregunta = normalizeText(pregunta);
  const normalizedChunk = normalizeText(chunk.text);
  const normalizedTitle = normalizeText(chunk.titulo || "");
  const normalizedDescription = normalizeText(chunk.descripcion || "");
  const normalizedNivel = normalizeText(chunk.nivel || "");
  const normalizedSeccion = normalizeText(chunk.seccion || "");
  const shortTreatmentQuery =
    dominio === "licencias" &&
    isShortTreatmentQuestion(pregunta) &&
    expansions.some((expansion) =>
      (expansion.preferredArticles || []).includes("16")
    );
  const illnessDuringWorkQuery =
    dominio === "licencias" && isIllnessDuringWorkQuestion(pregunta);
  const examQuery = dominio === "licencias" && isExamLicenseQuestion(pregunta);
  const bloodDonationQuery =
    dominio === "licencias" && isBloodDonationQuestion(pregunta);
  const trainingQuery =
    dominio === "licencias" && isTrainingLicenseQuestion(pregunta);

  const chunkSearchText = normalizeText(
    [
      chunk.filename,
      chunk.titulo,
      chunk.descripcion,
      chunk.articulo,
      chunk.otorgante,
      chunk.interviniente,
      chunk.seccion,
      chunk.nivel,
      chunk.tipo,
      chunk.text,
    ]
      .filter(Boolean)
      .join("\n")
  );

  let score = 0;

  for (const token of queryTokens) {
    if (chunkSearchText.includes(token)) {
      score += token.length >= 9 ? 5 : token.length >= 6 ? 4 : 2;
    }
  }

  for (const token of preguntaTokens) {
    if (normalizedTitle.includes(token)) score += 12;
    if (normalizedDescription.includes(token)) score += 14;
    if (normalizedChunk.includes(token)) score += 4;
    if (normalizedNivel.includes(token)) score += 18;
    if (normalizedSeccion.includes(token)) score += 8;
  }

  if (normalizedPregunta.length >= 8 && chunkSearchText.includes(normalizedPregunta)) {
    score += 30;
  }

  const articleNumbers = extractArticleNumbersFromQuery(pregunta);

  for (const articleNumber of articleNumbers) {
    if (articleMatches(chunk, articleNumber)) {
      score += 120;
    }
  }

  for (const expansion of expansions) {
    const preferredArticles = expansion.preferredArticles || [];

    if (preferredArticles.some((article) => articleMatches(chunk, article))) {
      score += chunk.kind === "articulo" ? 180 : 105;
    }

    for (const term of expansion.terms) {
      const termTokens = tokenize(term);
      let localMatches = 0;

      for (const token of termTokens) {
        if (chunkSearchText.includes(token)) {
          localMatches++;
        }
      }

      if (localMatches > 0) {
        score += Math.min(localMatches * 4, 55);
      }
    }
  }

  if (shortTreatmentQuery) {
    const isShortTreatmentArticle =
      articleMatches(chunk, "16") || articleMatches(chunk, "17");
    const hasShortTreatmentCategory =
      normalizedTitle.includes("corto tratamiento") ||
      normalizedDescription.includes("corto tratamiento") ||
      normalizedTitle.includes("afecciones comunes") ||
      normalizedDescription.includes("afecciones comunes");

    if (isShortTreatmentArticle) {
      score += chunk.kind === "articulo" ? 420 : 260;
      if (articleMatches(chunk, "16")) {
        score += chunk.kind === "articulo" ? 180 : 100;
      }
    } else if (hasShortTreatmentCategory) {
      score += 180;
    } else {
      score -= 240;
    }
  }

  if (examQuery && chunk.dominio === "licencias") {
    if (articleMatches(chunk, "35")) {
      score += chunk.kind === "articulo" ? 460 : 280;
    } else if (articleMatches(chunk, "36")) {
      score += chunk.kind === "articulo" ? 180 : 110;
    } else if (articleMatches(chunk, "41")) {
      score -= 320;
    }
  }

  if (bloodDonationQuery && chunk.dominio === "licencias") {
    if (articleMatches(chunk, "50")) {
      score += chunk.kind === "articulo" ? 460 : 280;
    } else if (articleMatches(chunk, "51")) {
      score += chunk.kind === "articulo" ? 120 : 80;
    } else if (articleMatches(chunk, "41")) {
      score -= 320;
    }
  }

  if (trainingQuery && chunk.dominio === "licencias") {
    if (articleMatches(chunk, "41")) {
      score += chunk.kind === "articulo" ? 460 : 280;
    } else if (articleMatches(chunk, "34")) {
      score += chunk.kind === "articulo" ? 160 : 100;
    } else if (articleMatches(chunk, "16") || articleMatches(chunk, "17")) {
      score -= 320;
    }
  }

  if (illnessDuringWorkQuery) {
    if (articleMatches(chunk, "18")) {
      score += chunk.kind === "articulo" ? 420 : 260;
    } else {
      score -= 240;
    }
  }

  if (chunk.kind === "resumen") {
    score -= 12;
  }

  if (!queryIsGeneral(pregunta) && isGenericLicenseArticle(chunk)) {
    score -= 35;
  }

  if (!queryIsGeneral(pregunta) && isGenericStatuteArticle(chunk)) {
    score -= 25;
  }

  if (chunk.kind === "articulo") {
    score += 5;
  }

  const isCapacitacionQuery = expansions.some((expansion) =>
    (expansion.preferredArticles || []).includes("41")
  );

  if (
    isCapacitacionQuery &&
    chunk.dominio === "licencias" &&
    chunk.kind === "articulo" &&
    !["34", "41"].includes(String(chunk.articulo || ""))
  ) {
    score -= 220;
  }

  if (
    isCapacitacionQuery &&
    chunk.dominio === "licencias" &&
    chunk.kind === "resumen" &&
    !chunk.articleRefs.includes("41")
  ) {
    score -= 180;
  }

  const climateQuery = dominio === "licencias" && isClimateQuestion(pregunta);

  const longTreatmentQuery =
    dominio === "licencias" && isLongTreatmentQuestion(pregunta);
  const familyCareQuery =
    dominio === "licencias" && isFamilyCareQuestion(pregunta);
  const bereavementQuery =
    dominio === "licencias" && isBereavementQuestion(pregunta);

  if (climateQuery && chunk.dominio === "licencias") {
    if (articleMatches(chunk, "49")) {
      score += chunk.kind === "articulo" ? 360 : 220;
    } else {
      score -= 360;
    }
  }

  if (longTreatmentQuery && chunk.dominio === "licencias") {
    if (articleMatches(chunk, "20")) {
      score += chunk.kind === "articulo" ? 460 : 280;
    } else if (articleMatches(chunk, "21") || articleMatches(chunk, "22")) {
      score += chunk.kind === "articulo" ? 220 : 140;
    } else if (
      ["72", "73", "74", "75", "76"].some((article) =>
        articleMatches(chunk, article)
      )
    ) {
      score -= 320;
    }
  }

  if (familyCareQuery && chunk.dominio === "licencias") {
    if (articleMatches(chunk, "30")) {
      score += chunk.kind === "articulo" ? 460 : 280;
    } else if (articleMatches(chunk, "27")) {
      score -= 320;
    }
  }

  if (bereavementQuery && chunk.dominio === "licencias") {
    if (articleMatches(chunk, "48")) {
      score += chunk.kind === "articulo" ? 460 : 280;
    } else if (articleMatches(chunk, "27") || articleMatches(chunk, "30")) {
      score -= 260;
    }
  }

  const unjustifiedAbsenceQuery =
    dominio === "licencias" && isUnjustifiedAbsenceQuestion(pregunta);

  if (unjustifiedAbsenceQuery && chunk.dominio === "licencias") {
    const isMainAbandonment = articleMatches(chunk, "85") || articleMatches(chunk, "86");
    const isSanctionContext = articleMatches(chunk, "80") || articleMatches(chunk, "84");

    if (isMainAbandonment) {
      score += chunk.kind === "articulo" ? 360 : 220;
    } else if (isSanctionContext) {
      score += chunk.kind === "articulo" ? 220 : 140;
    } else {
      score -= 180;
    }
  }

  const isCoberturasQuery = dominio === "coberturas";
  const isSecundariaQuery =
    normalizedPregunta.includes("secundaria") ||
    normalizedPregunta.includes("secundario") ||
    normalizedPregunta.includes("nivel secundario") ||
    normalizedPregunta.includes("horas catedra");

  const isPrimariaQuery =
    normalizedPregunta.includes("primaria") ||
    normalizedPregunta.includes("primario") ||
    normalizedPregunta.includes("inicial") ||
    normalizedPregunta.includes("nivel inicial") ||
    normalizedPregunta.includes("nivel primario");

  const isCabeceraCeroQuery =
    normalizedPregunta.includes("cabecera cero") ||
    normalizedPregunta.includes("cabecera 0");

  if (isCoberturasQuery) {
    if (normalizedPregunta.includes("cobertura") || normalizedPregunta.includes("coberturas")) {
      if (
        String(chunk.articulo || "") === "Decreto Artículo 1" ||
        String(chunk.articulo || "") === "Decreto Artículo 2"
      ) {
        score += 180;
      }

      if (
        String(chunk.articulo || "") === "ANEXO I Punto 1" ||
        String(chunk.articulo || "") === "ANEXO II Punto 1"
      ) {
        score += 140;
      }
    }

    if (normalizedPregunta.includes("asamblea") || normalizedPregunta.includes("asambleas")) {
      if (
        String(chunk.articulo || "") === "ANEXO I Punto 1" ||
        String(chunk.articulo || "") === "ANEXO II Punto 1"
      ) {
        score += 160;
      }

      if (
        String(chunk.articulo || "") === "Decreto Artículo 1" ||
        String(chunk.articulo || "") === "Decreto Artículo 2"
      ) {
        score += 120;
      }
    }
  }

  const isProcedimientoAsambleaQuery =
    isCoberturasQuery &&
    (normalizedPregunta.includes("procedimiento") ||
      normalizedPregunta.includes("solicitud de cobertura") ||
      normalizedPregunta.includes("pedido de cobertura"));

  if (isProcedimientoAsambleaQuery) {
    if (
      String(chunk.articulo || "") === "ANEXO II Punto 9" ||
      String(chunk.articulo || "") === "ANEXO I Punto 11"
    ) {
      score += 260;
    }

    if (
      String(chunk.articulo || "") === "ANEXO II Punto 1" ||
      String(chunk.articulo || "") === "ANEXO I Punto 1"
    ) {
      score += 130;
    }
  }

  if (isCoberturasQuery && isCabeceraCeroQuery) {
    if (String(chunk.articulo || "") === "ANEXO II Punto 10") {
      score += 220;
    }

    if (String(chunk.articulo || "") === "ANEXO I Punto 12") {
      score += 140;
    }

    if (
      String(chunk.articulo || "") === "ANEXO II Punto 4" ||
      String(chunk.articulo || "") === "ANEXO I Punto 6"
    ) {
      score -= 90;
    }
  }

  if (isCoberturasQuery && isSecundariaQuery) {
    if (
      chunk.nivel === "Secundario" ||
      normalizedNivel.includes("secundario") ||
      normalizedChunk.includes("nivel secundario")
    ) {
      score += 120;
    }

    if (
      chunk.nivel === "Inicial y Primaria" ||
      normalizedNivel.includes("inicial") ||
      normalizedNivel.includes("primaria") ||
      normalizedChunk.includes("nivel inicial") ||
      normalizedChunk.includes("educacion primaria")
    ) {
      score -= 160;
    }
  }

  if (isCoberturasQuery && isPrimariaQuery) {
    if (
      chunk.nivel === "Inicial y Primaria" ||
      normalizedNivel.includes("inicial") ||
      normalizedNivel.includes("primaria") ||
      normalizedChunk.includes("nivel inicial") ||
      normalizedChunk.includes("educacion primaria")
    ) {
      score += 120;
    }

    if (
      chunk.nivel === "Secundario" ||
      normalizedNivel.includes("secundario") ||
      normalizedChunk.includes("nivel secundario")
    ) {
      score -= 100;
    }
  }

  if (dominio === "servicios") {
    const serviceExpansions = getMatchedServiceExpansions(pregunta);
    const serviceId = getServiceItemId(chunk);

    for (const expansion of serviceExpansions) {
      const preferredIds = expansion.preferredIds.map(normalizeText);

      if (preferredIds.includes(serviceId)) {
        score += 340;
      }

      for (const term of expansion.terms) {
        const termTokens = tokenize(term);
        let localMatches = 0;

        for (const token of termTokens) {
          if (chunkSearchText.includes(token)) {
            localMatches++;
          }
        }

        if (localMatches > 0) {
          score += Math.min(localMatches * 8, 90);
        }
      }
    }

    if (normalizedPregunta.includes("whatsapp") || normalizedPregunta.includes("telefono") || normalizedPregunta.includes("contacto")) {
      if (chunkSearchText.includes("whatsapp") || chunkSearchText.includes("wa me")) {
        score += 65;
      }
    }

    if (serviceId === "sidca_contactos_institucionales" && serviceExpansions.length > 0) {
      score += 35;
    }
  }

  return score;
}

function truncateText(text: string, maxLength: number): string {
  const clean = text.replace(/\s+/g, " ").trim();

  if (clean.length <= maxLength) return clean;

  return `${clean.slice(0, maxLength).trim()}...`;
}

function getArticleChunksByRefs(
  allChunks: LocalChunk[],
  dominio: DominioBackend,
  refs: string[]
): LocalChunk[] {
  if (!refs.length) return [];

  const normalizedRefs = refs.map((ref) => normalizeText(ref));

  return allChunks.filter((chunk) => {
    if (chunk.dominio !== dominio) return false;
    if (chunk.kind !== "articulo") return false;

    const article = normalizeText(chunk.articulo || "");

    return normalizedRefs.includes(article);
  });
}

function getResumenChunksByRefs(
  allChunks: LocalChunk[],
  dominio: DominioBackend,
  refs: string[]
): LocalChunk[] {
  if (!refs.length) return [];

  const normalizedRefs = refs.map((ref) => normalizeText(ref));

  return allChunks.filter((chunk) => {
    if (chunk.dominio !== dominio) return false;
    if (chunk.kind !== "resumen") return false;

    return chunk.articleRefs.some((ref) =>
      normalizedRefs.includes(normalizeText(ref))
    );
  });
}

function uniquePushChunk(target: LocalChunk[], chunk: LocalChunk): void {
  const exists = target.some((item) => item.id === chunk.id);

  if (!exists) {
    target.push(chunk);
  }
}

function uniqueInsertChunk(target: LocalChunk[], index: number, chunk: LocalChunk): void {
  const exists = target.some((item) => item.id === chunk.id);

  if (!exists) {
    target.splice(index, 0, chunk);
  }
}

function collectArticleAndSummaryChunks(
  allChunks: LocalChunk[],
  dominio: DominioBackend,
  refs: string[],
  maxResults: number
): LocalChunk[] {
  const selected: LocalChunk[] = [];
  const articleChunks = getArticleChunksByRefs(allChunks, dominio, refs);
  const summaryChunks = getResumenChunksByRefs(allChunks, dominio, refs);

  for (const ref of refs) {
    for (const articleChunk of articleChunks) {
      if (selected.length >= maxResults) return selected;
      if (articleMatches(articleChunk, ref)) {
        uniquePushChunk(selected, articleChunk);
      }
    }

    for (const summaryChunk of summaryChunks) {
      if (selected.length >= maxResults) return selected;
      if (summaryChunk.articleRefs.some((item) => normalizeText(item) === normalizeText(ref))) {
        uniquePushChunk(selected, summaryChunk);
      }
    }
  }

  return selected;
}

async function searchLocalFragments(
  input: ChatbotQueryInput,
  dominio: DominioBackend,
  documentos?: LocalChunk[],
): Promise<LocalChunk[]> {
  const chunks = documentos || (await loadLocalChunks());
  const maxResults = Math.min(Math.max(input.maxResults ?? 5, 1), 8);
  const explicitlyRequestedNorms = Array.from(new Map(parseNormativeReferences(input.pregunta).map((reference) => [
    [reference.kind, reference.number, reference.year || "", reference.qualifier || "", reference.jurisdiction || ""].join("|"), reference,
  ])).values());
  const requestedArticles = extractArticleNumbersFromQuery(input.pregunta);
  const asksReasonsAndAffairs = /\brazones? particulares\b/.test(normalizeText(input.pregunta)) && /\basuntos? particulares\b/.test(normalizeText(input.pregunta));

  if (dominio === "licencias" && explicitlyRequestedNorms.length === 0) {
    if (requestedArticles.length > 1) {
      const allRequestedArticles = requestedArticles.flatMap((article) =>
        chunks.filter((chunk) => chunk.dominio === dominio && chunk.kind === "articulo" && articleMatches(chunk, article)));
      return Array.from(new Map(allRequestedArticles.map((chunk) => [chunk.id, chunk])).values());
    }
    if (asksReasonsAndAffairs) {
      return ["45", "52"].flatMap((article) => chunks.filter((chunk) => chunk.dominio === dominio && chunk.kind === "articulo" && articleMatches(chunk, article)));
    }
  }

  if (dominio === "licencias" && explicitlyRequestedNorms.length === 0 && requestedArticles.length <= 1) {
    const targetArticle = getSpecificLicenseArticle(input.pregunta);
    if (targetArticle && ["30", "35", "40", "45", "49", "52", "55"].includes(targetArticle)) {
      const targetChunks = chunks.filter((chunk) => chunk.dominio === dominio && chunk.kind === "articulo" && articleMatches(chunk, targetArticle));
      if (targetChunks.length) return targetChunks.slice(0, maxResults);
    }
  }

  const isMultiNormComparison = /\b(?:comparar|comparacion|compara|comparan|comparando|comparado|diferencias?|relacion|entre|versus|vs)\b/i.test(normalizeText(input.pregunta)) && explicitlyRequestedNorms.length > 1;
  if (isMultiNormComparison) {
    const perNorm = Math.max(1, Math.floor(maxResults / explicitlyRequestedNorms.length));
    const selected: LocalChunk[] = [];
    for (const requestedNorm of explicitlyRequestedNorms) {
      const matches = chunks.filter((chunk) => chunkMatchesNormativeReference({
        sourceId: chunk.sourceId || chunk.filename, source: chunk.referencia, norm: chunk.norma || chunk.referencia,
        jurisdiction: String(chunk.metadata?.jurisdiccion || ""), metadata: chunk.metadata, text: chunk.text,
      }, requestedNorm)).map((chunk) => ({ chunk, score: scoreChunk(chunk, input.pregunta, chunk.dominio) }))
        .map((match) => {
          const normalizedQuestion = normalizeText(input.pregunta);
          const normalizedText = normalizeText(match.chunk.text);
          const asksSecondaryTechnical = /secundaria/.test(normalizedQuestion) && /tecnica/.test(normalizedQuestion);
          const directlyAddressesSecondaryTechnical = /secundaria|nivel medio/.test(normalizedText) && /tecnic/.test(normalizedText);
          return { ...match, score: match.score + (asksSecondaryTechnical && directlyAddressesSecondaryTechnical ? 120 : 0) };
        })
        .sort((a, b) => b.score - a.score);
      selected.push(...matches.slice(0, perNorm).map(({ chunk }) => chunk));
    }
    return selected;
  }

  if (requestedArticles.length > 1) {
    const normScopedChunks = explicitlyRequestedNorms.length
      ? chunks.filter((chunk) => explicitlyRequestedNorms.some((norm) => chunkMatchesNormativeReference({
        sourceId: chunk.sourceId || chunk.filename, source: chunk.referencia, norm: chunk.norma || chunk.referencia,
        jurisdiction: String(chunk.metadata?.jurisdiccion || ""), metadata: chunk.metadata, text: chunk.text,
      }, norm)))
      : chunks.filter((chunk) => chunk.dominio === dominio);
    return Array.from(new Map(requestedArticles.flatMap((article) => normScopedChunks.filter((chunk) =>
      chunk.kind === "articulo" && articleMatches(chunk, article))).map((chunk) => [chunk.id, chunk])).values());
  }

  const decreeAndModification = input.pregunta.match(/\b(?:decreto(?:\s+acuerdo)?|dcto\.?\s*acdo\.?)\s*(?:n(?:ro)?[º°.]?\s*)?(\d{2,5})\s*\/\s*(\d{4})/i);
  const resolutionModification = input.pregunta.match(/\bRESRH[- ]?(\d{4})[- ]?(\d+)/i);
  if (decreeAndModification && resolutionModification && /modific|reemplaz/i.test(input.pregunta)) {
    const requestedDecree = parseNormativeReferences(decreeAndModification[0])[0];
    const resolutionNumber = `resrh ${resolutionModification[1]} ${resolutionModification[2]}`;
    const decreeChunks = chunks.filter((chunk) => {
      return Boolean(requestedDecree && chunkMatchesNormativeReference({ sourceId: chunk.sourceId || chunk.filename, source: chunk.referencia, norm: chunk.norma || chunk.referencia, jurisdiction: String(chunk.metadata?.jurisdiccion || ""), metadata: chunk.metadata, text: chunk.text }, requestedDecree));
    });
    const modifyingChunks = chunks.filter((chunk) => {
      const matchesResolution = (value: string) => Array.from(value.matchAll(/\bRESRH[- ]?(\d{4})[- ]?(\d+)(?:-[A-Z0-9]+)*\b/gi))
        .some((match) => `${match[1]} ${match[2]}` === resolutionNumber.replace(/^resrh\s*/i, ""));
      return matchesResolution(`${chunk.norma || ""} ${chunk.metadata?.documento || ""} ${chunk.referencia} ${chunk.text}`);
    });
    const annexChunk = decreeChunks.find((chunk) => /articulo\s*2\b/.test(normalizeText(String(chunk.articulo || ""))) || /como anexo ii/.test(normalizeText(chunk.text)));
    const selected = [annexChunk, ...modifyingChunks.filter((chunk) => /anexo ii|modificacion y reemplazo/i.test(`${chunk.titulo || ""} ${chunk.text}`))]
      .filter((chunk): chunk is LocalChunk => Boolean(chunk));
    if (selected.length) return selected.slice(0, maxResults);
  }

  if (dominio === "servicios") {
    const preferredServiceChunks = collectPreferredServiceChunks(
      chunks,
      input.pregunta,
      maxResults
    );

    if (preferredServiceChunks.length) {
      return preferredServiceChunks;
    }
  }

  if (dominio === "licencias" && isClimateQuestion(input.pregunta)) {
    const onlyArticle49 = collectArticleAndSummaryChunks(
      chunks,
      dominio,
      ["49"],
      maxResults
    );

    if (onlyArticle49.length) {
      return onlyArticle49;
    }
  }

  if (dominio === "licencias" && isLongTreatmentQuestion(input.pregunta)) {
    const longTreatmentChunks = collectArticleAndSummaryChunks(
      chunks,
      dominio,
      ["20", "21", "22"],
      maxResults
    );

    if (longTreatmentChunks.length) {
      return longTreatmentChunks;
    }
  }

  if (dominio === "licencias" && isBereavementQuestion(input.pregunta)) {
    const bereavementChunks = collectArticleAndSummaryChunks(
      chunks,
      dominio,
      ["48"],
      maxResults
    );

    if (bereavementChunks.length) {
      return bereavementChunks;
    }
  }

  if (dominio === "licencias" && isFamilyCareQuestion(input.pregunta)) {
    const familyCareChunks = collectArticleAndSummaryChunks(
      chunks,
      dominio,
      ["30"],
      maxResults
    );

    if (familyCareChunks.length) {
      return familyCareChunks;
    }
  }

  if (dominio === "licencias" && isUnjustifiedAbsenceQuestion(input.pregunta)) {
    const absenceChunks = collectArticleAndSummaryChunks(
      chunks,
      dominio,
      ["85", "86", "80", "84"],
      maxResults
    );

    if (absenceChunks.length) {
      return absenceChunks;
    }
  }

  if (dominio === "licencias" && isMaternityQuestion(input.pregunta)) {
    const maternityChunks = collectArticleAndSummaryChunks(
      chunks,
      dominio,
      ["27"],
      maxResults
    );

    if (maternityChunks.length) {
      return maternityChunks;
    }
  }

  if (dominio === "licencias" && isIllnessDuringWorkQuestion(input.pregunta)) {
    const illnessDuringWorkChunks = collectArticleAndSummaryChunks(
      chunks,
      dominio,
      ["18"],
      maxResults,
    );

    if (illnessDuringWorkChunks.length) {
      return illnessDuringWorkChunks;
    }
  }

  if (dominio === "licencias" && isSuperpositionQuestion(input.pregunta)) {
    const superpositionChunks = collectArticleAndSummaryChunks(
      chunks,
      dominio,
      ["53", "54"],
      maxResults
    );

    if (superpositionChunks.length) {
      return superpositionChunks;
    }
  }

  if (dominio === "licencias" && isExamLicenseQuestion(input.pregunta)) {
    const examChunks = collectArticleAndSummaryChunks(
      chunks,
      dominio,
      ["35", "36"],
      maxResults
    );

    if (examChunks.length) {
      return examChunks;
    }
  }

  if (dominio === "licencias" && isBloodDonationQuestion(input.pregunta)) {
    const bloodDonationChunks = collectArticleAndSummaryChunks(
      chunks,
      dominio,
      ["50"],
      maxResults
    );

    if (bloodDonationChunks.length) {
      return bloodDonationChunks;
    }
  }

  if (dominio === "licencias" && isTrainingLicenseQuestion(input.pregunta)) {
    const trainingChunks = collectArticleAndSummaryChunks(
      chunks,
      dominio,
      ["41", "34"],
      maxResults
    );

    if (trainingChunks.length) {
      return trainingChunks;
    }
  }

  const matchedExpansions = getMatchedExpansions(input.pregunta, dominio);
  const preferredArticles = Array.from(
    new Set(matchedExpansions.flatMap((item) => item.preferredArticles || []))
  );

  if (
    dominio === "licencias" &&
    preferredArticles.includes("16") &&
    isShortTreatmentQuestion(input.pregunta)
  ) {
    const shortTreatmentChunks = collectArticleAndSummaryChunks(
      chunks,
      dominio,
      ["16", "17"],
      maxResults
    );

    if (shortTreatmentChunks.length) {
      return shortTreatmentChunks;
    }
  }

  if (dominio === "licencias" && preferredArticles.includes("53")) {
    const superpositionChunks = collectArticleAndSummaryChunks(
      chunks,
      dominio,
      ["53", "54"],
      maxResults
    );

    if (superpositionChunks.length) {
      return superpositionChunks;
    }
  }

  const scored = chunks
    .map((chunk) => ({
      chunk,
      score: scoreChunk(chunk, input.pregunta, dominio),
    }))
    .filter((item) => item.score > 0)
    .sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score;

      if (a.chunk.kind === "articulo" && b.chunk.kind !== "articulo") return -1;
      if (b.chunk.kind === "articulo" && a.chunk.kind !== "articulo") return 1;

      return 0;
    });

  if (!scored.length) return [];

  const topScore = scored[0]?.score || 0;
  const hasExpansion = matchedExpansions.length > 0;
  const hasDirectArticle = extractArticleNumbersFromQuery(input.pregunta).length > 0;

  if (topScore < 12 && !hasExpansion && !hasDirectArticle) {
    return [];
  }

  const selected: LocalChunk[] = [];

  for (const item of scored) {
    if (selected.length >= maxResults) break;

    const chunk = item.chunk;

    if (chunk.kind === "resumen" && chunk.articleRefs.length) {
      const articleChunks = getArticleChunksByRefs(
        chunks,
        dominio,
        chunk.articleRefs
      );

      for (const articleChunk of articleChunks) {
        if (selected.length >= maxResults) break;
        uniquePushChunk(selected, articleChunk);
      }

      if (selected.length < maxResults) {
        uniquePushChunk(selected, chunk);
      }

      continue;
    }

    uniquePushChunk(selected, chunk);
  }

  if (dominio === "licencias" && preferredArticles.includes("41")) {
    const article34 = chunks.find(
      (chunk) =>
        chunk.dominio === "licencias" &&
        chunk.kind === "articulo" &&
        String(chunk.articulo) === "34"
    );

    if (article34) {
      uniqueInsertChunk(selected, 1, article34);
    }
  }

  if (preferredArticles.length) {
    const preferredSummaries = getResumenChunksByRefs(
      chunks,
      dominio,
      preferredArticles
    );

    for (const summary of preferredSummaries) {
      if (selected.length >= maxResults) break;
      uniquePushChunk(selected, summary);
    }
  }

  return selected.slice(0, maxResults);
}

function buildContext(chunks: LocalChunk[]): string {
  return chunks
    .map((chunk, index) => {
      const structuredRecord = chunk.dominio === "nomenclador" ? getNomenclatorRecord(chunk) : null;
      const content = structuredRecord && (structuredRecord.categoria || structuredRecord.titulo_articulo)
        ? renderNomenclatorRecord(chunk, structuredRecord)
        : chunk.kind === "articulo" ? buildVerbatimArticleAnswer(chunk) : truncateText(chunk.text, 1200);
      const parts = [
        `### Fragmento ${index + 1}`,
        `Archivo: ${chunk.filename}`,
        `Tipo: ${chunk.kind}`,
        `Fuente: ${chunk.referencia}`,
        chunk.norma ? `Norma identificada: ${chunk.norma}` : "",
        chunk.articulo ? `Artículo/s: ${chunk.articulo}` : "",
        chunk.seccion ? `Sección: ${chunk.seccion}` : "",
        chunk.nivel ? `Nivel: ${chunk.nivel}` : "",
        chunk.metadata?.jurisdiccion ? `Jurisdicción: ${chunk.metadata.jurisdiccion}` : "",
        chunk.titulo ? `Título: ${chunk.titulo}` : "",
      chunk.metadata ? `Metadatos documentales: ${JSON.stringify(Object.fromEntries(Object.entries(chunk.metadata).filter(([key]) => /vigencia|actualiz|derog|sustit|histor|observ|nota|control|integridad|faltant|repetid|url_texto/i.test(key))))}` : "",
        chunk.descripcion ? `Descripción: ${chunk.descripcion}` : "",
        chunk.otorgante ? `Funcionario otorgante: ${chunk.otorgante}` : "",
        chunk.interviniente
          ? `Funcionario interviniente: ${chunk.interviniente}`
          : "",
        `Contenido:\n${content}`,
      ];

      return parts.filter(Boolean).join("\n");
    })
    .join("\n\n");
}

function buildLocalFallbackAnswer(
  chunks: LocalChunk[],
  pregunta: string,
): string {
  if (!chunks.length) return NO_ENCONTRADO;

  const requestedArticles = extractArticleNumbersFromQuery(pregunta);
  const multiNormOrArticleQuery = parseNormativeReferences(pregunta).length > 1 || requestedArticles.length > 1;
  const directArticle = requestedArticles.length === 1
    ? chunks.find((chunk) => chunk.kind === "articulo" && articleMatches(chunk, requestedArticles[0]))
    : undefined;
  if (!multiNormOrArticleQuery && directArticle) return buildVerbatimArticleAnswer(directArticle);
  if (!multiNormOrArticleQuery && chunks[0]?.kind === "articulo" && scoreChunk(chunks[0], pregunta, chunks[0].dominio) >= 20) {
    return buildVerbatimArticleAnswer(chunks[0]);
  }

  if (chunks[0]?.dominio === "servicios") {
    const primary = chunks[0];
    const sugerida = getObjectField(primary.raw, ["respuesta_sugerida"]);
    const titulo =
      primary.titulo ||
      getObjectField(primary.raw, ["categoria", "titulo", "nombre"]) ||
      "Servicio SiDCa";

    return [
      `Encontré información relacionada con ${titulo}.`,
      sugerida
        ? truncateText(sugerida, 500)
        : "La documentación disponible contiene información relacionada, pero no detalla una orientación adicional.",
      "La información de respaldo queda disponible en la tarjeta documental.",
    ].join("\n\n");
  }

  const primary = chunks[0];
  const title = primary.articulo
    ? `Artículo ${primary.articulo}${
        primary.descripcion ? ` – ${primary.descripcion}` : ""
      }`
    : primary.descripcion || primary.titulo || "fragmento relacionado";
  const asksForProcedure = /\b(c[oó]mo|qu[eé] tengo que|d[oó]nde|qui[eé]n|cu[aá]ntos?|presentar|tr[aá]mite|procedimiento)\b/i.test(
    pregunta,
  );
  const hasProcedureSignals = /\b(presentar|presentaci[oó]n|solicitud|tr[aá]mite|plazo|deber[aá]|ante)\b/i.test(
    primary.text,
  );

  return [
    `Encontré información relacionada con ${title}.`,
    asksForProcedure && !hasProcedureSignals
      ? "La documentación disponible permite identificar el encuadre normativo, pero no detalla en este fragmento el procedimiento que debés realizar. Para recibir orientación sobre el trámite, comunicate con SiDCa."
      : asksForProcedure
        ? "La documentación disponible contiene indicaciones relacionadas. Podés abrir el artículo para revisar el texto completo antes de realizar el trámite."
      : "La documentación disponible identifica el artículo relacionado. Podés abrirlo para consultar el texto completo.",
    `Base normativa: ${title}.`,
  ].join("\n\n");
}

function buildVerbatimArticleAnswer(chunk: LocalChunk): string {
  let content = getObjectField(chunk.raw, ["contenido", "content", "texto", "text"]) || chunk.text;
  const requestedArticle = String(chunk.articulo || "").match(/\d+/)?.[0];
  const headings = Array.from(content.matchAll(/^[\t ]*\[?ART[IÍ]CULO[\t ]*(\d+)(?=\D|$)[^\n]*/gim));
  const firstHeadingIndex = headings.findIndex((heading) => !requestedArticle || heading[1] === requestedArticle);
  if (firstHeadingIndex >= 0) {
    const firstHeading = headings[firstHeadingIndex];
    const nextHeading = headings.slice(firstHeadingIndex + 1).find((heading) => heading[1] !== requestedArticle);
    const startIndex = firstHeading.index ?? 0;
    const endIndex = nextHeading?.index ?? content.length;
    content = content.slice(startIndex, endIndex).trim();
  }
  return `${chunk.referencia}\n\n${content}`.trim();
}

function buildDocumentedServiceAnswer(chunks: LocalChunk[], pregunta: string): { answer: string; chunk: LocalChunk | null } | null {
  const normalizedQuestion = normalizeText(pregunta);
  const isCertificateRequest = /\bcertificados?\b/.test(normalizedQuestion) && !/certificado medico/.test(normalizedQuestion);
  const primary = isCertificateRequest
    ? chunks.find((chunk) => getServiceItemId(chunk) === normalizeText("sidca_certificados"))
    : undefined;
  if (isCertificateRequest && !primary) {
    return { answer: "No encontré una ficha documental específica de certificados con instrucciones confirmadas. La información disponible sobre capacitaciones no permite inferir el procedimiento para obtener o descargar un certificado; hace falta actualizar la documentación de certificados antes de orientar el trámite.", chunk: null };
  }
  const service = primary || chunks.find((chunk) => chunk.dominio === "servicios");
  if (!service) return null;
  const suggested = getObjectField(service.raw, ["respuesta_sugerida"]);
  const title = service.titulo || getObjectField(service.raw, ["titulo_articulo", "categoria", "titulo"]) || "el servicio";
  if (!suggested) return { answer: `La documentación identifica ${title}, pero no incluye una respuesta_sugerida con instrucciones suficientes. Hace falta documentar el procedimiento confirmado antes de orientar al usuario.`, chunk: service };
  const asksDownload = /\b(?:descarg|imprim|obten|baj[aá]r?)\w*\b/.test(normalizedQuestion);
  if (isCertificateRequest && asksDownload) {
    return { answer: `La ficha documental “${title}” indica: ${suggested.trim()}\n\nLa fuente no confirma que la ruta ni el estado de impresión reflejen la versión actual de la APP SIDCA. No puedo confirmar el procedimiento vigente; hace falta actualizar y validar esta ficha antes de orientar la descarga.`, chunk: service };
  }
  return { answer: suggested.trim(), chunk: service };
}

type NomenclatorResolution = { answer: string; chunks: LocalChunk[]; clarification?: boolean };

type NomenclatorCandidate = {
  chunk: LocalChunk;
  record: Record<string, any>;
  category: string;
  identity: string;
  sourceKind: "common" | "centers" | "gardens" | "other";
  roleAnchor: string;
  roleDetails: string[];
  pof: boolean;
  institution: string;
};

function getNomenclatorRecord(chunk: LocalChunk): Record<string, any> | null {
  return chunk.raw && typeof chunk.raw === "object" && !Array.isArray(chunk.raw)
    ? chunk.raw as Record<string, any>
    : null;
}

function nomenclatorRecordIdentity(chunk: LocalChunk, record: Record<string, any>): string {
  return normalizeText(`${chunk.sourceId || ""} ${chunk.filename} ${chunk.referencia} ${record.normativa || ""} ${record.documento || ""} ${record.titulo_documento || ""}`);
}

function isNomenclatorPof(record: Record<string, any>): boolean {
  return /pof|anexo unico\s*[–-]?\s*escuela|planta organica/.test(normalizeText(`${record.tipo || ""} ${record.alcance || ""} ${record.titulo_articulo || ""} ${record.articulo || ""}`));
}

function normalizeCargoLabel(value: string): string {
  return normalizeText(value)
    .replace(/\b(director|vicedirector|maestro|profesor|secretario|prosecretario|preceptor|bibliotecario|supervisor|rector|vicerrector) a\b/g, "$1")
    .replace(/\b(vicedirector|vicerrector)\s+de\b/g, "$1 de");
}

function cargoAnchorAndDetails(value: string): { anchor: string; details: string[] } {
  const normalized = normalizeCargoLabel(value).replace(/\bjornada\b.*$/i, "");
  const tokens = normalized.split(/\s+/).filter(Boolean);
  const ignored = new Set(["cargo", "docente", "nivel", "categoria", "de", "del", "la", "el", "las", "los", "y", "en", "para", "con"]);
  const roleIndex = tokens.findIndex((token) => !ignored.has(token));
  if (roleIndex < 0) return { anchor: "", details: [] };
  const anchor = tokens[roleIndex];
  const detailIgnored = new Set([...ignored, "educacion", "comun", "unidad", "institucional", "jardin", "maternal", "escuela", "inicial", "primario", "primaria", "secundario", "secundaria", "superior", "artistica", "tecnico", "tecnica", "jornada", "completa", "extendida", "simple", "js", "jc"]);
  return { anchor, details: tokens.slice(roleIndex + 1).filter((token) => !detailIgnored.has(token) && token.length > 2) };
}

function nomenclatorSourceKind(identity: string): NomenclatorCandidate["sourceKind"] {
  if (/2453|centros educativos/.test(identity)) return "centers";
  if (/1480|jardines maternales|pasitos cuidados/.test(identity)) return "gardens";
  if (/2431|nomenclador docente/.test(identity)) return "common";
  return "other";
}

function nomenclatorJornada(candidate: NomenclatorCandidate): string {
  const normalized = normalizeText(`${candidate.record.jornada || ""} ${candidate.category}`);
  if (/jornada completa|\bjc\b/.test(normalized)) return "completa";
  if (/jornada extendida/.test(normalized)) return "extendida";
  if (/jornada simple|\bjs\b/.test(normalized)) return "simple";
  return "";
}

function nomenclatorLevel(candidate: NomenclatorCandidate): string {
  return normalizeText(`${candidate.record.nivel_modalidad || ""} ${candidate.record.nivel_educativo || ""} ${candidate.record.nivel || ""} ${candidate.record.modalidad || ""}`);
}

function nomenclatorComparisonSegments(question: string, roleAnchors: string[]): string[] {
  const normalized = normalizeCargoLabel(question);
  const explicit = normalized.split(/\s+\b(?:vs\.?|versus|contra)\b\s+/i).map((part) => part.trim()).filter(Boolean);
  if (explicit.length > 1) return explicit;
  const anchors = roleAnchors.sort((left, right) => right.length - left.length).map((value) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  if (!anchors.length) return [normalized];
  const anchorPattern = `(?:${anchors.join("|")})`;
  const separator = new RegExp(`\\s+(?:con|y)\\s+(?=(?:(?:un|una|el|la)\\s+)?${anchorPattern}\\b)`, "i");
  return normalized.split(separator).map((part) => part.trim()).filter(Boolean);
}

function getNomenclatorNumber(record: Record<string, any>, keys: string[]): number | null {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "number" && Number.isFinite(value)) return value;
    if (typeof value === "string" && /^\d+(?:\.\d+)?$/.test(value.trim())) return Number(value);
  }
  return null;
}

function renderNomenclatorRecord(chunk: LocalChunk, record: Record<string, any>): string {
  const category = String(record.categoria || record.titulo_articulo || record.titulo || record.nombre || "Cargo sin denominación");
  const norm = String(record.normativa || record.documento || record.fuente || chunk.referencia);
  const normalizedCategory = normalizeText(category);
  const jornada = record.jornada || (normalizedCategory.includes("jornada completa") || /\bjc\b/.test(normalizedCategory) ? "Completa" : normalizedCategory.includes("jornada extendida") ? "Extendida" : normalizedCategory.includes("jornada simple") || /\bjs\b/.test(normalizedCategory) ? "Simple" : "");
  const hours = getNomenclatorNumber(record, ["horas_reloj_semanales"]);
  const chairHours = getNomenclatorNumber(record, ["horas_catedra_semanales", "horas_catedra"]);
  const clockMinutes = getNomenclatorNumber(record, ["minutos_reloj_semanales", "minutos_reloj"]);
  const salary = getNomenclatorNumber(record, ["puntos_salariales", "puntos_nomenclador", "puntaje_salarial", "puntos"]);
  const incompatibility = getNomenclatorNumber(record, ["puntaje_incompatibilidad", "puntos_incompatibilidad", "puntaje_compatibilidad"]);
  const fields = [
    `Cargo: ${category}`,
    record.nivel_modalidad || record.nivel_educativo || record.nivel ? `Nivel/modalidad: ${record.nivel_modalidad || record.nivel_educativo || record.nivel}` : "",
    record.modalidad ? `Modalidad: ${record.modalidad}` : "",
    record.institucion || record.tipo_institucion ? `Institución: ${record.institucion || record.tipo_institucion}` : "",
    jornada ? `Jornada: ${jornada}` : "",
    `Horas reloj semanales: ${hours === null ? "sin dato en el registro" : hours}`,
    chairHours === null ? "" : `Horas cátedra semanales: ${chairHours}`,
    clockMinutes === null ? "" : `Minutos reloj semanales: ${clockMinutes}`,
    `Puntos salariales/nomenclador: ${salary === null ? "sin dato en el registro" : salary}`,
    `Puntos de incompatibilidad: ${incompatibility === null ? "sin dato en el registro" : incompatibility}`,
    record.unidad ? `Unidad: ${record.unidad}` : "",
    record.tipo ? `Tipo de registro: ${record.tipo}` : "",
    record.alcance ? `Alcance: ${record.alcance}` : "",
    `Norma: ${norm}`,
    record.articulo ? `Artículo/registro: ${record.articulo}` : "",
    record.id ? `ID documental: ${record.id}` : "",
    record.pagina_pdf ? `Página PDF: ${record.pagina_pdf}` : "",
  ].filter(Boolean);
  return fields.join("; ");
}

function resolveNomenclatorQuery(pregunta: string, chunks: LocalChunk[]): NomenclatorResolution | null {
  const text = normalizeText(pregunta);
  const compare = /\b(?:comparar|comparacion|compara|comparan|comparando|comparado|diferencias?|entre|versus|vs|contra)\b/.test(text);
  const pofAsked = /\bpof\b|planta organica funcional/.test(text);
  const explicitlyRequestedNorms = parseNormativeReferences(pregunta);
  const asksMetrics = /\b(?:horas?|carga horaria|puntos?|puntaje|incompatibilidad|jornada)\b/.test(text);
  const candidates = chunks.flatMap((chunk): NomenclatorCandidate[] => {
    if (chunk.dominio !== "nomenclador") return [];
    const record = getNomenclatorRecord(chunk);
    if (!record) return [];
    const identity = nomenclatorRecordIdentity(chunk, record);
    const category = normalizeText(String(record.categoria || record.titulo_articulo || record.titulo || ""));
    const institution = normalizeText(String(record.institucion || record.establecimiento || record.escuela || record.nombre_institucion || record.unidad_institucional || ""));
    const signature = cargoAnchorAndDetails(category);
    const sourceKind = nomenclatorSourceKind(identity);
    const pof = isNomenclatorPof(record);
    if (!signature.anchor || (pof && !pofAsked) || (!pof && pofAsked)) return [];
    if (explicitlyRequestedNorms.length && !explicitlyRequestedNorms.some((norm) => chunkMatchesNormativeReference({
      sourceId: chunk.sourceId || chunk.filename, source: chunk.referencia, norm: chunk.norma || chunk.referencia,
      jurisdiction: String(chunk.metadata?.jurisdiccion || ""), metadata: chunk.metadata, text: chunk.text,
    }, norm))) return [];
    return [{ chunk, record, category, identity, sourceKind, roleAnchor: signature.anchor, roleDetails: signature.details, pof, institution }];
  });
  if (!asksMetrics && !compare || !candidates.length) return null;

  const availableAnchors = Array.from(new Set(candidates.map((candidate) => candidate.roleAnchor)));
  const explicitAnchors = availableAnchors.filter((anchor) => new RegExp(`(?:^|\\s)${anchor.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?:$|\\s)`).test(text));
  const candidateDetails = new Set(candidates.flatMap((candidate) => candidate.roleDetails));
  const explicitDetails = Array.from(candidateDetails).filter((detail) => new RegExp(`(?:^|\\s)${detail.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?:$|\\s)`).test(text));
  const requestedJornadas = ["completa", "extendida", "simple"].filter((jornada) =>
    new RegExp(`\\bjornada\\s+${jornada}\\b|\\b${jornada}\\b`).test(text));
  const specificLevels = ["primario comun", "primario especial", "primario adulto", "educacion inicial", "nivel inicial", "educacion tecnica", "nivel secundario"];
  const requestedSpecificLevels = specificLevels.filter((level) => text.includes(level));
  const requestedLevels = requestedSpecificLevels.length ? requestedSpecificLevels : ["inicial", "primaria", "primario", "secundaria", "secundario", "superior"].filter((level) => new RegExp(`\\b${level}\\b`).test(text));
  const requestedInstitutions: NomenclatorCandidate["sourceKind"][] = [];
  if (/centros? educativos?/.test(text)) requestedInstitutions.push("centers");
  if (/jardines? maternales?|pasitos cuidados?/.test(text)) requestedInstitutions.push("gardens");
  if (/escuela comun|escuela primaria|escuela secundaria/.test(text)) requestedInstitutions.push("common");

  const matchesSegment = (candidate: NomenclatorCandidate, segment: string, dimension?: { kind: string; value: string }): boolean => {
    const segmentText = normalizeCargoLabel(segment);
    const segmentAnchors = availableAnchors.filter((anchor) => new RegExp(`(?:^|\\s)${anchor.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?:$|\\s)`).test(segmentText));
    if (segmentAnchors.length && !segmentAnchors.includes(candidate.roleAnchor)) return false;
    const segmentDetails = Array.from(candidateDetails).filter((detail) => new RegExp(`(?:^|\\s)${detail.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?:$|\\s)`).test(segmentText));
    const detailsForRole = new Set(candidates.filter((item) => item.roleAnchor === candidate.roleAnchor).flatMap((item) => item.roleDetails));
    if (segmentDetails.some((detail) => detailsForRole.has(detail) && !candidate.roleDetails.includes(detail))) return false;
    const centers = /centros? educativos?/.test(segmentText);
    const gardens = /jardines? maternales?|pasitos cuidados?/.test(segmentText);
    const common = /escuela comun|escuela primaria|escuela secundaria/.test(segmentText);
    if (centers && candidate.sourceKind !== "centers" || gardens && candidate.sourceKind !== "gardens" || common && candidate.sourceKind !== "common") return false;
    const namedInstitutions = Array.from(new Set(candidates.map((item) => item.institution).filter(Boolean)));
    const requestedNamedInstitutions = namedInstitutions.filter((institution) => segmentText.includes(institution));
    if (requestedNamedInstitutions.length && !requestedNamedInstitutions.includes(candidate.institution)) return false;
    const journeyOptions = ["completa", "extendida", "simple"].filter((journey) => new RegExp(`\\bjornada\\s+${journey}\\b|\\b${journey}\\b`).test(segmentText));
    const specificLevelOptions = specificLevels.filter((level) => segmentText.includes(level));
    const levelOptions = specificLevelOptions.length ? specificLevelOptions : ["inicial", "primaria", "primario", "secundaria", "secundario", "superior"].filter((level) => new RegExp(`\\b${level}\\b`).test(segmentText));
    const jornada = nomenclatorJornada(candidate);
    const level = nomenclatorLevel(candidate);
    if (journeyOptions.length && !journeyOptions.includes(jornada)) return false;
    if (levelOptions.length && !levelOptions.some((option) => level.includes(option))) return false;
    if (dimension?.kind === "role" && candidate.roleAnchor !== dimension.value) return false;
    if (dimension?.kind === "journey" && jornada !== dimension.value) return false;
    if (dimension?.kind === "level" && !level.includes(dimension.value)) return false;
    if (dimension?.kind === "institution" && candidate.sourceKind !== dimension.value) return false;
    return true;
  };

  const roleAnchors = explicitAnchors.length ? explicitAnchors : availableAnchors.filter((anchor) => new RegExp(`(?:^|\\s)${anchor.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?:$|\\s)`).test(text));
  const segments = compare ? nomenclatorComparisonSegments(pregunta, roleAnchors) : [pregunta];
  let groups: NomenclatorCandidate[][] = [];
  if (compare && segments.length > 1) {
    groups = segments.map((segment) => candidates.filter((candidate) => matchesSegment(candidate, segment)));
  } else if (compare && requestedJornadas.length > 1) {
    groups = requestedJornadas.map((journey) => candidates.filter((candidate) => matchesSegment(candidate, text, { kind: "journey", value: journey })));
  } else if (compare && roleAnchors.length > 1) {
    groups = roleAnchors.map((role) => candidates.filter((candidate) => matchesSegment(candidate, text, { kind: "role", value: role })));
  } else if (compare && requestedLevels.length > 1) {
    groups = Array.from(new Set(requestedLevels.map((level) => level.replace(/o$/, "a")))).map((level) => candidates.filter((candidate) => matchesSegment(candidate, text, { kind: "level", value: level })));
  } else if (compare && requestedInstitutions.length > 1) {
    groups = requestedInstitutions.map((institution) => candidates.filter((candidate) => matchesSegment(candidate, text, { kind: "institution", value: institution })));
  }

  const uniqueCandidates = (items: NomenclatorCandidate[]) => Array.from(new Map(items.map((item) => [String(item.record.id || item.chunk.id), item])).values());
  if (groups.length >= 2) {
    groups = groups.map(uniqueCandidates);
    if (groups.some((group) => group.length !== 1)) {
      const alternatives = uniqueCandidates(groups.flat());
      return { answer: "No pude identificar de manera unívoca cada cargo de la comparación. Aclará cargo, institución, nivel o jornada.", chunks: alternatives.map((item) => item.chunk), clarification: true };
    }
    const selected = groups.map((group) => group[0]);
    const label = (item: NomenclatorCandidate) => `${item.record.categoria || item.record.titulo_articulo || "Cargo"} (${item.chunk.referencia})`;
    const metrics = [
      { label: "Horas reloj semanales", keys: ["horas_reloj_semanales"], unit: "" },
      { label: "Horas cátedra semanales", keys: ["horas_catedra_semanales", "horas_catedra"], unit: "" },
      { label: "Minutos reloj semanales", keys: ["minutos_reloj_semanales", "minutos_reloj"], unit: "" },
      { label: "Puntos salariales", keys: ["puntos_salariales", "puntos_nomenclador", "puntaje_salarial", "puntos"], unit: "" },
      { label: "Puntos de incompatibilidad", keys: ["puntaje_incompatibilidad", "puntos_incompatibilidad", "puntaje_compatibilidad"], unit: "" },
    ];
    const comparisons = metrics.map((metric) => {
      const values = selected.map((item) => getNomenclatorNumber(item.record, metric.keys));
      if (values.length !== 2 || values.some((value) => value === null)) return `${metric.label}: no hay datos compatibles en todos los registros para comparar.`;
      const difference = Math.abs(values[0]! - values[1]!);
      if (values[0] === values[1]) return `${metric.label}: mismo valor (${values[0]}).`;
      const higher = values[0]! > values[1]! ? 0 : 1;
      return `${metric.label}: ${label(selected[higher])} registra ${difference} más que ${label(selected[1 - higher])}.`;
    });
    const answer = selected.map((item) => `${label(item)}\n${renderNomenclatorRecord(item.chunk, item.record)}`).join("\n\n") + `\n\nDiferencias registradas:\n${comparisons.join("\n")}`;
    return { answer, chunks: selected.map((item) => item.chunk) };
  }

  const selected = uniqueCandidates(candidates.filter((candidate) => matchesSegment(candidate, pregunta)));
  if (selected.length === 1) return { answer: renderNomenclatorRecord(selected[0].chunk, selected[0].record), chunks: [selected[0].chunk] };
  if (selected.length > 1) {
    return { answer: `Encontré ${selected.length} registros compatibles; necesito que aclares cargo, jornada, nivel, modalidad o institución.\n${selected.map(({ chunk, record }) => `- ${renderNomenclatorRecord(chunk, record)}`).join("\n")}`, chunks: selected.map(({ chunk }) => chunk), clarification: true };
  }
  return null;
}

function toSearchHits(
  chunks: LocalChunk[],
  pregunta: string,
  dominio: DominioBackend
): SearchHit[] {
  return chunks.map((chunk) => ({
    filename: chunk.filename,
    score: scoreChunk(chunk, pregunta, dominio),
    text: truncateText(chunk.text, 1200),
  }));
}

function toArticulos(chunks: LocalChunk[]): BackendArticulo[] {
  return chunks.map((chunk) => ({
    articulo: chunk.articulo,
    titulo_articulo: chunk.descripcion || chunk.titulo,
    resumen: truncateText(
      chunk.descripcion || chunk.titulo || chunk.text,
      chunk.descripcion || chunk.titulo ? 320 : 420,
    ),
    texto: truncateText(chunk.text, 1800),
    fuente: chunk.referencia,
    otorgante: chunk.otorgante,
    interviniente: chunk.interviniente,
    seccion: chunk.seccion,
    nivel: chunk.nivel,
    tipo: chunk.tipo,
  }));
}

function getGroqClient(): OpenAI | null {
  if (workflowTestDependencies.groqClient !== undefined) return workflowTestDependencies.groqClient;
  const apiKey = getEnv("GROQ_API_KEY");

  if (!apiKey) return null;

  return new OpenAI({
    apiKey,
    baseURL: getEnv("GROQ_BASE_URL") || "https://api.groq.com/openai/v1",
    timeout: 10000,
    maxRetries: 0,
  });
}

function buildGroqSystemPrompt(): string {
  return [
    "Sos el Asistente Virtual de SiDCa. Respondé en español argentino claro y directo, usando exclusivamente los fragmentos entregados. No inventes normas, requisitos, plazos, autoridades, trámites, contactos ni datos faltantes; si falta respaldo, explicitalo. No uses conocimiento externo ni menciones proveedores o tecnología. Preferí respuestas breves y citá la norma/artículo respaldados.",
    "Para trámites, indicá pasos solo si están documentados; si no, reconocé la limitación y recomendá consultar a SiDCa. En servicios, priorizá respuesta_sugerida, ubicación, contactos y enlaces del fragmento, sin alterar números ni mezclar áreas.",
    "Licencias: razones particulares = art. 52; asuntos particulares = art. 45, no son intercambiables. Corto tratamiento: arts. 16-17; largo: 20-22; familiar: 30; maternidad: 27; duelo: 48; clima: 49; capacitación: 34/41; exámenes: 35; superposición: 53-54. Respetá unidades y condiciones literales; no calcules derechos individuales sin datos suficientes.",
    "Inasistencias injustificadas: distinguí art. 85 (cinco o más días laborales consecutivos) de art. 86 (diez no consecutivos en el año); considerá descuento del art. 80 y sanciones del 84 si aparecen. Si no se indica continuidad, explicá ambos escenarios. No afirmes que una solicitud individual es requisito si la norma no lo dice.",
    "Para nomencladores, mantené separados horas reloj, horas cátedra, minutos, puntos salariales e incompatibilidad; no combines instituciones, jornadas ni registros POF. Si la consulta es ambigua, pedí precisión. No confundas suplente, interino y titular; para niveles, priorizá el anexo pertinente. No presentes texto derogado/sustituido como vigente ni reconstruyas fragmentos truncados.",
  ].join(" ");
}

function parseProviderDelaySeconds(value: unknown): number {
  if (typeof value === "number" && Number.isFinite(value) && value > 0) return value;
  const text = String(value || "").trim();
  if (!text) return 0;
  const numeric = Number(text);
  if (Number.isFinite(numeric) && numeric > 0) return numeric;
  const date = Date.parse(text);
  if (Number.isFinite(date)) return Math.max(0, (date - Date.now()) / 1000);
  let seconds = 0;
  for (const match of text.matchAll(/(\d+(?:\.\d+)?)\s*(d|h|m|s)/gi)) {
    const value = Number(match[1]);
    const unit = match[2].toLowerCase();
    seconds += value * (unit === "d" ? 86400 : unit === "h" ? 3600 : unit === "m" ? 60 : 1);
  }
  return seconds;
}

function isComplexRetrievalQuery(pregunta: string): boolean {
  const text = normalizeText(pregunta);
  return /\b(?:comparar|comparacion|compara|comparan|comparando|comparado|diferencia|diferencias|relacion|entre|versus|contra|modificar|modifica|modificacion|modificaciones|reemplazar|reemplaza|reemplazo|reemplazado|ademas|por un lado|por otro lado)\b/.test(text) ||
    parseNormativeReferences(pregunta).length > 1 || extractArticleNumbersFromQuery(pregunta).length > 1;
}

async function acquireGroqQueueSlot(): Promise<() => void> {
  if (groqQueueDepth >= GROQ_MAX_QUEUED_REQUESTS) {
    const error: any = new Error("La cola local de Groq está saturada.");
    error.code = "provider_queue_saturated";
    throw error;
  }
  const previous = groqQueue;
  let release!: () => void;
  groqQueue = new Promise<void>((resolve) => { release = resolve; });
  groqQueueDepth++;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const acquired = await Promise.race([
    previous.then(() => true, () => true),
    new Promise<boolean>((resolve) => { timer = setTimeout(() => resolve(false), GROQ_QUEUE_WAIT_TIMEOUT_MS); }),
  ]);
  if (timer) clearTimeout(timer);
  if (!acquired) {
    groqQueueDepth--;
    // Mantiene el orden de los siguientes turnos, sin bloquear esta solicitud.
    void previous.then(release, release);
    const error: any = new Error("Se agotó el tiempo máximo de espera en la cola de Groq.");
    error.code = "provider_queue_timeout";
    throw error;
  }
  groqQueueDepth--;
  return release;
}

type GroqGeneration = {
  text: string;
  finishReason: string;
  completionTokens: number | null;
  status: number;
};

function respuestaPareceIncompleta(
  text: string,
  finishReason: string,
): boolean {
  const clean = text.trim();

  if (!clean || finishReason === "length") return true;

  const markdownMarkers = (clean.match(/\*\*|`/g) || []).length;
  if (markdownMarkers % 2 !== 0) return true;

  if (/(?:^|\s)(?:artículo|articulo|art\.)$/i.test(clean)) return true;
  if (/[,:;]\s*$/.test(clean)) return true;

  return false;
}

async function solicitarGeneracionGroq(
  client: OpenAI,
  model: string,
  maxTokens: number,
  pregunta: string,
  dominio: DominioBackend,
  context: string,
  referencias: string[],
  retryInstruction = "",
): Promise<GroqGeneration> {
  const reasoningOptions = model.startsWith("openai/gpt-oss-")
    ? {
        reasoning_effort: "low",
        include_reasoning: false,
      }
    : {};

  const requestBody: any = {
    model,
    temperature: 0.12,
    max_tokens: maxTokens,
    ...reasoningOptions,
    messages: [
      {
        role: "system",
        content: buildGroqSystemPrompt(),
      },
      {
        role: "user",
        content:
          `Dominio detectado: ${dominio}\n\n` +
          `Consulta del docente:\n${pregunta}\n\n` +
          `Fragmentos recuperados:\n\n${context}\n\n` +
          `Fuentes disponibles para citar al final: ${referencias.join(" | ")}\n\n` +
          "Redactá una respuesta clara, precisa y directa. No uses información externa a los fragmentos." +
          retryInstruction,
      },
    ],
  };
  const response: any = await client.chat.completions.create(requestBody);

  const content = response?.choices?.[0]?.message?.content;
  const finishReason = String(response?.choices?.[0]?.finish_reason || "unknown");
  const completionTokens = Number.isFinite(response?.usage?.completion_tokens)
    ? Number(response.usage.completion_tokens)
    : null;
  const promptTokens = Number.isFinite(response?.usage?.prompt_tokens)
    ? Number(response.usage.prompt_tokens)
    : null;
  const totalTokens = Number.isFinite(response?.usage?.total_tokens)
    ? Number(response.usage.total_tokens)
    : null;
  const text = typeof content === "string" ? content.trim() : "";

  console.log("[groqWorkflow] usage", {
    promptTokens,
    finishReason,
    completionTokens,
    totalTokens,
  });

  if (!text) {
    throw new Error("Groq no devolvió contenido utilizable.");
  }

  return {
    text,
    finishReason,
    completionTokens,
    status: Number(response?.status || 200),
  };
}

async function askGroq(
  pregunta: string,
  dominio: DominioBackend,
  chunks: LocalChunk[]
): Promise<GroqGeneration> {
  if (groqCooldownUntil > workflowNow()) {
    const error: any = new Error(`Groq en período de enfriamiento: ${groqCooldownReason || "rate_limit"}`);
    error.status = 429;
    error.cooldownUntil = groqCooldownUntil;
    throw error;
  }

  const client = getGroqClient();

  if (!client) {
    throw new Error("Falta GROQ_API_KEY.");
  }

  const model = getEnv("GROQ_MODEL") || "openai/gpt-oss-20b";
  const configuredMaxTokens = Number(getEnv("GROQ_MAX_TOKENS") || 900);
  const complexQuery = isComplexRetrievalQuery(pregunta);
  const maxTokens = complexQuery ? Math.min(Math.max(configuredMaxTokens, 1200), 1800) : Math.min(configuredMaxTokens, 600);
  const context = buildContext(chunks);
  const referencias = Array.from(new Set(chunks.map((chunk) => chunk.referencia)));
  const providerStartedAt = Date.now();
  const estimatedInputTokens = Math.ceil((buildGroqSystemPrompt().length + context.length + pregunta.length + referencias.join(" | ").length + 400) / 4);
  const estimatedCombinedTokens = estimatedInputTokens + maxTokens;
  console.log("[groqWorkflow] provider_request", { dominio, model, fragmentos: chunks.length,
    maxTokens, estimatedInputTokens, estimatedCombinedTokens, estimatedBudget: GROQ_ESTIMATED_TPM_BUDGET, estimatedBy: "chars_div_4" });
  if (estimatedCombinedTokens > GROQ_ESTIMATED_TPM_BUDGET) {
    const error: any = new Error("El contexto completo más la salida solicitada excede el presupuesto estimado por solicitud.");
    error.code = "context_budget_exceeded";
    throw error;
  }
  const release = await acquireGroqQueueSlot();
  try {
  if (groqCooldownUntil > workflowNow()) {
    const error: any = new Error(`Groq en período de enfriamiento: ${groqCooldownReason || "rate_limit"}`);
    error.status = 429;
    throw error;
  }
  let first: GroqGeneration;
  try {
    first = await solicitarGeneracionGroq(client, model, maxTokens, pregunta, dominio, context, referencias);
  } catch (error: any) {
    const headers = error?.headers || error?.response?.headers;
    let retryAfterSeconds = 0;
    try {
      retryAfterSeconds = parseProviderDelaySeconds(typeof headers?.get === "function" ? headers.get("retry-after") : headers?.["retry-after"] || error?.retryAfter || 0);
    } catch { retryAfterSeconds = 0; }
    if (!isRateLimitError(error) || retryAfterSeconds <= 0 || retryAfterSeconds > 1) throw error;
    const jitterMs = 100 + Math.floor(Math.random() * 250);
    console.warn("[groqWorkflow] provider_retry", { reason: "short_rate_limit", retryAfterSeconds, jitterMs, model });
    await new Promise((resolve) => setTimeout(resolve, retryAfterSeconds * 1000 + jitterMs));
    first = await solicitarGeneracionGroq(client, model, maxTokens, pregunta, dominio, context, referencias);
  }

  if (!respuestaPareceIncompleta(first.text, first.finishReason)) {
    console.log("[groqWorkflow] provider_complete", { dominio, model, duracionMs: Date.now() - providerStartedAt,
      finishReason: first.finishReason });
    return first;
  }

  console.warn("[groqWorkflow] generation_retry", {
    reason: first.finishReason === "length" ? "max_tokens" : "incomplete_text",
  });

  const retryMaxTokens = Math.min(Math.max(maxTokens + 300, Math.ceil(maxTokens * 1.5)), 1800);
  const estimatedRetryTotalTokens = estimatedCombinedTokens + estimatedInputTokens + retryMaxTokens;
  if (estimatedRetryTotalTokens > GROQ_ESTIMATED_TPM_BUDGET) {
    const error: any = new Error("El reintento por respuesta incompleta excede el presupuesto estimado acumulado.");
    error.code = "context_budget_exceeded";
    throw error;
  }
  console.log("[groqWorkflow] generation_retry_budget", { estimatedRetryTotalTokens, estimatedBudget: GROQ_ESTIMATED_TPM_BUDGET, estimatedBy: "chars_div_4" });
  const retry = await solicitarGeneracionGroq(
    client,
    model,
    retryMaxTokens,
    pregunta,
    dominio,
    context,
    referencias,
    " Respondé de forma completa y cerrá todas las frases y marcas Markdown antes de terminar.",
  );

  if (respuestaPareceIncompleta(retry.text, retry.finishReason)) {
    throw new Error("Groq devolvió una respuesta incompleta.");
  }

  console.log("[groqWorkflow] provider_complete", { dominio, model, duracionMs: Date.now() - providerStartedAt,
    finishReason: retry.finishReason, retriedIncomplete: true });
  return retry;
  } catch (error: any) {
    if (isRateLimitError(error)) {
      const message = String(error?.message || "").toLowerCase();
      const providerCode = String(error?.code || error?.error?.code || error?.type || "").toLowerCase();
      const headers = error?.headers || error?.response?.headers;
      const getHeader = (name: string) => {
        try { return typeof headers?.get === "function" ? headers.get(name) : headers?.[name] ?? headers?.[name.toLowerCase()]; }
        catch { return undefined; }
      };
      const retryAfter = parseProviderDelaySeconds(getHeader("retry-after") || error?.retryAfter || 0);
      // Groq documenta las cabeceras genéricas requests=RPD y tokens=TPM; las variantes day/minute son opcionales.
      const dailyRequestRemaining = getHeader("x-ratelimit-remaining-requests-day") ?? getHeader("x-ratelimit-remaining-requests");
      const dailyTokenRemaining = getHeader("x-ratelimit-remaining-tokens-day");
      const minuteRequestRemaining = getHeader("x-ratelimit-remaining-requests-minute");
      const minuteTokenRemaining = getHeader("x-ratelimit-remaining-tokens-minute") ?? getHeader("x-ratelimit-remaining-tokens");
      const dailyRequestLimit = getHeader("x-ratelimit-limit-requests-day") ?? getHeader("x-ratelimit-limit-requests");
      const dailyTokenLimit = getHeader("x-ratelimit-limit-tokens-day");
      const explicitDaily = /(?:requests?|tokens?)\s*(?:per|\/|a)\s*day|daily|requests_per_day|tokens_per_day|quota.*day/.test(`${message} ${providerCode}`);
      const explicitMinute = /(?:requests?|tokens?)\s*(?:per|\/|a)\s*minute|rpm|tpm|requests_per_minute|tokens_per_minute/.test(`${message} ${providerCode}`);
      const dailyTokenExhausted = /tokens?\s*(?:per|\/|a)\s*day|tpd|tokens_per_day/.test(`${message} ${providerCode}`) || (dailyTokenRemaining !== undefined && dailyTokenRemaining !== null && Number(dailyTokenRemaining) <= 0);
      const dailyRequestExhausted = /requests?\s*(?:per|\/|a)\s*day|rpd|requests_per_day/.test(`${message} ${providerCode}`) || (!explicitMinute && dailyRequestRemaining !== undefined && dailyRequestRemaining !== null && Number(dailyRequestRemaining) <= 0);
      const dailyExhausted = explicitDaily || dailyTokenExhausted || dailyRequestExhausted;
      const tpmExhausted = /tokens?\s*(?:per|\/|a)\s*minute|tpm|tokens_per_minute/.test(`${message} ${providerCode}`) || (minuteTokenRemaining !== undefined && minuteTokenRemaining !== null && Number(minuteTokenRemaining) <= 0);
      const rpmExhausted = /requests?\s*(?:per|\/|a)\s*minute|rpm|requests_per_minute/.test(`${message} ${providerCode}`) || (minuteRequestRemaining !== undefined && minuteRequestRemaining !== null && Number(minuteRequestRemaining) <= 0);
      const rateKind = dailyExhausted
        ? (dailyTokenExhausted || (/token/.test(`${message} ${providerCode}`) && !dailyRequestExhausted) ? "tpd" : "rpd")
        : tpmExhausted ? "tpm" : rpmExhausted ? "rpm" : "rate_limit";
      const rateReset = dailyExhausted
        ? rateKind === "tpd"
          ? getHeader("x-ratelimit-reset-tokens-day")
          : getHeader("x-ratelimit-reset-requests-day") ?? getHeader("x-ratelimit-reset-requests")
        : tpmExhausted
          ? getHeader("x-ratelimit-reset-tokens-minute") ?? getHeader("x-ratelimit-reset-tokens")
          : getHeader("x-ratelimit-reset-requests-minute");
      const resetSeconds = parseProviderDelaySeconds(rateReset);
      const dailyLimitValue = Number(rateKind === "tpd" ? dailyTokenLimit : dailyRequestLimit);
      const providerCooldownSeconds = Math.max(retryAfter, resetSeconds);
      const safeFallbackSeconds = dailyExhausted ? (providerCooldownSeconds > 0 ? 0 : 24 * 60 * 60) : 2;
      const cooldownSeconds = Math.max(providerCooldownSeconds, safeFallbackSeconds);
      const cooldownMs = cooldownSeconds * 1000;
      groqCooldownUntil = workflowNow() + cooldownMs;
      groqCooldownReason = rateKind;
      console.warn("[groqWorkflow] provider_circuit_open", { reason: groqCooldownReason, retryAfterSeconds: Math.ceil(cooldownMs / 1000),
        limit: Number.isFinite(dailyLimitValue) ? dailyLimitValue : undefined,
        remaining: rateKind === "tpd" ? Number(dailyTokenRemaining) : rateKind === "rpd" ? Number(dailyRequestRemaining) : rateKind === "tpm" ? Number(minuteTokenRemaining) : Number(minuteRequestRemaining),
        resetSeconds: resetSeconds || undefined, providerRetryAfterSeconds: retryAfter || undefined, model });
    } else if (Number(error?.status || error?.response?.status) === 404) {
      console.warn("[groqWorkflow] provider_model_not_found", { model });
    }
    throw error;
  } finally {
    release();
  }
}

function isRateLimitError(error: any): boolean {
  const status = Number(error?.status || error?.response?.status || error?.code);
  const message = String(error?.message || "").toLowerCase();

  return status === 429 || message.includes("rate limit") || message.includes("429");
}

function extractCitedArticleRefs(text: string): string[] {
  const normalized = normalizeText(text);
  const refs = new Set<string>();
  const patterns = [
    /\b(?:articulos?|arts?)\s+([0-9]+(?:\s*(?:y|e|,)\s*[0-9]+)*)/g,
  ];

  for (const pattern of patterns) {
    for (const match of normalized.matchAll(pattern)) {
      for (const ref of String(match[1] || "").match(/\d+/g) || []) {
        refs.add(ref);
      }
    }
  }

  return Array.from(refs);
}

function citedArticlesAreSupported(
  respuesta: string,
  chunks: LocalChunk[]
): boolean {
  const cited = extractCitedArticleRefs(respuesta);
  if (!cited.length) return true;

  const available = new Set<string>();
  for (const chunk of chunks) {
    for (const ref of [
      ...(chunk.articleRefs || []),
      ...(String(chunk.articulo || "").match(/\d+/g) || []),
    ]) {
      available.add(String(ref));
    }
  }

  return cited.every((ref) => available.has(ref));
}

function citedNormsAreSupported(answer: string, chunks: LocalChunk[]): boolean {
  const citations = parseNormativeReferences(answer);
  if (!citations.length) return true;
  return citations.every((citation) => chunks.some((chunk) => chunkMatchesNormativeReference({
    sourceId: chunk.sourceId || chunk.filename, source: chunk.referencia, norm: chunk.norma || chunk.referencia,
    jurisdiction: String(chunk.metadata?.jurisdiccion || ""), metadata: chunk.metadata, text: chunk.text,
  }, citation)));
}

function citedNormArticlePairsAreSupported(answer: string, chunks: LocalChunk[]): boolean {
  const pairs: Array<{ norm: ReturnType<typeof parseNormativeReferences>[number]; article: string }> = [];
  const normFirst = /\b(?:ley(?:\s+(?:nacional|provincial))?|decreto(?:\s*[-–]\s*ley|\s+acuerdo|\s+ecyt)?|resoluci[oó]n(?:\s+(?:nacional|provincial))?)\s*(?:(?:n(?:ro)?|numero)\s*(?:[º°ª]\.?|\.\s*[º°ª]?|)?\s*)?\d[\d.]*(?:\s*\/\s*\d{2,4})?[^\n.;]{0,80}?\b(?:art(?:i|í)culo|art)\.?\s*(\d+)\b/gi;
  for (const match of answer.matchAll(normFirst)) {
    const norm = parseNormativeReferences(match[0])[0];
    const article = match[1];
    if (norm && article) pairs.push({ norm, article });
  }
  const articleFirst = /\b(?:art(?:i|í)culo|art)\.?\s*(\d+)\s+(?:de\s+)?(?:la\s+)?((?:ley(?:\s+(?:nacional|provincial))?|decreto(?:\s*[-–]\s*ley|\s+acuerdo|\s+ecyt)?|resoluci[oó]n(?:\s+(?:nacional|provincial))?)\s*(?:(?:n(?:ro)?|numero)\s*(?:[º°ª]\.?|\.\s*[º°ª]?|)?\s*)?\d[\d.]*(?:\s*\/\s*\d{2,4})?)/gi;
  for (const match of answer.matchAll(articleFirst)) {
    const norm = parseNormativeReferences(match[2])[0];
    if (norm && match[1]) pairs.push({ norm, article: match[1] });
  }
  return pairs.every(({ norm, article }) => chunks.some((chunk) => {
    const exactNorm = chunkMatchesNormativeReference({
      sourceId: chunk.sourceId || chunk.filename, source: chunk.referencia, norm: chunk.norma || chunk.referencia,
      jurisdiction: String(chunk.metadata?.jurisdiccion || ""), metadata: chunk.metadata, text: chunk.text,
    }, norm);
    const hasArticle = [...(chunk.articleRefs || []).map(String), ...(String(chunk.articulo || "").match(/\d+/g) || [])].includes(article);
    return exactNorm && hasArticle;
  }));
}

function responseCacheKey(
  dominio: DominioBackend,
  pregunta: string
): string {
  return `${cachedChunksVersion}|${dominio}|${normalizeText(pregunta)}`;
}

function getCachedResponse(key: string): ChatbotWorkflowOutput | null {
  const cached = responseCache.get(key);
  if (!cached) return null;

  if (cached.expiresAt <= workflowNow()) {
    responseCache.delete(key);
    return null;
  }

  responseCache.delete(key);
  responseCache.set(key, cached);
  console.log("[chatbot] response_cache_hit");
  return cached.response;
}

function setCachedResponse(key: string, response: ChatbotWorkflowOutput): void {
  responseCache.delete(key);
  responseCache.set(key, {
    expiresAt: workflowNow() + RESPONSE_CACHE_TTL_MS,
    response,
  });

  while (responseCache.size > RESPONSE_CACHE_MAX_ENTRIES) {
    const oldestKey = responseCache.keys().next().value;
    if (!oldestKey) break;
    responseCache.delete(oldestKey);
  }
}

function chunksForSearch(chunks: LocalChunk[]): LocalChunk[] {
  return chunks.slice(0, 8);
}

function buildOutput(params: {
  ok: boolean;
  tipo: string;
  origen: string;
  dominio: DominioBackend;
  consulta: string;
  respuesta: string;
  chunks: LocalChunk[];
  referenciasAdicionales?: string[];
  asesoramientoGremial?: BackendServiceContact;
  error?: string;
}): ChatbotWorkflowOutput {
  const referencias = Array.from(
    new Set([...params.chunks.map((chunk) => chunk.referencia), ...(params.referenciasAdicionales || [])].filter(Boolean))
  );

  return {
    ok: params.ok,
    tipo: params.tipo,
    dominio: params.dominio,
    origen: params.origen,
    consulta: params.consulta,
    consultaNormalizada: params.consulta,
    respuesta: params.respuesta,
    articulos: toArticulos(params.chunks),
    referencias,
    busqueda: toSearchHits(
      chunksForSearch(params.chunks),
      params.consulta,
      params.dominio
    ),
    conversationId: null,
    asesoramientoGremial: params.asesoramientoGremial,
    error: params.error,
  };
}

export async function runGroqWorkflow(
  input: ChatbotQueryInput
): Promise<ChatbotWorkflowOutput> {
  const workflowStartedAt = Date.now();
  const preguntaEfectiva = input.contextoConversacional
    ? `${input.pregunta}\nContexto conversacional previo del mismo usuario: ${truncateText(input.contextoConversacional, 1200)}`
    : input.pregunta;
  const explicitNormInQuestion = parseNormativeReferences(input.pregunta).length > 0;
  const asksSingleArticle = /\b(?:art(?:i|í)culo|art)\.?\s*\d+\b/i.test(input.pregunta) && !explicitNormInQuestion;
  const preliminaryDomain = classifyDomain(preguntaEfectiva, input.dominio);
  if (asksSingleArticle && !input.contextoConversacional && preliminaryDomain === "general") {
    return buildOutput({ ok: true, tipo: "aclaracion", origen: "local_rag", dominio: input.dominio || "general", consulta: input.pregunta,
      respuesta: "¿De qué norma querés consultar ese artículo? Hay varias normas documentales con artículos de igual numeración; indicá la ley o decreto para evitar una respuesta ambigua.", chunks: [] });
  }
  const dominio = explicitNormInQuestion
    ? classifyDomain(input.pregunta, input.dominio)
    : classifyDomain(preguntaEfectiva, input.dominio);
  console.log("[chatbot] request_start", { dominio, conversationContext: Boolean(input.contextoConversacional) });

  if (isStandaloneGreeting(input.pregunta)) {
    return buildOutput({ ok: true, tipo: "respuesta_local", origen: "local_rag", dominio: "general", consulta: input.pregunta,
      respuesta: "¡Hola! ¿En qué puedo ayudarte?", chunks: [] });
  }

  if (isGenericTeacherQuestion(input.pregunta)) {
    return buildOutput({
      ok: true,
      tipo: "sin_resultados",
      origen: "local_rag",
      dominio,
      consulta: input.pregunta,
      respuesta:
        "Necesito que especifiques qué querés consultar sobre el docente. Por ejemplo: licencia, estatuto, coberturas, docente suplente, docente interino, titularidad, cabecera cero o funciones del cargo.",
      chunks: [],
    });
  }

  if (isIncompleteTeacherComparisonQuestion(input.pregunta)) {
    return buildOutput({
      ok: true,
      tipo: "sin_resultados",
      origen: "local_rag",
      dominio: "coberturas",
      consulta: input.pregunta,
      respuesta:
        "Necesito que completes la consulta. ¿Querés comparar docente suplente con docente interino, docente titular u otra situación de revista?",
      chunks: [],
    });
  }

  if (isMedicalLicenseAfterAppointmentQuestion(input.pregunta)) {
    return buildOutput({
      ok: true,
      tipo: "sin_resultados",
      origen: "local_rag",
      dominio: "licencias",
      consulta: input.pregunta,
      respuesta:
        "No encontré en los fragmentos cargados una regla que establezca una cantidad mínima de días desde haber optado, tomado o sido designado en un cargo para solicitar licencia médica. La base sí contiene licencias médicas o de salud, como corto tratamiento, largo tratamiento y accidentes o enfermedades profesionales, pero no se encontró un plazo específico vinculado a la fecha de opción o designación del cargo. Fuente consultada: Decreto Acuerdo Nº 1092/2015 – Régimen de Licencias Docentes.",
      chunks: [],
    });
  }

  if (dominio === "licencias" && isAmbiguousParticularLicenseQuestion(input.pregunta)) {
    return buildOutput({ ok: true, tipo: "aclaracion", origen: "local_rag", dominio,
      consulta: input.pregunta,
      respuesta: "¿Te referís a la licencia por razones particulares del artículo 52 o a la licencia por asuntos particulares del artículo 45? Son figuras distintas y tienen condiciones diferentes.",
      chunks: [] });
  }

  if (dominio === "licencias" && isGenericLicenseQuestion(input.pregunta)) {
    return buildOutput({
      ok: true,
      tipo: "sin_resultados",
      origen: "local_rag",
      dominio,
      consulta: input.pregunta,
      respuesta:
        "Necesito que especifiques qué tipo de licencia querés consultar. Por ejemplo: maternidad, corto tratamiento, largo tratamiento, grupo familiar, capacitación, superposición horaria, razones climáticas, examen o razones particulares.",
      chunks: [],
    });
  }

  try {
    const documentos = await loadLocalChunks();
    const cacheKey = responseCacheKey(dominio, preguntaEfectiva);
    const useResponseCache = !input.contextoConversacional;
    const cachedResponse = useResponseCache ? getCachedResponse(cacheKey) : null;

    if (cachedResponse) {
      return cachedResponse;
    }

    const isComplexRetrieval = isComplexRetrievalQuery(input.pregunta);
    const adaptiveMaxResults = input.maxResults ?? (isComplexRetrieval ? 5 : 3);
    const searchInput = { ...input, maxResults: adaptiveMaxResults, pregunta: preguntaEfectiva };
    const chunks = await searchLocalFragments(searchInput, dominio, documentos);
    console.log("[chatbot] retrieval_complete", { dominio, motor: "local_rag", fragmentos: chunks.length,
      corpusFingerprint: cachedChunksVersion.slice(0, 12), duracionMs: Date.now() - workflowStartedAt });

    if (dominio === "nomenclador") {
      const resolved = resolveNomenclatorQuery(input.pregunta, documentos);
      if (resolved) {
        const response = buildOutput({ ok: true, tipo: resolved.clarification ? "aclaracion" : "respuesta_local",
          origen: "local_rag", dominio, consulta: input.pregunta, respuesta: resolved.answer, chunks: resolved.chunks });
        return response;
      }
    }

    const requestedArticles = extractArticleNumbersFromQuery(input.pregunta);
    const requestedArticle = requestedArticles.length === 1 ? requestedArticles[0] : null;
    const asksPrivateDecree = /3387/.test(normalizeText(input.pregunta));
    const missingControlChunk = asksPrivateDecree && requestedArticle && documentos.find((chunk) => {
      const control: any = chunk.metadata?.control_documental;
      const refs = Array.isArray(control?.articulos_faltantes) ? control.articulos_faltantes.map(String) : [];
      return refs.includes(requestedArticle) && /3387/.test(normalizeText(`${chunk.referencia} ${chunk.filename}`));
    });
    if (missingControlChunk) {
      return buildOutput({ ok: true, tipo: "sin_resultados", origen: "local_rag", dominio, consulta: input.pregunta,
        respuesta: `El artículo ${requestedArticle} del Decreto-Ley 3387 no está disponible en la fuente documental. El control de integridad indica que los artículos 33 a 75 están ausentes y que no deben completarse por inferencia.`, chunks: [missingControlChunk] });
    }

    if (requestedArticles.length > 1) {
      const coveredArticles = new Set(chunks.filter((chunk) => chunk.kind === "articulo")
        .flatMap((chunk) => requestedArticles.filter((article) => articleMatches(chunk, article))));
      const missingArticles = requestedArticles.filter((article) => !coveredArticles.has(article));
      if (missingArticles.length) {
        return buildOutput({ ok: true, tipo: "sin_resultados", origen: "local_rag", dominio, consulta: input.pregunta,
          respuesta: `La documentación activa no contiene todos los artículos solicitados (${missingArticles.join(", ")}). No voy a responder la comparación como completa ni completar el texto faltante.`, chunks });
      }
    }

    if (!chunks.length) {
      const questionNumbers = (preguntaEfectiva.match(/\d+/g) || []).map((value) => value.replace(/\D/g, ""));
      const compactQuestionDigits = preguntaEfectiva.replace(/\D/g, "");
      const inactive = cachedInactiveSourceLabels.find((label) => {
        const normalizedLabel = normalizeText(label).replace(/[^a-z0-9 ]/g, " ");
        const nums = normalizedLabel.match(/\d+/g) || [];
        return nums.some((number) => {
          if (number.length < 4) return false;
          if (compactQuestionDigits.includes(number)) return true;
          return questionNumbers.some((asked) => asked.length >= 4 && (asked.includes(number) || number.includes(asked)));
        });
      });
      if (inactive) {
        return buildOutput({ ok: true, tipo: "sin_resultados", origen: "local_rag", dominio, consulta: input.pregunta,
          respuesta: `La norma solicitada figura en el índice documental, pero su fuente está inactiva y no se utiliza para responder. No hay fragmentos habilitados suficientes para informar su contenido.`, chunks: [] });
      }
      const asesoramientoGremial = getAsesoramientoGremialContact(
        documentos
      );

      return buildOutput({
        ok: true,
        tipo: "sin_resultados",
        origen: "local_rag",
        dominio,
        consulta: input.pregunta,
        respuesta: NO_ENCONTRADO,
        chunks: [],
        asesoramientoGremial,
      });
    }

    const nonCurrentChunks = chunks.filter((chunk) => {
      const update: any = chunk.metadata?.actualizacion;
      const status = normalizeText(String(update?.estado || chunk.metadata?.estado_vigencia || ""));
      return requestedArticles.some((article) => articleMatches(chunk, article)) && /derogado|sustituido/.test(status);
    });
    const nonCurrent = requestedArticles.length === 1 ? nonCurrentChunks[0] : undefined;
    if (nonCurrent) {
      const update: any = nonCurrent.metadata?.actualizacion;
      const status = normalizeText(String(update?.estado || nonCurrent.metadata?.estado_vigencia || ""));
      const summary = String(update?.resumen_no_literal || "").trim();
      return buildOutput({ ok: true, tipo: "respuesta_local", origen: "local_rag", dominio, consulta: input.pregunta,
        respuesta: `El artículo solicitado se conserva en la fuente como texto histórico y figura como ${status}. No corresponde presentarlo como disposición vigente. ${update?.norma ? `La ficha atribuye esta actualización a ${update.norma}.` : ""} ${summary ? `La fuente incluye este resumen de actualización, no una transcripción oficial: ${summary}` : ""}`.trim(), chunks: [nonCurrent] });
    }
    if (requestedArticles.length > 1 && nonCurrentChunks.length) {
      const notices = nonCurrentChunks.map((chunk) => {
        const update: any = chunk.metadata?.actualizacion;
        const status = normalizeText(String(update?.estado || chunk.metadata?.estado_vigencia || "no verificado"));
        return `Artículo ${chunk.articulo}: figura como ${status}${update?.norma ? ` según ${update.norma}` : ""}.`;
      });
      return buildOutput({ ok: true, tipo: "respuesta_local", origen: "local_rag", dominio, consulta: input.pregunta,
        respuesta: `La comparación incluye artículos que la fuente marca como no vigentes; no los presentaré como vigentes ni compararé su texto histórico con disposiciones actuales. ${notices.join(" ")} Consultá por separado los artículos vigentes que quieras analizar.`, chunks: nonCurrentChunks });
    }

    const truncated = requestedArticles.includes("32") && /3387/.test(normalizeText(input.pregunta))
      ? chunks.find((chunk) => articleMatches(chunk, "32") && /articulo 32 truncado/.test(normalizeText(String((chunk.metadata?.control_documental as any)?.contenido || ""))))
      : undefined;
    if (truncated) {
      return buildOutput({ ok: true, tipo: "respuesta_local", origen: "local_rag", dominio, consulta: input.pregunta,
        respuesta: "La fuente del Decreto-Ley 3387 advierte que el artículo 32 está truncado. El fragmento disponible no es íntegro; no puedo reconstruir el texto faltante. La advertencia es un control documental, no parte del texto normativo.", chunks: [truncated] });
    }

    if (dominio === "servicios") {
      const documented = buildDocumentedServiceAnswer(chunks, input.pregunta);
      if (documented) {
        const serviceChunk = documented.chunk;
        const response = buildOutput({ ok: true, tipo: "respuesta_local", origen: "local_rag", dominio,
          consulta: input.pregunta, respuesta: documented.answer, chunks: serviceChunk ? [serviceChunk] : [],
          referenciasAdicionales: serviceChunk ? [serviceChunk.titulo || getObjectField(serviceChunk.raw, ["titulo_articulo", "categoria", "titulo"])].filter((reference): reference is string => Boolean(reference)) : [] });
        if (useResponseCache) setCachedResponse(cacheKey, response);
        return response;
      }
    }

    const requestedArticleForLocalAnswer = extractArticleNumbersFromQuery(input.pregunta);
    const mappedLicenseArticle = dominio === "licencias" ? getSpecificLicenseArticle(input.pregunta) : null;
    const deterministicLicenseArticle = dominio === "licencias" && ["45", "52"].includes(mappedLicenseArticle || "");
    const literalArticleRequest = /\b(?:texto literal|texto completo|transcrib(?:i|ime)|copi(?:a|ame|a)(?:\s+el)?\s+art[ií]culo)\b/i.test(normalizeText(input.pregunta));
    const explicitArticleQuery = deterministicLicenseArticle || (requestedArticleForLocalAnswer.length === 1 && literalArticleRequest &&
      (explicitNormInQuestion || dominio === "licencias"));
    if (explicitArticleQuery) {
      const targetArticle = mappedLicenseArticle || requestedArticleForLocalAnswer[0];
      const exactArticleChunk = chunks.find((chunk) => chunk.kind === "articulo" && targetArticle && articleMatches(chunk, targetArticle));
      if (exactArticleChunk) {
        const response = buildOutput({ ok: true, tipo: "respuesta_local", origen: "local_rag", dominio,
          consulta: input.pregunta, respuesta: buildVerbatimArticleAnswer(exactArticleChunk), chunks: [exactArticleChunk] });
        if (useResponseCache) setCachedResponse(cacheKey, response);
        return response;
      }
    }

    try {
      const generacionGroq = await askGroq(preguntaEfectiva, dominio, chunks);

      const supportedArticles = citedArticlesAreSupported(generacionGroq.text, chunks);
      const supportedNorms = citedNormsAreSupported(generacionGroq.text, chunks);
      const supportedPairs = citedNormArticlePairsAreSupported(generacionGroq.text, chunks);
      if (!supportedArticles || !supportedNorms || !supportedPairs) {
        console.warn("[groqWorkflow] unsupported_normative_citation_rejected");
        return buildOutput({
          ok: true,
          tipo: "respuesta_local",
          origen: "local_rag_fallback",
          dominio,
          consulta: input.pregunta,
          respuesta: buildLocalFallbackAnswer(chunks, input.pregunta),
          chunks,
          error: "unsupported_normative_citation_rejected",
        });
      }

      const response = buildOutput({
        ok: true,
        tipo: "respuesta_ia",
        origen: "groq_rag",
        dominio,
        consulta: input.pregunta,
        respuesta: generacionGroq.text,
        chunks,
      });

      if (useResponseCache) setCachedResponse(cacheKey, response);
      return response;
    } catch (groqError: any) {
      const providerErrorCode = String(groqError?.code || "");
      const localAnswer = providerErrorCode === "context_budget_exceeded"
        ? "Los fragmentos completos necesarios exceden el presupuesto estimado para una consulta segura. No recorté artículos ni sus excepciones. Dividí la consulta por artículo o grupo de normas para conservar el texto íntegro."
        : buildLocalFallbackAnswer(chunks, input.pregunta);
      const rateLimit = isRateLimitError(groqError);

      console.warn(
        "[groqWorkflow] provider_unavailable",
        { code: rateLimit ? "provider_rate_limit" : "provider_error" },
      );

      return buildOutput({
        ok: true,
        tipo: "respuesta_local",
        origen: rateLimit ? "local_rag_fallback_429" : "local_rag_fallback",
        dominio,
        consulta: input.pregunta,
        respuesta: localAnswer,
        chunks,
        error: rateLimit ? "provider_rate_limit" : providerErrorCode || "provider_error",
      });
    }
  } catch {
    console.error("[groqWorkflow] workflow_error", {
      code: "workflow_error",
    });

    return buildOutput({
      ok: false,
      tipo: "error",
      origen: "groq_rag",
      dominio,
      consulta: input.pregunta,
      respuesta: SERVICIO_NO_DISPONIBLE,
      chunks: [],
      error: "workflow_error",
    });
  }
}
