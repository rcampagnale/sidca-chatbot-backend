import OpenAI from "openai";
import { readFile } from "node:fs/promises";

export type DominioBackend =
  | "licencias"
  | "estatuto"
  | "general"
  | "coberturas";

export type ChatbotQueryInput = {
  pregunta: string;
  dominio?: DominioBackend;
  maxResults?: number;
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
  [key: string]: unknown;
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
  error?: string;
};

type SourceConfig = {
  dominio: DominioBackend;
  source: string;
  filename: string;
  referencia: string;
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

let cachedChunks: LocalChunk[] | null = null;
let cachedAt = 0;

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
      "enfermedad comun",
      "enfermedad común",
      "afeccion comun",
      "afección común",
      "operacion menor",
      "operación menor",
    ],
    terms: [
      "afecciones de corto tratamiento enfermedad comun operaciones quirurgicas menores veinticinco dias",
    ],
    preferredArticles: ["16", "17"],
  },
  {
    dominio: "licencias",
    triggers: [
      "enfermedad hora",
      "me retire enfermo",
      "retirarme enfermo",
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
      "enfermedad grave",
      "junta medica",
      "junta médica",
      "complejo diagnostico",
      "complejo diagnóstico",
    ],
    terms: [
      "afecciones lesiones largo tratamiento junta medica un año seis meses haberes",
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
    ],
    terms: [
      "maternidad embarazo parto nacimiento multiple prematuro defuncion fetal ciento veinte dias",
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
      "atencion grupo familiar enfermo accidentado quince dias cuarenta cinco dias con goce haberes",
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
      "final",
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
      "clima",
      "lluvia",
      "temporal",
      "tormenta",
      "meteorologico",
      "meteorologicos",
      "meteorológica",
      "meteorológicas",
      "fenomeno meteorologico",
      "fenómeno meteorológico",
      "fuerza mayor",
      "razones climaticas",
      "razones climáticas",
    ],
    terms: [
      "razones extraordinarias fenomenos meteorologicos fuerza mayor inasistencia justificacion articulo 49",
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
      "superposicion de horarios",
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

function classifyDomain(
  pregunta: string,
  dominio?: DominioBackend
): DominioBackend {
  if (
    dominio === "licencias" ||
    dominio === "estatuto" ||
    dominio === "general" ||
    dominio === "coberturas"
  ) {
    return dominio;
  }

  const text = normalizeText(pregunta);

  if (text.includes("estatuto") || text.includes("ley 3122")) {
    return "estatuto";
  }

  if (
    text.includes("cobertura") ||
    text.includes("asamblea") ||
    text.includes("cabecera") ||
    text.includes("vacante") ||
    text.includes("lom") ||
    text.includes("fua")
  ) {
    return "coberturas";
  }

  if (
    text.includes("licencia") ||
    text.includes("licencias") ||
    text.includes("inasistencia") ||
    text.includes("justificacion") ||
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
      "cargo",
      "cargos",
      "horas",
      "catedra",
      "interino",
      "interinato",
      "suplente",
      "suplencia",
      "cabecera",
      "lom",
      "listado",
      "vacante",
      "vacantes",
      "fua",
      "opcion",
      "destino",
      "renuncia",
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

function getSourceConfigs(): SourceConfig[] {
  const licenciasUrl =
    getEnv("SIDCA_DOCS_LICENCIAS_URL") || DEFAULT_LICENCIAS_URL;

  const estatutoUrl =
    getEnv("SIDCA_DOCS_ESTATUTO_URL") || DEFAULT_ESTATUTO_URL;

  const generalUrl = getEnv("SIDCA_DOCS_GENERAL_URL") || DEFAULT_GENERAL_URL;

  const coberturasUrl = getEnv("SIDCA_DOCS_COBERTURAS_URL");

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
  ];

  if (coberturasUrl) {
    sources.push({
      dominio: "coberturas",
      source: coberturasUrl,
      filename: "coberturas_docentes.json",
      referencia: "Normativa de coberturas docentes",
    });
  }

  return sources.filter((s) => Boolean(s.source));
}

async function loadJsonFromSource(source: string): Promise<unknown> {
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
    kind: articulo ? "articulo" : "documento",
    dominio: source.dominio,
    filename: source.filename,
    source: source.source,
    referencia: source.referencia,
    articulo,
    articleRefs,
    titulo: prefix && titulo ? `${prefix} - ${titulo}` : titulo || prefix || undefined,
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

  const text = [
    prefix,
    articulo ? `Artículo/s: ${articulo}` : "",
    descripcion ? `Descripción: ${descripcion}` : "",
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
    text,
    raw: row,
  };
}

function extractChunksFromJson(data: unknown, source: SourceConfig): LocalChunk[] {
  if (Array.isArray(data)) {
    const chunks: LocalChunk[] = [];

    data.forEach((item, index) => {
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

async function loadLocalChunks(): Promise<LocalChunk[]> {
  const ttl = Number(getEnv("LOCAL_RAG_CACHE_TTL_MS") || 5 * 60 * 1000);
  const now = Date.now();

  if (cachedChunks && now - cachedAt < ttl) {
    return cachedChunks;
  }

  const sources = getSourceConfigs();
  const chunks: LocalChunk[] = [];

  for (const source of sources) {
    try {
      const json = await loadJsonFromSource(source.source);
      chunks.push(...extractChunksFromJson(json, source));
    } catch (error: any) {
      console.warn(
        `[groqWorkflow] No se pudo cargar fuente ${source.filename}:`,
        error?.message || error
      );
    }
  }

  cachedChunks = chunks;
  cachedAt = now;

  return chunks;
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
      ? "asamblea publica cobertura cargos horas catedra interinato suplencia"
      : "consulta general sindicato afiliado tramite formulario beneficio";

  return `${pregunta} ${domainBase} ${expandedTerms}`.trim();
}

function extractArticleNumbersFromQuery(pregunta: string): string[] {
  const normalized = normalizeText(pregunta);
  const results = new Set<string>();

  const regexes = [/articulo\s+(\d+)/g, /art\s+(\d+)/g, /art\s*(\d+)/g];

  for (const regex of regexes) {
    let match: RegExpExecArray | null;

    while ((match = regex.exec(normalized)) !== null) {
      if (match[1]) results.add(match[1]);
    }
  }

  return Array.from(results);
}

function articleMatches(chunk: LocalChunk, articleNumber: string): boolean {
  if (!articleNumber) return false;

  const article = normalizeText(chunk.articulo || "");
  const refs = chunk.articleRefs.map((ref) => normalizeText(ref));

  return article === articleNumber || refs.includes(articleNumber);
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

function scoreChunk(
  chunk: LocalChunk,
  pregunta: string,
  dominio: DominioBackend
): number {
  if (chunk.dominio !== dominio) return -1000;

  const expanded = expandQuestion(pregunta, dominio);
  const queryTokens = tokenize(expanded);
  const preguntaTokens = tokenize(pregunta);
  const expansions = getMatchedExpansions(pregunta, dominio);

  const normalizedPregunta = normalizeText(pregunta);
  const normalizedChunk = normalizeText(chunk.text);
  const normalizedTitle = normalizeText(chunk.titulo || "");
  const normalizedDescription = normalizeText(chunk.descripcion || "");

  const chunkSearchText = normalizeText(
    [
      chunk.filename,
      chunk.titulo,
      chunk.descripcion,
      chunk.articulo,
      chunk.otorgante,
      chunk.interviniente,
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

async function searchLocalFragments(
  input: ChatbotQueryInput,
  dominio: DominioBackend
): Promise<LocalChunk[]> {
  const chunks = await loadLocalChunks();
  const maxResults = Math.min(Math.max(input.maxResults ?? 5, 1), 8);
  const matchedExpansions = getMatchedExpansions(input.pregunta, dominio);
  const preferredArticles = Array.from(
    new Set(matchedExpansions.flatMap((item) => item.preferredArticles || []))
  );

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
      const parts = [
        `### Fragmento ${index + 1}`,
        `Archivo: ${chunk.filename}`,
        `Tipo: ${chunk.kind}`,
        `Fuente: ${chunk.referencia}`,
        chunk.articulo ? `Artículo/s: ${chunk.articulo}` : "",
        chunk.titulo ? `Título: ${chunk.titulo}` : "",
        chunk.descripcion ? `Descripción: ${chunk.descripcion}` : "",
        chunk.otorgante ? `Funcionario otorgante: ${chunk.otorgante}` : "",
        chunk.interviniente
          ? `Funcionario interviniente: ${chunk.interviniente}`
          : "",
        `Contenido:\n${truncateText(chunk.text, 2400)}`,
      ];

      return parts.filter(Boolean).join("\n");
    })
    .join("\n\n");
}

function buildLocalFallbackAnswer(chunks: LocalChunk[]): string {
  if (!chunks.length) return NO_ENCONTRADO;

  const main = chunks
    .slice(0, 3)
    .map((chunk) => {
      const title = chunk.articulo
        ? `Artículo ${chunk.articulo}${
            chunk.descripcion ? ` – ${chunk.descripcion}` : ""
          }`
        : chunk.descripcion || chunk.titulo || "Fragmento encontrado";

      const extra = [
        chunk.otorgante ? `Funcionario otorgante: ${chunk.otorgante}` : "",
        chunk.interviniente
          ? `Funcionario interviniente: ${chunk.interviniente}`
          : "",
      ]
        .filter(Boolean)
        .join("\n");

      return `${title}:\n${truncateText(chunk.text, 1200)}${
        extra ? `\n${extra}` : ""
      }`;
    })
    .join("\n\n");

  const referencias = Array.from(new Set(chunks.map((chunk) => chunk.referencia)));

  return [
    "No pude generar la respuesta con Groq en este momento, pero encontré información local relacionada:",
    "",
    main,
    "",
    `Fuente consultada: ${referencias.join(" | ")}`,
  ].join("\n");
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
    texto: truncateText(chunk.text, 1800),
    fuente: chunk.referencia,
    otorgante: chunk.otorgante,
    interviniente: chunk.interviniente,
  }));
}

function getGroqClient(): OpenAI | null {
  const apiKey = getEnv("GROQ_API_KEY");

  if (!apiKey) return null;

  return new OpenAI({
    apiKey,
    baseURL: getEnv("GROQ_BASE_URL") || "https://api.groq.com/openai/v1",
  });
}

function buildGroqSystemPrompt(): string {
  return [
    "Sos el Asistente Virtual de SiDCa.",
    "Respondé únicamente con la información incluida en los fragmentos normativos proporcionados.",
    "No inventes artículos, plazos, requisitos, autoridades ni procedimientos.",
    "Si la información no está en los fragmentos, indicá que no se encontró información suficiente.",
    "Usá lenguaje claro, formal y útil para docentes afiliados.",
    "Cuando corresponda, mencioná artículos, días, requisitos, funcionarios otorgantes e intervinientes presentes en los fragmentos.",
    "Si el fragmento de resumen indica funcionario otorgante o interviniente, incluilo al final de la respuesta.",
    "No confundas encabezados de secciones siguientes con el contenido del artículo consultado.",
    "Para Capacitación y Perfeccionamiento Docente, si aparece el Artículo 34 junto al Artículo 41, debe considerarse licencia extraordinaria con goce de haberes.",
    "No digas que algo no existe si en los fragmentos aparece información relacionada y suficiente.",
    "Al final incluí una línea con el formato: Fuente consultada: nombre de la fuente.",
  ].join(" ");
}

async function askGroq(
  pregunta: string,
  dominio: DominioBackend,
  chunks: LocalChunk[]
): Promise<string> {
  const client = getGroqClient();

  if (!client) {
    throw new Error("Falta GROQ_API_KEY.");
  }

  const model = getEnv("GROQ_MODEL") || "llama-3.1-8b-instant";
  const maxTokens = Number(getEnv("GROQ_MAX_TOKENS") || 900);
  const context = buildContext(chunks);
  const referencias = Array.from(new Set(chunks.map((chunk) => chunk.referencia)));

  const response: any = await client.chat.completions.create({
    model,
    temperature: 0.12,
    max_tokens: maxTokens,
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
          "Redactá una respuesta clara, precisa y directa. No uses información externa a los fragmentos.",
      },
    ],
  });

  const content = response?.choices?.[0]?.message?.content;

  if (typeof content !== "string" || !content.trim()) {
    throw new Error("Groq no devolvió contenido utilizable.");
  }

  return content.trim();
}

function isRateLimitError(error: any): boolean {
  const status = Number(error?.status || error?.response?.status || error?.code);
  const message = String(error?.message || "").toLowerCase();

  return status === 429 || message.includes("rate limit") || message.includes("429");
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
  error?: string;
}): ChatbotWorkflowOutput {
  const referencias = Array.from(
    new Set(params.chunks.map((chunk) => chunk.referencia).filter(Boolean))
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
    error: params.error,
  };
}

export async function runGroqWorkflow(
  input: ChatbotQueryInput
): Promise<ChatbotWorkflowOutput> {
  const dominio = classifyDomain(input.pregunta, input.dominio);

  try {
    const chunks = await searchLocalFragments(input, dominio);

    if (!chunks.length) {
      return buildOutput({
        ok: true,
        tipo: "sin_resultados",
        origen: "local_rag",
        dominio,
        consulta: input.pregunta,
        respuesta: NO_ENCONTRADO,
        chunks: [],
      });
    }

    try {
      const respuestaGroq = await askGroq(input.pregunta, dominio, chunks);

      return buildOutput({
        ok: true,
        tipo: "respuesta_ia",
        origen: "groq_rag",
        dominio,
        consulta: input.pregunta,
        respuesta: respuestaGroq,
        chunks,
      });
    } catch (groqError: any) {
      const localAnswer = buildLocalFallbackAnswer(chunks);
      const rateLimit = isRateLimitError(groqError);

      console.warn(
        "[groqWorkflow] Groq no respondió. Se usa fallback local:",
        groqError?.message || groqError
      );

      return buildOutput({
        ok: true,
        tipo: "respuesta_local",
        origen: rateLimit ? "local_rag_fallback_429" : "local_rag_fallback",
        dominio,
        consulta: input.pregunta,
        respuesta: localAnswer,
        chunks,
        error: groqError?.message || "Error al consultar Groq",
      });
    }
  } catch (error: any) {
    console.error("[groqWorkflow] Error general:", error);

    return buildOutput({
      ok: false,
      tipo: "error",
      origen: "groq_rag",
      dominio,
      consulta: input.pregunta,
      respuesta: SERVICIO_NO_DISPONIBLE,
      chunks: [],
      error: error?.message || "Error interno",
    });
  }
}