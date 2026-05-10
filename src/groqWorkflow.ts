import OpenAI from "openai";
import { readFile } from "node:fs/promises";

export type DominioBackend =
  | "licencias"
  | "estatuto"
  | "general"
  | "coberturas"
  | "servicios";

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
  seccion?: string;
  nivel?: string;
  tipo?: string;
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
  seccion?: string;
  nivel?: string;
  tipo?: string;
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
  "https://raw.githubusercontent.com/rcampagnale/sidca-chatbot-docs/main/decreto_636_coberturas_docentes.json";

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
    dominio === "servicios"
  ) {
    return dominio;
  }

  const text = normalizeText(pregunta);

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
  return (
    value === "licencias" ||
    value === "estatuto" ||
    value === "general" ||
    value === "coberturas" ||
    value === "servicios"
  );
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

      const manifestSources = (manifest.fuentes || [])
        .filter((fuente) => fuente.activo !== false)
        .filter((fuente) => fuente.url && isDominioBackend(fuente.dominio))
        .map((fuente) => {
          const source = String(fuente.url);
          const dominio = fuente.dominio as DominioBackend;

          return {
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
        console.log(
          `[groqWorkflow] Fuentes cargadas desde índice: ${manifestSources.length}`
        );

        return manifestSources;
      }

      console.warn(
        "[groqWorkflow] El índice de fuentes no tiene fuentes activas válidas. Se usa fallback local."
      );
    } catch (error: any) {
      console.warn(
        "[groqWorkflow] No se pudo cargar SIDCA_DOCS_MANIFEST_URL. Se usa fallback local:",
        error?.message || error
      );
    }
  }

  return getFallbackSourceConfigs();
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
    descripcion: getObjectField(item, ["descripcion", "descripción"]),
    otorgante: getObjectField(item, ["otorgante"]),
    interviniente: getObjectField(item, ["interviniente"]),
    seccion: getObjectField(item, ["seccion", "sección"]),
    nivel: getObjectField(item, ["nivel"]),
    tipo: getObjectField(item, ["tipo"]),
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

  const sources = await getSourceConfigs();
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
      ? "asamblea publica cobertura cargos horas catedra interinato suplencia cabecera cero lom fua decreto 636"
      : dominio === "servicios"
      ? "servicios app sidca contacto whatsapp beneficios turismo viajes casa del docente convenios capacitaciones certificados oficina gestion asesoramiento"
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
    text.includes("lluvia") ||
    text.includes("temporal") ||
    text.includes("tormenta") ||
    compact.includes("climatic") ||
    compact.includes("climtic") ||
    compact.includes("clim")
  );
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

  if (climateQuery && chunk.dominio === "licencias") {
    if (articleMatches(chunk, "49")) {
      score += chunk.kind === "articulo" ? 360 : 220;
    } else {
      score -= 360;
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
  dominio: DominioBackend
): Promise<LocalChunk[]> {
  const chunks = await loadLocalChunks();
  const maxResults = Math.min(Math.max(input.maxResults ?? 5, 1), 8);

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

  const matchedExpansions = getMatchedExpansions(input.pregunta, dominio);
  const preferredArticles = Array.from(
    new Set(matchedExpansions.flatMap((item) => item.preferredArticles || []))
  );

  if (dominio === "licencias" && preferredArticles.includes("16")) {
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
      const parts = [
        `### Fragmento ${index + 1}`,
        `Archivo: ${chunk.filename}`,
        `Tipo: ${chunk.kind}`,
        `Fuente: ${chunk.referencia}`,
        chunk.articulo ? `Artículo/s: ${chunk.articulo}` : "",
        chunk.seccion ? `Sección: ${chunk.seccion}` : "",
        chunk.nivel ? `Nivel: ${chunk.nivel}` : "",
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

  if (chunks[0]?.dominio === "servicios") {
    const respuestas = chunks
      .slice(0, 3)
      .map((chunk) => {
        const sugerida = getObjectField(chunk.raw, ["respuesta_sugerida"]);
        const titulo =
          chunk.titulo ||
          getObjectField(chunk.raw, ["categoria", "titulo", "nombre"]) ||
          "Servicio SiDCa";

        return sugerida
          ? `${titulo}:\n${sugerida}`
          : `${titulo}:\n${truncateText(chunk.text, 1200)}`;
      })
      .join("\n\n");

    const referencias = Array.from(
      new Set(chunks.map((chunk) => chunk.referencia).filter(Boolean))
    );

    return [
      respuestas,
      "",
      `Fuente consultada: ${referencias.join(" | ")}`,
    ].join("\n");
  }

  const main = chunks
    .slice(0, 3)
    .map((chunk) => {
      const title = chunk.articulo
        ? `Artículo ${chunk.articulo}${
            chunk.descripcion ? ` – ${chunk.descripcion}` : ""
          }`
        : chunk.descripcion || chunk.titulo || "Fragmento encontrado";

      const extra = [
        chunk.seccion ? `Sección: ${chunk.seccion}` : "",
        chunk.nivel ? `Nivel: ${chunk.nivel}` : "",
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
    seccion: chunk.seccion,
    nivel: chunk.nivel,
    tipo: chunk.tipo,
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
    "Si el dominio detectado es servicios, respondé sobre servicios, contactos, beneficios y secciones de la App SiDCa. Priorizá la respuesta_sugerida, la ubicación en la app, el WhatsApp, la dirección, el aula virtual o el enlace que aparezca en los fragmentos. No inventes números, direcciones, horarios ni enlaces.",
    "Para consultas de servicios como viajes, turismo, Casa del Docente, hotelería, convenios, certificados, capacitaciones, Oficina de Gestión, soporte técnico, asesoramiento gremial o legal, indicá el camino dentro de la app cuando esté disponible y el contacto correspondiente si figura en los fragmentos.",
    "Si el usuario pide asesoramiento gremial, contacto gremial, número gremial o con quién comunicarse por asesoramiento gremial del sindicato, respondé con el servicio de Asesoramiento Gremial y el WhatsApp de Asesoramiento General si aparece en los fragmentos. No lo confundas con licencia por representación gremial, delegados gremiales ni franquicias gremiales.",
    "Si el usuario pregunta en qué parte de la app debe ingresar, dónde buscar o dónde encontrar decretos, resoluciones, leyes o normativa, respondé con la sección Asesoramiento > Legal. Si además menciona titularización, trámite, documentación o formularios, agregá también Asesoramiento > Oficina de Gestión.",
    "Cuando respondas sobre contactos, escribí el número de WhatsApp completo y, si aparece, el área a la que corresponde. No mezcles contactos de áreas no relacionadas salvo que el fragmento principal los incluya como medios generales de contacto.",

    "Si la consulta es por razones climáticas o fenómenos meteorológicos, respondé solo con el Artículo 49 si ese es el fragmento disponible; no mezcles el Artículo 52 ni menciones plazos de razones particulares.",
    "Si la consulta es por afecciones, lesiones o licencia de corto tratamiento, respondé solo con los artículos 16 y 17 y el resumen correspondiente si están disponibles. No desarrolles el Artículo 20 de largo tratamiento ni el Artículo 27 de maternidad salvo que el usuario los pregunte expresamente.",
    "Si la consulta es por superposición de horarios, respondé solo con los artículos 53 y 54 y el resumen correspondiente si están disponibles. No desarrolles el Artículo 36 salvo que el usuario pregunte expresamente por prácticas obligatorias.",
    "Si la consulta es por maternidad, mamá, madre, embarazo o parto, respondé con el Artículo 27 y el resumen correspondiente si está disponible. No mezcles artículos de corto o largo tratamiento salvo que el fragmento del Artículo 27 los mencione como referencia accesoria.",

    "Si la consulta trata sobre faltas sin justificar, inasistencias injustificadas o abandono de servicio, priorizá los artículos 85 y 86, y usá los artículos 80 y 84 como contexto de descuento y sanciones si están presentes.",
    "REGLA NUMÉRICA OBLIGATORIA: cuando una norma establece un umbral mínimo, por ejemplo cinco (5) días, cualquier cantidad igual o superior cumple ese umbral. Ejemplo: si la consulta dice que faltó 6 días y el artículo establece abandono de servicio por cinco (5) días laborales consecutivos injustificados, debe interpretarse que 6 días sí supera el umbral de 5 días.",
    "Para inasistencias injustificadas: si son cinco (5) o más días laborales consecutivos, corresponde analizar el Artículo 85 sobre abandono de servicio. Si son no consecutivas, corresponde analizar el Artículo 86, que establece abandono al llegar a diez (10) días injustificados en el año calendario. Siempre considerar el Artículo 80 para descuento del día no trabajado y el Artículo 84 para sanciones acumulativas por ausencia injustificada.",
    "Si la pregunta no aclara si las inasistencias fueron consecutivas o no consecutivas, no afirmes una sola consecuencia como definitiva. Respondé diferenciando ambos escenarios: 1) si fueron consecutivas; 2) si no fueron consecutivas.",
    "No uses la expresión 'pagar el descuento'. Cuando corresponda, indicá que se aplicará el descuento de la remuneración del día no trabajado.",

    "Usá lenguaje claro, formal y útil para docentes afiliados.",
    "Cuando corresponda, mencioná artículos, puntos, anexos, niveles, días, requisitos, funcionarios otorgantes e intervinientes presentes en los fragmentos.",
    "Si la consulta menciona secundaria, priorizá fragmentos del Anexo II o nivel Secundario.",
    "Si la consulta menciona inicial o primaria, priorizá fragmentos del Anexo I o nivel Inicial y Primaria.",
    "Si el fragmento de resumen indica funcionario otorgante o interviniente, incluilo al final de la respuesta.",
    "No confundas encabezados de secciones siguientes con el contenido del artículo consultado.",
    "Para Capacitación y Perfeccionamiento Docente, si aparece el Artículo 34 junto al Artículo 41, debe considerarse licencia extraordinaria con goce de haberes.",
    "No infieras diferencias entre docente suplente, interino o titular si los fragmentos no contienen una definición explícita. Si la consulta está incompleta, pedí que se aclare qué situaciones de revista desea comparar.",
    "Si la consulta pregunta desde cuándo o a partir de cuántos días después de optar, tomar o ser designado en un cargo se puede solicitar licencia médica, no respondas con el Artículo 52 de razones particulares. Si no hay un fragmento con ese plazo específico, indicá que no se encontró una regla específica sobre ese plazo.",
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