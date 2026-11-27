import test from "node:test";
import assert from "node:assert/strict";
const { runGroqWorkflow } = await import("../src/groqWorkflow.js");
const { setWorkflowTestDependencies } = await import("../src/groqWorkflow.js");
const { updateWorkflowTestDependencies } = await import("../src/groqWorkflow.js");
import { readFile } from "node:fs/promises";
const source = JSON.parse(await readFile(new URL("./fixtures/movilidad.json", import.meta.url), "utf8"));
const generalList = JSON.parse(await readFile(new URL("./fixtures/nomenclador-general.json", import.meta.url), "utf8"));
const centersList = JSON.parse(await readFile(new URL("./fixtures/nomenclador-centros.json", import.meta.url), "utf8"));
const gardensList = JSON.parse(await readFile(new URL("./fixtures/nomenclador-jardines.json", import.meta.url), "utf8"));
const nationalLaw = JSON.parse(await readFile(new URL("./fixtures/ley_26206_educacion_nacional.json", import.meta.url), "utf8"));
const privateLaw = JSON.parse(await readFile(new URL("./fixtures/decreto_ley_3387_educacion_privada.json", import.meta.url), "utf8"));
const provincialLaw = JSON.parse(await readFile(new URL("./fixtures/ley_5381_educacion_catamarca.json", import.meta.url), "utf8"));
const coberturas636 = JSON.parse(await readFile(new URL("./fixtures/decreto_636_coberturas_actualizado_res3335.json", import.meta.url), "utf8"));
const ley26058 = JSON.parse(await readFile(new URL("./fixtures/ley_26058_educacion_tecnico_profesional.json", import.meta.url), "utf8"));
const licenciaRegression = JSON.parse(await readFile(new URL("./fixtures/regimen-licencias-regresion.json", import.meta.url), "utf8"));
const estatutoRegression = JSON.parse(await readFile(new URL("./fixtures/estatuto-3122-regresion.json", import.meta.url), "utf8"));
const consultasGenerales = JSON.parse(await readFile(new URL("./fixtures/consultas-generales-regresion.json", import.meta.url), "utf8"));
const serviciosSidca = JSON.parse(await readFile(new URL("./fixtures/servicios-sidca-regresion.json", import.meta.url), "utf8"));
const decreto533 = JSON.parse(await readFile(new URL("./fixtures/decreto-533-regresion.json", import.meta.url), "utf8"));
let now = 1000;
let contents = source;
let manifest: any = { version: "1", fuentes: [{ id: "movilidad_docente_decreto_533", dominio: "coberturas", referencia: "Decreto Acuerdo 533/2021 Catamarca", url: "mock://movilidad", activo: true }] };
const requests: any[] = [];
let downloads = 0;
let mockAnswer = "El artículo 6 establece el cambio automático de suplente a interino.";
const mockClient: any = {
  chat: {
    completions: {
      create: async (request: any) => {
        requests.push(request);
        return { choices: [{ message: { content: mockAnswer }, finish_reason: "stop" }], usage: {} };
      },
    },
  },
};
setWorkflowTestDependencies({
  now: () => now,
  fetchJson: async (url) => { downloads++; if (url === "mock://manifest") return manifest; if (url === "mock://movilidad") return contents; throw new Error("unexpected URL"); },
  groqClient: mockClient,
});
process.env.SIDCA_DOCS_MANIFEST_URL = "mock://manifest";
process.env.LOCAL_RAG_CACHE_TTL_MS = "300000";
process.env.GROQ_API_KEY = "test-only";

test("pipeline operativo recupera el artículo solicitado de la fuente configurada", async () => {
  const result = await runGroqWorkflow({ pregunta: "Decreto 533/2021 artículo 6" });
  assert.equal(result.dominio, "coberturas");
  assert.ok(result.articulos.some((item) => String(item.articulo).includes("6")));
  assert.ok(result.referencias.some((item) => item.includes("533/2021")));
  assert.ok(result.busqueda.some((item) => item.text?.includes("cambios de Situación de Revista")));
  assert.match(requests[0].messages[1].content, /Decreto Acuerdo 533\/2021 Catamarca/);
  assert.match(requests[0].messages[1].content, /Artículo\/s: Decreto Artículo 6/);
});

test("caché operativa: mantiene corpus dentro del TTL y actualiza tras vencer", async () => {
  setWorkflowTestDependencies({ now: () => now, fetchJson: async (url) => { downloads++; return url === "mock://manifest" ? manifest : contents; }, groqClient: mockClient });
  const start = downloads;
  await runGroqWorkflow({ pregunta: "Decreto 533/2021 artículo 6" });
  const afterWarm = downloads;
  assert.ok(afterWarm > start);
  now += 1000;
  await runGroqWorkflow({ pregunta: "Decreto 533/2021 artículo 6" });
  assert.equal(downloads, afterWarm);
  now += 6 * 60 * 1000;
  manifest = { ...manifest, version: "2" };
  await runGroqWorkflow({ pregunta: "Decreto 533/2021 artículo 6" });
  assert.ok(downloads > afterWarm);
});

function configureCorpus(manifestValue: any, documentValue: unknown, options: { failManifest?: boolean; failDocument?: boolean; delay?: () => Promise<void> } = {}) {
  setWorkflowTestDependencies({
    now: () => now,
    fetchJson: async (url) => {
      downloads++;
      if (url === "mock://manifest") { if (options.failManifest) throw new Error("manifest unavailable"); return manifestValue; }
      await options.delay?.();
      if (options.failDocument) throw new Error("document unavailable");
      return documentValue;
    },
    groqClient: mockClient,
  });
}

const question = "Decreto 533/2021 artículo 6";
const oneSourceManifest = { version: "stable", fuentes: [{ id: "movilidad_docente_decreto_533", dominio: "coberturas", referencia: "Decreto Acuerdo 533/2021 Catamarca", url: "mock://movilidad", activo: true }] };

test("A1 mismo número de versión y contenido cambiado invalida respuesta", async () => {
  now += 10_000; const fresh = structuredClone(source); configureCorpus(oneSourceManifest, fresh);
  await runGroqWorkflow({ pregunta: question }); const oldContextCount = requests.length;
  now += 6 * 60_000; const changed = structuredClone(source); changed[0].contenido += " Texto agregado de fixture.";
  updateWorkflowTestDependencies({ fetchJson: async (url) => url === "mock://manifest" ? oneSourceManifest : changed }); await runGroqWorkflow({ pregunta: question });
  assert.ok(requests.length > oldContextCount); assert.match(requests.at(-1).messages[1].content, /Texto agregado de fixture/);
});

test("A2-A3 fuente desactivada se elimina del corpus y no genera contexto", async () => {
  now += 10_000; configureCorpus(oneSourceManifest, source); await runGroqWorkflow({ pregunta: question });
  now += 6 * 60_000; const disabled = { ...oneSourceManifest, fuentes: oneSourceManifest.fuentes.map((s) => ({ ...s, activo: false })) };
  updateWorkflowTestDependencies({ fetchJson: async (url) => url === "mock://manifest" ? disabled : source }); const before = requests.length; const result = await runGroqWorkflow({ pregunta: question });
  assert.equal(requests.length, before); assert.equal(result.tipo, "error");
});

test("A4 descarga parcial conserva el corpus completo previo", async () => {
  now += 10_000; const two = { ...oneSourceManifest, fuentes: [...oneSourceManifest.fuentes, { ...oneSourceManifest.fuentes[0], id: "second", url: "mock://second" }] };
  configureCorpus(two, source); await runGroqWorkflow({ pregunta: question });
  now += 6 * 60_000; updateWorkflowTestDependencies({ fetchJson: async (url) => { if (url === "mock://manifest") return two; if (url === "mock://movilidad") return source; throw new Error("temporary partial failure"); } });
  const result = await runGroqWorkflow({ pregunta: question }); assert.ok(result.busqueda.length > 0);
});

test("A5-A6 error total de manifiesto no usa fallback fijo", async () => {
  now += 10_000; configureCorpus(oneSourceManifest, source); await runGroqWorkflow({ pregunta: question });
  const before = requests.length;
  updateWorkflowTestDependencies({ fetchJson: async (url) => { if (url === "mock://manifest") throw new Error("manifest unavailable"); return source; } });
  const result = await runGroqWorkflow({ pregunta: question }); assert.ok(result.articulos.length > 0); assert.equal(requests.length, before);
});

test("A7 el proveedor documental puede recuperarse tras un fallo temporal", async () => {
  now += 10_000; configureCorpus(oneSourceManifest, source); await runGroqWorkflow({ pregunta: question });
  updateWorkflowTestDependencies({ fetchJson: async (url) => { if (url === "mock://manifest") throw new Error("temporary outage"); return source; } }); await runGroqWorkflow({ pregunta: question });
  updateWorkflowTestDependencies({ fetchJson: async (url) => url === "mock://manifest" ? oneSourceManifest : source }); const result = await runGroqWorkflow({ pregunta: question });
  assert.equal(result.dominio, "coberturas"); assert.ok(result.articulos.length > 0);
});

test("A8 solicitudes concurrentes comparten carga documental", async () => {
  now += 10_000; let releases!: () => void; const gate = new Promise<void>((resolve) => { releases = resolve; });
  configureCorpus(oneSourceManifest, source, { delay: () => gate });
  const first = runGroqWorkflow({ pregunta: question }); const second = runGroqWorkflow({ pregunta: question });
  releases(); const results = await Promise.all([first, second]); assert.ok(results.every((result) => result.articulos.length > 0));
});

async function queryNomencladores(pregunta: string) {
  now += 6 * 60_000;
  const sources = [
    { id: "nomenclador_docente_catamarca", dominio: "general", referencia: "Decreto Acuerdo 2431/2016", url: "mock://general", activo: true },
    { id: "nomenclador_centros_educativos", dominio: "general", referencia: "Decreto ECyT 2453/2019, Artículo 2", url: "mock://centros", activo: true },
    { id: "nomenclador_jardines_maternales", dominio: "general", referencia: "Decreto ECyT 1480/2019", url: "mock://jardines", activo: true },
  ];
  setWorkflowTestDependencies({ now: () => now, fetchJson: async (url) => url === "mock://manifest" ? { version: "1.9.0", fuentes: sources } : url === "mock://general" ? generalList : url === "mock://centros" ? centersList : gardensList, groqClient: mockClient });
  return runGroqWorkflow({ pregunta });
}

test("Caso 9: pipeline recupera escuela común del Decreto 2431/2016", async () => {
  const result = await queryNomencladores("Maestro de Grado Jornada Extendida de escuela común, nomenclador 2431/2016");
  const userMessage = requests.at(-1).messages[1].content;
  assert.match(userMessage, /CAT2431-046/); assert.match(userMessage, /Nivel Primario Común/); assert.match(userMessage, /Decreto Acuerdo 2431\/2016/);
  assert.doesNotMatch(userMessage, /CAT2453-CE/); assert.equal(result.dominio, "nomenclador");
});

test("Caso 10: contexto operativo incluye Centros Educativos 2453/2019", async () => {
  await queryNomencladores("Director General de Centros Educativos Decreto 2453/2019");
  const userMessage = requests.at(-1).messages[1].content;
  assert.match(userMessage, /CAT2453-CE-001/); assert.match(userMessage, /Artículo 2/); assert.match(userMessage, /Centros Educativos/);
});

test("Caso 11: pipeline recupera la categoría de Jardines Pasitos Cuidados", async () => {
  await queryNomencladores("Director de Jardín Maternal Pasitos Cuidados Decreto 1480/2019");
  const userMessage = requests.at(-1).messages[1].content;
  assert.match(userMessage, /CAT-1480-2019-09/); assert.match(userMessage, /Director\/a de Jardín Maternal/); assert.match(userMessage, /Decreto ECyT N.º 1480\/2019/);
});

test("Caso 12: prompt identifica horas, incompatibilidad y puntos del director maternal por separado", async () => {
  await queryNomencladores("¿Cuántas horas reloj semanales, puntos de incompatibilidad y puntos salariales tiene un director de Jardín Maternal Pasitos Cuidados?");
  const prompt = requests.at(-1).messages[1].content;
  assert.match(prompt, /horas_reloj_semanales: 30/); assert.match(prompt, /puntaje_incompatibilidad: 30/); assert.match(prompt, /puntos_nomenclador: 436/);
});

test("Caso 13: preceptor usa registro general de Centros, sin mezclar POF", async () => {
  await queryNomencladores("¿Cuántas horas reloj semanales y puntos de incompatibilidad tiene un preceptor de Centros Educativos?");
  const prompt = requests.at(-1).messages[1].content;
  assert.match(prompt, /CAT2453-CE-015/); assert.match(prompt, /horas_reloj_semanales: 30/); assert.match(prompt, /puntaje_incompatibilidad: 18/); assert.match(prompt, /Artículo 2/); assert.doesNotMatch(prompt, /POF específica/);
});

test("Caso 14: comparación recupera y mantiene identidad de ambas instituciones", async () => {
  await queryNomencladores("¿Qué diferencias existen entre un director de Centro Educativo y un director de Jardín Maternal Pasitos Cuidados?");
  const prompt = requests.at(-1).messages[1].content;
  assert.match(prompt, /CAT2453-CE-001/); assert.match(prompt, /CAT-1480-2019-09/); assert.match(prompt, /Decreto ECyT 2453\/2019/); assert.match(prompt, /Decreto ECyT N.º 1480\/2019/);
});

async function queryIntegrity(pregunta: string, id: string, dominio: string, data: unknown) {
  now += 6 * 60_000;
  const url = `mock://${id}`;
  setWorkflowTestDependencies({ now: () => now, fetchJson: async (requested) => requested === "mock://manifest"
    ? { version: "test", fuentes: [{ id, dominio, referencia: id === "ley_26206_educacion_nacional" ? "Ley Nacional 26.206" : id === "decreto_ley_3387_educacion_privada" ? "Decreto-Ley 3387" : id === "coberturas_decreto_636" ? "Decreto Acuerdo 636/2021 y Resolución RESRH-2021-3335-E-CAT-MTRH" : "Ley Provincial 5381 Catamarca", url, activo: true }] }
    : data, groqClient: mockClient });
  const before = requests.length;
  const result = await runGroqWorkflow({ pregunta });
  return { result, prompt: requests.length > before ? requests.at(-1).messages[1].content : "", generated: requests.length > before };
}

test("Caso 15: artículo derogado se responde como histórico y no vigente", async () => {
  const { result } = await queryIntegrity("Ley 26.206 artículo 9", "ley_26206_educacion_nacional", "general", nationalLaw);
  assert.match(result.respuesta, /histórico/); assert.match(result.respuesta, /derogado/); assert.match(result.respuesta, /Ley 27.798/); assert.doesNotMatch(result.respuesta, /no será inferior al seis por ciento.*vigente/i);
});

test("Caso 16: sustitución del artículo 16 prioriza resumen no literal", async () => {
  const { result } = await queryIntegrity("Ley 26.206 artículo 16", "ley_26206_educacion_nacional", "general", nationalLaw);
  assert.match(result.respuesta, /sustituido/); assert.match(result.respuesta, /Ley 27.045/); assert.match(result.respuesta, /resumen.*no una transcripción oficial/i); assert.match(result.respuesta, /cuatro años/); assert.doesNotMatch(result.respuesta, /desde la edad de cinco años como vigente/i);
});

test("Caso 17: artículo 92 conserva modificación y resumen en prompt Groq", async () => {
  const { prompt, generated } = await queryIntegrity("Ley 26.206 artículo 92", "ley_26206_educacion_nacional", "general", nationalLaw);
  assert.equal(generated, true); assert.match(prompt, /Artículo 92/); assert.match(prompt, /Ley 27\.621, art\. 25/); assert.match(prompt, /resumen_no_literal/); assert.match(prompt, /conciencia ambiental/); assert.match(prompt, /histórico/);
});

test("Caso 18: artículo 50 ausente informa incompletitud y no inventa texto", async () => {
  const { result, generated } = await queryIntegrity("Decreto-Ley 3387 artículo 50", "decreto_ley_3387_educacion_privada", "general", privateLaw);
  assert.equal(generated, false); assert.match(result.respuesta, /no está disponible/); assert.match(result.respuesta, /33 a 75 están ausentes/); assert.match(result.respuesta, /no deben completarse por inferencia/); assert.equal(result.articulos.some((item) => String(item.articulo).includes("50")), false);
});

test("Caso 19: artículo 32 truncado entrega advertencia editorial por separado", async () => {
  const { result, generated } = await queryIntegrity("Decreto-Ley 3387 artículo 32", "decreto_ley_3387_educacion_privada", "general", privateLaw);
  assert.equal(generated, false); assert.match(result.respuesta, /está truncado/); assert.match(result.respuesta, /no es íntegro/); assert.match(result.respuesta, /no puedo reconstruir/); assert.match(JSON.stringify(result.articulos), /control_documental/);
});

test("Caso 20: Ley 5381 marca vigencia no verificada en contexto Groq", async () => {
  const { prompt, generated } = await queryIntegrity("Ley Provincial 5381 Catamarca artículo 49", "ley_5381_educacion_catamarca", "general", provincialLaw);
  assert.equal(generated, true); assert.match(prompt, /Ley Provincial 5381 Catamarca/); assert.match(prompt, /jurisdiccion: Catamarca/); assert.match(prompt, /vigencia y modificaciones posteriores no verificadas/);
});

test("Caso 21: Decreto 636 y Resolución 3335 conservan anexo reemplazado", async () => {
  const previousAnswer = mockAnswer;
  mockAnswer = "La Ley Nacional 26.206, artículo 2, dispone una regla distinta.";
  const { result, prompt, generated } = await queryIntegrity("¿Qué anexo del Decreto Acuerdo 636/2021 modifica la Resolución RESRH-2021-3335-E-CAT-MTRH?", "coberturas_decreto_636", "coberturas", coberturas636);
  mockAnswer = previousAnswer;
  assert.equal(generated, true); assert.match(prompt, /Dcto\. Acdo\. N° 636\/2021/); assert.match(prompt, /RESRH-2021-3335/); assert.match(prompt, /como Anexo II/); assert.match(prompt, /Educación Secundaria en todas sus Modalidades/); assert.match(prompt, /será reemplazado por el presente Anexo II/);
  assert.equal(result.origen, "local_rag_fallback"); assert.doesNotMatch(result.respuesta, /Ley Nacional 26\.206/);
});

test("Caso 22: comparación recupera Ley 26058 nacional y Ley 5381 provincial", async () => {
  const data = { v: ley26058, p: provincialLaw };
  now += 6 * 60_000;
  setWorkflowTestDependencies({ now: () => now, fetchJson: async (url) => url === "mock://manifest" ? { version:"test", fuentes:[
    {id:"ley_26058_educacion_tecnico_profesional",dominio:"general",referencia:"Ley Nacional 26.058",url:"mock://26058",activo:true},
    {id:"ley_5381_educacion_catamarca",dominio:"general",referencia:"Ley Provincial 5381 Catamarca",url:"mock://5381",activo:true},
  ]} : url === "mock://26058" ? data.v : data.p, groqClient:mockClient });
  const before=requests.length; await runGroqWorkflow({pregunta:"¿Qué diferencias existen entre la Ley Nacional 26.058 de Educación Técnico Profesional y la Ley Provincial 5381 de Educación de Catamarca respecto de la educación secundaria técnica?"});
  const prompt=requests[before].messages[1].content; assert.match(prompt,/Ley Nacional 26\.058/); assert.match(prompt,/Ley Provincial 5381/);
  const blocks=prompt.split(/### Fragmento \d+/).slice(1); const national=blocks.filter((block:string)=>block.includes("Norma identificada: Ley 26.058")).join("\n"); const catamarca=blocks.filter((block:string)=>block.includes("Norma identificada: Ley 5381")).join("\n");
  assert.ok(national); assert.ok(catamarca); assert.match(national,/Artículo\/s: Artículo \d+/); assert.match(national,/Jurisdicción: Nacional/); assert.match(catamarca,/Artículo\/s: Artículo \d+/); assert.match(catamarca,/Jurisdicción: Catamarca/); assert.match(prompt,/vigencia y modificaciones posteriores no verificadas/);
  assert.match(national,/Artículo\/s: Artículo 22/); assert.match(catamarca,/Artículo\/s: Artículo 54/); assert.match(catamarca,/Artículo\/s: Artículo 35/);
  const nationalArticles=new Set(Array.from(national.matchAll(/Artículo\/s: Artículo (\d+)/g),match=>match[1]));
  const crossNormArticle=Array.from(catamarca.matchAll(/Artículo\/s: Artículo (\d+)/g),match=>match[1]).find(article=>!nationalArticles.has(article)); assert.ok(crossNormArticle,"se necesita un artículo recuperado solo de la norma provincial");
  mockAnswer=`La Ley Nacional 26.058, artículo ${crossNormArticle}, establece esa misma regla.`;
  setWorkflowTestDependencies({now:()=>now,fetchJson:async(url)=>url==="mock://manifest"?{version:"test",fuentes:[
    {id:"ley_26058_educacion_tecnico_profesional",dominio:"general",referencia:"Ley Nacional 26.058",url:"mock://26058",activo:true},
    {id:"ley_5381_educacion_catamarca",dominio:"general",referencia:"Ley Provincial 5381 Catamarca",url:"mock://5381",activo:true},
  ]}:url==="mock://26058"?data.v:data.p,groqClient:mockClient});
  const invalidCitation=await runGroqWorkflow({pregunta:"¿Qué diferencias existen entre la Ley Nacional 26.058 de Educación Técnico Profesional y la Ley Provincial 5381 de Educación de Catamarca respecto de la educación secundaria técnica?"});
  assert.equal(invalidCitation.origen,"local_rag_fallback"); mockAnswer="El artículo 6 establece el cambio automático de suplente a interino.";
});

test("regresión: consulta temática de artículo resuelve el régimen de licencias sin pedir norma", async () => {
  now += 6 * 60_000;
  setWorkflowTestDependencies({ now: () => now, fetchJson: async (url) => url === "mock://manifest"
    ? { version:"theme", fuentes:[{id:"licencias_docentes_provincial",dominio:"licencias",referencia:"Régimen Provincial de Licencias Docentes",url:"mock://licenses",activo:true}] }
    : licenciaRegression, groqClient:mockClient });
  const before = requests.length;
  const result = await runGroqWorkflow({ pregunta:"¿Qué establece el artículo 16 del régimen de licencias docentes?" });
  assert.equal(requests.length, before + 1); assert.notEqual(result.tipo, "aclaracion");
  assert.match(requests.at(-1).messages[1].content, /Artículo\/s: 16/); assert.match(requests.at(-1).messages[1].content, /veinticinco \(25\) días corridos/);
});

test("regresión: advertencia del artículo 32 no bloquea artículo completo del Decreto-Ley 3387", async () => {
  const previousAnswer = mockAnswer; mockAnswer = "El fragmento consultado está disponible en la fuente documental.";
  const { result, prompt, generated } = await queryIntegrity("Decreto-Ley 3387 artículo 1", "decreto_ley_3387_educacion_privada", "general", privateLaw);
  mockAnswer = previousAnswer;
  assert.equal(generated, true); assert.match(prompt, /Artículo 1/); assert.doesNotMatch(result.respuesta, /está truncado/);
});

test("regresión: artículo derogado de otra sección no interrumpe artículo vigente consultado", async () => {
  const previousAnswer = mockAnswer; mockAnswer = "El artículo consultado establece que la educación es un derecho personal y social.";
  const { result, prompt, generated } = await queryIntegrity("Ley Nacional 26.206 artículo 2", "ley_26206_educacion_nacional", "general", nationalLaw);
  mockAnswer = previousAnswer;
  assert.equal(generated, true); assert.match(prompt, /Artículo 2/); assert.doesNotMatch(result.respuesta, /texto histórico|derogado|sustituido/i);
});

test("regresión: acepta ambos órdenes de cita al validar identidad normativa y artículo", async () => {
  const previousAnswer = mockAnswer; mockAnswer = "La Ley 3122, artículo 6, reconoce derechos; también, el artículo 6 de la Ley Provincial 3122 trata esos derechos.";
  now += 6 * 60_000;
  setWorkflowTestDependencies({ now: () => now, fetchJson: async (url) => url === "mock://manifest"
    ? { version:"citations", fuentes:[{id:"estatuto_docente_provincial",dominio:"estatuto",referencia:"Ley Provincial 3122 Catamarca",url:"mock://estatuto",activo:true}] }
    : estatutoRegression, groqClient:mockClient });
  const result = await runGroqWorkflow({ pregunta:"Ley Provincial 3122 artículo 6", dominio:"estatuto" }); mockAnswer = previousAnswer;
  assert.equal(result.origen, "groq_rag"); assert.match(result.respuesta, /Ley 3122, artículo 6/); assert.match(result.respuesta, /artículo 6 de la Ley Provincial 3122/);
});

test("regresión: rechazo de cita cuyo número solo comparte una subcadena", async () => {
  const previousAnswer = mockAnswer; mockAnswer = "La Ley Provincial 31220, artículo 6, establece esta regla.";
  now += 6 * 60_000;
  setWorkflowTestDependencies({ now: () => now, fetchJson: async (url) => url === "mock://manifest"
    ? { version:"citation-mismatch", fuentes:[{id:"estatuto_docente_provincial",dominio:"estatuto",referencia:"Ley Provincial 3122 Catamarca",url:"mock://estatuto",activo:true}] }
    : estatutoRegression, groqClient:mockClient });
  const result = await runGroqWorkflow({ pregunta:"Ley Provincial 3122 artículo 6", dominio:"estatuto" }); mockAnswer = previousAnswer;
  assert.equal(result.origen, "local_rag_fallback"); assert.doesNotMatch(result.respuesta, /31220/);
});

test("regresión: mismo corpus recargado después del TTL conserva la respuesta cacheada", async () => {
  const previousAnswer = mockAnswer; mockAnswer = "Respuesta estable del corpus de prueba.";
  const stableManifest = { version:"cache-stable", fuentes:[{id:"movilidad_docente_decreto_533",dominio:"coberturas",referencia:"Decreto Acuerdo 533/2021 Catamarca",url:"mock://movilidad",activo:true}] };
  now += 6 * 60_000;
  setWorkflowTestDependencies({ now: () => now, fetchJson: async (url) => url === "mock://manifest" ? stableManifest : source, groqClient:mockClient });
  const questionStable = "Decreto 533/2021 artículo 6, consulta estable de prueba";
  await runGroqWorkflow({ pregunta:questionStable }); const generatedAtFirst = requests.length;
  now += 6 * 60_000;
  updateWorkflowTestDependencies({ now: () => now, fetchJson: async (url) => url === "mock://manifest" ? stableManifest : structuredClone(source) });
  const response = await runGroqWorkflow({ pregunta:questionStable }); mockAnswer = previousAnswer;
  assert.equal(requests.length, generatedAtFirst); assert.match(response.respuesta, /Respuesta estable/);
});

test("regresión: Ley 5381 activa se asigna al dominio educacion_provincial por su ID", async () => {
  now += 6 * 60_000;
  setWorkflowTestDependencies({ now: () => now, fetchJson: async (url) => url === "mock://manifest"
    ? { version:"law5381", fuentes:[{id:"ley_5381_educacion_catamarca",dominio:"general",referencia:"Ley Provincial 5381 Catamarca",url:"mock://5381",activo:true}] }
    : provincialLaw, groqClient:mockClient });
  const result = await runGroqWorkflow({ pregunta:"Ley Provincial 5381 Catamarca artículo 49", dominio:"educacion_provincial" });
  assert.equal(result.dominio, "educacion_provincial"); assert.ok(result.articulos.length > 0);
});

test("regresión: comparación multinormativa no usa Ley 5381 para la referencia parcial 381", async () => {
  now += 6 * 60_000;
  setWorkflowTestDependencies({ now: () => now, fetchJson: async (url) => url === "mock://manifest"
    ? { version:"exact-multinorm", fuentes:[{id:"ley_5381_educacion_catamarca",dominio:"general",referencia:"Ley Provincial 5381 Catamarca",url:"mock://5381",activo:true}] }
    : provincialLaw, groqClient:mockClient });
  const before = requests.length;
  await runGroqWorkflow({ pregunta:"¿Qué diferencias existen entre la Ley Provincial 5381 y la Ley Provincial 381?", dominio:"educacion_provincial" });
  const prompt = requests.at(-1).messages[1].content;
  assert.equal(requests.length, before + 1); assert.equal((prompt.match(/### Fragmento \d+/g) || []).length, 2);
  assert.match(prompt, /Ley Provincial 5381 Catamarca/); assert.doesNotMatch(prompt, /Norma identificada: Ley 381\b/);
});

test("Caso 23: artículo sin norma ni contexto solicita aclaración sin retrieval", async () => {
  const before=downloads; const beforeRequests=requests.length; const result=await runGroqWorkflow({pregunta:"¿Qué establece el artículo 35?"});
  assert.equal(result.tipo,"aclaracion"); assert.match(result.respuesta,/¿De qué norma/); assert.equal(downloads,before); assert.equal(requests.length,beforeRequests);
});

test("Caso 24: contexto por solicitud resuelve seguimiento y nueva norma prevalece", async () => {
  const manifestValue={version:"test",fuentes:[
    {id:"movilidad_docente_decreto_533",dominio:"coberturas",referencia:"Decreto Acuerdo 533/2021 Catamarca",url:"mock://mob",activo:true},
    {id:"ley_26058_educacion_tecnico_profesional",dominio:"general",referencia:"Ley Nacional 26.058",url:"mock://law",activo:true},
  ]};
  now+=6*60_000; setWorkflowTestDependencies({now:()=>now,fetchJson:async(url)=>url==="mock://manifest"?manifestValue:url==="mock://mob"?source:ley26058,groqClient:mockClient});
  const first=await runGroqWorkflow({pregunta:"¿Qué establece el Decreto Acuerdo 533/2021 sobre la movilidad docente?"});
  const before=requests.length; const follow=await runGroqWorkflow({pregunta:"¿Y qué establece el artículo 6?",contextoConversacional:"Decreto Acuerdo 533/2021, movilidad docente"});
  assert.equal(follow.dominio,"coberturas"); assert.ok(follow.articulos.some((a)=>String(a.articulo).includes("6"))); assert.match(requests[before].messages[1].content,/Decreto Acuerdo 533\/2021/);
  const switched=await runGroqWorkflow({pregunta:"¿Y el artículo 6 de la Ley 26.058?",contextoConversacional:"Decreto Acuerdo 533/2021"});
  assert.equal(switched.dominio,"educacion_tecnico_profesional"); assert.ok(switched.referencias.some((r)=>r.includes("26.058"))); assert.notEqual(first.dominio,switched.dominio);
});

test("Caso 25: fuentes inactivas del manifiesto v1.9 no se descargan ni llegan a Groq", async () => {
  const productionManifest=JSON.parse(await readFile(new URL("./fixtures/manifest.json",import.meta.url),"utf8"));
  const inactiveUrls=productionManifest.fuentes.filter((f:any)=>f.activo===false).map((f:any)=>f.url);
  const fetched:string[]=[]; now+=6*60_000;
  setWorkflowTestDependencies({now:()=>now,fetchJson:async(url)=>{fetched.push(url); if(url==="mock://manifest")return productionManifest; if(url.includes("decreto_533_2021"))return source; if(url.includes("decreto_636"))return coberturas636; return [];},groqClient:mockClient});
  for(const pregunta of ["¿Qué establece la Ley Nacional 26.206?","¿Qué establece el artículo 35 de la Ley Provincial 5381?","¿Qué derechos reconoce el Decreto-Ley 3387 a los docentes de gestión privada?","¿Cómo funciona un sumario según el Decreto 413/2000?"]){
    const before=requests.length; const result=await runGroqWorkflow({pregunta}); assert.match(result.respuesta,/fuente está inactiva/i); assert.equal(requests.length,before);
  }
  const unknown=await runGroqWorkflow({pregunta:"¿Qué establece la Ley Nacional 9999?"}); assert.doesNotMatch(unknown.respuesta,/fuente está inactiva/i); assert.match(unknown.respuesta,/No encontré fragmentos suficientes/);
  assert.ok(inactiveUrls.every((url:string)=>!fetched.includes(url)));
});

async function runWithSources(pregunta: string, fuentes: any[], payloads: Record<string, unknown>, answer: string) {
  now += 6 * 60_000;
  const fetched: string[] = [];
  const before = requests.length;
  const previousAnswer = mockAnswer;
  mockAnswer = answer;
  setWorkflowTestDependencies({ now: () => now, fetchJson: async (url) => {
    fetched.push(url);
    if (url === "mock://manifest") return { version: "regresion-v1", fuentes };
    if (url in payloads) return payloads[url];
    throw new Error(`unexpected URL ${url}`);
  }, groqClient: mockClient });
  try {
    const result = await runGroqWorkflow({ pregunta });
    const prompt = requests.length > before ? requests.at(-1).messages[1].content : "";
    return { result, prompt, fetched, generated: requests.length > before };
  } finally {
    mockAnswer = previousAnswer;
  }
}

test("Caso 26: regresión de corto/largo tratamiento y razones particulares", async () => {
  const fuentes = [
    { id: "licencias_docentes_provincial", dominio: "licencias", referencia: "Decreto Acuerdo Nº 1092/2015 – Régimen de Licencias Docentes", url: "mock://licencias-regresion", activo: true },
    { id: "ley_26206_educacion_nacional", dominio: "general", referencia: "Ley Nacional 26.206", url: "mock://ley-nacional", activo: true },
  ];
  const payloads = { "mock://licencias-regresion": licenciaRegression, "mock://ley-nacional": nationalLaw };
  const short = await runWithSources("¿Cuántos días de licencia por corto tratamiento tiene un docente de Catamarca?", fuentes, payloads, "El artículo 16 concede veinticinco (25) días corridos en el año calendario.");
  assert.equal(short.result.dominio, "licencias"); assert.equal(short.generated, true); assert.ok(short.result.articulos.some((a:any)=>String(a.articulo)==="16"));
  assert.match(short.prompt,/Decreto Acuerdo Nº 1092\/2015/); assert.match(short.prompt,/Artículo\/s: 16/); assert.match(short.prompt,/veinticinco \(25\) días corridos/); assert.doesNotMatch(short.prompt,/Ley Nacional 26\.206/); assert.match(short.result.respuesta,/veinticinco \(25\) días corridos/);

  const long = await runWithSources("¿Cuántos días de licencia por largo tratamiento corresponden a un docente?", fuentes, payloads, "El artículo 20 prevé hasta un año con haberes íntegros, seis meses con el 50% y seis meses sin goce de haberes.");
  assert.equal(long.result.dominio,"licencias"); assert.ok(long.result.articulos.some((a:any)=>String(a.articulo)==="20")); assert.match(long.prompt,/Artículo\/s: 20/); assert.match(long.prompt,/seis \(6\) meses/);

  const particular = await runWithSources("¿Cuántos días por razones particulares puede solicitar un docente?", fuentes, payloads, "El artículo 52 permite hasta dos días de inasistencia al mes y no más de cinco días hábiles al año.");
  assert.equal(particular.result.dominio,"licencias"); assert.ok(particular.result.articulos.some((a:any)=>String(a.articulo)==="52")); assert.match(particular.prompt,/Artículo\/s: 52/); assert.match(particular.prompt,/DOS \(2\) días/); assert.match(particular.prompt,/CINCO \(5\) días hábiles/);
});

test("Caso 27: regresión de derechos y deberes bajo Ley Provincial 3122", async () => {
  const fuentes = [
    { id: "estatuto_docente_provincial", dominio: "estatuto", referencia: "Ley Nº 3122 – Estatuto del Docente Provincial", url: "mock://estatuto-3122", activo: true },
    { id: "ley_26206_educacion_nacional", dominio: "general", referencia: "Ley Nacional 26.206", url: "mock://ley-nacional", activo: true },
  ];
  const payloads = { "mock://estatuto-3122": estatutoRegression, "mock://ley-nacional": nationalLaw };
  const duties = await runWithSources("¿Qué deberes tiene el personal docente según el Estatuto Provincial Ley 3122?", fuentes, payloads, "El artículo 5 enumera los deberes del personal docente, entre ellos desempeñar con dignidad, eficiencia y lealtad las funciones del cargo.");
  assert.equal(duties.result.dominio,"estatuto"); assert.ok(duties.result.referencias.some((x:string)=>x.includes("3122"))); assert.match(duties.prompt,/Ley Nº 3122/); assert.match(duties.prompt,/Artículo\/s: 5/); assert.match(duties.prompt,/dignidad, eficiencia y lealtad/); assert.doesNotMatch(duties.prompt,/Ley Nacional 26\.206/);
  const rights = await runWithSources("¿Qué derechos reconoce la Ley Provincial 3122 al docente?", fuentes, payloads, "El artículo 6 reconoce, entre otros, estabilidad en el cargo y el goce de una remuneración y jubilación justas.");
  assert.equal(rights.result.dominio,"estatuto"); assert.match(rights.prompt,/Artículo\/s: 6/); assert.match(rights.prompt,/La estabilidad en el cargo/); assert.match(rights.prompt,/remuneración y jubilación justas/); assert.match(rights.result.respuesta,/estabilidad en el cargo/);
});

test("Caso 28: regresión de coberturas 636/3335 y movilidad 533", async () => {
  const fuentes = [
    { id: "coberturas_decreto_636", dominio: "coberturas", referencia: "Decreto Acuerdo 636/2021 y Resolución RESRH-2021-3335-E-CAT-MTRH", url: "mock://docs/decreto_636_coberturas_actualizado_res3335.json", activo: true },
    { id: "movilidad_docente_decreto_533", dominio: "coberturas", referencia: "Decreto Acuerdo 533/2021 Catamarca", url: "mock://docs/decreto_533_2021_movilidad_docente.json", activo: true },
  ];
  const payloads = { [fuentes[0].url]: coberturas636, [fuentes[1].url]: decreto533 };
  const coverage = await runWithSources("¿Cómo se ofrecen los cargos interinos y suplentes mediante una Asamblea Pública?", fuentes, payloads, "El Decreto 636 regula el sistema de asambleas públicas para cubrir cargos interinos y suplentes.");
  assert.equal(coverage.result.dominio,"coberturas"); assert.ok(coverage.result.referencias.some((x:string)=>x.includes("636"))); assert.match(coverage.prompt,/Asamblea Pública/); assert.match(coverage.prompt,/cargos/); assert.match(coverage.prompt,/interino.*suplente/i);
  const replaced = await runWithSources("¿Qué anexo del Decreto Acuerdo 636/2021 modifica la Resolución RESRH-2021-3335-E-CAT-MTRH para Educación Secundaria?", fuentes, payloads, "El artículo 1 de la Resolución reemplaza el Anexo II del Decreto 636/2021 para Educación Secundaria.");
  assert.equal(replaced.result.dominio,"coberturas"); assert.match(replaced.prompt,/RESRH-2021-3335/); assert.match(replaced.prompt,/Artículo\/s: Decreto Artículo 2/); assert.match(replaced.prompt,/reemplazado por el presente Anexo II/);
  const mobility = await runWithSources("¿Qué establece el Decreto 533/2021 sobre movilidad docente y el cambio de suplente a interino?", fuentes, payloads, "El artículo 6 faculta el cambio automático de suplente a interino en las condiciones del Decreto 533/2021.");
  assert.equal(mobility.result.dominio,"coberturas"); assert.ok(mobility.result.referencias.some((x:string)=>x.includes("533/2021"))); assert.match(mobility.prompt,/Decreto Acuerdo N.º 533\/2021/); assert.match(mobility.prompt,/Artículo\/s: Decreto Artículo 6/); assert.doesNotMatch(mobility.prompt,/RESRH-2021-3335/);
  const fetched = [...coverage.fetched,...replaced.fetched,...mobility.fetched]; assert.ok(fetched.includes("mock://docs/decreto_636_coberturas_actualizado_res3335.json")); assert.ok(fetched.includes("mock://docs/decreto_533_2021_movilidad_docente.json")); assert.equal(fetched.some((x)=>x.includes("decreto_636_coberturas_docentes.json")),false);
});

test("Caso 29: regresión de servicios, beneficios, contactos y app SIDCA", async () => {
  const fuentes = [{ id: "servicios_sidca_app", dominio: "servicios", referencia: "Información institucional disponible en la App SiDCa", url: "mock://servicios-sidca", activo: true }];
  const payloads = { "mock://servicios-sidca": serviciosSidca };
  const scenarios = [
    { q: "¿Dónde veo las capacitaciones y cursos disponibles?", id: "sidca_capacitaciones", evidence: /Aula Virtual/, answer: "Las capacitaciones están en Información > Capacitaciones. Allí se consultan cursos disponibles, aprobados y el registro de asistencia." },
    { q: "¿Cómo obtengo mi certificado de una capacitación?", id: "sidca_certificados", evidence: /5493834325816/, answer: "Consultá en Información > Capacitaciones > Cursos Aprobados. Para entrega, el contacto indicado es 5493834325816; la impresión figura como no disponible." },
    { q: "¿Cómo accedo a los beneficios de la Red de Convenios?", id: "sidca_red_convenios", evidence: /credencial digital/, answer: "La app indica ingresar en Beneficios > Red de Convenios y presentar la credencial digital para acceder a los beneficios." },
    { q: "¿Dónde encuentro los contactos institucionales y la sede de SiDCa?", id: "sidca_contactos_institucionales", evidence: /Ayacucho 227/, answer: "Los contactos están en Nosotros > Contacto. La sede central informada está en Ayacucho 227 - 1° Piso, San Fernando del Valle de Catamarca." },
    { q: "¿Qué secciones principales tiene la app SiDCa?", id: "sidca_servicios_home_resumen", evidence: /Nosotros.*Información.*Asesoramiento.*Beneficios/s, answer: "La app organiza sus servicios en Nosotros, Información, Asesoramiento y Beneficios." },
  ];
  for (const item of scenarios) {
    const outcome = await runWithSources(item.q, fuentes, payloads, item.answer);
    assert.equal(outcome.result.dominio,"servicios"); assert.equal(outcome.generated,true); assert.ok(outcome.result.articulos.length>0);
    assert.match(outcome.prompt,new RegExp(item.id)); assert.match(outcome.prompt,item.evidence); assert.equal(outcome.result.respuesta,item.answer);
  }
});

test("Caso 30: regresión de consultas generales e información sin respaldo", async () => {
  const fuentes = [{ id: "consultas_generales_sidca", dominio: "general", referencia: "Consultas generales SiDCa", url: "mock://consultas-generales", activo: true }];
  const payloads = { "mock://consultas-generales": consultasGenerales };
  const affiliation = await runWithSources("¿Cómo me afilio al sindicato?", fuentes, payloads, "La afiliación se realiza presentando la documentación requerida en la sede correspondiente o por los canales habilitados por el sindicato.");
  assert.equal(affiliation.result.dominio,"general"); assert.match(affiliation.prompt,/¿Cómo me afilio al sindicato\?/); assert.match(affiliation.prompt,/canales habilitados por el sindicato/); assert.match(affiliation.result.respuesta,/canales habilitados/);
  const greeting = await runWithSources("Hola, ¿cómo me afilio al sindicato?", fuentes, payloads, "La afiliación se realiza presentando la documentación requerida en la sede correspondiente o por los canales habilitados por el sindicato.");
  assert.equal(greeting.result.dominio,"general"); assert.match(greeting.prompt,/documentación requerida/); assert.match(greeting.result.respuesta,/canales habilitados/);
  const standaloneGreeting = await runWithSources("Hola", fuentes, payloads, "This answer must not be requested from the model.");
  assert.equal(standaloneGreeting.result.dominio,"general"); assert.equal(standaloneGreeting.result.respuesta,"¡Hola! ¿En qué puedo ayudarte?"); assert.equal(standaloneGreeting.generated,false); assert.equal(standaloneGreeting.fetched.length,0);
  const schedule = await runWithSources("¿Cuál es el horario de atención del sindicato?", fuentes, payloads, "El horario puede variar según la sede; conviene verificarlo por los medios oficiales antes de asistir.");
  assert.equal(schedule.result.dominio,"general"); assert.match(schedule.prompt,/horario de atención/); assert.doesNotMatch(schedule.result.respuesta,/\b(?:08|09|10|11|12|13|14|15|16|17|18):\d{2}\b/);
  const unsupported = await runWithSources("¿Qué dispone la Ley 9999 sobre el régimen docente?", fuentes, payloads, "No hay respaldo para esa norma.");
  assert.equal(unsupported.result.dominio,"general"); assert.equal(unsupported.generated,false); assert.match(unsupported.result.respuesta,/No encontré fragmentos suficientes/); assert.doesNotMatch(unsupported.result.respuesta,/artículo|dispone|establece/i);
});

test("Contrato HTTP: acepta el cuerpo previo, contexto opcional y los dominios nuevos", async () => {
  const { createServer } = await import("node:http");
  const { spawn } = await import("node:child_process");
  const docsServer = createServer((req,res) => {
    res.setHeader("Content-Type","application/json");
    if(req.url==="/manifest") res.end(JSON.stringify({version:"http-test",fuentes:[{id:"consultas_generales_sidca",dominio:"general",referencia:"Consultas generales SiDCa",url:`http://127.0.0.1:${docsPort}/consultas.json`,activo:true}]}));
    else if(req.url==="/consultas.json") res.end(JSON.stringify(consultasGenerales));
    else { res.statusCode=404; res.end("{}"); }
  });
  await new Promise<void>((resolve,reject)=>{docsServer.once("error",reject);docsServer.listen(0,"127.0.0.1",()=>resolve());});
  const docsPort=(docsServer.address() as any).port;
  const probe=createServer(); await new Promise<void>((resolve,reject)=>{probe.once("error",reject);probe.listen(0,"127.0.0.1",()=>resolve());}); const apiPort=(probe.address() as any).port; await new Promise<void>(resolve=>probe.close(()=>resolve()));
  const apiBase=`http://127.0.0.1:${apiPort}`; let childOutput="";
  const child=spawn(process.execPath,["--import","tsx","src/server.ts"],{cwd:process.cwd(),env:{...process.env,PORT:String(apiPort),GROQ_API_KEY:"",SIDCA_DOCS_MANIFEST_URL:`http://127.0.0.1:${docsPort}/manifest`},stdio:["ignore","pipe","pipe"]});
  child.stdout.on("data",chunk=>{childOutput+=String(chunk);}); child.stderr.on("data",chunk=>{childOutput+=String(chunk);});
  try {
    let ready=false;
    for(let attempt=0;attempt<60;attempt++){
      if(child.exitCode!==null) throw new Error(`server exited before readiness: ${childOutput}`);
      try { await fetch(`${apiBase}/api/chatbot/query`); ready=true; break; } catch { await new Promise(resolve=>setTimeout(resolve,100)); }
    }
    assert.equal(ready,true,`server did not become ready: ${childOutput}`);
    const post=async(body:unknown)=>fetch(`${apiBase}/api/chatbot/query`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    const legacy=await post({pregunta:"¿Cómo me afilio al sindicato?"}); const legacyBody=await legacy.json() as any;
    assert.equal(legacy.status,200); for(const key of ["ok","tipo","dominio","origen","consulta","consultaNormalizada","respuesta","articulos","referencias","busqueda","conversationId"]) assert.ok(key in legacyBody,`missing response field ${key}`);
    const withContext=await post({pregunta:"¿Dónde consulto por afiliación?",contextoConversacional:"La conversación trata de consultas generales SiDCa."}); assert.equal(withContext.status,200);
    for(const domain of ["nomenclador","sumarios","educacion_nacional","educacion_provincial","educacion_privada","educacion_tecnico_profesional"]){const response=await post({pregunta:"consulta de compatibilidad",dominio:domain});assert.equal(response.status,200,`domain ${domain} rejected`);const body=await response.json() as any;assert.equal(body.dominio,domain);}
    const invalid=await post({pregunta:"x"}); const invalidBody=await invalid.json() as any; assert.equal(invalid.status,400); assert.equal(invalidBody.status,400); for(const key of ["ok","tipo","dominio","origen","consulta","consultaNormalizada","respuesta","articulos","referencias","busqueda","conversationId"]) assert.ok(key in invalidBody,`missing error response field ${key}`);
  } finally {
    child.kill(); await new Promise<void>(resolve=>{if(child.exitCode!==null)resolve();else{const timer=setTimeout(resolve,2000);child.once("close",()=>{clearTimeout(timer);resolve();});}});
    await new Promise<void>(resolve=>docsServer.close(()=>resolve()));
  }
});
