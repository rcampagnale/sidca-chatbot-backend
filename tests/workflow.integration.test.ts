import test from "node:test";
import assert from "node:assert/strict";
const { runGroqWorkflow } = await import("../src/groqWorkflow.js");
const { setWorkflowTestDependencies } = await import("../src/groqWorkflow.js");
const { updateWorkflowTestDependencies } = await import("../src/groqWorkflow.js");
import { readFile } from "node:fs/promises";
const source = JSON.parse(await readFile(new URL("./fixtures/movilidad.json", import.meta.url), "utf8"));
const generalList = JSON.parse(await readFile(new URL("./fixtures/nomenclador-general.json", import.meta.url), "utf8"));
const diverseGeneralList = JSON.parse(await readFile(new URL("./fixtures/nomenclador-general-diversidad.json", import.meta.url), "utf8"));
const centersList = JSON.parse(await readFile(new URL("./fixtures/nomenclador-centros.json", import.meta.url), "utf8"));
const completeCentersList = JSON.parse(await readFile(new URL("./fixtures/nomenclador-centros-completo.json", import.meta.url), "utf8"));
const generalRealList = JSON.parse(await readFile(new URL("./fixtures/nomenclador-general-real.json", import.meta.url), "utf8"));
const centersRealList = JSON.parse(await readFile(new URL("./fixtures/nomenclador-centros-real.json", import.meta.url), "utf8"));
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
const licenseParticulars = JSON.parse(await readFile(new URL("./fixtures/licencias-particulares-regresion.json", import.meta.url), "utf8"));
const licenseParticularsReal = JSON.parse(await readFile(new URL("./fixtures/licencias-particulares-real.json", import.meta.url), "utf8"));
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

async function queryNomencladores(pregunta: string, generalData: unknown = generalList, centersData: unknown = centersList) {
  now += 6 * 60_000;
  const sources = [
    { id: "nomenclador_docente_catamarca", dominio: "general", referencia: "Decreto Acuerdo 2431/2016", url: "mock://general", activo: true },
    { id: "nomenclador_centros_educativos", dominio: "general", referencia: "Decreto ECyT 2453/2019, Artículo 2", url: "mock://centros", activo: true },
    { id: "nomenclador_jardines_maternales", dominio: "general", referencia: "Decreto ECyT 1480/2019", url: "mock://jardines", activo: true },
  ];
  setWorkflowTestDependencies({ now: () => now, fetchJson: async (url) => url === "mock://manifest" ? { version: "1.9.0", fuentes: sources } : url === "mock://general" ? generalData : url === "mock://centros" ? centersData : gardensList, groqClient: mockClient });
  return runGroqWorkflow({ pregunta });
}

test("Caso 9: pipeline recupera escuela común del Decreto 2431/2016", async () => {
  const before = requests.length;
  const result = await queryNomencladores("Maestro de Grado Jornada Extendida de escuela común, nomenclador 2431/2016");
  assert.equal(requests.length, before); assert.match(result.respuesta, /Maestro de Grado Jornada Extendida/); assert.match(result.respuesta, /35/); assert.match(result.respuesta, /Decreto Acuerdo 2431\/2016/);
  assert.doesNotMatch(result.respuesta, /CAT2453-CE|Jardín Maternal/); assert.equal(result.dominio, "nomenclador");
});

test("v3: maestro de grado de jornada completa usa el registro general de escuela común", async () => {
  const before = requests.length;
  const result = await queryNomencladores("¿Cuántas horas reloj semanales y puntos de incompatibilidad tiene un maestro de grado de jornada completa de una escuela común?");
  assert.equal(requests.length, before); assert.ok(result.referencias.some((reference: string) => /CAT2431-052/.test(reference))); assert.match(result.respuesta, /Jornada: Completa/);
  assert.match(result.respuesta, /Horas reloj semanales: 40/); assert.match(result.respuesta, /Puntos de incompatibilidad: 36/);
  assert.match(result.respuesta, /Nivel Primario Común/); assert.doesNotMatch(result.respuesta, /CAT2453-CE|CAT-1480/);
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
  const before = requests.length;
  const result = await queryNomencladores("¿Cuántas horas reloj semanales, puntos de incompatibilidad y puntos salariales tiene un director de Jardín Maternal Pasitos Cuidados?");
  assert.equal(requests.length, before); assert.match(result.respuesta, /Horas reloj semanales: 30/); assert.match(result.respuesta, /Puntos de incompatibilidad: 30/); assert.match(result.respuesta, /Puntos salariales\/nomenclador: 436/);
});

test("Caso 13: preceptor usa registro general de Centros, sin mezclar POF", async () => {
  const before = requests.length;
  const result = await queryNomencladores("¿Cuántas horas reloj semanales y puntos de incompatibilidad tiene un preceptor de Centros Educativos?");
  assert.equal(requests.length, before); assert.ok(result.referencias.some((reference: string) => /CAT2453-CE-015/.test(reference))); assert.match(result.respuesta, /Horas reloj semanales: 30/); assert.match(result.respuesta, /Puntos de incompatibilidad: 18/); assert.match(result.respuesta, /Norma: .*Artículo 2/); assert.doesNotMatch(result.respuesta, /POF específica/);
});

test("Caso 14: comparación recupera y mantiene identidad de ambas instituciones", async () => {
  const before = requests.length;
  const result = await queryNomencladores("¿Qué diferencias existen entre un director de Centro Educativo y un director de Jardín Maternal Pasitos Cuidados?");
  assert.equal(requests.length, before); assert.ok(result.referencias.some((reference: string) => /CAT2453-CE-001/.test(reference))); assert.ok(result.referencias.some((reference: string) => /CAT-1480-2019-09/.test(reference))); assert.match(result.respuesta, /Decreto ECyT 2453\/2019/); assert.match(result.respuesta, /Decreto ECyT N.º 1480\/2019/);
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

test("auditoría final: comparación no presenta un artículo derogado como vigente", async () => {
  const { result, generated } = await queryIntegrity("Compará los artículos 8 y 9 de la Ley Nacional 26.206 según su vigencia", "ley_26206_educacion_nacional", "general", nationalLaw);
  assert.equal(generated, false); assert.match(result.respuesta, /incluye artículos que la fuente marca como no vigentes/i);
  assert.match(result.respuesta, /Artículo 9: figura como derogado/); assert.doesNotMatch(result.respuesta, /seis por ciento.*vigente/i);
});

test("auditoría final: artículo truncado bloquea una comparación como si ambos textos fueran completos", async () => {
  const { result, generated } = await queryIntegrity("Compará los artículos 31 y 32 del Decreto-Ley 3387", "decreto_ley_3387_educacion_privada", "general", privateLaw);
  assert.equal(generated, false); assert.match(result.respuesta, /artículo 32 está truncado/i); assert.match(result.respuesta, /no puedo reconstruir/);
  assert.doesNotMatch(result.respuesta, /presentaré.*como vigente/i);
});

test("Caso 20: Ley 5381 marca vigencia no verificada en contexto Groq", async () => {
  const { prompt, generated } = await queryIntegrity("Ley Provincial 5381 Catamarca artículo 49", "ley_5381_educacion_catamarca", "general", provincialLaw);
  assert.equal(generated, true); assert.match(prompt, /Ley Provincial 5381 Catamarca/); assert.match(prompt, /Jurisdicción: Catamarca/); assert.match(prompt, /vigencia y modificaciones posteriores no verificadas/);
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
  assert.match(requests.at(-1).messages[1].content, /Artículo\/s: 16/); assert.match(requests.at(-1).messages[1].content, /veinticinco \(25\) días corridos/); assert.match(requests.at(-1).messages[1].content, /Artículo\/s: 17/);
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
  assert.equal(particular.result.dominio,"licencias"); assert.equal(particular.generated,false); assert.ok(particular.result.articulos.some((a:any)=>String(a.articulo)==="52")); assert.match(particular.result.respuesta,/ART[IÍ]CULO 52/); assert.match(particular.result.respuesta,/DOS \(2\) días/); assert.match(particular.result.respuesta,/CINCO \(5\) días hábiles/);
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
    assert.equal(outcome.result.dominio,"servicios"); assert.equal(outcome.generated,false); assert.ok(outcome.result.articulos.length>0);
    assert.match(outcome.result.respuesta,item.evidence); assert.doesNotMatch(outcome.result.respuesta,/horario.*\d{2}:\d{2}/i);
  }
});

test("v3 auditoría: descarga de certificado usa su ficha y no sustituye la respuesta por capacitaciones", async () => {
  const fuentes = [{ id: "servicios_sidca_app", dominio: "servicios", referencia: "Información institucional disponible en la App SiDCa", url: "mock://certificado-especifico", activo: true }];
  const outcome = await runWithSources("¿Cómo descargo el certificado de una capacitación?", fuentes, { "mock://certificado-especifico": serviciosSidca }, "no debe llamarse");
  assert.equal(outcome.generated, false); assert.match(outcome.result.respuesta, /La ficha documental “Certificados de cursos y entrega de certificados” indica:/);
  assert.match(outcome.result.respuesta, /no confirma que la ruta ni el estado de impresión reflejen la versión actual/i);
  assert.doesNotMatch(outcome.result.respuesta, /Aula Virtual|cursos disponibles/i);
  assert.ok(outcome.result.referencias.some((reference: string) => /Certificados de cursos y entrega de certificados/.test(reference)));
});

test("auditoría final: sin sidca_certificados no usa una ficha de capacitaciones como sustituto", async () => {
  const fuentes = [{ id: "servicios_sidca_app", dominio: "servicios", referencia: "Información institucional disponible en la App SiDCa", url: "mock://solo-capacitaciones", activo: true }];
  const trainingOnly = { ...serviciosSidca, servicios: serviciosSidca.servicios.filter((item: any) => item.id !== "sidca_certificados") };
  const outcome = await runWithSources("¿Cómo descargo el certificado de una capacitación?", fuentes, { "mock://solo-capacitaciones": trainingOnly }, "no debe llamarse");
  assert.equal(outcome.generated, false); assert.match(outcome.result.respuesta, /no encontré una ficha documental específica de certificados/i);
  assert.match(outcome.result.respuesta, /no permite inferir el procedimiento/); assert.equal(outcome.result.articulos.length, 0); assert.deepEqual(outcome.result.referencias, []);
  assert.doesNotMatch(outcome.result.respuesta, /Aula Virtual|Cursos Aprobados|botón|sección/);
});

test("v3 auditoría: certificados sin procedimiento documentado informan la carencia sin inventar navegación", async () => {
  const fuentes = [{ id: "servicios_sidca_app", dominio: "servicios", referencia: "Información institucional disponible en la App SiDCa", url: "mock://certificado-sin-procedimiento", activo: true }];
  const incomplete = { ...serviciosSidca, servicios: serviciosSidca.servicios.map((item: any) => {
    if (item.id !== "sidca_certificados") return item;
    const { respuesta_sugerida: _omitted, ...withoutProcedure } = item; return withoutProcedure;
  }) };
  const outcome = await runWithSources("¿Cómo descargo el certificado de una capacitación?", fuentes, { "mock://certificado-sin-procedimiento": incomplete }, "no debe llamarse");
  assert.equal(outcome.generated, false); assert.match(outcome.result.respuesta, /no incluye una respuesta_sugerida/i);
  assert.doesNotMatch(outcome.result.respuesta, /botón|descargar desde|ir a la sección/i);
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

test("v3: razones particulares (52) y asuntos particulares (45) se resuelven desde sus textos reales", async () => {
  const fuentes = [{ id: "licencias_docentes_provincial", dominio: "licencias", referencia: "Decreto Acuerdo 1092/2015 – Régimen de Licencias Docentes", url: "mock://licencias-v3", activo: true }];
  const reasons = await runWithSources("¿Cuántos días de licencia por razones particulares tiene un docente?", fuentes, { "mock://licencias-v3": licenseParticulars }, "no debe llamarse");
  assert.equal(reasons.generated, false); assert.match(reasons.result.respuesta, /ART[IÍ]CULO 52/i); assert.match(reasons.result.respuesta, /DOS \(2\) días de inasistencias al mes/); assert.match(reasons.result.respuesta, /CINCO \(5\) días hábiles al año/); assert.doesNotMatch(reasons.result.respuesta, /seis \(6\) meses/);
  const affairs = await runWithSources("¿Qué condiciones tiene la licencia por asuntos particulares?", fuentes, { "mock://licencias-v3": licenseParticulars }, "no debe llamarse");
  assert.equal(affairs.generated, false); assert.match(affairs.result.respuesta, /ART[IÍ]CULO 45/i); assert.match(affairs.result.respuesta, /seis \(6\) meses dentro de cada decenio/); assert.match(affairs.result.respuesta, /QUINCE \(15\) días/); assert.doesNotMatch(affairs.result.respuesta, /CINCO \(5\) días hábiles al año/);
  const ambiguous = await runGroqWorkflow({ pregunta: "¿Cuántos días de licencia particular tiene un docente?" });
  assert.equal(ambiguous.tipo, "aclaracion"); assert.match(ambiguous.respuesta, /artículo 52.*artículo 45/i);
});

test("v3 auditoría: comparación de razones y asuntos particulares conserva ambos artículos", async () => {
  const fuentes = [{ id: "licencias_docentes_provincial", dominio: "licencias", referencia: "Decreto Acuerdo 1092/2015 – Régimen de Licencias Docentes", url: "mock://licencias-particulares-comparison", activo: true }];
  const outcome = await runWithSources("Compará las licencias por razones particulares y asuntos particulares", fuentes, { "mock://licencias-particulares-comparison": licenseParticulars }, "Las dos licencias tienen condiciones diferentes según los artículos citados.");
  assert.equal(outcome.generated, false); assert.match(outcome.result.respuesta, /ART[IÍ]CULO 45/); assert.match(outcome.result.respuesta, /ART[IÍ]CULO 52/);
  assert.match(outcome.result.respuesta, /anticipación mínima de QUINCE \(15\) días/); assert.match(outcome.result.respuesta, /docentes por horas cátedra/);
  assert.equal(outcome.result.dominio, "licencias");
});

test("operativo v3.1: la comparación local de artículos 45 y 52 conserva condiciones sin atribuciones", async () => {
  const fuentes = [{ id: "licencias_docentes_provincial", dominio: "licencias", referencia: "Decreto Acuerdo 1092/2015 – Régimen de Licencias Docentes", url: "mock://licencias-particulares-real", activo: true }];
  const outcome = await runWithSources("¿Qué diferencias existen entre licencia por razones particulares y licencia por asuntos particulares?", fuentes, { "mock://licencias-particulares-real": licenseParticularsReal }, "No debe consultarse Groq para esta comparación documental.");
  assert.equal(outcome.generated, false);
  assert.match(outcome.result.respuesta, /ART[IÍ]CULO 45/); assert.match(outcome.result.respuesta, /ART[IÍ]CULO 52/);
  assert.match(outcome.result.respuesta, /a\) Común:[\s\S]*DOS \(2\) años de antigüedad ininterrumpida/);
  assert.match(outcome.result.respuesta, /en este caso, en el período inmediato anterior/);
  assert.match(outcome.result.respuesta, /anticipación mínima de QUINCE \(15\) días/);
  assert.match(outcome.result.respuesta, /con veinticuatro \(24\) horas de anticipación,[\s\S]*con goce de haberes/);
  assert.match(outcome.result.respuesta, /docentes por horas cátedra[\s\S]*equivalentes al número de horas cátedras semanal/);
  assert.doesNotMatch(outcome.result.respuesta, /licencia común (?:se concede|es|tiene) con goce/i);
});

test("operativo v3.1: artículo 52 literal excluye el encabezado de la sección siguiente", async () => {
  const fuentes = [{ id: "licencias_docentes_provincial", dominio: "licencias", referencia: "Régimen de Licencias Docentes", url: "mock://licencia-52-real", activo: true }];
  const outcome = await runWithSources("Dame el texto literal del artículo 52 del régimen de licencias docentes", fuentes, { "mock://licencia-52-real": licenseParticularsReal }, "No debe consultarse Groq para texto literal.");
  assert.equal(outcome.generated, false); assert.match(outcome.result.respuesta, /con goce de haberes/);
  assert.match(outcome.result.respuesta, /horas anuales de inasistencias equivalentes al número de horas cátedras semanal/);
  assert.doesNotMatch(outcome.result.respuesta, /SUPERPOSICIÓN DE HORARIOS/);
});

test("auditoría final RAG v3.1: artículo literal conserva una línea normativa final en mayúsculas", async () => {
  const fuente = [{ id: "licencias_docentes_provincial", dominio: "licencias", referencia: "Decreto Acuerdo 1092/2015 – Régimen de Licencias Docentes", url: "mock://licencia-upper-final", activo: true }];
  const document = structuredClone(licenseParticularsReal);
  const article = document.find((item: any) => String(item.articulo).includes("52"));
  assert.ok(article);
  const normativeFinalLine = "Los docentes por horas cátedra podrán solicitar hasta un máximo de horas anuales de inasistencias equivalentes al número de horas cátedras semanal del cargo de que se trate.";
  article.contenido = `${article.contenido.replace(/\n_{5,}\s*SUPERPOSICI[ÓO]N DE HORARIOS[\s\S]*$/i, "").trimEnd()}\n${normativeFinalLine.toUpperCase()}`;
  const outcome = await runWithSources("Dame el texto literal del artículo 52 del régimen de licencias docentes", fuente, { "mock://licencia-upper-final": document }, "No debe consultarse Groq.");
  assert.equal(outcome.generated, false); assert.match(outcome.result.respuesta, new RegExp(normativeFinalLine.toUpperCase().replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
});

test("operativo v3.1: Decreto 533/2021 conserva el primer suplente sin agregar trámites", async () => {
  const fuentes = [{ id: "movilidad_docente_decreto_533", dominio: "coberturas", referencia: "Decreto Acuerdo 533/2021 Catamarca", url: "mock://movilidad-first-substitute", activo: true }];
  const outcome = await runWithSources("¿A quién corresponde el cambio automático de situación de revista según el Decreto Acuerdo 533/2021?", fuentes, { "mock://movilidad-first-substitute": decreto533 }, "No debe consultarse Groq para esta regla documentada.");
  assert.equal(outcome.generated, false); assert.match(outcome.result.respuesta, /primer docente suplente/);
  assert.match(outcome.result.respuesta, /cuando se produzca la baja del interino o titular suplido/);
  assert.doesNotMatch(outcome.result.respuesta, /presentar un trámite|iniciar una solicitud|requisito adicional/i);
});

test("auditoría final RAG v3.1: pregunta operativa exacta del artículo 6 del Decreto 533 se resuelve localmente", async () => {
  const fuentes = [{ id: "movilidad_docente_decreto_533", dominio: "coberturas", referencia: "Decreto Acuerdo 533/2021 Catamarca", url: "mock://mobility-exact-art6", activo: true }];
  const before = requests.length;
  const outcome = await runWithSources("¿Qué establece el artículo 6 del Decreto 533/2021?", fuentes, { "mock://mobility-exact-art6": decreto533 }, "No debe consultarse Groq.");
  assert.equal(outcome.generated, false); assert.equal(requests.length, before);
  assert.match(outcome.result.respuesta, /primer docente suplente/);
  assert.match(outcome.result.respuesta, /cuando se produzca la baja del interino o titular suplido/);
  assert.doesNotMatch(outcome.result.respuesta, /presentar un trámite|iniciar una solicitud|requisito adicional/i);
});

test("auditoría final RAG v3.1: artículo 6 determinista no intercepta otro artículo de Decreto 533", async () => {
  const fuentes = [{ id: "movilidad_docente_decreto_533", dominio: "coberturas", referencia: "Decreto Acuerdo 533/2021 Catamarca", url: "mock://mobility-non-art6", activo: true }];
  const outcome = await runWithSources("¿Qué establece el artículo 5 del Decreto 533/2021?", fuentes, { "mock://mobility-non-art6": decreto533 }, "La respuesta del artículo 5 no debe sustituirse por el artículo 6.");
  assert.equal(outcome.generated, true); assert.match(outcome.prompt, /ART[IÍ]CULO 5/); assert.doesNotMatch(outcome.result.respuesta, /primer docente suplente/);
});

test("auditoría final RAG v3.1: artículo 6 de otra norma no activa la regla del Decreto 533", async () => {
  const fuentes = [{ id: "movilidad_docente_decreto_533", dominio: "coberturas", referencia: "Decreto Acuerdo 533/2021 Catamarca", url: "mock://mobility-other-norm", activo: true }];
  const outcome = await runWithSources("¿Qué establece el artículo 6 de la Ley Provincial 5381?", fuentes, { "mock://mobility-other-norm": decreto533 }, "No hay respaldo para esa norma.");
  assert.equal(outcome.generated, false); assert.doesNotMatch(outcome.result.respuesta, /primer docente suplente|baja del interino o titular suplido/);
});

test("v3 auditoría: artículos 16 y 17 se recuperan juntos sin agregar el artículo 20", async () => {
  const fuentes = [{ id: "licencias_docentes_provincial", dominio: "licencias", referencia: "Decreto Acuerdo 1092/2015", url: "mock://licencias-16-17", activo: true }];
  const outcome = await runWithSources("¿Qué disponen conjuntamente los artículos 16 y 17 del régimen de licencias?", fuentes, { "mock://licencias-16-17": licenciaRegression }, "Los artículos regulan afecciones comunes y la derivación a junta médica.");
  assert.equal(outcome.generated, true); assert.match(outcome.prompt, /ART[IÍ]CULO 16/); assert.match(outcome.prompt, /ART[IÍ]CULO 17/); assert.doesNotMatch(outcome.prompt, /ART[IÍ]CULO 20/);
});

test("v3 auditoría: artículos 20, 21 y 22 conservan las condiciones complementarias", async () => {
  const fuentes = [{ id: "licencias_docentes_provincial", dominio: "licencias", referencia: "Decreto Acuerdo 1092/2015", url: "mock://licencias-20-22", activo: true }];
  const outcome = await runWithSources("Explicá los artículos 20, 21 y 22 sobre licencia por largo tratamiento", fuentes, { "mock://licencias-20-22": licenciaRegression }, "Los artículos establecen duración, cómputo y condiciones para una nueva licencia.");
  assert.equal(outcome.generated, true); for (const article of [20, 21, 22]) assert.match(outcome.prompt, new RegExp(`ART[IÍ]CULO ${article}`));
  assert.match(outcome.prompt, /cinco \(5\) años/); assert.match(outcome.prompt, /tres \(3\) años/);
});

test("v3 auditoría: una norma explícita ajena no hereda artículos inferidos de licencias", async () => {
  const fuentes = [{ id: "licencias_docentes_provincial", dominio: "licencias", referencia: "Decreto Acuerdo 1092/2015", url: "mock://licencias-explicit-mismatch", activo: true }];
  const outcome = await runWithSources("¿Qué establece la Ley Provincial 5381, artículo 52, sobre razones particulares?", fuentes, { "mock://licencias-explicit-mismatch": licenseParticulars }, "No hay texto aplicable.");
  assert.equal(outcome.generated, false); assert.doesNotMatch(outcome.result.respuesta, /ART[IÍ]CULO 52/); assert.doesNotMatch(outcome.result.respuesta, /DOS \(2\) días/);
});

test("v3: nomenclador solicita precisión si hay más de una jornada compatible", async () => {
  const before = requests.length;
  const result = await queryNomencladores("¿Cuántas horas reloj semanales tiene un maestro de grado de escuela común?");
  assert.equal(result.tipo, "aclaracion"); assert.match(result.respuesta, /Jornada Extendida/); assert.match(result.respuesta, /Jornada Completa/); assert.equal(requests.length, before);
});

test("v3 auditoría: comparación de jornadas resuelve los registros por separado", async () => {
  const result = await queryNomencladores("Compará Maestro de Grado Jornada Completa vs. Maestro de Grado Jornada Extendida de escuela común");
  assert.equal(result.dominio, "nomenclador"); assert.ok(result.referencias.some((reference: string) => /CAT2431-052/.test(reference))); assert.ok(result.referencias.some((reference: string) => /CAT2431-046/.test(reference)));
  assert.match(result.respuesta, /Jornada Completa/); assert.match(result.respuesta, /Jornada Extendida/); assert.match(result.respuesta, /2431\/2016/);
});

test("v3 auditoría: consultas acentuadas y sin acentos mantienen la resolución documental", async () => {
  const accented = await queryNomencladores("¿Cuántas horas reloj tiene un maestro de grado de jornada completa de escuela común?");
  const unaccented = await queryNomencladores("Cuantas horas reloj tiene un maestro de grado de jornada completa de escuela comun?");
  assert.ok(accented.referencias.some((reference: string) => /CAT2431-052/.test(reference))); assert.ok(unaccented.referencias.some((reference: string) => /CAT2431-052/.test(reference)));
  assert.match(accented.respuesta, /Horas reloj semanales: 40/); assert.match(unaccented.respuesta, /Horas reloj semanales: 40/);
});

test("v3 auditoría: más de cinco registros reales se examinan y diferencia Maestro Especial de Maestro de Grado", async () => {
  assert.ok(diverseGeneralList.registros.length > 5);
  const result = await queryNomencladores("¿Cuántas horas reloj tiene un Maestro Especial de Jornada Simple de Nivel Primario Común?", diverseGeneralList);
  assert.ok(result.referencias.some((reference: string) => /CAT2431-065/.test(reference))); assert.match(result.respuesta, /Maestro Especial Escuela JS - Primario Común/);
  assert.match(result.respuesta, /Horas reloj semanales: 12/); assert.doesNotMatch(result.respuesta, /CAT2431-052|Maestro de Grado Jornada Completa/);
});

test("v3 auditoría: cargos compuestos Secretario y Prosecretario no se confunden", async () => {
  const result = await queryNomencladores("Compará Secretario de 2da. de Nivel Secundario y Educación Artística con Prosecretario de 2da. de Nivel Secundario y Educación Artística", diverseGeneralList);
  assert.ok(result.referencias.some((reference: string) => /CAT2431-115/.test(reference))); assert.ok(result.referencias.some((reference: string) => /CAT2431-116/.test(reference)));
  assert.match(result.respuesta, /Secretario de 2da\./); assert.match(result.respuesta, /Prosecretario de 2da\./);
});

test("v3 auditoría: Director y Vicedirector se comparan como roles independientes", async () => {
  const result = await queryNomencladores("Compará Director común 1ra. cat. Anexo Albergue JC vs. Vicedirector Común 1ra. cat. JC", diverseGeneralList);
  assert.ok(result.referencias.some((reference: string) => /CAT2431-028/.test(reference))); assert.ok(result.referencias.some((reference: string) => /CAT2431-059/.test(reference)));
  assert.match(result.respuesta, /Director común/); assert.match(result.respuesta, /Vicedirector Común/);
});

test("v3 auditoría: comparación entre niveles identifica cada registro y su norma", async () => {
  const result = await queryNomencladores("Compará Director JIN 1ra. cat. vs Director común 1ra. cat. Anexo Albergue JC", diverseGeneralList);
  assert.ok(result.referencias.some((reference: string) => /CAT2431-018/.test(reference))); assert.ok(result.referencias.some((reference: string) => /CAT2431-028/.test(reference)));
  assert.match(result.respuesta, /Educación Inicial/); assert.match(result.respuesta, /Nivel Primario Común/); assert.match(result.respuesta, /Decreto Acuerdo 2431\/2016/);
});

test("operativo v3.1: abreviatura Director de Centro Educativo resuelve el cargo institucional", async () => {
  assert.equal(generalRealList.registros.length, 186); assert.equal(centersRealList.registros.length, 20);
  const before = requests.length;
  const result = await queryNomencladores("Compara un Director de Centro Educativo con un Director de Jardín Maternal Pasitos Cuidados", generalRealList, centersRealList);
  assert.equal(requests.length, before); assert.equal(result.tipo, "respuesta_local", result.respuesta);
  assert.match(result.respuesta, /Director General/); assert.match(result.respuesta, /Director\/a de Jardín Maternal/);
  assert.doesNotMatch(result.respuesta, /CAT2453-CE|CAT-1480/);
  assert.ok(result.referencias.some((reference: string) => /2453\/2019/.test(reference)));
  assert.ok(result.referencias.some((reference: string) => /1480\/2019/.test(reference)));
  assert.ok(result.referencias.some((reference: string) => /CAT2453-CE-001/.test(reference)));
  assert.ok(result.referencias.some((reference: string) => /CAT-1480-2019-09/.test(reference)));
});

test("auditoría final RAG v3.1: alias Director de Centro Educativo aclara si el catálogo ofrece dos categorías", async () => {
  const centersWithAlternative = structuredClone(centersRealList);
  const director = centersWithAlternative.registros.find((record: any) => record.id === "CAT2453-CE-001");
  assert.ok(director);
  centersWithAlternative.registros.push({ ...director, id: "TEST-CAT2453-CE-001-ALT", nivel_educativo: "Inicial" });
  const result = await queryNomencladores("Compara un Director de Centro Educativo con un Director de Jardín Maternal Pasitos Cuidados", generalRealList, centersWithAlternative);
  assert.equal(result.tipo, "aclaracion"); assert.match(result.respuesta, /Alternativas documentadas/);
  assert.match(result.respuesta, /Director General/); assert.match(result.respuesta, /nivel educativo/);
});

test("auditoría final RAG v3.1: institución al final no resuelve un Director General ambiguo", async () => {
  const centersWithAlternative = structuredClone(centersRealList);
  const director = centersWithAlternative.registros.find((record: any) => record.id === "CAT2453-CE-001");
  assert.ok(director);
  centersWithAlternative.registros.push({ ...director, id: "TEST-CAT2453-CE-001-SCOPE", nivel_educativo: "Inicial" });
  const result = await queryNomencladores("Compara Director General con Vicedirector de Centros Educativos", generalRealList, centersWithAlternative);
  assert.equal(result.tipo, "aclaracion"); assert.match(result.respuesta, /Alternativas documentadas|Alternativas:/);
  assert.match(result.respuesta, /Director General/); assert.match(result.respuesta, /Vicedirector/);
});

test("operativo v3.1: calificadores compartidos resuelven comparación de jornadas con corpus completo", async () => {
  assert.equal(generalRealList.registros.length, 186);
  const before = requests.length;
  const result = await queryNomencladores("Compara un maestro de grado de jornada completa con un maestro de grado de jornada extendida de nivel primario común", generalRealList);
  assert.equal(requests.length, before); assert.equal(result.tipo, "respuesta_local", result.respuesta);
  assert.match(result.respuesta, /Horas reloj semanales: 40/); assert.match(result.respuesta, /Horas reloj semanales: 35/);
  assert.match(result.respuesta, /Jornada: Completa/); assert.match(result.respuesta, /Jornada: Extendida/);
  assert.match(result.respuesta, /Cargo: Maestro de Grado Jornada Completa\nNivel\/modalidad:/);
  assert.doesNotMatch(result.respuesta, /CAT2431-/); assert.doesNotMatch(result.respuesta, /;.*Horas reloj/);
  assert.ok(result.referencias.some((reference: string) => /2431\/2016/.test(reference)));
  assert.ok(result.referencias.some((reference: string) => /CAT2431-052/.test(reference)));
  assert.ok(result.referencias.some((reference: string) => /CAT2431-046/.test(reference)));
});

test("auditoría final RAG v3.1: el corpus real completo identifica la consulta operativa de Maestro de Grado JC", async () => {
  assert.equal(generalRealList.registros.length, 186); assert.equal(centersRealList.registros.length, 20);
  const before = requests.length;
  const result = await queryNomencladores("¿Cuántas horas reloj semanales y puntos de incompatibilidad tiene un maestro de grado de jornada completa de una escuela común?", generalRealList, centersRealList);
  assert.equal(requests.length, before); assert.equal(result.tipo, "respuesta_local");
  assert.match(result.respuesta, /Horas reloj semanales: 40/); assert.match(result.respuesta, /Puntos de incompatibilidad: 36/);
  assert.match(result.respuesta, /Decreto Acuerdo 2431\/2016/); assert.ok(result.referencias.some((reference: string) => /CAT2431-052/.test(reference)));
  assert.doesNotMatch(result.respuesta, /CAT2453-CE|CAT-1480/);
});

test("auditoría final RAG v3.1: métricas iguales permiten respuesta; otras métricas divergentes piden desambiguar", async () => {
  const variedCenters = structuredClone(centersRealList);
  const firstVice = variedCenters.registros.find((record: any) => /vicedirector/i.test(record.categoria));
  assert.ok(firstVice);
  firstVice.puntos_salariales += 1;
  firstVice.puntaje_incompatibilidad += 1;
  const hoursOnly = await queryNomencladores("¿Cuántas horas reloj tiene un vicedirector de Centros Educativos?", generalRealList, variedCenters);
  assert.equal(hoursOnly.tipo, "respuesta_local"); assert.match(hoursOnly.respuesta, /Horas reloj semanales: 35/);
  assert.doesNotMatch(hoursOnly.respuesta, /Puntos salariales:|Puntos de incompatibilidad:/);
  const salary = await queryNomencladores("¿Cuántos puntos salariales tiene un vicedirector de Centros Educativos?", generalRealList, variedCenters);
  assert.equal(salary.tipo, "aclaracion"); assert.match(salary.respuesta, /Alternativas:/);
  const incompatibility = await queryNomencladores("¿Cuántos puntos de incompatibilidad tiene un vicedirector de Centros Educativos?", generalRealList, variedCenters);
  assert.equal(incompatibility.tipo, "aclaracion"); assert.match(incompatibility.respuesta, /Alternativas:/);
});

test("auditoría final RAG v3.1: comparación conserva instituciones y niveles distintos", async () => {
  const institutions = await queryNomencladores("Compara Director de Centro Comunitario con Director de Centro Educativo", generalRealList, centersRealList);
  assert.equal(institutions.tipo, "respuesta_local", institutions.respuesta);
  assert.ok(institutions.referencias.some((reference: string) => /CAT2431-086/.test(reference)));
  assert.ok(institutions.referencias.some((reference: string) => /CAT2453-CE-001/.test(reference)));
  const levels = await queryNomencladores("Compara Director JIN 1ra. cat. con Director común 1ra. cat. Anexo Albergue JC", generalRealList, centersRealList);
  assert.equal(levels.tipo, "respuesta_local", levels.respuesta);
  assert.ok(levels.referencias.some((reference: string) => /CAT2431-018/.test(reference)));
  assert.ok(levels.referencias.some((reference: string) => /CAT2431-028/.test(reference)));
});

test("operativo v3.1: el nomenclador solicita precisión y ofrece alternativas cuando la comparación sigue ambigua", async () => {
  const result = await queryNomencladores("Compara un Director con un Vicedirector", generalRealList, centersRealList);
  assert.equal(result.tipo, "aclaracion"); assert.match(result.respuesta, /Alternativas documentadas/);
  assert.match(result.respuesta, /nivel educativo|institución|denominación exacta del cargo/);
  assert.match(result.respuesta, /director|vicedirector/i);
});

test("operativo v3.1: una institución compartida se propaga sin colapsar cargos distintos", async () => {
  const result = await queryNomencladores("Compara Director General con Vicedirector de nivel inicial de Centros Educativos", generalRealList, centersRealList);
  assert.equal(result.tipo, "respuesta_local", result.respuesta); assert.match(result.respuesta, /Director General/);
  assert.match(result.respuesta, /Vicedirector Educación Inicial/); assert.match(result.respuesta, /Centros Educativos/);
});

test("auditoría final RAG v3.1: un nivel al final no se impone a cargos con descriptores distintos", async () => {
  const result = await queryNomencladores("Compara Director JIN 1ra. cat. con Director común 1ra. cat. de Nivel Primario Común", generalRealList, centersRealList);
  assert.equal(result.tipo, "aclaracion"); assert.match(result.respuesta, /Alternativas documentadas/);
  assert.match(result.respuesta, /Director JIN|Director común/);
});

test("operativo v3.1: vicedirectores de Centros con métricas coincidentes se responden sin aclaración", async () => {
  const before = requests.length;
  const result = await queryNomencladores("¿Cuántas horas reloj y puntos de incompatibilidad tiene un vicedirector de Centros Educativos?", generalRealList, centersRealList);
  assert.equal(requests.length, before); assert.equal(result.tipo, "respuesta_local");
  assert.match(result.respuesta, /Inicial/); assert.match(result.respuesta, /Primaria/); assert.match(result.respuesta, /Secundaria/);
  assert.match(result.respuesta, /Horas reloj semanales: 35/); assert.match(result.respuesta, /Puntos de incompatibilidad: 30/);
});

test("v3 auditoría: comparación informa ausencia de horas reloj y compara solo magnitudes presentes", async () => {
  const result = await queryNomencladores("Compará Maestro de Grado Jornada Completa con Hora Cátedra Nivel Secundario", diverseGeneralList);
  assert.ok(result.referencias.some((reference: string) => /CAT2431-052/.test(reference)), `${JSON.stringify(result.referencias)} ${result.respuesta}`); assert.ok(result.referencias.some((reference: string) => /CAT2431-106/.test(reference)), JSON.stringify(result.referencias));
  assert.match(result.respuesta, /Horas reloj semanales: no hay datos compatibles en todos los registros/);
  assert.match(result.respuesta, /Puntos de incompatibilidad:.*registra 35 más que/s);
});

test("v3 auditoría: una POF se incluye solo si se solicita y aporta el registro institucional", async () => {
  assert.equal(completeCentersList.registros.length, 20);
  const result = await queryNomencladores("¿Cuántas horas reloj tiene un Maestro de Grado Jornada Extendida de la POF de Centros Educativos?", generalList, completeCentersList);
  assert.ok(result.referencias.some((reference: string) => /CAT2453-CE-017/.test(reference))); assert.match(result.respuesta, /POF Centro Educativo Nº 1 Riberas del Valle/);
  assert.match(result.respuesta, /Horas reloj semanales: 35/); assert.match(result.respuesta, /Jornada: Extendida/);
});

test("v3: consulta explicativa de artículo llega a Groq con el artículo aislado", async () => {
  const before = requests.length;
  const sources = [{ id: "estatuto_docente_provincial", dominio: "estatuto", referencia: "Ley Provincial 3122 Catamarca", url: "mock://statute-local", activo: true }];
  setWorkflowTestDependencies({ now: () => now, fetchJson: async (url) => url === "mock://manifest" ? { version: "direct-article", fuentes: sources } : estatutoRegression, groqClient: mockClient });
  const result = await runGroqWorkflow({ pregunta: "¿Qué establece el artículo 6 de la Ley Provincial 3122?", dominio: "estatuto" });
  assert.equal(requests.length, before + 1); assert.equal(result.origen, "groq_rag"); assert.match(requests.at(-1).messages[1].content, /ART[IÍ]CULO 6/); assert.match(requests.at(-1).messages[1].content, /estabilidad en el cargo/i); assert.doesNotMatch(requests.at(-1).messages[1].content, /ART[IÍ]CULO 7/); assert.ok(result.referencias.some((reference) => reference.includes("3122")));
});

test("auditoría final: literal aísla el artículo intermedio y conserva encabezados duplicados del mismo artículo", async () => {
  const literalDocument = [{
    articulo: 7, tipo: "articulo", dominio: "estatuto", documento: "Ley Provincial 3122", titulo_articulo: "Artículo 7",
    contenido: "ARTÍCULO 6°.- TEXTO ANTERIOR QUE NO DEBE APARECER.\n[ARTÍCULO 7°]\nCONTENIDO PERTINENTE A7 UNO.\nARTÍCULO 7°.- CONTENIDO PERTINENTE A7 DOS.\nARTÍCULO 8°.- TEXTO POSTERIOR QUE NO DEBE APARECER.",
  }];
  const sources = [{ id: "estatuto_docente_provincial", dominio: "estatuto", referencia: "Ley Provincial 3122 Catamarca", url: "mock://literal-articulo-intermedio", activo: true }];
  setWorkflowTestDependencies({ now: () => now, fetchJson: async (url) => url === "mock://manifest" ? { version: "literal-middle-article", fuentes: sources } : literalDocument, groqClient: mockClient });
  const before = requests.length;
  const result = await runGroqWorkflow({ pregunta: "Solicito el texto literal del artículo 7 de la Ley Provincial 3122" });
  assert.equal(requests.length, before); assert.equal(result.tipo, "respuesta_local");
  assert.match(result.respuesta, /CONTENIDO PERTINENTE A7 UNO/); assert.match(result.respuesta, /CONTENIDO PERTINENTE A7 DOS/);
  assert.doesNotMatch(result.respuesta, /TEXTO ANTERIOR|TEXTO POSTERIOR|ART[IÍ]CULO 6|ART[IÍ]CULO 8/);
});

test("v3: cuotas por minuto abren enfriamiento y el fallback conserva el texto normativo", async () => {
  now += 10_000;
  let providerCalls = 0;
  const rateLimited: any = { chat: { completions: { create: async () => { providerCalls++; const error: any = new Error("rate limit exceeded per minute"); error.status = 429; error.headers = new Headers({ "retry-after": "30", "x-ratelimit-limit-requests-minute": "30" }); throw error; } } } };
  const sources = [{ id: "movilidad_docente_decreto_533", dominio: "coberturas", referencia: "Decreto Acuerdo 533/2021 Catamarca", url: "mock://mobility-429", activo: true }];
  setWorkflowTestDependencies({ now: () => now, fetchJson: async (url) => url === "mock://manifest" ? { version: "429-minute", fuentes: sources } : source, groqClient: rateLimited });
  const [first, second] = await Promise.all([
    runGroqWorkflow({ pregunta: "¿Qué regla documental aplica a la movilidad docente? Decreto Acuerdo 533/2021" }),
    runGroqWorkflow({ pregunta: "¿Qué regla documental aplica a la movilidad docente? Decreto Acuerdo 533/2021" }),
  ]);
  assert.equal(first.origen, "local_rag_fallback_429"); assert.equal(second.origen, "local_rag_fallback_429"); assert.match(first.respuesta, /Facúltase.*cambios de Situación de Revista/s); assert.equal(providerCalls, 1);
});

test("v3: cuota diaria evita reintentos hasta que vence el enfriamiento", async () => {
  now += 10_000;
  let providerCalls = 0;
  const dailyLimited: any = { chat: { completions: { create: async () => { providerCalls++; if (providerCalls === 1) { const error: any = new Error("daily requests per day quota exceeded"); error.status = 429; throw error; } return { choices: [{ message: { content: "La documentación está disponible.", }, finish_reason: "stop" }], usage: {} }; } } } };
  const sources = [{ id: "movilidad_docente_decreto_533", dominio: "coberturas", referencia: "Decreto Acuerdo 533/2021 Catamarca", url: "mock://mobility-daily", activo: true }];
  setWorkflowTestDependencies({ now: () => now, fetchJson: async (url) => url === "mock://manifest" ? { version: "429-daily", fuentes: sources } : source, groqClient: dailyLimited });
  await runGroqWorkflow({ pregunta: "¿Qué regla documental aplica a la movilidad docente? Decreto Acuerdo 533/2021" });
  await runGroqWorkflow({ pregunta: "¿Qué regla documental aplica a la movilidad docente? Decreto Acuerdo 533/2021" });
  assert.equal(providerCalls, 1); now += 25 * 60 * 60 * 1000;
  await runGroqWorkflow({ pregunta: "¿Qué regla documental aplica a la movilidad docente? Decreto Acuerdo 533/2021" });
  assert.equal(providerCalls, 2);
});

test("auditoría final: agotamiento diario respeta un reset RPD informado de dos horas", async () => {
  now += 10_000; let providerCalls = 0;
  const limited: any = { chat: { completions: { create: async () => {
    providerCalls++;
    if (providerCalls === 1) { const error: any = new Error("daily requests per day quota exceeded"); error.status = 429; error.headers = new Headers({ "x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "2h" }); throw error; }
    return { choices: [{ message: { content: "La documentación está disponible." }, finish_reason: "stop" }], usage: {} };
  } } } };
  const sources = [{ id: "movilidad_docente_decreto_533", dominio: "coberturas", referencia: "Decreto Acuerdo 533/2021 Catamarca", url: "mock://rpd-reset-2h", activo: true }];
  setWorkflowTestDependencies({ now: () => now, fetchJson: async (url) => url === "mock://manifest" ? { version: "rpd-reset-2h", fuentes: sources } : source, groqClient: limited });
  const query = "¿Qué regla documental aplica a la movilidad docente? Decreto Acuerdo 533/2021";
  await runGroqWorkflow({ pregunta: query }); now += 2 * 60 * 60 * 1000 - 1; await runGroqWorkflow({ pregunta: query }); assert.equal(providerCalls, 1);
  now += 2; await runGroqWorkflow({ pregunta: query }); assert.equal(providerCalls, 2);
});

test("auditoría final: TPD y RPD eligen su reset sin intercambiar cabeceras", async () => {
  now += 10_000; let tokenCalls = 0;
  const tokenLimited: any = { chat: { completions: { create: async () => {
    tokenCalls++;
    if (tokenCalls === 1) { const error: any = new Error("daily tokens per day quota exceeded"); error.status = 429; error.headers = new Headers({ "x-ratelimit-remaining-tokens-day": "0", "x-ratelimit-reset-tokens-day": "2h", "x-ratelimit-reset-requests": "12h" }); throw error; }
    return { choices: [{ message: { content: "Respuesta documentada." }, finish_reason: "stop" }], usage: {} };
  } } } };
  const sources = [{ id: "movilidad_docente_decreto_533", dominio: "coberturas", referencia: "Decreto Acuerdo 533/2021 Catamarca", url: "mock://tpd-rpd-reset", activo: true }];
  setWorkflowTestDependencies({ now: () => now, fetchJson: async (url) => url === "mock://manifest" ? { version: "tpd-reset", fuentes: sources } : source, groqClient: tokenLimited });
  const query = "¿Qué regla documental aplica a la movilidad docente? Decreto Acuerdo 533/2021";
  await runGroqWorkflow({ pregunta: query }); now += 2 * 60 * 60 * 1000 + 1; await runGroqWorkflow({ pregunta: query }); assert.equal(tokenCalls, 2);

  now += 10_000; let requestCalls = 0;
  const requestLimited: any = { chat: { completions: { create: async () => {
    requestCalls++;
    if (requestCalls === 1) { const error: any = new Error("daily requests per day quota exceeded"); error.status = 429; error.headers = new Headers({ "x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "2h", "x-ratelimit-reset-tokens-day": "12h" }); throw error; }
    return { choices: [{ message: { content: "Respuesta documentada." }, finish_reason: "stop" }], usage: {} };
  } } } };
  setWorkflowTestDependencies({ now: () => now, fetchJson: async (url) => url === "mock://manifest" ? { version: "rpd-reset-selection", fuentes: sources.map((item) => ({ ...item, url: "mock://rpd-reset-selection" })) } : source, groqClient: requestLimited });
  await runGroqWorkflow({ pregunta: query }); now += 2 * 60 * 60 * 1000 + 1; await runGroqWorkflow({ pregunta: query }); assert.equal(requestCalls, 2);
});

test("auditoría final: agotamiento diario sin Retry-After ni reset conserva 24 horas", async () => {
  now += 10_000; let providerCalls = 0;
  const limited: any = { chat: { completions: { create: async () => {
    providerCalls++;
    if (providerCalls === 1) { const error: any = new Error("daily requests per day quota exceeded"); error.status = 429; error.headers = new Headers({ "x-ratelimit-remaining-requests": "0" }); throw error; }
    return { choices: [{ message: { content: "La documentación está disponible." }, finish_reason: "stop" }], usage: {} };
  } } } };
  const sources = [{ id: "movilidad_docente_decreto_533", dominio: "coberturas", referencia: "Decreto Acuerdo 533/2021 Catamarca", url: "mock://rpd-no-reset", activo: true }];
  setWorkflowTestDependencies({ now: () => now, fetchJson: async (url) => url === "mock://manifest" ? { version: "rpd-no-reset", fuentes: sources } : source, groqClient: limited });
  const query = "¿Qué regla documental aplica a la movilidad docente? Decreto Acuerdo 533/2021";
  await runGroqWorkflow({ pregunta: query }); now += 23 * 60 * 60 * 1000; await runGroqWorkflow({ pregunta: query }); assert.equal(providerCalls, 1);
  now += 2 * 60 * 60 * 1000; await runGroqWorkflow({ pregunta: query }); assert.equal(providerCalls, 2);
});

test("v3 auditoría: cuota RPM prevalece sobre cabeceras diarias y usa el reset por minuto", async () => {
  now += 10_000; let providerCalls = 0;
  const limited: any = { chat: { completions: { create: async () => {
    providerCalls++;
    if (providerCalls === 1) { const error: any = new Error("requests per minute rate limit"); error.status = 429; error.headers = new Headers({ "retry-after": "5", "x-ratelimit-limit-requests-day": "1000", "x-ratelimit-remaining-requests-day": "0", "x-ratelimit-reset-requests-day": "24h", "x-ratelimit-remaining-requests-minute": "0", "x-ratelimit-reset-requests-minute": "20s" }); throw error; }
    return { choices: [{ message: { content: "Respuesta documentada." }, finish_reason: "stop" }], usage: {} };
  } } } };
  const sources = [{ id: "movilidad_docente_decreto_533", dominio: "coberturas", referencia: "Decreto Acuerdo 533/2021 Catamarca", url: "mock://mobility-rpm-day-headers", activo: true }];
  setWorkflowTestDependencies({ now: () => now, fetchJson: async (url) => url === "mock://manifest" ? { version: "rpm-day-headers", fuentes: sources } : source, groqClient: limited });
  const query = "¿Qué regla documental aplica a la movilidad docente? Decreto Acuerdo 533/2021";
  await runGroqWorkflow({ pregunta: query }); now += 19_000; await runGroqWorkflow({ pregunta: query }); assert.equal(providerCalls, 1);
  now += 2_000; await runGroqWorkflow({ pregunta: query }); assert.equal(providerCalls, 2);
});

test("v3 auditoría: Retry-After superior a 60 segundos no se recorta", async () => {
  now += 10_000; let providerCalls = 0;
  const limited: any = { chat: { completions: { create: async () => {
    providerCalls++; if (providerCalls === 1) { const error: any = new Error("rate limit per minute"); error.status = 429; error.headers = new Headers({ "retry-after": "120" }); throw error; }
    return { choices: [{ message: { content: "Respuesta documentada." }, finish_reason: "stop" }], usage: {} };
  } } } };
  const sources = [{ id: "movilidad_docente_decreto_533", dominio: "coberturas", referencia: "Decreto Acuerdo 533/2021 Catamarca", url: "mock://mobility-retry-120", activo: true }];
  setWorkflowTestDependencies({ now: () => now, fetchJson: async (url) => url === "mock://manifest" ? { version: "retry-120", fuentes: sources } : source, groqClient: limited });
  const query = "¿Qué regla documental aplica a la movilidad docente? Decreto Acuerdo 533/2021";
  await runGroqWorkflow({ pregunta: query }); now += 65_000; await runGroqWorkflow({ pregunta: query }); assert.equal(providerCalls, 1);
  now += 56_000; await runGroqWorkflow({ pregunta: query }); assert.equal(providerCalls, 2);
});

test("v3 auditoría: cola Groq limita la espera y responde localmente cuando se satura", async () => {
  now += 10_000; let providerCalls = 0;
  const slow: any = { chat: { completions: { create: async () => {
    providerCalls++; await new Promise((resolve) => setTimeout(resolve, 250));
    return { choices: [{ message: { content: "Respuesta documentada." }, finish_reason: "stop" }], usage: {} };
  } } } };
  const sources = [{ id: "movilidad_docente_decreto_533", dominio: "coberturas", referencia: "Decreto Acuerdo 533/2021 Catamarca", url: "mock://mobility-queue", activo: true }];
  setWorkflowTestDependencies({ now: () => now, fetchJson: async (url) => url === "mock://manifest" ? { version: "queue-bounded", fuentes: sources } : source, groqClient: slow });
  const started = Date.now();
  const outcomes = await Promise.all(Array.from({ length: 10 }, (_, index) => runGroqWorkflow({ pregunta: `¿Qué regla aplica a movilidad docente caso ${index}? Decreto Acuerdo 533/2021` })));
  assert.ok(Date.now() - started < 2_500); assert.equal(outcomes.length, 10); assert.ok(outcomes.some((result) => result.origen === "local_rag_fallback"));
  assert.ok(providerCalls <= 9);
});

test("v3: reintenta una sola vez con backoff corto ante Retry-After breve", async () => {
  now += 10_000;
  let providerCalls = 0;
  const transientLimit: any = { chat: { completions: { create: async () => {
    providerCalls++;
    if (providerCalls === 1) { const error: any = new Error("rate limit"); error.status = 429; error.headers = new Headers({ "retry-after": "0.01" }); throw error; }
    return { choices: [{ message: { content: "La movilidad docente se rige por el fragmento recuperado." }, finish_reason: "stop" }], usage: {} };
  } } } };
  const sources = [{ id: "movilidad_docente_decreto_533", dominio: "coberturas", referencia: "Decreto Acuerdo 533/2021 Catamarca", url: "mock://mobility-retry", activo: true }];
  setWorkflowTestDependencies({ now: () => now, fetchJson: async (url) => url === "mock://manifest" ? { version: "429-retry", fuentes: sources } : source, groqClient: transientLimit });
  const result = await runGroqWorkflow({ pregunta: "¿Qué regla documental aplica a la movilidad docente? Decreto Acuerdo 533/2021" });
  assert.equal(providerCalls, 2); assert.equal(result.origen, "groq_rag");
});

test("v3: 404 de modelo no cambia el modelo y responde con fallback documental", async () => {
  now += 10_000;
  let providerCalls = 0;
  const missingModel: any = { chat: { completions: { create: async () => { providerCalls++; const error: any = new Error("model not found"); error.status = 404; throw error; } } } };
  const sources = [{ id: "movilidad_docente_decreto_533", dominio: "coberturas", referencia: "Decreto Acuerdo 533/2021 Catamarca", url: "mock://mobility-404", activo: true }];
  setWorkflowTestDependencies({ now: () => now, fetchJson: async (url) => url === "mock://manifest" ? { version: "404", fuentes: sources } : source, groqClient: missingModel });
  const result = await runGroqWorkflow({ pregunta: "¿Qué regla documental aplica a la movilidad docente? Decreto Acuerdo 533/2021" });
  assert.equal(providerCalls, 1); assert.equal(result.origen, "local_rag_fallback"); assert.match(result.respuesta, /cambios de Situación de Revista/); assert.doesNotMatch(result.respuesta, /llama-3\.1-8b-instant/);
});

test("v3: consulta simple mantiene contexto completo con presupuesto de tokens adaptativo", async () => {
  now += 10_000;
  const sources = [{ id: "movilidad_docente_decreto_533", dominio: "coberturas", referencia: "Decreto Acuerdo 533/2021 Catamarca", url: "mock://mobility-budget", activo: true }];
  setWorkflowTestDependencies({ now: () => now, fetchJson: async (url) => url === "mock://manifest" ? { version: "budget", fuentes: sources } : source, groqClient: mockClient });
  const before = requests.length;
  await runGroqWorkflow({ pregunta: "¿Qué regla documental aplica a la movilidad docente? Decreto Acuerdo 533/2021" });
  const request = requests[before]; const serialized = JSON.stringify(request.messages);
  const estimatedInputTokens = Math.ceil(request.messages.reduce((total: number, message: any) => total + String(message.content || "").length, 0) / 4);
  assert.ok(request.max_tokens <= 600); assert.ok(estimatedInputTokens <= 900); assert.ok(estimatedInputTokens + request.max_tokens <= 1500);
  assert.ok(Math.ceil(serialized.length / 4) < 1500); assert.match(serialized, /cambios de Situación de Revista/);
});

test("auditoría final: el reintento por respuesta incompleta cuenta entrada y salida en el presupuesto combinado", async () => {
  now += 10_000;
  const sources = [{ id: "movilidad_docente_decreto_533", dominio: "coberturas", referencia: "Decreto Acuerdo 533/2021 Catamarca", url: "mock://mobility-retry-budget", activo: true }];
  let providerCalls = 0;
  const incompleteThenComplete: any = { chat: { completions: { create: async (request: any) => {
    requests.push(request); providerCalls++;
    return providerCalls === 1
      ? { choices: [{ message: { content: "La norma dispone" }, finish_reason: "length" }], usage: {} }
      : { choices: [{ message: { content: "La documentación indica el cambio de situación de revista según el fragmento recuperado." }, finish_reason: "stop" }], usage: {} };
  } } } };
  setWorkflowTestDependencies({ now: () => now, fetchJson: async (url) => url === "mock://manifest" ? { version: "retry-budget", fuentes: sources } : source, groqClient: incompleteThenComplete });
  const before = requests.length;
  const result = await runGroqWorkflow({ pregunta: "Explicá la regla documental aplicable a la movilidad docente según el Decreto Acuerdo 533/2021" });
  const calls = requests.slice(before);
  const estimatedCombinedTokens = calls.reduce((total, request) => {
    const estimatedInput = Math.ceil(request.messages.reduce((sum: number, message: any) => sum + String(message.content || "").length, 0) / 4);
    return total + estimatedInput + request.max_tokens;
  }, 0);
  assert.equal(providerCalls, 2); assert.equal(calls.length, 2); assert.equal(result.origen, "groq_rag");
  assert.ok(estimatedCombinedTokens <= 7000, `estimated combined generation cost ${estimatedCombinedTokens} exceeded the per-request budget`);
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
    // El servidor TypeScript puede tardar más en iniciar bajo carga de CI.
    for(let attempt=0;attempt<180;attempt++){
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
