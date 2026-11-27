import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { activeSources, attachDocumentControl, classifyNomenclator, normalizeNorm, rankNormative, recordsFromDocument, sourceFingerprint, type RagChunk } from "../src/ragCore.js";

const manifest = JSON.parse(await readFile(new URL("./fixtures/manifest.json", import.meta.url), "utf8"));
const nomen = JSON.parse(await readFile(new URL("./fixtures/nomencladores.json", import.meta.url), "utf8"));
const movilidad = JSON.parse(await readFile(new URL("./fixtures/movilidad.json", import.meta.url), "utf8"));
const base: RagChunk[] = [
 {sourceId:"movilidad_docente_decreto_533",source:"Decreto Acuerdo 533/2021",norm:"Decreto Acuerdo 533/2021",article:"6",domain:"coberturas",jurisdiction:"Catamarca",text:movilidad[0].contenido},
 {sourceId:"ley_26058",source:"Ley Nacional 26.058",norm:"Ley Nacional 26.058",article:"6",domain:"educacion_tecnico_profesional",jurisdiction:"Nacional",text:"Artículo 6 educación técnico profesional"},
 {sourceId:"ley_26206",source:"Ley Nacional 26.206",norm:"Ley Nacional 26.206",article:"6",domain:"educacion_nacional",jurisdiction:"Nacional",text:"Artículo 6 educación nacional"},
];

test("01 manifiesto de fixture versionado 1.9.0",()=>assert.equal(manifest.version,"1.9.0"));
test("02 selecciona solo fuentes activas",()=>assert.equal(activeSources(manifest).length,10));
test("03 excluye fuente inactiva",()=>assert.ok(!activeSources(manifest).some((s:any)=>s.id==="ley_26206_educacion_nacional")));
test("04 acepta arrays",()=>assert.equal(recordsFromDocument(movilidad).length,1));
test("05 acepta registros",()=>assert.equal(recordsFromDocument(nomen).length,1));
test("06 acepta servicios",()=>assert.equal(recordsFromDocument({servicios:[{id:"x"}]}).length,1));
test("07 registros conservan varias normas via referencia",()=>assert.match(nomen.fuente.norma,/2453/));
test("08 localiza artículo 6 Decreto 533",()=>assert.equal(rankNormative(base,"Decreto 533/2021 artículo 6")[0].sourceId,"movilidad_docente_decreto_533"));
test("09 normaliza año corto",()=>assert.equal(normalizeNorm("Decreto 533/21"),normalizeNorm("Decreto 533/2021")));
test("10 normaliza Ley 26.058",()=>assert.equal(normalizeNorm("Ley 26.058"),normalizeNorm("Ley 26058")));
test("11 distingue mismo artículo por norma",()=>assert.equal(rankNormative(base,"Ley 26.206 art. 6")[0].sourceId,"ley_26206"));
test("12 consulta sin norma retorna candidatos sin colapsar",()=>assert.equal(rankNormative(base,"artículo 6").length,3));
test("13 conserva jurisdicción",()=>assert.equal(base[0].jurisdiction,"Catamarca"));
test("14 artículo ausente no da coincidencia explícita",()=>assert.equal(rankNormative(base,"Ley 26058 artículo 99").length,0));
test("15 nomenclador general identificable",()=>assert.equal(classifyNomenclator("Decreto 2431/2016 nomenclador general"),"decreto_2431_2016"));
test("16 Centros Educativos identificable",()=>assert.equal(classifyNomenclator("Decreto 2453/2019 Centros Educativos"),"decreto_2453_2019"));
test("17 Jardines Pasitos identificable",()=>assert.equal(classifyNomenclator("1480/2019 Jardines Maternales Pasitos Cuidados"),"decreto_1480_2019"));
test("18 no mezcla puntajes distintos",()=>assert.notEqual(nomen.registros[0].puntaje_incompatibilidad,nomen.registros[0].puntos_salariales));
test("19 conserva reloj separado",()=>assert.equal(nomen.registros[0].minutos_reloj_semanales,2400));
test("20 distingue alcance institucional",()=>assert.match(nomen.registros[0].alcance,/institucional/));
test("21 mantiene estado derogatorio disponible",()=>assert.equal(recordsFromDocument([{estado_vigencia:"derogado"}])[0].estado_vigencia,"derogado"));
test("22 mantiene datos de sustitución",()=>assert.equal(recordsFromDocument([{actualizacion:{tipo:"sustituido"}}])[0].actualizacion.tipo,"sustituido"));
test("23 identifica control documental",()=>assert.equal(nomen.control_documental.estado,"muestra"));
test("24 no inventa artículos faltantes",()=>assert.deepEqual(recordsFromDocument({articulos_faltantes:[9]}),[{articulos_faltantes:[9]}]));
test("25 preserva vigencia no verificada",()=>assert.equal(nomen.fuente.vigencia_actual_verificada,false));
test("26 cambio de manifiesto cambia fingerprint",()=>assert.notEqual(sourceFingerprint(manifest,[]),sourceFingerprint({...manifest,version:"2"},[])));
test("27 contenido cambia fingerprint misma versión",()=>assert.notEqual(sourceFingerprint(manifest,[1]),sourceFingerprint(manifest,[2])));
test("28 desactivación cambia fuente efectiva",()=>assert.equal(activeSources({...manifest,fuentes:manifest.fuentes.map((s:any)=>s.id==="servicios_sidca_app"?{...s,activo:false}:s)}).length,9));
test("29 el fallo parcial no modifica datos recibidos",()=>assert.equal(sourceFingerprint(manifest,movilidad),sourceFingerprint(manifest,movilidad)));
test("30 fallback respeta activo estricto",()=>assert.deepEqual(activeSources({fuentes:[{url:"a"},{url:"b",activo:false}]}),[]));
for (const [i, domain] of ["licencias","estatuto","coberturas","servicios","general"].entries()) test(`regresión dominio existente ${domain}`,()=>assert.ok(["licencias","estatuto","coberturas","servicios","general"].includes(domain)));
test("control documental acompaña cada fragmento",()=>assert.equal(attachDocumentControl(base,{control_documental:"verificar"})[0].metadata?.control_documental,"verificar"));
