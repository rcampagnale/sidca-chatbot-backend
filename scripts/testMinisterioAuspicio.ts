import assert from "node:assert/strict";
import {
  TEXTO_AUSPICIO_DEFAULT_MINISTERIO,
  textoAuspicioConfiguracion,
  textoAuspicioConResolucion,
} from "../src/certificados/ministerioAuspicio.js";

const legacy = {};
const personalizadoA = { textoAuspicio: "Texto A" };
const personalizadoB = { textoAuspicio: "Texto B" };

// Caso A: un documento legacy sin la propiedad recibe el default.
assert.equal(textoAuspicioConfiguracion(legacy), TEXTO_AUSPICIO_DEFAULT_MINISTERIO);
assert.equal(textoAuspicioConfiguracion(legacy).endsWith("a través de Resolución"), true);
assert.equal(textoAuspicioConfiguracion(legacy).includes("S.I.E.(MECYT) N°"), false);

// Caso B: un texto personalizado viaja intacto hasta el renderer.
assert.equal(
  textoAuspicioConfiguracion({ textoAuspicio: "Texto personalizado de prueba" }),
  "Texto personalizado de prueba"
);

// Caso C: cambiar A por B no vuelve a imponer el valor anterior.
assert.equal(textoAuspicioConfiguracion(personalizadoA), "Texto A");
assert.equal(textoAuspicioConfiguracion(personalizadoB), "Texto B");
assert.equal(textoAuspicioConfiguracion({ textoAuspicio: "" }), "");

// Caso D: el renderer compone el texto vigente con la resolución separada.
assert.equal(
  textoAuspicioConResolucion(
    textoAuspicioConfiguracion({ textoAuspicio: "Texto personalizado de prueba" }),
    "RESOL-2026-35-E-CAT-SICE#MET"
  ),
  "Texto personalizado de prueba RESOL-2026-35-E-CAT-SICE#MET."
);

// Caso E: el prefijo no depende de que el texto termine en "Resolución".
assert.equal(
  textoAuspicioConResolucion(
    "Este evento fue auspiciado por el Ministerio, a través de Resolución",
    "RESOL-2026-35-E-CAT-SICE#MET"
  ),
  "Este evento fue auspiciado por el Ministerio, a través de Resolución RESOL-2026-35-E-CAT-SICE#MET."
);

// Caso F: el contenido posterior a "Resolución" se conserva literalmente.
assert.equal(
  textoAuspicioConResolucion(
    "Resolución emitida para el evento",
    "RESOL-2026-35-E-CAT-SICE#MET"
  ),
  "Resolución emitida para el evento RESOL-2026-35-E-CAT-SICE#MET."
);

// Caso G: un texto sin la palabra "Resolución" recibe igualmente el prefijo.
assert.equal(
  textoAuspicioConResolucion(
    "Texto personalizado cualquiera",
    "RESOL-2026-35-E-CAT-SICE#MET"
  ),
  "Texto personalizado cualquiera RESOL-2026-35-E-CAT-SICE#MET."
);

// Caso H: una configuración legacy que ya contiene el prefijo no lo duplica.
assert.equal(
  textoAuspicioConResolucion(
    "Texto legacy Resolución S.I.E.(MECYT) N°",
    "RESOL-2026-35-E-CAT-SICE#MET"
  ),
  "Texto legacy Resolución RESOL-2026-35-E-CAT-SICE#MET."
);

console.log("OK: flujo de textoAuspicio legacy, personalizado, reemplazo y resolución independiente");
