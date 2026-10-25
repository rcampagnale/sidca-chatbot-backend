import assert from "node:assert/strict";
import {
  nombreInstitucionOrganizadora,
  normalizarInstitucionOrganizadora,
  resolverInstitucionOrganizadora,
} from "../src/certificados/institucionOrganizadora.js";

assert.equal(normalizarInstitucionOrganizadora("SIDCA"), "sidca");
assert.equal(normalizarInstitucionOrganizadora("SiDCa"), "sidca");
assert.equal(normalizarInstitucionOrganizadora("ministerio_sidca"), "sidca");
assert.equal(
  nombreInstitucionOrganizadora({
    institucionCertificado: "ministerio",
    institucionOrganizadora: "sidca",
  }),
  "SINDICATO DE DOCENTES DE CATAMARCA - SI.D.CA."
);

assert.equal(normalizarInstitucionOrganizadora("ITM"), "itm");
assert.equal(
  nombreInstitucionOrganizadora({
    institucionCertificado: "ministerio",
    institucionOrganizadora: "instituto tecnológico municipal",
  }),
  "INSTITUTO TECNOLÓGICO MUNICIPAL"
);

assert.equal(resolverInstitucionOrganizadora({ institucionCertificado: "ministerio" }), "");
assert.equal(nombreInstitucionOrganizadora({ institucionCertificado: "ministerio" }), "");

console.log("OK: resolver institucional Ministerio SIDCA/ITM y legacy");
