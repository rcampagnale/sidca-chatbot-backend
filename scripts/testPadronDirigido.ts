import assert from "node:assert/strict";
import { resolverPadronDesdeFuentes } from "../src/certificados/afiliacion.js";

const texto = (valor: string) => ({ stringValue: valor });
const entero = (valor: number) => ({ integerValue: String(valor) });
const booleano = (valor: boolean) => ({ booleanValue: valor });

const documento = (
  dni: string | number,
  activo?: boolean,
  departamento?: string
) => ({
  dni: typeof dni === "number" ? entero(dni) : texto(dni),
  ...(typeof activo === "boolean" ? { activo: booleano(activo) } : {}),
  ...(departamento ? { departamento: texto(departamento) } : {}),
});

const normalizar = (dni: unknown) => String(dni ?? "").replace(/\D/g, "");
const dniDocumento = (campos: Record<string, any>) =>
  normalizar(
    campos.dni?.stringValue ??
      campos.dni?.integerValue ??
      campos.dni?.doubleValue ??
      ""
  );

const dnis = ["100", "200", "300", "400", "500"];
const fuentesCompletas = {
  // 100: adherente bloqueado por nuevoAfiliado, aunque usuarios diga true.
  // 200: adherente habilitado por el respaldo de usuarios.
  adherentes: [documento("100"), documento(200), documento("500")],
  nuevoAfiliado: [
    documento(100, false, "Valle Viejo"),
    documento("300", undefined, "Tinogasta"),
    documento("500", true, "Capital"),
    documento(500, false, "Pomán"),
  ],
  usuarios: [
    documento("100", true, "Capital"),
    documento("200", true, "Andalgalá"),
    documento(300, false, "Belén"),
  ],
};

const solicitados = new Set(dnis);
const fuentesDirigidas = Object.fromEntries(
  Object.entries(fuentesCompletas).map(([coleccion, documentos]) => [
    coleccion,
    documentos.filter((campos) => solicitados.has(dniDocumento(campos))),
  ])
) as typeof fuentesCompletas;

const resultadoCompleto = resolverPadronDesdeFuentes(dnis, fuentesCompletas);
const resultadoDirigido = resolverPadronDesdeFuentes(dnis, fuentesDirigidas);

assert.deepEqual([...resultadoDirigido], [...resultadoCompleto]);
assert.equal(resultadoDirigido.get("100")?.afiliacion.habilitadoCertificado, false);
assert.equal(resultadoDirigido.get("200")?.afiliacion.habilitadoCertificado, true);
assert.equal(resultadoDirigido.get("300")?.afiliacion.tipo, "cotizante");
assert.equal(resultadoDirigido.get("400")?.afiliacion.tipo, "no_verificada");
assert.equal(resultadoDirigido.get("500")?.afiliacion.habilitadoCertificado, false);
assert.equal(resultadoDirigido.get("100")?.departamento.crudo, "Valle Viejo");

console.log(
  "[padron-test] OK: la proyección dirigida coincide con la completa para 5 DNI."
);
