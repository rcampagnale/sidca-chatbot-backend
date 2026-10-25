export const TEXTO_AUSPICIO_DEFAULT_MINISTERIO =
  "Este evento de capacitación, fortalecimiento y actualización docente fue auspiciado por el Ministerio de Educación y Trabajo, a través de Resolución";

const texto = (valor: unknown) => String(valor ?? "").replace(/\s+/g, " ").trim();

/**
 * Distingue una propiedad legacy ausente de una propiedad guardada vacía.
 * El default sólo corresponde al primer caso.
 */
export const textoAuspicioConfiguracion = (registro: Record<string, any> | null | undefined) => {
  if (registro && Object.prototype.hasOwnProperty.call(registro, "textoAuspicio")) {
    return String(registro.textoAuspicio ?? "");
  }
  return TEXTO_AUSPICIO_DEFAULT_MINISTERIO;
};

// El texto administrado es literal. El campo no se modifica al hidratar ni
// guardar; la composición visual sólo elimina un prefijo legacy si aparece.
export const normalizarTextoAuspicio = (valor: unknown) => texto(valor);

const normalizarNumeroResolucion = (valor: unknown) =>
  texto(valor).replace(/^resoluci[oó]n\b[\s:.-]*/i, "");

export const textoAuspicioConResolucion = (auspicio: unknown, resolucion: unknown) => {
  const base = texto(auspicio)
    .replace(/\s*S\.I\.E\.\(MECYT\)\s*N[°º]\s*/gi, " ")
    .replace(/\s+/g, " ")
    .trim();
  const numero = normalizarNumeroResolucion(resolucion);
  const conPunto = (valor: string) =>
    valor && !/[.!?]$/.test(valor) ? `${valor}.` : valor;
  if (!base) {
    return numero
      ? `Resolución ${numero}.`
      : "";
  }
  const partes = [base];
  if (numero) partes.push(numero);
  return conPunto(partes.join(" "));
};
