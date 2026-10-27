export const NOMBRES_INSTITUCIONES_ORGANIZADORAS = {
  sidca: "SINDICATO DE DOCENTES DE CATAMARCA - SI.D.CA.",
  itm: "INSTITUTO TECNOLOGICO MUNICIPAL (I.T.M)",
} as const;

export type InstitucionOrganizadora = keyof typeof NOMBRES_INSTITUCIONES_ORGANIZADORAS;

const normalizarClave = (valor: unknown) =>
  String(valor ?? "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .trim()
    .toLowerCase()
    .replace(/[._-]+/g, " ")
    .replace(/\s+/g, " ");

/**
 * Sólo acepta instituciones conocidas. Un valor ausente o desconocido queda
 * vacío para conservar la salida histórica sin inventar una institución.
 */
export const normalizarInstitucionOrganizadora = (
  valor: unknown
): InstitucionOrganizadora | "" => {
  const clave = normalizarClave(valor);
  if (!clave) return "";

  if (
    clave === "itm" ||
    clave.includes("instituto tecnologico municipal")
  ) {
    return "itm";
  }

  if (
    clave === "sidca" ||
    clave.includes("sindicato de docentes de catamarca") ||
    clave.includes("sidca")
  ) {
    return "sidca";
  }

  return "";
};

/**
 * Contrato común para configuración y snapshots. Los nombres alternativos
 * sólo sirven para leer datos legacy; `institucionOrganizadora` es la fuente
 * actual y explícita.
 */
export const resolverInstitucionOrganizadora = (
  record: Record<string, any> | null | undefined
): InstitucionOrganizadora | "" => {
  const candidatos = [
    record?.institucionOrganizadora,
    record?.institutionKey,
    record?.institucion,
    record?.organizador,
    record?.entidad,
    record?.issuer,
    record?.institucionCertificado,
  ];

  for (const candidato of candidatos) {
    const clave = normalizarInstitucionOrganizadora(candidato);
    if (clave) return clave;
  }

  return "";
};

export const nombreInstitucionOrganizadora = (
  record: Record<string, any> | null | undefined
) => {
  const clave = resolverInstitucionOrganizadora(record);
  return clave ? NOMBRES_INSTITUCIONES_ORGANIZADORAS[clave] : "";
};
