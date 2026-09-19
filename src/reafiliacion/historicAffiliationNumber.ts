export type HistoricAffiliationRecord = Record<string, any> & {
  id?: string;
  path?: string;
};

export type HistoricAffiliationResolution = {
  nroAfiliacion: number;
  source: "usuario" | "counter" | "usuarioId" | "dni" | "counter_last";
  documento: HistoricAffiliationRecord | null;
};

export function normalizarNumeroAfiliacion(value: unknown): number | null {
  if (typeof value === "number") {
    return Number.isSafeInteger(value) && value > 0 ? value : null;
  }

  const raw = String(value ?? "").trim();
  if (!/^\d+$/.test(raw)) return null;

  const parsed = Number(raw);
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : null;
}

function conflicto(message: string): never {
  throw Object.assign(new Error(message), { statusCode: 409 });
}

function claveDocumento(documento: HistoricAffiliationRecord): string {
  return String(documento.path || documento.id || "");
}

function documentosUnicos(documentos: HistoricAffiliationRecord[]): HistoricAffiliationRecord[] {
  const unicos = new Map<string, HistoricAffiliationRecord>();
  for (const documento of documentos) {
    const clave = claveDocumento(documento);
    if (!unicos.has(clave)) unicos.set(clave, documento);
  }
  return [...unicos.values()].sort((a, b) => claveDocumento(a).localeCompare(claveDocumento(b)));
}

function dniNormalizado(value: unknown): string {
  return String(value ?? "").replace(/\D/g, "");
}

function seleccionarDocumento(
  documentos: HistoricAffiliationRecord[],
  dni: string
): HistoricAffiliationRecord | null {
  const unicos = documentosUnicos(documentos);
  return unicos.find((documento) => dniNormalizado(documento.dni) === dni) || unicos[0] || null;
}

function resolverDocumentos(
  documentos: HistoricAffiliationRecord[],
  usuarioIdHistorico: string,
  dni: string,
  mensajeConflicto: string
): { nroAfiliacion: number; documento: HistoricAffiliationRecord | null } | null {
  const unicos = documentosUnicos(documentos);
  if (!unicos.length) return null;

  const conDni = unicos.filter((documento) => dniNormalizado(documento.dni) === dni);
  const candidatos = conDni.length ? conDni : unicos;
  const deOtraPersona = candidatos.find((documento) => {
    const propietario = String(documento.usuarioId || "").trim();
    return propietario && propietario !== usuarioIdHistorico;
  });
  if (deOtraPersona) {
    conflicto("El nuevoAfiliado existente pertenece a otra persona.");
  }

  const numeros = [...new Set(
    candidatos
      .map((documento) => normalizarNumeroAfiliacion(documento.nroAfiliacion))
      .filter((numero): numero is number => numero !== null)
  )];
  if (numeros.length > 1) {
    conflicto(mensajeConflicto);
  }
  if (numeros.length === 0) return null;

  return { nroAfiliacion: numeros[0], documento: candidatos[0] || null };
}

export function resolverNumeroAfiliacionHistorico(input: {
  usuario: HistoricAffiliationRecord;
  counter: HistoricAffiliationRecord;
  usuarioIdHistorico: string;
  dni: string;
  nuevoAfiliadosPorUsuarioId: HistoricAffiliationRecord[];
  nuevoAfiliadosPorDni: HistoricAffiliationRecord[];
}): HistoricAffiliationResolution {
  const numeroUsuario = normalizarNumeroAfiliacion(input.usuario.nroAfiliacion);
  if (numeroUsuario !== null) {
    return {
      nroAfiliacion: numeroUsuario,
      source: "usuario",
      documento: seleccionarDocumento(input.nuevoAfiliadosPorUsuarioId, input.dni),
    };
  }

  for (const campo of ["nroAfiliacion", "nroAfiliacionHistorico", "nroAfiliacionReafiliacion"]) {
    const numeroCounter = normalizarNumeroAfiliacion(input.counter[campo]);
    if (numeroCounter !== null) {
      return {
        nroAfiliacion: numeroCounter,
        source: "counter",
        documento: seleccionarDocumento(input.nuevoAfiliadosPorUsuarioId, input.dni),
      };
    }
  }

  const porUsuario = resolverDocumentos(
    input.nuevoAfiliadosPorUsuarioId,
    input.usuarioIdHistorico,
    input.dni,
    "Existen múltiples números históricos de afiliación. Requiere revisión administrativa."
  );
  if (porUsuario) {
    return { ...porUsuario, source: "usuarioId" };
  }

  const porDni = resolverDocumentos(
    input.nuevoAfiliadosPorDni,
    input.usuarioIdHistorico,
    input.dni,
    "Existen múltiples números históricos de afiliación. Requiere revisión administrativa."
  );
  if (porDni) {
    return { ...porDni, source: "dni" };
  }

  const numeroLast = normalizarNumeroAfiliacion(input.counter.last);
  if (numeroLast !== null) {
    return {
      nroAfiliacion: numeroLast,
      source: "counter_last",
      documento: null,
    };
  }

  conflicto("No se pudo determinar el número histórico de afiliación. Requiere revisión administrativa.");
}
