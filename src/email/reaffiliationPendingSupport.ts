export type PendingEmailDecision = "procesar" | "omitido" | "ya_enviado" | "procesando";
export type ApprovedEmailDecision = "procesar" | "omitido" | "ya_enviado" | "procesando";

export type PendingEmailState = Record<string, any> & {
  estadoReafiliacion?: unknown;
  requiereRevisionComision?: unknown;
  fechaSolicitudReafiliacion?: unknown;
  usuarioIdHistorico?: unknown;
  correoReafiliacionPendienteEstado?: unknown;
  correoReafiliacionPendienteSolicitudId?: unknown;
  correoReafiliacionPendienteProcesandoAt?: unknown;
  fechaResolucionReafiliacion?: unknown;
  fechaAprobacion?: unknown;
  correoReafiliacionAprobadaEstado?: unknown;
  correoReafiliacionAprobadaSolicitudId?: unknown;
  correoReafiliacionAprobadaProcesandoAt?: unknown;
};

export function normalizarEstadoReafiliacion(value: unknown): string {
  return String(value ?? "")
    .trim()
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "");
}

export function solicitudIdDesdeFecha(value: unknown): string | null {
  const raw = String(value ?? "").trim();
  if (!raw || !Number.isFinite(Date.parse(raw))) return null;
  return raw;
}

export function esSolicitudPendienteValida(state: PendingEmailState): boolean {
  return (
    normalizarEstadoReafiliacion(state.estadoReafiliacion) === "pendiente" &&
    state.requiereRevisionComision === true &&
    solicitudIdDesdeFecha(state.fechaSolicitudReafiliacion) !== null &&
    String(state.usuarioIdHistorico ?? "").trim().length > 0
  );
}

export function decidirCorreoPendiente(
  state: PendingEmailState,
  solicitudId: string,
  ahora = Date.now(),
): PendingEmailDecision {
  if (!esSolicitudPendienteValida(state)) return "omitido";

  const mismaSolicitud = String(state.correoReafiliacionPendienteSolicitudId ?? "") === solicitudId;
  const estadoCorreo = String(state.correoReafiliacionPendienteEstado ?? "");
  if (mismaSolicitud && estadoCorreo === "enviado") return "ya_enviado";

  if (mismaSolicitud && estadoCorreo === "procesando") {
    const procesandoEn = Date.parse(String(state.correoReafiliacionPendienteProcesandoAt ?? ""));
    if (Number.isFinite(procesandoEn) && ahora - procesandoEn < 15 * 60 * 1000) {
      return "procesando";
    }
  }

  return "procesar";
}

export function esSolicitudAprobadaValida(state: PendingEmailState): boolean {
  return (
    normalizarEstadoReafiliacion(state.estadoReafiliacion) === "aprobada" &&
    solicitudIdDesdeFecha(state.fechaSolicitudReafiliacion) !== null &&
    String(state.usuarioIdHistorico ?? "").trim().length > 0
  );
}

export function decidirCorreoAprobada(
  state: PendingEmailState,
  solicitudId: string,
  ahora = Date.now(),
): ApprovedEmailDecision {
  if (!esSolicitudAprobadaValida(state)) return "omitido";

  const mismaSolicitud = String(state.correoReafiliacionAprobadaSolicitudId ?? "") === solicitudId;
  const estadoCorreo = String(state.correoReafiliacionAprobadaEstado ?? "");
  if (mismaSolicitud && estadoCorreo === "enviado") return "ya_enviado";

  if (mismaSolicitud && estadoCorreo === "procesando") {
    const procesandoEn = Date.parse(String(state.correoReafiliacionAprobadaProcesandoAt ?? ""));
    if (Number.isFinite(procesandoEn) && ahora - procesandoEn < 15 * 60 * 1000) {
      return "procesando";
    }
  }

  return "procesar";
}

export function emailValidoReafiliacion(value: unknown): value is string {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(String(value ?? "").trim());
}
