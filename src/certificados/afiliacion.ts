// src/certificados/afiliacion.ts
//
// Condición sindical de un participante, para decidir si puede emitir o
// descargar certificados.
//
// La regla NO se inventa acá: replica la que ya usa
// Admin → Adherentes → Estado en el frontend (Adherentes.js, `estadoUI` +
// `getActivoFromNuevo`). Este módulo es la ÚNICA implementación en el backend,
// y la comparten el listado de aprobados, la emisión individual, la masiva y
// la selección de certificados del PDF masivo.
//
// Resumen de la regla:
//
//   TIPO
//     está en `adherentes`            -> adherente
//     no está, pero existe como
//     afiliado en usuarios/nuevoAfiliado -> cotizante
//     ninguna de las dos             -> no verificada
//
//   ESTADO (sólo para adherentes)
//     lo resuelve `nuevoAfiliado.activo`; sólo si esa colección no aporta
//     ningún booleano se cae a `usuarios.activo`. NUNCA sale de
//     `adherentes.activo`: esa colección guarda todos sus registros en true
//     —el propio código del padrón anota "NO guardamos 'estado' en
//     adherentes"— así que usarla no distinguiría a nadie.
//
//     Sin ningún booleano resoluble el adherente queda NO habilitado, que es
//     lo mismo que hace estadoUI() al recibir undefined.
//
//   COTIZANTE
//     siempre habilitado. `activo` no bloquea cotizantes: esa política es
//     sólo para adherentes.
//
// Esto es independiente del estado ACADÉMICO: no toca `aprobo`, ni las
// aprobaciones, ni las exclusiones administrativas. Sólo decide habilitación.

import {
  canonizarDepartamento,
  resolverSegmentoDepartamento,
} from "./segmentos.js";

const COLECCION_ADHERENTES = "adherentes";
const COLECCION_NUEVO_AFILIADO = "nuevoAfiliado";
const COLECCION_USUARIOS = "usuarios";

export type TipoAfiliacion = "adherente" | "cotizante" | "no_verificada";

/**
 * Departamento del afiliado, resuelto y clasificado.
 *
 * `crudo` conserva el valor tal como está guardado, sin canonizar: es lo que
 * permite ver en el Excel un "Fray M.Esquiú" y corregirlo. `canonico` está
 * vacío cuando el valor no se pudo reconocer.
 */
export type Departamento = {
  crudo: string;
  canonico: string;
  segmentoId: string;
};

export type Afiliacion = {
  tipo: TipoAfiliacion;
  activo: boolean | null;
  habilitadoCertificado: boolean;
  etiqueta: string;
  motivoBloqueo: string;
};

export const AFILIACION_NO_VERIFICADA: Afiliacion = {
  tipo: "no_verificada",
  activo: null,
  habilitadoCertificado: false,
  etiqueta: "No verificado",
  motivoBloqueo: "No se pudo verificar la condición de afiliación.",
};

const MOTIVO_ADHERENTE_BLOQUEADO =
  "El adherente no se encuentra habilitado para emitir certificados.";

/** Mismo criterio que normalizeDni() en server.ts: sólo dígitos. */
const normalizar = (dni: unknown) => String(dni ?? "").replace(/\D/g, "");

/**
 * Lista una colección raíz de Firestore y conserva sus campos REST.
 *
 * La resolución del padrón se hace por lote: cada colección se lee una vez
 * y luego se indexa por DNI. Así no se dispara una consulta REST por cada
 * participante (1.000 aprobados siguen siendo tres lecturas paginadas, no
 * 3.000 solicitudes).
 */
async function listarColeccion(
  proyecto: string,
  accessToken: string,
  coleccion: string
): Promise<Record<string, any>[]> {
  const baseUrl = `https://firestore.googleapis.com/v1/projects/${proyecto}/databases/(default)/documents/${encodeURIComponent(
    coleccion
  )}`;
  const documentos: Record<string, any>[] = [];
  let pageToken = "";
  let paginas = 0;
  const inicio = Date.now();

  do {
    const parametros = new URLSearchParams({ pageSize: "1000" });
    parametros.append("mask.fieldPaths", "dni");
    parametros.append("mask.fieldPaths", "activo");
    parametros.append("mask.fieldPaths", "departamento");
    if (pageToken) parametros.set("pageToken", pageToken);

    const respuesta = await fetch(`${baseUrl}?${parametros.toString()}`, {
      headers: { Authorization: `Bearer ${accessToken}` },
    });

    if (!respuesta.ok) {
      throw new Error(`Firestore ${respuesta.status} al listar ${coleccion}.`);
    }

    const datos: any = await respuesta.json();
    paginas += 1;
    for (const documento of Array.isArray(datos?.documents)
      ? datos.documents
      : []) {
      documentos.push(documento?.fields || {});
    }

    pageToken = String(datos?.nextPageToken || "");
  } while (pageToken);

  console.log(
    `[perf] listarColeccion coleccion=${coleccion} paginas=${paginas} documentos=${documentos.length} tiempo=${Date.now() - inicio}ms`
  );

  return documentos;
}

/**
 * Caché corta de las TRES colecciones completas (adherentes, nuevoAfiliado,
 * usuarios).
 *
 * listarColeccion pagina la colección ENTERA sin filtro — miles de documentos,
 * varias decenas de páginas — y antes se repetía sin caché en CADA llamada a
 * resolverPadronPorDni: una vez por curso al listar Registro de Aprobados,
 * una vez por segmento al generar un PDF, una vez en cada emisión masiva. Con
 * treinta cursos abiertos en la misma pantalla eso eran treinta repaginados
 * completos de las mismas tres colecciones.
 *
 * TTL corto (45s): si un administrador cambia el `activo` de un afiliado, el
 * cambio tarda como máximo eso en reflejarse en afiliación y departamento, no
 * el resto de la sesión. Vive en memoria del proceso, por proyecto+colección;
 * cada instancia de Cloud Run tiene la suya y se pierde en cada reinicio.
 */
type ColeccionCacheEntry = {
  documentos: Record<string, any>[];
  expiraEn: number;
};

const COLECCION_CACHE_TTL_MS = 45_000;
const coleccionCache = new Map<string, ColeccionCacheEntry>();
// Evita que varios cache miss simultáneos paginen la misma colección más de
// una vez: todos esperan la misma Promise y sólo el resultado exitoso entra
// en la caché de 45 segundos.
const coleccionEnCurso = new Map<string, Promise<Record<string, any>[]>>();

async function listarColeccionCacheada(
  proyecto: string,
  accessToken: string,
  coleccion: string
): Promise<Record<string, any>[]> {
  const clave = `${proyecto}/${coleccion}`;
  const cacheado = coleccionCache.get(clave);

  if (cacheado && cacheado.expiraEn > Date.now()) {
    console.log(`[perf] listarColeccionCacheada proyecto=${proyecto} coleccion=${coleccion} cache=hit`);
    return cacheado.documentos;
  }

  const cargaExistente = coleccionEnCurso.get(clave);
  if (cargaExistente) {
    console.log(`[perf] listarColeccionCacheada proyecto=${proyecto} coleccion=${coleccion} cache=shared`);
    return cargaExistente;
  }

  console.log(`[perf] listarColeccionCacheada proyecto=${proyecto} coleccion=${coleccion} cache=miss`);
  const carga = listarColeccion(proyecto, accessToken, coleccion)
    .then((documentos) => {
      coleccionCache.set(clave, {
        documentos,
        expiraEn: Date.now() + COLECCION_CACHE_TTL_MS,
      });
      return documentos;
    })
    .finally(() => {
      // También se borra si falla: ningún error queda cacheado.
      coleccionEnCurso.delete(clave);
    });

  coleccionEnCurso.set(clave, carga);
  return carga;
}

const valorFirestore = (campo: any): string => {
  if (typeof campo?.stringValue === "string") return campo.stringValue;
  if (campo?.integerValue !== undefined) return String(campo.integerValue);
  if (campo?.doubleValue !== undefined) return String(campo.doubleValue);
  return "";
};

function indexarPorDni(
  documentos: Record<string, any>[],
  dnisSolicitados: Set<string>
): Map<string, Record<string, any>[]> {
  const indice = new Map<string, Record<string, any>[]>();

  for (const campos of documentos) {
    const dni = normalizar(valorFirestore(campos?.dni));
    if (!dni || !dnisSolicitados.has(dni)) continue;

    const existentes = indice.get(dni) || [];
    existentes.push(campos);
    indice.set(dni, existentes);
  }

  return indice;
}

/**
 * Valores booleanos de `activo` presentes en un conjunto de documentos.
 *
 * Los que no lo traen se descartan: "ausente" no es lo mismo que false.
 */
const booleanosActivo = (docs: Record<string, any>[]): boolean[] =>
  docs
    .map((campos) => campos?.activo?.booleanValue)
    .filter((valor): valor is boolean => typeof valor === "boolean");

/**
 * Resuelve el estado dentro de UNA fuente.
 *
 * El código del padrón toma "el primer booleano", lo que depende del orden en
 * que Firestore devuelva los documentos. Acá se desempata de forma
 * determinista: si algún documento de esa fuente dice false, el resultado es
 * false. Hoy no hay ningún caso en conflicto, así que el resultado coincide
 * con el de la pantalla; la diferencia es que no puede cambiar de una consulta
 * a la otra.
 *
 * Devuelve null cuando la fuente no aporta ningún booleano, para que el
 * llamador sepa que tiene que caer a la siguiente.
 */
const resolverEnFuente = (docs: Record<string, any>[]): boolean | null => {
  const valores = booleanosActivo(docs);
  if (!valores.length) return null;
  return !valores.includes(false);
};

/** Proyección final a partir de lo que se encontró en cada colección. */
export function proyectarAfiliacion(
  enAdherentes: Record<string, any>[],
  enNuevoAfiliado: Record<string, any>[],
  enUsuarios: Record<string, any>[]
): Afiliacion {
  if (enAdherentes.length > 0) {
    // Prioridad de fuentes: nuevoAfiliado manda, y usuarios sólo se mira si
    // aquella no aportó ningún booleano. Las dos NO se mezclan.
    const desdeNuevo = resolverEnFuente(enNuevoAfiliado);
    const activo = desdeNuevo !== null ? desdeNuevo : resolverEnFuente(enUsuarios);

    // Sin dato resoluble queda NO habilitado: es lo que hace estadoUI() del
    // padrón cuando recibe undefined.
    const habilitado = activo === true;

    return {
      tipo: "adherente",
      activo: activo === null ? false : activo,
      habilitadoCertificado: habilitado,
      etiqueta: habilitado ? "Adherente · Habilitado" : "Adherente · No habilitado",
      motivoBloqueo: habilitado ? "" : MOTIVO_ADHERENTE_BLOQUEADO,
    };
  }

  // No figura en el padrón de adherentes. Si existe como afiliado, cotiza: el
  // sistema histórico trata la ausencia del campo `adherente` como "no
  // adherente", así que no se exige que esté marcado en false.
  if (enUsuarios.length > 0 || enNuevoAfiliado.length > 0) {
    return {
      tipo: "cotizante",
      activo: null,
      habilitadoCertificado: true,
      etiqueta: "Cotizante",
      motivoBloqueo: "",
    };
  }

  return { ...AFILIACION_NO_VERIFICADA };
}

/**
 * Resuelve la afiliación de varios DNI.
 *
 * Devuelve un mapa por DNI normalizado. Las tres colecciones se leen por lote
 * y se indexan en memoria; los DNI repetidos no agregan lecturas remotas.
 */
export async function resolverAfiliacionPorDni(
  dnis: Array<string | number | null | undefined>,
  opciones: { proyecto: string; accessToken: string; tamanoLote?: number }
): Promise<Map<string, Afiliacion>> {
  const padron = await resolverPadronPorDni(dnis, opciones);
  return new Map([...padron].map(([dni, d]) => [dni, d.afiliacion]));
}

/**
 * Afiliación Y departamento en una sola pasada.
 *
 * Las tres colecciones ya se consultan para resolver la afiliación, así que
 * el departamento sale del mismo viaje: no agrega ni una consulta. Esto es lo
 * que evita el N+1 al clasificar mil certificados por segmento.
 */
export async function resolverPadronPorDni(
  dnis: Array<string | number | null | undefined>,
  opciones: { proyecto: string; accessToken: string; tamanoLote?: number }
): Promise<Map<string, { afiliacion: Afiliacion; departamento: Departamento }>> {
  const { proyecto, accessToken } = opciones;

  const unicos = [...new Set(dnis.map(normalizar).filter(Boolean))];
  const salida = new Map<
    string,
    { afiliacion: Afiliacion; departamento: Departamento }
  >();

  if (!unicos.length) return salida;

  const inicio = Date.now();
  const clavesCache = [COLECCION_ADHERENTES, COLECCION_NUEVO_AFILIADO, COLECCION_USUARIOS]
    .map((coleccion) => `${proyecto}/${coleccion}`);
  const cacheHit = clavesCache.every(
    (clave) => (coleccionCache.get(clave)?.expiraEn || 0) > Date.now()
  );

  const solicitados = new Set(unicos);
  const [documentosAdherentes, documentosNuevoAfiliado, documentosUsuarios] =
    await Promise.all([
      listarColeccionCacheada(proyecto, accessToken, COLECCION_ADHERENTES),
      listarColeccionCacheada(proyecto, accessToken, COLECCION_NUEVO_AFILIADO),
      listarColeccionCacheada(proyecto, accessToken, COLECCION_USUARIOS),
    ]);

  console.log(`[perf] resolverPadron dnis=${unicos.length} cache=${cacheHit ? "hit" : "miss"} tiempo=${Date.now() - inicio}ms`);

  const adherentesPorDni = indexarPorDni(documentosAdherentes, solicitados);
  const nuevoAfiliadoPorDni = indexarPorDni(
    documentosNuevoAfiliado,
    solicitados
  );
  const usuariosPorDni = indexarPorDni(documentosUsuarios, solicitados);

  for (const dni of unicos) {
    const adherentes = adherentesPorDni.get(dni) || [];
    const nuevoAfiliado = nuevoAfiliadoPorDni.get(dni) || [];
    const usuarios = usuariosPorDni.get(dni) || [];

    salida.set(dni, {
      afiliacion: proyectarAfiliacion(adherentes, nuevoAfiliado, usuarios),
      departamento: proyectarDepartamento(
        nuevoAfiliado,
        usuarios,
        adherentes
      ),
    });
  }

  return salida;
}

/** Primer `departamento` no vacío de un conjunto de documentos. */
const departamentoDe = (docs: Record<string, any>[]): string => {
  for (const campos of docs) {
    const valor = String(campos?.departamento?.stringValue ?? "").trim();
    if (valor) return valor;
  }
  return "";
};

/**
 * Departamento vigente del afiliado.
 *
 * Prioridad `nuevoAfiliado` → `usuarios`: es la que ya aplica la pantalla de
 * Afiliado Actualizado al fusionar los dos registros —`pick(nr.departamento,
 * ur.departamento)`—, así que se respeta tal cual. En el padrón real sólo 7
 * DNI de 6655 tienen valores distintos entre ambas, pero el criterio queda
 * definido igual.
 *
 * `adherentes` se agrega como TERCER respaldo, que la pantalla original no
 * contempla porque sólo fusiona dos colecciones. Recupera 13 personas cuyo
 * departamento no está en ninguna de las otras dos y que, si no, irían al
 * grupo de revisión sin motivo. Es una extensión, no un cambio de prioridad:
 * sólo actúa cuando las dos fuentes principales están vacías.
 */
export function proyectarDepartamento(
  enNuevoAfiliado: Record<string, any>[],
  enUsuarios: Record<string, any>[],
  enAdherentes: Record<string, any>[]
): Departamento {
  const crudo =
    departamentoDe(enNuevoAfiliado) ||
    departamentoDe(enUsuarios) ||
    departamentoDe(enAdherentes);

  return {
    crudo,
    canonico: canonizarDepartamento(crudo),
    segmentoId: resolverSegmentoDepartamento(crudo),
  };
}

/** Atajo para un solo DNI. Reutiliza el resolver por lote. */
export async function resolverAfiliacionDeUnDni(
  dni: string | number | null | undefined,
  opciones: { proyecto: string; accessToken: string }
): Promise<Afiliacion> {
  const clave = normalizar(dni);
  if (!clave) return { ...AFILIACION_NO_VERIFICADA };

  const mapa = await resolverAfiliacionPorDni([clave], opciones);
  return mapa.get(clave) || { ...AFILIACION_NO_VERIFICADA };
}
