// src/certificados/emisionesCurso.ts
//
// Lee las emisiones vigentes de UN curso y les resuelve la configuración
// efectiva con la que hay que imprimirlas.
//
// La resolución vive acá y no en el renderer a propósito: renderCertificadoPdfPage
// dibuja lo que recibe y no consulta Firestore. Acá se garantiza que siempre
// reciba una emisión completa.

import { resolverPadronPorDni } from "./afiliacion.js";
import { SEGMENTO_SIN_DEPARTAMENTO } from "./segmentos.js";

type FirestoreValue = any;

const projectId = () =>
  process.env.FIREBASE_PROJECT_ID ||
  process.env.GOOGLE_CLOUD_PROJECT ||
  process.env.GCLOUD_PROJECT ||
  "";

const decode = (v: FirestoreValue): any =>
  v?.stringValue ??
  v?.integerValue ??
  v?.doubleValue ??
  v?.booleanValue ??
  v?.timestampValue ??
  (v?.mapValue
    ? Object.fromEntries(
        Object.entries(v.mapValue.fields || {}).map(([k, x]) => [k, decode(x)])
      )
    : v?.arrayValue
    ? (v.arrayValue.values || []).map(decode)
    : null);

const decodeDoc = (doc: any) =>
  Object.fromEntries(
    Object.entries(doc?.fields || {}).map(([k, v]) => [k, decode(v)])
  );

const texto = (valor: any) => String(valor ?? "").trim();

/**
 * ¿El snapshot trae autoridades utilizables?
 *
 * Un arreglo vacío no cuenta, y uno con entradas sin nombre ni cargo tampoco:
 * imprimirlas dejaría el bloque de firma en blanco igual.
 */
function tieneAutoridadesUtiles(autoridades: any): boolean {
  return (
    Array.isArray(autoridades) &&
    autoridades.some(
      (autoridad: any) =>
        texto(autoridad?.nombre) !== "" || texto(autoridad?.cargo) !== ""
    )
  );
}

/** Deja una autoridad con los cuatro textos y el orden, sin campos extra. */
const normalizarAutoridad = (autoridad: any, indice: number) => ({
  nombre: texto(autoridad?.nombre),
  cargo: texto(autoridad?.cargo),
  organismo: texto(autoridad?.organismo),
  referencia: texto(autoridad?.referencia),
  orden: Number(autoridad?.orden) || indice + 1,
});

/**
 * Autoridades de la CONFIGURACIÓN del curso, con compatibilidad legacy.
 *
 * Prioridad: `autoridades` (modelo actual) y, si no hay, `firmas` (modelo
 * anterior con imágenes), de las que sólo se rescatan nombre y cargo. Las
 * firmas nunca tuvieron organismo ni referencia: quedan vacíos y esos
 * renglones no se dibujan.
 */
function autoridadesDeConfiguracion(configuracion: any): any[] {
  if (tieneAutoridadesUtiles(configuracion?.autoridades)) {
    return configuracion.autoridades.slice(0, 2).map(normalizarAutoridad);
  }

  if (tieneAutoridadesUtiles(configuracion?.firmas)) {
    return configuracion.firmas
      .slice(0, 2)
      .map((firma: any, indice: number) => ({
        nombre: texto(firma?.nombre),
        cargo: texto(firma?.cargo),
        organismo: "",
        referencia: "",
        orden: indice + 1,
      }));
  }

  return [];
}

/**
 * Lee la configuración del curso. UNA sola vez por generación.
 *
 * Devuelve null si no existe: en ese caso no hay nada con qué completar y las
 * emisiones se imprimen tal como fueron guardadas.
 */
async function obtenerConfiguracionCurso(cursoId: string, accessToken: string) {
  const url = `https://firestore.googleapis.com/v1/projects/${projectId()}/databases/(default)/documents/certificados/${encodeURIComponent(
    cursoId
  )}`;

  const response = await fetch(url, {
    headers: { Authorization: `Bearer ${accessToken}` },
  });

  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`Firestore ${response.status}`);

  return decodeDoc(await response.json());
}

export async function listarEmisionesVigentesCurso(
  cursoId: string,
  accessToken: string
) {
  const id = texto(cursoId);
  if (!id) throw new Error("cursoId es obligatorio.");

  const base = `https://firestore.googleapis.com/v1/projects/${projectId()}/databases/(default)/documents/certificados/${encodeURIComponent(
    id
  )}/emitidos`;

  const documentos: any[] = [];
  let pageToken = "";

  do {
    const url = new URL(base);
    url.searchParams.set("pageSize", "300");
    if (pageToken) url.searchParams.set("pageToken", pageToken);

    const response = await fetch(url, {
      headers: { Authorization: `Bearer ${accessToken}` },
    });

    if (!response.ok) throw new Error(`Firestore ${response.status}`);

    const data = await response.json();
    documentos.push(...(data.documents || []));
    pageToken = data.nextPageToken || "";
  } while (pageToken);

  const vigentes = documentos
    .map(decodeDoc)
    .filter((emision) => texto(emision.estado) === "vigente");

  // Aislamiento por curso: una emisión ajena mezclaría certificados de otra
  // capacitación en el mismo archivo.
  if (
    vigentes.some((emision) => {
      const otro = texto(emision.cursoId);
      return otro && otro !== id;
    })
  ) {
    throw new Error(
      "Se detectó una emisión perteneciente a otro curso. Se cancela la generación para evitar un PDF mezclado."
    );
  }

  // Las emisiones anteriores al modelo de institución y autoridades tienen un
  // snapshot sin esos campos: se guardaron cuando el certificado sólo imprimía
  // los datos documentales. Sin completarlas, el PDF sale sin bloque de firma.
  //
  // La configuración del curso se lee UNA vez y se reutiliza para todas: con
  // mil certificados esto son mil renders y UNA lectura extra, no mil.
  //
  // Sólo se completa lo que falta. Un snapshot que YA trae autoridades manda
  // siempre: es el estado congelado en el momento de emitir y no debe quedar
  // pisado por una configuración cambiada después. Nada de esto se escribe en
  // Firestore; el documento histórico queda intacto.
  const necesitanCompletar = vigentes.some(
    (emision) =>
      !tieneAutoridadesUtiles(emision?.certificado?.autoridades) ||
      !texto(emision?.certificado?.institucionCertificado)
  );

  if (!necesitanCompletar) return vigentes;

  const configuracion = await obtenerConfiguracionCurso(id, accessToken);
  if (!configuracion) return vigentes;

  const autoridadesCurso = autoridadesDeConfiguracion(configuracion);
  const institucionCurso =
    texto(configuracion.institucionCertificado) === "itm" ? "itm" : "sidca";

  return proyectarEmisiones(vigentes, autoridadesCurso, institucionCurso);
}

/**
 * Emisiones que además pueden IMPRIMIRSE hoy, opcionalmente de un solo
 * segmento geográfico.
 *
 * Un certificado emitido no se borra ni se anula porque el participante deje
 * de estar habilitado, pero tampoco puede salir en el PDF masivo mientras esa
 * condición no se cumpla: si no, la descarga masiva sería una puerta trasera
 * para obtener lo que la pantalla bloquea.
 *
 * Con `segmentoId` se queda además sólo con quienes pertenecen a ese segmento.
 * El departamento sale de resolverPadronPorDni, que lo trae en el MISMO viaje
 * que la afiliación: filtrar por segmento no agrega ni una consulta. Es el
 * departamento VIGENTE del afiliado y no el que quedó congelado en el
 * certificado, porque la descarga se organiza por dónde está hoy la gente.
 *
 * Sin `segmentoId` se comporta exactamente como antes.
 *
 * El orden importa y es el de la especificación: primero se descarta a quien
 * no está habilitado y recién después se clasifica por segmento. Así
 * `omitidosAfiliacion` sigue significando lo mismo que hasta ahora.
 *
 * Devuelve también cuántos quedaron fuera, para poder informarlo en el
 * trabajo sin cambiar el formato de lo que ya existía.
 */
export async function filtrarEmisionesHabilitadas(
  emisiones: any[],
  accessToken: string,
  segmentoId?: string
): Promise<{
  habilitadas: any[];
  omitidosAfiliacion: number;
  omitidosSegmento: number;
  sinDepartamento: number;
}> {
  if (!emisiones.length) {
    return {
      habilitadas: [],
      omitidosAfiliacion: 0,
      omitidosSegmento: 0,
      sinDepartamento: 0,
    };
  }

  const padron = await resolverPadronPorDni(
    emisiones.map((emision) => emision?.participante?.dni),
    { proyecto: projectId(), accessToken }
  );

  const clave = (emision: any) =>
    String(emision?.participante?.dni ?? "").replace(/\D/g, "");

  const segmentoDe = (emision: any) =>
    padron.get(clave(emision))?.departamento?.segmentoId ||
    SEGMENTO_SIN_DEPARTAMENTO;

  const habilitadasPorAfiliacion = emisiones.filter(
    (emision) =>
      padron.get(clave(emision))?.afiliacion?.habilitadoCertificado === true
  );

  const sinDepartamento = habilitadasPorAfiliacion.filter(
    (emision) => segmentoDe(emision) === SEGMENTO_SIN_DEPARTAMENTO
  ).length;

  const habilitadas = segmentoId
    ? habilitadasPorAfiliacion.filter(
        (emision) => segmentoDe(emision) === segmentoId
      )
    : habilitadasPorAfiliacion;

  return {
    habilitadas,
    omitidosAfiliacion: emisiones.length - habilitadasPorAfiliacion.length,
    omitidosSegmento: habilitadasPorAfiliacion.length - habilitadas.length,
    sinDepartamento,
  };
}

function proyectarEmisiones(vigentes: any[], autoridadesCurso: any[], institucionCurso: string) {
  return vigentes.map((emision) => {
    const certificado = emision.certificado || {};

    const autoridades = tieneAutoridadesUtiles(certificado.autoridades)
      ? certificado.autoridades.slice(0, 2).map(normalizarAutoridad)
      : autoridadesCurso;

    const institucionCertificado =
      texto(certificado.institucionCertificado) || institucionCurso;

    return {
      ...emision,
      certificado: { ...certificado, autoridades, institucionCertificado },
    };
  });
}
