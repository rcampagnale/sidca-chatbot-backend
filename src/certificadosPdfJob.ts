// src/certificadosPdfJob.ts
//
// Cloud Run Job: genera UN PDF con todos los certificados vigentes de UN
// curso y lo deja en Cloud Storage.
//
// Es un proceso batch: no levanta Express, no escucha PORT, corre y termina.
// La misma imagen sirve para el servicio (npm start) y para este Job
// (node dist/src/certificadosPdfJob.js); lo único que cambia es el comando.
//
// Debe ejecutarse con tasks=1 y parallelism=1: todas las páginas van al mismo
// archivo y el orden —alfabético por apellido y nombre— tiene que ser
// determinístico. Repartir páginas entre tareas produciría archivos pisados.
//
// El uso de memoria no depende de la cantidad de certificados: hay UNA
// instancia de PDFDocument conectada por pipe al stream de Storage, y cada
// página se escribe y se suelta. Mil páginas ocupan lo mismo que una.

import "dotenv/config";
import PDFDocument from "pdfkit";
import { Storage } from "@google-cloud/storage";
import { renderCertificadoPdfPage } from "./certificados/certificadoPdfRenderer.js";
import {
  listarEmisionesVigentesCurso,
  filtrarEmisionesHabilitadas,
} from "./certificados/emisionesCurso.js";
import { esSegmentoValido, obtenerSegmento } from "./certificados/segmentos.js";

const proyecto =
  process.env.FIREBASE_PROJECT_ID ||
  process.env.GCLOUD_PROJECT ||
  process.env.GOOGLE_CLOUD_PROJECT ||
  "";

const firestoreBase = `https://firestore.googleapis.com/v1/projects/${proyecto}/databases/(default)/documents`;

/**
 * Access token de la cuenta de servicio.
 *
 * En Cloud Run sale del Metadata Server; en local, de
 * GOOGLE_OAUTH_ACCESS_TOKEN, igual que el resto del backend. Nunca se invoca
 * gcloud desde Node.
 *
 * Se cachea: un PDF de mil páginas hace cientos de escrituras de progreso, y
 * pedir un token nuevo en cada una sería absurdo.
 */
let tokenCache: { valor: string; expiraEn: number } | null = null;

async function obtenerAccessToken(): Promise<string> {
  const explicito = process.env.GOOGLE_OAUTH_ACCESS_TOKEN?.trim();
  if (explicito) return explicito;

  if (tokenCache && tokenCache.expiraEn > Date.now() + 60_000) {
    return tokenCache.valor;
  }

  const respuesta = await fetch(
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
    { headers: { "Metadata-Flavor": "Google" } }
  );

  if (!respuesta.ok) {
    throw new Error(`No se pudo obtener el access token (${respuesta.status}).`);
  }

  const datos: any = await respuesta.json();

  tokenCache = {
    valor: String(datos.access_token),
    expiraEn: Date.now() + Number(datos.expires_in || 3600) * 1000,
  };

  return tokenCache.valor;
}

/** Serializa un valor JS al formato de Firestore REST. */
function aValorFirestore(valor: any): any {
  if (valor === null || valor === undefined) return { nullValue: null };
  if (valor instanceof Date) return { timestampValue: valor.toISOString() };
  if (typeof valor === "boolean") return { booleanValue: valor };
  if (typeof valor === "number") {
    return Number.isInteger(valor)
      ? { integerValue: String(valor) }
      : { doubleValue: valor };
  }
  return { stringValue: String(valor) };
}

/**
 * Actualiza SÓLO los campos indicados del documento del trabajo.
 *
 * El updateMask no es opcional: un PATCH sin máscara reemplaza el documento
 * entero, así que cada escritura de progreso borraría jobId, cursoId,
 * creadoPor y todo lo que el backend dejó al crearlo.
 */
async function actualizarTrabajo(
  rutaTrabajo: string,
  datos: Record<string, any>
): Promise<void> {
  const campos = Object.keys(datos);
  if (!campos.length) return;

  const mascara = campos
    .map((campo) => `updateMask.fieldPaths=${encodeURIComponent(campo)}`)
    .join("&");

  const respuesta = await fetch(`${rutaTrabajo}?${mascara}`, {
    method: "PATCH",
    headers: {
      Authorization: `Bearer ${await obtenerAccessToken()}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      fields: Object.fromEntries(
        Object.entries(datos).map(([clave, valor]) => [
          clave,
          aValorFirestore(valor),
        ])
      ),
    }),
  });

  if (!respuesta.ok) {
    throw new Error(`No se pudo actualizar el trabajo (${respuesta.status}).`);
  }
}

/** Estados del trabajo. Los comparte con el backend y con la pantalla. */
const ESTADO_PROCESANDO = "procesando";
const ESTADO_COMPLETADO = "completado";
const ESTADO_ERROR = "error";

/** Progreso: como mucho una escritura por segundo, o cada 5 certificados. */
const PROGRESO_CADA_CERTIFICADOS = 5;
const PROGRESO_CADA_MS = 1000;

/** Con menos muestras que esto, cualquier estimación de tiempo es ruido. */
const MUESTRAS_MINIMAS_ETA = 2;

const main = async () => {
  // 1. Variables de entorno. Se acepta el nombre anterior del identificador
  //    para no romper una ejecución que ya esté en vuelo.
  const cursoId = String(process.env.CERTIFICADOS_PDF_CURSO_ID || "").trim();
  const jobId = String(
    process.env.CERTIFICADOS_PDF_TRABAJO_ID ||
      process.env.CERTIFICADOS_PDF_JOB_ID ||
      ""
  ).trim();
  const bucketNombre = String(process.env.CERTIFICADOS_PDF_BUCKET || "").trim();

  // Segmento geográfico. Es OPCIONAL: sin él el Job hace exactamente lo de
  // siempre —un PDF con todo el curso— y ninguna ejecución anterior cambia de
  // comportamiento.
  const segmentoId = String(
    process.env.CERTIFICADOS_PDF_SEGMENTO_ID || ""
  ).trim();

  if (!cursoId || !jobId || !proyecto || !bucketNombre) {
    throw new Error(
      "Faltan variables del Job: se requieren CERTIFICADOS_PDF_CURSO_ID, CERTIFICADOS_PDF_TRABAJO_ID, CERTIFICADOS_PDF_BUCKET y el proyecto."
    );
  }

  // Se valida contra la lista de segmentos y no sólo contra caracteres: el
  // valor se interpola en la ruta del objeto de Storage, y un segmento
  // inventado generaría un archivo que después nadie podría encontrar.
  if (segmentoId && !esSegmentoValido(segmentoId)) {
    throw new Error(`CERTIFICADOS_PDF_SEGMENTO_ID inválido: ${segmentoId}`);
  }

  const segmento = segmentoId ? obtenerSegmento(segmentoId) : null;

  const rutaTrabajo = `${firestoreBase}/certificados/${encodeURIComponent(
    cursoId
  )}/trabajosPdf/${encodeURIComponent(jobId)}`;

  const inicio = Date.now();

  try {
    // 2. En marcha.
    await actualizarTrabajo(rutaTrabajo, {
      estado: ESTADO_PROCESANDO,
      procesados: 0,
      porcentaje: 0,
      iniciadoEn: new Date(),
      actualizadoEn: new Date(),
      error: null,
    });

    // 3. Emisiones del curso. listarEmisionesVigentesCurso lee SÓLO
    //    certificados/{cursoId}/emitidos —nada de collectionGroup— y aborta si
    //    encuentra una emisión con otro cursoId.
    const vigentes = await listarEmisionesVigentesCurso(
      cursoId,
      await obtenerAccessToken()
    );

    // 3 bis. Condición sindical. Un certificado ya emitido no se borra ni se
    //        anula porque el participante deje de estar habilitado, pero
    //        tampoco puede salir en el PDF masivo: si no, la descarga masiva
    //        sería una puerta trasera para obtener lo que la pantalla bloquea.
    //
    // 3 ter. Segmento geográfico, cuando lo hay. Se aplica DESPUÉS de la
    //        afiliación, sobre el departamento vigente del afiliado. El mismo
    //        llamado resuelve las dos cosas: no hay una segunda pasada por el
    //        padrón.
    const {
      habilitadas: emisiones,
      omitidosAfiliacion,
      omitidosSegmento,
      sinDepartamento,
    } = await filtrarEmisionesHabilitadas(
      vigentes,
      await obtenerAccessToken(),
      segmentoId || undefined
    );

    // Un solo log con los contadores del filtrado. Sin tokens, sin documentos.
    console.log(
      `[pdf-segmentado] curso=${cursoId} segmento=${segmentoId || "-"} vigentes=${
        vigentes.length
      } habilitados=${vigentes.length - omitidosAfiliacion} omitidosAfiliacion=${omitidosAfiliacion} omitidosSegmento=${omitidosSegmento} sinDepartamento=${sinDepartamento} paginas=${emisiones.length}`
    );

    // 4. Orden determinístico.
    emisiones.sort((a: any, b: any) =>
      String(a.participante?.apellidoNombre || "").localeCompare(
        String(b.participante?.apellidoNombre || ""),
        "es",
        { sensitivity: "base" }
      )
    );

    const total = emisiones.length;

    if (!total) {
      throw new Error(
        segmento
          ? `No hay certificados descargables en el segmento ${segmento.nombre}.`
          : omitidosAfiliacion > 0
          ? "Todos los certificados vigentes corresponden a participantes sin afiliación habilitada."
          : "No hay certificados vigentes emitidos para este curso."
      );
    }

    await actualizarTrabajo(rutaTrabajo, {
      total,
      omitidosAfiliacion,
      actualizadoEn: new Date(),
    });

    // 5. Streaming. El PDF nunca existe entero en memoria: se escribe en
    //    Storage a medida que se generan las páginas.
    // La ruta lleva el segmento cuando lo hay. Tiene que coincidir exactamente
    // con la que arma el backend en pdfSegmentoObjectName(): el endpoint de
    // descarga rechaza cualquier objeto que no sea el determinístico.
    const objectName = segmentoId
      ? `certificados-pdf/${cursoId}/${segmentoId}/${jobId}.pdf`
      : `certificados-pdf/${cursoId}/${jobId}.pdf`;
    const storage = new Storage();
    const archivo = storage.bucket(bucketNombre).file(objectName);

    const salida = archivo.createWriteStream({
      contentType: "application/pdf",
      metadata: { cacheControl: "private, max-age=0, no-transform" },
    });

    const documento = new PDFDocument({
      autoFirstPage: false,
      size: "A4",
      layout: "landscape",
      margin: 0,
    });

    // El fallo del stream se captura acá y se relanza al esperar el cierre:
    // sin esto un error de Storage sería un unhandled rejection.
    const cierre = new Promise<void>((resolve, reject) => {
      salida.once("finish", resolve);
      salida.once("error", reject);
    });

    documento.pipe(salida);

    let ultimaEscritura = 0;

    for (let indice = 0; indice < total; indice += 1) {
      await renderCertificadoPdfPage(documento, emisiones[indice]);

      const procesados = indice + 1;
      const ahora = Date.now();
      const ultimo = procesados === total;

      const toca =
        ultimo ||
        procesados % PROGRESO_CADA_CERTIFICADOS === 0 ||
        ahora - ultimaEscritura >= PROGRESO_CADA_MS;

      if (!toca) continue;

      ultimaEscritura = ahora;

      const transcurridoMs = ahora - inicio;

      // Velocidad media real. Con una sola muestra la estimación oscila
      // demasiado como para mostrarla, así que hasta la segunda no se informa
      // ningún tiempo restante.
      const hayMuestra = procesados >= MUESTRAS_MINIMAS_ETA && transcurridoMs > 0;
      const porSegundo = hayMuestra ? procesados / (transcurridoMs / 1000) : 0;
      const restanteEstimadoMs =
        porSegundo > 0
          ? Math.max(0, Math.round(((total - procesados) / porSegundo) * 1000))
          : null;

      await actualizarTrabajo(rutaTrabajo, {
        procesados,
        total,
        porcentaje: total > 0 ? Math.floor((procesados * 100) / total) : 0,
        transcurridoMs,
        restanteEstimadoMs,
        finalizacionEstimada:
          restanteEstimadoMs === null
            ? null
            : new Date(ahora + restanteEstimadoMs),
        actualizadoEn: new Date(),
      });
    }

    // 6. Cierre. doc.end() no significa "subido": hay que esperar el finish
    //    real del stream de Storage antes de dar el archivo por disponible.
    documento.end();
    await cierre;

    const [metadatos] = await archivo.getMetadata();
    const tamanioBytes = Number(metadatos.size || 0);

    await actualizarTrabajo(rutaTrabajo, {
      estado: ESTADO_COMPLETADO,
      procesados: total,
      total,
      porcentaje: 100,
      objectName,
      // Se conserva el nombre histórico del campo: los trabajos anteriores lo
      // usan y el endpoint de descarga lo lee.
      storagePath: objectName,
      tamanioBytes,
      transcurridoMs: Date.now() - inicio,
      restanteEstimadoMs: 0,
      finalizacionEstimada: null,
      finalizadoEn: new Date(),
      actualizadoEn: new Date(),
      error: null,
    });

    console.log(
      `[certificados-pdf-job] completado curso=${cursoId} segmento=${
        segmentoId || "-"
      } job=${jobId} paginas=${total} bytes=${tamanioBytes}`
    );
  } catch (error: any) {
    // Sin esto el trabajo quedaba para siempre en "procesando" y la pantalla
    // encuestaba sin fin. El mensaje se acota: va a una interfaz.
    const mensaje = String(
      error?.message || "No se pudo generar el PDF masivo."
    ).slice(0, 300);

    console.error("[certificados-pdf-job] error", error);

    await actualizarTrabajo(rutaTrabajo, {
      estado: ESTADO_ERROR,
      error: mensaje,
      finalizadoEn: new Date(),
      actualizadoEn: new Date(),
    }).catch((fallo) => {
      console.error("[certificados-pdf-job] no se pudo registrar el error", fallo);
    });

    throw error;
  }
};

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
