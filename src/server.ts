import "dotenv/config";
import crypto from "node:crypto";
import express from "express";
import cors from "cors";
import multer from "multer";
import OpenAI, { toFile } from "openai";
import { z } from "zod";
import { createRemoteJWKSet, jwtVerify } from "jose";
import { runChatbotWorkflow } from "./openaiWorkflow.js";
// El SDK de Storage ya no se usa acá: la descarga va por la API JSON con el
// mismo helper de credenciales que el resto del backend. El Cloud Run Job sí
// lo sigue usando, en su propio proceso.
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import {
  AFILIACION_NO_VERIFICADA,
  resolverAfiliacionDeUnDni,
  resolverPadronPorDni,
  type Afiliacion,
  type Departamento,
} from "./certificados/afiliacion.js";
import {
  SEGMENTOS,
  SEGMENTO_SIN_DEPARTAMENTO,
  esSegmentoValido,
  obtenerSegmento,
} from "./certificados/segmentos.js";

const app = express();
const PORT = Number(process.env.PORT || 8080);
const REQUEST_TIMEOUT_MS = Number(process.env.REQUEST_TIMEOUT_MS || 15000);
const WEBHOOK_MAX_AGE_MS = Number(process.env.MP_WEBHOOK_MAX_AGE_MS || 5 * 60 * 1000);

type MercadoPagoEnvironment = "test" | "production";
type PagoTipo = "cuota_adherente" | "orden_administrativa";

type RateLimitOptions = {
  windowMs: number;
  max: number;
  message: string;
};

type RateLimitEntry = {
  count: number;
  resetAt: number;
};

const rateLimitStore = new Map<string, RateLimitEntry>();


const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 25 * 1024 * 1024,
  },
});

const registroInscriptosUpload = multer({
  storage: multer.memoryStorage(),
  limits: { files: 10, fileSize: 15 * 1024 * 1024 },
});
const registroExcelMimeTypes = new Set([
  "application/vnd.ms-excel",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  "application/octet-stream",
]);

function nombreArchivoRegistroSeguro(nombre: string): string {
  const base = String(nombre || "archivo.xlsx").replace(/[\\/\0-\x1f]/g, "_").replace(/[^a-zA-Z0-9._-]+/g, "_");
  return base.slice(0, 180) || "archivo.xlsx";
}

function validarArchivoRegistro(file: Express.Multer.File): void {
  const extension = String(file.originalname || "").toLowerCase().split(".").pop();
  if (!["xls", "xlsx"].includes(extension || "")) {
    throw Object.assign(new Error("Sólo se permiten archivos Excel .xls o .xlsx."), { statusCode: 400 });
  }
  if (!registroExcelMimeTypes.has(String(file.mimetype || "").toLowerCase())) {
    throw Object.assign(new Error("El archivo no tiene un tipo Excel válido."), { statusCode: 400 });
  }
  if (!file.buffer?.length || file.size > 15 * 1024 * 1024) {
    throw Object.assign(new Error("Cada planilla debe pesar como máximo 15 MB."), { statusCode: 400 });
  }
}

function normalizarBucket(valor: unknown): string {
  return String(valor || "").trim().replace(/^gs:\/\//i, "").replace(/\/+$/, "");
}

function bucketRegistroInscriptos() {
  const bucketNombre = normalizarBucket(
    process.env.FIREBASE_STORAGE_BUCKET ||
      process.env.STORAGE_BUCKET ||
      process.env.GCS_BUCKET ||
      process.env.CERTIFICADOS_PDF_BUCKET
  );
  if (!bucketNombre) throw Object.assign(new Error("La descarga de planillas no está configurada."), { statusCode: 500 });
  return bucketNombre;
}

async function storageRegistroRequest(
  method: "GET" | "POST" | "DELETE",
  bucket: string,
  objectName: string,
  options: { mimeType?: string; body?: Buffer; media?: boolean } = {}
): Promise<Response> {
  const url = method === "POST"
    ? `https://storage.googleapis.com/upload/storage/v1/b/${encodeURIComponent(bucket)}/o?uploadType=media&name=${encodeURIComponent(objectName)}`
    : `https://storage.googleapis.com/storage/v1/b/${encodeURIComponent(bucket)}/o/${encodeURIComponent(objectName)}${method === "GET" ? "?alt=media" : ""}`;
  return fetch(url, {
    method,
    headers: {
      Authorization: `Bearer ${await getGoogleAccessToken()}`,
      ...(options.mimeType ? { "Content-Type": options.mimeType } : {}),
      ...(options.body ? { "Content-Length": String(options.body.length) } : {}),
    },
    ...(options.body ? { body: options.body as any } : {}),
  });
}

async function storageRegistroError(response: Response): Promise<never> {
  const detail = (await response.text().catch(() => "")).replace(/\s+/g, " ").slice(0, 240);
  throw Object.assign(new Error(`Google Storage ${response.status}${detail ? `: ${detail}` : ""}`), { storageHttpStatus: response.status, statusCode: response.status >= 500 ? 502 : response.status });
}

async function uploadGoogleStorageObject(bucket: string, objectName: string, body: Buffer, mimeType: string): Promise<void> {
  const response = await storageRegistroRequest("POST", bucket, objectName, { body, mimeType, media: true });
  if (!response.ok) await storageRegistroError(response);
}

async function deleteGoogleStorageObject(bucket: string, objectName: string): Promise<void> {
  const response = await storageRegistroRequest("DELETE", bucket, objectName);
  if (!response.ok && response.status !== 404) await storageRegistroError(response);
}

const bodySchema = z.object({
  pregunta: z.string().trim().min(2, "La pregunta es obligatoria"),
  dominio: z
    .enum(["licencias", "estatuto", "general", "coberturas"])
    .optional(),
  maxResults: z.number().int().min(1).max(8).optional(),
});

const secureMercadoPagoPreferenceSchema = z.object({
  dni: z.string().trim().min(5, "El DNI es obligatorio"),
  pagoId: z
    .string()
    .trim()
    .regex(
      /^[A-Za-z0-9_-]{1,128}$/,
      "El identificador de la orden es inválido."
    )
    .optional(),
  forzarNuevaPreferencia: z.boolean().optional().default(false),
});

const firebaseBootstrapSchema = z.object({
  dni: z.string().trim().min(5, "El DNI es obligatorio"),
  usuarioId: z
    .string()
    .trim()
    .regex(
      /^[A-Za-z0-9:_-]{1,128}$/,
      "El identificador del usuario es inválido."
    ),
});

type FirestoreDocument = {
  name: string;
  fields?: Record<string, FirestoreValue>;
  createTime?: string;
  updateTime?: string;
};

type FirestoreValue = {
  stringValue?: string;
  integerValue?: string;
  doubleValue?: number;
  booleanValue?: boolean;
  timestampValue?: string;
  nullValue?: null;
  mapValue?: { fields?: Record<string, FirestoreValue> };
  arrayValue?: { values?: FirestoreValue[] };
};

type FirestoreRecord = Record<string, any> & {
  id?: string;
  path?: string;
  _name?: string;
};

type AuthenticatedUser = {
  uid: string;
  email?: string;
};

type FirebaseServiceAccount = {
  client_email?: string;
  private_key?: string;
  project_id?: string;
};

type CuotaAdherenteConfig = {
  habilitada: boolean;
  periodo: number;
  importe: number;
  moneda: string;
  concepto: string;
  detalle: string;
  cuotasMaximas: number;
};

type MercadoPagoPayment = {
  id: number | string;
  status?: string;
  status_detail?: string;
  transaction_amount?: number;
  currency_id?: string;
  external_reference?: string;
  payment_method_id?: string;
  payment_type_id?: string;
  date_created?: string;
  date_approved?: string;
  live_mode?: boolean;
};

const firebaseProjectId =
  process.env.FIREBASE_PROJECT_ID?.trim() ||
  process.env.GOOGLE_CLOUD_PROJECT?.trim() ||
  process.env.GCLOUD_PROJECT?.trim() ||
  "sidca-a33f0";

const firebaseIssuer = `https://securetoken.google.com/${firebaseProjectId}`;
const firebaseJwks = createRemoteJWKSet(
  new URL(
    "https://www.googleapis.com/service_accounts/v1/jwk/securetoken@system.gserviceaccount.com"
  )
);

const firestoreBaseUrl = `https://firestore.googleapis.com/v1/projects/${firebaseProjectId}/databases/(default)/documents`;
function base64UrlEncode(input: Buffer | string): string {
  return Buffer.from(input).toString("base64url");
}

function getFirebaseServiceAccount(): FirebaseServiceAccount {
  const encoded = process.env.FIREBASE_SERVICE_ACCOUNT_BASE64?.trim();
  const rawJson = process.env.FIREBASE_SERVICE_ACCOUNT_JSON?.trim();

  if (!encoded && !rawJson) {
    throw Object.assign(
      new Error(
        "Falta configurar FIREBASE_SERVICE_ACCOUNT_JSON o FIREBASE_SERVICE_ACCOUNT_BASE64 para emitir tokens Firebase."
      ),
      { statusCode: 500 }
    );
  }

  try {
    const raw = encoded ? Buffer.from(encoded, "base64").toString("utf8") : rawJson!;
    const account = JSON.parse(raw) as FirebaseServiceAccount;

    if (!account.client_email || !account.private_key) {
      throw new Error("El service account no contiene client_email o private_key.");
    }

    account.private_key = account.private_key.replace(/\\n/g, "\n");
    return account;
  } catch (error: any) {
    throw Object.assign(
      new Error(`Credenciales Firebase Admin inválidas: ${error?.message || "no se pudo leer el service account"}`),
      { statusCode: 500 }
    );
  }
}

function createFirebaseCustomToken(uid: string, claims: Record<string, unknown> = {}): string {
  const account = getFirebaseServiceAccount();
  const now = Math.floor(Date.now() / 1000);
  const header = { alg: "RS256", typ: "JWT" };
  const payload = {
    iss: account.client_email,
    sub: account.client_email,
    aud: "https://identitytoolkit.googleapis.com/google.identity.identitytoolkit.v1.IdentityToolkit",
    iat: now,
    exp: now + 3600,
    uid,
    claims,
  };

  const unsignedToken = `${base64UrlEncode(JSON.stringify(header))}.${base64UrlEncode(JSON.stringify(payload))}`;
  const signature = crypto.sign("RSA-SHA256", Buffer.from(unsignedToken), account.private_key!);
  return `${unsignedToken}.${base64UrlEncode(signature)}`;
}

function getOpenAITranscriptionClient(): OpenAI {
  const apiKey = process.env.OPENAI_API_KEY?.trim();

  if (!apiKey) {
    throw new Error(
      "Falta OPENAI_API_KEY. La consulta del chatbot usa Groq, pero la transcripción de audio todavía requiere OpenAI."
    );
  }

  return new OpenAI({ apiKey });
}

function getMercadoPagoEnvironment(): MercadoPagoEnvironment {
  const environment = String(process.env.MP_ENV || "test")
    .trim()
    .toLowerCase();

  if (environment !== "test" && environment !== "production") {
    throw Object.assign(
      new Error("MP_ENV debe ser test o production."),
      { statusCode: 500 }
    );
  }

  return environment;
}

function getMercadoPagoAccessToken(): string {
  const environment = getMercadoPagoEnvironment();
  const accessToken =
    environment === "production"
      ? process.env.MP_ACCESS_TOKEN?.trim()
      : process.env.MP_ACCESS_TOKEN_TEST?.trim();

  if (!accessToken) {
    throw Object.assign(
      new Error(
        environment === "production"
          ? "Falta configurar MP_ACCESS_TOKEN en el backend."
          : "Falta configurar MP_ACCESS_TOKEN_TEST en el backend."
      ),
      { statusCode: 500 }
    );
  }

  return accessToken;
}

function buildMercadoPagoPayer(
  environment: MercadoPagoEnvironment,
  afiliadoNombre: string,
  dni: string
): Record<string, unknown> {
  if (environment === "test") {
    return {};
  }

  return {
    payer: {
      name: afiliadoNombre,
      identification: {
        type: "DNI",
        number: dni,
      },
    },
  };
}

function getMercadoPagoBackUrls(): {
  success: string;
  pending: string;
  failure: string;
} {
  const success = process.env.MP_BACK_URL_SUCCESS?.trim();
  const pending = process.env.MP_BACK_URL_PENDING?.trim();
  const failure = process.env.MP_BACK_URL_FAILURE?.trim();
  const faltantes = [
    !success ? "MP_BACK_URL_SUCCESS" : null,
    !pending ? "MP_BACK_URL_PENDING" : null,
    !failure ? "MP_BACK_URL_FAILURE" : null,
  ].filter(Boolean);

  if (faltantes.length > 0) {
    throw new Error(
      `Faltan configurar las URL de retorno de Mercado Pago: ${faltantes.join(
        ", "
      )}.`
    );
  }

  return {
    success: success as string,
    pending: pending as string,
    failure: failure as string,
  };
}

function normalizeDni(dni: string | number | null | undefined): string {
  return String(dni ?? "").replace(/\D/g, "");
}

function assertValidDni(dni: string): string {
  if (!/^\d{6,9}$/.test(dni)) {
    throw Object.assign(new Error("DNI inválido."), {
      statusCode: 400,
    });
  }

  return dni;
}

function getDocumentoId(name: string): string {
  return name.split("/").pop() || name;
}

function firestoreValueToJs(value: FirestoreValue): any {
  if ("stringValue" in value) return value.stringValue;
  if ("integerValue" in value) return Number(value.integerValue);
  if ("doubleValue" in value) return value.doubleValue;
  if ("booleanValue" in value) return value.booleanValue;
  if ("timestampValue" in value) return value.timestampValue;
  if ("nullValue" in value) return null;
  if ("arrayValue" in value) {
    return (value.arrayValue?.values || []).map((v) => firestoreValueToJs(v));
  }
  if ("mapValue" in value) {
    return firestoreFieldsToJs(value.mapValue?.fields || {});
  }
  return undefined;
}

function firestoreFieldsToJs(fields: Record<string, FirestoreValue>): FirestoreRecord {
  return Object.fromEntries(
    Object.entries(fields).map(([key, value]) => [key, firestoreValueToJs(value)])
  );
}

function firestoreDocToJs(doc: FirestoreDocument): FirestoreRecord {
  const data = firestoreFieldsToJs(doc.fields || {});
  return {
    ...data,
    id: getDocumentoId(doc.name),
    path: doc.name,
    _name: doc.name,
  };
}

function jsToFirestoreValue(value: any): FirestoreValue {
  if (value === null || value === undefined) return { nullValue: null };
  if (value instanceof Date) return { timestampValue: value.toISOString() };
  if (typeof value === "boolean") return { booleanValue: value };
  if (typeof value === "number") {
    return Number.isInteger(value)
      ? { integerValue: String(value) }
      : { doubleValue: value };
  }
  if (Array.isArray(value)) {
    return { arrayValue: { values: value.map((item) => jsToFirestoreValue(item)) } };
  }
  if (typeof value === "object") {
    return {
      mapValue: {
        fields: Object.fromEntries(
          Object.entries(value).map(([key, item]) => [key, jsToFirestoreValue(item)])
        ),
      },
    };
  }
  return { stringValue: String(value) };
}

function jsToFirestoreFields(data: Record<string, any>): Record<string, FirestoreValue> {
  return Object.fromEntries(
    Object.entries(data).map(([key, value]) => [key, jsToFirestoreValue(value)])
  );
}

let googleAccessTokenCache:
  | {
      token: string;
      expiresAt: number;
    }
  | null = null;

async function getGoogleAccessToken(): Promise<string> {
  const explicitToken = process.env.GOOGLE_OAUTH_ACCESS_TOKEN?.trim();
  if (explicitToken) return explicitToken;

  if (
    googleAccessTokenCache &&
    googleAccessTokenCache.expiresAt > Date.now() + 60_000
  ) {
    return googleAccessTokenCache.token;
  }

  const metadataResponse = await fetch(
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
    {
      headers: {
        "Metadata-Flavor": "Google",
      },
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    }
  );

  if (!metadataResponse.ok) {
    throw new Error(
      "No se pudo obtener token de Google para Firestore. Configurá GOOGLE_OAUTH_ACCESS_TOKEN en local o ejecutá en Cloud Run con service account."
    );
  }

  const data = await metadataResponse.json();
  if (!data?.access_token) {
    throw new Error("La metadata de Google no devolvió access_token para Firestore.");
  }

  const expiresInSeconds = Number(data.expires_in || 3000);
  googleAccessTokenCache = {
    token: String(data.access_token),
    expiresAt: Date.now() + Math.max(60, expiresInSeconds) * 1000,
  };

  return googleAccessTokenCache.token;
}

async function firestoreRequest<T>(
  url: string,
  init: RequestInit = {}
): Promise<T | null> {
  const accessToken = await getGoogleAccessToken();
  const response = await fetch(url, {
    ...init,
    signal: init.signal || AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    headers: {
      Authorization: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
      ...(init.headers || {}),
    },
  });

  if (response.status === 404) return null;

  const data = await response.json().catch(() => null);

  if (!response.ok) {
    const detail = data?.error?.message || data?.message || response.statusText;
    // El mensaje se mantiene igual para no cambiar lo que ya loguean o
    // muestran los llamadores. Se adjunta además el estado estructurado, que
    // hace falta para distinguir una precondición incumplida de un error real
    // sin tener que leer el texto.
    throw Object.assign(new Error(`Firestore ${response.status}: ${detail}`), {
      firestoreStatus: String(data?.error?.status || ""),
      firestoreHttpStatus: response.status,
    });
  }

  return data as T;
}

async function getFirestoreDoc(path: string): Promise<FirestoreRecord | null> {
  const doc = await firestoreRequest<FirestoreDocument>(
    `${firestoreBaseUrl}/${path}`
  );
  return doc ? firestoreDocToJs(doc) : null;
}

async function setFirestoreDoc(
  path: string,
  data: Record<string, any>
): Promise<FirestoreRecord> {
  const doc = await firestoreRequest<FirestoreDocument>(`${firestoreBaseUrl}/${path}`, {
    method: "PATCH",
    body: JSON.stringify({ fields: jsToFirestoreFields(data) }),
  });

  if (!doc) throw new Error("No se pudo guardar el documento en Firestore.");
  return firestoreDocToJs(doc);
}

/**
 * Firestore REST responses expose document paths as full resource names
 * (projects/.../documents/collection/id), while the REST helpers above expect
 * the relative path after /documents/.
 */
function getFirestoreRelativePath(value: FirestoreRecord | string): string {
  const raw = typeof value === "string"
    ? value
    : String(value?.path || value?._name || "");
  const normalized = raw.trim();
  if (!normalized) throw new Error("Ruta Firestore vacía.");
  const marker = "/documents/";
  const markerIndex = normalized.indexOf(marker);
  return markerIndex >= 0
    ? normalized.slice(markerIndex + marker.length)
    : normalized.replace(/^\/+/, "");
}

async function updateFirestoreDoc(
  path: string,
  data: Record<string, any>
): Promise<FirestoreRecord> {
  const updateMask = Object.keys(data)
    .map((field) => `updateMask.fieldPaths=${encodeURIComponent(field)}`)
    .join("&");
  const url = `${firestoreBaseUrl}/${path}${updateMask ? `?${updateMask}` : ""}`;
  const doc = await firestoreRequest<FirestoreDocument>(url, {
    method: "PATCH",
    body: JSON.stringify({ fields: jsToFirestoreFields(data) }),
  });

  if (!doc) throw new Error("No se pudo actualizar el documento en Firestore.");
  return firestoreDocToJs(doc);
}

async function createFirestoreDoc(
  collectionPath: string,
  documentId: string,
  data: Record<string, any>
): Promise<FirestoreRecord> {
  const doc = await firestoreRequest<FirestoreDocument>(
    `${firestoreBaseUrl}/${collectionPath}?documentId=${encodeURIComponent(documentId)}`,
    {
      method: "POST",
      body: JSON.stringify({ fields: jsToFirestoreFields(data) }),
    }
  );

  if (!doc) throw new Error("No se pudo crear el documento en Firestore.");
  return firestoreDocToJs(doc);
}

async function addFirestoreDoc(
  collectionPath: string,
  data: Record<string, any>
): Promise<FirestoreRecord> {
  const doc = await firestoreRequest<FirestoreDocument>(
    `${firestoreBaseUrl}/${collectionPath}`,
    {
      method: "POST",
      body: JSON.stringify({ fields: jsToFirestoreFields(data) }),
    }
  );

  if (!doc) throw new Error("No se pudo crear el documento en Firestore.");
  return firestoreDocToJs(doc);
}

/**
 * Elimina UN documento de Firestore.
 *
 * Firestore REST borra únicamente el documento indicado: sus subcolecciones,
 * si las hubiera, quedan huérfanas pero intactas, y ningún otro documento se
 * ve afectado. No hay borrado en cascada.
 *
 * Devuelve true si la operación se ejecutó y false si el documento no existía
 * (firestoreRequest traduce el 404 a null).
 */
async function deleteFirestoreDoc(path: string): Promise<boolean> {
  const resultado = await firestoreRequest<Record<string, never>>(
    `${firestoreBaseUrl}/${path}`,
    { method: "DELETE" }
  );

  return resultado !== null;
}

function makeFieldFilter(field: string, op: string, value: any) {
  return {
    fieldFilter: {
      field: { fieldPath: field },
      op,
      value: jsToFirestoreValue(value),
    },
  };
}

async function queryFirestoreCollection(
  collectionId: string,
  filters: Array<{ field: string; op?: string; value: any }>,
  limit = 50
): Promise<FirestoreRecord[]> {
  const where =
    filters.length === 0
      ? undefined
      : filters.length === 1
      ? makeFieldFilter(filters[0].field, filters[0].op || "EQUAL", filters[0].value)
      : {
          compositeFilter: {
            op: "AND",
            filters: filters.map((filter) =>
              makeFieldFilter(filter.field, filter.op || "EQUAL", filter.value)
            ),
          },
        };

  const result = await firestoreRequest<Array<{ document?: FirestoreDocument }>>(
    `${firestoreBaseUrl}:runQuery`,
    {
      method: "POST",
      body: JSON.stringify({
        structuredQuery: {
          from: [{ collectionId }],
          ...(where ? { where } : {}),
          limit,
        },
      }),
    }
  );

  return (result || [])
    .map((row) => row.document)
    .filter((doc): doc is FirestoreDocument => Boolean(doc))
    .map((doc) => firestoreDocToJs(doc));
}

/**
 * Consulta un COLLECTION GROUP: todas las subcolecciones que se llaman igual,
 * sin importar bajo qué documento cuelguen.
 *
 * Es una función aparte de queryFirestoreCollection() a propósito: aquella
 * consulta colecciones raíz y la usan Mercado Pago, el bootstrap y las
 * búsquedas por DNI. Acá se agrega allDescendants: true, que cambia por
 * completo el conjunto consultado, así que no se mezclan.
 *
 * Pagina internamente ordenando por __name__ y avanzando con un cursor
 * startAt/before:false (equivalente a startAfter), porque runQuery no
 * devuelve pageToken. Así un curso con cientos o miles de aprobaciones se
 * resuelve completo y no queda truncado por el límite de una sola llamada.
 *
 * Nota sobre índices: se recomienda pasar un único filtro de igualdad. Una
 * igualdad + orderBy __name__ la resuelve el índice de campo único que
 * Firestore mantiene solo. Con dos o más igualdades haría falta un índice
 * compuesto de ámbito COLLECTION_GROUP creado a mano.
 */
async function queryFirestoreCollectionGroup(
  collectionId: string,
  filters: Array<{ field: string; op?: string; value: any }>,
  limit = 10_000,
  pageSize = 300
): Promise<FirestoreRecord[]> {
  const where =
    filters.length === 0
      ? undefined
      : filters.length === 1
      ? makeFieldFilter(filters[0].field, filters[0].op || "EQUAL", filters[0].value)
      : {
          compositeFilter: {
            op: "AND",
            filters: filters.map((filter) =>
              makeFieldFilter(filter.field, filter.op || "EQUAL", filter.value)
            ),
          },
        };

  const documentos: FirestoreRecord[] = [];
  let cursor: string | null = null;

  while (documentos.length < limit) {
    const restantes = limit - documentos.length;
    const tamanoPagina = Math.min(pageSize, restantes);

    const structuredQuery: Record<string, any> = {
      from: [{ collectionId, allDescendants: true }],
      ...(where ? { where } : {}),
      orderBy: [{ field: { fieldPath: "__name__" }, direction: "ASCENDING" }],
      limit: tamanoPagina,
    };

    if (cursor) {
      structuredQuery.startAt = {
        values: [{ referenceValue: cursor }],
        before: false,
      };
    }

    const result = await firestoreRequest<Array<{ document?: FirestoreDocument }>>(
      `${firestoreBaseUrl}:runQuery`,
      {
        method: "POST",
        body: JSON.stringify({ structuredQuery }),
      }
    );

    const pagina = (result || [])
      .map((row) => row.document)
      .filter((doc): doc is FirestoreDocument => Boolean(doc));

    if (pagina.length === 0) break;

    for (const doc of pagina) documentos.push(firestoreDocToJs(doc));

    cursor = pagina[pagina.length - 1].name;

    // Página incompleta ⇒ no quedan más resultados.
    if (pagina.length < tamanoPagina) break;
  }

  return documentos;
}

/**
 * Consulta una subcolección concreta, colgada de UN documento padre.
 *
 * Distinta de las otras dos:
 *   queryFirestoreCollection      → colección raíz
 *   queryFirestoreCollectionGroup → todas las subcolecciones con ese nombre
 *   esta                          → sólo la de ESE padre
 *
 * Firestore REST lo resuelve poniendo el padre en la URL del runQuery, sin
 * allDescendants: así la consulta queda acotada al documento indicado y no
 * necesita índices compuestos.
 *
 * Se usa para leer certificados/{cursoId}/emitidos sin mezclar los emitidos
 * de otros cursos.
 */
async function queryFirestoreChildCollection(
  parentPath: string,
  collectionId: string,
  filters: Array<{ field: string; op?: string; value: any }>,
  limit = 20
): Promise<FirestoreRecord[]> {
  const where =
    filters.length === 0
      ? undefined
      : filters.length === 1
      ? makeFieldFilter(filters[0].field, filters[0].op || "EQUAL", filters[0].value)
      : {
          compositeFilter: {
            op: "AND",
            filters: filters.map((filter) =>
              makeFieldFilter(filter.field, filter.op || "EQUAL", filter.value)
            ),
          },
        };

  const result = await firestoreRequest<Array<{ document?: FirestoreDocument }>>(
    `${firestoreBaseUrl}/${parentPath}:runQuery`,
    {
      method: "POST",
      body: JSON.stringify({
        structuredQuery: {
          from: [{ collectionId }],
          ...(where ? { where } : {}),
          limit,
        },
      }),
    }
  );

  return (result || [])
    .map((row) => row.document)
    .filter((doc): doc is FirestoreDocument => Boolean(doc))
    .map((doc) => firestoreDocToJs(doc));
}

function cursoIdDesdeRegistroInscriptos(archivo: FirestoreRecord): string {
  const cursoId = String(archivo?.cursoId || "").trim();
  if (cursoId) return cursoId;

  const relativePath = getFirestoreRelativePath(archivo);
  const match = relativePath.match(/^certificados\/([^/]+)\/registroInscriptos\/[^/]+$/);
  return String(match?.[1] || "").trim();
}

function fechaRegistroInscriptos(archivo: FirestoreRecord): number {
  const timestamp = Date.parse(String(archivo?.subidoEn || ""));
  return Number.isFinite(timestamp) ? timestamp : 0;
}

/**
 * Caché corta de la metadata de Registro Inscriptos (nunca de los archivos).
 *
 * listarRegistroInscriptosActivos hace una collection-group query sin filtro
 * sobre TODOS los cursos y, para los que no tienen título en el propio
 * documento, un getFirestoreDoc adicional. Repetirlo en cada apertura de la
 * pestaña —y cada vez que el validador vuelve a ella— es exactamente el
 * patrón de "escanear miles de documentos en cada request" que hay que
 * evitar.
 *
 * Sólo se cachean cursoId, título, archivoId, nombreOriginal, size, mimeType
 * y subidoEn: nunca el contenido de un archivo. TTL 45s. Se invalida de
 * inmediato al subir o eliminar una planilla, así que un administrador que
 * sube un archivo lo ve reflejado sin esperar el vencimiento; el botón
 * "Actualizar" del validador fuerza una recarga real vía forzar=true.
 */
let registroInscriptosCache: { datos: any[]; expiraEn: number } | null = null;
const REGISTRO_INSCRIPTOS_CACHE_TTL_MS = 45_000;

function invalidarRegistroInscriptosCache() {
  registroInscriptosCache = null;
}

async function listarRegistroInscriptosActivos(forzar = false): Promise<any[]> {
  if (!forzar && registroInscriptosCache && registroInscriptosCache.expiraEn > Date.now()) {
    return registroInscriptosCache.datos;
  }

  const datos = await listarRegistroInscriptosActivosSinCache();
  registroInscriptosCache = { datos, expiraEn: Date.now() + REGISTRO_INSCRIPTOS_CACHE_TTL_MS };
  return datos;
}

async function listarRegistroInscriptosActivosSinCache(): Promise<any[]> {
  let archivos: FirestoreRecord[];
  try {
    archivos = await queryFirestoreCollectionGroup(
      "registroInscriptos",
      [],
      10_000
    );
  } catch (error) {
    console.error("[sidca-chatbot-backend] Error consultando collection group registroInscriptos:", error);
    throw error;
  }

  const activos = archivos.filter((archivo) => archivo.activo === true);
  const porCurso = new Map<string, { cursoId: string; titulo?: string; archivos: any[] }>();
  for (const archivo of activos) {
    const cursoId = cursoIdDesdeRegistroInscriptos(archivo);
    if (!cursoId) {
      console.warn("[sidca-chatbot-backend] Planilla activa sin cursoId ni ruta válida:", archivo.path || archivo.id);
      continue;
    }
    const tituloCurso = String(archivo.tituloCurso || "").trim();
    const curso = porCurso.get(cursoId) || { cursoId, titulo: tituloCurso, archivos: [] };
    if (!curso.titulo && tituloCurso) curso.titulo = tituloCurso;
    curso.archivos.push({
      archivoId: String(archivo.archivoId || archivo.id || ""),
      nombreOriginal: String(archivo.nombreOriginal || "planilla.xlsx"),
      size: archivo.size,
      mimeType: archivo.mimeType,
      subidoEn: archivo.subidoEn,
    });
    porCurso.set(cursoId, curso);
  }

  const cursos = await Promise.all([...porCurso.values()].map(async (curso) => {
    if (!curso.titulo) {
      const documentoCurso = await getFirestoreDoc(`cursos/${curso.cursoId}`);
      curso.titulo = String(documentoCurso?.titulo || "").trim();
    }
    curso.titulo = curso.titulo || "Capacitación sin título";
    curso.archivos.sort((a, b) => fechaRegistroInscriptos(b) - fechaRegistroInscriptos(a));
    return {
      cursoId: curso.cursoId,
      titulo: curso.titulo,
      cantidadArchivos: curso.archivos.length,
      archivos: curso.archivos,
    };
  }));

  const cursosOrdenados = cursos.sort((a, b) => String(a.titulo).localeCompare(String(b.titulo), "es", { sensitivity: "base" }));
  console.info(`[registro-inscriptos] documentos collection-group=${archivos.length} activos=${activos.length} cursos=${cursosOrdenados.length}`);
  return cursosOrdenados;
}

async function verifyFirebaseIdToken(authorization?: string): Promise<AuthenticatedUser> {
  const token = authorization?.startsWith("Bearer ")
    ? authorization.slice("Bearer ".length).trim()
    : "";

  if (!token) {
    throw Object.assign(new Error("Falta Authorization Bearer con Firebase ID Token."), {
      statusCode: 401,
    });
  }

  const { payload } = await jwtVerify(token, firebaseJwks, {
    issuer: firebaseIssuer,
    audience: firebaseProjectId,
  });

  if (!payload.sub) {
    throw Object.assign(new Error("Token Firebase inválido: falta UID."), {
      statusCode: 401,
    });
  }

  return {
    uid: payload.sub,
    email: typeof payload.email === "string" ? payload.email : undefined,
  };
}

/**
 * Busca el documento del usuario autenticado.
 * Primero intenta usuarios/{uid}; si el proyecto usa otro ID de documento,
 * busca el UID en los campos históricos más habituales.
 */
async function findUsuarioByAuthUid(uid: string): Promise<FirestoreRecord | null> {
  for (const collection of ["usuarios", "nuevoAfiliado"]) {
    const direct = await getFirestoreDoc(`${collection}/${uid}`);
    if (direct) return direct;
  }

  const uidFields = ["uid", "usuarioId", "userId", "authUid"];

  for (const field of uidFields) {
    for (const collection of ["usuarios", "nuevoAfiliado"]) {
      const matches = await queryFirestoreCollection(collection, [{ field, value: uid }], 2);
      if (matches.length > 0) return matches[0];
    }
  }

  return null;
}

/**
 * Busca los registros del usuario autenticado en usuarios/nuevoAfiliado, por
 * uid y, si hace falta, por email.
 *
 * Antes cada consulta se esperaba una por una (hasta 16 round-trips
 * secuenciales en el peor caso, dos de ellos un escaneo COMPLETO sin filtro
 * de hasta 10.000 documentos). Acá cada nivel dispara sus consultas
 * INDEPENDIENTES con Promise.all: lo que tarda el nivel es el máximo de sus
 * consultas, no la suma. El conjunto de documentos que devuelve es
 * exactamente el mismo — mismas colecciones, mismos campos, mismo orden de
 * prioridad (uid antes que email, email antes que el escaneo completo) — sólo
 * cambia que ya no se esperan de a una.
 *
 * El escaneo completo sigue siendo el último recurso: caro, pero el resultado
 * de esta función se cachea con TTL corto en requireValidadorCertificados, así
 * que un validador cuyo uid no está indexado no vuelve a pagarlo en cada
 * request, sólo cuando vence la caché.
 */
async function findRegistrosValidadorByAuth(authUser: AuthenticatedUser): Promise<FirestoreRecord[]> {
  const encontrados = new Map<string, FirestoreRecord>();
  const agregar = (docs: FirestoreRecord[]) => docs.forEach((doc) => {
    const key = String(doc.path || doc.name || `${doc.dni}:${doc.email || doc.correo || ""}`);
    encontrados.set(key, doc);
  });
  const colecciones = ["usuarios", "nuevoAfiliado"] as const;

  const inicioFaseUid = Date.now();
  const [directos, porCampoUid] = await Promise.all([
    Promise.all(
      colecciones.map((coleccion) => getFirestoreDoc(`${coleccion}/${authUser.uid}`))
    ),
    Promise.all(
      ["uid", "usuarioId", "userId", "authUid"].flatMap((campo) =>
        colecciones.map((coleccion) =>
          queryFirestoreCollection(coleccion, [{ field: campo, value: authUser.uid }], 50)
        )
      )
    ),
  ]);
  directos.forEach((doc) => { if (doc) agregar([doc]); });
  porCampoUid.forEach((docs) => agregar(docs));
  console.log(`[perf] buscar-validador faseUid=${Date.now() - inicioFaseUid}ms encontrados=${encontrados.size}`);

  if (encontrados.size === 0 && authUser.email) {
    const email = authUser.email.trim().toLowerCase();

    const porCampoEmail = await Promise.all(
      ["email", "correo", "mail"].flatMap((campo) =>
        colecciones.map(async (coleccion) => {
          const candidatos = await queryFirestoreCollection(coleccion, [{ field: campo, value: authUser.email }], 100);
          return candidatos.filter((doc) => String(doc[campo] || "").trim().toLowerCase() === email);
        })
      )
    );
    porCampoEmail.forEach((docs) => agregar(docs));

    if (encontrados.size === 0) {
      const escaneos = await Promise.all(
        colecciones.map((coleccion) => queryFirestoreCollection(coleccion, [], 10000))
      );
      escaneos.forEach((todos) => {
        agregar(todos.filter((doc) => [doc.email, doc.correo, doc.mail].some((valor) => String(valor || "").trim().toLowerCase() === email)));
      });
    }
  }
  return [...encontrados.values()];
}

/**
 * Resuelve el permiso de validador SIN caché. Es el cuerpo original de
 * requireValidadorCertificados, extraído tal cual: ninguna regla de
 * autorización cambia acá, sólo se envuelve en una caché más abajo.
 */
async function resolverValidadorCertificadosSinCache(
  authUser: AuthenticatedUser
): Promise<FirestoreRecord> {
  const registros = await findRegistrosValidadorByAuth(authUser);
  if (registros.length === 0) {
    throw Object.assign(
      new Error("El usuario autenticado no está registrado en SIDCA."),
      { statusCode: 403 }
    );
  }

  const dnis = new Set(registros.map((doc) => normalizeDni(doc.dni)).filter(Boolean));
  if (dnis.size > 1) {
    console.warn(`[sidca-chatbot-backend] validador email ambiguo uid=${authUser.uid}`);
    throw Object.assign(new Error("No tenés autorización para validar certificados SIDCA."), { statusCode: 403 });
  }
  const autorizado = registros.find((doc) => doc.validarCertificados === true);
  console.info(`[sidca-chatbot-backend] validador resuelto uid=${authUser.uid} origen=${registros.map((d) => String(d.path || "").split("/")[0]).filter(Boolean).join("+")} dni=${[...dnis][0] || "-"} permiso=${Boolean(autorizado)}`);
  if (!autorizado) {
    throw Object.assign(
      new Error("No tenés autorización para validar certificados SIDCA."),
      { statusCode: 403 }
    );
  }

  return autorizado;
}

/**
 * Caché corta de permiso de validador: uid -> resultado ya resuelto.
 *
 * Resolver un validador puede costar hasta dos escaneos completos sin filtro
 * (usuarios + nuevoAfiliado, hasta 10.000 documentos cada uno) cuando su uid
 * no está indexado en ningún campo habitual. Sin caché, ESO se repetía en
 * cada pedido: cada certificado escaneado, cada apertura de Registro
 * Inscriptos, cada descarga.
 *
 * TTL corto (45s) a propósito: si un administrador le retira el permiso a un
 * validador, tarda como máximo 45s en reflejarse, no el resto de la sesión.
 * Nunca se guardan tokens ni contraseñas, sólo el documento ya resuelto (o el
 * motivo del rechazo) y su vencimiento.
 *
 * Vive en memoria del proceso: cada instancia de Cloud Run tiene la suya, se
 * pierde en cada reinicio/revisión nueva, y no se comparte entre instancias.
 * Es exactamente el comportamiento buscado para algo de este TTL.
 */
type ValidadorPermisoCacheEntry =
  | { ok: true; registro: FirestoreRecord; expiraEn: number }
  | { ok: false; mensaje: string; statusCode: number; expiraEn: number };

const VALIDADOR_PERMISO_CACHE_TTL_MS = 45_000;
const validadorPermisoCache = new Map<string, ValidadorPermisoCacheEntry>();
// Una misma identidad puede abrir varias acciones a la vez. Mientras se
// resuelve su permiso, las solicitudes simultáneas comparten una sola carga.
const validadorPermisoEnCurso = new Map<string, Promise<FirestoreRecord>>();

/**
 * Exige que el usuario tenga permiso explícito para validar certificados.
 * El permiso se controla mediante:
 *
 * validarCertificados: true
 *
 * en el documento correspondiente de la colección usuarios.
 */
async function requireValidadorCertificados(
  authUser: AuthenticatedUser
): Promise<FirestoreRecord> {
  const cacheado = validadorPermisoCache.get(authUser.uid);

  if (cacheado && cacheado.expiraEn > Date.now()) {
    console.log(`[perf] resolverPermiso uid=${authUser.uid} cache=hit resultado=${cacheado.ok ? "autorizado" : "rechazado"}`);
    if (cacheado.ok) return cacheado.registro;
    throw Object.assign(new Error(cacheado.mensaje), { statusCode: cacheado.statusCode });
  }

  const resolucionExistente = validadorPermisoEnCurso.get(authUser.uid);
  if (resolucionExistente) {
    console.log(`[perf] resolverPermiso uid=${authUser.uid} cache=shared`);
    return resolucionExistente;
  }

  const inicio = Date.now();
  const resolucion = resolverValidadorCertificadosSinCache(authUser)
    .then((autorizado) => {
      validadorPermisoCache.set(authUser.uid, {
        ok: true,
        registro: autorizado,
        expiraEn: Date.now() + VALIDADOR_PERMISO_CACHE_TTL_MS,
      });
      console.log(`[perf] resolverPermiso uid=${authUser.uid} tiempo=${Date.now() - inicio}ms cache=miss resultado=autorizado`);
      return autorizado;
    })
    .catch((error: any) => {
      const statusCode = Number(error?.statusCode);

      // Sólo un rechazo funcional 403 se cachea negativamente. Los fallos de
      // infraestructura se reintentan de inmediato en el siguiente pedido.
      if (statusCode === 403) {
        validadorPermisoCache.set(authUser.uid, {
          ok: false,
          mensaje: String(error?.message || "No tenés autorización para validar certificados SIDCA."),
          statusCode,
          expiraEn: Date.now() + VALIDADOR_PERMISO_CACHE_TTL_MS,
        });
      }
      console.log(`[perf] resolverPermiso uid=${authUser.uid} tiempo=${Date.now() - inicio}ms cache=miss resultado=rechazado status=${Number.isFinite(statusCode) ? statusCode : "sin_status"}`);
      throw error;
    })
    .finally(() => {
      validadorPermisoEnCurso.delete(authUser.uid);
    });

  validadorPermisoEnCurso.set(authUser.uid, resolucion);
  return resolucion;
}

/**
 * Firebase UIDs autorizados a administrar el módulo de certificados.
 *
 * Se configuran exclusivamente por variable de entorno, separados por coma.
 * No se hardcodea ningún UID en el código.
 *
 * Si la variable está vacía, el Set queda vacío y ningún usuario obtiene
 * permisos administrativos de forma automática.
 */
const certificadosAdminUids = new Set(
  String(process.env.CERTIFICADOS_ADMIN_UIDS || "")
    .split(",")
    .map((uid) => uid.trim())
    .filter(Boolean)
);

/**
 * Exige que el usuario autenticado sea administrador del módulo de
 * certificados.
 *
 * Recibe únicamente el resultado ya verificado de verifyFirebaseIdToken(),
 * de modo que el UID comparado proviene de la firma del Firebase ID Token
 * y nunca de body, query o headers enviados por el cliente.
 *
 * Este permiso es independiente de validarCertificados, que corresponde a
 * los validadores designados.
 */
async function requireAdministrador(
  authUser: AuthenticatedUser
): Promise<AuthenticatedUser> {
  if (!certificadosAdminUids.has(authUser.uid)) {
    throw Object.assign(
      new Error(
        "No tenés autorización administrativa para gestionar certificados SIDCA."
      ),
      { statusCode: 403 }
    );
  }

  return authUser;
}

/** Permiso resuelto para operar sobre la validación de certificados. */
type PermisoCertificados =
  | { tipo: "administrador"; usuario: null }
  | { tipo: "validador"; usuario: FirestoreRecord };

/**
 * Exige ser administrador del módulo O validador designado.
 *
 * Es el permiso de la VALIDACIÓN de certificados, más amplio que el
 * administrativo: un validador designado puede verificar un certificado
 * escaneado sin poder configurar ni emitir.
 *
 * Comprueba la allowlist administrativa de forma directa en vez de llamar a
 * requireAdministrador() y atrapar su 403: capturar excepciones como control
 * de flujo confundiría un error de infraestructura con una falta de permiso.
 */
async function requireAdministradorOValidadorCertificados(
  authUser: AuthenticatedUser
): Promise<PermisoCertificados> {
  if (certificadosAdminUids.has(authUser.uid)) {
    return { tipo: "administrador", usuario: null };
  }

  // Lanza 403 con su propio mensaje si no está designado.
  const usuario = await requireValidadorCertificados(authUser);

  return { tipo: "validador", usuario };
}

async function findDocsByDni(collectionId: "usuarios" | "nuevoAfiliado", dni: string) {
  const numericDni = Number(dni);
  const byString = await queryFirestoreCollection(collectionId, [
    { field: "dni", value: dni },
  ]);
  const byNumber = Number.isFinite(numericDni)
    ? await queryFirestoreCollection(collectionId, [{ field: "dni", value: numericDni }])
    : [];

  const merged = new Map<string, FirestoreRecord>();
  for (const doc of [...byString, ...byNumber]) {
    if (doc.path) merged.set(doc.path, doc);
  }
  return [...merged.values()];
}

function buildNombreAfiliado(doc: FirestoreRecord): string {
  const apellidoNombre = String(
    doc.apellidoNombre || doc.apellido_y_nombre || doc.apellidoYNombre || ""
  ).trim();
  if (apellidoNombre) return apellidoNombre;

  const apellido = String(doc.apellido || "").trim();
  const nombre = String(doc.nombre || "").trim();
  return [apellido, nombre].filter(Boolean).join(", ") || "Afiliado SIDCA";
}

function docBelongsToUid(doc: FirestoreRecord, uid: string): boolean {
  return [doc.uid, doc.usuarioId, doc.userId, doc.authUid, doc.id].some(
    (value) => String(value || "").trim() === uid
  );
}

async function findAfiliadoByDni(dni: string): Promise<FirestoreRecord | null> {
  const usuarios = await findDocsByDni("usuarios", dni);
  if (usuarios.length > 0) return usuarios[0];

  const nuevosAfiliados = await findDocsByDni("nuevoAfiliado", dni);
  return nuevosAfiliados[0] || null;
}

async function validateDniBelongsToUser(dni: string, uid: string): Promise<void> {
  const afiliado = await findAfiliadoByDni(dni);

  if (!afiliado) {
    throw Object.assign(new Error("No existe un afiliado para el DNI indicado."), {
      statusCode: 404,
    });
  }

  if (!docBelongsToUid(afiliado, uid)) {
    throw Object.assign(
      new Error("No se encontró vínculo entre este DNI y el usuario autenticado."),
      { statusCode: 403 }
    );
  }
}

async function getAfiliadoDocs(dni: string) {
  const usuarios = await findDocsByDni("usuarios", dni);
  const nuevoAfiliado = await findDocsByDni("nuevoAfiliado", dni);
  const source = usuarios[0] || nuevoAfiliado[0];

  if (!source) {
    throw Object.assign(new Error("No se encontró el afiliado para el DNI indicado."), {
      statusCode: 404,
    });
  }

  return {
    usuarios,
    nuevoAfiliado,
    afiliadoNombre: buildNombreAfiliado(source),
  };
}

async function getCuotaAdherenteConfig(): Promise<CuotaAdherenteConfig> {
  const config = await getFirestoreDoc("config/cuotaAdherente");

  if (!config) {
    throw Object.assign(
      new Error("No existe la configuración config/cuotaAdherente."),
      { statusCode: 500 }
    );
  }

  const parsed = {
    habilitada: Boolean(config.habilitada),
    periodo: Number(config.periodo),
    importe: Number(config.importe),
    moneda: String(config.moneda || "ARS"),
    concepto: String(config.concepto || "Cuota sindical de adherente SIDCA"),
    detalle: String(config.detalle || ""),
    cuotasMaximas: Number(config.cuotasMaximas || 1),
  };

  if (!parsed.habilitada) {
    throw Object.assign(new Error("El pago de cuota adherente no está habilitado."), {
      statusCode: 409,
    });
  }

  if (
    !Number.isFinite(parsed.periodo) ||
    !Number.isFinite(parsed.importe) ||
    parsed.importe <= 0 ||
    parsed.moneda !== "ARS"
  ) {
    throw Object.assign(
      new Error("La configuración de cuota adherente está incompleta o inválida."),
      { statusCode: 500 }
    );
  }

  return parsed;
}

function isRecentlyCreated(payment: FirestoreRecord): boolean {
  const createdAt = new Date(payment.createdAt || payment.fechaCreacion || 0).getTime();
  const fifteenMinutesAgo = Date.now() - 15 * 60 * 1000;
  return Number.isFinite(createdAt) && createdAt >= fifteenMinutesAgo;
}

async function findExistingPagoAdherente(uid: string, dni: string, periodo: number) {
  const pagos = await queryFirestoreCollection(
    "pagos_adherentes",
    [
      { field: "uid", value: uid },
      { field: "dni", value: dni },
      { field: "periodo", value: periodo },
    ],
    50
  );

  return pagos;
}

function getPagoEstadoInterno(payment: FirestoreRecord): string {
  const raw = String(
    payment.estadoInterno ||
      payment.estadoMercadoPago ||
      payment.estado ||
      "pendiente"
  )
    .trim()
    .toLowerCase();

  const aliases: Record<string, string> = {
    approved: "aprobado",
    pending: "pendiente",
    in_process: "en_proceso",
    rejected: "rechazado",
    cancelled: "cancelado",
    refunded: "devuelto",
    charged_back: "contracargo",
  };

  return aliases[raw] || raw;
}

function inferPagoTipo(payment: FirestoreRecord): PagoTipo {
  const explicit = String(payment.tipoPago || "").trim().toLowerCase();
  if (explicit === "cuota_adherente") return "cuota_adherente";
  if (explicit === "orden_administrativa") return "orden_administrativa";

  const concepto = String(payment.concepto || "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase();

  return concepto.includes("cuota sindical") && concepto.includes("adherente")
    ? "cuota_adherente"
    : "orden_administrativa";
}

function shouldActivateAdherente(payment: FirestoreRecord): boolean {
  const tipoPago = inferPagoTipo(payment);
  if (tipoPago !== "cuota_adherente") return false;

  if (payment.habilitaAdherente === false) return false;
  return true;
}

function assertPagoAdminValido(payment: FirestoreRecord, dni: string) {
  if (normalizeDni(payment.dni) !== dni) {
    throw Object.assign(new Error("La orden de pago no corresponde al DNI indicado."), {
      statusCode: 403,
    });
  }

  const estado = getPagoEstadoInterno(payment);
  if (["aprobado", "approved", "pagado", "cancelado", "cancelled", "vencido", "rechazado", "rejected"].includes(estado)) {
    throw Object.assign(new Error("La orden de pago no se encuentra disponible para abonar."), {
      statusCode: 409,
    });
  }

  const importe = Number(payment.importe);
  const moneda = String(payment.moneda || "ARS");
  if (!Number.isFinite(importe) || importe <= 0 || moneda !== "ARS") {
    throw Object.assign(new Error("La orden de pago tiene importe o moneda inválidos."), {
      statusCode: 500,
    });
  }
}
function getCheckoutUrl(data: any): string {
  const environment = getMercadoPagoEnvironment();
  if (environment === "test" && data?.sandbox_init_point) {
    return data.sandbox_init_point;
  }

  return data?.init_point || data?.sandbox_init_point || "";
}

async function createMercadoPagoPreference(preferenceBody: Record<string, any>) {
  const response = await fetch("https://api.mercadopago.com/checkout/preferences", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${getMercadoPagoAccessToken()}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(preferenceBody),
    signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
  });

  const data = await response.json();

  if (!response.ok) {
    console.error("[sidca-chatbot-backend] Mercado Pago preference error", {
      status: response.status,
      message: data?.message || data?.error || "Error de Mercado Pago",
    });
    throw Object.assign(
      new Error(data?.message || data?.error || "No se pudo crear la preferencia."),
      { statusCode: response.status }
    );
  }

  return data;
}

async function fetchMercadoPagoPayment(paymentId: string): Promise<MercadoPagoPayment> {
  const response = await fetch(
    `https://api.mercadopago.com/v1/payments/${encodeURIComponent(paymentId)}`,
    {
      headers: {
        Authorization: `Bearer ${getMercadoPagoAccessToken()}`,
      },
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    }
  );

  const data = await response.json();

  if (!response.ok) {
    console.error("[sidca-chatbot-backend] Mercado Pago payment error", {
      status: response.status,
      paymentId,
      message: data?.message || data?.error || "Error de Mercado Pago",
    });
    throw Object.assign(
      new Error(data?.message || data?.error || "No se pudo consultar el pago."),
      { statusCode: response.status }
    );
  }

  return data;
}

function extractMercadoPagoSignature(signatureHeader: string | undefined) {
  const parts = Object.fromEntries(
    String(signatureHeader || "")
      .split(",")
      .map((part) => part.trim().split("="))
      .filter(([key, value]) => key && value)
  );

  return {
    ts: parts.ts,
    v1: parts.v1,
  };
}

function verifyMercadoPagoWebhookSignature(
  paymentId: string,
  requestId: string | undefined,
  signatureHeader: string | undefined
): void {
  const secret = process.env.MP_WEBHOOK_SECRET?.trim();
  if (!secret) {
    throw Object.assign(new Error("Falta configurar MP_WEBHOOK_SECRET."), {
      statusCode: 500,
    });
  }

  const { ts, v1 } = extractMercadoPagoSignature(signatureHeader);
  if (!paymentId || !requestId || !ts || !v1) {
    throw Object.assign(new Error("Firma de webhook Mercado Pago incompleta."), {
      statusCode: 401,
    });
  }

  const timestamp = Number(ts);
  const timestampMs = timestamp > 1_000_000_000_000
    ? timestamp
    : timestamp * 1000;
  const age = Math.abs(Date.now() - timestampMs);

  if (!Number.isFinite(timestamp) || age > WEBHOOK_MAX_AGE_MS) {
    throw Object.assign(new Error("La firma del webhook está vencida."), {
      statusCode: 401,
    });
  }

  const manifest = `id:${paymentId};request-id:${requestId};ts:${ts};`;
  const expected = crypto.createHmac("sha256", secret).update(manifest).digest("hex");
  const expectedBuffer = Buffer.from(expected, "hex");
  const receivedBuffer = Buffer.from(v1, "hex");

  if (
    expectedBuffer.length !== receivedBuffer.length ||
    !crypto.timingSafeEqual(expectedBuffer, receivedBuffer)
  ) {
    throw Object.assign(new Error("Firma de webhook Mercado Pago inválida."), {
      statusCode: 401,
    });
  }
}

function mapMercadoPagoStatus(status: string | undefined) {
  switch (status) {
    case "approved":
      return { estadoInterno: "aprobado", procesado: true, revision: false };
    case "pending":
      return { estadoInterno: "pendiente", procesado: false, revision: false };
    case "in_process":
      return { estadoInterno: "en_proceso", procesado: false, revision: false };
    case "rejected":
      return { estadoInterno: "rechazado", procesado: false, revision: false };
    case "cancelled":
      return { estadoInterno: "cancelado", procesado: false, revision: false };
    case "refunded":
      return { estadoInterno: "devuelto", procesado: true, revision: true };
    case "charged_back":
      return { estadoInterno: "contracargo", procesado: true, revision: true };
    default:
      return { estadoInterno: "desconocido", procesado: false, revision: true };
  }
}

async function activateAdherenteAfterPayment(
  dni: string,
  periodo: number,
  mercadoPagoPaymentId: string,
  pagoAdherenteId: string
) {
  if (!Number.isFinite(periodo)) {
    throw Object.assign(
      new Error("El período de la cuota adherente es inválido."),
      { statusCode: 409 }
    );
  }

  const { usuarios, nuevoAfiliado } = await getAfiliadoDocs(dni);
  const update = {
    adherente: true,
    activo: true,
    suspendido: false,
    periodoCuotaPagada: periodo,
    fechaRegularizacion: new Date(),
    medioRegularizacion: "mercado_pago",
    mercadoPagoPaymentId,
    pagoAdherenteId,
    updatedAt: new Date(),
  };

  for (const doc of [...usuarios, ...nuevoAfiliado]) {
    if (doc.path) {
      const relativePath = doc.path.split("/documents/")[1];
      await updateFirestoreDoc(relativePath, update);
    }
  }
}

async function appendPagoEvento(
  pagoId: string,
  data: {
    tipo: string;
    estadoAnterior?: string | null;
    estadoNuevo: string;
    origen: string;
    detalle?: string;
  }
) {
  await addFirestoreDoc(`pagos_adherentes/${pagoId}/eventos`, {
    ...data,
    fecha: new Date(),
  });
}

function publicPagoFields(payment: FirestoreRecord) {
  const estado = getPagoEstadoInterno(payment);

  return {
    pagoId: payment.pagoId || payment.id,
    dni: payment.dni,
    afiliadoNombre: payment.afiliadoNombre,
    periodo: payment.periodo,
    importe: payment.importe,
    moneda: payment.moneda,
    concepto: payment.concepto,
    detalle: payment.detalle,
    tipoPago: inferPagoTipo(payment),
    habilitaAdherente: shouldActivateAdherente(payment),
    estado,
    estadoInterno: estado,
    estadoMercadoPago: payment.estadoMercadoPago || null,
    estadoMercadoPagoDetalle: payment.estadoMercadoPagoDetalle || null,
    mercadoPagoPreferenceId:
      payment.mercadoPagoPreferenceId || payment.preferenceId || null,
    mercadoPagoPaymentId: payment.mercadoPagoPaymentId || null,
    paymentMethodId: payment.paymentMethodId || null,
    paymentTypeId: payment.paymentTypeId || null,
    fechaCreacion: payment.createdAt || payment.fechaCreacion || null,
    fechaPago: payment.fechaPago || null,
    requiereRevisionAdministrativa: Boolean(
      payment.requiereRevisionAdministrativa ||
        payment.revisionAdministrativa
    ),
    comprobante:
      estado === "aprobado"
        ? {
            leyenda: "Comprobante de pago. No válido como factura.",
            mercadoPagoPaymentId: payment.mercadoPagoPaymentId || null,
            comprobanteUrl: payment.comprobanteUrl || null,
          }
        : null,
  };
}

function getAllowedOrigins(): Set<string> {
  const configured = String(process.env.CORS_ALLOWED_ORIGINS || "")
    .split(",")
    .map((origin) => origin.trim())
    .filter(Boolean);

  return new Set([
    "http://localhost:3000",
    "https://sidcagremio.com",
    "https://www.sidcagremio.com",
    "https://sidca-a33f0.web.app",
    "https://sidca-a33f0.firebaseapp.com",
    ...configured,
  ]);
}

function createRateLimit(options: RateLimitOptions) {
  return (
    req: express.Request,
    res: express.Response,
    next: express.NextFunction
  ) => {
    const now = Date.now();
    const clientIp =
      String(req.headers["x-forwarded-for"] || "")
        .split(",")[0]
        .trim() ||
      req.ip ||
      req.socket.remoteAddress ||
      "unknown";
    const key = `${req.path}:${clientIp}`;
    const current = rateLimitStore.get(key);

    if (!current || current.resetAt <= now) {
      rateLimitStore.set(key, {
        count: 1,
        resetAt: now + options.windowMs,
      });
      next();
      return;
    }

    if (current.count >= options.max) {
      const retryAfterSeconds = Math.max(
        1,
        Math.ceil((current.resetAt - now) / 1000)
      );
      res.setHeader("Retry-After", String(retryAfterSeconds));
      res.status(429).json({
        ok: false,
        error: options.message,
      });
      return;
    }

    current.count += 1;
    rateLimitStore.set(key, current);

    if (rateLimitStore.size > 10_000) {
      for (const [storedKey, entry] of rateLimitStore.entries()) {
        if (entry.resetAt <= now) rateLimitStore.delete(storedKey);
      }
    }

    next();
  };
}

const allowedOrigins = getAllowedOrigins();
const bootstrapRateLimit = createRateLimit({
  windowMs: 60_000,
  max: 10,
  message: "Demasiados intentos de autenticación. Esperá un minuto.",
});
const paymentRateLimit = createRateLimit({
  windowMs: 60_000,
  max: 20,
  message: "Demasiados intentos de pago. Esperá un minuto.",
});
const chatbotRateLimit = createRateLimit({
  windowMs: 60_000,
  max: 60,
  message: "Demasiadas consultas. Esperá un minuto.",
});

app.disable("x-powered-by");
app.set("trust proxy", 1);
app.use(
  cors({
    origin(origin, callback) {
      if (!origin || allowedOrigins.has(origin)) {
        callback(null, true);
        return;
      }

      callback(new Error("Origen no permitido por CORS."));
    },
    credentials: true,
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allowedHeaders: [
      "Content-Type",
      "Authorization",
      "X-Request-Id",
      "X-Signature",
    ],
  })
);
app.use(express.json({ limit: "1mb" }));

app.get("/health", (_req, res) => {
  res.status(200).json({
    ok: true,
    service: "sidca-chatbot-backend",
    status: "running",
    timestamp: new Date().toISOString(),
  });
});

// ============================================================
// CERTIFICADOS SIDCA
// Endpoint inicial protegido para comprobar autenticación.
// No modifica Firestore y no afecta Mercado Pago ni chatbot.
//
// DECISIONES DEFINIDAS PARA ETAPAS POSTERIORES (todavía NO implementadas):
//
//   Colección raíz : "certificados" (ya existente en producción).
//                    Las nuevas estructuras serán subcolecciones:
//                      certificados/{certificadoId}/emitidos/{token}
//                      certificados/{certificadoId}/emitidos/{token}/verificaciones/{id}
//                    NO se crea ninguna colección "certificaciones".
//
//   Token del QR   : crypto.randomBytes(24).toString("hex")
//                    -> 48 caracteres hexadecimales en minúscula
//                    -> validación prevista: /^[a-f0-9]{48}$/
//
//   certificadoId  : validación prevista: /^[A-Za-z0-9_-]{1,128}$/
//                    No se limita a 20 caracteres (formato actual de los IDs
//                    automáticos de Firestore) para mantener compatibilidad
//                    con IDs existentes o futuros.
// ============================================================

/**
 * Códigos de error de jose que indican un Firebase ID Token inválido
 * y por lo tanto deben responderse como 401.
 */
const JOSE_UNAUTHORIZED_CODES = new Set([
  "ERR_JWT_EXPIRED", // token vencido
  "ERR_JWT_INVALID", // token malformado
  "ERR_JWS_INVALID", // JWS inválido
  "ERR_JWS_SIGNATURE_VERIFICATION_FAILED", // firma inválida
  "ERR_JWT_CLAIM_VALIDATION_FAILED", // claims inválidos (iss / aud / exp)
  "ERR_JWKS_NO_MATCHING_KEY", // firmado con una clave desconocida
  "ERR_JOSE_ALG_NOT_ALLOWED", // algoritmo no permitido
  "ERR_JOSE_NOT_SUPPORTED", // alg del header no soportado por el JWKS (ej. "none")
]);

/**
 * Códigos HTTP que el módulo de certificados propaga tal cual cuando el
 * error los declara explícitamente.
 */
const CERTIFICADOS_STATUS_CODES = new Set([400, 401, 403, 404, 409, 503]);

/**
 * Traduce un error del módulo de certificados a una respuesta HTTP segura.
 *
 * - Respeta statusCode / status cuando vale 400, 401, 403, 404 o 409.
 * - Mapea los errores de verificación JWT de jose a 401.
 * - Cualquier otro error responde 500 con un mensaje genérico; el detalle
 *   real queda únicamente en console.error, para no exponer service account,
 *   variables de entorno, tokens, credenciales ni rutas internas.
 */
function sendCertificadosError(
  res: express.Response,
  error: any
): express.Response {
  const explicitStatus = Number(error?.statusCode ?? error?.status);

  if (CERTIFICADOS_STATUS_CODES.has(explicitStatus)) {
    const mensajeSeguro = explicitStatus === 503
      ? "La cuenta de servicio no tiene permisos para administrar Firebase Authentication."
      : String(error?.message || "No se pudo completar la operación.");
    return res.status(explicitStatus).json({
      ok: false,
      modulo: "certificados",
      error: String(error?.message || "No se pudo completar la operación."),
    });
  }

  if (JOSE_UNAUTHORIZED_CODES.has(String(error?.code || ""))) {
    return res.status(401).json({
      ok: false,
      modulo: "certificados",
      error: "Token de autenticación inválido o vencido.",
    });
  }

  console.error("[sidca-chatbot-backend] Error certificados:", error);

  return res.status(500).json({
    ok: false,
    modulo: "certificados",
    error: "Error interno del servicio de certificados.",
  });
}

app.get("/api/certificados/health", async (req, res) => {
  try {
    const authUser = await verifyFirebaseIdToken(
      req.headers.authorization
    );

    const usuario = await requireValidadorCertificados(authUser);

    return res.status(200).json({
      ok: true,
      modulo: "certificados",
      mensaje: "Servicio de certificados SIDCA operativo.",
      autenticado: true,
      autorizado: true,
      usuario: {
        uid: authUser.uid,
        email: authUser.email || null,
        nombre: buildNombreAfiliado(usuario),
        dni: usuario.dni || null,
      },
    });
  } catch (error: any) {
    return sendCertificadosError(res, error);
  }
});

// Endpoint de comprobación administrativa.
// Autoriza exclusivamente por CERTIFICADOS_ADMIN_UIDS; no consulta Firestore
// y no expone la lista de UIDs configurados.
app.get("/api/certificados/admin/health", async (req, res) => {
  try {
    const authUser = await verifyFirebaseIdToken(req.headers.authorization);

    await requireAdministrador(authUser);

    return res.status(200).json({
      ok: true,
      modulo: "certificados",
      administrador: true,
      usuario: {
        uid: authUser.uid,
        email: authUser.email || null,
      },
    });
  } catch (error: any) {
    return sendCertificadosError(res, error);
  }
});

// ============================================================
// CONFIGURACIÓN DE CERTIFICADO POR CURSO
//
// Documento: certificados/{cursoId}
//
// El ID del documento es el ID real de cursos/{cursoId}, de modo que la
// configuración principal de un curso se resuelve con un lookup directo y
// no puede duplicarse.
//
// Los documentos históricos de "certificados" tienen IDs automáticos y no
// se tocan: estos endpoints sólo leen/escriben el documento cuyo ID es un
// cursoId válido, y las escrituras usan updateMask, por lo que cualquier
// campo ajeno al módulo (por ejemplo el histórico "imagen") permanece
// intacto.
// ============================================================

/**
 * Formato admitido para cursoId / certificadoId.
 * No se limita a 20 caracteres para no atarse al formato de los IDs
 * automáticos de Firestore.
 */
const CERTIFICADO_ID_REGEX = /^[A-Za-z0-9_-]{1,128}$/;

/**
 * Valida el parámetro de ruta antes de construir cualquier path de Firestore.
 * Evita que un valor con barras o puntos escape de la colección esperada.
 */
function parseCursoIdParam(valor: unknown): string {
  const cursoId = String(valor ?? "").trim();

  if (!CERTIFICADO_ID_REGEX.test(cursoId)) {
    throw Object.assign(new Error("El identificador del curso es inválido."), {
      statusCode: 400,
    });
  }

  return cursoId;
}

/**
 * Formato del token de validación que viaja en el QR.
 *
 * Es exactamente lo que produce crypto.randomBytes(24).toString("hex"):
 * 48 caracteres hexadecimales en minúscula. Deliberadamente más estricta que
 * CERTIFICADO_ID_REGEX — un token no es un ID de documento cualquiera, y
 * cerrar el formato reduce la superficie de lo que puede llegar a Firestore.
 */
const CERTIFICADO_TOKEN_REGEX = /^[a-f0-9]{48}$/;

/** Una emisiÃ³n vigente sÃ³lo estÃ¡ disponible si puede resolver su QR. */
function emisionTieneQrValido(emision: FirestoreRecord): boolean {
  const token = String(emision.token || emision.id || "").trim();
  const urlValidacion = String(emision.urlValidacion || "").trim();

  return CERTIFICADO_TOKEN_REGEX.test(token) && Boolean(urlValidacion);
}

/**
 * Valida el token del certificado antes de usarlo en un path de Firestore.
 * Rechaza barras, espacios, mayúsculas y cualquier longitud distinta de 48.
 */
function parseCertificadoTokenParam(valor: unknown): string {
  const token = String(valor ?? "").trim();

  if (!CERTIFICADO_TOKEN_REGEX.test(token)) {
    throw Object.assign(
      new Error("El código de validación del certificado es inválido."),
      { statusCode: 400 }
    );
  }

  return token;
}

/**
 * Institución que emite el certificado.
 *
 * Determina qué plantilla usa el frontend. Firestore guarda SÓLO el valor
 * semántico —"sidca" o "itm"—, nunca el nombre del archivo PNG: el asset es
 * un detalle del frontend y puede cambiar sin tocar los datos.
 */
const INSTITUCIONES_CERTIFICADO = ["sidca", "itm"] as const;

type InstitucionCertificado = (typeof INSTITUCIONES_CERTIFICADO)[number];

/**
 * Resuelve la institución de una configuración o de un emitido.
 *
 * Las configuraciones anteriores a este campo no lo tienen: se interpretan
 * como "sidca", que era la única plantilla que existía. No se migra nada —
 * el campo queda persistido recién cuando el administrador vuelve a guardar.
 */
function normalizarInstitucionCertificado(
  record: FirestoreRecord | null | undefined
): InstitucionCertificado {
  return record?.institucionCertificado === "itm" ? "itm" : "sidca";
}

/**
 * Autoridad que firma el certificado, en TEXTO.
 *
 * Reemplaza al modelo anterior de firmas con imagen (Cloudinary): ahora el
 * certificado sólo imprime texto, hasta cuatro renglones por autoridad.
 *
 * nombre y cargo admiten vacío a propósito: la configuración se guarda como
 * borrador y puede quedar incompleta. La obligatoriedad real se comprueba al
 * EMITIR, que es cuando el dato se vuelve irreversible.
 *
 * organismo y referencia son opcionales incluso al emitir: no toda autoridad
 * necesita las cuatro líneas. Al faltar llegan como "" y el certificado
 * simplemente no dibuja ese renglón.
 */
const autoridadCertificadoSchema = z.strictObject({
  nombre: z.string().trim().max(160),
  cargo: z.string().trim().max(200),
  organismo: z.string().trim().max(300).optional().default(""),
  referencia: z.string().trim().max(300).optional().default(""),
  orden: z.number().int().min(1).max(2),
});

/**
 * Cuerpo aceptado por PUT.
 *
 * Es estricto a propósito: el cliente no puede inyectar campos arbitrarios.
 * cursoId, cursoTitulo, estadoConfiguracion y la auditoría NO se aceptan
 * desde el body; los resuelve el backend.
 *
 * "firmas" ya no se acepta: strictObject lo rechaza si alguien lo enviara.
 */
const configuracionCertificadoSchema = z.strictObject({
  titulo: z.string().trim().min(1, "El título del certificado es obligatorio.").max(300),
  resolucion: z.string().trim().min(1, "La resolución es obligatoria.").max(200),
  cargaHoraria: z.string().trim().min(1, "La carga horaria es obligatoria.").max(100),
  dias: z.string().trim().min(1, "Las fechas de realización son obligatorias.").max(300),
  fecha: z.string().trim().min(1, "La fecha del certificado es obligatoria.").max(200),
  modalidad: z.string().trim().min(1, "La modalidad es obligatoria.").max(120),
  institucionCertificado: z.enum(INSTITUCIONES_CERTIFICADO),
  autoridades: z.array(autoridadCertificadoSchema).max(2).optional().default([]),
});

const ESTADOS_CONFIGURACION_CERTIFICADO = new Set(["borrador", "lista"]);

/**
 * Autoridades de una configuración, con compatibilidad de LECTURA hacia el
 * modelo anterior.
 *
 * Prioridad:
 *   1. record.autoridades — modelo actual.
 *   2. record.firmas      — configuraciones anteriores: se toman nombre y
 *                           cargo de las dos primeras y se ignoran
 *                           imagenUrl, imagenPublicId y proveedor.
 *
 * Las autoridades guardadas antes de existir organismo/referencia no tienen
 * esos campos: se resuelven a "" acá mismo, así que no hace falta migrar
 * ningún documento. Tampoco se intenta deducirlos partiendo el cargo: si el
 * administrador quiere las cuatro líneas, las carga.
 *
 * No escribe nada en Firestore: el documento legacy queda tal cual hasta que
 * el administrador lo guarde de nuevo.
 */
function obtenerAutoridadesConfiguracion(
  record: FirestoreRecord | null | undefined
): Record<string, any>[] {
  const autoridades = Array.isArray(record?.autoridades)
    ? record!.autoridades
    : null;

  if (autoridades) {
    return autoridades.slice(0, 2).map((autoridad: any, indice: number) => ({
      nombre: String(autoridad?.nombre || "").trim(),
      cargo: String(autoridad?.cargo || "").trim(),
      organismo: String(autoridad?.organismo || "").trim(),
      referencia: String(autoridad?.referencia || "").trim(),
      orden: indice + 1,
    }));
  }

  const firmas = Array.isArray(record?.firmas) ? record!.firmas : [];

  return firmas.slice(0, 2).map((firma: any, indice: number) => ({
    nombre: String(firma?.nombre || "").trim(),
    cargo: String(firma?.cargo || "").trim(),
    organismo: "",
    referencia: "",
    orden: indice + 1,
  }));
}

/**
 * Proyecta el documento de Firestore a la respuesta pública del módulo.
 *
 * Devuelve únicamente los campos que administra esta pantalla. No expone
 * _name / path ni campos históricos como "imagen", cuyo significado no está
 * definido en el código actual.
 */
function mapConfiguracionCertificado(
  record: FirestoreRecord,
  cursoId: string
): Record<string, any> {
  return {
    cursoId: record.cursoId || cursoId,
    cursoTitulo: record.cursoTitulo || "",
    titulo: record.titulo || "",
    resolucion: record.resolucion || "",
    cargaHoraria: record.cargaHoraria || "",
    dias: record.dias || "",
    fecha: record.fecha || "",
    modalidad: record.modalidad || "",
    institucionCertificado: normalizarInstitucionCertificado(record),
    autoridades: obtenerAutoridadesConfiguracion(record),
    estadoConfiguracion: record.estadoConfiguracion || "borrador",
    creadoEn: record.creadoEn || null,
    actualizadoEn: record.actualizadoEn || null,
    creadoPor: record.creadoPor || null,
    actualizadoPor: record.actualizadoPor || null,
  };
}

/**
 * Ordena las autoridades y reasigna orden 1 y 2.
 *
 * Guarda los cuatro textos y el orden: ningún rastro del modelo de imágenes.
 */
function normalizarAutoridadesCertificado(
  autoridades: z.infer<typeof autoridadCertificadoSchema>[]
): Record<string, any>[] {
  return [...autoridades]
    .sort((a, b) => a.orden - b.orden)
    .slice(0, 2)
    .map((autoridad, indice) => ({
      nombre: autoridad.nombre,
      cargo: autoridad.cargo,
      organismo: autoridad.organismo || "",
      referencia: autoridad.referencia || "",
      orden: indice + 1,
    }));
}

const mapValidadorAdmin = (doc: FirestoreRecord) => ({
  usuarioDocId: String(doc.path || "").split("/").pop() || "",
  dni: doc.dni ?? "",
  apellidoNombre: doc.apellidoNombre || buildNombreAfiliado(doc) || "",
  email: doc.email || doc.correo || doc.mail || "",
  validarCertificados: doc.validarCertificados === true,
});

const parseUsuarioDocId = (value: string) => {
  const id = String(value || "").trim();
  if (!id || /[\\/]/.test(id) || id.includes("..")) {
    throw Object.assign(new Error("Identificador de usuario invÃ¡lido."), { statusCode: 400 });
  }
  return id;
};

async function identityToolkitAdminRequest(path: string, body: Record<string, any>) {
  const token = await getGoogleAccessToken();
  const quotaProject = String(process.env.GOOGLE_CLOUD_QUOTA_PROJECT || firebaseProjectId).trim() || firebaseProjectId;
  const response = await fetch(`https://identitytoolkit.googleapis.com/v1/projects/${encodeURIComponent(firebaseProjectId)}${path}`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
      "x-goog-user-project": quotaProject,
    },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const code = String(data?.error?.message || data?.error?.status || "");
    const statusCode = code === "EMAIL_EXISTS" ? 409 : code === "WEAK_PASSWORD" ? 400 : response.status === 403 ? 503 : response.status;
    throw Object.assign(new Error(code), { statusCode, identityCode: code });
  }
  return data || {};
}

async function buscarFirebaseAuthPorEmail(email: string) {
  let data;
  try { data = await identityToolkitAdminRequest("/accounts:lookup", { email: [email] }); }
  catch (error: any) { if (["EMAIL_NOT_FOUND", "USER_NOT_FOUND"].includes(error.identityCode)) return { existe: false }; throw error; }
  const user = Array.isArray(data.users) ? data.users[0] : null;
  return user ? { existe: true, uid: user.localId, email: String(user.email || email).trim().toLowerCase(), disabled: user.disabled === true, displayName: user.displayName || "" } : { existe: false };
}

async function buscarFirebaseAuthPorUid(uid: string) {
  let data;
  try { data = await identityToolkitAdminRequest("/accounts:lookup", { localId: [uid] }); }
  catch (error: any) { if (["EMAIL_NOT_FOUND", "USER_NOT_FOUND"].includes(error.identityCode)) return { existe: false }; throw error; }
  const user = Array.isArray(data.users) ? data.users[0] : null;
  return user ? { existe: true, uid: user.localId, email: String(user.email || "").trim().toLowerCase(), disabled: user.disabled === true, displayName: user.displayName || "" } : { existe: false };
}

async function crearFirebaseAuthValidador(email: string, password: string, displayName: string) {
  const apiKey = String(process.env.FIREBASE_WEB_API_KEY || "").trim();
  if (!apiKey) throw Object.assign(new Error("Falta configurar FIREBASE_WEB_API_KEY para crear accesos de validadores."), { statusCode: 500 });
  return identityToolkitAdminRequest(`/accounts?key=${encodeURIComponent(apiKey)}`, { email, password, displayName, disabled: false });
}

async function actualizarFirebaseAuthValidador(uid: string, disableUser: boolean) {
  return identityToolkitAdminRequest("/accounts:update", { localId: uid, disableUser });
}

async function resolverPersonaPorDocId(id: string) {
  const usuarios = await getFirestoreDoc(`usuarios/${id}`);
  if (usuarios) return usuarios;
  return getFirestoreDoc(`nuevoAfiliado/${id}`);
}

async function registrosPorPersona(dni: string) {
  const normalizado = normalizeDni(dni);
  if (!normalizado) return [];
  return (await Promise.all([findDocsByDni("usuarios", normalizado), findDocsByDni("nuevoAfiliado", normalizado)])).flat();
}

app.get("/api/certificados/admin/validadores", async (req, res) => {
  try {
    const authUser = await verifyFirebaseIdToken(req.headers.authorization);
    await requireAdministrador(authUser);
    const [usuarios, nuevos] = await Promise.all([
      queryFirestoreCollection("usuarios", [{ field: "validarCertificados", value: true }], 500),
      queryFirestoreCollection("nuevoAfiliado", [{ field: "validarCertificados", value: true }], 500),
    ]);
    const unicos = new Map<string, FirestoreRecord>();
    [...usuarios, ...nuevos].forEach((doc) => unicos.set(normalizeDni(doc.dni) || String(doc.path), doc));
    const validadores = [...unicos.values()].map(mapValidadorAdmin).sort((a, b) => a.apellidoNombre.localeCompare(b.apellidoNombre, "es"));
    return res.json({ ok: true, validadores });
  } catch (error: any) { return sendCertificadosError(res, error); }
});

app.get("/api/certificados/admin/validadores/buscar", async (req, res) => {
  try {
    const authUser = await verifyFirebaseIdToken(req.headers.authorization);
    await requireAdministrador(authUser);
    const consulta = String(req.query.q || "").trim();
    if (!consulta) return res.json({ ok: true, usuarios: [] });
    const consultaDni = consulta.replace(/\D/g, "");
    const esBusquedaDni = Boolean(consultaDni) && /^[\d.\s]+$/.test(consulta);
    const claveDocumento = (doc: FirestoreRecord) => {
      const dni = normalizeDni(doc.dni);
      return dni ? `dni:${dni}` : `registro:${String(doc.path || doc.name || "")}`;
    };
    const fusionar = (anterior: FirestoreRecord | undefined, actual: FirestoreRecord) => {
      if (!anterior) return actual;
      return {
        ...anterior,
        ...Object.fromEntries(Object.entries(actual).filter(([, valor]) => valor !== undefined && valor !== null && valor !== "")),
      };
    };
    const unicos = new Map<string, FirestoreRecord>();
    if (esBusquedaDni) {
      const [porUsuarios, porNuevos] = await Promise.all([
        findDocsByDni("usuarios", consultaDni),
        findDocsByDni("nuevoAfiliado", consultaDni),
      ]);
      [...porUsuarios, ...porNuevos].forEach((doc) => {
        if (claveDocumento(doc)) unicos.set(claveDocumento(doc), fusionar(unicos.get(claveDocumento(doc)), doc));
      });
      return res.json({ ok: true, usuarios: [...unicos.values()].map(mapValidadorAdmin) });
    }
    const [documentosUsuarios, documentosNuevos] = await Promise.all([
      queryFirestoreCollection("usuarios", [], 10000),
      queryFirestoreCollection("nuevoAfiliado", [], 10000),
    ]);
    const todos = [...documentosUsuarios, ...documentosNuevos];
    const normal = (v: any) => String(v || "").normalize("NFD").replace(/[\u0300-\u036f]/g, "").replace(/\s+/g, " ").trim().toLowerCase();
    const consultaNormalizada = normal(consulta);
    todos.filter((doc) => [doc.dni, doc.apellido, doc.nombre, doc.apellidoNombre, doc.apellido_y_nombre, doc.apellidoYNombre, doc.email, doc.correo, doc.mail].some((v) => normal(v).includes(consultaNormalizada))).forEach((doc) => {
      if (claveDocumento(doc)) unicos.set(claveDocumento(doc), fusionar(unicos.get(claveDocumento(doc)), doc));
    });
    const usuariosCoincidentes = [...unicos.values()].sort((a, b) => mapValidadorAdmin(a).apellidoNombre.localeCompare(mapValidadorAdmin(b).apellidoNombre, "es", { sensitivity: "base" })).slice(0, 25).map(mapValidadorAdmin);
    return res.json({ ok: true, usuarios: usuariosCoincidentes });
  } catch (error: any) { return sendCertificadosError(res, error); }
});

app.put("/api/certificados/admin/validadores/:usuarioDocId", async (req, res) => {
  try {
    const authUser = await verifyFirebaseIdToken(req.headers.authorization); await requireAdministrador(authUser);
    const id = parseUsuarioDocId(req.params.usuarioDocId);
    const actual = await resolverPersonaPorDocId(id);
    if (!actual) throw Object.assign(new Error("Usuario no encontrado."), { statusCode: 404 });
    const dni = normalizeDni(actual.dni);
    const documentos = dni ? await registrosPorPersona(dni) : [actual];
    const emails = [...new Set(documentos.flatMap((doc) => [doc.email, doc.correo, doc.mail]).map((value) => String(value || "").trim().toLowerCase()).filter((value) => value.includes("@")))];
    const emailSolicitado = String(req.body?.email || "").trim().toLowerCase();
    if (emailSolicitado && emails.length && !emails.includes(emailSolicitado)) throw Object.assign(new Error("El correo no coincide con los datos de la persona seleccionada."), { statusCode: 409 });
    const email = emails[0];
    if (!email) throw Object.assign(new Error("La persona no tiene un correo registrado."), { statusCode: 409 });
    const nombreCompleto = buildNombreAfiliado(actual);
    let cuenta = actual.authUid ? await buscarFirebaseAuthPorUid(String(actual.authUid)) : { existe: false };
    if (!cuenta.existe) cuenta = await buscarFirebaseAuthPorEmail(email);
    const password = String(req.body?.passwordInicial || "");
    let authGestionada = actual.authCertificadosGestionado === true;
    if (!cuenta.existe) {
      if (password.length < 8 || password.length > 128) throw Object.assign(new Error("La contraseña inicial debe tener entre 8 y 128 caracteres."), { statusCode: 400 });
      const creada = await crearFirebaseAuthValidador(email, password, nombreCompleto);
      cuenta = { existe: true, uid: creada.localId, email, disabled: false, displayName: nombreCompleto };
      authGestionada = true;
    } else if (cuenta.email && cuenta.email !== email) {
      throw Object.assign(new Error("El correo ya está asociado a otra cuenta de acceso."), { statusCode: 409 });
    } else if (cuenta.disabled) {
      if (!authGestionada) throw Object.assign(new Error("La cuenta Firebase existente está deshabilitada y no fue gestionada por este módulo."), { statusCode: 409 });
      await actualizarFirebaseAuthValidador(cuenta.uid, false);
      cuenta = { ...cuenta, disabled: false };
    }
    const cambios = { validarCertificados: true, authUid: cuenta.uid, authEmail: email, authCertificadosGestionado: authGestionada, ...(authGestionada ? { authCertificadosCreadoEn: actual.authCertificadosCreadoEn || new Date().toISOString(), authCertificadosCreadoPor: actual.authCertificadosCreadoPor || authUser.uid } : {}), validarCertificadosActualizadoEn: new Date().toISOString(), validarCertificadosActualizadoPor: authUser.uid };
    const actualizados = await Promise.all(documentos.map((doc) => updateFirestoreDoc(getFirestoreRelativePath(doc), cambios)));
    return res.json({ ok: true, usuario: mapValidadorAdmin(actualizados[0] || { ...actual, ...cambios }), acceso: { existe: true, habilitada: !cuenta.disabled, gestionadaPorModulo: authGestionada, email, tieneUidVinculado: true } });
  } catch (error: any) { return sendCertificadosError(res, error); }
});

app.get("/api/certificados/admin/validadores/:usuarioDocId/acceso", async (req, res) => {
  try {
    const authUser = await verifyFirebaseIdToken(req.headers.authorization); await requireAdministrador(authUser);
    const id = parseUsuarioDocId(req.params.usuarioDocId);
    const persona = await resolverPersonaPorDocId(id);
    if (!persona) throw Object.assign(new Error("Usuario no encontrado."), { statusCode: 404 });
    const documentos = await registrosPorPersona(normalizeDni(persona.dni));
    const gestionada = documentos.some((doc) => doc.authCertificadosGestionado === true);
    const uid = documentos.map((doc) => String(doc.authUid || "").trim()).find(Boolean);
    const emails = [...new Set(documentos.flatMap((doc) => [doc.email, doc.correo, doc.mail]).map((value) => String(value || "").trim().toLowerCase()).filter((value) => value.includes("@")))];
    const cuenta = uid ? await buscarFirebaseAuthPorUid(uid) : emails[0] ? await buscarFirebaseAuthPorEmail(emails[0]) : { existe: false };
    return res.json({ ok: true, acceso: { existe: cuenta.existe === true, habilitada: cuenta.existe === true && cuenta.disabled !== true, gestionadaPorModulo: gestionada, email: cuenta.email || emails[0] || "", tieneUidVinculado: Boolean(uid && cuenta.existe) } });
  } catch (error: any) { return sendCertificadosError(res, error); }
});

app.delete("/api/certificados/admin/validadores/:usuarioDocId", async (req, res) => {
  try {
    const authUser = await verifyFirebaseIdToken(req.headers.authorization); await requireAdministrador(authUser);
    const id = parseUsuarioDocId(req.params.usuarioDocId); const actual = await resolverPersonaPorDocId(id);
    if (!actual) throw Object.assign(new Error("Usuario no encontrado."), { statusCode: 404 });
    const dni = normalizeDni(actual.dni);
    const documentos = dni ? await registrosPorPersona(dni) : [actual];
    const cambios = { validarCertificados: false, validarCertificadosActualizadoEn: new Date().toISOString(), validarCertificadosActualizadoPor: authUser.uid };
    await Promise.all(documentos.map((doc) => updateFirestoreDoc(getFirestoreRelativePath(doc), cambios)));
    const gestionada = documentos.some((doc) => doc.authCertificadosGestionado === true);
    const uid = documentos.map((doc) => String(doc.authUid || "").trim()).find(Boolean);
    let accesoFirebaseActualizado = false;
    let advertencia = "";
    if (gestionada && uid) {
      try {
        await actualizarFirebaseAuthValidador(uid, true);
        accesoFirebaseActualizado = true;
      } catch (error: any) {
        advertencia = "El permiso fue retirado, pero no se pudo actualizar la cuenta de acceso.";
        console.error(`[sidca-chatbot-backend] quitar validador: fallo secundario Auth dni=${dni || "-"} codigo=${error?.identityCode || error?.statusCode || "desconocido"}`);
      }
    }
    console.info(`[sidca-chatbot-backend] quitar validador dni=${dni || "-"} registrosUsuarios=${documentos.filter((doc) => String(doc.path || "").startsWith("usuarios/")).length} registrosNuevoAfiliado=${documentos.filter((doc) => String(doc.path || "").startsWith("nuevoAfiliado/")).length} gestionadoAuth=${gestionada} authEncontrado=${Boolean(uid)}`);
    return res.json({ ok: true, usuarioDocId: id, validarCertificados: false, accesoFirebaseActualizado, ...(advertencia ? { advertencia } : {}) });
  } catch (error: any) { return sendCertificadosError(res, error); }
});

// Planillas documentales de inscriptos. Se guarda el archivo privado en
// Storage y únicamente sus metadatos en Firestore; nunca se publican URLs.
app.get("/api/certificados/admin/registro-inscriptos/:cursoId", async (req, res) => {
  try {
    const authUser = await verifyFirebaseIdToken(req.headers.authorization);
    await requireAdministrador(authUser);
    const cursoId = parseCursoIdParam(req.params.cursoId);
    const archivos = await queryFirestoreChildCollection(`certificados/${cursoId}`, "registroInscriptos", [{ field: "activo", value: true }], 100);
    return res.json({ ok: true, archivos: archivos.map((archivo) => ({ ...archivo, storagePath: undefined })) });
  } catch (error: any) { return sendCertificadosError(res, error); }
});

app.post("/api/certificados/admin/registro-inscriptos/:cursoId", registroInscriptosUpload.array("archivos", 10), async (req, res) => {
  const subidos: string[] = [];
  try {
    const authUser = await verifyFirebaseIdToken(req.headers.authorization);
    await requireAdministrador(authUser);
    const cursoId = parseCursoIdParam(req.params.cursoId);
    const curso = await getFirestoreDoc(`cursos/${cursoId}`);
    if (!curso) throw Object.assign(new Error("La capacitación no existe."), { statusCode: 404 });
    const archivos = (req.files as Express.Multer.File[]) || [];
    if (!archivos.length || archivos.length > 10) throw Object.assign(new Error("Seleccioná entre 1 y 10 planillas."), { statusCode: 400 });
    archivos.forEach(validarArchivoRegistro);
    const bucket = bucketRegistroInscriptos();
    const creados = [];
    for (const archivo of archivos) {
      const archivoId = crypto.randomUUID();
      const storagePath = `certificados/registro-inscriptos/${cursoId}/${archivoId}-${nombreArchivoRegistroSeguro(archivo.originalname)}`;
      await uploadGoogleStorageObject(bucket, storagePath, archivo.buffer, archivo.mimetype);
      subidos.push(storagePath);
      try {
        const metadata = await createFirestoreDoc(`certificados/${cursoId}/registroInscriptos`, archivoId, {
          archivoId, cursoId, tituloCurso: String(curso.titulo || "").trim(), nombreOriginal: String(archivo.originalname || "archivo.xlsx"), storagePath,
          mimeType: archivo.mimetype, size: archivo.size, activo: true,
          subidoEn: new Date(), subidoPorUid: authUser.uid, subidoPorEmail: authUser.email || "",
        });
        creados.push({ ...metadata, storagePath: undefined });
      } catch (error) {
        await deleteGoogleStorageObject(bucket, storagePath).catch(() => undefined);
        throw error;
      }
    }
    // La caché de metadata acaba de quedar desactualizada: se invalida ahora,
    // no en 45s, para que el próximo listado vea la planilla nueva.
    invalidarRegistroInscriptosCache();
    return res.status(201).json({ ok: true, archivos: creados });
  } catch (error: any) {
    if (subidos.length) {
      const bucket = (() => { try { return bucketRegistroInscriptos(); } catch { return null; } })();
      if (bucket) await Promise.all(subidos.map((path) => deleteGoogleStorageObject(bucket, path).catch(() => undefined)));
    }
    return sendCertificadosError(res, error);
  }
});

app.delete("/api/certificados/admin/registro-inscriptos/:cursoId/:archivoId", async (req, res) => {
  try {
    const authUser = await verifyFirebaseIdToken(req.headers.authorization);
    await requireAdministrador(authUser);
    const cursoId = parseCursoIdParam(req.params.cursoId);
    const archivoId = parseTrabajoPdfIdParam(req.params.archivoId);
    const path = `certificados/${cursoId}/registroInscriptos/${archivoId}`;
    const archivo = await getFirestoreDoc(path);
    if (!archivo || archivo.activo !== true) throw Object.assign(new Error("La planilla no existe."), { statusCode: 404 });
    await updateFirestoreDoc(path, { activo: false, eliminadoEn: new Date(), eliminadoPor: authUser.uid });
    if (archivo.storagePath) await deleteGoogleStorageObject(bucketRegistroInscriptos(), String(archivo.storagePath));
    invalidarRegistroInscriptosCache();
    return res.json({ ok: true, eliminado: true });
  } catch (error: any) { return sendCertificadosError(res, error); }
});

app.get("/api/certificados/registro-inscriptos", async (req, res) => {
  const inicio = Date.now();
  try {
    const marcaAuth = Date.now();
    const authUser = await verifyFirebaseIdToken(req.headers.authorization);
    const tiempoAuth = Date.now() - marcaAuth;

    const marcaPermiso = Date.now();
    await requireAdministradorOValidadorCertificados(authUser);
    const tiempoPermiso = Date.now() - marcaPermiso;

    // El botón "Actualizar" del validador pide ?forzar=true para saltear la
    // caché de 45s; cualquier otro valor usa la caché normalmente.
    const forzar = String(req.query.forzar || "") === "true";

    const marcaFirestore = Date.now();
    const cursos = await listarRegistroInscriptosActivos(forzar);
    const tiempoFirestore = Date.now() - marcaFirestore;

    console.log(`[perf] registro-inscriptos firebaseAuth=${tiempoAuth}ms resolverPermiso=${tiempoPermiso}ms firestore=${tiempoFirestore}ms total=${Date.now() - inicio}ms forzar=${forzar}`);
    return res.json({ ok: true, cursos });
  } catch (error: any) { return sendCertificadosError(res, error); }
});

app.get("/api/certificados/registro-inscriptos/:cursoId/:archivoId/descargar", async (req, res) => {
  const inicio = Date.now();
  try {
    const marcaAuth = Date.now();
    const authUser = await verifyFirebaseIdToken(req.headers.authorization);
    const tiempoAuth = Date.now() - marcaAuth;

    const marcaPermiso = Date.now();
    const permiso = await requireAdministradorOValidadorCertificados(authUser);
    const tiempoPermiso = Date.now() - marcaPermiso;

    const cursoId = parseCursoIdParam(req.params.cursoId);
    const archivoId = parseTrabajoPdfIdParam(req.params.archivoId);

    const marcaMetadata = Date.now();
    const archivo = await getFirestoreDoc(`certificados/${cursoId}/registroInscriptos/${archivoId}`);
    const tiempoMetadata = Date.now() - marcaMetadata;

    // Diagnóstico del 404: cada condición se comprueba y se registra por
    // separado, en vez de un "no está disponible" genérico que después obliga
    // a adivinar cuál de las cuatro falló.
    if (!archivo) {
      console.warn(`[registro-inscriptos-404] curso=${cursoId} archivo=${archivoId} motivo=documento_no_existe`);
      throw Object.assign(new Error("La planilla no está disponible."), { statusCode: 404 });
    }
    if (archivo.activo !== true) {
      console.warn(`[registro-inscriptos-404] curso=${cursoId} archivo=${archivoId} motivo=inactivo`);
      throw Object.assign(new Error("La planilla no está disponible."), { statusCode: 404 });
    }
    if (String(archivo.cursoId) !== cursoId) {
      console.warn(`[registro-inscriptos-404] curso=${cursoId} archivo=${archivoId} motivo=cursoId_no_coincide archivoCursoId=${String(archivo.cursoId)}`);
      throw Object.assign(new Error("La planilla no está disponible."), { statusCode: 404 });
    }
    if (!archivo.storagePath) {
      console.warn(`[registro-inscriptos-404] curso=${cursoId} archivo=${archivoId} motivo=sin_storagePath`);
      throw Object.assign(new Error("La planilla no está disponible."), { statusCode: 404 });
    }

    const bucket = bucketRegistroInscriptos();
    const storagePath = String(archivo.storagePath);

    // GET directo a la API de Storage: se necesita el Response crudo, con su
    // body como stream. La versión anterior esperaba el archivo COMPLETO en
    // un Buffer antes de mandar el primer byte al navegador; con planillas de
    // varios MB, eso era buena parte del retraso.
    const marcaStorage = Date.now();
    const objeto = await storageRegistroRequest("GET", bucket, storagePath);
    const tiempoStorage = Date.now() - marcaStorage;

    if (!objeto.ok || !objeto.body) {
      const detalle = await objeto.text().catch(() => "");
      console.warn(`[registro-inscriptos-404] curso=${cursoId} archivo=${archivoId} motivo=storage_${objeto.status} bucket=${bucket} storagePath=${storagePath} detalle=${detalle.slice(0, 200)}`);

      if (objeto.status === 404) {
        throw Object.assign(new Error("La planilla no está disponible."), { statusCode: 404 });
      }
      if (objeto.status === 401 || objeto.status === 403) {
        throw Object.assign(new Error("El servidor no tiene permiso para leer la planilla."), { statusCode: 500 });
      }
      throw Object.assign(new Error("No se pudo leer la planilla."), { statusCode: 502 });
    }

    // La auditoría se inicia en paralelo con el stream: no retrasa el primer
    // byte, pero se espera y registra su resultado antes de cerrar el handler.
    const auditoriaPromise = addFirestoreDoc(`certificados/${cursoId}/registroInscriptos/${archivoId}/descargas`, {
      descargadoEn: new Date(), descargadoPorUid: authUser.uid, descargadoPorEmail: authUser.email || "", tipoValidador: permiso.tipo,
    });

    res.setHeader("Content-Type", String(archivo.mimeType || "application/octet-stream"));
    res.setHeader("Content-Disposition", `attachment; filename="${nombreArchivoRegistroSeguro(String(archivo.nombreOriginal || "planilla.xlsx"))}"`);
    res.setHeader("Cache-Control", "private, no-store");
    const largo = objeto.headers.get("content-length");
    if (largo) res.setHeader("Content-Length", largo);

    const preparacion = Date.now() - inicio;
    console.log(`[perf] descargar-planilla-preparacion curso=${cursoId} archivo=${archivoId} auth=${tiempoAuth}ms permiso=${tiempoPermiso}ms metadata=${tiempoMetadata}ms storageHeaders=${tiempoStorage}ms preparacion=${preparacion}ms`);

    // Las dos tareas arrancan al mismo tiempo: la auditoría no demora el
    // primer byte y pipeline conserva backpressure y reporta el cierre real.
    const inicioTransferencia = Date.now();
    const transferenciaPromise = pipeline(Readable.fromWeb(objeto.body as any), res);
    const [resultadoAuditoria, resultadoTransferencia] = await Promise.allSettled([
      auditoriaPromise,
      transferenciaPromise,
    ]);

    if (resultadoAuditoria.status === "rejected") {
      console.error(`[registro-inscriptos] no se pudo registrar la auditoría de descarga curso=${cursoId} archivo=${archivoId}`, resultadoAuditoria.reason);
    }

    if (resultadoTransferencia.status === "rejected") {
      console.error("[registro-inscriptos] error al transmitir la planilla", resultadoTransferencia.reason);
      if (!res.headersSent && !res.destroyed) {
        res.status(502).json({ ok: false, error: "No se pudo leer la planilla." });
      } else if (!res.writableEnded && !res.destroyed) {
        res.destroy(resultadoTransferencia.reason as Error);
      }
      return;
    }

    console.log(`[perf] descargar-planilla-fin curso=${cursoId} archivo=${archivoId} transferencia=${Date.now() - inicioTransferencia}ms totalRequest=${Date.now() - inicio}ms`);
  } catch (error: any) { return sendCertificadosError(res, error); }
});

app.get("/api/certificados/admin/configuracion/:cursoId", async (req, res) => {
  try {
    const authUser = await verifyFirebaseIdToken(req.headers.authorization);
    await requireAdministrador(authUser);

    const cursoId = parseCursoIdParam(req.params.cursoId);
    const record = await getFirestoreDoc(`certificados/${cursoId}`);

    if (!record) {
      throw Object.assign(
        new Error("Todavía no hay una configuración de certificado para este curso."),
        { statusCode: 404 }
      );
    }

    return res.status(200).json({
      ok: true,
      modulo: "certificados",
      configuracion: mapConfiguracionCertificado(record, cursoId),
    });
  } catch (error: any) {
    return sendCertificadosError(res, error);
  }
});

app.put("/api/certificados/admin/configuracion/:cursoId", async (req, res) => {
  try {
    const authUser = await verifyFirebaseIdToken(req.headers.authorization);
    await requireAdministrador(authUser);

    const cursoId = parseCursoIdParam(req.params.cursoId);

    // El curso debe existir: certificados/{cursoId} sólo tiene sentido si
    // cursos/{cursoId} existe realmente.
    const curso = await getFirestoreDoc(`cursos/${cursoId}`);

    if (!curso) {
      throw Object.assign(new Error("El curso indicado no existe."), {
        statusCode: 404,
      });
    }

    let datosValidados: z.infer<typeof configuracionCertificadoSchema>;

    try {
      datosValidados = configuracionCertificadoSchema.parse(req.body);
    } catch (error: any) {
      if (error instanceof z.ZodError) {
        const detalle = error.issues[0];
        throw Object.assign(
          new Error(
            detalle?.message
              ? `${detalle.path.join(".") || "body"}: ${detalle.message}`
              : "Los datos enviados no son válidos."
          ),
          { statusCode: 400 }
        );
      }
      throw error;
    }

    const existente = await getFirestoreDoc(`certificados/${cursoId}`);
    const ahora = new Date();

    // El estado no se promueve automáticamente: si ya estaba en "lista" se
    // respeta, y cualquier configuración nueva nace como "borrador".
    const estadoPrevio = String(existente?.estadoConfiguracion || "");
    const estadoConfiguracion = ESTADOS_CONFIGURACION_CERTIFICADO.has(estadoPrevio)
      ? estadoPrevio
      : "borrador";

    const datos: Record<string, any> = {
      cursoId,
      // cursoTitulo proviene siempre del documento real de cursos, nunca del
      // cliente. El campo "titulo" del certificado sí es libre y editable.
      cursoTitulo: String(curso.titulo || ""),

      titulo: datosValidados.titulo,
      resolucion: datosValidados.resolucion,
      cargaHoraria: datosValidados.cargaHoraria,
      dias: datosValidados.dias,
      fecha: datosValidados.fecha,
      modalidad: datosValidados.modalidad,

      institucionCertificado: datosValidados.institucionCertificado,

      // Sustituye a "firmas". El campo legacy puede seguir existiendo
      // físicamente en documentos anteriores: updateFirestoreDoc usa
      // updateMask, así que no se toca lo que no se nombra. No se borra
      // automáticamente — es historial, y quitarlo no aporta nada.
      autoridades: normalizarAutoridadesCertificado(
        datosValidados.autoridades || []
      ),

      estadoConfiguracion,

      actualizadoEn: ahora,
      actualizadoPor: authUser.uid,
    };

    // creadoEn / creadoPor se escriben una sola vez. Al no incluirlos en el
    // updateMask cuando ya existen, quedan intactos y además se evita
    // reescribir el timestamp como string al releerlo.
    if (!existente?.creadoEn) datos.creadoEn = ahora;
    if (!existente?.creadoPor) datos.creadoPor = authUser.uid;

    // updateFirestoreDoc usa updateMask: los campos no listados (por ejemplo
    // el histórico "imagen") no se borran ni se sobrescriben.
    const guardado = await updateFirestoreDoc(`certificados/${cursoId}`, datos);

    return res.status(200).json({
      ok: true,
      modulo: "certificados",
      creado: !existente,
      certificadoId: cursoId,
      configuracion: mapConfiguracionCertificado(guardado, cursoId),
    });
  } catch (error: any) {
    return sendCertificadosError(res, error);
  }
});

/**
 * Elimina la configuración de certificado de un curso.
 *
 * Borra EXCLUSIVAMENTE el documento certificados/{cursoId}. No toca —ni
 * podría, porque es un borrado de documento único sin cascada—:
 *
 *   cursos/{cursoId}                     el curso académico sigue existiendo
 *   usuarios/{usuarioDocId}              los usuarios siguen existiendo
 *   usuarios/{usuarioDocId}/cursos/*     las aprobaciones siguen existiendo
 *   nuevoAfiliado                        intacta
 *
 * Efecto: el curso desaparece de Emitir (que lista sólo documentos de
 * certificados) y vuelve a aparecer en Configurar como curso sin configurar,
 * porque Configurar se alimenta de la colección cursos.
 *
 * Nota para más adelante: cuando exista la subcolección emitidos, borrar la
 * configuración dejaría certificados emitidos huérfanos. En esa etapa habrá
 * que bloquear el borrado si hay emisiones.
 */
app.delete("/api/certificados/admin/configuracion/:cursoId", async (req, res) => {
  try {
    const authUser = await verifyFirebaseIdToken(req.headers.authorization);
    await requireAdministrador(authUser);

    const cursoId = parseCursoIdParam(req.params.cursoId);

    const existente = await getFirestoreDoc(`certificados/${cursoId}`);

    if (!existente) {
      throw Object.assign(
        new Error("Este curso no tiene configuración de certificado."),
        { statusCode: 404 }
      );
    }

    // Firestore NO borra subcolecciones en cascada: si se elimina la
    // configuración con emisiones vivas, certificados/{cursoId}/emitidos/*
    // quedaría huérfano — documentos inalcanzables desde el módulo, con sus
    // tokens de validación todavía activos. Se bloquea sin importar el
    // estado: un anulado o reemplazado también es historial que no se tira.
    const emitidos = await queryFirestoreChildCollection(
      `certificados/${cursoId}`,
      "emitidos",
      [],
      1
    );

    if (emitidos.length > 0) {
      throw Object.assign(
        new Error(
          "No se puede eliminar la configuración porque ya existen certificados emitidos para este curso."
        ),
        { statusCode: 409 }
      );
    }

    await deleteFirestoreDoc(`certificados/${cursoId}`);

    console.log(
      `[sidca-chatbot-backend] configuracion de certificado eliminada curso=${cursoId} por=${authUser.uid}`
    );

    return res.status(200).json({
      ok: true,
      modulo: "certificados",
      eliminado: true,
      certificadoId: cursoId,
      cursoId,
    });
  } catch (error: any) {
    return sendCertificadosError(res, error);
  }
});

// ============================================================
// LISTADO DE CONFIGURACIONES DE CERTIFICADO
//
// Fuente de la pestaña EMITIR. A diferencia de CONFIGURAR — que lista la
// colección "cursos" para poder elegir cuál configurar — acá sólo interesan
// los cursos que YA tienen configuración creada por este módulo.
//
// La colección "certificados" es anterior al módulo y contiene documentos
// históricos con ID automático que no siguen este modelo. Se los distingue
// por el campo cursoId, que sólo escribe el PUT de configuración: los
// históricos no lo tienen. Se filtran en memoria porque Firestore no ofrece
// un operador "existe campo"; la alternativa (cursoId != "") obligaría a
// ordenar por ese campo y no aporta nada con este volumen.
//
// Los documentos históricos no se leen para modificarlos: se ignoran.
// ============================================================

const CONFIGURACIONES_MAX_RESULTADOS = 2000;

app.get("/api/certificados/admin/configuraciones", async (req, res) => {
  try {
    const authUser = await verifyFirebaseIdToken(req.headers.authorization);
    await requireAdministrador(authUser);

    const documentos = await queryFirestoreCollection(
      "certificados",
      [],
      CONFIGURACIONES_MAX_RESULTADOS
    );

    const configuraciones = documentos
      // Sólo el modelo nuevo: los históricos no tienen cursoId.
      .filter((record) => Boolean(String(record.cursoId || "").trim()))
      // Cursos apartados de la emisión. El documento sigue existiendo con
      // toda su configuración: sólo deja de ofrecerse para emitir.
      .filter((record) => record.ocultarEnEmitir !== true)
      .map((record) => ({
        certificadoId: record.id || record.cursoId,
        cursoId: record.cursoId,
        cursoTitulo: record.cursoTitulo || "",
        titulo: record.titulo || "",
        resolucion: record.resolucion || "",
        modalidad: record.modalidad || "",
        estadoConfiguracion: record.estadoConfiguracion || "borrador",
        actualizadoEn: record.actualizadoEn || null,
      }))
      .sort((a, b) =>
        String(a.cursoTitulo || a.titulo).localeCompare(
          String(b.cursoTitulo || b.titulo),
          "es",
          { sensitivity: "base" }
        )
      );

    return res.status(200).json({
      ok: true,
      modulo: "certificados",
      total: configuraciones.length,
      revisados: documentos.length,
      configuraciones,
    });
  } catch (error: any) {
    return sendCertificadosError(res, error);
  }
});

// ============================================================
// APROBADOS DE UN CURSO
//
// La aprobación NO vive en el módulo de certificados: la crea el importador
// de Excel existente (uploadUserCursosInfo) en
//
//   usuarios/{usuarioDocId}/cursos/{aprobacionId}
//
// con { aprobo: true, curso: "cursos/{cursoId}", ... }. El documento de
// aprobación no guarda DNI: la identidad sale del documento padre.
//
// Por eso hay que consultar el collection group "cursos" (las subcolecciones
// de usuarios), que es distinto de la colección raíz "cursos" (el catálogo de
// capacitaciones). Ambos se llaman igual.
//
// El importador usa addDoc sin verificar existencia previa, así que un mismo
// usuario puede tener varias aprobaciones del mismo curso. Se deduplica por
// usuarioDocId, que es la identidad real.
// ============================================================

const APROBADOS_MAX_RESULTADOS = 10_000;
const APROBADOS_CACHE_TTL_MS = 45_000;
const USUARIOS_BATCH_GET_CHUNK_SIZE = 200;

type MetricasUsuariosAprobados = {
  usuariosCacheHit: number;
  usuariosCacheMiss: number;
};

type UsuarioAcademicoCacheEntry = {
  usuario: FirestoreRecord | null;
  expiraEn: number;
};

const usuariosAcademicosCache = new Map<
  string,
  UsuarioAcademicoCacheEntry
>();
const usuariosAcademicosEnCurso = new Map<
  string,
  Promise<FirestoreRecord | null>
>();

const aprobadosCursoCache = new Map<
  string,
  { datos: Awaited<ReturnType<typeof resolverAprobadosCursoSinCache>>; expiraEn: number }
>();
const aprobadosCursoEnCurso = new Map<
  string,
  Promise<Awaited<ReturnType<typeof resolverAprobadosCursoSinCache>>>
>();
const aprobadosCursoVersion = new Map<string, number>();

/**
 * Extrae el ID del documento de usuario desde el name completo de una
 * aprobación:
 *
 *   projects/.../documents/usuarios/{usuarioDocId}/cursos/{aprobacionId}
 *
 * Devuelve null si la ruta no tiene esa forma, para poder clasificar el
 * registro como anómalo en vez de romper toda la respuesta.
 */
function extraerUsuarioDocIdDeAprobacion(name: unknown): string | null {
  const ruta = String(name ?? "");
  const marcador = "/documents/";
  const indice = ruta.indexOf(marcador);

  if (indice === -1) return null;

  const segmentos = ruta.slice(indice + marcador.length).split("/");

  if (segmentos.length !== 4) return null;
  if (segmentos[0] !== "usuarios" || segmentos[2] !== "cursos") return null;
  if (!segmentos[1]) return null;

  return segmentos[1];
}

/** Enmascara un DNI para logs de diagnóstico. */
function enmascararDni(dni: string): string {
  if (dni.length <= 4) return "***";
  return `${dni.slice(0, 2)}***${dni.slice(-2)}`;
}

type FirestoreBatchGetResponse = {
  found?: FirestoreDocument;
  missing?: string;
};

/** Lee usuarios académicos con documents:batchGet, sin modificar Firestore. */
async function batchGetUsuarios(
  usuarioDocIds: string[]
): Promise<Map<string, FirestoreRecord | null>> {
  const documentos = usuarioDocIds.map(
    (usuarioDocId) =>
      `projects/${firebaseProjectId}/databases/(default)/documents/usuarios/${usuarioDocId}`
  );
  const respuesta = await firestoreRequest<FirestoreBatchGetResponse[]>(
    `${firestoreBaseUrl}:batchGet`,
    {
      method: "POST",
      body: JSON.stringify({ documents: documentos }),
    }
  );

  if (!Array.isArray(respuesta)) {
    throw new Error("Firestore batchGet devolvió una respuesta inválida.");
  }

  const solicitados = new Set(usuarioDocIds);
  const encontrados = new Map<string, FirestoreRecord | null>();

  for (const elemento of respuesta) {
    if (elemento?.found?.name) {
      const usuarioDocId = getDocumentoId(elemento.found.name);
      if (!solicitados.has(usuarioDocId)) {
        throw new Error("Firestore batchGet devolvió un documento no solicitado.");
      }
      if (encontrados.has(usuarioDocId)) {
        throw new Error("Firestore batchGet devolvió un documento duplicado.");
      }
      encontrados.set(usuarioDocId, firestoreDocToJs(elemento.found));
      continue;
    }

    if (typeof elemento?.missing === "string" && elemento.missing) {
      const usuarioDocId = getDocumentoId(elemento.missing);
      if (!solicitados.has(usuarioDocId)) {
        throw new Error("Firestore batchGet devolvió un missing no solicitado.");
      }
      if (encontrados.has(usuarioDocId)) {
        throw new Error("Firestore batchGet devolvió un documento duplicado.");
      }
      encontrados.set(usuarioDocId, null);
      continue;
    }

    throw new Error("Firestore batchGet devolvió una entrada inválida.");
  }

  // Sólo una respuesta missing explícita produce null; una respuesta incompleta falla.
  if (usuarioDocIds.some((usuarioDocId) => !encontrados.has(usuarioDocId))) {
    throw new Error("Firestore batchGet devolvió una respuesta incompleta.");
  }

  return encontrados;
}

/** Resuelve usuarios con caché negativa y single-flight por documento. */
async function resolverUsuariosPorIds(
  usuarioDocIds: string[],
  metricas?: MetricasUsuariosAprobados
): Promise<Map<string, FirestoreRecord | null>> {
  const ids = [
    ...new Set(
      usuarioDocIds.map((valor) => String(valor || "").trim()).filter(Boolean)
    ),
  ];
  const resultado = new Map<string, FirestoreRecord | null>();
  const solicitudes = new Map<string, Promise<FirestoreRecord | null>>();
  const pendientes: string[] = [];
  const ahora = Date.now();

  for (const usuarioDocId of ids) {
    const cacheado = usuariosAcademicosCache.get(usuarioDocId);
    if (cacheado && cacheado.expiraEn > ahora) {
      resultado.set(usuarioDocId, cacheado.usuario);
      if (metricas) metricas.usuariosCacheHit += 1;
      continue;
    }
    if (cacheado) usuariosAcademicosCache.delete(usuarioDocId);

    if (metricas) metricas.usuariosCacheMiss += 1;
    const enCurso = usuariosAcademicosEnCurso.get(usuarioDocId);
    if (enCurso) {
      solicitudes.set(usuarioDocId, enCurso);
    } else {
      pendientes.push(usuarioDocId);
    }
  }

  for (
    let inicio = 0;
    inicio < pendientes.length;
    inicio += USUARIOS_BATCH_GET_CHUNK_SIZE
  ) {
    const lote = pendientes.slice(
      inicio,
      inicio + USUARIOS_BATCH_GET_CHUNK_SIZE
    );
    const batch = batchGetUsuarios(lote);

    for (const usuarioDocId of lote) {
      const solicitud = batch
        .then((usuarios) => {
          const usuario = usuarios.get(usuarioDocId) || null;
          usuariosAcademicosCache.set(usuarioDocId, {
            usuario,
            expiraEn: Date.now() + APROBADOS_CACHE_TTL_MS,
          });
          return usuario;
        })
        .finally(() => {
          if (usuariosAcademicosEnCurso.get(usuarioDocId) === solicitud) {
            usuariosAcademicosEnCurso.delete(usuarioDocId);
          }
        });

      usuariosAcademicosEnCurso.set(usuarioDocId, solicitud);
      solicitudes.set(usuarioDocId, solicitud);
    }
  }

  const resueltos = await Promise.all(
    ids.map(async (usuarioDocId) => {
      const usuario = resultado.has(usuarioDocId)
        ? resultado.get(usuarioDocId) || null
        : await solicitudes.get(usuarioDocId);
      return [usuarioDocId, usuario || null] as const;
    })
  );

  return new Map(resueltos);
}

/**
 * Participante aprobado de un curso, con la calidad de sus datos resuelta.
 *
 * `estado` describe la calidad del dato ACADÉMICO, no la disponibilidad:
 *   aprobado          → tiene DNI y nombre, se le puede emitir
 *   datos_incompletos → falta DNI o nombre
 *   sin_usuario       → su documento de usuarios ya no existe
 *
 * `apartado` es una condición ADMINISTRATIVA independiente: quien fue quitado
 * de la emisión conserva su estado académico tal cual.
 */
type ParticipanteAprobado = {
  usuarioDocId: string;
  /** Todos los documentos de usuarios consolidados en esta persona. */
  usuarioDocIds: string[];
  dni: string;
  apellidoNombre: string;
  apellido?: string;
  nombre?: string;
  estado: "aprobado" | "datos_incompletos" | "sin_usuario";
  aprobaciones: number;
  certificadoEmitido: boolean;
  certificadoQrValido: boolean;
  apartado?: boolean;
  /** Condición sindical. La resuelve resolverPadronPorDni; ver afiliacion.ts. */
  afiliacion?: Afiliacion;
  /** Departamento vigente y segmento al que corresponde su descarga. */
  departamento?: Departamento;
};

/** Proyecta un usuario resuelto a la forma común del participante. */
function construirParticipanteAprobado(
  usuarioDocId: string,
  usuario: FirestoreRecord | null,
  aprobaciones: number,
  apartado = false,
  certificadoEmitido = false,
  certificadoQrValido = false
): ParticipanteAprobado {
  const comun = {
    usuarioDocId,
    usuarioDocIds: [usuarioDocId],
    aprobaciones,
    certificadoEmitido,
    certificadoQrValido,
    ...(apartado ? { apartado: true } : {}),
  };

  if (!usuario) {
    return {
      ...comun,
      dni: "",
      apellidoNombre: "",
      estado: "sin_usuario",
    };
  }

  const dni = normalizeDni(usuario.dni);
  const apellidoNombre = buildNombreAfiliado(usuario);

  // buildNombreAfiliado nunca devuelve vacío: si no hay datos cae en su texto
  // por defecto, así que se comprueban los campos de origen.
  const tieneNombre = Boolean(
    String(usuario.apellidoNombre || usuario.apellido_y_nombre || "").trim() ||
      String(usuario.apellido || "").trim() ||
      String(usuario.nombre || "").trim()
  );

  // apellido / nombre sólo si existen de verdad: no se parte artificialmente
  // un apellidoNombre.
  const apellido = String(usuario.apellido || "").trim();
  const nombre = String(usuario.nombre || "").trim();

  return {
    ...comun,
    dni,
    apellidoNombre,
    ...(apellido ? { apellido } : {}),
    ...(nombre ? { nombre } : {}),
    estado: Boolean(dni) && tieneNombre ? "aprobado" : "datos_incompletos",
  };
}

/**
 * El DNI es la identidad de emisión. Dos documentos de `usuarios` con el
 * mismo DNI representan una sola persona para este curso, aunque sus IDs
 * sean distintos. Los registros sin DNI conservan como clave su usuarioDocId
 * para no fusionar personas desconocidas entre sí.
 */
function consolidarParticipantesPorDni(
  participantes: ParticipanteAprobado[]
): ParticipanteAprobado[] {
  const grupos = new Map<string, ParticipanteAprobado[]>();

  for (const participante of participantes) {
    const dni = normalizeDni(participante.dni);
    const usuarioDocId = String(participante.usuarioDocId || "").trim();
    const clave = dni ? `dni:${dni}` : `usuario:${usuarioDocId}`;
    const grupo = grupos.get(clave) || [];
    grupo.push(participante);
    grupos.set(clave, grupo);
  }

  return [...grupos.values()].map((grupo) => {
    // La selección es estable: primero certificado/QR, luego cantidad de
    // datos reales y finalmente el orden original de Firestore.
    const canonical = [...grupo].sort((a, b) => {
      const score = (valor: ParticipanteAprobado) =>
        (valor.certificadoEmitido ? 1_000_000 : 0) +
        (valor.certificadoQrValido ? 100_000 : 0) +
        (valor.afiliacion && valor.afiliacion.tipo !== "no_verificada"
          ? 1_000
          : 0) +
        (valor.departamento?.canonico ? 10 : 0) +
        (valor.dni ? 10 : 0) +
        (valor.apellido ? 100 : 0) +
        (valor.nombre ? 100 : 0) +
        (valor.apellidoNombre ? String(valor.apellidoNombre).length : 0);
      return score(b) - score(a);
    })[0];

    const textoMasCompleto = (...valores: unknown[]) =>
      valores
        .map((valor) => String(valor || "").trim())
        .filter(Boolean)
        .sort((a, b) => b.length - a.length)[0] || "";

    const usuarioDocIds = [
      ...new Set(
        grupo.flatMap((valor) =>
          Array.isArray(valor.usuarioDocIds) && valor.usuarioDocIds.length
            ? valor.usuarioDocIds
            : [valor.usuarioDocId]
        )
      ),
    ];

    const fusionado: ParticipanteAprobado = {
      ...canonical,
      usuarioDocId: canonical.usuarioDocId,
      usuarioDocIds,
      dni: normalizeDni(textoMasCompleto(...grupo.map((valor) => valor.dni))),
      apellidoNombre: textoMasCompleto(
        ...grupo.map((valor) => valor.apellidoNombre)
      ),
      ...(textoMasCompleto(...grupo.map((valor) => valor.apellido))
        ? { apellido: textoMasCompleto(...grupo.map((valor) => valor.apellido)) }
        : {}),
      ...(textoMasCompleto(...grupo.map((valor) => valor.nombre))
        ? { nombre: textoMasCompleto(...grupo.map((valor) => valor.nombre)) }
        : {}),
      estado: grupo.some((valor) => valor.estado === "aprobado")
        ? "aprobado"
        : grupo.some((valor) => valor.estado === "datos_incompletos")
        ? "datos_incompletos"
        : "sin_usuario",
      // Una persona tiene una aprobación académica para este curso, aunque
      // haya más de un documento de usuario asociado al mismo DNI.
      aprobaciones: 1,
      certificadoEmitido: grupo.some((valor) => valor.certificadoEmitido),
      certificadoQrValido: grupo.some((valor) => valor.certificadoQrValido),
      ...(grupo.some((valor) => valor.apartado)
        ? { apartado: true }
        : {}),
    };

    return fusionado;
  });
}

/**
 * Resuelve los documentos de usuario de un conjunto de aprobados y devuelve la
 * lista ordenada por apellido y nombre.
 *
 * Se usa igual para los disponibles y para los apartados: la única diferencia
 * es la marca `apartado`.
 */
async function resolverParticipantesAprobados(
  porUsuario: Map<string, number>,
  apartado = false,
  usuariosConCertificadoVigente = new Set<string>(),
  usuariosConCertificadoQrValido = usuariosConCertificadoVigente,
  metricas?: MetricasUsuariosAprobados & { padron: number; usuarios: number }
): Promise<ParticipanteAprobado[]> {
  const inicioUsuarios = Date.now();
  const usuarios = await resolverUsuariosPorIds([...porUsuario.keys()], metricas);
  if (metricas) metricas.usuarios += Date.now() - inicioUsuarios;

  const participantes = consolidarParticipantesPorDni(
    [...porUsuario.keys()].map((usuarioDocId) =>
      construirParticipanteAprobado(
        usuarioDocId,
        usuarios.get(usuarioDocId) || null,
        porUsuario.get(usuarioDocId) || 1,
        apartado,
        usuariosConCertificadoVigente.has(usuarioDocId),
        usuariosConCertificadoQrValido.has(usuarioDocId)
      )
    )
  );

  // Condición sindical. Se resuelve en UNA sola pasada para todos los DNI de
  // la lista: cada DNI distinto se consulta una vez, con concurrencia acotada.
  // No modifica el estado académico ni las exclusiones; sólo agrega el dato
  // que decide la habilitación para emitir y descargar.
  const inicioPadron = Date.now();
  const padron = await resolverPadronPorDni(
    participantes.map((participante) => participante.dni),
    { proyecto: firebaseProjectId, accessToken: await getGoogleAccessToken() }
  );
  if (metricas) metricas.padron += Date.now() - inicioPadron;

  return participantes
    .map((participante) => {
      const resuelto = padron.get(normalizeDni(participante.dni));

      return {
        ...participante,
        afiliacion: resuelto?.afiliacion || { ...AFILIACION_NO_VERIFICADA },
        // Departamento VIGENTE del afiliado, no el del snapshot del
        // certificado: es el que decide en qué segmento se descarga.
        departamento: resuelto?.departamento || {
          crudo: "",
          canonico: "",
          segmentoId: SEGMENTO_SIN_DEPARTAMENTO,
        },
      };
    })
    .sort((a, b) =>
      a.apellidoNombre.localeCompare(b.apellidoNombre, "es", {
        sensitivity: "base",
      })
    );
}

/**
 * Resuelve el padrón completo de aprobados de un curso.
 *
 * Es el cuerpo del GET /aprobados extraído tal cual, sin req ni res, para que
 * la emisión masiva pueda reconstruir la misma lista desde el backend en vez
 * de confiar en la que tiene abierta el navegador. Una sola implementación:
 * si mañana cambia el criterio de "aprobado", cambia para las dos.
 */
async function resolverAprobadosCursoSinCache(cursoId: string) {
  const inicioResolver = Date.now();
  const metricas: MetricasUsuariosAprobados & {
    padron: number;
    usuarios: number;
  } = {
    usuariosCacheHit: 0,
    usuariosCacheMiss: 0,
    padron: 0,
    usuarios: 0,
  };
  const inicioBase = Date.now();
  {
    // Estas cuatro lecturas son independientes entre sí: ninguna necesita el
    // resultado de otra para saber QUÉ pedir, todas conocen cursoId de
    // entrada. Antes se esperaban una por una; en Promise.all lo que tardan es
    // el máximo de las cuatro, no la suma. Sólo el chequeo de "el curso no
    // existe" tiene que esperar a que las cuatro terminen, porque hasta
    // entonces no se sabe si hubo que abortar — en el caso normal (curso
    // existente) no se pierde nada.
    const [curso, certificado, emitidosVigentes, aprobaciones] = await Promise.all([
      getFirestoreDoc(`cursos/${cursoId}`),
      getFirestoreDoc(`certificados/${cursoId}`),
      queryFirestoreChildCollection(
        `certificados/${cursoId}`,
        "emitidos",
        [{ field: "estado", value: EMITIDOS_ESTADO_VIGENTE }],
        APROBADOS_MAX_RESULTADOS
      ),
      // Un solo filtro de igualdad: lo resuelve el índice de campo único que
      // Firestore mantiene automáticamente. "aprobo" se evalúa en memoria para
      // no exigir un índice compuesto de ámbito COLLECTION_GROUP y, de paso,
      // poder contar los registros que no están aprobados.
      queryFirestoreCollectionGroup(
        "cursos",
        [{ field: "curso", value: `cursos/${cursoId}` }],
        APROBADOS_MAX_RESULTADOS
      ),
    ]);
    const tiempoBase = Date.now() - inicioBase;

    if (!curso) {
      throw Object.assign(new Error("El curso indicado no existe."), {
        statusCode: 404,
      });
    }

    // Exclusiones administrativas de la emisión. Viven en el documento del
    // certificado: la aprobación original en usuarios/{id}/cursos NUNCA se
    // toca, sólo se omite de esta respuesta.
    const usuariosExcluidos = new Set(
      (Array.isArray(certificado?.usuariosExcluidos)
        ? certificado.usuariosExcluidos
        : []
      ).map((valor: any) => String(valor || "").trim())
    );

    const usuariosConCertificadoVigente = new Set<string>();
    const usuariosConCertificadoQrValido = new Set<string>();
    emitidosVigentes.forEach((emitido) => {
      const usuarioDocId = String(emitido.usuarioDocId || "").trim();
      if (!usuarioDocId) return;
      usuariosConCertificadoVigente.add(usuarioDocId);
      if (emisionTieneQrValido(emitido)) {
        usuariosConCertificadoQrValido.add(usuarioDocId);
      }
    });

    const documentosAprobacion = aprobaciones.length;
    const truncado = documentosAprobacion >= APROBADOS_MAX_RESULTADOS;

    let rutasInesperadas = 0;
    let noAprobados = 0;
    let aprobacionesAprobadas = 0;

    // usuarioDocId -> cantidad de documentos de aprobación de ese usuario.
    // Dos mapas: quienes siguen disponibles para emitir y quienes fueron
    // apartados. Los apartados también se resuelven, para poder mostrarlos y
    // recuperarlos: contarlos no alcanzaba, porque sin nombre ni DNI la UI no
    // tenía nada que ofrecer.
    const porUsuario = new Map<string, number>();
    const porUsuarioExcluido = new Map<string, number>();

    for (const aprobacion of aprobaciones) {
      const usuarioDocId = extraerUsuarioDocIdDeAprobacion(aprobacion._name);

      if (!usuarioDocId) {
        rutasInesperadas += 1;
        continue;
      }

      if (aprobacion.aprobo !== true) {
        noAprobados += 1;
        continue;
      }

      aprobacionesAprobadas += 1;

      const apartado = usuariosExcluidos.has(usuarioDocId);
      const mapa = apartado ? porUsuarioExcluido : porUsuario;

      const previos = mapa.get(usuarioDocId);

      if (previos === undefined) {
        mapa.set(usuarioDocId, 1);
      } else {
        // Los duplicados del importador se cuentan en ambos grupos: son
        // documentos de aprobación repetidos, no una cuestión de
        // disponibilidad.
        mapa.set(usuarioDocId, previos + 1);
      }
    }

    const [participantesDisponibles, participantesApartados] = await Promise.all([
      resolverParticipantesAprobados(
        porUsuario,
        false,
        usuariosConCertificadoVigente,
        usuariosConCertificadoQrValido,
        metricas
      ),
      resolverParticipantesAprobados(
        porUsuarioExcluido,
        true,
        usuariosConCertificadoVigente,
        usuariosConCertificadoQrValido,
        metricas
      ),
    ]);

    const participantesConsolidados = consolidarParticipantesPorDni([
      ...participantesDisponibles,
      ...participantesApartados,
    ]);
    const participantes = participantesConsolidados.filter(
      (participante) => participante.apartado !== true
    );
    const participantesExcluidos = participantesConsolidados.filter(
      (participante) => participante.apartado === true
    );
    const duplicados = Math.max(
      0,
      aprobacionesAprobadas - participantesConsolidados.length
    );
    const duplicadosDetalle = participantesConsolidados
      .filter((participante) => participante.usuarioDocIds.length > 1)
      .map((participante) => ({
        dni: participante.dni,
        usuarioDocIds: participante.usuarioDocIds,
        cantidad: participante.usuarioDocIds.length,
      }));

    // Los contadores de calidad de datos se refieren a los DISPONIBLES, que es
    // lo que muestran los indicadores de la pantalla. Así se mantienen las dos
    // invariantes:
    //   identificados + datosIncompletos + sinUsuario = participantes.length
    //   participantes.length + participantesExcluidos.length = aprobados
    const contarEstado = (
      lista: ParticipanteAprobado[],
      estado: ParticipanteAprobado["estado"]
    ) => lista.filter((participante) => participante.estado === estado).length;

    const identificados = contarEstado(participantes, "aprobado");
    const datosIncompletos = contarEstado(participantes, "datos_incompletos");
    const sinUsuario = contarEstado(participantes, "sin_usuario");

    const excluidos = participantesExcluidos.length;

    console.log(
      `[sidca-chatbot-backend] aprobados curso=${cursoId} documentos=${documentosAprobacion} disponibles=${participantes.length} apartados=${excluidos} duplicados=${duplicados}`
    );
    console.log(
      `[perf] aprobados-resolver curso=${cursoId} base=${tiempoBase}ms usuarios=${metricas.usuarios}ms usuariosCacheHit=${metricas.usuariosCacheHit} usuariosCacheMiss=${metricas.usuariosCacheMiss} padron=${metricas.padron}ms total=${Date.now() - inicioResolver}ms`
    );

    return {
      curso: {
        id: cursoId,
        titulo: curso.titulo || "",
        estado: curso.estado || "",
        categoria: curso.categoria || "",
      },
      resumen: {
        documentosAprobacion,
        // Total de personas aprobadas del curso, INCLUYENDO las apartadas de
        // la emisión. Es el dato académico y no baja al apartar a alguien:
        // apartar no borra ninguna aprobación.
        aprobados: participantes.length + excluidos,
        // Personas que siguen disponibles para emitir.
        disponibles: participantes.length,
        identificados,
        sinUsuario,
        datosIncompletos,
        duplicados,
        duplicadosDetalle,
        noAprobados,
        rutasInesperadas,
        excluidos,
        truncado,
      },
      ocultarEnEmitir: certificado?.ocultarEnEmitir === true,
      participantes,
      // Apartados de la emisión. Conservan su aprobación intacta y pueden
      // recuperarse con PUT .../reincluir-usuario.
      participantesExcluidos,
    };
  }
}

async function resolverAprobadosCurso(cursoId: string) {
  const version = aprobadosCursoVersion.get(cursoId) || 0;
  const cacheado = aprobadosCursoCache.get(cursoId);
  if (cacheado && cacheado.expiraEn > Date.now()) {
    console.log(`[perf] aprobados-curso curso=${cursoId} cache=hit`);
    return cacheado.datos;
  }
  if (cacheado) aprobadosCursoCache.delete(cursoId);

  const enCurso = aprobadosCursoEnCurso.get(cursoId);
  if (enCurso) {
    console.log(`[perf] aprobados-curso curso=${cursoId} cache=shared`);
    return enCurso;
  }

  console.log(`[perf] aprobados-curso curso=${cursoId} cache=miss`);
  const carga = resolverAprobadosCursoSinCache(cursoId)
    .then((datos) => {
      if ((aprobadosCursoVersion.get(cursoId) || 0) === version) {
        aprobadosCursoCache.set(cursoId, {
          datos,
          expiraEn: Date.now() + APROBADOS_CACHE_TTL_MS,
        });
      }
      return datos;
    })
    .finally(() => {
      if (aprobadosCursoEnCurso.get(cursoId) === carga) {
        aprobadosCursoEnCurso.delete(cursoId);
      }
    });

  aprobadosCursoEnCurso.set(cursoId, carga);
  return carga;
}

function invalidarCacheAprobadosCurso(cursoId: string): void {
  const id = String(cursoId || "").trim();
  if (!id) return;
  aprobadosCursoVersion.set(id, (aprobadosCursoVersion.get(id) || 0) + 1);
  aprobadosCursoCache.delete(id);
  aprobadosCursoEnCurso.delete(id);
}

app.get("/api/certificados/admin/aprobados/:cursoId", async (req, res) => {
  try {
    const authUser = await verifyFirebaseIdToken(req.headers.authorization);
    await requireAdministrador(authUser);

    const cursoId = parseCursoIdParam(req.params.cursoId);
    const datos = await resolverAprobadosCurso(cursoId);

    return res.status(200).json({
      ok: true,
      modulo: "certificados",
      ...datos,
    });
  } catch (error: any) {
    return sendCertificadosError(res, error);
  }
});

// ============================================================
// REGISTRO DE APROBADOS PARA EL PORTAL DE VALIDADORES
//
// Este listado usa exactamente el mismo resolver que la emisión. No crea
// una colección paralela ni decide aprobaciones en el cliente: sólo proyecta
// los participantes que ya fueron aprobados, tienen datos completos, no están
// apartados y cuentan con afiliación habilitada.

function esAprobadoDisponibleParaRegistro(participante: ParticipanteAprobado) {
  return (
    participante.estado === "aprobado" &&
    participante.apartado !== true &&
    participante.afiliacion?.habilitadoCertificado === true &&
    participante.afiliacion.tipo !== "no_verificada" &&
    participante.certificadoEmitido === true &&
    participante.certificadoQrValido === true
  );
}

function proyectarRegistroAprobado(participante: ParticipanteAprobado) {
  return {
    usuarioDocId: participante.usuarioDocId,
    usuarioDocIds: participante.usuarioDocIds,
    apellidoNombre: participante.apellidoNombre,
    dni: participante.dni,
    departamento: {
      crudo: participante.departamento?.crudo || "",
      canonico: participante.departamento?.canonico || "",
    },
  };
}

async function obtenerRegistrosAprobadosCurso(
  cursoId: string,
  configuracion?: FirestoreRecord
) {
  const datos = await resolverAprobadosCurso(cursoId);
  const certificado = configuracion || (await getFirestoreDoc(`certificados/${cursoId}`));
  const aprobados = datos.participantes
    .filter(esAprobadoDisponibleParaRegistro)
    .map(proyectarRegistroAprobado)
    .sort((a, b) =>
      a.apellidoNombre.localeCompare(b.apellidoNombre, "es", {
        sensitivity: "base",
      })
    );

  return {
    curso: {
      ...datos.curso,
      resolucion: String(certificado?.resolucion || "").trim(),
    },
    cantidad: aprobados.length,
    aprobados,
  };
}

app.get("/api/certificados/registro-aprobados", async (req, res) => {
  const inicio = Date.now();
  try {
    const marcaAuth = Date.now();
    const authUser = await verifyFirebaseIdToken(req.headers.authorization);
    const tiempoAuth = Date.now() - marcaAuth;

    const marcaPermiso = Date.now();
    await requireAdministradorOValidadorCertificados(authUser);
    const tiempoPermiso = Date.now() - marcaPermiso;

    const marcaFirestore = Date.now();

    const documentos = await queryFirestoreCollection(
      "certificados",
      [],
      CONFIGURACIONES_MAX_RESULTADOS
    );
    const configuraciones = documentos
      .filter((record) => Boolean(String(record.cursoId || "").trim()))
      .filter((record) => record.ocultarEnEmitir !== true);

    // Un curso por vez sería la suma de todos; en paralelo es el máximo. La
    // caché single-flight de resolverPadronPorDni (afiliacion.ts) hace que los
    // cursos simultáneos compartan una sola paginación por colección en un
    // cache miss; durante el TTL reutilizan esos documentos completos.
    const cursos = await Promise.all(
      configuraciones.map(async (configuracion) => {
        const cursoId = String(configuracion.cursoId || "").trim();
        try {
          const registro = await obtenerRegistrosAprobadosCurso(cursoId, configuracion);
          return {
            cursoId,
            titulo: registro.curso?.titulo || configuracion.cursoTitulo || configuracion.titulo || "",
            resolucion: registro.curso?.resolucion || "",
            cantidadAprobados: registro.cantidad,
          };
        } catch (error: any) {
          if (Number(error?.statusCode) === 404) return null;
          throw error;
        }
      })
    );

    const disponibles = cursos
      .filter((curso): curso is NonNullable<typeof curso> => curso !== null)
      .sort((a, b) => a.titulo.localeCompare(b.titulo, "es", { sensitivity: "base" }));

    const tiempoFirestore = Date.now() - marcaFirestore;
    console.log(`[perf] registro-aprobados firebaseAuth=${tiempoAuth}ms resolverPermiso=${tiempoPermiso}ms firestore=${tiempoFirestore}ms cursos=${disponibles.length} total=${Date.now() - inicio}ms`);

    return res.status(200).json({
      ok: true,
      modulo: "certificados",
      cursos: disponibles,
    });
  } catch (error: any) {
    return sendCertificadosError(res, error);
  }
});

app.get("/api/certificados/registro-aprobados/:cursoId", async (req, res) => {
  const inicio = Date.now();
  try {
    const marcaAuth = Date.now();
    const authUser = await verifyFirebaseIdToken(req.headers.authorization);
    const tiempoAuth = Date.now() - marcaAuth;

    const marcaPermiso = Date.now();
    await requireAdministradorOValidadorCertificados(authUser);
    const tiempoPermiso = Date.now() - marcaPermiso;

    const cursoId = parseCursoIdParam(req.params.cursoId);

    const marcaFirestore = Date.now();
    const registro = await obtenerRegistrosAprobadosCurso(cursoId);
    const tiempoFirestore = Date.now() - marcaFirestore;

    console.log(`[perf] registro-aprobados-curso curso=${cursoId} firebaseAuth=${tiempoAuth}ms resolverPermiso=${tiempoPermiso}ms firestore=${tiempoFirestore}ms aprobados=${registro.cantidad} total=${Date.now() - inicio}ms`);

    return res.status(200).json({
      ok: true,
      modulo: "certificados",
      ...registro,
    });
  } catch (error: any) {
    return sendCertificadosError(res, error);
  }
});

// ============================================================
// EXCLUSIONES ADMINISTRATIVAS DE LA EMISIÓN
//
// "Quitar de la emisión" NO borra nada del sistema académico. La aprobación
// original vive en usuarios/{usuarioDocId}/cursos y queda intacta; el curso
// y su configuración también. Lo único que ocurre es que este módulo deja de
// ofrecerlos para emitir certificados.
//
// Por eso la ÚNICA escritura de estos endpoints es sobre
// certificados/{cursoId}, y sólo sobre dos campos:
//
//   usuariosExcluidos : string[]  - participantes apartados
//   ocultarEnEmitir   : boolean   - curso apartado
//
// Se usa updateFirestoreDoc, que envía updateMask, así que ningún otro campo
// del documento se toca. Todo es reversible.
// ============================================================

const excluirUsuarioSchema = z.strictObject({
  usuarioDocId: z
    .string()
    .trim()
    .regex(CERTIFICADO_ID_REGEX, "El identificador del participante es inválido."),
});

/**
 * Lee el documento de certificado exigiendo que exista.
 * Sin configuración previa no hay dónde registrar la exclusión.
 */
async function obtenerCertificadoParaEmision(
  cursoId: string
): Promise<FirestoreRecord> {
  const certificado = await getFirestoreDoc(`certificados/${cursoId}`);

  if (!certificado) {
    throw Object.assign(
      new Error("Todavía no hay una configuración de certificado para este curso."),
      { statusCode: 404 }
    );
  }

  return certificado;
}

/** Lista de excluidos normalizada y sin duplicados. */
function leerUsuariosExcluidos(certificado: FirestoreRecord): string[] {
  const valores = Array.isArray(certificado.usuariosExcluidos)
    ? certificado.usuariosExcluidos
    : [];

  return [
    ...new Set(
      valores.map((valor: any) => String(valor || "").trim()).filter(Boolean)
    ),
  ];
}

app.put(
  "/api/certificados/admin/emision/:cursoId/excluir-usuario",
  async (req, res) => {
    try {
      const authUser = await verifyFirebaseIdToken(req.headers.authorization);
      await requireAdministrador(authUser);

      const cursoId = parseCursoIdParam(req.params.cursoId);
      const { usuarioDocId } = excluirUsuarioSchema.parse(req.body);

      const certificado = await obtenerCertificadoParaEmision(cursoId);
      const actuales = leerUsuariosExcluidos(certificado);

      const yaEstaba = actuales.includes(usuarioDocId);
      const usuariosExcluidos = yaEstaba
        ? actuales
        : [...actuales, usuarioDocId];

      if (!yaEstaba) {
        await updateFirestoreDoc(`certificados/${cursoId}`, {
          usuariosExcluidos,
          actualizadoEn: new Date().toISOString(),
          actualizadoPor: authUser.uid,
        });
        invalidarCacheAprobadosCurso(cursoId);
      }

      return res.status(200).json({
        ok: true,
        modulo: "certificados",
        cursoId,
        usuarioDocId,
        yaEstaba,
        usuariosExcluidos,
      });
    } catch (error: any) {
      if (error instanceof z.ZodError) {
        return sendCertificadosError(
          res,
          Object.assign(
            new Error(error.issues.map((issue) => issue.message).join(" | ")),
            { statusCode: 400 }
          )
        );
      }
      return sendCertificadosError(res, error);
    }
  }
);

app.put(
  "/api/certificados/admin/emision/:cursoId/reincluir-usuario",
  async (req, res) => {
    try {
      const authUser = await verifyFirebaseIdToken(req.headers.authorization);
      await requireAdministrador(authUser);

      const cursoId = parseCursoIdParam(req.params.cursoId);
      const { usuarioDocId } = excluirUsuarioSchema.parse(req.body);

      const certificado = await obtenerCertificadoParaEmision(cursoId);
      const actuales = leerUsuariosExcluidos(certificado);

      const estaba = actuales.includes(usuarioDocId);
      const usuariosExcluidos = actuales.filter((id) => id !== usuarioDocId);

      if (estaba) {
        await updateFirestoreDoc(`certificados/${cursoId}`, {
          usuariosExcluidos,
          actualizadoEn: new Date().toISOString(),
          actualizadoPor: authUser.uid,
        });
        invalidarCacheAprobadosCurso(cursoId);
      }

      return res.status(200).json({
        ok: true,
        modulo: "certificados",
        cursoId,
        usuarioDocId,
        estaba,
        usuariosExcluidos,
      });
    } catch (error: any) {
      if (error instanceof z.ZodError) {
        return sendCertificadosError(
          res,
          Object.assign(
            new Error(error.issues.map((issue) => issue.message).join(" | ")),
            { statusCode: 400 }
          )
        );
      }
      return sendCertificadosError(res, error);
    }
  }
);

/** Aparta o reincorpora el curso completo. No borra nada. */
async function cambiarVisibilidadEmision(
  req: express.Request,
  res: express.Response,
  ocultar: boolean
) {
  try {
    const authUser = await verifyFirebaseIdToken(req.headers.authorization);
    await requireAdministrador(authUser);

    const cursoId = parseCursoIdParam(req.params.cursoId);
    await obtenerCertificadoParaEmision(cursoId);

    await updateFirestoreDoc(`certificados/${cursoId}`, {
      ocultarEnEmitir: ocultar,
      actualizadoEn: new Date().toISOString(),
      actualizadoPor: authUser.uid,
    });
    invalidarCacheAprobadosCurso(cursoId);

    return res.status(200).json({
      ok: true,
      modulo: "certificados",
      cursoId,
      ocultarEnEmitir: ocultar,
    });
  } catch (error: any) {
    return sendCertificadosError(res, error);
  }
}

app.put("/api/certificados/admin/emision/:cursoId/ocultar", (req, res) =>
  cambiarVisibilidadEmision(req, res, true)
);

app.put("/api/certificados/admin/emision/:cursoId/mostrar", (req, res) =>
  cambiarVisibilidadEmision(req, res, false)
);

// ============================================================
// EMISIÓN REAL DE UN CERTIFICADO
//
// Crea certificados/{cursoId}/emitidos/{token}.
//
// El cliente sólo aporta usuarioDocId: a quién emitir. TODO lo demás se
// vuelve a verificar y a leer del servidor — que aprobó, que no está
// excluido, que el usuario existe, qué nombre y qué DNI tiene, qué dice la
// configuración. Nada del body llega al documento guardado.
//
// El documento es un SNAPSHOT: congela participante y configuración tal como
// estaban al emitir. Si mañana cambia el nombre del usuario o la resolución
// del curso, el certificado ya emitido sigue diciendo lo que decía. Por eso
// no guarda referencias, guarda copias.
// ============================================================

/** Cuerpo del POST de emisión. Reutiliza la regex de IDs del módulo. */
const emitirCertificadoSchema = z.strictObject({
  usuarioDocId: z
    .string()
    .trim()
    .regex(CERTIFICADO_ID_REGEX, "El identificador del participante es inválido."),
});

/**
 * Valida usuarioDocId cuando llega como parámetro de RUTA.
 *
 * Mismo criterio que el schema del body, pero acá el valor se concatena en un
 * path de Firestore, así que la validación no es opcional: un valor con barras
 * o puntos podría escapar de la colección esperada.
 */
function parseUsuarioDocIdParam(valor: unknown): string {
  const usuarioDocId = String(valor ?? "").trim();

  if (!CERTIFICADO_ID_REGEX.test(usuarioDocId)) {
    throw Object.assign(
      new Error("El identificador del participante es inválido."),
      { statusCode: 400 }
    );
  }

  return usuarioDocId;
}

const EMITIDOS_ESTADO_VIGENTE = "vigente";
const TOKEN_BYTES = 24;
const TOKEN_MAX_INTENTOS = 3;

/** Base pública de la URL de validación que llevará el QR. */
const CERTIFICADOS_VALIDACION_BASE_URL = (
  process.env.CERTIFICADOS_VALIDACION_BASE_URL ||
  "https://sidcagremio.com/validar-certificado"
).replace(/\/+$/, "");

/**
 * Genera un token único para el certificado.
 *
 * crypto.randomBytes: 24 bytes → 48 caracteres hexadecimales en minúscula.
 * No se deriva de DNI, UID, cursoId ni de la hora: es imposible adivinarlo o
 * enumerarlo a partir de datos conocidos.
 *
 * La colisión es prácticamente imposible (2^192), pero se comprueba igual:
 * un choque silencioso sobrescribiría un certificado ajeno.
 */
async function generarTokenCertificado(cursoId: string): Promise<string> {
  for (let intento = 0; intento < TOKEN_MAX_INTENTOS; intento += 1) {
    const token = crypto.randomBytes(TOKEN_BYTES).toString("hex");
    const existente = await getFirestoreDoc(
      `certificados/${cursoId}/emitidos/${token}`
    );

    if (!existente) return token;
  }

  throw Object.assign(
    new Error("No se pudo generar un identificador único para el certificado."),
    { statusCode: 500 }
  );
}

/**
 * Proyecta una emisión guardada a la respuesta pública del módulo.
 *
 * Devuelve el SNAPSHOT tal como quedó guardado: no recalcula participante,
 * certificado ni urlValidacion. Si el nombre del usuario o la resolución del
 * curso cambiaron después de emitir, el certificado sigue diciendo lo que
 * decía — que es exactamente el sentido de guardar una foto.
 *
 * emitidoPor queda fuera: es dato de auditoría interna y el frontend no lo
 * necesita.
 */
function mapEmisionCertificado(emision: FirestoreRecord): Record<string, any> {
  return {
    certificadoId: emision.certificadoId || emision.id,
    token: emision.token || emision.id,
    cursoId: emision.cursoId || "",
    usuarioDocId: emision.usuarioDocId || "",
    estado: emision.estado || "",
    participante: emision.participante || null,
    certificado: emision.certificado || null,
    urlValidacion: emision.urlValidacion || "",
    emitidoEn: emision.emitidoEn || null,
  };
}

/**
 * Devuelve el certificado VIGENTE ya emitido para un participante y curso.
 *
 * Sólo LECTURA. Existe para que la pantalla de emisión pueda reconocer, al
 * abrir el preview, que ese participante ya tiene certificado — y mostrar su
 * QR — sin depender del estado en memoria, que se pierde al recargar.
 *
 * 404 cuando no hay emisión vigente: no es un error, es la respuesta normal
 * para alguien que todavía no fue emitido.
 */
app.get(
  "/api/certificados/admin/emision/:cursoId/usuario/:usuarioDocId",
  async (req, res) => {
    try {
      const authUser = await verifyFirebaseIdToken(req.headers.authorization);
      await requireAdministrador(authUser);

      const cursoId = parseCursoIdParam(req.params.cursoId);
      const usuarioDocId = parseUsuarioDocIdParam(req.params.usuarioDocId);

      // Sólo la subcolección de ESTE curso: no mezcla emitidos de otros.
      const emitidos = await queryFirestoreChildCollection(
        `certificados/${cursoId}`,
        "emitidos",
        [{ field: "usuarioDocId", value: usuarioDocId }],
        20
      );

      const vigentes = emitidos.filter(
        (emitido) => emitido.estado === EMITIDOS_ESTADO_VIGENTE
      );

      if (vigentes.length === 0) {
        throw Object.assign(
          new Error(
            "Este participante todavía no tiene un certificado vigente emitido para este curso."
          ),
          { statusCode: 404 }
        );
      }

      // No debería haber más de uno: el POST de emisión lo impide. Si aparece,
      // se avisa en el log sin elegir en silencio y se devuelve el más
      // reciente por emitidoEn, que Firestore guarda como timestamp ISO y por
      // lo tanto ordena bien de forma lexicográfica.
      if (vigentes.length > 1) {
        console.warn(
          `[sidca-chatbot-backend] multiples certificados vigentes curso=${cursoId} usuario=${usuarioDocId} cantidad=${vigentes.length}`
        );
      }

      const emision = [...vigentes].sort((a, b) =>
        String(b.emitidoEn || "").localeCompare(String(a.emitidoEn || ""))
      )[0];

      return res.status(200).json({
        ok: true,
        modulo: "certificados",
        emision: mapEmisionCertificado(emision),
      });
    } catch (error: any) {
      return sendCertificadosError(res, error);
    }
  }
);

app.get("/api/certificados/admin/emision/:cursoId/emitidos", async (req, res) => {
  try {
    const authUser = await verifyFirebaseIdToken(req.headers.authorization);
    await requireAdministrador(authUser);
    const cursoId = parseCursoIdParam(req.params.cursoId);
    const emitidos = await queryFirestoreChildCollection(
      `certificados/${cursoId}`,
      "emitidos",
      [{ field: "estado", value: EMITIDOS_ESTADO_VIGENTE }],
      APROBADOS_MAX_RESULTADOS
    );
    return res.status(200).json({
      ok: true,
      modulo: "certificados",
      cursoId,
      total: emitidos.length,
      emisiones: emitidos.map(mapEmisionCertificado),
    });
  } catch (error: any) {
    return sendCertificadosError(res, error);
  }
});

/**
 * Índice de unicidad por participante.
 *
 * El documento emitido vive en emitidos/{token} y el token es aleatorio, así
 * que dos administradores que emitan a la vez a la misma persona generarían
 * dos identificadores distintos y ninguna de las dos escrituras chocaría con
 * la otra: la comprobación previa de "¿ya tiene certificado vigente?" es una
 * lectura, y entre esa lectura y la escritura hay una ventana.
 *
 * Este índice cierra la ventana. Su identificador NO es aleatorio: es el
 * propio usuarioDocId, así que sólo puede existir una vez por curso, y
 * Firestore lo hace cumplir en el servidor.
 */
const EMISION_INDICE_COLECCION = "emisionUsuarios";

/** Ruta absoluta de un documento, como la exige documents:commit. */
const rutaDocumentoFirestore = (path: string) =>
  `projects/${firebaseProjectId}/databases/(default)/documents/${path}`;

/**
 * ¿El commit falló porque el documento ya existía?
 *
 * Firestore no es del todo consistente con qué devuelve al incumplirse una
 * precondición `exists: false` —según el caso, ALREADY_EXISTS o
 * FAILED_PRECONDITION—, así que se aceptan las dos y, como red, el texto.
 */
function esPrecondicionIncumplida(error: any): boolean {
  const estado = String(error?.firestoreStatus || "");

  if (estado === "ALREADY_EXISTS" || estado === "FAILED_PRECONDITION") {
    return true;
  }

  return /already exists|precondition/i.test(String(error?.message || ""));
}

/**
 * Escribe el certificado emitido y su índice de unicidad en UNA sola
 * transacción.
 *
 * Las dos escrituras viajan en el mismo `documents:commit`, cada una con
 * precondición `exists: false`. Firestore aplica el commit entero o ninguna
 * parte: no puede quedar un certificado sin índice ni un índice sin
 * certificado.
 *
 * Si el índice ya existía, el commit completo se rechaza y se traduce a un
 * 409 con `codigo: "ya_emitido"`, que es exactamente lo que ya devolvía la
 * comprobación previa. Para quien llama, ganar o perder la carrera se ve
 * igual.
 *
 * No se escribe nada sensible en el índice: usuarioDocId, token, estado y
 * fecha. El día que existan anulaciones, ese `estado` es el que habrá que
 * actualizar o el documento el que habrá que borrar para liberar al
 * participante; esta tarea no toca esa lógica.
 */
async function crearEmisionCertificadoAtomica({
  cursoId,
  usuarioDocId,
  token,
  emision,
  creadoEn,
}: {
  cursoId: string;
  usuarioDocId: string;
  token: string;
  emision: Record<string, any>;
  creadoEn: Date;
}): Promise<void> {
  const indice = {
    usuarioDocId,
    token,
    estado: EMITIDOS_ESTADO_VIGENTE,
    creadoEn,
  };

  const cuerpo = {
    writes: [
      {
        update: {
          name: rutaDocumentoFirestore(
            `certificados/${cursoId}/${EMISION_INDICE_COLECCION}/${usuarioDocId}`
          ),
          fields: jsToFirestoreFields(indice),
        },
        currentDocument: { exists: false },
      },
      {
        update: {
          name: rutaDocumentoFirestore(
            `certificados/${cursoId}/emitidos/${token}`
          ),
          fields: jsToFirestoreFields(emision),
        },
        currentDocument: { exists: false },
      },
    ],
  };

  try {
    await firestoreRequest(`${firestoreBaseUrl}:commit`, {
      method: "POST",
      body: JSON.stringify(cuerpo),
    });
  } catch (error: any) {
    if (esPrecondicionIncumplida(error)) {
      throw Object.assign(
        new Error(
          "Este participante ya tiene un certificado vigente emitido para este curso."
        ),
        { statusCode: 409, codigo: "ya_emitido" }
      );
    }

    throw error;
  }
}

/**
 * Datos del curso que NO dependen del participante.
 *
 * Se resuelven una sola vez por pedido: en la emisión masiva, releer la
 * configuración y todo el grupo de aprobaciones por cada persona convertiría
 * cien emisiones en cientos de consultas idénticas.
 */
type ContextoEmision = {
  certificado: FirestoreRecord;
  curso: FirestoreRecord;
  usuariosAprobados: Set<string>;
  dnisConCertificadoVigente: Set<string>;
};

/**
 * Prepara el contexto y aplica las validaciones de CURSO.
 *
 * Son los pasos 1 a 3 de la emisión, que no miran a ningún participante en
 * particular: la configuración tiene que existir, el curso académico tiene que
 * existir y el curso no puede estar apartado de la emisión.
 */
async function prepararContextoEmision(cursoId: string): Promise<ContextoEmision> {
  // 1. Configuración del certificado.
  const certificado = await obtenerCertificadoParaEmision(cursoId);

  // 2. El curso académico debe existir de verdad, no alcanza con el
  //    cursoTitulo copiado en la configuración.
  const curso = await getFirestoreDoc(`cursos/${cursoId}`);

  if (!curso) {
    throw Object.assign(new Error("El curso indicado no existe."), {
      statusCode: 404,
    });
  }

  // 3. Curso apartado de la emisión. Se comprueba acá también para que no
  //    baste con llamar al endpoint directamente salteando la UI.
  if (certificado.ocultarEnEmitir === true) {
    throw Object.assign(
      new Error("Este curso se encuentra apartado de la emisión de certificados."),
      { statusCode: 409 }
    );
  }

  // Aprobaciones reales del curso, resueltas una sola vez. Se guarda el
  // conjunto de usuarios con al menos una aprobación válida: los duplicados
  // del importador no emiten de más porque acá sólo importa la existencia.
  const aprobaciones = await queryFirestoreCollectionGroup(
    "cursos",
    [{ field: "curso", value: `cursos/${cursoId}` }],
    APROBADOS_MAX_RESULTADOS
  );

  const usuariosAprobados = new Set<string>();

  for (const aprobacion of aprobaciones) {
    if (aprobacion.aprobo !== true) continue;
    const usuarioDocId = extraerUsuarioDocIdDeAprobacion(aprobacion._name);
    if (usuarioDocId) usuariosAprobados.add(usuarioDocId);
  }

  const emitidosVigentes = await queryFirestoreChildCollection(
    `certificados/${cursoId}`,
    "emitidos",
    [{ field: "estado", value: EMITIDOS_ESTADO_VIGENTE }],
    APROBADOS_MAX_RESULTADOS
  );
  const dnisConCertificadoVigente = new Set<string>();
  emitidosVigentes.forEach((emitido) => {
    const dniEmitido = normalizeDni(
      emitido?.participante?.dni || emitido?.dni || ""
    );
    if (dniEmitido) dnisConCertificadoVigente.add(dniEmitido);
  });

  return {
    certificado,
    curso,
    usuariosAprobados,
    dnisConCertificadoVigente,
  };
}

/**
 * Emite UN certificado. Es la única implementación: la usan tanto el endpoint
 * individual como el masivo.
 *
 * El orden de las comprobaciones es deliberado y no debe alterarse: define qué
 * error ve el administrador cuando un participante incumple más de una
 * condición a la vez.
 *
 * `contexto` es opcional: sin él la función se autoabastece, con él se evita
 * releer lo mismo N veces. En ambos casos las validaciones son idénticas.
 *
 * Lanza con `statusCode` y, cuando el motivo es una emisión vigente previa,
 * también con `codigo: "ya_emitido"`, para que la masiva pueda clasificar ese
 * caso sin comparar textos.
 */
async function emitirCertificadoParaUsuario({
  cursoId,
  usuarioDocId,
  authUser,
  contexto,
}: {
  cursoId: string;
  usuarioDocId: string;
  authUser: AuthenticatedUser;
  contexto?: ContextoEmision;
}) {
  const {
    certificado,
    curso,
    usuariosAprobados,
    dnisConCertificadoVigente,
  } =
    contexto || (await prepararContextoEmision(cursoId));

  {
    // 4. Participante excluido por decisión administrativa.
    if (leerUsuariosExcluidos(certificado).includes(usuarioDocId)) {
      throw Object.assign(
        new Error("Este participante fue excluido de la emisión de certificados."),
        { statusCode: 409 }
      );
    }

    // 5. Aprobación real. No se confía en que la UI lo haya mostrado: la lista
    //    sale de las aprobaciones releídas del curso.
    if (!usuariosAprobados.has(usuarioDocId)) {
      throw Object.assign(
        new Error("El participante no registra una aprobación válida para este curso."),
        { statusCode: 409 }
      );
    }

    // 6. Documento académico del participante. Es el padre de la aprobación:
    //    no se busca un reemplazo por DNI, porque sería otra persona.
    const usuario = await getFirestoreDoc(`usuarios/${usuarioDocId}`);

    if (!usuario) {
      throw Object.assign(
        new Error(
          "No se puede emitir el certificado porque el usuario académico ya no existe."
        ),
        { statusCode: 409 }
      );
    }

    // 7. Datos mínimos para imprimir. buildNombreAfiliado nunca devuelve
    //    vacío —cae en un texto por defecto—, así que se comprueban los
    //    campos de origen y no su resultado.
    const dni = normalizeDni(usuario.dni);
    const apellidoNombre = buildNombreAfiliado(usuario);
    const apellido = String(usuario.apellido || "").trim();
    const nombre = String(usuario.nombre || "").trim();

    const tieneNombreReal = Boolean(
      String(usuario.apellidoNombre || usuario.apellido_y_nombre || "").trim() ||
        apellido ||
        nombre
    );

    if (!dni || !tieneNombreReal) {
      throw Object.assign(
        new Error(
          "No se puede emitir el certificado porque los datos del participante están incompletos."
        ),
        { statusCode: 409 }
      );
    }

    // La unicidad funcional es por curso + DNI, no por usuarioDocId. Un mismo
    // afiliado puede existir en más de un documento de `usuarios`; en ese
    // caso, la emisión individual reconoce el certificado previo antes de
    // intentar escribir otro para el documento duplicado.
    if (dnisConCertificadoVigente.has(dni)) {
      throw Object.assign(
        new Error(
          "Este participante ya tiene un certificado vigente emitido para este curso."
        ),
        { statusCode: 409, codigo: "ya_emitido" }
      );
    }

    // 7 bis. Condición sindical.
    //
    //   El bloqueo del navegador no alcanza: sin esta comprobación bastaría
    //   con llamar al endpoint a mano para saltearlo. Va acá, en la función
    //   compartida, así rige igual para la emisión individual y la masiva sin
    //   duplicar la regla.
    //
    //   Se resuelve DESPUÉS de tener el DNI y ANTES de escribir nada.
    const afiliacion = await resolverAfiliacionDeUnDni(dni, {
      proyecto: firebaseProjectId,
      accessToken: await getGoogleAccessToken(),
    });

    if (afiliacion.habilitadoCertificado !== true) {
      throw Object.assign(
        new Error(
          afiliacion.tipo === "adherente"
            ? "No se puede emitir el certificado porque el adherente no se encuentra habilitado."
            : "No se puede emitir el certificado porque no se pudo verificar la condición de afiliación."
        ),
        // El código permite a la emisión masiva contabilizar este caso aparte
        // en vez de tratarlo como una falla.
        { statusCode: 409, codigo: "afiliacion_no_habilitada" }
      );
    }

    // 8. Doble emisión. Se consulta sólo la subcolección de ESTE curso.
    const emitidosPrevios = await queryFirestoreChildCollection(
      `certificados/${cursoId}`,
      "emitidos",
      [{ field: "usuarioDocId", value: usuarioDocId }],
      20
    );

    const vigente = emitidosPrevios.find(
      (emitido) => emitido.estado === EMITIDOS_ESTADO_VIGENTE
    );

    if (vigente) {
      throw Object.assign(
        new Error(
          "Este participante ya tiene un certificado vigente emitido para este curso."
        ),
        // El código permite a la emisión masiva contarlo como "ya emitido" en
        // vez de como error. Es la carrera real: mientras corre la masiva,
        // otro administrador puede emitirle a esta misma persona.
        { statusCode: 409, codigo: "ya_emitido" }
      );
    }

    // 9. Autoridades. El PUT permite guardar borradores incompletos, pero un
    //    certificado emitido es irreversible: tiene que salir con las dos
    //    autoridades completas. Se leen con el helper, que además resuelve las
    //    configuraciones legacy que todavía tienen "firmas".
    //    "Completa" significa nombre + cargo: organismo y referencia son
    //    renglones opcionales y no pueden bloquear una emisión.
    const autoridadesCertificado = obtenerAutoridadesConfiguracion(certificado);

    const autoridadesCompletas =
      autoridadesCertificado.length === 2 &&
      autoridadesCertificado.every(
        (autoridad) =>
          String(autoridad.nombre || "").trim() !== "" &&
          String(autoridad.cargo || "").trim() !== ""
      );

    if (!autoridadesCompletas) {
      throw Object.assign(
        new Error(
          "No se puede emitir el certificado porque faltan completar las dos autoridades."
        ),
        { statusCode: 409 }
      );
    }

    // 10. Institución. Las configuraciones anteriores al campo se emiten como
    //     "sidca", que era la única plantilla existente.
    const institucionCertificado =
      normalizarInstitucionCertificado(certificado);

    // 11. Token y URL de validación.
    const token = await generarTokenCertificado(cursoId);
    const urlValidacion = `${CERTIFICADOS_VALIDACION_BASE_URL}/${cursoId}/${token}`;

    const ahora = new Date();

    // 10. Snapshot. El título del curso sale del curso REAL; el resto, de la
    //     configuración. Nada viene del body.
    const participante = {
      usuarioDocId,
      dni,
      apellidoNombre,
      ...(apellido ? { apellido } : {}),
      ...(nombre ? { nombre } : {}),
    };

    // La institución y las autoridades forman parte del SNAPSHOT: si mañana
    // se cambia la institución del curso o el cargo de una autoridad, el
    // certificado ya emitido sigue diciendo lo que decía cuando se emitió.
    const certificadoSnapshot = {
      institucionCertificado: institucionCertificado,
      cursoTitulo: String(curso.titulo || certificado.cursoTitulo || ""),
      titulo: String(certificado.titulo || ""),
      resolucion: String(certificado.resolucion || ""),
      cargaHoraria: String(certificado.cargaHoraria || ""),
      dias: String(certificado.dias || ""),
      fecha: String(certificado.fecha || ""),
      modalidad: String(certificado.modalidad || ""),
      autoridades: autoridadesCertificado,
    };

    const datos = {
      version: 1,
      certificadoId: token,
      token,
      cursoId,
      usuarioDocId,
      estado: EMITIDOS_ESTADO_VIGENTE,
      participante,
      certificado: certificadoSnapshot,
      urlValidacion,
      emitidoEn: ahora,
      emitidoPor: authUser.uid,
    };

    // Un único commit con las dos escrituras y sus precondiciones. La
    // comprobación del paso 8 sigue siendo la primera línea de defensa —y la
    // que cubre los certificados históricos, anteriores a este índice—, pero
    // es una lectura: quien de verdad impide la doble emisión simultánea es
    // esta transacción.
    await crearEmisionCertificadoAtomica({
      cursoId,
      usuarioDocId,
      token,
      emision: datos,
      creadoEn: ahora,
    });

    console.log(
      `[sidca-chatbot-backend] certificado emitido curso=${cursoId} usuario=${usuarioDocId} token=${token} por=${authUser.uid}`
    );

    return {
      certificadoId: token,
      token,
      cursoId,
      usuarioDocId,
      estado: EMITIDOS_ESTADO_VIGENTE,
      participante,
      certificado: certificadoSnapshot,
      urlValidacion,
      emitidoEn: ahora.toISOString(),
    };
  }
}

app.post("/api/certificados/admin/emision/:cursoId/emitir", async (req, res) => {
  try {
    const authUser = await verifyFirebaseIdToken(req.headers.authorization);
    await requireAdministrador(authUser);

    const cursoId = parseCursoIdParam(req.params.cursoId);

    let usuarioDocId: string;
    try {
      ({ usuarioDocId } = emitirCertificadoSchema.parse(req.body));
    } catch (error: any) {
      throw Object.assign(
        new Error(
          error?.issues?.map((issue: any) => issue.message).join(" | ") ||
            "El cuerpo del pedido es inválido."
        ),
        { statusCode: 400 }
      );
    }

    const emision = await emitirCertificadoParaUsuario({
      cursoId,
      usuarioDocId,
      authUser,
    });
    invalidarCacheAprobadosCurso(cursoId);

    // Misma forma de respuesta que antes del refactor: la pantalla de emisión
    // individual no se entera de que la lógica se movió.
    return res.status(201).json({
      ok: true,
      modulo: "certificados",
      creado: true,
      emision,
    });
  } catch (error: any) {
    return sendCertificadosError(res, error);
  }
});

// ============================================================
// EMISIÓN MASIVA
//
// Emite de una sola vez a todos los aprobados elegibles que todavía no tienen
// certificado vigente. NO genera PDF: eso es la descarga masiva, que sólo lee
// lo ya emitido. Son dos responsabilidades separadas a propósito.
//
// El cuerpo del pedido se ignora por completo. El navegador manda un cursoId
// en la URL y nada más: quién corresponde emitir lo decide el backend
// releyendo el padrón, nunca la lista que el administrador tiene en pantalla.
// ============================================================

/** Emisiones simultáneas. Con lotes de 10 mil personas no abren mil conexiones. */
const EMISION_MASIVA_LOTE = 10;

app.post("/api/certificados/admin/emision/:cursoId/emitir-masivo", async (req, res) => {
  try {
    const authUser = await verifyFirebaseIdToken(req.headers.authorization);
    await requireAdministrador(authUser);

    const cursoId = parseCursoIdParam(req.params.cursoId);

    // El padrón se reconstruye acá: mismo helper que alimenta la pantalla, así
    // que "aprobado", "apartado" y "datos incompletos" significan lo mismo en
    // los dos lados.
    const padron = await resolverAprobadosCurso(cursoId);

    // Valida configuración, curso y curso apartado antes de tocar a nadie, y
    // deja resueltas las aprobaciones para todo el lote.
    const contexto = await prepararContextoEmision(cursoId);

    // Elegibles: aprobados de ESTE curso, con usuario, con nombre y DNI (eso
    // es lo que significa estado "aprobado"), no apartados —los apartados
    // viven en otra lista— y sin certificado vigente.
    const candidatos = padron.participantes.filter(
      (participante) =>
        participante.estado === "aprobado" &&
        participante.apartado !== true &&
        participante.certificadoEmitido !== true &&
        Boolean(participante.usuarioDocId)
    );

    let emitidos = 0;
    // Arranca en los que ya venían emitidos y suma los que resulten emitidos
    // por otro administrador mientras esto corre.
    let yaEmitidos = padron.participantes.filter(
      (participante) => participante.certificadoEmitido === true
    ).length;

    // Los adherentes no habilitados no son un error: se cuentan aparte y la
    // tanda sigue. Un bloqueo administrativo no puede frenar las demás emisiones.
    let afiliacionNoHabilitada = 0;

    const errores: { usuarioDocId: string; apellidoNombre: string; mensaje: string }[] = [];

    for (let i = 0; i < candidatos.length; i += EMISION_MASIVA_LOTE) {
      const lote = candidatos.slice(i, i + EMISION_MASIVA_LOTE);

      // allSettled y no all: que una persona falle no puede abortar el lote ni
      // deshacer los certificados ya escritos. Cada emisión es independiente.
      const resultados = await Promise.allSettled(
        lote.map((participante) =>
          emitirCertificadoParaUsuario({
            cursoId,
            usuarioDocId: participante.usuarioDocId,
            authUser,
            contexto,
          })
        )
      );

      resultados.forEach((resultado, indice) => {
        const participante = lote[indice];

        if (resultado.status === "fulfilled") {
          emitidos += 1;
          return;
        }

        const error: any = resultado.reason;

        if (error?.codigo === "ya_emitido") {
          yaEmitidos += 1;
          return;
        }

        if (error?.codigo === "afiliacion_no_habilitada") {
          afiliacionNoHabilitada += 1;
          return;
        }

        errores.push({
          usuarioDocId: participante.usuarioDocId,
          apellidoNombre: participante.apellidoNombre || "",
          mensaje: String(error?.message || "No se pudo emitir el certificado."),
        });
      });
    }

    if (emitidos > 0) invalidarCacheAprobadosCurso(cursoId);

    console.log(
      `[sidca-chatbot-backend] emisión masiva curso= candidatos= emitidos= yaEmitidos= afiliacion= errores= por=`
    );

    // 200 y no 201 aunque haya creado documentos: es un resultado agregado, y
    // "no había nadie pendiente" es un desenlace normal, no un error.
    return res.status(200).json({
      ok: true,
      modulo: "certificados",
      cursoId,
      totalAprobados: padron.resumen.aprobados,
      candidatos: candidatos.length,
      emitidos,
      yaEmitidos,
      omitidos: {
        apartados: padron.resumen.excluidos,
        datosIncompletos: padron.resumen.datosIncompletos,
        sinUsuario: padron.resumen.sinUsuario,
        afiliacionNoHabilitada,
      },
      // Sin tokens ni URLs: son datos sensibles y no hacen falta para el
      // resumen. Quien necesite uno lo pide por el endpoint individual.
      errores,
    });
  } catch (error: any) {
    return sendCertificadosError(res, error);
  }
});

// ============================================================
// VALIDACIÓN DE UN CERTIFICADO ESCANEADO
//
// Es el endpoint al que llega la página que abre el QR.
//
// NO va bajo /admin porque también lo usan los validadores designados, que no
// son administradores. Pero NO es público: el token del QR es difícil de
// adivinar, y eso no reemplaza autenticación. Sin Firebase ID Token no
// devuelve nada.
//
// Sólo LECTURA: no registra el escaneo ni toca el documento. La auditoría de
// verificaciones, si se decide llevarla, será otra etapa.
// ============================================================

const EMITIDOS_ESTADOS_CONOCIDOS = new Set([
  "vigente",
  "anulado",
  "reemplazado",
]);

app.get("/api/certificados/validar/:cursoId/:token", async (req, res) => {
  try {
    const inicioValidacion = Date.now();

    const inicioFirebaseAuth = Date.now();
    const authUser = await verifyFirebaseIdToken(req.headers.authorization);
    const firebaseAuth = Date.now() - inicioFirebaseAuth;

    const cursoId = parseCursoIdParam(req.params.cursoId);
    const token = parseCertificadoTokenParam(req.params.token);

    // Auth ya terminó: recién ahora se pueden iniciar en paralelo la
    // autorización y la lectura directa del certificado.
    let resolverPermiso = 0;
    const inicioResolverPermiso = Date.now();
    const permisoPromise = requireAdministradorOValidadorCertificados(authUser)
      .finally(() => {
        resolverPermiso = Date.now() - inicioResolverPermiso;
      });

    // El token ES el ID del documento: lectura directa, sin query.
    let leerCertificado = 0;
    const inicioLeerCertificado = Date.now();
    const emisionPromise = getFirestoreDoc(
      `certificados/${cursoId}/emitidos/${token}`
    ).finally(() => {
      leerCertificado = Date.now() - inicioLeerCertificado;
    });

    const [permiso, emision] = await Promise.all([
      permisoPromise,
      emisionPromise,
    ]);

    const noEncontrado = () =>
      Object.assign(
        new Error("Certificado no encontrado o código de validación inválido."),
        { statusCode: 404 }
      );

    if (!emision) throw noEncontrado();

    // Comprobación defensiva: el path ya identifica el documento, pero si su
    // contenido no coincide con la ruta, algo está mal y no se afirma nada.
    // No se intenta reparar el documento.
    if (String(emision.cursoId || "") !== cursoId) throw noEncontrado();
    if (emision.token && String(emision.token) !== token) throw noEncontrado();

    const estado = String(emision.estado || "");
    const valido = estado === EMITIDOS_ESTADO_VIGENTE;

    // Un certificado anulado o reemplazado EXISTE: responde 200 con
    // valido:false, no 404. La diferencia importa — el validador tiene que
    // poder distinguir "este QR es falso" de "este certificado es real pero
    // ya no está vigente".
    if (!EMITIDOS_ESTADOS_CONOCIDOS.has(estado)) {
      console.warn(
        `[sidca-chatbot-backend] estado de certificado desconocido curso=${cursoId} estado=${estado || "(vacio)"}`
      );
    }

    console.log(
      `[sidca-chatbot-backend] certificado validado curso=${cursoId} estado=${estado} por=${permiso.tipo}`
    );

    const inicioRegistrarVerificacion = Date.now();
    const verificado = await addFirestoreDoc(
      `certificados/${cursoId}/emitidos/${token}/verificaciones`,
      {
        cursoId,
        token,
        estadoCertificado: estado,
        valido,
        validadorUid: authUser.uid,
        validadorEmail: authUser.email || "",
        validadorNombre: permiso.tipo === "validador" ? buildNombreAfiliado(permiso.usuario) : "Administrador SIDCA",
        tipoValidador: permiso.tipo,
        validadoEn: new Date().toISOString(),
      }
    );
    const registrarVerificacion = Date.now() - inicioRegistrarVerificacion;

    console.log(
      `[perf] validar-certificado firebaseAuth=${firebaseAuth}ms resolverPermiso=${resolverPermiso}ms leerCertificado=${leerCertificado}ms registrarVerificacion=${registrarVerificacion}ms total=${Date.now() - inicioValidacion}ms estado=${estado || "(vacio)"} tipoValidador=${permiso.tipo}`
    );

    return res.status(200).json({
      ok: true,
      modulo: "certificados",
      validacion: {
        valido,
        estado,
        certificadoId: emision.certificadoId || token,
        cursoId,
        participante: emision.participante || null,
        certificado: emision.certificado || null,
        emitidoEn: emision.emitidoEn || null,
        registroCurso: emision.registroCurso || null,
      },
      verificacion: {
        validadoEn: verificado.validadoEn,
        validador: {
          nombre: verificado.validadorNombre,
          email: verificado.validadorEmail || null,
          tipo: verificado.tipoValidador,
        },
      },
    });
  } catch (error: any) {
    return sendCertificadosError(res, error);
  }
});

/** Registra una sola vez que un certificado fue presentado y validado. */
app.post("/api/certificados/validar/:cursoId/:token/registrar", async (req, res) => {
  try {
    const authUser = await verifyFirebaseIdToken(req.headers.authorization);
    const permiso = await requireAdministradorOValidadorCertificados(authUser);
    const cursoId = parseCursoIdParam(req.params.cursoId);
    const token = parseCertificadoTokenParam(req.params.token);
    const path = `certificados/${cursoId}/emitidos/${token}`;
    const emision = await getFirestoreDoc(path);
    if (!emision || String(emision.estado || "") !== EMITIDOS_ESTADO_VIGENTE) {
      throw Object.assign(new Error("El certificado no está vigente y no puede registrarse."), { statusCode: 409 });
    }
    if (String(emision.cursoId || cursoId) !== cursoId || (emision.token && String(emision.token) !== token)) {
      throw Object.assign(new Error("El certificado no coincide con el código escaneado."), { statusCode: 404 });
    }
    if (emision.registroCurso) {
      return res.status(200).json({ ok: true, yaRegistrado: true, registro: emision.registroCurso });
    }
    const registradoEn = new Date().toISOString();
    const registro = {
      registrado: true,
      registradoEn,
      registradoPorUid: authUser.uid,
      registradoPorEmail: authUser.email || "",
      registradoPorNombre: permiso.tipo === "validador" ? buildNombreAfiliado(permiso.usuario) : "Administrador SIDCA",
      tipoValidador: permiso.tipo,
    };
    await updateFirestoreDoc(path, { registroCurso: registro });
    return res.status(200).json({ ok: true, registro });
  } catch (error: any) {
    return sendCertificadosError(res, error);
  }
});

// ============================================================
// VALIDACION DE CENA
//
// Cena comparte la autorización institucional de certificados, pero conserva
// sus propios datos funcionales y su transacción de acreditación. La lectura
// devuelve una foto completa de la reserva; la escritura se hace sólo en
// /registrar, mediante una transacción REST de Firestore para que dos escaneos
// simultáneos no acrediten dos veces.
// ============================================================

const CENA_TOKEN_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const CENA_ANIO_REGEX = /^\d{4}$/;

function parseCenaTokenParam(valor: unknown): string {
  const token = String(valor ?? "").trim();
  if (!CENA_TOKEN_REGEX.test(token)) {
    throw Object.assign(new Error("El código de validación de Cena es inválido."), { statusCode: 400 });
  }
  return token;
}

function parseCenaAnioParam(valor: unknown): string {
  const anio = String(valor ?? "").trim();
  if (!CENA_ANIO_REGEX.test(anio)) {
    throw Object.assign(new Error("El año de la Cena es inválido."), { statusCode: 400 });
  }
  return anio;
}

async function requireAdministradorOValidadorCena(authUser: AuthenticatedUser): Promise<PermisoCertificados> {
  try {
    return await requireAdministradorOValidadorCertificados(authUser);
  } catch (error: any) {
    if (Number(error?.statusCode) === 403) {
      throw Object.assign(
        new Error("No tenés autorización para validar la Cena SIDCA."),
        { statusCode: 403 }
      );
    }
    throw error;
  }
}

function getUbicacionTarjetaCena(tarjeta: FirestoreRecord) {
  const path = getFirestoreRelativePath(tarjeta);
  const match = path.match(/^gestion_cena\/(\d{4})\/tarjetas\/([^/]+)$/);
  if (!match) {
    throw Object.assign(new Error("La tarjeta de Cena tiene una ubicación inválida."), { statusCode: 500 });
  }

  const reservaId = String(tarjeta.reservaId || "").trim();
  if (!/^[A-Za-z0-9_-]{1,128}$/.test(reservaId)) {
    throw Object.assign(new Error("La tarjeta de Cena no tiene una reserva válida."), { statusCode: 500 });
  }

  return {
    anio: match[1],
    tarjetaId: match[2],
    tarjetaPath: path,
    reservaId,
    reservaPath: `gestion_cena/${match[1]}/reservas/${reservaId}`,
  };
}

async function buscarTarjetaCenaPorToken(token: string): Promise<FirestoreRecord> {
  const tarjetas = await queryFirestoreCollectionGroup("tarjetas", [{ field: "token", value: token }], 2);
  if (tarjetas.length === 0) {
    throw Object.assign(new Error("Tarjeta de Cena no encontrada o código inválido."), { statusCode: 404 });
  }
  if (tarjetas.length > 1) {
    console.error(`[sidca-chatbot-backend] token de cena duplicado token=${token}`);
    throw Object.assign(new Error("No se pudo resolver la tarjeta de Cena."), { statusCode: 409 });
  }

  const tarjeta = tarjetas[0];
  // El token es el valor firmado en el QR. En tarjetas heredadas el ID del
  // documento puede diferir de ese token, por lo que no se lo usa como regla
  // de validación ni se invalida un QR legítimo por esa diferencia.
  if (String(tarjeta.token || "") !== token) {
    throw Object.assign(new Error("Tarjeta de Cena no encontrada o código inválido."), { statusCode: 404 });
  }
  return tarjeta;
}

const ordenarTarjetasCena = (a: FirestoreRecord, b: FirestoreRecord) =>
  Number(a.numeroTarjeta || 0) - Number(b.numeroTarjeta || 0) ||
  Number(Boolean(a.anulada)) - Number(Boolean(b.anulada)) ||
  Number(a.numeroReemision || 0) - Number(b.numeroReemision || 0) ||
  String(a.id || "").localeCompare(String(b.id || ""));

function esTarjetaVigenteCena(tarjeta: FirestoreRecord | null): tarjeta is FirestoreRecord {
  return tarjeta !== null && tarjeta.anulada !== true && tarjeta.reemplazada !== true;
}

function tarjetaCenaAcreditada(tarjeta: FirestoreRecord | null): boolean {
  return esTarjetaVigenteCena(tarjeta) && (tarjeta.validada === true || tarjeta.estado === "validada");
}

function resumirTarjetasCena(tarjetas: FirestoreRecord[]) {
  const vigentes = tarjetas.filter(esTarjetaVigenteCena);
  const acreditadas = vigentes.filter(tarjetaCenaAcreditada).length;
  return { vigentes, acreditadas, pendientes: vigentes.length - acreditadas };
}

function normalizarFechaCena(valor: unknown) {
  if (!valor) return { iso: null, display: null };
  const fecha = new Date(String(valor));
  if (Number.isNaN(fecha.getTime())) return { iso: null, display: null };
  const fechaTexto = new Intl.DateTimeFormat("es-AR", {
    timeZone: "America/Argentina/Buenos_Aires",
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
  }).format(fecha);
  const horaTexto = new Intl.DateTimeFormat("es-AR", {
    timeZone: "America/Argentina/Buenos_Aires",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(fecha);
  return { iso: fecha.toISOString(), display: `${fechaTexto} - ${horaTexto} hs` };
}

function normalizarUsuarioCena(usuario: any, nombre: any, email: any, uid: any) {
  const nombreNormalizado = String(nombre || usuario?.nombre || "").trim();
  const emailNormalizado = String(email || usuario?.email || usuario?.correo || "").trim();
  const uidNormalizado = String(uid || usuario?.uid || "").trim();
  const valorTexto = typeof usuario === "string" ? usuario.trim() : "";
  const display = nombreNormalizado && emailNormalizado && nombreNormalizado !== emailNormalizado
    ? `${nombreNormalizado} (${emailNormalizado})`
    : nombreNormalizado || emailNormalizado || valorTexto || uidNormalizado || "Usuario autorizado";
  return {
    nombre: nombreNormalizado || null,
    email: emailNormalizado || null,
    uid: uidNormalizado || null,
    display,
  };
}

function completarAuditoriaTarjetaCena(tarjeta: FirestoreRecord, auditoria: FirestoreRecord | null) {
  if (!auditoria) return tarjeta;
  return {
    ...tarjeta,
    fechaValidacion: tarjeta.fechaValidacion || auditoria.fechaValidacion || null,
    validadoPor: tarjeta.validadoPor || auditoria.validadoPor || null,
    validadoPorUid: tarjeta.validadoPorUid || auditoria.validadoPorUid || null,
    validadoPorEmail: tarjeta.validadoPorEmail || auditoria.validadoPorEmail || null,
    validadoPorNombre: tarjeta.validadoPorNombre || auditoria.validadoPorNombre || null,
  };
}

function presentarTarjetaCena(tarjeta: FirestoreRecord) {
  const validadoPor = tarjeta.validadoPor || null;
  const anuladaPor = tarjeta.anuladaPor || null;
  const fechaValidacion = normalizarFechaCena(tarjeta.fechaValidacion);
  const fechaAnulacion = normalizarFechaCena(tarjeta.fechaAnulacion);
  const usuarioValidacion = normalizarUsuarioCena(
    validadoPor,
    tarjeta.validadoPorNombre,
    tarjeta.validadoPorEmail,
    tarjeta.validadoPorUid
  );
  const usuarioAnulacion = normalizarUsuarioCena(
    anuladaPor,
    tarjeta.anuladaPorNombre,
    tarjeta.anuladaPorEmail,
    tarjeta.anuladaPorUid
  );
  return {
    id: tarjeta.id,
    tipo: tarjeta.tipo || null,
    numeroTarjeta: Number(tarjeta.numeroTarjeta || 0),
    numeroAcompanante: tarjeta.numeroAcompanante ?? null,
    totalAcompanantes: Number(tarjeta.totalAcompanantes || 0),
    estado: tarjeta.estado || "pendiente",
    validada: tarjeta.validada === true,
    anulada: tarjeta.anulada === true,
    fechaValidacion: fechaValidacion.iso,
    fechaValidacionIso: fechaValidacion.iso,
    fechaValidacionDisplay: fechaValidacion.display,
    validadoPor,
    validadoPorUid: usuarioValidacion.uid,
    validadoPorEmail: usuarioValidacion.email,
    validadoPorNombre: usuarioValidacion.nombre,
    validadoPorDisplay: usuarioValidacion.display,
    fechaAnulacion: fechaAnulacion.iso,
    fechaAnulacionIso: fechaAnulacion.iso,
    fechaAnulacionDisplay: fechaAnulacion.display,
    anuladaPor,
    anuladaPorUid: usuarioAnulacion.uid,
    anuladaPorEmail: usuarioAnulacion.email,
    anuladaPorNombre: usuarioAnulacion.nombre,
    anuladaPorDisplay: usuarioAnulacion.display,
    motivoAnulacion: tarjeta.motivoAnulacion || null,
    reemplazada: tarjeta.reemplazada === true,
    reemplazadaPor: tarjeta.reemplazadaPor || null,
    esReemision: tarjeta.esReemision === true,
    numeroReemision: Number(tarjeta.numeroReemision || 0),
  };
}

function presentarReservaCena(reserva: FirestoreRecord) {
  return {
    id: reserva.id,
    anio: Number(reserva.anio || 0),
    estado: reserva.estado || "activa",
    afiliado: reserva.afiliado || null,
    cantidadTarjetas: Number(reserva.cantidadTarjetas || 0),
    cantidadTitular: Number(reserva.cantidadTitular || 0),
    cantidadAcompanantes: Number(reserva.cantidadAcompanantes || 0),
  };
}

function reservaCenaActiva(reserva: FirestoreRecord): boolean {
  return String(reserva.estado || "activa") === "activa";
}

function estadoTarjetaCena(tarjeta: FirestoreRecord | null, reserva: FirestoreRecord) {
  if (!reservaCenaActiva(reserva)) return "reserva_anulada";
  if (!tarjeta) return "consulta_reserva";
  if (tarjeta.reemplazada === true) return "reemplazada";
  if (!esTarjetaVigenteCena(tarjeta)) return "anulada";
  if (tarjetaCenaAcreditada(tarjeta)) return "validada";
  return "pendiente";
}

async function construirSnapshotReservaCena(
  tarjeta: FirestoreRecord | null,
  reserva: FirestoreRecord,
  anio: string
) {
  const tarjetas = await queryFirestoreChildCollection(
    `gestion_cena/${anio}`,
    "tarjetas",
    [{ field: "reservaId", value: reserva.id }],
    500
  );

  const requiereFallbackAuditoria = tarjetas.some(
    (tarjetaActual) => tarjetaCenaAcreditada(tarjetaActual) &&
      (!tarjetaActual.fechaValidacion ||
        (!tarjetaActual.validadoPor && !tarjetaActual.validadoPorNombre && !tarjetaActual.validadoPorEmail))
  );
  const auditorias = requiereFallbackAuditoria
    ? await queryFirestoreChildCollection(`gestion_cena/${anio}`, "validaciones", [], 2000)
    : [];
  const auditoriaPorTarjeta = new Map<string, FirestoreRecord>();
  auditorias.forEach((auditoria) => {
    const tarjetaId = String(auditoria.tarjetaId || "");
    const token = String(auditoria.token || auditoria.id || "");
    if (tarjetaId) auditoriaPorTarjeta.set(tarjetaId, auditoria);
    if (token) auditoriaPorTarjeta.set(token, auditoria);
  });
  const tarjetasConAuditoria = tarjetas.map((tarjetaActual) => completarAuditoriaTarjetaCena(
    tarjetaActual,
    auditoriaPorTarjeta.get(String(tarjetaActual.id || "")) || auditoriaPorTarjeta.get(String(tarjetaActual.token || "")) || null
  ));
  const tarjetaEscaneada = tarjeta
    ? tarjetasConAuditoria.find((tarjetaActual) => tarjetaActual.id === tarjeta.id) || completarAuditoriaTarjetaCena(
      tarjeta,
      auditoriaPorTarjeta.get(String(tarjeta.id || "")) || auditoriaPorTarjeta.get(String(tarjeta.token || "")) || null
    )
    : null;
  const resumen = resumirTarjetasCena(tarjetasConAuditoria);
  const tarjetaReemplazo = tarjetaEscaneada?.reemplazadaPor
    ? tarjetasConAuditoria.find((tarjetaActual) => tarjetaActual.id === String(tarjetaEscaneada.reemplazadaPor)) || null
    : null;

  return {
    estado: estadoTarjetaCena(tarjetaEscaneada, reserva),
    puedeAcreditar:
      Boolean(tarjetaEscaneada) &&
      reservaCenaActiva(reserva) &&
      esTarjetaVigenteCena(tarjetaEscaneada) &&
      !tarjetaCenaAcreditada(tarjetaEscaneada),
    reserva: presentarReservaCena(reserva),
    tarjeta: tarjetaEscaneada ? presentarTarjetaCena(tarjetaEscaneada) : null,
    tarjetaReemplazo: tarjetaReemplazo ? presentarTarjetaCena(tarjetaReemplazo) : null,
    tarjetas: resumen.vigentes.sort(ordenarTarjetasCena).map(presentarTarjetaCena),
    tarjetasHistoricas: tarjetasConAuditoria.filter((tarjetaActual) => !esTarjetaVigenteCena(tarjetaActual)).sort(ordenarTarjetasCena).map(presentarTarjetaCena),
    resumen: {
      total: resumen.vigentes.length,
      acreditadas: resumen.acreditadas,
      pendientes: resumen.pendientes,
    },
  };
}

async function beginFirestoreTransaction(): Promise<string> {
  const respuesta = await firestoreRequest<{ transaction?: string }>(`${firestoreBaseUrl}:beginTransaction`, {
    method: "POST",
    body: JSON.stringify({}),
  });
  const transaction = String(respuesta?.transaction || "");
  if (!transaction) throw new Error("No se pudo iniciar la transacción de Firestore.");
  return transaction;
}

async function getFirestoreDocInTransaction(path: string, transaction: string) {
  const doc = await firestoreRequest<FirestoreDocument>(
    `${firestoreBaseUrl}/${path}?transaction=${encodeURIComponent(transaction)}`
  );
  return doc ? firestoreDocToJs(doc) : null;
}

async function rollbackFirestoreTransaction(transaction: string) {
  try {
    await firestoreRequest(`${firestoreBaseUrl}:rollback`, {
      method: "POST",
      body: JSON.stringify({ transaction }),
    });
  } catch {
    // La transacción expira sola; un rollback fallido no tapa el resultado real.
  }
}

function esConflictoTransaccionFirestore(error: any): boolean {
  return String(error?.firestoreStatus || "") === "ABORTED" || /\baborted\b/i.test(String(error?.message || ""));
}

async function registrarTarjetaCenaAtomica({
  tarjetaPath,
  reservaPath,
  anio,
  token,
  authUser,
  permiso,
}: {
  tarjetaPath: string;
  reservaPath: string;
  anio: string;
  token: string;
  authUser: AuthenticatedUser;
  permiso: PermisoCertificados;
}) {
  for (let intento = 0; intento < 2; intento += 1) {
    const transaction = await beginFirestoreTransaction();
    let confirmar = false;

    try {
      const [tarjeta, reserva] = await Promise.all([
        getFirestoreDocInTransaction(tarjetaPath, transaction),
        getFirestoreDocInTransaction(reservaPath, transaction),
      ]);

      if (!tarjeta || !reserva) {
        throw Object.assign(new Error("La tarjeta o su reserva ya no están disponibles."), { statusCode: 404 });
      }

      const estado = estadoTarjetaCena(tarjeta, reserva);
      if (estado !== "pendiente") {
        return { resultado: estado === "validada" ? "ya_registrada" : "no_acreditable", tarjeta, reserva };
      }

      const validadoPor = {
        uid: authUser.uid,
        email: authUser.email || "",
        nombre: permiso.tipo === "validador" ? buildNombreAfiliado(permiso.usuario) : "Administrador SIDCA",
        tipo: permiso.tipo,
      };
      const validadoPorUid = authUser.uid;
      const validadoPorEmail = authUser.email || "";
      const validadoPorNombre = validadoPor.nombre || "";
      const validacionPath = `gestion_cena/${anio}/validaciones/${token}`;
      const cuerpo = {
        transaction,
        writes: [
          {
            update: {
              name: rutaDocumentoFirestore(tarjetaPath),
              fields: jsToFirestoreFields({
                estado: "validada",
                validada: true,
                validadoPor,
                validadoPorUid,
                validadoPorEmail,
                validadoPorNombre,
              }),
            },
            // Sin esta máscara, el Commit REST reemplaza el documento completo
            // con los campos de acreditación y pierde su identidad funcional.
            updateMask: {
              fieldPaths: [
                "validada",
                "estado",
                "fechaValidacion",
                "validadoPor",
                "validadoPorUid",
                "validadoPorEmail",
                "validadoPorNombre",
              ],
            },
            updateTransforms: [{ fieldPath: "fechaValidacion", setToServerValue: "REQUEST_TIME" }],
          },
          {
            update: {
              name: rutaDocumentoFirestore(validacionPath),
              fields: jsToFirestoreFields({
                anio: Number(anio),
                tarjetaId: tarjeta.id,
                reservaId: reserva.id,
                token,
                codigoVisible: tarjeta.codigoVisible || null,
                afiliadoDni: tarjeta.afiliadoDni || reserva.afiliado?.dni || null,
                tipo: tarjeta.tipo || null,
                validadoPor,
                validadoPorUid,
                validadoPorEmail,
                validadoPorNombre,
              }),
            },
            updateTransforms: [{ fieldPath: "fechaValidacion", setToServerValue: "REQUEST_TIME" }],
          },
        ],
      };

      await firestoreRequest(`${firestoreBaseUrl}:commit`, {
        method: "POST",
        body: JSON.stringify(cuerpo),
      });
      confirmar = true;

      return {
        resultado: "registrada",
        tarjeta: { ...tarjeta, estado: "validada", validada: true, validadoPor },
        reserva,
      };
    } catch (error: any) {
      if (esConflictoTransaccionFirestore(error) && intento === 0) continue;
      throw error;
    } finally {
      if (!confirmar) await rollbackFirestoreTransaction(transaction);
    }
  }

  throw new Error("No se pudo registrar la tarjeta de Cena.");
}

function sendCenaError(res: express.Response, error: any) {
  const statusCode = [400, 401, 403, 404, 409].includes(Number(error?.statusCode))
    ? Number(error.statusCode)
    : 500;
  if (statusCode === 500) console.error("[sidca-chatbot-backend] Error validación Cena:", error);
  return res.status(statusCode).json({
    ok: false,
    modulo: "cena",
    error: statusCode === 500 ? "Error interno del servicio de validación de Cena." : String(error?.message || "No se pudo completar la operación."),
  });
}

app.get("/api/cena/validar/:token", async (req, res) => {
  try {
    const authUser = await verifyFirebaseIdToken(req.headers.authorization);
    const token = parseCenaTokenParam(req.params.token);
    const [permiso, tarjeta] = await Promise.all([
      requireAdministradorOValidadorCena(authUser),
      buscarTarjetaCenaPorToken(token),
    ]);
    const ubicacion = getUbicacionTarjetaCena(tarjeta);
    const reserva = await getFirestoreDoc(ubicacion.reservaPath);
    if (!reserva) throw Object.assign(new Error("No se encontró la reserva de esta tarjeta."), { statusCode: 404 });
    const snapshot = await construirSnapshotReservaCena(tarjeta, reserva, ubicacion.anio);
    return res.status(200).json({ ok: true, modulo: "cena", validacion: snapshot, permiso: permiso.tipo });
  } catch (error: any) {
    return sendCenaError(res, error);
  }
});

app.get("/api/cena/reserva", async (req, res) => {
  try {
    const authUser = await verifyFirebaseIdToken(req.headers.authorization);
    await requireAdministradorOValidadorCena(authUser);
    const anio = parseCenaAnioParam(req.query.anio);
    const dni = assertValidDni(normalizeDni(String(req.query.dni || "")));
    const reservas = await queryFirestoreChildCollection(
      `gestion_cena/${anio}`,
      "reservas",
      [{ field: "afiliado.dni", value: dni }],
      2
    );
    if (reservas.length === 0) {
      throw Object.assign(new Error("No se encontró una reserva de Cena para el DNI indicado."), { statusCode: 404 });
    }
    if (reservas.length > 1) {
      throw Object.assign(new Error("Hay más de una reserva para ese DNI. Consultá al administrador."), { statusCode: 409 });
    }
    const snapshot = await construirSnapshotReservaCena(null, reservas[0], anio);
    return res.status(200).json({ ok: true, modulo: "cena", consulta: "dni", validacion: snapshot });
  } catch (error: any) {
    return sendCenaError(res, error);
  }
});

app.post("/api/cena/validar/:token/registrar", async (req, res) => {
  try {
    const authUser = await verifyFirebaseIdToken(req.headers.authorization);
    const token = parseCenaTokenParam(req.params.token);
    const [permiso, tarjetaEncontrada] = await Promise.all([
      requireAdministradorOValidadorCena(authUser),
      buscarTarjetaCenaPorToken(token),
    ]);
    const ubicacion = getUbicacionTarjetaCena(tarjetaEncontrada);
    const registro = await registrarTarjetaCenaAtomica({
      tarjetaPath: ubicacion.tarjetaPath,
      reservaPath: ubicacion.reservaPath,
      anio: ubicacion.anio,
      token,
      authUser,
      permiso,
    });
    const snapshot = await construirSnapshotReservaCena(registro.tarjeta, registro.reserva, ubicacion.anio);
    return res.status(200).json({
      ok: true,
      modulo: "cena",
      resultado: registro.resultado,
      validacion: snapshot,
    });
  } catch (error: any) {
    return sendCenaError(res, error);
  }
});

app.post("/api/chatbot/query", chatbotRateLimit, async (req, res) => {
  try {
    const input = bodySchema.parse(req.body);
    const result = await runChatbotWorkflow(input);

    res.status(200).json(result);
  } catch (error: any) {
    if (error instanceof z.ZodError) {
      res.status(400).json({
        ok: false,
        tipo: "error",
        dominio: req.body?.dominio || "general",
        origen: "backend",
        consulta: req.body?.pregunta || "",
        consultaNormalizada: req.body?.pregunta || "",
        respuesta: "Solicitud inválida.",
        articulos: [],
        referencias: [],
        busqueda: [],
        conversationId: null,
        error: error.issues.map((i: any) => i.message).join(" | "),
      });
      return;
    }

    console.error("[sidca-chatbot-backend] Error:", error);

    res.status(500).json({
      ok: false,
      tipo: "error",
      dominio: req.body?.dominio || "general",
      origen: "backend",
      consulta: req.body?.pregunta || "",
      consultaNormalizada: req.body?.pregunta || "",
      respuesta: "Servicio no disponible por el momento.",
      articulos: [],
      referencias: [],
      busqueda: [],
      conversationId: null,
      error: error?.message || "Error interno",
    });
  }
});

app.post("/api/auth/firebase/bootstrap", bootstrapRateLimit, async (req, res) => {
  try {
    const input = firebaseBootstrapSchema.parse(req.body);
    const dni = assertValidDni(normalizeDni(input.dni));
    const usuarioId = input.usuarioId.trim();

    const affiliateDoc = await findAfiliadoByDni(dni);

    if (!affiliateDoc) {
      res.status(404).json({
        ok: false,
        error: "No se encontró el afiliado para el DNI indicado.",
      });
      return;
    }

    if (!docBelongsToUid(affiliateDoc, usuarioId)) {
      res.status(403).json({ ok: false, error: "El DNI no corresponde al usuario autenticado." });
      return;
    }

    const customToken = createFirebaseCustomToken(usuarioId, { dni });

    res.status(200).json({
      ok: true,
      customToken,
      uid: usuarioId,
      dni,
      afiliadoNombre: affiliateDoc ? buildNombreAfiliado(affiliateDoc) : null,
    });
  } catch (error: any) {
    const statusCode = Number(error?.statusCode || (error?.name === "ZodError" ? 400 : 500));
    console.error("[sidca-chatbot-backend] Error Firebase bootstrap:", {
      statusCode,
      message: error?.message || "Error interno",
    });
    res.status(statusCode).json({
      ok: false,
      error: error?.message || "No se pudo iniciar la sesión segura de Firebase.",
    });
  }
});
app.post("/api/pagos/mercadopago/preference", paymentRateLimit, async (req, res) => {
  try {
    const authUser = await verifyFirebaseIdToken(req.headers.authorization);
    const input = secureMercadoPagoPreferenceSchema.parse(req.body);
    const dni = assertValidDni(normalizeDni(input.dni));

    await validateDniBelongsToUser(dni, authUser.uid);

    if (input.pagoId) {
      const pagoId = input.pagoId;
      const orden = await getFirestoreDoc(`pagos_adherentes/${pagoId}`);

      if (!orden) {
        res.status(404).json({ ok: false, error: "No se encontró la orden de pago." });
        return;
      }

      assertPagoAdminValido(orden, dni);

      if (orden.uid && String(orden.uid) !== authUser.uid) {
        res.status(403).json({
          ok: false,
          error: "La orden de pago pertenece a otro usuario.",
        });
        return;
      }

      const estado = getPagoEstadoInterno(orden);
      if (
        !input.forzarNuevaPreferencia &&
        ["creada", "preferencia_creada", "pendiente", "en_proceso"].includes(estado) &&
        orden.checkoutUrl &&
        isRecentlyCreated(orden)
      ) {
        res.status(200).json({
          ok: true,
          pagoId,
          preferenceId: orden.preferenceId || orden.mercadoPagoPreferenceId || null,
          checkoutUrl: orden.checkoutUrl,
          ambiente: orden.ambiente || getMercadoPagoEnvironment(),
          reutilizada: true,
        });
        return;
      }

      const importe = Number(orden.importe);
      const moneda = String(orden.moneda || "ARS");
      const concepto = String(orden.concepto || "Pago SIDCA").trim();
      const detalle = String(orden.detalle || concepto).trim();
      const afiliadoNombre = String(orden.afiliadoNombre || "Afiliado SIDCA").trim();
      const ambiente = getMercadoPagoEnvironment();
      const externalReference = orden.externalReference || `SIDCA-PAGO-${pagoId}`;
      const backUrls = getMercadoPagoBackUrls();
      const notificationUrl = process.env.MP_WEBHOOK_URL?.trim();
      const tipoPago = inferPagoTipo(orden);
      const habilitaAdherente =
        tipoPago === "cuota_adherente" &&
        orden.habilitaAdherente !== false;
      const createdAt =
        orden.createdAt || orden.fechaCreacion || new Date();

      await updateFirestoreDoc(`pagos_adherentes/${pagoId}`, {
        pagoId,
        uid: authUser.uid,
        dni,
        afiliadoNombre,
        moneda,
        ambiente,
        tipoPago,
        habilitaAdherente,
        estadoInterno: "creada",
        estado: "pendiente",
        externalReference,
        procesado: false,
        requiereRevisionAdministrativa: Boolean(
          orden.requiereRevisionAdministrativa ||
            orden.revisionAdministrativa
        ),
        createdAt,
        fechaCreacion: orden.fechaCreacion || createdAt,
        updatedAt: new Date(),
      });

      const preferenceBody = {
        items: [
          {
            id: pagoId,
            title: concepto,
            description: detalle,
            quantity: 1,
            currency_id: moneda,
            unit_price: importe,
          },
        ],
        ...buildMercadoPagoPayer(ambiente, afiliadoNombre, dni),
        external_reference: externalReference,
        metadata: {
          pagoId,
          periodo: orden.periodo || null,
          concepto,
        },
        back_urls: backUrls,
        auto_return: "approved",
        ...(notificationUrl ? { notification_url: notificationUrl } : {}),
        statement_descriptor: "SIDCA",
      };

      const mpPreference = await createMercadoPagoPreference(preferenceBody);
      const checkoutUrl = getCheckoutUrl(mpPreference);

      await updateFirestoreDoc(`pagos_adherentes/${pagoId}`, {
        preferenceId: mpPreference.id,
        mercadoPagoPreferenceId: mpPreference.id,
        initPoint: mpPreference.init_point || null,
        sandboxInitPoint: mpPreference.sandbox_init_point || null,
        checkoutUrl,
        estadoInterno: "preferencia_creada",
        estado: "pendiente",
        updatedAt: new Date(),
      });

      await appendPagoEvento(pagoId, {
        tipo: "preferencia_creada",
        estadoAnterior: estado,
        estadoNuevo: "preferencia_creada",
        origen: "backend",
        detalle: "Preferencia creada desde orden administrativa.",
      });

      res.status(200).json({
        ok: true,
        pagoId,
        preferenceId: mpPreference.id,
        checkoutUrl,
        ambiente,
      });
      return;
    }

    const config = await getCuotaAdherenteConfig();
    const { afiliadoNombre, usuarios, nuevoAfiliado } = await getAfiliadoDocs(dni);
    const afiliadoDocs = [...usuarios, ...nuevoAfiliado];

    const alreadyActive = afiliadoDocs.some(
      (doc) => doc.adherente === true && doc.activo === true
    );

    if (alreadyActive) {
      res.status(409).json({
        ok: false,
        error: "El adherente ya figura activo. No corresponde generar otra orden.",
      });
      return;
    }

    const existingPayments = await findExistingPagoAdherente(
      authUser.uid,
      dni,
      config.periodo
    );
    const approved = existingPayments.find((payment) => payment.estadoInterno === "aprobado");

    if (approved) {
      res.status(409).json({
        ok: false,
        error: "La cuota adherente de este período ya fue abonada.",
        pago: publicPagoFields(approved),
      });
      return;
    }

    const reusable = input.forzarNuevaPreferencia
      ? undefined
      : existingPayments.find(
          (payment) =>
            ["creada", "preferencia_creada", "pendiente", "en_proceso"].includes(
              getPagoEstadoInterno(payment)
            ) &&
            payment.checkoutUrl &&
            isRecentlyCreated(payment)
        );

    if (reusable) {
      res.status(200).json({
        ok: true,
        pagoId: reusable.pagoId || reusable.id,
        preferenceId: reusable.preferenceId,
        checkoutUrl: reusable.checkoutUrl,
        ambiente: reusable.ambiente || getMercadoPagoEnvironment(),
        reutilizada: true,
      });
      return;
    }

    const pagoId = crypto.randomUUID();
    const externalReference = `SIDCA-CUOTA-${config.periodo}-${pagoId}`;
    const ambiente = getMercadoPagoEnvironment();
    const createdAt = new Date();

    await createFirestoreDoc("pagos_adherentes", pagoId, {
      pagoId,
      uid: authUser.uid,
      dni,
      afiliadoNombre,
      periodo: config.periodo,
      importe: config.importe,
      moneda: config.moneda,
      concepto: config.concepto,
      detalle: config.detalle,
      ambiente,
      tipoPago: "cuota_adherente",
      habilitaAdherente: true,
      estadoInterno: "creada",
      estado: "pendiente",
      estadoMercadoPago: null,
      externalReference,
      procesado: false,
      requiereRevisionAdministrativa: false,
      createdAt,
      fechaCreacion: createdAt,
      updatedAt: createdAt,
    });

    const backUrls = getMercadoPagoBackUrls();
    const notificationUrl = process.env.MP_WEBHOOK_URL?.trim();
    const preferenceBody = {
      items: [
        {
          id: pagoId,
          title: config.concepto,
          description: config.detalle || config.concepto,
          quantity: 1,
          currency_id: config.moneda,
          unit_price: config.importe,
        },
      ],
      ...buildMercadoPagoPayer(ambiente, afiliadoNombre, dni),
      external_reference: externalReference,
      metadata: {
        pagoId,
        periodo: config.periodo,
        concepto: "cuota_adherente",
      },
      back_urls: backUrls,
      auto_return: "approved",
      ...(notificationUrl ? { notification_url: notificationUrl } : {}),
      statement_descriptor: "SIDCA",
    };

    const mpPreference = await createMercadoPagoPreference(preferenceBody);
    const checkoutUrl = getCheckoutUrl(mpPreference);

    await updateFirestoreDoc(`pagos_adherentes/${pagoId}`, {
      preferenceId: mpPreference.id,
      mercadoPagoPreferenceId: mpPreference.id,
      initPoint: mpPreference.init_point || null,
      sandboxInitPoint: mpPreference.sandbox_init_point || null,
      checkoutUrl,
      estadoInterno: "preferencia_creada",
      estado: "pendiente",
      updatedAt: new Date(),
    });

    await appendPagoEvento(pagoId, {
      tipo: "preferencia_creada",
      estadoAnterior: "creada",
      estadoNuevo: "preferencia_creada",
      origen: "backend",
    });

    res.status(200).json({
      ok: true,
      pagoId,
      preferenceId: mpPreference.id,
      checkoutUrl,
      ambiente,
    });
  } catch (error: any) {
    if (error instanceof z.ZodError) {
      res.status(400).json({
        ok: false,
        error: error.issues.map((i: any) => i.message).join(" | "),
      });
      return;
    }

    const statusCode = Number(error?.statusCode || 500);
    console.error("[sidca-chatbot-backend] Error MP secure preference:", {
      statusCode,
      message: error?.message || "Error interno",
    });

    res.status(statusCode).json({
      ok: false,
      error: error?.message || "No se pudo preparar el pago.",
    });
  }
});

app.post("/api/pagos/mercadopago/webhook", async (req, res) => {
  try {
    const topic = String(req.query.type || req.query.topic || req.body?.type || "").toLowerCase();
    const paymentId = String(
      req.query["data.id"] || req.query.id || req.body?.data?.id || req.body?.id || ""
    ).trim();

    if (!["payment", "payments"].includes(topic) || !paymentId) {
      res.status(200).json({ ok: true, ignored: true });
      return;
    }

    verifyMercadoPagoWebhookSignature(
      paymentId,
      req.headers["x-request-id"] as string | undefined,
      req.headers["x-signature"] as string | undefined
    );

    const payment = await fetchMercadoPagoPayment(paymentId);
    const idempotencyDoc = await getFirestoreDoc(
      `pagos_mercadopago/${paymentId}`
    );

    if (
      idempotencyDoc &&
      String(idempotencyDoc.status || "") === String(payment.status || "") &&
      String(idempotencyDoc.statusDetail || "") ===
        String(payment.status_detail || "")
    ) {
      res.status(200).json({ ok: true, duplicate: true });
      return;
    }

    const externalReference = String(payment.external_reference || "");
    const internalPayments = await queryFirestoreCollection(
      "pagos_adherentes",
      [{ field: "externalReference", value: externalReference }],
      2
    );

    if (internalPayments.length > 1) {
      throw Object.assign(
        new Error("Existen varias órdenes internas para la misma referencia."),
        { statusCode: 409 }
      );
    }

    const internalPayment = internalPayments[0];

    if (!internalPayment) {
      throw Object.assign(new Error("No existe una orden interna para el pago informado."), {
        statusCode: 404,
      });
    }

    const expectedAmount = Number(internalPayment.importe);
    const actualAmount = Number(payment.transaction_amount);
    const expectedCurrency = String(internalPayment.moneda || "ARS");
    const actualCurrency = String(payment.currency_id || "");
    const expectedLiveMode =
      String(internalPayment.ambiente || "test") === "production";

    if (
      externalReference !== internalPayment.externalReference ||
      !Number.isFinite(actualAmount) ||
      Math.abs(actualAmount - expectedAmount) > 0.01 ||
      actualCurrency !== expectedCurrency ||
      Boolean(payment.live_mode) !== expectedLiveMode
    ) {
      await updateFirestoreDoc(`pagos_adherentes/${internalPayment.id}`, {
        requiereRevisionAdministrativa: true,
        updatedAt: new Date(),
      });
      throw Object.assign(
        new Error("El pago recibido no coincide con la orden interna."),
        { statusCode: 409 }
      );
    }

    const mappedStatus = mapMercadoPagoStatus(payment.status);
    const previousStatus = getPagoEstadoInterno(internalPayment);
    const fechaPago =
      payment.date_approved || internalPayment.fechaPago || null;

    await updateFirestoreDoc(`pagos_adherentes/${internalPayment.id}`, {
      estado: mappedStatus.estadoInterno,
      estadoInterno: mappedStatus.estadoInterno,
      estadoMercadoPago: payment.status || null,
      estadoMercadoPagoDetalle: payment.status_detail || null,
      mercadoPagoPaymentId: String(payment.id),
      paymentMethodId: payment.payment_method_id || null,
      paymentTypeId: payment.payment_type_id || null,
      fechaPago,
      mercadoPagoDateCreated: payment.date_created || null,
      procesado: mappedStatus.procesado,
      requiereRevisionAdministrativa: mappedStatus.revision,
      updatedAt: new Date(),
    });

    if (
      mappedStatus.estadoInterno === "aprobado" &&
      shouldActivateAdherente(internalPayment)
    ) {
      await activateAdherenteAfterPayment(
        String(internalPayment.dni),
        Number(internalPayment.periodo),
        String(payment.id),
        String(internalPayment.id)
      );
    }

    await appendPagoEvento(String(internalPayment.id), {
      tipo: "webhook_payment",
      estadoAnterior: previousStatus,
      estadoNuevo: mappedStatus.estadoInterno,
      origen: "mercado_pago_webhook",
      detalle: payment.status_detail || undefined,
    });

    await setFirestoreDoc(`pagos_mercadopago/${paymentId}`, {
      mercadoPagoPaymentId: String(payment.id),
      pagoAdherenteId: internalPayment.id,
      externalReference,
      status: payment.status || null,
      statusDetail: payment.status_detail || null,
      updatedAt: new Date(),
      createdAt: idempotencyDoc?.createdAt || new Date(),
    });

    res.status(200).json({ ok: true, estado: mappedStatus.estadoInterno });
  } catch (error: any) {
    const statusCode = Number(error?.statusCode || 500);
    console.error("[sidca-chatbot-backend] Error MP webhook:", {
      statusCode,
      message: error?.message || "Error interno",
    });
    res.status(statusCode).json({
      ok: false,
      error: error?.message || "No se pudo procesar el webhook.",
    });
  }
});

app.get("/api/pagos/mercadopago/estado/:pagoId", async (req, res) => {
  try {
    const authUser = await verifyFirebaseIdToken(req.headers.authorization);
    const reference = String(req.params.pagoId || "").trim();

    if (!/^[A-Za-z0-9_-]{1,200}$/.test(reference)) {
      res.status(400).json({
        ok: false,
        error: "El identificador del pago es inválido.",
      });
      return;
    }

    let payment = await getFirestoreDoc(`pagos_adherentes/${reference}`);

    if (!payment) {
      const matches = await queryFirestoreCollection(
        "pagos_adherentes",
        [{ field: "externalReference", value: reference }],
        2
      );

      if (matches.length > 1) {
        res.status(409).json({
          ok: false,
          error: "La referencia corresponde a más de una orden.",
        });
        return;
      }

      payment = matches[0] || null;
    }

    if (!payment) {
      res.status(404).json({ ok: false, error: "No se encontró el pago." });
      return;
    }

    if (String(payment.uid || "") !== authUser.uid) {
      res.status(403).json({ ok: false, error: "No tenés acceso a este pago." });
      return;
    }

    res.status(200).json({ ok: true, pago: publicPagoFields(payment) });
  } catch (error: any) {
    const statusCode = Number(error?.statusCode || 500);
    res.status(statusCode).json({
      ok: false,
      error: error?.message || "No se pudo consultar el pago.",
    });
  }
});

app.get("/api/pagos/mercadopago/mis-pagos", async (req, res) => {
  try {
    const authUser = await verifyFirebaseIdToken(req.headers.authorization);
    const payments = await queryFirestoreCollection(
      "pagos_adherentes",
      [{ field: "uid", value: authUser.uid }],
      100
    );

    const sorted = payments.sort((a, b) => {
      const aDate = new Date(a.createdAt || a.fechaCreacion || 0).getTime();
      const bDate = new Date(b.createdAt || b.fechaCreacion || 0).getTime();
      return bDate - aDate;
    });

    res.status(200).json({
      ok: true,
      pagos: sorted.map((payment) => publicPagoFields(payment)),
    });
  } catch (error: any) {
    const statusCode = Number(error?.statusCode || 500);
    res.status(statusCode).json({
      ok: false,
      error: error?.message || "No se pudieron consultar tus pagos.",
    });
  }
});

app.post(
  "/api/chatbot/transcribe",
  upload.single("audio"),
  async (req, res) => {
    try {
      if (!req.file) {
        res.status(400).json({
          ok: false,
          error: "No se recibió ningún archivo de audio.",
        });
        return;
      }

      const openai = getOpenAITranscriptionClient();

      const audioFile = await toFile(
        req.file.buffer,
        req.file.originalname || "audio.m4a",
        {
          type: req.file.mimetype || "audio/m4a",
        }
      );

      const transcription = await openai.audio.transcriptions.create({
        file: audioFile,
        model: process.env.OPENAI_TRANSCRIBE_MODEL || "gpt-4o-mini-transcribe",
        language: "es",
      });

      res.status(200).json({
        ok: true,
        texto: transcription.text?.trim() || "",
      });
    } catch (error: any) {
      console.error("[sidca-chatbot-backend] Error STT:", error);

      res.status(500).json({
        ok: false,
        error: error?.message || "No se pudo transcribir el audio.",
      });
    }
  }
);

// ============================================================
// DESCARGA MASIVA EN PDF
//
// El navegador sólo orquesta: pide iniciar, consulta el estado y descarga.
// La generación corre en un Cloud Run Job aparte, porque un PDF de mil
// páginas no entra en el tiempo ni en la memoria de un request HTTP.
//
// Esto NO emite certificados: sólo lee los que ya están emitidos y vigentes.
// ============================================================

/** Estados del trabajo. Los comparte con el Job y con la pantalla. */
const PDF_ESTADO_PENDIENTE = "pendiente";
const PDF_ESTADO_PROCESANDO = "procesando";
const PDF_ESTADO_COMPLETADO = "completado";
const PDF_ESTADO_ERROR = "error";

/**
 * Estados que significan "todavía trabajando".
 *
 * Incluye el vocabulario anterior —preparando, generando, finalizando, listo—
 * para que un trabajo que quedó en vuelo con la versión previa se siga
 * interpretando bien en lugar de aparecer como colgado.
 */
const PDF_ESTADOS_EN_CURSO = new Set([
  PDF_ESTADO_PENDIENTE,
  PDF_ESTADO_PROCESANDO,
  "preparando",
  "generando",
  "finalizando",
]);

const pdfTrabajoCompletado = (trabajo: FirestoreRecord | null) =>
  trabajo?.estado === PDF_ESTADO_COMPLETADO || trabajo?.estado === "listo";

/** Nombre del objeto en Storage. Determinístico y sin datos personales. */
const pdfObjectName = (cursoId: string, jobId: string) =>
  `certificados-pdf/${cursoId}/${jobId}.pdf`;

/**
 * Identificador del trabajo tal como llega en la URL.
 *
 * Se valida con el mismo formato que el resto de los identificadores del
 * módulo: termina interpolado en rutas de Firestore y de Storage, así que no
 * puede llevar barras ni puntos suspensivos.
 */
function parseTrabajoPdfIdParam(valor: unknown): string {
  const jobId = String(valor ?? "").trim();

  if (!CERTIFICADO_ID_REGEX.test(jobId)) {
    throw Object.assign(new Error("El identificador del trabajo es inválido."), {
      statusCode: 400,
    });
  }

  return jobId;
}

/** Quita del nombre de archivo lo que Windows y los navegadores no aceptan. */
const sanitizarNombreArchivo = (valor: string) =>
  String(valor || "")
    .replace(/[\\/:*?"<>|\r\n]/g, "-")
    .trim()
    .slice(0, 120) || "certificados";

// ============================================================
// DESCARGA POR SEGMENTOS GEOGRÁFICOS
//
// Un PDF con mil certificados es inmanejable. Estos endpoints permiten
// generarlo y bajarlo por región, de a un segmento por vez.
//
// La clasificación es UNA sola —clasificarSegmentosCurso—, y la comparten el
// resumen, el Excel y el Job que arma el PDF. Si cada uno tuviera la suya,
// el resumen podría decir Valle Central y el PDF salir con otra gente.
// ============================================================

/** Ruta del objeto de un segmento. Incluye el segmento para poder ubicarlo. */
const pdfSegmentoObjectName = (
  cursoId: string,
  segmentoId: string,
  jobId: string
) => `certificados-pdf/${cursoId}/${segmentoId}/${jobId}.pdf`;

function parseSegmentoIdParam(valor: unknown): string {
  const segmentoId = String(valor ?? "").trim();

  // Lista blanca: el frontend no puede pedir un segmento arbitrario, y menos
  // uno con barras que terminen escapando de la ruta de Storage.
  if (!esSegmentoValido(segmentoId)) {
    throw Object.assign(new Error("El segmento indicado no existe."), {
      statusCode: 400,
    });
  }

  return segmentoId;
}

/**
 * Clasifica a los participantes del curso por segmento.
 *
 * Fuente única para el resumen, el Excel y la selección del PDF. Resuelve el
 * padrón UNA vez —afiliación y departamento salen del mismo viaje— y después
 * clasifica en memoria: con dos mil participantes no se dispara una consulta
 * por fila ni un recorrido remoto por segmento.
 *
 * `participantes` cuenta identificados y disponibles; `descargables` cuenta
 * los que además tienen certificado vigente y afiliación habilitada. Son
 * conceptos distintos y se llevan por separado a propósito.
 */
async function clasificarSegmentosCurso(cursoId: string) {
  // resolverAprobadosCurso ya trae afiliación, departamento y la marca de
  // certificado vigente de este curso. Reutilizarlo evita repetir el padrón y
  // garantiza que el resumen coincida con la tabla de la pantalla.
  const padron = await resolverAprobadosCurso(cursoId);

  // Los apartados quedan fuera: fueron excluidos de la emisión, así que no
  // cuentan ni en el PDF ni en la planilla de control.
  //
  // Los "sin usuario asociado" tampoco entran en ningún segmento: no hay
  // afiliado sobre el cual resolver departamento. Eso es un problema distinto
  // del de "sin departamento cargado" y no se mezcla con él.
  const identificados = padron.participantes.filter(
    (participante) => participante.estado !== "sin_usuario"
  );

  const porSegmento = new Map<string, ParticipanteAprobado[]>(
    SEGMENTOS.map((segmento) => [segmento.id, []])
  );

  for (const participante of identificados) {
    const segmentoId = participante.departamento?.segmentoId || "";

    // Un segmento desconocido cae en revisión en lugar de perderse.
    const destino =
      porSegmento.get(segmentoId) ||
      porSegmento.get(SEGMENTO_SIN_DEPARTAMENTO)!;

    destino.push(participante);
  }

  const resumen = SEGMENTOS.map((segmento) => {
    const gente = porSegmento.get(segmento.id) || [];

    return {
      id: segmento.id,
      nombre: segmento.nombre,
      departamentos: segmento.departamentos,
      // Identificados y disponibles del segmento, con o sin certificado.
      participantes: gente.length,
      certificadosEmitidos: gente.filter((p) => p.certificadoEmitido).length,
      // Lo que realmente va a entrar al PDF: emitido y vigente Y con la
      // afiliación habilitada HOY.
      certificadosDescargables: gente.filter(
        (p) => p.certificadoEmitido && p.afiliacion?.habilitadoCertificado === true
      ).length,
      adherentesNoHabilitados: gente.filter(
        (p) =>
          p.afiliacion?.tipo === "adherente" &&
          p.afiliacion?.habilitadoCertificado !== true
      ).length,
    };
  });

  return { padron, porSegmento, resumen };
}

app.get(
  "/api/certificados/admin/pdf-segmentado/:cursoId/segmentos",
  async (req, res) => {
    try {
      const authUser = await verifyFirebaseIdToken(req.headers.authorization);
      await requireAdministrador(authUser);

      const cursoId = parseCursoIdParam(req.params.cursoId);
      const { resumen } = await clasificarSegmentosCurso(cursoId);

      // Los ocho van siempre, aunque estén en cero: un segmento que
      // desaparece de la lista se lee como "no existe", no como "vacío".
      return res.status(200).json({
        ok: true,
        modulo: "certificados",
        cursoId,
        segmentos: resumen,
      });
    } catch (error: any) {
      return sendCertificadosError(res, error);
    }
  }
);

/**
 * Datos para la planilla de control de un segmento.
 *
 * Incluye a TODOS los identificados del segmento, habilitados o no: el sentido
 * de la planilla es poder ver por qué alguien no entró al PDF. Por eso el
 * Excel suele tener más filas que páginas tiene el PDF.
 *
 * El backend entrega el modelo ya resuelto; el frontend sólo lo vuelca a
 * Excel. No se le pide que reinterprete afiliación ni departamento.
 */
app.get(
  "/api/certificados/admin/pdf-segmentado/:cursoId/:segmentoId/excel",
  async (req, res) => {
    try {
      const authUser = await verifyFirebaseIdToken(req.headers.authorization);
      await requireAdministrador(authUser);

      const cursoId = parseCursoIdParam(req.params.cursoId);
      const segmentoId = parseSegmentoIdParam(req.params.segmentoId);

      const { porSegmento } = await clasificarSegmentosCurso(cursoId);
      const segmento = obtenerSegmento(segmentoId)!;
      const gente = porSegmento.get(segmentoId) || [];

      const filas = gente
        .map((participante: any) => {
          const afiliacion = participante.afiliacion || {};
          const departamento = participante.departamento || {};

          const estadoAfiliado =
            afiliacion.tipo === "adherente"
              ? afiliacion.habilitadoCertificado === true
                ? "Habilitado"
                : "No habilitado"
              : afiliacion.tipo === "cotizante"
              ? "Cotizante"
              : "No verificado";

          return {
            apellidoNombre: participante.apellidoNombre || "",
            // Sólo dígitos. El frontend lo escribe como texto para que Excel
            // no lo convierta en notación científica.
            dni: String(participante.dni || "").replace(/\D/g, ""),
            adherente: afiliacion.tipo === "adherente" ? "Sí" : "No",
            estadoAfiliado,
            // Un valor no reconocido se conserva crudo: es lo que permite
            // localizarlo y corregirlo después.
            departamento:
              departamento.canonico || departamento.crudo || "Sin departamento",
          };
        })
        .sort((a: any, b: any) =>
          a.apellidoNombre.localeCompare(b.apellidoNombre, "es", {
            sensitivity: "base",
          })
        );

      return res.status(200).json({
        ok: true,
        modulo: "certificados",
        cursoId,
        segmentoId,
        segmentoNombre: segmento.nombre,
        filas,
      });
    } catch (error: any) {
      return sendCertificadosError(res, error);
    }
  }
);

/**
 * Último trabajo de un segmento.
 *
 * Los trabajos segmentados viven en la MISMA subcolección que los masivos y se
 * distinguen por el campo `segmentoId`. Así el Job de Cloud Run, el endpoint de
 * estado y el de descarga siguen leyendo un único lugar, y los trabajos
 * antiguos —que no tienen el campo— nunca se confunden con los de un segmento.
 *
 * El puntero "actual" se resuelve por consulta y no por un campo en el
 * certificado: guardar un mapa de ocho punteros obligaría a escribir rutas de
 * campo con guiones en las máscaras de actualización, que es una fuente de
 * errores silenciosos.
 */
async function ultimoTrabajoSegmento(cursoId: string, segmentoId: string) {
  const trabajos = await queryFirestoreChildCollection(
    `certificados/${cursoId}`,
    "trabajosPdf",
    [{ field: "segmentoId", value: segmentoId }],
    50
  );

  if (!trabajos.length) return null;

  // Se ordena en memoria: son unos pocos documentos por segmento y evita
  // exigir un índice compuesto sólo para esto.
  return [...trabajos].sort(
    (a, b) =>
      Date.parse(String(b.creadoEn || "")) - Date.parse(String(a.creadoEn || ""))
  )[0];
}

/** Un trabajo del curso que además pertenece a ESTE segmento. */
async function obtenerTrabajoSegmento(
  cursoId: string,
  segmentoId: string,
  jobId: string
) {
  const trabajo = await getFirestoreDoc(
    `certificados/${cursoId}/trabajosPdf/${jobId}`
  );

  // Se comprueba la pertenencia al segmento: sin esto, el jobId de Valle
  // Central serviría para descargar por la ruta de cualquier otro segmento.
  if (!trabajo || String(trabajo.segmentoId || "") !== segmentoId) {
    throw Object.assign(new Error("El trabajo indicado no existe."), {
      statusCode: 404,
    });
  }

  return trabajo;
}

app.post(
  "/api/certificados/admin/pdf-segmentado/:cursoId/:segmentoId/iniciar",
  async (req, res) => {
    try {
      const authUser = await verifyFirebaseIdToken(req.headers.authorization);
      await requireAdministrador(authUser);

      const cursoId = parseCursoIdParam(req.params.cursoId);
      const segmentoId = parseSegmentoIdParam(req.params.segmentoId);
      const segmento = obtenerSegmento(segmentoId)!;

      const certificado = await getFirestoreDoc(`certificados/${cursoId}`);

      if (!certificado) {
        throw Object.assign(
          new Error("No existe la configuración de certificado de este curso."),
          { statusCode: 404 }
        );
      }

      // Un trabajo activo por segmento. Dos clics seguidos devuelven el mismo,
      // y un segmento en curso no bloquea a los otros siete.
      const enCurso = await ultimoTrabajoSegmento(cursoId, segmentoId);

      if (enCurso && PDF_ESTADOS_EN_CURSO.has(String(enCurso.estado))) {
        return res
          .status(200)
          .json({ ok: true, trabajo: enCurso, reutilizado: true });
      }

      // Cuántos certificados va a contener realmente el PDF. Sale de la misma
      // clasificación que alimenta el resumen y el Excel, así que la barra de
      // progreso no puede contradecir a la tarjeta.
      const { resumen } = await clasificarSegmentosCurso(cursoId);
      const total =
        resumen.find((fila) => fila.id === segmentoId)?.certificadosDescargables ||
        0;

      if (!total) {
        throw Object.assign(
          new Error(
            "No hay certificados descargables en este segmento para generar el PDF."
          ),
          { statusCode: 409 }
        );
      }

      const curso = await getFirestoreDoc(`cursos/${cursoId}`);
      const tituloCurso = String(
        curso?.titulo || certificado.cursoTitulo || cursoId
      );

      const jobId = crypto.randomUUID();
      const ahora = new Date();
      const objectName = pdfSegmentoObjectName(cursoId, segmentoId, jobId);

      const trabajo = {
        jobId,
        cursoId,
        // Campos nuevos. Los existentes se conservan tal cual para que el Job,
        // la pantalla y la descarga sigan funcionando sin ramificar.
        segmentoId,
        segmentoNombre: segmento.nombre,
        estado: PDF_ESTADO_PENDIENTE,
        total,
        procesados: 0,
        porcentaje: 0,
        creadoEn: ahora,
        iniciadoEn: null,
        actualizadoEn: ahora,
        finalizadoEn: null,
        transcurridoMs: 0,
        restanteEstimadoMs: null,
        finalizacionEstimada: null,
        objectName,
        storagePath: objectName,
        tamanioBytes: 0,
        creadoPor: authUser.uid,
        error: null,
        nombreArchivo: `Certificados_${sanitizarNombreArchivo(
          segmento.nombre
        ).replace(/\s+/g, "_")}.pdf`,
        cursoTitulo: tituloCurso,
      };

      await createFirestoreDoc(
        `certificados/${cursoId}/trabajosPdf`,
        jobId,
        trabajo
      );

      const proyecto =
        process.env.GCLOUD_PROJECT || process.env.FIREBASE_PROJECT_ID || "";
      const region = process.env.CERTIFICADOS_PDF_JOB_REGION || "us-central1";
      const nombreJob = process.env.CERTIFICADOS_PDF_JOB_NAME;
      const bucket = process.env.CERTIFICADOS_PDF_BUCKET;

      try {
        if (!nombreJob)
          throw new Error("Falta configurar CERTIFICADOS_PDF_JOB_NAME.");
        if (!bucket) throw new Error("Falta configurar CERTIFICADOS_PDF_BUCKET.");

        // Mismo Job de Cloud Run que la descarga masiva. La única diferencia es
        // CERTIFICADOS_PDF_SEGMENTO_ID: cuando está, el Job filtra por segmento;
        // cuando no, se comporta como siempre.
        const cuerpoRun = {
          overrides: {
            containerOverrides: [
              {
                env: [
                  { name: "CERTIFICADOS_PDF_CURSO_ID", value: cursoId },
                  { name: "CERTIFICADOS_PDF_TRABAJO_ID", value: jobId },
                  { name: "CERTIFICADOS_PDF_JOB_ID", value: jobId },
                  { name: "CERTIFICADOS_PDF_BUCKET", value: bucket },
                  { name: "CERTIFICADOS_PDF_SEGMENTO_ID", value: segmentoId },
                ],
              },
            ],
            taskCount: 1,
          },
        };

        const urlRun = `https://run.googleapis.com/v2/projects/${proyecto}/locations/${region}/jobs/${nombreJob}:run`;

        console.log("[pdf-segmentado] Ejecutando Cloud Run Job", {
          cursoId,
          segmentoId,
          jobId,
          bucket,
          jobName: nombreJob,
          region,
          body: JSON.stringify(cuerpoRun),
        });

        const respuesta = await fetch(urlRun, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${await getGoogleAccessToken()}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify(cuerpoRun),
        });

        const textoRespuesta = await respuesta.text();

        if (!respuesta.ok) {
          throw new Error(
            `Cloud Run respondió ${respuesta.status}: ${textoRespuesta.slice(
              0,
              300
            )}`
          );
        }

        let operacion: any = null;
        try {
          operacion = JSON.parse(textoRespuesta);
        } catch {
          operacion = null;
        }

        const operationName = String(operacion?.name || "");
        const executionName = String(operacion?.metadata?.name || "");

        if (operationName || executionName) {
          await updateFirestoreDoc(
            `certificados/${cursoId}/trabajosPdf/${jobId}`,
            {
              cloudRunOperationName: operationName,
              cloudRunExecutionName: executionName,
              actualizadoEn: new Date(),
            }
          ).catch(() => undefined);
        }
      } catch (fallo: any) {
        // Sin esto el segmento queda con un trabajo "pendiente" eterno que
        // bloquea todos los intentos siguientes.
        await updateFirestoreDoc(
          `certificados/${cursoId}/trabajosPdf/${jobId}`,
          {
            estado: PDF_ESTADO_ERROR,
            error: "No se pudo iniciar la generación del PDF.",
            finalizadoEn: new Date(),
            actualizadoEn: new Date(),
          }
        ).catch(() => undefined);

        console.error(
          "[pdf-segmentado] no se pudo lanzar el Job PDF",
          fallo
        );

        throw Object.assign(
          new Error("No se pudo iniciar la generación del PDF del segmento."),
          { statusCode: 502 }
        );
      }

      console.log(
        `[pdf-segmentado] iniciado curso=${cursoId} segmento=${segmentoId} job=${jobId} certificados=${total} por=${authUser.uid}`
      );

      return res.status(202).json({ ok: true, trabajo });
    } catch (error: any) {
      return sendCertificadosError(res, error);
    }
  }
);

// `actual` se registra ANTES que `:jobId`: las dos rutas tienen la misma forma
// y Express resuelve por orden de declaración.
app.get(
  "/api/certificados/admin/pdf-segmentado/:cursoId/:segmentoId/actual",
  async (req, res) => {
    try {
      const authUser = await verifyFirebaseIdToken(req.headers.authorization);
      await requireAdministrador(authUser);

      const cursoId = parseCursoIdParam(req.params.cursoId);
      const segmentoId = parseSegmentoIdParam(req.params.segmentoId);

      const trabajo = await ultimoTrabajoSegmento(cursoId, segmentoId);

      return res.status(200).json({
        ok: true,
        trabajo: trabajo
          ? {
              ...trabajo,
              objectName: undefined,
              storagePath: undefined,
              listoParaDescargar: pdfTrabajoCompletado(trabajo),
            }
          : null,
      });
    } catch (error: any) {
      return sendCertificadosError(res, error);
    }
  }
);

app.get(
  "/api/certificados/admin/pdf-segmentado/:cursoId/:segmentoId/:jobId",
  async (req, res) => {
    try {
      const authUser = await verifyFirebaseIdToken(req.headers.authorization);
      await requireAdministrador(authUser);

      const cursoId = parseCursoIdParam(req.params.cursoId);
      const segmentoId = parseSegmentoIdParam(req.params.segmentoId);
      const jobId = parseTrabajoPdfIdParam(req.params.jobId);

      const trabajo = await obtenerTrabajoSegmento(cursoId, segmentoId, jobId);

      return res.status(200).json({
        ok: true,
        trabajo: {
          ...trabajo,
          objectName: undefined,
          storagePath: undefined,
          listoParaDescargar: pdfTrabajoCompletado(trabajo),
        },
      });
    } catch (error: any) {
      return sendCertificadosError(res, error);
    }
  }
);

app.get(
  "/api/certificados/admin/pdf-segmentado/:cursoId/:segmentoId/:jobId/descargar",
  async (req, res) => {
    try {
      const authUser = await verifyFirebaseIdToken(req.headers.authorization);
      await requireAdministrador(authUser);

      const cursoId = parseCursoIdParam(req.params.cursoId);
      const segmentoId = parseSegmentoIdParam(req.params.segmentoId);
      const jobId = parseTrabajoPdfIdParam(req.params.jobId);

      const trabajo = await obtenerTrabajoSegmento(cursoId, segmentoId, jobId);

      if (!pdfTrabajoCompletado(trabajo)) {
        throw Object.assign(new Error("El PDF todavía no está listo."), {
          statusCode: 409,
        });
      }

      // El objeto no se toma del pedido y además tiene que coincidir con la
      // ruta determinística de ESTE curso, ESTE segmento y ESTE trabajo.
      const objectName = String(trabajo.objectName || trabajo.storagePath || "");
      const esperado = pdfSegmentoObjectName(cursoId, segmentoId, jobId);

      if (objectName !== esperado) {
        console.error(
          `[pdf-segmentado] objectName inesperado curso=${cursoId} segmento=${segmentoId} job=${jobId}`
        );
        throw Object.assign(new Error("El archivo del trabajo no es válido."), {
          statusCode: 409,
        });
      }

      const bucket = process.env.CERTIFICADOS_PDF_BUCKET;

      if (!bucket) {
        throw Object.assign(new Error("La descarga no está configurada."), {
          statusCode: 500,
        });
      }

      const urlObjeto = `https://storage.googleapis.com/storage/v1/b/${encodeURIComponent(
        bucket
      )}/o/${encodeURIComponent(objectName)}?alt=media`;

      const objeto = await fetch(urlObjeto, {
        headers: { Authorization: `Bearer ${await getGoogleAccessToken()}` },
      });

      if (!objeto.ok || !objeto.body) {
        const detalle = await objeto.text().catch(() => "");

        console.error(
          `[pdf-segmentado] Storage ${objeto.status} al leer ${objectName}: ${detalle.slice(
            0,
            300
          )}`
        );

        // Cada causa con su código: un problema de permisos no puede volver a
        // disfrazarse de "todavía no está listo".
        if (objeto.status === 404) {
          throw Object.assign(
            new Error(
              "El archivo del PDF no está disponible en el almacenamiento."
            ),
            { statusCode: 409 }
          );
        }

        if (objeto.status === 401 || objeto.status === 403) {
          throw Object.assign(
            new Error("El servidor no tiene permiso para leer el PDF generado."),
            { statusCode: 500 }
          );
        }

        throw Object.assign(new Error("No se pudo leer el PDF generado."), {
          statusCode: 502,
        });
      }

      res.setHeader("Content-Type", "application/pdf");
      res.setHeader(
        "Content-Disposition",
        `attachment; filename="${sanitizarNombreArchivo(
          String(trabajo.nombreArchivo || "certificados.pdf")
        )}"`
      );
      res.setHeader("Transfer-Encoding", "chunked");
      res.flushHeaders();

      // Streaming, igual que el masivo: el PDF no pasa por la memoria del
      // backend y el bucket sigue privado.
      Readable.fromWeb(objeto.body as any)
        .on("error", (fallo) => {
          console.error("[pdf-segmentado] error al transmitir el PDF", fallo);
          if (!res.headersSent) {
            res
              .status(502)
              .json({ ok: false, error: "No se pudo leer el PDF generado." });
            return;
          }
          res.end();
        })
        .pipe(res);
    } catch (error: any) {
      return sendCertificadosError(res, error);
    }
  }
);

app.post("/api/certificados/admin/pdf-masivo/:cursoId/iniciar", async (req, res) => {
  try {
    const authUser = await verifyFirebaseIdToken(req.headers.authorization);
    await requireAdministrador(authUser);

    // parseCursoIdParam y no un String() suelto: el cursoId se interpola en
    // rutas de Firestore y de Storage.
    const cursoId = parseCursoIdParam(req.params.cursoId);

    const certificado = await getFirestoreDoc(`certificados/${cursoId}`);

    if (!certificado) {
      throw Object.assign(
        new Error("No existe la configuración de certificado de este curso."),
        { statusCode: 404 }
      );
    }

    // Un solo trabajo activo por curso: dos clics seguidos no generan dos
    // PDFs iguales, se devuelve el que ya está corriendo.
    const jobActual = String(certificado.pdfMasivoJobActual || "").trim();

    if (jobActual) {
      const enCurso = await getFirestoreDoc(
        `certificados/${cursoId}/trabajosPdf/${jobActual}`
      );

      if (enCurso && PDF_ESTADOS_EN_CURSO.has(String(enCurso.estado))) {
        return res.status(200).json({ ok: true, trabajo: enCurso, reutilizado: true });
      }
    }

    // Aislamiento por curso: sólo la subcolección de ESTE curso, sólo
    // vigentes. Nunca un collectionGroup.
    const emitidos = await queryFirestoreChildCollection(
      `certificados/${cursoId}`,
      "emitidos",
      [{ field: "estado", value: EMITIDOS_ESTADO_VIGENTE }],
      APROBADOS_MAX_RESULTADOS
    );

    if (!emitidos.length) {
      throw Object.assign(
        new Error("No hay certificados vigentes emitidos para generar el PDF."),
        { statusCode: 409 }
      );
    }

    // El título real sale del curso académico. Si no está, se cae al de la
    // configuración; el cursoId es el último recurso.
    const curso = await getFirestoreDoc(`cursos/${cursoId}`);
    const tituloCurso = String(
      curso?.titulo || certificado.cursoTitulo || cursoId
    );

    const jobId = crypto.randomUUID();
    const ahora = new Date();

    const trabajo = {
      jobId,
      cursoId,
      estado: PDF_ESTADO_PENDIENTE,
      total: emitidos.length,
      procesados: 0,
      porcentaje: 0,
      creadoEn: ahora,
      iniciadoEn: null,
      actualizadoEn: ahora,
      finalizadoEn: null,
      transcurridoMs: 0,
      restanteEstimadoMs: null,
      finalizacionEstimada: null,
      objectName: pdfObjectName(cursoId, jobId),
      storagePath: pdfObjectName(cursoId, jobId),
      tamanioBytes: 0,
      creadoPor: authUser.uid,
      error: null,
      nombreArchivo: `Certificados - ${sanitizarNombreArchivo(tituloCurso)}.pdf`,
    };

    await createFirestoreDoc(`certificados/${cursoId}/trabajosPdf`, jobId, trabajo);
    await updateFirestoreDoc(`certificados/${cursoId}`, { pdfMasivoJobActual: jobId });

    const proyecto =
      process.env.GCLOUD_PROJECT || process.env.FIREBASE_PROJECT_ID || "";
    const region = process.env.CERTIFICADOS_PDF_JOB_REGION || "us-central1";
    const nombreJob = process.env.CERTIFICADOS_PDF_JOB_NAME;
    const bucket = process.env.CERTIFICADOS_PDF_BUCKET;

    try {
      if (!nombreJob) throw new Error("Falta configurar CERTIFICADOS_PDF_JOB_NAME.");
      if (!bucket) throw new Error("Falta configurar CERTIFICADOS_PDF_BUCKET.");

      // Overrides de ESTA ejecución solamente: el Job queda igual para la
      // siguiente. Nunca se reconfigura el recurso por cada pedido.
      //
      // La forma importa: `overrides` en la raíz, `containerOverrides` como
      // arreglo dentro suyo y `taskCount` hermano de containerOverrides. Si
      // algo de eso se anida mal, Cloud Run acepta el pedido igual —los
      // campos desconocidos se descartan— y la ejecución arranca SIN las
      // variables, que es exactamente el síntoma a diagnosticar.
      const cuerpoRun = {
        overrides: {
          containerOverrides: [
            {
              env: [
                { name: "CERTIFICADOS_PDF_CURSO_ID", value: cursoId },
                { name: "CERTIFICADOS_PDF_TRABAJO_ID", value: jobId },
                // Se mantiene el nombre anterior por si la imagen desplegada
                // todavía no estuviera actualizada.
                { name: "CERTIFICADOS_PDF_JOB_ID", value: jobId },
                { name: "CERTIFICADOS_PDF_BUCKET", value: bucket },
              ],
            },
          ],
          taskCount: 1,
        },
      };

      const urlRun = `https://run.googleapis.com/v2/projects/${proyecto}/locations/${region}/jobs/${nombreJob}:run`;

      // Log de diagnóstico. Deliberadamente NO incluye el header Authorization
      // ni el access token: sólo el JSON que realmente sale.
      console.log("[certificados-pdf] Ejecutando Cloud Run Job", {
        cursoId,
        jobId,
        bucket,
        jobName: nombreJob,
        region,
        url: urlRun,
        body: JSON.stringify(cuerpoRun),
      });

      const respuesta = await fetch(urlRun, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${await getGoogleAccessToken()}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(cuerpoRun),
      });

      // Se lee como texto antes de interpretar: si Cloud Run devuelve un error
      // en HTML o vacío, un .json() directo enmascararía la causa real.
      const textoRespuesta = await respuesta.text();

      if (!respuesta.ok) {
        throw new Error(
          `Cloud Run respondió ${respuesta.status}: ${textoRespuesta.slice(0, 300)}`
        );
      }

      // :run devuelve una Operation de larga duración, no la Execution. El
      // nombre de la ejecución, cuando viene, está en metadata.name.
      let operacion: any = null;
      try {
        operacion = JSON.parse(textoRespuesta);
      } catch {
        operacion = null;
      }

      const operationName = String(operacion?.name || "");
      const executionName = String(operacion?.metadata?.name || "");

      console.log("[certificados-pdf] Cloud Run aceptó la ejecución", {
        operationName,
        executionName,
      });

      // Se guarda para poder correlacionar el trabajo con la ejecución al
      // depurar. Campo nuevo: no rompe nada de lo que ya lee la pantalla.
      if (operationName || executionName) {
        await updateFirestoreDoc(
          `certificados/${cursoId}/trabajosPdf/${jobId}`,
          {
            cloudRunOperationName: operationName,
            cloudRunExecutionName: executionName,
            actualizadoEn: new Date(),
          }
        ).catch(() => undefined);
      }
    } catch (fallo: any) {
      // El documento ya existe y el curso ya lo apunta como actual: si no se
      // marca el error, el curso queda con un trabajo "pendiente" eterno que
      // bloquea todos los intentos siguientes.
      await updateFirestoreDoc(`certificados/${cursoId}/trabajosPdf/${jobId}`, {
        estado: PDF_ESTADO_ERROR,
        error: "No se pudo iniciar la generación del PDF.",
        finalizadoEn: new Date(),
        actualizadoEn: new Date(),
      }).catch(() => undefined);

      console.error("[sidca-chatbot-backend] no se pudo lanzar el Job PDF", fallo);

      throw Object.assign(
        new Error("No se pudo iniciar la generación del PDF masivo."),
        { statusCode: 502 }
      );
    }

    console.log(
      `[sidca-chatbot-backend] pdf masivo iniciado curso=${cursoId} job=${jobId} certificados=${emitidos.length} por=${authUser.uid}`
    );

    return res.status(202).json({ ok: true, trabajo });
  } catch (error: any) {
    return sendCertificadosError(res, error);
  }
});

/**
 * Trabajo vigente del curso. Es lo que permite retomar el progreso después de
 * un F5, de cerrar el navegador o de perder la conexión: la generación vive en
 * Firestore, no en la pestaña.
 */
app.get("/api/certificados/admin/pdf-masivo/:cursoId/actual", async (req, res) => {
  try {
    const authUser = await verifyFirebaseIdToken(req.headers.authorization);
    await requireAdministrador(authUser);

    const cursoId = parseCursoIdParam(req.params.cursoId);
    const certificado = await getFirestoreDoc(`certificados/${cursoId}`);
    const jobId = String(certificado?.pdfMasivoJobActual || "").trim();

    const trabajo = jobId
      ? await getFirestoreDoc(`certificados/${cursoId}/trabajosPdf/${jobId}`)
      : null;

    return res.status(200).json({
      ok: true,
      trabajo: trabajo
        ? { ...trabajo, listoParaDescargar: pdfTrabajoCompletado(trabajo) }
        : null,
    });
  } catch (error: any) {
    return sendCertificadosError(res, error);
  }
});

app.get("/api/certificados/admin/pdf-masivo/:cursoId/:jobId", async (req, res) => {
  try {
    const authUser = await verifyFirebaseIdToken(req.headers.authorization);
    await requireAdministrador(authUser);

    const cursoId = parseCursoIdParam(req.params.cursoId);
    const jobId = parseTrabajoPdfIdParam(req.params.jobId);

    const trabajo = await getFirestoreDoc(
      `certificados/${cursoId}/trabajosPdf/${jobId}`
    );

    if (!trabajo) {
      throw Object.assign(new Error("El trabajo indicado no existe."), {
        statusCode: 404,
      });
    }

    return res.status(200).json({
      ok: true,
      // La ruta del objeto no viaja: el navegador no la necesita y el endpoint
      // de descarga la lee del documento, no del pedido.
      trabajo: {
        ...trabajo,
        objectName: undefined,
        storagePath: undefined,
        listoParaDescargar: pdfTrabajoCompletado(trabajo),
      },
    });
  } catch (error: any) {
    return sendCertificadosError(res, error);
  }
});

app.get("/api/certificados/admin/pdf-masivo/:cursoId/:jobId/descargar", async (req, res) => {
  try {
    const authUser = await verifyFirebaseIdToken(req.headers.authorization);
    await requireAdministrador(authUser);

    const cursoId = parseCursoIdParam(req.params.cursoId);
    const jobId = parseTrabajoPdfIdParam(req.params.jobId);

    const trabajo = await getFirestoreDoc(
      `certificados/${cursoId}/trabajosPdf/${jobId}`
    );

    if (!trabajo) {
      throw Object.assign(new Error("El trabajo indicado no existe."), {
        statusCode: 404,
      });
    }

    if (!pdfTrabajoCompletado(trabajo)) {
      throw Object.assign(new Error("El PDF todavía no está listo."), {
        statusCode: 409,
      });
    }

    // El objeto NUNCA se toma del pedido. Se lee del documento del trabajo y,
    // además, se exige que coincida con la ruta determinística de ESTE curso y
    // ESTE trabajo: aunque alguien lograra escribir el documento, no podría
    // hacer que el endpoint sirva un archivo arbitrario del bucket.
    const objectName = String(trabajo.objectName || trabajo.storagePath || "");
    const esperado = pdfObjectName(cursoId, jobId);

    if (objectName !== esperado) {
      console.error(
        `[sidca-chatbot-backend] objectName inesperado curso=${cursoId} job=${jobId}`
      );
      throw Object.assign(new Error("El archivo del trabajo no es válido."), {
        statusCode: 409,
      });
    }

    const bucket = process.env.CERTIFICADOS_PDF_BUCKET;

    if (!bucket) {
      throw Object.assign(new Error("La descarga no está configurada."), {
        statusCode: 500,
      });
    }

    console.log("[certificados-pdf] Descarga", {
      cursoId,
      jobId,
      estado: trabajo.estado,
      objectName: trabajo.objectName,
      tamanioBytes: trabajo.tamanioBytes,
      esperado,
    });

    // Se lee por la API JSON de Storage y NO con `new Storage()`.
    //
    // El SDK resuelve credenciales por ADC, que en Cloud Run funciona pero en
    // una máquina de desarrollo exige `gcloud auth application-default login`:
    // sin eso falla con "Could not load the default credentials" antes de
    // emitir un solo pedido. getGoogleAccessToken() es el helper que ya usa
    // todo el backend y cubre los dos casos —GOOGLE_OAUTH_ACCESS_TOKEN en
    // local, Metadata Server en producción—, así que la descarga se comporta
    // igual en ambos entornos y con la misma identidad de siempre.
    const urlObjeto = `https://storage.googleapis.com/storage/v1/b/${encodeURIComponent(
      bucket
    )}/o/${encodeURIComponent(objectName)}?alt=media`;

    console.log("[certificados-pdf] Storage descarga", { bucket, objectName });

    const objeto = await fetch(urlObjeto, {
      headers: { Authorization: `Bearer ${await getGoogleAccessToken()}` },
    });

    if (!objeto.ok || !objeto.body) {
      const detalle = await objeto.text().catch(() => "");

      console.error(
        `[sidca-chatbot-backend] Storage ${objeto.status} al leer ${objectName}: ${detalle.slice(0, 300)}`
      );

      // Cada causa con su código: un problema de permisos no puede volver a
      // disfrazarse de "todavía no está listo".
      if (objeto.status === 404) {
        throw Object.assign(
          new Error("El archivo del PDF no está disponible en el almacenamiento."),
          { statusCode: 409 }
        );
      }

      if (objeto.status === 401 || objeto.status === 403) {
        throw Object.assign(
          new Error("El servidor no tiene permiso para leer el PDF generado."),
          { statusCode: 500 }
        );
      }

      throw Object.assign(new Error("No se pudo leer el PDF generado."), {
        statusCode: 502,
      });
    }

    res.setHeader("Content-Type", "application/pdf");
    res.setHeader(
      "Content-Disposition",
      `attachment; filename="${sanitizarNombreArchivo(
        String(trabajo.nombreArchivo || "certificados.pdf")
      )}"`
    );
    res.setHeader("Transfer-Encoding", "chunked");
    res.flushHeaders();

    // Streaming: el PDF no pasa por la memoria del backend. Nada de download()
    // ni de Buffer. El bucket sigue privado; esta es la única puerta y está
    // detrás del ID Token administrativo.
    Readable.fromWeb(objeto.body as any)
      .on("error", (fallo) => {
        console.error("[sidca-chatbot-backend] error al transmitir el PDF", fallo);
        // Con el cuerpo ya empezado no se puede cambiar el código de estado:
        // sólo queda cortar.
        if (!res.headersSent) {
          res.status(502).json({ ok: false, error: "No se pudo leer el PDF generado." });
          return;
        }
        res.end();
      })
      .pipe(res);
  } catch (error: any) {
    return sendCertificadosError(res, error);
  }
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`[sidca-chatbot-backend] running on http://0.0.0.0:${PORT}`);
  console.log("[sidca-chatbot-backend] chatbot mode: groq_rag");
});
