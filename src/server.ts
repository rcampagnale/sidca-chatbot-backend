import "dotenv/config";
import crypto from "node:crypto";
import express from "express";
import cors from "cors";
import multer from "multer";
import OpenAI, { toFile } from "openai";
import { z } from "zod";
import { createRemoteJWKSet, jwtVerify } from "jose";
import { runChatbotWorkflow } from "./openaiWorkflow.js";

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
    throw new Error(`Firestore ${response.status}: ${detail}`);
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
  const direct = await getFirestoreDoc(`usuarios/${uid}`);
  if (direct) return direct;

  const uidFields = ["uid", "usuarioId", "userId", "authUid"];

  for (const field of uidFields) {
    const matches = await queryFirestoreCollection(
      "usuarios",
      [{ field, value: uid }],
      2
    );

    if (matches.length > 0) {
      return matches[0];
    }
  }

  return null;
}

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
  const usuario = await findUsuarioByAuthUid(authUser.uid);

  if (!usuario) {
    throw Object.assign(
      new Error("El usuario autenticado no está registrado en SIDCA."),
      { statusCode: 403 }
    );
  }

  if (usuario.validarCertificados !== true) {
    throw Object.assign(
      new Error("No tenés autorización para validar certificados SIDCA."),
      { statusCode: 403 }
    );
  }

  return usuario;
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
const CERTIFICADOS_STATUS_CODES = new Set([400, 401, 403, 404, 409]);

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
 * Firma que se imprime en el certificado.
 *
 * Las imágenes se alojan en Cloudinary (el proyecto migró desde Firebase
 * Storage por límite de cuota), por eso se guarda imagenPublicId y no
 * storagePath. Durante el borrador la imagen puede faltar todavía.
 */
const firmaCertificadoSchema = z.strictObject({
  nombre: z.string().trim().min(1, "El nombre de la firma es obligatorio.").max(160),
  cargo: z.string().trim().min(1, "El cargo de la firma es obligatorio.").max(200),
  imagenUrl: z.string().trim().max(1000).optional().default(""),
  imagenPublicId: z.string().trim().max(300).optional().default(""),
  proveedor: z.string().trim().max(40).optional().default("cloudinary"),
  orden: z.number().int().positive().max(50),
});

/**
 * Cuerpo aceptado por PUT.
 *
 * Es estricto a propósito: el cliente no puede inyectar campos arbitrarios.
 * cursoId, cursoTitulo, estadoConfiguracion y la auditoría NO se aceptan
 * desde el body; los resuelve el backend.
 */
const configuracionCertificadoSchema = z.strictObject({
  titulo: z.string().trim().min(1, "El título del certificado es obligatorio.").max(300),
  resolucion: z.string().trim().min(1, "La resolución es obligatoria.").max(200),
  cargaHoraria: z.string().trim().min(1, "La carga horaria es obligatoria.").max(100),
  dias: z.string().trim().min(1, "Las fechas de realización son obligatorias.").max(300),
  fecha: z.string().trim().min(1, "La fecha del certificado es obligatoria.").max(200),
  modalidad: z.string().trim().min(1, "La modalidad es obligatoria.").max(120),
  firmas: z.array(firmaCertificadoSchema).max(10).optional().default([]),
});

const ESTADOS_CONFIGURACION_CERTIFICADO = new Set(["borrador", "lista"]);

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
    firmas: Array.isArray(record.firmas) ? record.firmas : [],
    estadoConfiguracion: record.estadoConfiguracion || "borrador",
    creadoEn: record.creadoEn || null,
    actualizadoEn: record.actualizadoEn || null,
    creadoPor: record.creadoPor || null,
    actualizadoPor: record.actualizadoPor || null,
  };
}

/**
 * Ordena las firmas y reasigna orden 1..n.
 * Así el orden guardado es siempre consistente aunque el cliente envíe
 * valores repetidos o con huecos.
 */
function normalizarFirmasCertificado(
  firmas: z.infer<typeof firmaCertificadoSchema>[]
): Record<string, any>[] {
  return [...firmas]
    .sort((a, b) => a.orden - b.orden)
    .map((firma, indice) => ({
      nombre: firma.nombre,
      cargo: firma.cargo,
      imagenUrl: firma.imagenUrl || "",
      imagenPublicId: firma.imagenPublicId || "",
      proveedor: firma.proveedor || "cloudinary",
      orden: indice + 1,
    }));
}

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

      firmas: normalizarFirmasCertificado(datosValidados.firmas || []),

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
const USUARIOS_LOTE_CONCURRENTE = 20;

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

/** Ejecuta tareas en lotes para no abrir cientos de conexiones a la vez. */
async function resolverEnLotes<T, R>(
  items: T[],
  tamanoLote: number,
  tarea: (item: T) => Promise<R>
): Promise<R[]> {
  const salida: R[] = [];

  for (let i = 0; i < items.length; i += tamanoLote) {
    const lote = items.slice(i, i + tamanoLote);
    const resueltos = await Promise.all(lote.map((item) => tarea(item)));
    salida.push(...resueltos);
  }

  return salida;
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
  dni: string;
  apellidoNombre: string;
  apellido?: string;
  nombre?: string;
  estado: "aprobado" | "datos_incompletos" | "sin_usuario";
  aprobaciones: number;
  apartado?: boolean;
};

/** Proyecta un usuario resuelto a la forma común del participante. */
function construirParticipanteAprobado(
  usuarioDocId: string,
  usuario: FirestoreRecord | null,
  aprobaciones: number,
  apartado = false
): ParticipanteAprobado {
  const comun = {
    usuarioDocId,
    aprobaciones,
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
 * Resuelve los documentos de usuario de un conjunto de aprobados y devuelve la
 * lista ordenada por apellido y nombre.
 *
 * Se usa igual para los disponibles y para los apartados: la única diferencia
 * es la marca `apartado`.
 */
async function resolverParticipantesAprobados(
  porUsuario: Map<string, number>,
  apartado = false
): Promise<ParticipanteAprobado[]> {
  const resueltos = await resolverEnLotes(
    [...porUsuario.keys()],
    USUARIOS_LOTE_CONCURRENTE,
    async (usuarioDocId) => {
      const usuario = await getFirestoreDoc(`usuarios/${usuarioDocId}`);
      return { usuarioDocId, usuario };
    }
  );

  return resueltos
    .map(({ usuarioDocId, usuario }) =>
      construirParticipanteAprobado(
        usuarioDocId,
        usuario,
        porUsuario.get(usuarioDocId) || 1,
        apartado
      )
    )
    .sort((a, b) =>
      a.apellidoNombre.localeCompare(b.apellidoNombre, "es", {
        sensitivity: "base",
      })
    );
}

app.get("/api/certificados/admin/aprobados/:cursoId", async (req, res) => {
  try {
    const authUser = await verifyFirebaseIdToken(req.headers.authorization);
    await requireAdministrador(authUser);

    const cursoId = parseCursoIdParam(req.params.cursoId);

    const curso = await getFirestoreDoc(`cursos/${cursoId}`);

    if (!curso) {
      throw Object.assign(new Error("El curso indicado no existe."), {
        statusCode: 404,
      });
    }

    // Exclusiones administrativas de la emisión. Viven en el documento del
    // certificado: la aprobación original en usuarios/{id}/cursos NUNCA se
    // toca, sólo se omite de esta respuesta.
    const certificado = await getFirestoreDoc(`certificados/${cursoId}`);
    const usuariosExcluidos = new Set(
      (Array.isArray(certificado?.usuariosExcluidos)
        ? certificado.usuariosExcluidos
        : []
      ).map((valor: any) => String(valor || "").trim())
    );

    // Un solo filtro de igualdad: lo resuelve el índice de campo único que
    // Firestore mantiene automáticamente. "aprobo" se evalúa en memoria para
    // no exigir un índice compuesto de ámbito COLLECTION_GROUP y, de paso,
    // poder contar los registros que no están aprobados.
    const aprobaciones = await queryFirestoreCollectionGroup(
      "cursos",
      [{ field: "curso", value: `cursos/${cursoId}` }],
      APROBADOS_MAX_RESULTADOS
    );

    const documentosAprobacion = aprobaciones.length;
    const truncado = documentosAprobacion >= APROBADOS_MAX_RESULTADOS;

    let rutasInesperadas = 0;
    let noAprobados = 0;
    let duplicados = 0;

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
        duplicados += 1;
      }
    }

    const [participantes, participantesExcluidos] = await Promise.all([
      resolverParticipantesAprobados(porUsuario),
      resolverParticipantesAprobados(porUsuarioExcluido, true),
    ]);

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

    return res.status(200).json({
      ok: true,
      modulo: "certificados",
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

    // 4. Participante excluido por decisión administrativa.
    if (leerUsuariosExcluidos(certificado).includes(usuarioDocId)) {
      throw Object.assign(
        new Error("Este participante fue excluido de la emisión de certificados."),
        { statusCode: 409 }
      );
    }

    // 5. Aprobación real. No se confía en que la UI lo haya mostrado: se
    //    releen las aprobaciones del curso y se busca la de este usuario.
    //    Con una válida alcanza; los duplicados del importador no emiten de
    //    más porque acá sólo se comprueba existencia.
    const aprobaciones = await queryFirestoreCollectionGroup(
      "cursos",
      [{ field: "curso", value: `cursos/${cursoId}` }],
      APROBADOS_MAX_RESULTADOS
    );

    const tieneAprobacion = aprobaciones.some(
      (aprobacion) =>
        aprobacion.aprobo === true &&
        extraerUsuarioDocIdDeAprobacion(aprobacion._name) === usuarioDocId
    );

    if (!tieneAprobacion) {
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
        { statusCode: 409 }
      );
    }

    // 9. Token y URL de validación.
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

    const certificadoSnapshot = {
      cursoTitulo: String(curso.titulo || certificado.cursoTitulo || ""),
      titulo: String(certificado.titulo || ""),
      resolucion: String(certificado.resolucion || ""),
      cargaHoraria: String(certificado.cargaHoraria || ""),
      dias: String(certificado.dias || ""),
      fecha: String(certificado.fecha || ""),
      modalidad: String(certificado.modalidad || ""),
      firmas: Array.isArray(certificado.firmas) ? certificado.firmas : [],
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

    // createFirestoreDoc y no setFirestoreDoc: si el documento ya existiera,
    // debe fallar en vez de sobrescribir un certificado ajeno.
    await createFirestoreDoc(`certificados/${cursoId}/emitidos`, token, datos);

    console.log(
      `[sidca-chatbot-backend] certificado emitido curso=${cursoId} usuario=${usuarioDocId} token=${token} por=${authUser.uid}`
    );

    return res.status(201).json({
      ok: true,
      modulo: "certificados",
      creado: true,
      emision: {
        certificadoId: token,
        token,
        cursoId,
        usuarioDocId,
        estado: EMITIDOS_ESTADO_VIGENTE,
        participante,
        certificado: certificadoSnapshot,
        urlValidacion,
        emitidoEn: ahora.toISOString(),
      },
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
    const authUser = await verifyFirebaseIdToken(req.headers.authorization);
    const permiso = await requireAdministradorOValidadorCertificados(authUser);

    const cursoId = parseCursoIdParam(req.params.cursoId);
    const token = parseCertificadoTokenParam(req.params.token);

    // El token ES el ID del documento: lectura directa, sin query.
    const emision = await getFirestoreDoc(
      `certificados/${cursoId}/emitidos/${token}`
    );

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
      },
    });
  } catch (error: any) {
    return sendCertificadosError(res, error);
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

app.listen(PORT, "0.0.0.0", () => {
  console.log(`[sidca-chatbot-backend] running on http://0.0.0.0:${PORT}`);
  console.log("[sidca-chatbot-backend] chatbot mode: groq_rag");
});