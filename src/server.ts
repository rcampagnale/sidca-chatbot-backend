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
    methods: ["GET", "POST", "OPTIONS"],
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