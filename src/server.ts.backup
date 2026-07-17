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
  pagoId: z.string().trim().min(1, "El identificador de la orden es obligatorio").optional(),
});

const firebaseBootstrapSchema = z.object({
  dni: z.string().trim().min(5, "El DNI es obligatorio"),
  usuarioId: z.string().trim().min(1, "El usuario autenticado es obligatorio"),
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
      "Falta OPENAI_API_KEY. La consulta del chatbot usa Groq, pero la transcripciÃ³n de audio todavÃ­a requiere OpenAI."
    );
  }

  return new OpenAI({ apiKey });
}

function getMercadoPagoAccessToken(): string {
  const accessToken =
    process.env.MP_ACCESS_TOKEN_TEST?.trim() ||
    process.env.MP_ACCESS_TOKEN?.trim();

  if (!accessToken) {
    throw new Error(
      "Falta configurar MP_ACCESS_TOKEN_TEST o MP_ACCESS_TOKEN en el backend."
    );
  }

  return accessToken;
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

async function getGoogleAccessToken(): Promise<string> {
  const explicitToken = process.env.GOOGLE_OAUTH_ACCESS_TOKEN?.trim();
  if (explicitToken) return explicitToken;

  const metadataResponse = await fetch(
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
    {
      headers: {
        "Metadata-Flavor": "Google",
      },
    }
  );

  if (!metadataResponse.ok) {
    throw new Error(
      "No se pudo obtener token de Google para Firestore. ConfigurÃ¡ GOOGLE_OAUTH_ACCESS_TOKEN en local o ejecutÃ¡ en Cloud Run con service account."
    );
  }

  const data = await metadataResponse.json();
  if (!data?.access_token) {
    throw new Error("La metadata de Google no devolviÃ³ access_token para Firestore.");
  }

  return data.access_token;
}

async function firestoreRequest<T>(
  url: string,
  init: RequestInit = {}
): Promise<T | null> {
  const accessToken = await getGoogleAccessToken();
  const response = await fetch(url, {
    ...init,
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
    throw Object.assign(new Error("Token Firebase invÃ¡lido: falta UID."), {
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
      new Error("No se encontrÃ³ vÃ­nculo entre este DNI y el usuario autenticado."),
      { statusCode: 403 }
    );
  }
}

async function getAfiliadoDocs(dni: string) {
  const usuarios = await findDocsByDni("usuarios", dni);
  const nuevoAfiliado = await findDocsByDni("nuevoAfiliado", dni);
  const source = usuarios[0] || nuevoAfiliado[0];

  if (!source) {
    throw Object.assign(new Error("No se encontrÃ³ el afiliado para el DNI indicado."), {
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
      new Error("No existe la configuraciÃ³n config/cuotaAdherente."),
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
    throw Object.assign(new Error("El pago de cuota adherente no estÃ¡ habilitado."), {
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
      new Error("La configuraciÃ³n de cuota adherente estÃ¡ incompleta o invÃ¡lida."),
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
  return String(payment.estadoInterno || payment.estado || "pendiente").toLowerCase();
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
  const ambiente = process.env.MP_ENV?.trim() || "test";
  if (ambiente === "test" && data?.sandbox_init_point) return data.sandbox_init_point;
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

  const manifest = `id:${paymentId};request-id:${requestId};ts:${ts};`;
  const expected = crypto.createHmac("sha256", secret).update(manifest).digest("hex");
  const expectedBuffer = Buffer.from(expected, "hex");
  const receivedBuffer = Buffer.from(v1, "hex");

  if (
    expectedBuffer.length !== receivedBuffer.length ||
    !crypto.timingSafeEqual(expectedBuffer, receivedBuffer)
  ) {
    throw Object.assign(new Error("Firma de webhook Mercado Pago invÃ¡lida."), {
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
  return {
    pagoId: payment.pagoId || payment.id,
    dni: payment.dni,
    afiliadoNombre: payment.afiliadoNombre,
    periodo: payment.periodo,
    importe: payment.importe,
    moneda: payment.moneda,
    concepto: payment.concepto,
    detalle: payment.detalle,
    estadoInterno: payment.estadoInterno,
    estadoMercadoPago: payment.estadoMercadoPago,
    fechaCreacion: payment.createdAt,
    fechaPago: payment.fechaPago,
    comprobante:
      payment.estadoInterno === "aprobado"
        ? {
            leyenda: "Comprobante de pago. No vÃ¡lido como factura.",
            mercadoPagoPaymentId: payment.mercadoPagoPaymentId,
            comprobanteUrl: payment.comprobanteUrl || null,
          }
        : null,
  };
}

app.use(cors());
app.use(express.json({ limit: "1mb" }));

app.get("/health", (_req, res) => {
  res.status(200).json({
    ok: true,
    service: "sidca-chatbot-backend",
    status: "running",
    timestamp: new Date().toISOString(),
  });
});

app.post("/api/chatbot/query", async (req, res) => {
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
        respuesta: "Solicitud invÃ¡lida.",
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

app.post("/api/auth/firebase/bootstrap", async (req, res) => {
  try {
    const input = firebaseBootstrapSchema.parse(req.body);
    const dni = normalizeDni(input.dni);
    const usuarioId = input.usuarioId.trim();

    if (!dni) {
      res.status(400).json({ ok: false, error: "DNI inválido." });
      return;
    }

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
app.post("/api/pagos/mercadopago/preference", async (req, res) => {
  try {
    const authUser = await verifyFirebaseIdToken(req.headers.authorization);
    const input = secureMercadoPagoPreferenceSchema.parse(req.body);
    const dni = normalizeDni(input.dni);

    if (input.pagoId) {
      const pagoId = input.pagoId;
      const orden = await getFirestoreDoc(`pagos_adherentes/${pagoId}`);

      if (!orden) {
        res.status(404).json({ ok: false, error: "No se encontró la orden de pago." });
        return;
      }

      assertPagoAdminValido(orden, dni);

      const estado = getPagoEstadoInterno(orden);
      if (
        ["creada", "preferencia_creada", "pendiente", "en_proceso"].includes(estado) &&
        orden.checkoutUrl &&
        isRecentlyCreated(orden)
      ) {
        res.status(200).json({
          ok: true,
          pagoId,
          preferenceId: orden.preferenceId || orden.mercadoPagoPreferenceId || null,
          checkoutUrl: orden.checkoutUrl,
          ambiente: orden.ambiente || process.env.MP_ENV?.trim() || "test",
          reutilizada: true,
        });
        return;
      }

      const importe = Number(orden.importe);
      const moneda = String(orden.moneda || "ARS");
      const concepto = String(orden.concepto || "Pago SIDCA").trim();
      const detalle = String(orden.detalle || concepto).trim();
      const afiliadoNombre = String(orden.afiliadoNombre || "Afiliado SIDCA").trim();
      const ambiente = process.env.MP_ENV?.trim() || "test";
      const externalReference = orden.externalReference || `SIDCA-PAGO-${pagoId}`;
      const backUrls = getMercadoPagoBackUrls();
      const notificationUrl = process.env.MP_WEBHOOK_URL?.trim();

      await updateFirestoreDoc(`pagos_adherentes/${pagoId}`, {
        pagoId,
        uid: authUser.uid,
        dni,
        afiliadoNombre,
        moneda,
        ambiente,
        estadoInterno: "creada",
        externalReference,
        procesado: false,
        requiereRevisionAdministrativa: Boolean(orden.requiereRevisionAdministrativa),
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
        payer: {
          name: afiliadoNombre,
          identification: { type: "DNI", number: dni },
        },
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

    await validateDniBelongsToUser(dni, authUser.uid);

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
        error: "La cuota adherente de este perÃ­odo ya fue abonada.",
        pago: publicPagoFields(approved),
      });
      return;
    }

    const reusable = existingPayments.find(
      (payment) =>
        ["creada", "preferencia_creada", "pendiente", "en_proceso"].includes(
          String(payment.estadoInterno || "")
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
        ambiente: reusable.ambiente || process.env.MP_ENV?.trim() || "test",
        reutilizada: true,
      });
      return;
    }

    const pagoId = crypto.randomUUID();
    const externalReference = `SIDCA-CUOTA-${config.periodo}-${pagoId}`;
    const ambiente = process.env.MP_ENV?.trim() || "test";
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
      estadoInterno: "creada",
      estadoMercadoPago: null,
      externalReference,
      procesado: false,
      requiereRevisionAdministrativa: false,
      createdAt,
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
      payer: {
        name: afiliadoNombre,
        identification: {
          type: "DNI",
          number: dni,
        },
      },
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
      initPoint: mpPreference.init_point || null,
      sandboxInitPoint: mpPreference.sandbox_init_point || null,
      checkoutUrl,
      estadoInterno: "preferencia_creada",
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

    const idempotencyDoc = await getFirestoreDoc(`pagos_mercadopago/${paymentId}`);
    if (idempotencyDoc) {
      res.status(200).json({ ok: true, duplicate: true });
      return;
    }

    const payment = await fetchMercadoPagoPayment(paymentId);
    const externalReference = String(payment.external_reference || "");
    const internalPayments = await queryFirestoreCollection(
      "pagos_adherentes",
      [{ field: "externalReference", value: externalReference }],
      1
    );
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
    const expectedLiveMode = String(internalPayment.ambiente || "test") !== "test";

    if (
      externalReference !== internalPayment.externalReference ||
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

    await createFirestoreDoc("pagos_mercadopago", paymentId, {
      mercadoPagoPaymentId: String(payment.id),
      pagoAdherenteId: internalPayment.id,
      externalReference,
      status: payment.status || null,
      statusDetail: payment.status_detail || null,
      createdAt: new Date(),
    });

    const mappedStatus = mapMercadoPagoStatus(payment.status);
    const previousStatus = internalPayment.estadoInterno || null;

    await updateFirestoreDoc(`pagos_adherentes/${internalPayment.id}`, {
      estadoInterno: mappedStatus.estadoInterno,
      estadoMercadoPago: payment.status || null,
      estadoMercadoPagoDetalle: payment.status_detail || null,
      mercadoPagoPaymentId: String(payment.id),
      paymentMethodId: payment.payment_method_id || null,
      paymentTypeId: payment.payment_type_id || null,
      fechaPago: payment.date_approved || null,
      mercadoPagoDateCreated: payment.date_created || null,
      procesado: mappedStatus.procesado,
      requiereRevisionAdministrativa: mappedStatus.revision,
      updatedAt: new Date(),
    });

    await appendPagoEvento(String(internalPayment.id), {
      tipo: "webhook_payment",
      estadoAnterior: previousStatus,
      estadoNuevo: mappedStatus.estadoInterno,
      origen: "mercado_pago_webhook",
      detalle: payment.status_detail || undefined,
    });

    if (mappedStatus.estadoInterno === "aprobado") {
      await activateAdherenteAfterPayment(
        String(internalPayment.dni),
        Number(internalPayment.periodo),
        String(payment.id),
        String(internalPayment.id)
      );
    }

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
    const pagoId = String(req.params.pagoId || "").trim();
    const payment = await getFirestoreDoc(`pagos_adherentes/${pagoId}`);

    if (!payment) {
      res.status(404).json({ ok: false, error: "No se encontrÃ³ el pago." });
      return;
    }

    if (payment.uid !== authUser.uid) {
      res.status(403).json({ ok: false, error: "No tenÃ©s acceso a este pago." });
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
      const aDate = new Date(a.createdAt || 0).getTime();
      const bDate = new Date(b.createdAt || 0).getTime();
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
          error: "No se recibiÃ³ ningÃºn archivo de audio.",
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
