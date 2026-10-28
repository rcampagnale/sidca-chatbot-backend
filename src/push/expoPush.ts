const EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send";
const EXPO_RECEIPTS_URL = "https://exp.host/--/api/v2/push/getReceipts";
export const EXPO_PUSH_BATCH_SIZE = 100;
export const EXPO_PUSH_THROTTLE_MS = 250;
export const EXPO_PUSH_MAX_RETRIES = 3;
const EXPO_RECEIPTS_BATCH_SIZE = 300;
const EXPO_RETRY_BACKOFF_MS = [1000, 2000, 4000] as const;
const EXPO_RETRYABLE_STATUSES = new Set([429, 500, 502, 503, 504]);

export type ExpoPushMessage = {
  to: string;
  sound: "default";
  title: string;
  body: string;
  data: Record<string, unknown>;
};

export type ExpoPushTicket = {
  status?: string;
  id?: string;
  message?: string;
  details?: { error?: string; [key: string]: unknown };
  [key: string]: unknown;
};

export type ExpoPushReceipt = {
  id?: string;
  status?: string;
  message?: string;
  details?: { error?: string; [key: string]: unknown };
  [key: string]: unknown;
};

export type ExpoPushResult = {
  status: number;
  tickets: ExpoPushTicket[];
  receipts: ExpoPushReceipt[];
  receiptsUnavailable: boolean;
  deviceNotRegisteredTokens: string[];
};

export type ExpoPushFailureContext = {
  statusCode: number;
  expoStatus: number;
  expoErrors: unknown;
  batchNumber: number;
  totalBatches: number;
  tokensInBatch: number;
  batchesCompleted: number;
  tokensAcceptedBeforeError: number;
  attempts: number;
  retryable: boolean;
  retryAfterMs?: number;
};

export class ExpoPushError extends Error implements ExpoPushFailureContext {
  readonly statusCode: number;
  readonly expoStatus: number;
  readonly expoErrors: unknown;
  readonly batchNumber: number;
  readonly totalBatches: number;
  readonly tokensInBatch: number;
  readonly batchesCompleted: number;
  readonly tokensAcceptedBeforeError: number;
  readonly attempts: number;
  readonly retryable: boolean;
  readonly retryAfterMs?: number;

  constructor(context: ExpoPushFailureContext) {
    super("Expo Push API rechazó el envío.");
    this.name = "ExpoPushError";
    this.statusCode = context.statusCode;
    this.expoStatus = context.expoStatus;
    this.expoErrors = context.expoErrors;
    this.batchNumber = context.batchNumber;
    this.totalBatches = context.totalBatches;
    this.tokensInBatch = context.tokensInBatch;
    this.batchesCompleted = context.batchesCompleted;
    this.tokensAcceptedBeforeError = context.tokensAcceptedBeforeError;
    this.attempts = context.attempts;
    this.retryable = context.retryable;
    this.retryAfterMs = context.retryAfterMs;
  }
}

function chunks<T>(items: T[], size: number): T[][] {
  const result: T[][] = [];
  for (let index = 0; index < items.length; index += size) {
    result.push(items.slice(index, index + size));
  }
  return result;
}

function normalizeTokens(tokenOrTokens: string | string[]): string[] {
  const tokens = Array.isArray(tokenOrTokens) ? tokenOrTokens : [tokenOrTokens];
  return [...new Set(tokens.map((token) => String(token || "").trim()).filter(Boolean))];
}

export function isExpoPushToken(token: unknown): token is string {
  return typeof token === "string" && /^Expo(nent)?PushToken\[[^\]]+\]$/.test(token.trim());
}

function publicStatusCode(expoStatus: number): number {
  return expoStatus >= 500 ? 502 : 400;
}

function parseRetryAfter(response: Response): number | undefined {
  const value = response.headers?.get("retry-after")?.trim();
  if (!value) return undefined;

  const seconds = Number(value);
  if (Number.isFinite(seconds) && seconds >= 0) return seconds * 1000;

  const timestamp = Date.parse(value);
  if (!Number.isFinite(timestamp)) return undefined;
  return Math.max(0, timestamp - Date.now());
}

function esperar(milisegundos: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milisegundos));
}

const CLAVES_SENSIBLES_EN_LOG = /token|authorization|title|body|data|header/i;
const EXPO_TOKEN_PATTERN = /Expo(?:nent)?PushToken\[[^\]]+\]/g;

function sanitizarExpoErrors(valor: unknown, profundidad = 0): unknown {
  if (profundidad > 4) return "[truncado]";
  if (valor === null || typeof valor === "number" || typeof valor === "boolean") return valor;
  if (typeof valor === "string") return valor.replace(EXPO_TOKEN_PATTERN, "[token-redactado]");
  if (typeof valor === "undefined") return undefined;
  if (Array.isArray(valor)) {
    return valor.slice(0, 20).map((item) => sanitizarExpoErrors(item, profundidad + 1));
  }
  if (typeof valor !== "object") return String(valor);

  return Object.entries(valor as Record<string, unknown>)
    .slice(0, 20)
    .reduce<Record<string, unknown>>((resultado, [clave, contenido]) => {
      resultado[clave] = CLAVES_SENSIBLES_EN_LOG.test(clave)
        ? "[redactado]"
        : sanitizarExpoErrors(contenido, profundidad + 1);
      return resultado;
    }, {});
}

export function getExpoPushErrorLogDetails(error: unknown): ExpoPushFailureContext | null {
  if (!(error instanceof ExpoPushError)) return null;
  return {
    statusCode: error.statusCode,
    expoStatus: error.expoStatus,
    expoErrors: sanitizarExpoErrors(error.expoErrors),
    batchNumber: error.batchNumber,
    totalBatches: error.totalBatches,
    tokensInBatch: error.tokensInBatch,
    batchesCompleted: error.batchesCompleted,
    tokensAcceptedBeforeError: error.tokensAcceptedBeforeError,
    attempts: error.attempts,
    retryable: error.retryable,
    ...(error.retryAfterMs === undefined ? {} : { retryAfterMs: error.retryAfterMs }),
  };
}

export function formatExpoPushErrorLog(error: unknown): string | null {
  const details = getExpoPushErrorLogDetails(error);
  if (!details) return null;
  return [
    `[expo-push] batch=${details.batchNumber}/${details.totalBatches}`,
    `status=${details.expoStatus}`,
    `enviadosPrevios=${details.tokensAcceptedBeforeError}`,
    `tokensLote=${details.tokensInBatch}`,
    `lotesCompletados=${details.batchesCompleted}`,
    `intentos=${details.attempts}`,
    `reintentable=${details.retryable}`,
    `expoErrors=${JSON.stringify(details.expoErrors)}`,
  ].join(" ");
}

async function getExpoPushReceipts(ticketIds: string[]): Promise<ExpoPushReceipt[]> {
  const receipts: ExpoPushReceipt[] = [];
  for (let index = 0; index < ticketIds.length; index += EXPO_RECEIPTS_BATCH_SIZE) {
    const batch = ticketIds.slice(index, index + EXPO_RECEIPTS_BATCH_SIZE);
    const response = await fetch(EXPO_RECEIPTS_URL, {
      method: "POST",
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: JSON.stringify({ ids: batch }),
    });
    const responseBody = (await response.json().catch(() => ({}))) as {
      data?: Record<string, ExpoPushReceipt>;
    };
    if (!response.ok) throw new Error("Expo Push Receipts API rechazó la consulta.");
    Object.entries(responseBody.data || {}).forEach(([id, receipt]) => {
      receipts.push({ id, ...receipt });
    });
  }
  return receipts;
}

export async function sendExpoPushNotifications(input: {
  token: string | string[];
  title: string;
  body: string;
  data?: Record<string, unknown>;
  includeReceipts?: boolean;
}): Promise<ExpoPushResult> {
  const tokens = normalizeTokens(input.token);
  if (tokens.length === 0) {
    throw Object.assign(new Error("No hay tokens Expo válidos."), { statusCode: 400 });
  }

  const tickets: ExpoPushTicket[] = [];
  const ticketTokenById = new Map<string, string>();
  const deviceNotRegisteredTokens = new Set<string>();
  let lastStatus = 200;
  const batches = chunks(tokens, EXPO_PUSH_BATCH_SIZE);
  let batchesCompleted = 0;
  let tokensAcceptedBeforeError = 0;

  for (let batchIndex = 0; batchIndex < batches.length; batchIndex += 1) {
    const batch = batches[batchIndex];
    if (batchIndex > 0) await esperar(EXPO_PUSH_THROTTLE_MS);
    const payload: ExpoPushMessage[] = batch.map((to) => ({
      to,
      sound: "default",
      title: input.title,
      body: input.body,
      data: input.data || {},
    }));

    let attempts = 0;
    let responseBody: { data?: ExpoPushTicket[]; errors?: unknown } = {};
    let responseStatus = 200;
    for (;;) {
      attempts += 1;
      const response = await fetch(EXPO_PUSH_URL, {
        method: "POST",
        headers: { Accept: "application/json", "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      responseStatus = response.status;
      responseBody = (await response.json().catch(() => ({}))) as {
        data?: ExpoPushTicket[];
        errors?: unknown;
      };

      if (response.ok) break;

      const retryable = EXPO_RETRYABLE_STATUSES.has(response.status);
      const retryAfterMs = parseRetryAfter(response);
      if (!retryable || attempts > EXPO_PUSH_MAX_RETRIES) {
        throw new ExpoPushError({
          statusCode: publicStatusCode(response.status),
          expoStatus: response.status,
          expoErrors: responseBody.errors,
          batchNumber: batchIndex + 1,
          totalBatches: batches.length,
          tokensInBatch: batch.length,
          batchesCompleted,
          tokensAcceptedBeforeError,
          attempts,
          retryable,
          ...(retryAfterMs === undefined ? {} : { retryAfterMs }),
        });
      }

      const backoffMs = EXPO_RETRY_BACKOFF_MS[Math.min(attempts - 1, EXPO_RETRY_BACKOFF_MS.length - 1)];
      await esperar(Math.max(EXPO_PUSH_THROTTLE_MS, retryAfterMs ?? backoffMs));
    }

    lastStatus = responseStatus;
    if (Array.isArray(responseBody.data)) {
      responseBody.data.forEach((ticket, index) => {
        const token = batch[index];
        if (ticket.id && token) ticketTokenById.set(ticket.id, token);
        if (ticket.details?.error === "DeviceNotRegistered" && token) {
          deviceNotRegisteredTokens.add(token);
        }
        tickets.push(ticket);
      });
    }
    batchesCompleted += 1;
    tokensAcceptedBeforeError += batch.length;
  }

  let receipts: ExpoPushReceipt[] = [];
  let receiptsUnavailable = false;
  if (input.includeReceipts && tickets.some((ticket) => Boolean(ticket.id))) {
    await new Promise((resolve) => setTimeout(resolve, 1000));
    try {
      receipts = await getExpoPushReceipts(
        tickets.map((ticket) => ticket.id).filter((id): id is string => Boolean(id)),
      );
      receipts.forEach((receipt) => {
        if (receipt.details?.error !== "DeviceNotRegistered" || !receipt.id) return;
        const token = ticketTokenById.get(receipt.id);
        if (token) deviceNotRegisteredTokens.add(token);
      });
    } catch (error) {
      receiptsUnavailable = true;
      console.warn("[expo-push] No se pudieron consultar receipts:", error instanceof Error ? error.message : error);
    }
  }

  return {
    status: lastStatus,
    tickets,
    receipts,
    receiptsUnavailable,
    deviceNotRegisteredTokens: [...deviceNotRegisteredTokens],
  };
}

export function expoTicketHasDeviceNotRegistered(ticket: ExpoPushTicket): boolean {
  return ticket.details?.error === "DeviceNotRegistered";
}
