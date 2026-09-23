const EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send";
const EXPO_RECEIPTS_URL = "https://exp.host/--/api/v2/push/getReceipts";
const EXPO_BATCH_SIZE = 100;
const EXPO_RECEIPTS_BATCH_SIZE = 300;

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

  for (const batch of chunks(tokens, EXPO_BATCH_SIZE)) {
    const payload: ExpoPushMessage[] = batch.map((to) => ({
      to,
      sound: "default",
      title: input.title,
      body: input.body,
      data: input.data || {},
    }));
    const response = await fetch(EXPO_PUSH_URL, {
      method: "POST",
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    lastStatus = response.status;
    const responseBody = (await response.json().catch(() => ({}))) as {
      data?: ExpoPushTicket[];
      errors?: unknown;
    };
    if (!response.ok) {
      throw Object.assign(new Error("Expo Push API rechazó el envío."), {
        statusCode: response.status >= 500 ? 502 : 400,
        expoStatus: response.status,
        expoErrors: responseBody.errors,
      });
    }
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
