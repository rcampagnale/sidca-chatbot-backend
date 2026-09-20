const EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send";
const EXPO_BATCH_SIZE = 100;

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

export type ExpoPushResult = {
  status: number;
  tickets: ExpoPushTicket[];
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

export async function sendExpoPushNotifications(input: {
  token: string | string[];
  title: string;
  body: string;
  data?: Record<string, unknown>;
}): Promise<ExpoPushResult> {
  const tokens = normalizeTokens(input.token);
  if (tokens.length === 0) {
    throw Object.assign(new Error("No hay tokens Expo válidos."), { statusCode: 400 });
  }

  const tickets: ExpoPushTicket[] = [];
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
    if (Array.isArray(responseBody.data)) tickets.push(...responseBody.data);
  }

  return { status: lastStatus, tickets };
}

export function expoTicketHasDeviceNotRegistered(ticket: ExpoPushTicket): boolean {
  return ticket.details?.error === "DeviceNotRegistered";
}
