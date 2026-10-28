import assert from "node:assert/strict";
import {
  EXPO_PUSH_THROTTLE_MS,
  ExpoPushError,
  formatExpoPushErrorLog,
  sendExpoPushNotifications,
} from "../src/push/expoPush.js";

type FakeResponse = {
  ok: boolean;
  status: number;
  headers: { get(name: string): string | null };
  json(): Promise<unknown>;
};

type FetchCall = { body: { to: string[]; title: string; body: string } };

const fakeToken = (index: number) => `ExpoPushToken[test-${index}]`;

function response(status: number, body: unknown, headers: Record<string, string> = {}): FakeResponse {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: (name) => headers[name.toLowerCase()] || null },
    json: async () => body,
  };
}

function okResponse(tokens: string[], offset: number): FakeResponse {
  return response(200, {
    data: tokens.map((_token, index) => ({ status: "ok", id: `ticket-${offset + index}` })),
  });
}

async function runWithMockFetch(
  handler: (call: FetchCall, callNumber: number) => FakeResponse,
  operation: () => Promise<unknown>,
) {
  const originalFetch = globalThis.fetch;
  const originalSetTimeout = globalThis.setTimeout;
  const delays: number[] = [];
  const calls: FetchCall[] = [];
  let callNumber = 0;
  (globalThis as typeof globalThis & { setTimeout: typeof setTimeout }).setTimeout = ((callback: (...args: unknown[]) => void, delay = 0) => {
    delays.push(delay);
    callback();
    return 0 as unknown as ReturnType<typeof setTimeout>;
  }) as typeof setTimeout;
  globalThis.fetch = (async (_url, init) => {
    const rawBody = JSON.parse(String(init?.body || "{}")) as unknown;
    const body = Array.isArray(rawBody)
      ? {
          to: rawBody.map((message) => String((message as { to?: string }).to || "")),
          title: String((rawBody[0] as { title?: string } | undefined)?.title || ""),
          body: String((rawBody[0] as { body?: string } | undefined)?.body || ""),
        }
      : rawBody as { to: string[]; title: string; body: string };
    const call = { body };
    calls.push(call);
    const result = handler(call, callNumber);
    callNumber += 1;
    return result as Response;
  }) as typeof fetch;

  try {
    const result = await operation();
    return { result, calls, delays };
  } finally {
    globalThis.fetch = originalFetch;
    globalThis.setTimeout = originalSetTimeout;
  }
}

const baseInput = {
  title: "Prueba controlada",
  body: "Mensaje de prueba",
  data: { test: true },
  includeReceipts: false,
};

{
  const tokens = Array.from({ length: 100 }, (_item, index) => fakeToken(index));
  const { result, calls, delays } = await runWithMockFetch(
    (call) => okResponse(call.body.to, 0),
    () => sendExpoPushNotifications({ ...baseInput, token: tokens }),
  );
  assert.equal(calls.length, 1, "un lote debe hacer una llamada");
  assert.equal(calls[0].body.to.length, 100);
  assert.deepEqual(delays, []);
  assert.equal((result as { tickets: unknown[] }).tickets.length, 100);
}

{
  const tokens = Array.from({ length: 1600 }, (_item, index) => fakeToken(index));
  const { calls, delays } = await runWithMockFetch(
    (call, callNumber) => okResponse(call.body.to, callNumber * 100),
    () => sendExpoPushNotifications({ ...baseInput, token: tokens }),
  );
  assert.equal(calls.length, 16, "1600 tokens deben dividirse en 16 lotes");
  assert.ok(calls.every((call) => call.body.to.length <= 100));
  assert.deepEqual(delays, Array(15).fill(EXPO_PUSH_THROTTLE_MS));
}

{
  const tokens = Array.from({ length: 800 }, (_item, index) => fakeToken(index));
  const { calls } = await runWithMockFetch(
    (call, callNumber) => callNumber === 6 ? response(429, { errors: [{ code: "RATE_LIMIT" }] }) : okResponse(call.body.to, callNumber),
    () => sendExpoPushNotifications({ ...baseInput, token: tokens }),
  );
  assert.equal(calls.length, 9, "el lote 7 debe reintentarse una sola vez");
  assert.deepEqual(calls.map((call) => call.body.to[0]), [
    fakeToken(0), fakeToken(100), fakeToken(200), fakeToken(300), fakeToken(400),
    fakeToken(500), fakeToken(600), fakeToken(600), fakeToken(700),
  ]);
}

{
  const tokens = [fakeToken(1)];
  const { calls, delays } = await runWithMockFetch(
    (_call, callNumber) => callNumber === 0 ? response(503, { errors: [{ code: "TEMPORARY" }] }) : okResponse(tokens, 0),
    () => sendExpoPushNotifications({ ...baseInput, token: tokens }),
  );
  assert.equal(calls.length, 2);
  assert.deepEqual(delays, [1000]);
}

{
  const tokens = [fakeToken(1)];
  const { delays } = await runWithMockFetch(
    (_call, callNumber) => callNumber === 0
      ? response(503, { errors: [{ code: "TEMPORARY" }] }, { "retry-after": "2" })
      : okResponse(tokens, 0),
    () => sendExpoPushNotifications({ ...baseInput, token: tokens }),
  );
  assert.deepEqual(delays, [2000], "Retry-After válido debe prevalecer sobre el backoff");
}

{
  const tokens = [fakeToken(1)];
  const { calls } = await runWithMockFetch(
    () => response(400, { errors: [{ code: "VALIDATION_ERROR", message: "token inválido" }] }),
    async () => {
      await assert.rejects(
        () => sendExpoPushNotifications({ ...baseInput, token: tokens }),
        (error: unknown) => {
          assert.ok(error instanceof ExpoPushError);
          assert.equal(error.expoStatus, 400);
          assert.equal(error.batchNumber, 1);
          assert.equal(error.attempts, 1);
          assert.equal(error.retryable, false);
          return true;
        },
      );
    },
  );
  assert.equal(calls.length, 1, "400 no debe reintentarse");
}

{
  const tokens = Array.from({ length: 700 }, (_item, index) => fakeToken(index));
  const { calls } = await runWithMockFetch(
    (call, callNumber) => callNumber < 6
      ? okResponse(call.body.to, callNumber * 100)
      : response(503, { errors: [{ code: "TEMPORARY", token: call.body.to[0] }] }),
    async () => {
      await assert.rejects(
        () => sendExpoPushNotifications({ ...baseInput, token: tokens }),
        (error: unknown) => {
          assert.ok(error instanceof ExpoPushError);
          assert.equal(error.batchNumber, 7);
          assert.equal(error.batchesCompleted, 6);
          assert.equal(error.tokensAcceptedBeforeError, 600);
          assert.equal(error.attempts, 4);
          const log = formatExpoPushErrorLog(error);
          assert.ok(log);
          assert.equal(log.includes(fakeToken(0)), false);
          assert.equal(log.includes(fakeToken(600)), false);
          assert.equal(log.includes("Prueba controlada"), false);
          assert.equal(log.includes("Mensaje de prueba"), false);
          assert.match(log, /batch=7\/7 status=503 enviadosPrevios=600 tokensLote=100/);
          return true;
        },
      );
    },
  );
  assert.equal(calls.length, 10, "sólo el lote fallido debe agotar los reintentos");
  assert.deepEqual(calls.slice(0, 6).map((call) => call.body.to[0]), Array.from({ length: 6 }, (_item, index) => fakeToken(index * 100)));
  assert.ok(calls.slice(6).every((call) => call.body.to[0] === fakeToken(600)));
}

{
  const token = fakeToken(1);
  const { result, calls } = await runWithMockFetch(
    (_call, callNumber) => callNumber === 0
      ? okResponse([token], 0)
      : response(200, { data: { "ticket-0": { status: "error", details: { error: "DeviceNotRegistered" } } } }),
    () => sendExpoPushNotifications({ ...baseInput, token, includeReceipts: true }),
  );
  assert.equal(calls.length, 2);
  assert.equal((result as { receipts: unknown[] }).receipts.length, 1);
  assert.deepEqual((result as { deviceNotRegisteredTokens: string[] }).deviceNotRegisteredTokens, [token]);
}

console.log("OK: Expo Push batch, throttling, retry, logging seguro, partial failure y receipts");
