import protobuf from "protobufjs";

export const FIRESTORE_CREATED_EVENT_TYPE = "google.cloud.firestore.document.v1.created";
export const FIRESTORE_UPDATED_EVENT_TYPE = "google.cloud.firestore.document.v1.updated";

const USUARIO_DOCUMENT_NAME =
  /^projects\/[^/]+\/databases\/[^/]+\/documents\/usuarios\/([A-Za-z0-9_-]{1,128})$/;

// Eventarc delivers DocumentEventData as protobuf. The schema intentionally
// declares only the fields needed to locate the document; protobufjs skips
// the remaining fields while decoding them according to their wire types.
const documentEventRoot = protobuf.Root.fromJSON({
  nested: {
    google: {
      nested: {
        events: {
          nested: {
            cloud: {
              nested: {
                firestore: {
                  nested: {
                    v1: {
                      nested: {
                        DocumentEventData: {
                          fields: {
                            value: { type: "Document", id: 1 },
                          },
                        },
                        Document: {
                          fields: {
                            name: { type: "string", id: 1 },
                          },
                        },
                      },
                    },
                  },
                },
              },
            },
          },
        },
      },
    },
  },
});

const DocumentEventData = documentEventRoot.lookupType(
  "google.events.cloud.firestore.v1.DocumentEventData",
);

export function validarTipoCloudEvent(tipo: unknown): void {
  validarTipoCloudEventEsperado(tipo, FIRESTORE_CREATED_EVENT_TYPE);
}

export function validarTipoCloudEventEsperado(tipo: unknown, esperado: string): void {
  if (String(tipo || "").trim() !== esperado) {
    throw Object.assign(new Error("Tipo de CloudEvent no admitido."), { statusCode: 400 });
  }
}

function decodificarNombreRecursoDesdeDocumentEvent(body: unknown): string {
  if (!Buffer.isBuffer(body) || body.length === 0) {
    throw Object.assign(new Error("CloudEvent protobuf vacío o inválido."), { statusCode: 400 });
  }

  let decoded: { value?: { name?: unknown } };
  try {
    decoded = DocumentEventData.decode(body) as unknown as { value?: { name?: unknown } };
  } catch {
    throw Object.assign(new Error("CloudEvent protobuf inválido."), { statusCode: 400 });
  }

  if (typeof decoded.value?.name !== "string" || !decoded.value.name.trim()) {
    throw Object.assign(new Error("CloudEvent sin nombre de documento válido."), { statusCode: 400 });
  }

  return decoded.value.name.trim();
}

export function extraerDniDesdeNombreCounter(nombreRecurso: unknown): string {
  if (typeof nombreRecurso !== "string") {
    throw Object.assign(new Error("CloudEvent sin nombre de documento válido."), { statusCode: 400 });
  }

  const match = nombreRecurso.trim().match(
    /^projects\/[^/]+\/databases\/[^/]+\/documents\/nuevoAfiliado_counters\/(\d{6,9})$/,
  );
  if (!match) {
    throw Object.assign(new Error("CloudEvent sin recurso nuevoAfiliado_counters válido."), { statusCode: 400 });
  }

  return match[1];
}

export function decodificarDniDesdeCounterEvent(body: unknown): string {
  return extraerDniDesdeNombreCounter(decodificarNombreRecursoDesdeDocumentEvent(body));
}

export function extraerUsuarioIdDesdeNombreDocumento(nombreRecurso: unknown): string {
  if (typeof nombreRecurso !== "string") {
    throw Object.assign(new Error("CloudEvent sin nombre de documento válido."), { statusCode: 400 });
  }

  const match = nombreRecurso.trim().match(USUARIO_DOCUMENT_NAME);
  if (!match) {
    throw Object.assign(new Error("CloudEvent sin recurso usuarios válido."), { statusCode: 400 });
  }

  return match[1];
}

export function decodificarUsuarioIdDesdeDocumentEvent(body: unknown): string {
  return extraerUsuarioIdDesdeNombreDocumento(decodificarNombreRecursoDesdeDocumentEvent(body));
}

/** Test helper: creates a valid DocumentEventData body without hand-writing wire bytes. */
export function codificarDocumentEventParaPrueba(nombreRecurso: string): Buffer {
  const message = DocumentEventData.create({ value: { name: nombreRecurso } });
  return Buffer.from(DocumentEventData.encode(message).finish());
}
