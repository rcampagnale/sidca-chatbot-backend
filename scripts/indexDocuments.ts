import "dotenv/config";
import fs from "node:fs";
import path from "node:path";
import OpenAI from "openai";

type ManifestItem = {
  source: string;
  filename: string;
  attributes: {
    ambito: "provincial" | "municipal";
    dominio: "licencias" | "estatuto" | "general";
    [key: string]: string | number | boolean;
  };
};

const ROOT_DIR = process.cwd();
const MANIFEST_PATH = path.join(ROOT_DIR, "docs", "manifest.json");
const TMP_DIR = path.join(ROOT_DIR, ".tmp-index");

function getRequiredEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) {
    throw new Error(`Falta ${name} en .env`);
  }
  return value;
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function ensureDir(dirPath: string) {
  await fs.promises.mkdir(dirPath, { recursive: true });
}

async function readManifest(): Promise<ManifestItem[]> {
  const raw = await fs.promises.readFile(MANIFEST_PATH, "utf8");
  const parsed = JSON.parse(raw);

  if (!Array.isArray(parsed)) {
    throw new Error("docs/manifest.json debe ser un array");
  }

  return parsed.map((item, index) => {
    if (!item?.source || !item?.filename || !item?.attributes) {
      throw new Error(`Manifest inválido en posición ${index}`);
    }

    return {
      source: String(item.source),
      filename: String(item.filename),
      attributes: item.attributes,
    } as ManifestItem;
  });
}

async function downloadFile(url: string, outPath: string) {
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`No se pudo descargar ${url} (${response.status})`);
  }

  const arrayBuffer = await response.arrayBuffer();
  const buffer = Buffer.from(arrayBuffer);

  await fs.promises.writeFile(outPath, buffer);
}

async function getOrCreateVectorStore(client: OpenAI): Promise<{
  vectorStoreId: string;
  createdNow: boolean;
}> {
  const existing = process.env.OPENAI_VECTOR_STORE_ID?.trim();
  if (existing) {
    return { vectorStoreId: existing, createdNow: false };
  }

  const vectorStore = await client.vectorStores.create({
    name: "SIDCA Provincia",
  });

  return { vectorStoreId: vectorStore.id, createdNow: true };
}

async function uploadRawFile(client: OpenAI, localPath: string) {
  const uploaded = await client.files.create({
    file: fs.createReadStream(localPath),
    purpose: "user_data",
  });

  return uploaded;
}

async function attachFileToVectorStoreAndPoll(
  client: OpenAI,
  vectorStoreId: string,
  fileId: string,
  attributes: Record<string, string | number | boolean>
) {
  const vectorStoreFile = await client.vectorStores.files.create(vectorStoreId, {
    file_id: fileId,
    attributes,
  });

  let attempts = 0;
  const maxAttempts = 120;

  while (attempts < maxAttempts) {
    const current = await client.vectorStores.files.retrieve(vectorStoreFile.id, {
      vector_store_id: vectorStoreId,
    });

    const status = current.status;

    if (status === "completed") {
      return current;
    }

    if (status === "failed" || status === "cancelled") {
      throw new Error(
        `Falló el procesamiento del archivo ${fileId}. Estado: ${status}. ` +
          `Detalle: ${current.last_error?.message || "sin detalle"}`
      );
    }

    attempts += 1;
    await sleep(2000);
  }

  throw new Error(`Timeout esperando el procesamiento del archivo ${fileId}`);
}

async function main() {
  const apiKey = getRequiredEnv("OPENAI_API_KEY");
  const client = new OpenAI({ apiKey });

  await ensureDir(TMP_DIR);

  const manifest = await readManifest();

  if (manifest.length === 0) {
    throw new Error("docs/manifest.json está vacío");
  }

  const { vectorStoreId, createdNow } = await getOrCreateVectorStore(client);

  console.log("========================================");
  console.log("SIDCA - Indexación de documentos");
  console.log("Vector Store ID:", vectorStoreId);
  console.log("Creado ahora:", createdNow ? "sí" : "no");
  console.log("Documentos a procesar:", manifest.length);
  console.log("========================================");

  for (const item of manifest) {
    const localPath = path.join(TMP_DIR, item.filename);

    console.log(`\n[1/4] Descargando: ${item.filename}`);
    await downloadFile(item.source, localPath);

    console.log(`[2/4] Subiendo archivo a OpenAI: ${item.filename}`);
    const uploadedFile = await uploadRawFile(client, localPath);

    console.log(`[3/4] Adjuntando al vector store con atributos`);
    console.log(`      file_id=${uploadedFile.id}`);
    console.log(`      attributes=${JSON.stringify(item.attributes)}`);

    const processed = await attachFileToVectorStoreAndPoll(
      client,
      vectorStoreId,
      uploadedFile.id,
      item.attributes
    );

    console.log(`[4/4] Procesado OK`);
    console.log(`      vector_store_file_id=${processed.id}`);
    console.log(`      status=${processed.status}`);
    console.log(`      usage_bytes=${processed.usage_bytes ?? "n/d"}`);
  }

  console.log("\n========================================");
  console.log("Indexación finalizada.");
  console.log("OPENAI_VECTOR_STORE_ID =", vectorStoreId);
  console.log("========================================");

  if (createdNow) {
    console.log(
      "\nGuarda este valor en tu .env para reutilizar el mismo vector store:"
    );
    console.log(`OPENAI_VECTOR_STORE_ID=${vectorStoreId}`);
  }
}

main().catch((error) => {
  console.error("\n[ERROR indexDocuments]", error);
  process.exit(1);
});