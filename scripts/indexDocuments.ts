import "dotenv/config";
import fs from "node:fs";
import path from "node:path";
import OpenAI from "openai";

type Ambito = "provincial" | "municipal";
type Dominio = "licencias" | "estatuto" | "general";

type ManifestItem = {
  source: string;
  filename: string;
  attributes: {
    ambito: Ambito;
    dominio: Dominio;
    [key: string]: string | number | boolean;
  };
};

const ROOT_DIR = process.cwd();
const MANIFEST_PATH = path.join(ROOT_DIR, "docs", "manifest.json");
const TMP_DIR = path.join(ROOT_DIR, ".tmp-index");

const VALID_AMBITOS: Ambito[] = ["provincial", "municipal"];
const VALID_DOMINIOS: Dominio[] = ["licencias", "estatuto", "general"];

const VECTOR_STORE_ENV_BY_DOMINIO: Record<Dominio, string> = {
  licencias: "OPENAI_VECTOR_STORE_ID_LICENCIAS",
  estatuto: "OPENAI_VECTOR_STORE_ID_ESTATUTO",
  general: "OPENAI_VECTOR_STORE_ID_GENERAL",
};

const VECTOR_STORE_NAME_BY_DOMINIO: Record<Dominio, string> = {
  licencias: "SIDCA Licencias",
  estatuto: "SIDCA Estatuto",
  general: "SIDCA General",
};

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

function isValidAmbito(value: unknown): value is Ambito {
  return typeof value === "string" && VALID_AMBITOS.includes(value as Ambito);
}

function isValidDominio(value: unknown): value is Dominio {
  return typeof value === "string" && VALID_DOMINIOS.includes(value as Dominio);
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

    if (!isValidAmbito(item.attributes.ambito)) {
      throw new Error(
        `Manifest inválido en posición ${index}: ambito "${item.attributes.ambito}" no permitido`
      );
    }

    if (!isValidDominio(item.attributes.dominio)) {
      throw new Error(
        `Manifest inválido en posición ${index}: dominio "${item.attributes.dominio}" no permitido`
      );
    }

    return {
      source: String(item.source),
      filename: String(item.filename),
      attributes: {
        ...item.attributes,
        ambito: item.attributes.ambito,
        dominio: item.attributes.dominio,
      },
    };
  });
}

function groupManifestByDominio(manifest: ManifestItem[]) {
  return manifest.reduce<Record<Dominio, ManifestItem[]>>(
    (acc, item) => {
      acc[item.attributes.dominio].push(item);
      return acc;
    },
    {
      licencias: [],
      estatuto: [],
      general: [],
    }
  );
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

async function getOrCreateVectorStoreByDominio(
  client: OpenAI,
  dominio: Dominio
): Promise<{
  vectorStoreId: string;
  createdNow: boolean;
  envName: string;
}> {
  const envName = VECTOR_STORE_ENV_BY_DOMINIO[dominio];
  const existing = process.env[envName]?.trim();

  if (existing) {
    return {
      vectorStoreId: existing,
      createdNow: false,
      envName,
    };
  }

  const vectorStore = await client.vectorStores.create({
    name: VECTOR_STORE_NAME_BY_DOMINIO[dominio],
  });

  return {
    vectorStoreId: vectorStore.id,
    createdNow: true,
    envName,
  };
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

  const grouped = groupManifestByDominio(manifest);
  const dominios: Dominio[] = ["licencias", "estatuto", "general"];

  const createdStores: Array<{
    dominio: Dominio;
    envName: string;
    vectorStoreId: string;
    createdNow: boolean;
  }> = [];

  console.log("========================================");
  console.log("SIDCA - Indexación de documentos por dominio");
  console.log("========================================");

  for (const dominio of dominios) {
    const items = grouped[dominio];

    if (items.length === 0) {
      console.log(`\n[${dominio}] Sin archivos. Se omite.`);
      continue;
    }

    const storeInfo = await getOrCreateVectorStoreByDominio(client, dominio);

    createdStores.push({
      dominio,
      envName: storeInfo.envName,
      vectorStoreId: storeInfo.vectorStoreId,
      createdNow: storeInfo.createdNow,
    });

    console.log(`\n========================================`);
    console.log(`Dominio: ${dominio}`);
    console.log(`Vector Store ID: ${storeInfo.vectorStoreId}`);
    console.log(`Creado ahora: ${storeInfo.createdNow ? "sí" : "no"}`);
    console.log(`Documentos a procesar: ${items.length}`);
    console.log("========================================");

    for (const item of items) {
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
        storeInfo.vectorStoreId,
        uploadedFile.id,
        item.attributes
      );

      console.log(`[4/4] Procesado OK`);
      console.log(`      vector_store_file_id=${processed.id}`);
      console.log(`      status=${processed.status}`);
      console.log(`      usage_bytes=${processed.usage_bytes ?? "n/d"}`);
    }
  }

  console.log("\n========================================");
  console.log("Indexación finalizada.");
  console.log("========================================");

  for (const store of createdStores) {
    console.log(
      `${store.envName}=${store.vectorStoreId} ${store.createdNow ? "(nuevo)" : "(reutilizado)"}`
    );
  }

  const newStores = createdStores.filter((store) => store.createdNow);

  if (newStores.length > 0) {
    console.log("\nGuarda estos valores en tu .env para reutilizar los mismos vector stores:");
    for (const store of newStores) {
      console.log(`${store.envName}=${store.vectorStoreId}`);
    }
  }
}

main().catch((error) => {
  console.error("\n[ERROR indexDocuments]", error);
  process.exit(1);
});