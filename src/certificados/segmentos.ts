// src/certificados/segmentos.ts
//
// Segmentos geográficos para la descarga de certificados por partes.
//
// Es la ÚNICA definición: el frontend no tiene su propia copia, la recibe por
// la API. Mover un departamento de segmento se hace acá y nada más.
//
// La clasificación es deliberadamente ESTRICTA. Un valor que no coincide
// exactamente con un departamento canónico —una vez normalizado— cae en
// "sin-departamento" en lugar de asignarse por aproximación: mandar el
// certificado de alguien al segmento equivocado es peor que dejarlo en el
// grupo de revisión.
//
// Por eso NO se reutiliza normalizeDepartamentoLabel() del frontend, que
// resuelve por subcadena y por primera palabra: con esa lógica la clave
// "santa" empareja tanto Santa María como Santa Rosa, y el último gana.

/** Los 16 departamentos de Catamarca, con su grafía oficial. */
export const DEPARTAMENTOS_CANONICOS = [
  "Ambato",
  "Ancasti",
  "Andalgalá",
  "Antofagasta de la Sierra",
  "Belén",
  "Capayán",
  "Capital",
  "El Alto",
  "Fray Mamerto Esquiú",
  "La Paz",
  "Paclín",
  "Pomán",
  "Santa María",
  "Santa Rosa",
  "Tinogasta",
  "Valle Viejo",
] as const;

export type DepartamentoCanonico = (typeof DEPARTAMENTOS_CANONICOS)[number];

/** Identificador del grupo de revisión. No agrupa departamentos. */
export const SEGMENTO_SIN_DEPARTAMENTO = "sin-departamento";

export type Segmento = {
  id: string;
  nombre: string;
  departamentos: string[];
};

export const SEGMENTOS: Segmento[] = [
  {
    id: "valle-central",
    nombre: "Valle Central",
    departamentos: [
      "Ambato",
      "Capayán",
      "Capital",
      "El Alto",
      "Fray Mamerto Esquiú",
      "Paclín",
      "Valle Viejo",
      "Santa Rosa",
    ],
  },
  { id: "tinogasta", nombre: "Tinogasta", departamentos: ["Tinogasta"] },
  { id: "santa-maria", nombre: "Santa María", departamentos: ["Santa María"] },
  { id: "poman", nombre: "Pomán", departamentos: ["Pomán"] },
  { id: "andalgala", nombre: "Andalgalá", departamentos: ["Andalgalá"] },
  {
    id: "antofagasta-belen",
    nombre: "Antofagasta - Belén",
    departamentos: ["Antofagasta de la Sierra", "Belén"],
  },
  {
    id: "ancasti-la-paz",
    nombre: "Ancasti - La Paz",
    departamentos: ["Ancasti", "La Paz"],
  },
  {
    // Octavo grupo: no es una región, es el cajón de revisión. Recoge a quien
    // no tiene departamento cargado y a quien tiene un valor que no se pudo
    // reconocer. Existe para DETECTAR problemas de datos, así que se muestra
    // siempre, incluso vacío.
    id: SEGMENTO_SIN_DEPARTAMENTO,
    nombre: "Sin departamento",
    departamentos: [],
  },
];

export const SEGMENTOS_VALIDOS = new Set(SEGMENTOS.map((s) => s.id));

/**
 * Clave de comparación de un departamento.
 *
 * Quita diacríticos, pasa a minúsculas, unifica guiones bajos y medios con
 * espacios y colapsa los espacios. Con esto " CAPITAL " y "capital" son la
 * misma clave, y "fray mamerto esquiu" empareja con "Fray Mamerto Esquiú".
 *
 * NO hace coincidencias parciales: la clave resultante tiene que ser idéntica
 * a la de un departamento canónico.
 */
export const normalizarDepartamento = (valor: unknown): string =>
  String(valor ?? "")
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();

const CANONICO_POR_CLAVE = new Map<string, string>(
  DEPARTAMENTOS_CANONICOS.map((d) => [normalizarDepartamento(d), d])
);

/**
 * Nombre canónico de un departamento, o "" si el valor no se reconoce.
 *
 * El dato guardado en Firestore no se toca: esto es sólo para clasificar y
 * para mostrar.
 */
export const canonizarDepartamento = (valor: unknown): string =>
  CANONICO_POR_CLAVE.get(normalizarDepartamento(valor)) || "";

const SEGMENTO_POR_CLAVE = new Map<string, string>();

for (const segmento of SEGMENTOS) {
  for (const departamento of segmento.departamentos) {
    SEGMENTO_POR_CLAVE.set(normalizarDepartamento(departamento), segmento.id);
  }
}

/**
 * Segmento al que pertenece un valor de departamento.
 *
 * Todo lo que no sea un departamento canónico reconocido —vacío, nulo, sólo
 * espacios, abreviado, con errores de tipeo o de otra provincia— va al grupo
 * de revisión.
 */
export const resolverSegmentoDepartamento = (valor: unknown): string =>
  SEGMENTO_POR_CLAVE.get(normalizarDepartamento(valor)) ||
  SEGMENTO_SIN_DEPARTAMENTO;

export const esSegmentoValido = (id: unknown): boolean =>
  SEGMENTOS_VALIDOS.has(String(id ?? ""));

export const obtenerSegmento = (id: unknown): Segmento | null =>
  SEGMENTOS.find((s) => s.id === String(id ?? "")) || null;

/**
 * Comprueba que el mapa esté bien formado.
 *
 * Los 16 departamentos tienen que estar en exactamente un segmento: si uno
 * quedara afuera, su gente iría a parar al grupo de revisión sin motivo, y si
 * apareciera en dos, sus certificados se descargarían por duplicado. Se
 * ejecuta al cargar el módulo para que un error de edición se note enseguida y
 * no meses después.
 */
export function verificarCoberturaSegmentos(): {
  ok: boolean;
  faltantes: string[];
  duplicados: string[];
  desconocidos: string[];
} {
  const vistos = new Map<string, number>();
  const desconocidos: string[] = [];

  for (const segmento of SEGMENTOS) {
    for (const departamento of segmento.departamentos) {
      const clave = normalizarDepartamento(departamento);

      if (!CANONICO_POR_CLAVE.has(clave)) desconocidos.push(departamento);

      vistos.set(clave, (vistos.get(clave) || 0) + 1);
    }
  }

  const faltantes = DEPARTAMENTOS_CANONICOS.filter(
    (d) => !vistos.has(normalizarDepartamento(d))
  );

  const duplicados = [...vistos.entries()]
    .filter(([, veces]) => veces > 1)
    .map(([clave]) => CANONICO_POR_CLAVE.get(clave) || clave);

  return {
    ok: !faltantes.length && !duplicados.length && !desconocidos.length,
    faltantes: [...faltantes],
    duplicados,
    desconocidos,
  };
}

// Falla ruidosamente al arrancar si alguien rompe el mapa editando SEGMENTOS.
{
  const revision = verificarCoberturaSegmentos();

  if (!revision.ok) {
    console.error(
      "[certificados-segmentos] mapa inválido",
      JSON.stringify(revision)
    );
  }
}
