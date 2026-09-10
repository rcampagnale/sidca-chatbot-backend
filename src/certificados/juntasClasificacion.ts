export const JUNTA_INICIAL_PRIMARIA_ESPECIAL_ADULTO = "inicial_primaria_especial_adulto";
export const JUNTA_MEDIA_TECNICA_ARTISTICA = "media_tecnica_artistica";

const normalizarTextoNivel = (valor: unknown) => String(valor || "")
  .normalize("NFD")
  .replace(/[\u0300-\u036f]/g, "")
  .toLowerCase()
  .trim()
  .replace(/\s+/g, " ");

const dividirNivel = (valor: unknown): string[] => {
  if (Array.isArray(valor)) return valor.flatMap(dividirNivel);
  if (valor === null || valor === undefined) return [];
  return String(valor)
    .split(/\s+\+\s+|\s+y\s+|[,;/|]/i)
    .map((item) => item.trim())
    .filter(Boolean);
};

const nivelCanonico = (valor: unknown) => {
  const nivel = normalizarTextoNivel(valor);
  if (nivel === "inicial") return "inicial";
  if (["primario", "primaria"].includes(nivel)) return "primario";
  if (["secundario", "secundaria"].includes(nivel)) return "secundario";
  if (nivel === "especial") return "especial";
  if (["adulto", "adultos"].includes(nivel)) return "adulto";
  if (nivel === "media") return "media";
  if (["tecnica", "tecnico", "tecnico profesional"].includes(nivel)) return "tecnica";
  if (["artistica", "artistico"].includes(nivel)) return "artistica";
  return nivel;
};

export const normalizarNivelesClasificacion = (valores: unknown): string[] => {
  const niveles = Array.isArray(valores) ? valores.flatMap(dividirNivel) : dividirNivel(valores);
  return [...new Set(niveles.map(nivelCanonico).filter(Boolean))];
};

export const juntasParaNiveles = (niveles: string[]): string[] => {
  const conjunto = new Set(niveles);
  const juntas: string[] = [];
  if (["inicial", "primario", "especial", "adulto"].some((nivel) => conjunto.has(nivel))) {
    juntas.push(JUNTA_INICIAL_PRIMARIA_ESPECIAL_ADULTO);
  }
  if (["secundario", "media", "tecnica", "artistica"].some((nivel) => conjunto.has(nivel))) {
    juntas.push(JUNTA_MEDIA_TECNICA_ARTISTICA);
  }
  return juntas;
};
