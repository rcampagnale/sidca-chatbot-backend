import type {
  InstitutionalEmailOptions,
  InstitutionalEmailSection,
} from "./institutionalEmail.js";

/**
 * Contenido institucional compartido por los correos que anuncian una
 * afiliación activa: la bienvenida de una afiliación nueva y la aprobación
 * de una reafiliación.
 *
 * Vive en un único lugar a propósito. Si mañana se suma una prestación o
 * cambia un texto, se corrige acá y los dos correos quedan alineados; antes
 * la lista estaba escrita dentro de un solo builder y habría quedado
 * desincronizada al copiarla.
 *
 * NO se usa en los correos de trámite en curso ni de rechazo: en esos casos
 * la afiliación todavía no está activa y listar beneficios sería prometer
 * algo que la persona aún no tiene.
 */
export const SECCIONES_SERVICIOS: InstitutionalEmailSection[] = [
  {
    titulo: "Acompañamiento",
    items: [
      {
        icono: "🤝",
        nombre: "Asesoramiento gremial",
        detalle:
          "Orientación sobre derechos laborales, reclamos, titularización, trámites y gestiones sindicales.",
      },
      {
        icono: "⚖️",
        nombre: "Departamento Jurídico",
        detalle:
          "Acompañamiento legal ante situaciones vinculadas con la actividad docente y laboral.",
      },
    ],
  },
  {
    titulo: "Beneficios y servicios",
    items: [
      {
        icono: "🎓",
        nombre: "Capacitaciones",
        detalle:
          "Propuestas presenciales y virtuales, cursos, congresos y formación continua para el desarrollo profesional docente.",
      },
      {
        icono: "✈️",
        nombre: "Turismo y viajes",
        detalle:
          "Viajes, propuestas recreativas y beneficios turísticos gestionados por SiDCA.",
      },
      {
        icono: "🏷️",
        nombre: "Convenios con empresas",
        detalle:
          "Descuentos y beneficios en empresas y comercios adheridos utilizando tu credencial.",
      },
      {
        icono: "🏨",
        nombre: "Convenios hoteleros",
        detalle:
          "Beneficios de alojamiento y acuerdos especiales con hoteles en diferentes provincias y localidades.",
      },
      {
        icono: "🏠",
        nombre: "Casa del Docente",
        detalle: "Alojamiento temporal y servicios destinados a afiliados y afiliadas.",
      },
      {
        icono: "🩺",
        nombre: "Médica gremial",
        detalle:
          "Atención exclusiva para afiliados y afiliadas, con consultas presenciales y virtuales.",
      },
      {
        icono: "🧮",
        nombre: "Simulador de sueldo",
        detalle:
          "Herramienta disponible en la APP SiDCA para estimar y comprender mejor tu liquidación salarial.",
      },
    ],
  },
];

/** Franja destacada con lo que resuelve la APP. */
export const HIGHLIGHTS_APP: InstitutionalEmailOptions["highlights"] = {
  titulo: "Todo en tu APP SiDCA",
  items: [
    "Credencial digital",
    "Convenios y descuentos",
    "Cursos y capacitaciones",
    "Certificados y constancias",
    "Registro de asistencia",
    "Aula Virtual SiDCA",
  ],
};

/** Segunda línea del cierre institucional, común a ambos correos. */
export const FRASE_CIERRE =
  "Seguimos construyendo un sindicato presente, cercano y comprometido con quienes educan.";
