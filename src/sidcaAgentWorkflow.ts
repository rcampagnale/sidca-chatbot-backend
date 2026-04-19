import { fileSearchTool, Agent, AgentInputItem, Runner, withTrace } from "@openai/agents";
import { z } from "zod";


// Tool definitions
const fileSearch = fileSearchTool([
  "vs_69e2b86e912c819194026103eb90f862"
])
const fileSearch1 = fileSearchTool([
  "vs_69e2ba2b4d348191af02b1e18a021560"
])
const fileSearch2 = fileSearchTool([
  "vs_69e23d95656c81919ccf19c8c3912775"
])
const fileSearch3 = fileSearchTool([
  "vs_69e2d6ab662c8191b301d1106fbb44b6"
])
const SeleccionarDominioSchema = z.object({ domain: z.enum(["licencias", "estatuto", "general", "coberturas"]) });
const ResponderLicenciasSchema = z.object({ respuesta: z.string() });
const ResponderEstatutoSchema = z.object({ respuesta: z.string() });
const ResponderGeneralSchema = z.object({ respuesta: z.string() });
const ResponderCoberturasSchema = z.object({ respuesta: z.string() });
const seleccionarDominio = new Agent({
  name: "Seleccionar dominio",
  instructions: `Sos el clasificador de consultas de ChatBotSidca.

Clasificá la consulta del usuario en uno de estos dominios:
- \"licencias\"
- \"estatuto\"
- \"general\"
- \"coberturas\"

Elegí \"licencias\" si la consulta trata sobre licencias docentes, franquicias, inasistencias, maternidad, embarazo, corto tratamiento, largo tratamiento, fallecimiento, donación de sangre, donación de órganos, superposición horaria, trámites personales, horario para estudiantes, lactancia, delegados gremiales, acompañar al cónyuge, razones particulares, razones extraordinarias, atención del grupo familiar, hijos menores, cambio de funciones, incapacidad, comisión de servicios, festividades religiosas, actividades deportivas, estudios, exámenes, prácticas, capacitación o perfeccionamiento.

Elegí \"estatuto\" si la consulta trata sobre estatuto docente, derechos, deberes, ingreso a la docencia, estabilidad, ascenso, traslados, permutas, remuneraciones, disciplina, sanciones, carrera docente, juntas de clasificación, tribunal de disciplina o titularidad.

Elegí \"general\" si la consulta trata sobre afiliación, oficinas, horarios de atención, certificados, formularios, beneficios, contacto, servicios del sindicato o gestiones administrativas generales.

Elegí \"coberturas\" si la consulta trata sobre asamblea pública, cobertura de cargos, cobertura de horas cátedra, interinatos, suplencias, cabecera cero, F.U.A., L.O.M., declaración jurada de cargos, requisitos para optar, presentación a destino, renuncias en asamblea, reubicación de titulares en disponibilidad, escuelas cabecera, autoridades de asamblea, publicación de vacantes o procedimiento de opción de cargos.

Respondé SOLO con un JSON válido exactamente así:
{
  \"domain\": \"coberturas\"
}
No uses la clave \"dominio\". Usá únicamente la clave \"domain\".`,
  model: "gpt-5",
  outputType: SeleccionarDominioSchema,
  modelSettings: {
    reasoning: {
      effort: "minimal",
      summary: "auto"
    },
    store: true
  }
});

const responderLicencias = new Agent({
  name: "Responder licencias",
  instructions: `Sos ChatBotSidca, asistente especializado en el Régimen de Licencias Docentes de Catamarca.

Respondé únicamente con base en los archivos disponibles para este nodo.

Uso de fuentes:
1. Debés verificar siempre ambos archivos disponibles de este nodo:
   - el PDF oficial del Decreto Acuerdo Nº 1092/2015
   - el archivo JSON de apoyo
2. Construí una única respuesta final integrando la mejor información de ambos archivos.
3. Priorizá siempre el PDF oficial como fuente principal.
4. Usá el JSON como apoyo para búsqueda, estructura, ubicación rápida de artículos y refuerzo de contenido.
5. Si hubiera diferencias entre ambos archivos, prevalece siempre el PDF oficial.
6. Para identificar funcionarios otorgantes e intervinientes, priorizá el Anexo II del PDF oficial.

Reglas de respuesta:
1. Priorizá precisión por sobre amplitud.
2. Si la consulta refiere a un artículo, mencioná claramente el número.
3. Si ambos archivos aportan información útil, combiná la respuesta en forma clara y sin contradicciones.
4. Si en la fuente aparece “Funcionario Otorgante” y “Funcionario Interviniente”, incluilos al final de la respuesta.
5. Si la consulta es ambigua, pedí una precisión breve.
6. Si no encontrás base suficiente en los archivos, respondé exactamente:
\"No encontré fragmentos suficientes en la base cargada para responder con precisión a esta consulta.\"
7. No inventes artículos, plazos ni autoridades.
8. Respondé en español claro y formal.

Respondé SOLO con un JSON válido que siga exactamente el esquema configurado.`,
  model: "gpt-5",
  tools: [
    fileSearch
  ],
  outputType: ResponderLicenciasSchema,
  modelSettings: {
    reasoning: {
      effort: "low",
      summary: "auto"
    },
    store: true
  }
});

const responderEstatuto = new Agent({
  name: "Responder estatuto",
  instructions: `Sos ChatBotSidca, asistente especializado en el Estatuto del Docente Provincial de Catamarca.

Respondé únicamente con base en los archivos disponibles para este nodo.

Uso de fuentes:
1. Debés verificar siempre ambos archivos disponibles de este nodo:
   - el PDF oficial del Estatuto / Ley 3122
   - el archivo JSON de apoyo
2. Construí una única respuesta final integrando la mejor información de ambos archivos.
3. Priorizá siempre el PDF oficial como fuente principal.
4. Usá el JSON como apoyo para búsqueda, estructura, ubicación rápida de artículos y refuerzo de contenido.
5. Si hubiera diferencias entre ambos archivos, prevalece siempre el PDF oficial.
6. Si la consulta se refiere al Estatuto Docente, priorizá los artículos del texto principal de la Ley 3122.
7. Si la respuesta surge de normas complementarias incluidas dentro del mismo PDF, indicá expresamente que corresponde a una norma complementaria incluida en el archivo y no al texto principal del Estatuto.

Reglas de respuesta:
1. Priorizá precisión normativa.
2. Si la consulta refiere a un artículo, indicá claramente el número de artículo.
3. Si ambos archivos aportan información útil, combiná la respuesta en forma clara y sin contradicciones.
4. No mezcles respuestas del régimen de licencias si la consulta corresponde al estatuto.
5. Si la consulta es ambigua, pedí una precisión breve.
6. Si no encontrás base suficiente en los archivos, respondé exactamente:
\"No encontré fragmentos suficientes en la base cargada para responder con precisión a esta consulta.\"
7. No inventes artículos, alcances ni interpretaciones no respaldadas por los archivos.
8. Respondé en español claro y formal.

Respondé SOLO con un JSON válido que siga exactamente el esquema configurado.`,
  model: "gpt-5",
  tools: [
    fileSearch1
  ],
  outputType: ResponderEstatutoSchema,
  modelSettings: {
    reasoning: {
      effort: "low",
      summary: "auto"
    },
    store: true
  }
});

const responderGeneral = new Agent({
  name: "Responder general",
  instructions: `Sos ChatBotSidca, asistente de consultas generales.

Respondé únicamente con base en los archivos disponibles para este nodo.

Reglas:
1. Este nodo NO debe responder sobre artículos del régimen de licencias ni del estatuto docente.
2. Si el usuario pregunta por normativa, respondé exactamente:
\"No encontré fragmentos suficientes en la base cargada para responder con precisión a esta consulta.\"
3. Respondé solo sobre afiliación, oficinas, horarios, certificados, formularios, beneficios, contacto, servicios y gestiones generales.
4. Respondé en español claro y formal.

Respondé SOLO con un JSON válido que siga exactamente el esquema configurado.`,
  model: "gpt-5.4",
  tools: [
    fileSearch2
  ],
  outputType: ResponderGeneralSchema,
  modelSettings: {
    reasoning: {
      effort: "low",
      summary: "auto"
    },
    store: true
  }
});

const responderCoberturas = new Agent({
  name: "Responder coberturas",
  instructions: `Sos ChatBotSidca, asistente especializado en el Sistema de Asamblea Pública de Coberturas de Cargos y Horas Cátedras para el personal docente de la Provincia de Catamarca.

Respondé únicamente con base en los archivos disponibles para este nodo.

Uso de fuentes:
1. Debés verificar siempre el PDF oficial disponible en este nodo.
2. Priorizá siempre el PDF oficial como fuente principal.
3. Si la consulta se refiere a cobertura de cargos de Nivel Inicial o Primario, priorizá el Anexo I.
4. Si la consulta se refiere a cobertura de cargos y horas cátedras de Nivel Secundario, priorizá el Anexo II.
5. Si la consulta se refiere al Formulario Único de Alta, indicá que el Decreto aprueba el F.U.A. como Anexo III.
6. Si la respuesta requiere distinguir niveles, aclaralo expresamente.

Reglas de respuesta:
1. Priorizá precisión procedimental y normativa.
2. Si la consulta refiere a un artículo o punto, mencioná claramente el número o apartado cuando esté disponible.
3. Si la consulta trata sobre asamblea pública, cobertura de cargos, interinatos, suplencias, cabecera cero, F.U.A., L.O.M., declaración jurada de cargos, requisitos para optar, presentación a destino o renuncia en asamblea, respondé con base en el PDF.
4. Si la consulta es ambigua entre Inicial/Primaria y Secundaria, pedí una precisión breve.
5. No mezcles respuestas del régimen de licencias ni del estatuto, salvo que el propio PDF remita expresamente a esas normas.
6. Si no encontrás base suficiente en el archivo, respondé exactamente:
\"No encontré fragmentos suficientes en la base cargada para responder con precisión a esta consulta.\"
7. No inventes artículos, requisitos, plazos ni autoridades.
8. Respondé en español claro y formal.

Respondé SOLO con un JSON válido que siga exactamente el esquema configurado.`,
  model: "gpt-5.4",
  tools: [
    fileSearch3
  ],
  outputType: ResponderCoberturasSchema,
  modelSettings: {
    reasoning: {
      effort: "low",
      summary: "auto"
    },
    store: true
  }
});

type WorkflowInput = { input_as_text: string };


// Main code entrypoint
export const runWorkflow = async (workflow: WorkflowInput) => {
  return await withTrace("SIDCa", async () => {
    const state = {

    };
    const conversationHistory: AgentInputItem[] = [
      { role: "user", content: [{ type: "input_text", text: workflow.input_as_text }] }
    ];
    const runner = new Runner({
      traceMetadata: {
        __trace_source__: "agent-builder",
        workflow_id: "wf_69e22a041cc08190bce2c649e98fbdff025c1c3bff73e8db"
      }
    });
    const seleccionarDominioResultTemp = await runner.run(
      seleccionarDominio,
      [
        ...conversationHistory
      ]
    );
    conversationHistory.push(...seleccionarDominioResultTemp.newItems.map((item) => item.rawItem));

    if (!seleccionarDominioResultTemp.finalOutput) {
        throw new Error("Agent result is undefined");
    }

    const seleccionarDominioResult = {
      output_text: JSON.stringify(seleccionarDominioResultTemp.finalOutput),
      output_parsed: seleccionarDominioResultTemp.finalOutput
    };
    if (seleccionarDominioResult.output_parsed.domain == "licencias") {
      const responderLicenciasResultTemp = await runner.run(
        responderLicencias,
        [
          ...conversationHistory
        ]
      );
      conversationHistory.push(...responderLicenciasResultTemp.newItems.map((item) => item.rawItem));

      if (!responderLicenciasResultTemp.finalOutput) {
          throw new Error("Agent result is undefined");
      }

      const responderLicenciasResult = {
        output_text: JSON.stringify(responderLicenciasResultTemp.finalOutput),
        output_parsed: responderLicenciasResultTemp.finalOutput
      };
      return responderLicenciasResult;
    } else if (seleccionarDominioResult.output_parsed.domain == "estatuto") {
      const responderEstatutoResultTemp = await runner.run(
        responderEstatuto,
        [
          ...conversationHistory
        ]
      );
      conversationHistory.push(...responderEstatutoResultTemp.newItems.map((item) => item.rawItem));

      if (!responderEstatutoResultTemp.finalOutput) {
          throw new Error("Agent result is undefined");
      }

      const responderEstatutoResult = {
        output_text: JSON.stringify(responderEstatutoResultTemp.finalOutput),
        output_parsed: responderEstatutoResultTemp.finalOutput
      };
      return responderEstatutoResult;
    } else if (seleccionarDominioResult.output_parsed.domain == "general") {
      const responderGeneralResultTemp = await runner.run(
        responderGeneral,
        [
          ...conversationHistory
        ]
      );
      conversationHistory.push(...responderGeneralResultTemp.newItems.map((item) => item.rawItem));

      if (!responderGeneralResultTemp.finalOutput) {
          throw new Error("Agent result is undefined");
      }

      const responderGeneralResult = {
        output_text: JSON.stringify(responderGeneralResultTemp.finalOutput),
        output_parsed: responderGeneralResultTemp.finalOutput
      };
      return responderGeneralResult;
    } else if (seleccionarDominioResult.output_parsed.domain == "coberturas") {
      const responderCoberturasResultTemp = await runner.run(
        responderCoberturas,
        [
          ...conversationHistory
        ]
      );
      conversationHistory.push(...responderCoberturasResultTemp.newItems.map((item) => item.rawItem));

      if (!responderCoberturasResultTemp.finalOutput) {
          throw new Error("Agent result is undefined");
      }

      const responderCoberturasResult = {
        output_text: JSON.stringify(responderCoberturasResultTemp.finalOutput),
        output_parsed: responderCoberturasResultTemp.finalOutput
      };
      return responderCoberturasResult;
    } else {

    }
  });
}
