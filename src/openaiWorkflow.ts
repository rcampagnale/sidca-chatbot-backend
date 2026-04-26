import {
  runGroqWorkflow,
  type ChatbotQueryInput,
  type ChatbotWorkflowOutput,
  type DominioBackend,
} from "./groqWorkflow.js";

export type { ChatbotQueryInput, ChatbotWorkflowOutput, DominioBackend };

export async function runChatbotWorkflow(
  input: ChatbotQueryInput
): Promise<ChatbotWorkflowOutput> {
  return runGroqWorkflow(input);
}