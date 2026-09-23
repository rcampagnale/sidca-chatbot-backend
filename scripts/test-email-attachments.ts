import { institutionalEmailAttachments } from "../src/email/institutionalEmail.js";

const attachments = institutionalEmailAttachments();
const expectedAttachments = new Map([
  ["logos-institucionales", "logos-institucionales.png"],
  ["red-web", "red-web.png"],
  ["red-facebook", "red-facebook.png"],
  ["red-youtube", "red-youtube.png"],
  ["red-instagram", "red-instagram.png"],
]);
const forbiddenContentIds = new Set([
  "sidca-logo",
  "cea-logo",
  "ie-logo",
  "cgt-logo",
]);

if (attachments.length !== 5) {
  throw new Error(`Se esperaban 5 attachments inline y se obtuvieron ${attachments.length}.`);
}

for (const attachment of attachments) {
  if (typeof attachment.filename !== "string" || !attachment.filename.trim()) {
    throw new Error("Cada attachment debe tener filename.");
  }
  if (typeof attachment.contentId !== "string" || !attachment.contentId.trim()) {
    throw new Error("Cada attachment debe tener filename y contentId.");
  }
  if (forbiddenContentIds.has(attachment.contentId)) {
    throw new Error(`El attachment ${attachment.filename} todavía usa contentId ${attachment.contentId}.`);
  }
  if (attachment.path) {
    throw new Error(`El attachment ${attachment.filename} todavía contiene path.`);
  }
  if (!Buffer.isBuffer(attachment.content) || attachment.content.length === 0) {
    throw new Error(`El attachment ${attachment.filename} no tiene un Buffer con contenido.`);
  }
  if (attachment.content.subarray(0, 8).toString("hex") !== "89504e470d0a1a0a") {
    throw new Error(`El attachment ${attachment.filename} no contiene un PNG válido.`);
  }
}

for (const [contentId, filename] of expectedAttachments) {
  const attachment = attachments.find((item) => item.contentId === contentId);
  if (!attachment) {
    throw new Error(`Falta el attachment con contentId ${contentId}.`);
  }
  if (attachment.filename !== filename) {
    throw new Error(`El contentId ${contentId} debe usar filename ${filename}.`);
  }
}

console.log("OK: 5 attachments inline PNG, sin path local, con contentId y contenido.");
