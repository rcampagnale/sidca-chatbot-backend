import { institutionalEmailAttachments } from "../src/email/institutionalEmail.js";

const attachments = institutionalEmailAttachments();

if (attachments.length !== 8) {
  throw new Error(`Se esperaban 8 attachments inline y se obtuvieron ${attachments.length}.`);
}

for (const attachment of attachments) {
  if (!attachment.filename || !attachment.contentId) {
    throw new Error("Cada attachment debe tener filename y contentId.");
  }
  if (attachment.path) {
    throw new Error(`El attachment ${attachment.filename} todavía contiene path.`);
  }
  if (!attachment.content || (Buffer.isBuffer(attachment.content) && attachment.content.length === 0)) {
    throw new Error(`El attachment ${attachment.filename} no tiene contenido.`);
  }
  if (!Buffer.isBuffer(attachment.content) || attachment.content.subarray(0, 8).toString("hex") !== "89504e470d0a1a0a") {
    throw new Error(`El attachment ${attachment.filename} no contiene un PNG válido.`);
  }
}

console.log("OK: 8 attachments inline PNG, sin path local, con contentId y contenido.");
