import fs from "node:fs/promises";
import path from "node:path";
import sharp from "sharp";

const assetDirectory = path.resolve(process.cwd(), "src/email/assets");
const outputPath = path.join(assetDirectory, "logos-institucionales.png");
const separation = 48;
const logos = [
  { filename: "sidca.png", width: 264 },
  { filename: "cea.png", width: 160 },
  { filename: "internacional-educacion.png", width: 120 },
  { filename: "cgt.png", width: 80 },
];

const resized = await Promise.all(logos.map(async (logo) => {
  const inputPath = path.join(assetDirectory, logo.filename);
  const buffer = await sharp(inputPath)
    .resize({ width: logo.width, fit: "inside", withoutEnlargement: true })
    .png()
    .toBuffer();
  const metadata = await sharp(buffer).metadata();
  return {
    ...logo,
    buffer,
    height: metadata.height || 0,
  };
}));

const width = resized.reduce((total, logo) => total + logo.width, 0)
  + separation * (resized.length - 1);
const height = Math.max(...resized.map((logo) => logo.height));

await sharp({
  create: {
    width,
    height,
    channels: 4,
    background: { r: 255, g: 255, b: 255, alpha: 1 },
  },
})
  .composite(resized.map((logo, index) => ({
    input: logo.buffer,
    left: resized
      .slice(0, index)
      .reduce((total, previous) => total + previous.width + separation, 0),
    top: Math.floor((height - logo.height) / 2),
  })))
  .png({ compressionLevel: 9, adaptiveFiltering: true, palette: true })
  .toFile(outputPath);

const outputStats = await fs.stat(outputPath);
console.log(JSON.stringify({
  outputPath,
  width,
  height,
  bytes: outputStats.size,
  logos: resized.map(({ filename, width: logoWidth, height: logoHeight }) => ({
    filename,
    width: logoWidth,
    height: logoHeight,
  })),
}, null, 2));
