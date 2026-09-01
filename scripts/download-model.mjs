import { createWriteStream } from 'node:fs';
import { mkdir, rename, rm } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import { basename, dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';

/** The upstream ONNX export published by the MoGe authors. */
export const MODEL_URL =
  'https://huggingface.co/Ruicheng/moge-2-vits-normal-onnx/resolve/e50ffda41565591092adea54c6ac83d6212e1e23/model.onnx';
export const MODEL_PATH = resolve(
  process.cwd(),
  'public',
  'models',
  'moge-2-vits-normal.onnx',
);

/**
 * Download the model to a sibling temporary file and atomically publish it.
 * Keeping the temporary file next to the destination makes rename atomic on
 * the same filesystem and prevents a failed transfer from looking usable.
 */
export async function downloadModel(
  modelUrl = MODEL_URL,
  destination = MODEL_PATH,
) {
  await mkdir(dirname(destination), { recursive: true });

  const temporaryPath = `${destination}.tmp-${process.pid}-${randomUUID()}`;
  let published = false;
  try {
    const response = await fetch(modelUrl, { redirect: 'follow' });
    if (!response.ok) {
      throw new Error(`Model download failed with HTTP ${response.status} ${response.statusText}`);
    }
    if (!response.body) {
      throw new Error('Model download returned no response body.');
    }

    const body = Readable.fromWeb(response.body);
    await pipeline(body, createWriteStream(temporaryPath, { flags: 'wx' }));
    await rename(temporaryPath, destination);
    published = true;
    console.log(`Downloaded ${basename(destination)} from ${modelUrl}`);
  } finally {
    if (!published) {
      await rm(temporaryPath, { force: true });
    }
  }
}

const invokedPath = process.argv[1];
const isMain =
  typeof invokedPath === 'string' && pathToFileURL(resolve(invokedPath)).href === import.meta.url;

if (isMain) {
  try {
    await downloadModel();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(`Could not download the MoGe model: ${message}`);
    process.exitCode = 1;
  }
}
