import * as ort from 'onnxruntime-web/webgpu';

import { checkWebGpuSupport } from '../platform/webgpu';
import { postprocessMoGe } from './postprocess';
import { preprocessImage, type MoGeImageSource } from './preprocess';
import type {
  MoGeInferenceResult,
  MoGeResult,
  RawMoGeOutputs,
  StageTimings,
} from './types';

export const DEFAULT_MOGE_MODEL_URL =
  import.meta.env.VITE_MOGE_MODEL_URL ||
  'https://huggingface.co/Ruicheng/moge-2-vits-normal-onnx/resolve/e50ffda41565591092adea54c6ac83d6212e1e23/model.onnx';
export const DEFAULT_MOGE_NUM_TOKENS = 1800;

const EXPECTED_INPUT_NAMES = ['image', 'num_tokens'] as const;
const EXPECTED_OUTPUT_BASE_NAMES = ['points', 'normal', 'mask'] as const;
// The published Ruicheng graph calls this output `scale`; the name used in
// Microsoft's export example is `metric_scale`. Both carry the same scalar
// exp'd metric scale, so accept either exact four-output contract.
const SCALE_OUTPUT_NAMES = ['scale', 'metric_scale'] as const;

type ScaleOutputName = (typeof SCALE_OUTPUT_NAMES)[number];
type ExpectedOutputName = (typeof EXPECTED_OUTPUT_BASE_NAMES)[number] | ScaleOutputName;

export interface MoGeInferenceOptions {
  /** URL or Vite-served path for the official MoGe-2 ONNX model. */
  modelUrl?: string;
  /** Number of ViT tokens passed to the dynamic ONNX graph. */
  numTokens?: number;
}

/**
 * Structural metadata shape used by ORT Web. Keeping this structural avoids
 * coupling the application to an internal ORT metadata type while still
 * validating every field we depend on.
 */
interface TensorMetadataLike {
  type?: unknown;
  shape?: unknown;
}

type MetadataMap = Record<string, unknown>;

interface OrtTensorLike {
  data?: unknown;
  dims?: unknown;
}

type OrtOutputMap = Record<string, OrtTensorLike | undefined>;

function formatNames(names: readonly string[]): string {
  return names.length > 0 ? names.join(', ') : '(none)';
}

function assertExactNames(
  actual: unknown,
  expected: readonly string[],
  kind: 'input' | 'output',
): asserts actual is readonly string[] {
  if (!Array.isArray(actual) || actual.some((name) => typeof name !== 'string')) {
    throw new Error(
      `MoGe ONNX session is missing ${kind} names; expected ${formatNames(expected)}.`,
    );
  }

  const names = actual as readonly string[];
  const actualSet = new Set(names);
  const expectedSet = new Set(expected);
  const missing = expected.filter((name) => !actualSet.has(name));
  const unexpected = names.filter((name) => !expectedSet.has(name));
  if (names.length !== expected.length || missing.length > 0 || unexpected.length > 0) {
    const suffix = [
      missing.length > 0 ? `missing ${missing.join(', ')}` : '',
      unexpected.length > 0 ? `unexpected ${unexpected.join(', ')}` : '',
    ]
      .filter(Boolean)
      .join('; ');
    throw new Error(
      `MoGe ONNX session has unexpected ${kind} names. Expected [${formatNames(expected)}], got [${formatNames(names)}]${suffix ? ` (${suffix})` : ''}.`,
    );
  }
}

function metadataMap(value: unknown, kind: 'input' | 'output'): MetadataMap {
  if (value === null || typeof value !== 'object') {
    throw new Error(
      `MoGe ONNX session is missing ${kind} metadata; the model cannot be validated safely.`,
    );
  }

  // ORT Web exposes metadata as a named array (`ValueMetadata[]`), while a
  // few adapters/test doubles expose a name-keyed map. Normalize both forms
  // before validating the model contract.
  if (Array.isArray(value)) {
    const normalized: MetadataMap = {};
    for (const entry of value) {
      if (entry === null || typeof entry !== 'object' || Array.isArray(entry)) {
        throw new Error(`MoGe ONNX ${kind} metadata contains a malformed entry.`);
      }
      const name = (entry as { name?: unknown }).name;
      if (typeof name !== 'string' || name.length === 0) {
        throw new Error(`MoGe ONNX ${kind} metadata contains an entry without a name.`);
      }
      if (Object.prototype.hasOwnProperty.call(normalized, name)) {
        throw new Error(`MoGe ONNX ${kind} metadata contains duplicate '${name}' entries.`);
      }
      normalized[name] = entry;
    }
    return normalized;
  }

  return value as MetadataMap;
}

function tensorMetadata(
  metadata: MetadataMap,
  name: string,
  kind: 'input' | 'output',
): TensorMetadataLike {
  const entry = metadata[name];
  if (entry === null || typeof entry !== 'object' || Array.isArray(entry)) {
    throw new Error(
      `MoGe ONNX ${kind} metadata is missing the '${name}' entry. Expected metadata with type and shape.`,
    );
  }
  const typed = entry as TensorMetadataLike;
  if (typeof typed.type !== 'string') {
    throw new Error(`MoGe ONNX ${kind} '${name}' metadata has no tensor type.`);
  }
  if (!Array.isArray(typed.shape)) {
    throw new Error(`MoGe ONNX ${kind} '${name}' metadata has no tensor shape.`);
  }
  return typed;
}

function assertMetadataNames(
  metadata: MetadataMap,
  expected: readonly string[],
  kind: 'input' | 'output',
): void {
  const names = Object.keys(metadata);
  const expectedSet = new Set(expected);
  const unexpected = names.filter((name) => !expectedSet.has(name));
  const missing = expected.filter((name) => !Object.prototype.hasOwnProperty.call(metadata, name));
  if (missing.length > 0 || unexpected.length > 0 || names.length !== expected.length) {
    throw new Error(
      `MoGe ONNX ${kind} metadata names do not match the session contract. Expected [${formatNames(expected)}], got [${formatNames(names)}].${missing.length > 0 ? ` Missing: ${missing.join(', ')}.` : ''}${unexpected.length > 0 ? ` Unexpected: ${unexpected.join(', ')}.` : ''}`,
    );
  }
}

function assertOutputNames(actual: unknown): ScaleOutputName {
  if (!Array.isArray(actual) || actual.some((name) => typeof name !== 'string')) {
    throw new Error(
      `MoGe ONNX session is missing output names; expected [${formatNames(EXPECTED_OUTPUT_BASE_NAMES)}] plus exactly one of [${formatNames(SCALE_OUTPUT_NAMES)}].`,
    );
  }

  const names = actual as readonly string[];
  const required = new Set<string>(EXPECTED_OUTPUT_BASE_NAMES);
  const scaleNames = new Set<string>(SCALE_OUTPUT_NAMES);
  const unexpected = names.filter((name) => !required.has(name) && !scaleNames.has(name));
  const missing = EXPECTED_OUTPUT_BASE_NAMES.filter((name) => !names.includes(name));
  const scales = SCALE_OUTPUT_NAMES.filter((name) => names.includes(name));
  if (
    names.length !== EXPECTED_OUTPUT_BASE_NAMES.length + 1 ||
    missing.length > 0 ||
    scales.length !== 1 ||
    unexpected.length > 0
  ) {
    throw new Error(
      `MoGe ONNX session has unexpected output names. Expected [${formatNames(EXPECTED_OUTPUT_BASE_NAMES)}] plus exactly one of [${formatNames(SCALE_OUTPUT_NAMES)}], got [${formatNames(names)}].${missing.length > 0 ? ` Missing: ${missing.join(', ')}.` : ''}${scales.length === 0 ? ' Missing scale output.' : ''}${scales.length > 1 ? ' Both scale output aliases were returned.' : ''}${unexpected.length > 0 ? ` Unexpected: ${unexpected.join(', ')}.` : ''}`,
    );
  }
  const scaleName = scales[0];
  if (scaleName === undefined) {
    // The branch above proves this cannot happen, but retaining an explicit
    // guard keeps this function safe with noUncheckedIndexedAccess enabled.
    throw new Error('MoGe ONNX session did not expose a scale output.');
  }
  return scaleName;
}

function assertFloat32Metadata(
  metadata: TensorMetadataLike,
  name: string,
  kind: 'input' | 'output',
): void {
  if (metadata.type !== 'float32') {
    throw new Error(
      `MoGe ONNX ${kind} '${name}' must have float32 metadata, got ${String(metadata.type)}.`,
    );
  }
}

function assertInt64Metadata(metadata: TensorMetadataLike, name: string): void {
  if (metadata.type !== 'int64') {
    throw new Error(
      `MoGe ONNX input '${name}' must have int64 metadata, got ${String(metadata.type)}.`,
    );
  }
}

function assertRank(
  metadata: TensorMetadataLike,
  rank: number,
  label: string,
): readonly unknown[] {
  const shape = metadata.shape;
  if (!Array.isArray(shape) || shape.length !== rank) {
    throw new Error(
      `MoGe ONNX ${label} must have rank ${rank}; metadata shape is ${JSON.stringify(shape)}.`,
    );
  }
  return shape;
}

function assertStaticDimension(value: unknown, expected: number, label: string): void {
  // ONNX Runtime reports dynamic dimensions as symbolic strings. Some model
  // exports use null/-1/0 for an unknown dimension, which is also valid here.
  if (typeof value === 'number') {
    if (!Number.isInteger(value) || value < -1) {
      throw new Error(`MoGe ONNX ${label} has invalid dimension metadata ${String(value)}.`);
    }
    if (value > 0 && value !== expected) {
      throw new Error(
        `MoGe ONNX ${label} has static dimension ${value}, but inference requires ${expected}.`,
      );
    }
  } else if (typeof value !== 'string' && value !== null) {
    throw new Error(`MoGe ONNX ${label} has unexpected dimension metadata ${String(value)}.`);
  }
}

function validateStaticShape(
  shape: readonly unknown[],
  expected: readonly (number | null)[],
  label: string,
): void {
  if (shape.length !== expected.length) {
    throw new Error(
      `MoGe ONNX ${label} has rank ${shape.length}; expected rank ${expected.length}.`,
    );
  }
  shape.forEach((dimension, index) => {
    const wanted = expected[index];
    if (wanted === undefined) {
      throw new Error(`MoGe ONNX ${label} has an invalid expected shape.`);
    }
    if (wanted !== null) {
      assertStaticDimension(dimension, wanted, `${label}[${index}]`);
    } else if (
      typeof dimension !== 'number' &&
      typeof dimension !== 'string' &&
      dimension !== null
    ) {
      throw new Error(
        `MoGe ONNX ${label}[${index}] has unexpected dimension metadata ${String(dimension)}.`,
      );
    }
  });
}

function validateSessionMetadata(session: {
  inputMetadata?: unknown;
  outputMetadata?: unknown;
}, scaleName: ScaleOutputName): void {
  const inputs = metadataMap(session.inputMetadata, 'input');
  const outputs = metadataMap(session.outputMetadata, 'output');
  assertMetadataNames(inputs, EXPECTED_INPUT_NAMES, 'input');
  assertMetadataNames(outputs, [...EXPECTED_OUTPUT_BASE_NAMES, scaleName], 'output');

  const image = tensorMetadata(inputs, 'image', 'input');
  assertFloat32Metadata(image, 'image', 'input');
  const imageShape = assertRank(image, 4, "input 'image'");
  validateStaticShape(imageShape, [1, 3, null, null], "input 'image'");

  const token = tensorMetadata(inputs, 'num_tokens', 'input');
  assertInt64Metadata(token, 'num_tokens');
  // The official dynamic export receives torch.tensor(1800), i.e. a scalar
  // int64, rather than a one-element vector.
  assertRank(token, 0, "input 'num_tokens'");

  const points = tensorMetadata(outputs, 'points', 'output');
  assertFloat32Metadata(points, 'points', 'output');
  validateStaticShape(
    assertRank(points, 4, "output 'points'"),
    [1, null, null, 3],
    "output 'points'",
  );

  const normal = tensorMetadata(outputs, 'normal', 'output');
  assertFloat32Metadata(normal, 'normal', 'output');
  validateStaticShape(
    assertRank(normal, 4, "output 'normal'"),
    [1, null, null, 3],
    "output 'normal'",
  );

  const mask = tensorMetadata(outputs, 'mask', 'output');
  assertFloat32Metadata(mask, 'mask', 'output');
  validateStaticShape(
    assertRank(mask, 3, "output 'mask'"),
    [1, null, null],
    "output 'mask'",
  );

  const metricScale = tensorMetadata(outputs, scaleName, 'output');
  assertFloat32Metadata(metricScale, scaleName, 'output');
  validateStaticShape(
    assertRank(metricScale, 1, `output '${scaleName}'`),
    [1],
    `output '${scaleName}'`,
  );
}

function tensorDims(tensor: OrtTensorLike, name: string): readonly number[] {
  if (!Array.isArray(tensor.dims) || tensor.dims.some((dim) => !Number.isInteger(dim))) {
    throw new Error(`MoGe ONNX output '${name}' has no valid runtime dimensions.`);
  }
  const dims = tensor.dims as number[];
  if (dims.some((dim) => dim <= 0)) {
    throw new Error(`MoGe ONNX output '${name}' has non-positive runtime dimensions.`);
  }
  return dims;
}

function floatData(
  tensor: OrtTensorLike,
  name: string,
  allowMaskedSentinels = false,
): Float32Array {
  if (!(tensor.data instanceof Float32Array)) {
    throw new Error(
      `MoGe ONNX output '${name}' is not a float32 typed array; check model output metadata and WebGPU execution.`,
    );
  }
  const data = tensor.data;
  for (let index = 0; index < data.length; index += 1) {
    if (!allowMaskedSentinels && !Number.isFinite(data[index])) {
      throw new Error(`MoGe ONNX output '${name}' contains NaN or an infinite value at index ${index}.`);
    }
  }
  return new Float32Array(data);
}

function elementCount(dims: readonly number[]): number {
  return dims.reduce((count, dim) => count * dim, 1);
}

function outputTensor(
  outputs: OrtOutputMap,
  name: ExpectedOutputName,
): OrtTensorLike {
  const tensor = outputs[name];
  if (tensor === undefined || tensor === null || typeof tensor !== 'object') {
    throw new Error(`MoGe ONNX inference did not return the required '${name}' output.`);
  }
  return tensor;
}

function validateRuntimeOutputNames(outputs: OrtOutputMap, scaleName: ScaleOutputName): void {
  if (outputs === null || typeof outputs !== 'object' || Array.isArray(outputs)) {
    throw new Error('MoGe ONNX inference returned no output map.');
  }
  const names = Object.keys(outputs);
  const expected = new Set<string>([...EXPECTED_OUTPUT_BASE_NAMES, scaleName]);
  const unexpected = names.filter((name) => !expected.has(name));
  const missing = [...EXPECTED_OUTPUT_BASE_NAMES, scaleName].filter(
    (name) => !Object.prototype.hasOwnProperty.call(outputs, name),
  );
  if (
    missing.length > 0 ||
    unexpected.length > 0 ||
    names.length !== EXPECTED_OUTPUT_BASE_NAMES.length + 1
  ) {
    throw new Error(
      `MoGe ONNX inference returned unexpected outputs. Expected [${formatNames([...EXPECTED_OUTPUT_BASE_NAMES, scaleName])}], got [${formatNames(names)}].${missing.length > 0 ? ` Missing: ${missing.join(', ')}.` : ''}${unexpected.length > 0 ? ` Unexpected: ${unexpected.join(', ')}.` : ''}`,
    );
  }
}

function validateRuntimeShape(
  tensor: OrtTensorLike,
  name: ExpectedOutputName,
  expectedShape: readonly number[],
  allowMaskedSentinels = false,
): Float32Array {
  const dims = tensorDims(tensor, name);
  if (dims.length !== expectedShape.length || dims.some((dim, index) => dim !== expectedShape[index])) {
    throw new Error(
      `MoGe ONNX output '${name}' has runtime shape [${dims.join(', ')}], expected [${expectedShape.join(', ')}].`,
    );
  }
  const data = floatData(tensor, name, allowMaskedSentinels);
  const expectedElements = elementCount(expectedShape);
  if (data.length !== expectedElements) {
    throw new Error(
      `MoGe ONNX output '${name}' contains ${data.length} values for shape [${expectedShape.join(', ')}] (${expectedElements} expected).`,
    );
  }
  return data;
}

function now(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

function validateNumTokens(value: number): number {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`MoGe numTokens must be a positive safe integer; received ${String(value)}.`);
  }
  return value;
}

/**
 * A reusable WebGPU-only ONNX Runtime session for the official MoGe-2 graph.
 * The session is created lazily and reused for every subsequent image.
 */
export class MoGeInference {
  private readonly modelUrl: string;
  private readonly numTokens: number;
  private session: ort.InferenceSession | undefined;
  private loadingPromise: Promise<ort.InferenceSession> | undefined;
  private disposed = false;
  private _lastTimings: StageTimings | undefined;
  private _modelLoadMs = 0;
  private activeInferences = 0;
  private readonly idleWaiters = new Set<() => void>();

  public constructor(options: MoGeInferenceOptions | string = {}) {
    const normalized = typeof options === 'string' ? { modelUrl: options } : options;
    const modelUrl = normalized.modelUrl ?? DEFAULT_MOGE_MODEL_URL;
    if (typeof modelUrl !== 'string' || modelUrl.trim().length === 0) {
      throw new Error('MoGe modelUrl must be a non-empty URL or served model path.');
    }
    this.modelUrl = modelUrl;
    this.numTokens = validateNumTokens(normalized.numTokens ?? DEFAULT_MOGE_NUM_TOKENS);
  }

  /** Timings from the most recent successful inference, if any. */
  public get lastTimings(): StageTimings | undefined {
    return this._lastTimings;
  }

  public get modelLoadMs(): number {
    return this._modelLoadMs;
  }

  /** Load and validate one long-lived WebGPU session. */
  public load(): Promise<ort.InferenceSession> {
    if (this.disposed) {
      return Promise.reject(new Error('MoGe inference has been disposed.'));
    }
    if (this.session) {
      return Promise.resolve(this.session);
    }
    if (this.loadingPromise) {
      return this.loadingPromise;
    }

    const pending = this.createSession();
    this.loadingPromise = pending;
    void pending.then(
      () => {
        if (this.loadingPromise === pending) {
          this.loadingPromise = undefined;
        }
      },
      () => {
        if (this.loadingPromise === pending) {
          this.loadingPromise = undefined;
        }
      },
    );
    return pending;
  }

  private async createSession(): Promise<ort.InferenceSession> {
    const loadStart = now();
    const diagnostics = await checkWebGpuSupport();
    if (!diagnostics.supported) {
      throw new Error(`MoGe requires WebGPU. ${diagnostics.reason ?? 'No GPU adapter is available.'}`);
    }

    let session: ort.InferenceSession | undefined;
    try {
      // WebGPU is deliberately the only execution provider. Do not add a
      // WASM/CPU fallback: silently switching providers changes performance
      // and can hide unsupported-browser failures.
      session = await ort.InferenceSession.create(this.modelUrl, {
        executionProviders: ['webgpu'],
      });

      assertExactNames(session.inputNames, EXPECTED_INPUT_NAMES, 'input');
      const scaleName = assertOutputNames(session.outputNames);
      validateSessionMetadata(session, scaleName);

      if (this.disposed) {
        throw new Error('MoGe inference was disposed while the model was loading.');
      }

      this.session = session;
      this._modelLoadMs = now() - loadStart;
      return session;
    } catch (error) {
      if (session && session !== this.session) {
        try {
          await session.release();
        } catch {
          // Preserve the actionable validation/creation error below.
        }
      }
      if (error instanceof Error) {
        throw error;
      }
      throw new Error(`Could not create the MoGe ONNX WebGPU session: ${String(error)}`, {
        cause: error,
      });
    }
  }

  /**
   * Run one image through preprocessing, the reused ONNX session, and the
   * canonical postprocessor. Pixel data remains in browser memory.
   */
  public async infer(source: MoGeImageSource): Promise<MoGeInferenceResult> {
    if (this.disposed) throw new Error('MoGe inference has been disposed.');
    this.activeInferences += 1;
    try {
      return await this.runInference(source);
    } finally {
      this.activeInferences -= 1;
      if (this.activeInferences === 0) {
        for (const resolve of this.idleWaiters) resolve();
        this.idleWaiters.clear();
      }
    }
  }

  private async runInference(source: MoGeImageSource): Promise<MoGeInferenceResult> {
    const preprocessStart = now();
    const preprocessed = await preprocessImage(source);
    const preprocessMs = now() - preprocessStart;

    let inferenceImage = preprocessed.inferenceImage;
    try {
      const session = await this.load();
      const imageMetadata = tensorMetadata(
        metadataMap(session.inputMetadata, 'input'),
        'image',
        'input',
      );
      const imageShape = assertRank(imageMetadata, 4, "input 'image'");
      assertStaticDimension(imageShape[2], preprocessed.height, "input 'image'[2]");
      assertStaticDimension(imageShape[3], preprocessed.width, "input 'image'[3]");

      const imageTensor = new ort.Tensor(
        'float32',
        preprocessed.data,
        [1, 3, preprocessed.height, preprocessed.width],
      );
      // ORT Web accepts a scalar int64 tensor when dims is [], matching the
      // torch.tensor(1800) input used by the official dynamic export.
      const tokenTensor = new ort.Tensor(
        'int64',
        BigInt64Array.of(BigInt(this.numTokens)),
        [],
      );

      const inferenceStart = now();
      const outputs = (await session.run({
        image: imageTensor,
        num_tokens: tokenTensor,
      })) as unknown as OrtOutputMap;
      const inferenceMs = now() - inferenceStart;

      const scaleName = assertOutputNames(session.outputNames);
      validateRuntimeOutputNames(outputs, scaleName);
      const points = validateRuntimeShape(
        outputTensor(outputs, 'points'),
        'points',
        [1, preprocessed.height, preprocessed.width, 3],
        true,
      );
      const normal = validateRuntimeShape(
        outputTensor(outputs, 'normal'),
        'normal',
        [1, preprocessed.height, preprocessed.width, 3],
        true,
      );
      const mask = validateRuntimeShape(
        outputTensor(outputs, 'mask'),
        'mask',
        [1, preprocessed.height, preprocessed.width],
      );
      const metricScaleTensor = validateRuntimeShape(
        outputTensor(outputs, scaleName),
        scaleName,
        [1],
      );
      const metricScale = metricScaleTensor[0];
      if (metricScale === undefined || !Number.isFinite(metricScale) || metricScale <= 0) {
        throw new Error(
          `MoGe ONNX output '${scaleName}' must be a positive finite value; received ${String(metricScale)}.`,
        );
      }

      // The official graph emits [1,H,W,C] (or [1,H,W] for mask) in C-order;
      // removing the batch dimension leaves the required row-major HWC/HW
      // arrays without a transpose or an implicit layout guess.
      const raw: RawMoGeOutputs = {
        width: preprocessed.width,
        height: preprocessed.height,
        points: points.subarray(0),
        normal: normal.subarray(0),
        mask: mask.subarray(0),
        metricScale,
      };

      const postprocessStart = now();
      const result: MoGeResult = postprocessMoGe(raw);
      const postprocessMs = now() - postprocessStart;
      const timings: StageTimings = {
        modelLoadMs: this._modelLoadMs,
        preprocessMs,
        inferenceMs,
        postprocessMs,
      };
      this._lastTimings = timings;
      return { result, inferenceImage, timings };
    } catch (error) {
      // The caller owns a bitmap returned by a successful call. On failure,
      // release only bitmaps allocated by preprocessing; a caller-supplied
      // ImageBitmap remains caller-owned.
      if (preprocessed.ownsInferenceImage) {
        try {
          inferenceImage.close();
        } catch {
          // Some test doubles and detached bitmaps throw on close.
        }
      }
      throw error;
    }
  }

  /** Release the ORT session and make a future load create a fresh one. */
  public async dispose(): Promise<void> {
    this.disposed = true;
    const pending = this.loadingPromise;
    if (pending) {
      try {
        await pending;
      } catch {
        // Loading errors are already reported to the caller of load().
      }
    }
    if (this.activeInferences > 0) {
      await new Promise<void>((resolve) => this.idleWaiters.add(resolve));
    }

    const session = this.session;
    this.session = undefined;
    this.loadingPromise = undefined;
    if (session) {
      await session.release();
    }
  }
}
