/**
 * Small, dependency-free geometry helpers used by the MoGe-2 post processor.
 *
 * The array layout in this file is deliberately the same as the ONNX output:
 * points and normals are row-major HWC (index `((y * width + x) * 3) + c`),
 * while masks and depth maps are row-major HW (`y * width + x`).
 *
 * The formulas below follow `moge/utils/geometry_torch.py` in microsoft/MoGe
 * (notably `normalized_view_plane_uv` and `recover_focal_shift`).  The
 * upstream implementation delegates the one-dimensional solve to SciPy; this
 * file evaluates the same residual while minimizing it with a deterministic
 * scalar search so the browser build has no SciPy dependency.
 */

export type NumericArray = ArrayLike<number>;
export type MaskArray = ArrayLike<number | boolean>;

export interface RecoverFocalShiftOptions {
  /** Target width of the nearest-neighbour solve (upstream default: 64). */
  downsampleWidth?: number;
  /** Target height of the nearest-neighbour solve (upstream default: 64). */
  downsampleHeight?: number;
  /**
   * If supplied, keep focal fixed and solve only for the Z shift.  This is the
   * equivalent of the `focal=` argument accepted by the upstream helper.
   */
  focal?: number;
}

export interface FocalShiftRecovery {
  /** Focal length relative to half the image diagonal. */
  focal: number;
  /** Additive translation along camera +Z, in affine-point units. */
  shift: number;
}

export interface FiniteDepthStats {
  count: number;
  min: number;
  max: number;
  mean: number;
  p01: number;
  p05: number;
  p50: number;
  p95: number;
  p99: number;
}

export interface RecoverDimensions {
  width: number;
  height: number;
  mask?: MaskArray;
  options?: RecoverFocalShiftOptions;
}

interface Sample {
  u: number;
  v: number;
  x: number;
  y: number;
  z: number;
}

interface ShiftEvaluation {
  cost: number;
  focal: number;
}

const DEFAULT_DOWNSAMPLE = 64;
const GOLDEN_RATIO = (1 + Math.sqrt(5)) / 2;

function assertPositiveInteger(value: number, name: string): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new RangeError(`${name} must be a positive integer; received ${String(value)}`);
  }
}

function assertFinitePositive(value: number, name: string): void {
  if (!Number.isFinite(value) || value <= 0) {
    throw new RangeError(`${name} must be finite and positive; received ${String(value)}`);
  }
}

function isMaskValid(mask: MaskArray, index: number): boolean {
  const value = mask[index];
  if (typeof value === "boolean") {
    return value;
  }
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

function validatePointArray(points: NumericArray, width: number, height: number): void {
  assertPositiveInteger(width, "width");
  assertPositiveInteger(height, "height");
  const expected = width * height * 3;
  if (points.length !== expected) {
    throw new RangeError(
      `points must contain exactly height*width*3 values (${expected}); received ${points.length}`,
    );
  }
}

function validateMask(mask: MaskArray | undefined, width: number, height: number): void {
  if (mask === undefined) {
    return;
  }
  const expected = width * height;
  if (mask.length !== expected) {
    throw new RangeError(
      `mask must contain exactly height*width values (${expected}); received ${mask.length}`,
    );
  }
}

/**
 * Build MoGe's normalized view-plane coordinates in row-major HW2 order.
 *
 * The first pixel is `(-spanX * (W - 1) / W, -spanY * (H - 1) / H)` and the
 * last is the corresponding positive corner, exactly as the upstream
 * `torch.linspace` implementation.  Coordinates are pixel-centre coordinates
 * expressed relative to half the image diagonal; they are not [0, 1] UVs.
 */
export function normalizedViewPlaneUv(
  width: number,
  height: number,
  aspectRatio = width / height,
): Float32Array {
  assertPositiveInteger(width, "width");
  assertPositiveInteger(height, "height");
  assertFinitePositive(aspectRatio, "aspectRatio");

  const denominator = Math.sqrt(1 + aspectRatio * aspectRatio);
  const spanX = aspectRatio / denominator;
  const spanY = 1 / denominator;
  const output = new Float32Array(width * height * 2);

  // Writing through Float32Array intentionally mirrors the model's fp32
  // post-processing even though the scalar arithmetic here uses JS doubles.
  for (let y = 0; y < height; y += 1) {
    const v = spanY * (2 * y - (height - 1)) / height;
    for (let x = 0; x < width; x += 1) {
      const u = spanX * (2 * x - (width - 1)) / width;
      const uvIndex = (y * width + x) * 2;
      output[uvIndex] = u;
      output[uvIndex + 1] = v;
    }
  }
  return output;
}

/** PyTorch `interpolate(..., mode="nearest")` source index for one axis. */
export function nearestInterpolateSourceIndex(
  outputIndex: number,
  inputSize: number,
  outputSize: number,
): number {
  assertPositiveInteger(inputSize, "inputSize");
  assertPositiveInteger(outputSize, "outputSize");
  if (!Number.isInteger(outputIndex) || outputIndex < 0 || outputIndex >= outputSize) {
    throw new RangeError(
      `outputIndex must be an integer in [0, ${outputSize}); received ${String(outputIndex)}`,
    );
  }
  // torch's legacy `nearest` mode uses floor(i * input/output), unlike
  // `nearest-exact`, which is intentionally not used by MoGe.
  return Math.min(inputSize - 1, Math.floor(outputIndex * inputSize / outputSize));
}

function parseRecoverArguments(
  widthOrDimensions: number | RecoverDimensions,
  height?: number,
  mask?: MaskArray,
  options?: RecoverFocalShiftOptions,
): RecoverDimensions {
  if (typeof widthOrDimensions === "number") {
    if (height === undefined) {
      throw new TypeError("recoverFocalShift requires both width and height");
    }
    const dimensions: RecoverDimensions = { width: widthOrDimensions, height };
    if (mask !== undefined) {
      dimensions.mask = mask;
    }
    if (options !== undefined) {
      dimensions.options = options;
    }
    return dimensions;
  }
  if (height !== undefined || mask !== undefined || options !== undefined) {
    throw new TypeError(
      "When dimensions are supplied as an object, height, mask, and options must be omitted",
    );
  }
  return widthOrDimensions;
}

function validateDownsampleSize(options: RecoverFocalShiftOptions): [number, number] {
  const targetWidth = options.downsampleWidth ?? DEFAULT_DOWNSAMPLE;
  const targetHeight = options.downsampleHeight ?? DEFAULT_DOWNSAMPLE;
  assertPositiveInteger(targetWidth, "downsampleWidth");
  assertPositiveInteger(targetHeight, "downsampleHeight");
  return [targetWidth, targetHeight];
}

function collectNearestSamples(
  points: NumericArray,
  dimensions: RecoverDimensions,
): Sample[] {
  const { width, height, mask } = dimensions;
  const options = dimensions.options ?? {};
  const [targetWidth, targetHeight] = validateDownsampleSize(options);
  const uv = normalizedViewPlaneUv(width, height);
  const samples: Sample[] = [];

  for (let targetY = 0; targetY < targetHeight; targetY += 1) {
    const sourceY = nearestInterpolateSourceIndex(targetY, height, targetHeight);
    for (let targetX = 0; targetX < targetWidth; targetX += 1) {
      const sourceX = nearestInterpolateSourceIndex(targetX, width, targetWidth);
      const sourceIndex = sourceY * width + sourceX;
      if (mask !== undefined && !isMaskValid(mask, sourceIndex)) {
        continue;
      }

      const pointIndex = sourceIndex * 3;
      const x = points[pointIndex];
      const y = points[pointIndex + 1];
      const z = points[pointIndex + 2];
      const uvIndex = sourceIndex * 2;
      const u = uv[uvIndex];
      const v = uv[uvIndex + 1];
      if (
        Number.isFinite(x) &&
        Number.isFinite(y) &&
        Number.isFinite(z) &&
        Number.isFinite(u) &&
        Number.isFinite(v)
      ) {
        samples.push({ u: u!, v: v!, x: x!, y: y!, z: z! });
      }
    }
  }
  return samples;
}

function normalizeSamples(samples: Sample[]): { samples: Sample[]; scale: number } {
  let scale = 1;
  for (const sample of samples) {
    scale = Math.max(scale, Math.abs(sample.x), Math.abs(sample.y), Math.abs(sample.z));
  }
  if (!Number.isFinite(scale) || scale <= 0) {
    throw new Error("recoverFocalShift cannot normalize non-finite point geometry");
  }
  return {
    scale,
    samples: samples.map((sample) => ({
      u: sample.u,
      v: sample.v,
      x: sample.x / scale,
      y: sample.y / scale,
      z: sample.z / scale,
    })),
  };
}

function evaluateShift(samples: Sample[], shift: number, fixedFocal?: number): ShiftEvaluation {
  let numerator = 0;
  let denominator = 0;
  let valid = 0;

  for (const sample of samples) {
    const depth = sample.z + shift;
    // SciPy's residual becomes non-finite at a pole; it does not silently
    // drop that observation. Negative depths remain valid during recovery
    // and are masked only afterward by v2.py::infer.
    const epsilon = 1e-10 * Math.max(1, Math.abs(sample.z), Math.abs(shift));
    if (!Number.isFinite(depth) || Math.abs(depth) <= epsilon) {
      return { cost: Number.POSITIVE_INFINITY, focal: Number.NaN };
    }
    const qx = sample.x / depth;
    const qy = sample.y / depth;
    numerator += qx * sample.u + qy * sample.v;
    denominator += qx * qx + qy * qy;
    valid += 1;
  }

  if (valid < 2 || !Number.isFinite(denominator) || denominator <= Number.EPSILON) {
    return { cost: Number.POSITIVE_INFINITY, focal: Number.NaN };
  }

  const focal = fixedFocal ?? numerator / denominator;
  if (!Number.isFinite(focal)) {
    return { cost: Number.POSITIVE_INFINITY, focal: Number.NaN };
  }

  let cost = 0;
  for (const sample of samples) {
    const depth = sample.z + shift;
    const epsilon = 1e-10 * Math.max(1, Math.abs(sample.z), Math.abs(shift));
    if (!Number.isFinite(depth) || Math.abs(depth) <= epsilon) {
      return { cost: Number.POSITIVE_INFINITY, focal: Number.NaN };
    }
    const residualX = focal * sample.x / depth - sample.u;
    const residualY = focal * sample.y / depth - sample.v;
    cost += residualX * residualX + residualY * residualY;
  }
  return Number.isFinite(cost)
    ? { cost, focal }
    : { cost: Number.POSITIVE_INFINITY, focal: Number.NaN };
}

function solveAffineSeed(samples: Sample[]): { focal: number; shift: number } | undefined {
  // Linearized camera equation:
  //     focal * xy - uv * shift = uv * z.
  // It is only used as a high-quality deterministic starting point; the
  // selected answer is still scored with the upstream nonlinear residual.
  let a00 = 0;
  let a01 = 0;
  let a11 = 0;
  let b0 = 0;
  let b1 = 0;
  for (const sample of samples) {
    const ax = sample.x;
    const ay = sample.y;
    const bx = -sample.u;
    const by = -sample.v;
    const rhsX = sample.u * sample.z;
    const rhsY = sample.v * sample.z;
    a00 += ax * ax + ay * ay;
    a01 += ax * bx + ay * by;
    a11 += bx * bx + by * by;
    b0 += ax * rhsX + ay * rhsY;
    b1 += bx * rhsX + by * rhsY;
  }
  const determinant = a00 * a11 - a01 * a01;
  const scale = Math.max(1, Math.abs(a00), Math.abs(a01), Math.abs(a11));
  if (!Number.isFinite(determinant) || Math.abs(determinant) <= 1e-12 * scale * scale) {
    return undefined;
  }
  const focal = (b0 * a11 - a01 * b1) / determinant;
  const shift = (a00 * b1 - a01 * b0) / determinant;
  return Number.isFinite(focal) && Number.isFinite(shift) ? { focal, shift } : undefined;
}

function solveFixedFocalSeed(samples: Sample[], focal: number): number | undefined {
  // With focal fixed, the same camera equation gives a direct least-squares
  // seed for shift:
  //   shift * (u² + v²) = focal * (u*x + v*y) - (u² + v²) * z.
  let weight = 0;
  let numerator = 0;
  for (const sample of samples) {
    const uv2 = sample.u * sample.u + sample.v * sample.v;
    weight += uv2;
    numerator += focal * (sample.u * sample.x + sample.v * sample.y) - uv2 * sample.z;
  }
  if (!Number.isFinite(weight) || weight <= Number.EPSILON) {
    return undefined;
  }
  const shift = numerator / weight;
  return Number.isFinite(shift) ? shift : undefined;
}

function goldenMinimum(
  samples: Sample[],
  left: number,
  right: number,
  fixedFocal?: number,
): { shift: number; evaluation: ShiftEvaluation } {
  if (!(right > left)) {
    const evaluation = evaluateShift(samples, left, fixedFocal);
    return { shift: left, evaluation };
  }
  let a = left;
  let b = right;
  let c = b - (b - a) / GOLDEN_RATIO;
  let d = a + (b - a) / GOLDEN_RATIO;
  let fc = evaluateShift(samples, c, fixedFocal).cost;
  let fd = evaluateShift(samples, d, fixedFocal).cost;

  // 80 iterations is inexpensive for a 64x64 sample and reaches the limits
  // of double precision for ordinary model output scales.
  for (let iteration = 0; iteration < 80; iteration += 1) {
    if (fc <= fd) {
      b = d;
      d = c;
      fd = fc;
      c = b - (b - a) / GOLDEN_RATIO;
      fc = evaluateShift(samples, c, fixedFocal).cost;
    } else {
      a = c;
      c = d;
      fc = fd;
      d = a + (b - a) / GOLDEN_RATIO;
      fd = evaluateShift(samples, d, fixedFocal).cost;
    }
  }

  const candidateA = evaluateShift(samples, a, fixedFocal);
  const candidateB = evaluateShift(samples, b, fixedFocal);
  const candidateC = evaluateShift(samples, (a + b) / 2, fixedFocal);
  let shift = a;
  let evaluation = candidateA;
  if (candidateB.cost < evaluation.cost) {
    shift = b;
    evaluation = candidateB;
  }
  if (candidateC.cost < evaluation.cost) {
    shift = (a + b) / 2;
    evaluation = candidateC;
  }
  return { shift, evaluation };
}

function optimizeShift(
  samples: Sample[],
  fixedFocal?: number,
): FocalShiftRecovery {
  const normalized = normalizeSamples(samples);
  const normalizedSamples = normalized.samples;
  let maxAbsZ = 0;
  for (const sample of normalizedSamples) {
    maxAbsZ = Math.max(maxAbsZ, Math.abs(sample.z));
  }

  // Upstream scipy.optimize.least_squares is unconstrained. In particular,
  // some points may remain behind the recovered camera and v2.py masks them
  // only after solving. Search a wide symmetric interval around its x0=0.
  const searchRadius = 128 * Math.max(1, maxAbsZ);
  let lower = -searchRadius;
  let upper = searchRadius;
  const affineSeed = solveAffineSeed(normalizedSamples);
  const seedShift = fixedFocal === undefined
    ? affineSeed?.shift
    : solveFixedFocalSeed(normalizedSamples, fixedFocal);
  if (seedShift !== undefined && seedShift < lower) {
    lower = seedShift - Math.max(1, Math.abs(seedShift) * 0.25);
  }
  if (seedShift !== undefined && seedShift > upper) {
    upper = seedShift + Math.max(1, Math.abs(seedShift) * 0.25);
  }

  let bestShift = lower;
  let bestEvaluation = evaluateShift(normalizedSamples, bestShift, fixedFocal);
  const consider = (shift: number): void => {
    if (!Number.isFinite(shift)) {
      return;
    }
    const evaluation = evaluateShift(normalizedSamples, shift, fixedFocal);
    if (evaluation.cost < bestEvaluation.cost) {
      bestShift = shift;
      bestEvaluation = evaluation;
    }
  };

  const gridSize = 257;
  let bestGridIndex = 0;
  for (let index = 0; index < gridSize; index += 1) {
    const shift = lower + (upper - lower) * index / (gridSize - 1);
    const evaluation = evaluateShift(normalizedSamples, shift, fixedFocal);
    if (evaluation.cost < bestEvaluation.cost) {
      bestShift = shift;
      bestEvaluation = evaluation;
      bestGridIndex = index;
    }
  }

  if (seedShift !== undefined) {
    consider(seedShift);
    if (seedShift > lower) {
      const seedWidth = Math.max((upper - lower) / (gridSize - 1) * 2, Math.abs(seedShift) * 0.2, 1e-7);
      const refined = goldenMinimum(
        normalizedSamples,
        Math.max(lower, seedShift - seedWidth),
        Math.min(upper, seedShift + seedWidth),
        fixedFocal,
      );
      if (refined.evaluation.cost < bestEvaluation.cost) {
        bestShift = refined.shift;
        bestEvaluation = refined.evaluation;
      }
    }
  }

  const gridStep = (upper - lower) / (gridSize - 1);
  const bracketLeft = Math.max(lower, bestShift - 2 * gridStep);
  const bracketRight = Math.min(upper, bestShift + 2 * gridStep);
  const refined = goldenMinimum(normalizedSamples, bracketLeft, bracketRight, fixedFocal);
  if (refined.evaluation.cost < bestEvaluation.cost) {
    bestShift = refined.shift;
    bestEvaluation = refined.evaluation;
  }

  // If the global grid did not improve over its initial endpoint, retain the
  // first valid grid cell's index for diagnostics and deterministic behavior.
  // This assignment also keeps the intentional fallback visible to readers.
  if (!Number.isFinite(bestEvaluation.cost)) {
    bestGridIndex = Math.max(0, Math.min(gridSize - 1, bestGridIndex));
    bestShift = lower + gridStep * bestGridIndex;
    bestEvaluation = evaluateShift(normalizedSamples, bestShift, fixedFocal);
  }

  const focal = fixedFocal ?? bestEvaluation.focal;
  const shift = bestShift * normalized.scale;
  if (!Number.isFinite(focal) || !Number.isFinite(shift)) {
    throw new Error("recoverFocalShift could not find finite focal/shift geometry");
  }
  return { focal, shift };
}

/**
 * Recover focal and Z shift from an affine HWC point map.
 *
 * `width` and `height` are the original map dimensions (the same order as
 * `RawMoGeOutputs`).  `mask`, when present, is sampled with the exact floor
 * mapping used by PyTorch `interpolate(mode="nearest")` before optimization.
 * With exactly one sampled valid pixel, upstream returns `(1, 0)`; that
 * compatibility fallback is retained here.  An all-invalid sample set is
 * rejected explicitly so callers receive an actionable geometry error.
 */
export function recoverFocalShift(
  points: NumericArray,
  width: number,
  height: number,
  mask?: MaskArray,
  options?: RecoverFocalShiftOptions,
): FocalShiftRecovery;
export function recoverFocalShift(
  points: NumericArray,
  dimensions: RecoverDimensions,
): FocalShiftRecovery;
export function recoverFocalShift(
  points: NumericArray,
  widthOrDimensions: number | RecoverDimensions,
  height?: number,
  mask?: MaskArray,
  options?: RecoverFocalShiftOptions,
): FocalShiftRecovery {
  const dimensions = parseRecoverArguments(widthOrDimensions, height, mask, options);
  validatePointArray(points, dimensions.width, dimensions.height);
  validateMask(dimensions.mask, dimensions.width, dimensions.height);
  const effectiveOptions = dimensions.options ?? {};
  if (effectiveOptions.focal !== undefined) {
    assertFinitePositive(effectiveOptions.focal, "focal");
  }
  const samples = collectNearestSamples(points, dimensions);
  if (samples.length === 0) {
    throw new Error("recoverFocalShift received no finite samples under the supplied mask");
  }
  if (samples.length === 1) {
    return { focal: effectiveOptions.focal ?? 1, shift: 0 };
  }
  return optimizeShift(samples, effectiveOptions.focal);
}

/** Solve a fixed-focal shift from already sampled UV and XYZ arrays. */
export function solveOptimalShift(
  uv: NumericArray,
  xyz: NumericArray,
  focal: number,
): number {
  assertFinitePositive(focal, "focal");
  if (uv.length % 2 !== 0 || xyz.length % 3 !== 0 || uv.length / 2 !== xyz.length / 3) {
    throw new RangeError("uv and xyz must contain the same number of 2D/3D samples");
  }
  const samples: Sample[] = [];
  for (let index = 0; index < uv.length / 2; index += 1) {
    const uvIndex = index * 2;
    const pointIndex = index * 3;
    const u = uv[uvIndex]!;
    const v = uv[uvIndex + 1]!;
    const x = xyz[pointIndex]!;
    const y = xyz[pointIndex + 1]!;
    const z = xyz[pointIndex + 2]!;
    if (Number.isFinite(u) && Number.isFinite(v) && Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
      samples.push({ u, v, x, y, z });
    }
  }
  if (samples.length === 0) {
    throw new Error("solveOptimalShift received no finite samples");
  }
  if (samples.length === 1) {
    return 0;
  }
  return optimizeShift(samples, focal).shift;
}

/** Return finite values, optionally restricted by a positive mask. */
export function finiteValues(values: NumericArray, mask?: MaskArray): number[] {
  if (mask !== undefined && mask.length !== values.length) {
    throw new RangeError(`mask length (${mask.length}) must equal values length (${values.length})`);
  }
  const output: number[] = [];
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index]!;
    if ((mask === undefined || isMaskValid(mask, index)) && Number.isFinite(value)) {
      output.push(value);
    }
  }
  return output;
}

/**
 * Percentile of finite values using NumPy's default linear interpolation.
 * `percentile` is expressed in the conventional 0..100 range.
 */
export function finitePercentile(
  values: NumericArray,
  percentile: number,
  mask?: MaskArray,
): number {
  if (!Number.isFinite(percentile) || percentile < 0 || percentile > 100) {
    throw new RangeError(`percentile must be finite and in [0, 100]; received ${String(percentile)}`);
  }
  const finite = finiteValues(values, mask).sort((a, b) => a - b);
  if (finite.length === 0) {
    throw new Error("finitePercentile requires at least one finite value");
  }
  if (finite.length === 1) {
    return finite[0]!;
  }
  const position = percentile / 100 * (finite.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const lowerValue = finite[lower]!;
  const upperValue = finite[upper]!;
  return lowerValue + (upperValue - lowerValue) * (position - lower);
}

/** Quantile counterpart to `finitePercentile`, with `quantile` in 0..1. */
export function finiteQuantile(
  values: NumericArray,
  quantile: number,
  mask?: MaskArray,
): number {
  if (!Number.isFinite(quantile) || quantile < 0 || quantile > 1) {
    throw new RangeError(`quantile must be finite and in [0, 1]; received ${String(quantile)}`);
  }
  return finitePercentile(values, quantile * 100, mask);
}

/** Summarize finite depth values; non-finite placeholders are ignored. */
export function finiteDepthStats(depth: NumericArray, mask?: MaskArray): FiniteDepthStats {
  const finite = finiteValues(depth, mask).sort((a, b) => a - b);
  if (finite.length === 0) {
    throw new Error("finiteDepthStats requires at least one finite depth");
  }
  const sum = finite.reduce((total, value) => total + value, 0);
  const percentileFromSorted = (percentile: number): number => {
    const position = percentile / 100 * (finite.length - 1);
    const lower = Math.floor(position);
    const upper = Math.ceil(position);
    const lowerValue = finite[lower]!;
    const upperValue = finite[upper]!;
    return lowerValue + (upperValue - lowerValue) * (position - lower);
  };
  return {
    count: finite.length,
    min: finite[0]!,
    max: finite[finite.length - 1]!,
    mean: sum / finite.length,
    p01: percentileFromSorted(1),
    p05: percentileFromSorted(5),
    p50: percentileFromSorted(50),
    p95: percentileFromSorted(95),
    p99: percentileFromSorted(99),
  };
}
