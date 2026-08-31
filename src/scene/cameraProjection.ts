import { Matrix4 } from 'three';
import type { MoGeResult } from '../moge/types';

export interface DepthRange {
  near: number;
  far: number;
  minDepth: number;
  maxDepth: number;
  sampleCount: number;
}

export interface DepthRangeOptions {
  /** Lower and upper quantiles used to ignore isolated depth outliers. */
  nearQuantile?: number;
  farQuantile?: number;
  /** Multipliers leaving a margin around the robust range. */
  nearScale?: number;
  farScale?: number;
}

export interface ImageFitViewport {
  /** Pixel-space origin of the letterboxed image rectangle. */
  x: number;
  y: number;
  width: number;
  height: number;
  sourceWidth: number;
  sourceHeight: number;
  aspect: number;
}

export interface CameraProjectionOptions extends DepthRangeOptions {
  width?: number;
  height?: number;
  sourceWidth?: number;
  sourceHeight?: number;
  viewportWidth?: number;
  viewportHeight?: number;
  viewport?: { width: number; height: number };
  near?: number;
  far?: number;
  depth?: ArrayLike<number>;
}

interface ProjectionClipPlanes {
  near: number;
  far: number;
}

export interface CameraProjection {
  /** Projection from converted Three camera coordinates to clip space. */
  projectionMatrix: Matrix4;
  /** Inverse of projectionMatrix, suitable for unprojection/ray setup. */
  projectionMatrixInverse: Matrix4;
  /** Concise aliases for renderer adapters. */
  matrix: Matrix4;
  inverse: Matrix4;
  near: number;
  far: number;
  sourceWidth: number;
  sourceHeight: number;
  /** Aspect used in the projection. It is always sourceWidth/sourceHeight. */
  aspect: number;
  viewport: ImageFitViewport;
  /** Vertical and horizontal fields of view in radians. */
  fovY: number;
  fovX: number;
  fovYDegrees: number;
  fovXDegrees: number;
  /** `fov` follows Three's conventional vertical-FOV naming. */
  fov: number;
  depthRange: DepthRange;
}

interface ProjectionDimensions {
  sourceWidth: number;
  sourceHeight: number;
  viewportWidth: number;
  viewportHeight: number;
}

function assertPositiveDimension(name: string, value: number | undefined): number {
  if (value === undefined || !Number.isFinite(value) || value <= 0) {
    throw new RangeError(`${name} must be a positive finite number; got ${value}`);
  }
  return value;
}

function quantile(sorted: readonly number[], probability: number): number {
  if (sorted.length === 0) return Number.NaN;
  const clamped = Math.min(1, Math.max(0, probability));
  const position = (sorted.length - 1) * clamped;
  const lowerIndex = Math.floor(position);
  const upperIndex = Math.ceil(position);
  const lower = sorted[lowerIndex];
  const upper = sorted[upperIndex];
  if (lower === undefined || upper === undefined) return sorted[0] ?? Number.NaN;
  return lower + (upper - lower) * (position - lowerIndex);
}

/**
 * Derive conservative clipping planes from positive finite depths.  Quantile
 * bounds make a single bad prediction unable to collapse the useful depth
 * range; margins keep the nearest/farthest valid points just inside the clip
 * volume.  The returned planes are deterministic for a given depth array.
 */
export function deriveDepthRange(
  depth: ArrayLike<number>,
  options: DepthRangeOptions = {},
): DepthRange {
  const values: number[] = [];
  for (let index = 0; index < depth.length; index += 1) {
    const value = depth[index];
    if (value !== undefined && Number.isFinite(value) && value > 0) values.push(value);
  }
  values.sort((first, second) => first - second);
  if (values.length === 0) {
    return { near: 0.01, far: 100, minDepth: 0, maxDepth: 0, sampleCount: 0 };
  }

  const requestedNearQuantile = options.nearQuantile ?? 0.01;
  const requestedFarQuantile = options.farQuantile ?? 0.99;
  const nearQuantile = Number.isFinite(requestedNearQuantile) ? requestedNearQuantile : 0.01;
  const farQuantile = Number.isFinite(requestedFarQuantile) ? requestedFarQuantile : 0.99;
  const requestedNearScale = options.nearScale ?? 0.8;
  const requestedFarScale = options.farScale ?? 1.2;
  const nearScale = Number.isFinite(requestedNearScale) ? requestedNearScale : 0.8;
  const farScale = Number.isFinite(requestedFarScale) ? requestedFarScale : 1.2;
  const robustNear = quantile(values, Math.min(nearQuantile, farQuantile));
  const robustFar = quantile(values, Math.max(nearQuantile, farQuantile));
  const near = Math.max(Number.MIN_VALUE, robustNear * Math.max(Number.MIN_VALUE, nearScale));
  const farCandidate = robustFar * Math.max(Number.MIN_VALUE, farScale);
  const far = Math.max(farCandidate, near * 2, Number.MIN_VALUE * 2);
  return {
    near,
    far,
    minDepth: values[0] ?? robustNear,
    maxDepth: values[values.length - 1] ?? robustFar,
    sampleCount: values.length,
  };
}

export const robustDepthRange = deriveDepthRange;

/** Compute a centered pixel rectangle preserving the source image aspect. */
export function imageFitViewport(
  viewportWidth: number,
  viewportHeight: number,
  sourceWidth: number,
  sourceHeight: number,
): ImageFitViewport {
  const outputWidth = assertPositiveDimension('viewportWidth', viewportWidth);
  const outputHeight = assertPositiveDimension('viewportHeight', viewportHeight);
  const imageWidth = assertPositiveDimension('sourceWidth', sourceWidth);
  const imageHeight = assertPositiveDimension('sourceHeight', sourceHeight);
  const sourceAspect = imageWidth / imageHeight;
  const viewportAspect = outputWidth / outputHeight;
  let width: number;
  let height: number;
  if (viewportAspect > sourceAspect) {
    height = outputHeight;
    width = height * sourceAspect;
  } else {
    width = outputWidth;
    height = width / sourceAspect;
  }
  return {
    x: (outputWidth - width) / 2,
    y: (outputHeight - height) / 2,
    width,
    height,
    sourceWidth: imageWidth,
    sourceHeight: imageHeight,
    aspect: sourceAspect,
  };
}

export const getImageFitViewport = imageFitViewport;
export const computeImageFitViewport = imageFitViewport;
export const fitViewport = imageFitViewport;
export const imageFit = imageFitViewport;

function readNormalizedIntrinsics(intrinsics: ArrayLike<number>): {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
} {
  if (intrinsics.length < 9) throw new RangeError('Intrinsics must contain a row-major 3x3 matrix');
  const fx = intrinsics[0];
  const cx = intrinsics[2];
  const fy = intrinsics[4];
  const cy = intrinsics[5];
  if (fx === undefined || fy === undefined || cx === undefined || cy === undefined
    || !Number.isFinite(fx) || !Number.isFinite(fy)
    || !Number.isFinite(cx) || !Number.isFinite(cy) || fx <= 0 || fy <= 0) {
    throw new RangeError('Intrinsics must contain finite positive focal lengths');
  }
  return { fx, fy, cx, cy };
}

function normalizeProjectionDimensions(
  sourceWidth: number,
  sourceHeight: number,
  options: CameraProjectionOptions,
): ProjectionDimensions {
  const normalizedSourceWidth = assertPositiveDimension('sourceWidth', sourceWidth);
  const normalizedSourceHeight = assertPositiveDimension('sourceHeight', sourceHeight);
  return {
    sourceWidth: normalizedSourceWidth,
    sourceHeight: normalizedSourceHeight,
    viewportWidth: assertPositiveDimension(
      'viewportWidth',
      options.viewportWidth ?? options.viewport?.width ?? normalizedSourceWidth,
    ),
    viewportHeight: assertPositiveDimension(
      'viewportHeight',
      options.viewportHeight ?? options.viewport?.height ?? normalizedSourceHeight,
    ),
  };
}

/**
 * Construct an off-axis Three perspective matrix directly from normalized K.
 *
 * With a converted point `(X, -Y, -Z)`, this matrix yields
 * `ndc.x = 2*u - 1`, `ndc.y = 1 - 2*v`, and maps positive OpenCV depth Z to
 * the usual Three depth interval: near -> -1, far -> +1.  The viewport is
 * intentionally absent from this matrix; callers should render to the
 * returned imageFitViewport rectangle to preserve calibration while
 * letterboxing an arbitrary canvas.
 */
export function createProjectionMatrix(
  intrinsics: ArrayLike<number>,
  sourceWidth: number,
  sourceHeight: number,
  near: number,
  far: number,
): Matrix4;
export function createProjectionMatrix(
  intrinsics: ArrayLike<number>,
  sourceWidth: number,
  sourceHeight: number,
  clipPlanes: ProjectionClipPlanes,
): Matrix4;
export function createProjectionMatrix(
  intrinsics: ArrayLike<number>,
  sourceWidth: number,
  sourceHeight: number,
  nearOrClipPlanes: number | ProjectionClipPlanes,
  far?: number,
): Matrix4 {
  const { fx, fy, cx, cy } = readNormalizedIntrinsics(intrinsics);
  assertPositiveDimension('sourceWidth', sourceWidth);
  assertPositiveDimension('sourceHeight', sourceHeight);
  const near = typeof nearOrClipPlanes === 'number' ? nearOrClipPlanes : nearOrClipPlanes.near;
  const resolvedFar = typeof nearOrClipPlanes === 'number' ? far : nearOrClipPlanes.far;
  if (resolvedFar === undefined || !Number.isFinite(near) || !Number.isFinite(resolvedFar)
    || near <= 0 || resolvedFar <= near) {
    throw new RangeError(`Projection clipping planes must satisfy 0 < near < far; got ${near}, ${resolvedFar}`);
  }

  // K is normalized by source dimensions, so fx/fy/cx/cy are already in
  // [image units / image dimensions].  The converted Y and Z signs account
  // for OpenCV (+Y down, +Z forward) -> Three (+Y up, -Z forward).
  const m00 = 2 * fx;
  const m02 = 1 - 2 * cx;
  const m11 = 2 * fy;
  const m12 = 2 * cy - 1;
  const m22 = -(resolvedFar + near) / (resolvedFar - near);
  const m23 = -(2 * resolvedFar * near) / (resolvedFar - near);

  return new Matrix4().set(
    m00, 0, m02, 0,
    0, m11, m12, 0,
    0, 0, m22, m23,
    0, 0, -1, 0,
  );
}

export const projectionMatrixFromIntrinsics = createProjectionMatrix;
export const buildProjectionMatrix = createProjectionMatrix;
export const calibratedProjectionMatrix = createProjectionMatrix;
export const createCameraProjectionMatrix = createProjectionMatrix;

function extractProjectionInputs(
  source: MoGeResult | ArrayLike<number>,
  sourceWidthOrOptions: number | CameraProjectionOptions | undefined,
  sourceHeight: number | undefined,
  options: CameraProjectionOptions | undefined,
): {
  intrinsics: ArrayLike<number>;
  dimensions: ProjectionDimensions;
  options: CameraProjectionOptions;
  depth?: ArrayLike<number>;
} {
  if ('intrinsics' in source && 'width' in source && 'height' in source) {
    const result = source as MoGeResult;
    const inputOptions = options ?? (typeof sourceWidthOrOptions === 'object' ? sourceWidthOrOptions : {});
    const validDepth = inputOptions.depth ?? Float32Array.from(
      result.depth,
      (depth, index) => result.mask[index] === 1 ? depth : Number.NaN,
    );
    return {
      intrinsics: result.intrinsics,
      dimensions: normalizeProjectionDimensions(result.width, result.height, inputOptions),
      options: inputOptions,
      depth: validDepth,
    };
  }

  const inputOptions = typeof sourceWidthOrOptions === 'object' ? sourceWidthOrOptions : (options ?? {});
  const width = typeof sourceWidthOrOptions === 'number'
    ? sourceWidthOrOptions
    : inputOptions.sourceWidth ?? inputOptions.width;
  const height = sourceHeight ?? inputOptions.sourceHeight ?? inputOptions.height;
  if (width === undefined || height === undefined) {
    throw new RangeError('sourceWidth and sourceHeight are required with a bare intrinsics matrix');
  }
  return {
    intrinsics: source,
    dimensions: normalizeProjectionDimensions(width, height, inputOptions),
    options: inputOptions,
    ...(inputOptions.depth === undefined ? {} : { depth: inputOptions.depth }),
  };
}

export function createCameraProjection(
  result: MoGeResult,
  options?: CameraProjectionOptions,
): CameraProjection;
export function createCameraProjection(
  intrinsics: ArrayLike<number>,
  sourceWidth: number,
  sourceHeight: number,
  options?: CameraProjectionOptions,
): CameraProjection;
export function createCameraProjection(
  intrinsics: ArrayLike<number>,
  options: CameraProjectionOptions,
): CameraProjection;
export function createCameraProjection(
  source: MoGeResult | ArrayLike<number>,
  sourceWidthOrOptions?: number | CameraProjectionOptions,
  sourceHeight?: number,
  options?: CameraProjectionOptions,
): CameraProjection {
  const extracted = extractProjectionInputs(source, sourceWidthOrOptions, sourceHeight, options);
  const inputOptions = extracted.options;
  const { dimensions } = extracted;
  const depthRange = extracted.depth === undefined
    ? deriveDepthRange([])
    : deriveDepthRange(extracted.depth, inputOptions);
  const near = inputOptions.near ?? depthRange.near;
  const far = inputOptions.far ?? depthRange.far;
  const projectionMatrix = createProjectionMatrix(
    extracted.intrinsics,
    dimensions.sourceWidth,
    dimensions.sourceHeight,
    near,
    far,
  );
  const projectionMatrixInverse = projectionMatrix.clone().invert();
  const { fx, fy } = readNormalizedIntrinsics(extracted.intrinsics);
  const fovX = 2 * Math.atan(0.5 / fx);
  const fovY = 2 * Math.atan(0.5 / fy);
  const viewport = imageFitViewport(
    dimensions.viewportWidth,
    dimensions.viewportHeight,
    dimensions.sourceWidth,
    dimensions.sourceHeight,
  );

  return {
    projectionMatrix,
    projectionMatrixInverse,
    matrix: projectionMatrix,
    inverse: projectionMatrixInverse,
    near,
    far,
    sourceWidth: dimensions.sourceWidth,
    sourceHeight: dimensions.sourceHeight,
    aspect: dimensions.sourceWidth / dimensions.sourceHeight,
    viewport,
    fovY,
    fovX,
    fovYDegrees: fovY * 180 / Math.PI,
    fovXDegrees: fovX * 180 / Math.PI,
    fov: fovY,
    depthRange,
  };
}

export const buildCameraProjection = createCameraProjection;
export const createCalibratedProjection = createCameraProjection;
export const createProjection = createCameraProjection;
