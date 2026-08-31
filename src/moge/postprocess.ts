import type { MoGeResult, RawMoGeOutputs } from "./types";
import { recoverFocalShift } from "./math";

/**
 * Recreate a camera-space point map from normalized depth and MoGe intrinsics.
 * `depth` is row-major HW, and the returned map is row-major HWC.  The
 * coordinates are OpenCV (+X right, +Y down, +Z forward), matching
 * `utils3d.pt.depth_map_to_point_map` used in `moge/model/v2.py`.
 */
export function depthToPoints(
  depth: ArrayLike<number>,
  width: number,
  height: number,
  intrinsics: ArrayLike<number>,
): Float32Array {
  if (!Number.isInteger(width) || width <= 0 || !Number.isInteger(height) || height <= 0) {
    throw new RangeError(`depthToPoints requires positive integer dimensions; received ${width}x${height}`);
  }
  const pixelCount = width * height;
  if (depth.length !== pixelCount) {
    throw new RangeError(
      `depth must contain exactly height*width values (${pixelCount}); received ${depth.length}`,
    );
  }
  if (intrinsics.length !== 9) {
    throw new RangeError(`intrinsics must contain exactly 9 values; received ${intrinsics.length}`);
  }
  const fx = intrinsics[0]!;
  const fy = intrinsics[4]!;
  const cx = intrinsics[2]!;
  const cy = intrinsics[5]!;
  if (!Number.isFinite(fx) || !Number.isFinite(fy) || fx <= 0 || fy <= 0) {
    throw new Error(`depthToPoints received non-finite or non-positive focal intrinsics (${fx}, ${fy})`);
  }
  if (!Number.isFinite(cx) || !Number.isFinite(cy)) {
    throw new Error("depthToPoints received non-finite principal-point intrinsics");
  }

  const points = new Float32Array(pixelCount * 3);
  for (let row = 0; row < height; row += 1) {
    // utils3d.pt.uv_map uses pixel centres: (x + 0.5) / width, (y + 0.5) / height.
    const v = (row + 0.5) / height;
    for (let column = 0; column < width; column += 1) {
      const pixelIndex = row * width + column;
      const pointIndex = pixelIndex * 3;
      const d = depth[pixelIndex]!;
      const u = (column + 0.5) / width;
      points[pointIndex] = (u - cx) / fx * d;
      points[pointIndex + 1] = (v - cy) / fy * d;
      points[pointIndex + 2] = d;
    }
  }
  return points;
}

function assertDimensions(raw: RawMoGeOutputs): void {
  if (!raw || typeof raw !== "object") {
    throw new TypeError("postprocessMoGe requires a RawMoGeOutputs object");
  }
  if (!Number.isInteger(raw.width) || raw.width <= 0 || !Number.isInteger(raw.height) || raw.height <= 0) {
    throw new RangeError(
      `postprocessMoGe requires positive integer width/height; received ${String(raw.width)}x${String(raw.height)}`,
    );
  }
}

function assertArrayLength(
  name: string,
  values: ArrayLike<unknown> | undefined,
  expected: number,
): asserts values is ArrayLike<unknown> {
  if (values === undefined || values === null || typeof values.length !== "number") {
    throw new TypeError(`${name} must be an array-like value containing exactly ${expected} entries`);
  }
  if (values.length !== expected) {
    throw new RangeError(`${name} must contain exactly ${expected} values; received ${values.length}`);
  }
}

/**
 * Port of the post-processing section of `moge/model/v2.py::MoGeModel.infer`.
 *
 * Raw arrays are row-major HWC/HW and already contain forward's remapping,
 * sigmoid mask, normalized normal, and exponentiated metric scale.  The
 * implementation intentionally keeps all axes in the OpenCV convention; no
 * Three.js conversion is performed here.
 */
export function postprocessMoGe(raw: RawMoGeOutputs): MoGeResult {
  assertDimensions(raw);
  const { width, height } = raw;
  const pixelCount = width * height;
  const pointValueCount = pixelCount * 3;
  assertArrayLength("points", raw.points, pointValueCount);
  if (raw.normal !== undefined) {
    assertArrayLength("normal", raw.normal, pointValueCount);
  }
  if (raw.mask !== undefined) {
    assertArrayLength("mask", raw.mask, pixelCount);
    for (let index = 0; index < pixelCount; index += 1) {
      if (!Number.isFinite(raw.mask[index]!)) {
        throw new Error(`postprocessMoGe received non-finite mask value at pixel ${index}`);
      }
    }
  }

  const metricScale = raw.metricScale ?? 1;
  if (!Number.isFinite(metricScale) || metricScale <= 0) {
    throw new Error(
      `postprocessMoGe requires a finite positive metricScale (already exp'd by forward); received ${String(metricScale)}`,
    );
  }

  // v2.py uses `mask > 0.5`.  A missing optional mask is treated as all valid
  // so that a points-only checkpoint still produces the canonical result.
  const mask = new Uint8Array(pixelCount);
  let inputValidCount = 0;
  for (let index = 0; index < pixelCount; index += 1) {
    const isValid = raw.mask === undefined || raw.mask[index]! > 0.5;
    mask[index] = isValid ? 1 : 0;
    inputValidCount += isValid ? 1 : 0;
  }
  if (inputValidCount === 0) {
    throw new Error("postprocessMoGe received an all-invalid mask (no pixels above 0.5)");
  }

  // Inactive pixels are allowed to contain sentinel NaNs/Infinities because
  // the binary mask is authoritative.  Geometry under the mask must be
  // finite, otherwise focal/shift recovery would silently produce nonsense.
  for (let index = 0; index < pixelCount; index += 1) {
    if (mask[index] === 0) {
      continue;
    }
    const pointIndex = index * 3;
    if (
      !Number.isFinite(raw.points[pointIndex]!) ||
      !Number.isFinite(raw.points[pointIndex + 1]!) ||
      !Number.isFinite(raw.points[pointIndex + 2]!)
    ) {
      throw new Error(`postprocessMoGe received non-finite point geometry at valid pixel ${index}`);
    }
    if (raw.normal !== undefined && (
      !Number.isFinite(raw.normal[pointIndex]!) ||
      !Number.isFinite(raw.normal[pointIndex + 1]!) ||
      !Number.isFinite(raw.normal[pointIndex + 2]!)
    )) {
      throw new Error(`postprocessMoGe received non-finite normal geometry at valid pixel ${index}`);
    }
  }

  const { focal, shift } = recoverFocalShift(raw.points, width, height, mask);
  if (!Number.isFinite(focal) || focal <= 0 || !Number.isFinite(shift)) {
    throw new Error(
      `postprocessMoGe could not recover finite positive focal and finite shift (focal=${String(focal)}, shift=${String(shift)})`,
    );
  }

  const aspect = width / height;
  const diagonalFactor = Math.sqrt(1 + aspect * aspect);
  const fx = focal / 2 * diagonalFactor / aspect;
  const fy = focal / 2 * diagonalFactor;
  const intrinsics = new Float32Array([
    fx, 0, 0.5,
    0, fy, 0.5,
    0, 0, 1,
  ]);
  if (!Number.isFinite(intrinsics[0]) || !Number.isFinite(intrinsics[4])) {
    throw new Error("postprocessMoGe produced non-finite camera intrinsics");
  }

  const affineDepth = new Float64Array(pixelCount);
  let positiveDepthCount = 0;
  for (let index = 0; index < pixelCount; index += 1) {
    const shifted = raw.points[index * 3 + 2]! + shift;
    if (mask[index] === 1 && !Number.isFinite(shifted)) {
      throw new Error(`postprocessMoGe produced non-finite shifted depth at valid pixel ${index}`);
    }
    affineDepth[index] = shifted;
    if (mask[index]! === 1 && shifted > 0) {
      mask[index] = 1;
      positiveDepthCount += 1;
    } else {
      // This is the `mask_binary &= points[..., 2] > 0` branch in v2.py.
      mask[index] = 0;
    }
  }
  if (positiveDepthCount === 0) {
    throw new Error("postprocessMoGe invalidated every masked pixel because shifted depth is non-positive");
  }

  // v2.py has force_projection=True in its user-facing path.  Reprojection
  // is linear in depth, so multiplying the unprojected points below by the
  // metric scale is numerically equivalent to unprojecting scaled depth and
  // mirrors the order in the upstream code.
  const unprojected = depthToPoints(affineDepth, width, height, intrinsics);
  const points = new Float32Array(pointValueCount);
  const depth = new Float32Array(pixelCount);
  for (let index = 0; index < pixelCount; index += 1) {
    const pointIndex = index * 3;
    if (mask[index]! === 1) {
      const scaledDepth = affineDepth[index]! * metricScale;
      const scaledX = unprojected[pointIndex]! * metricScale;
      const scaledY = unprojected[pointIndex + 1]! * metricScale;
      const scaledZ = unprojected[pointIndex + 2]! * metricScale;
      // Check after Float32 rounding too: a finite JS number can overflow a
      // model-compatible Float32Array and become an Infinity placeholder.
      if (
        !Number.isFinite(scaledDepth) ||
        !Number.isFinite(scaledX) ||
        !Number.isFinite(scaledY) ||
        !Number.isFinite(scaledZ) ||
        !Number.isFinite(Math.fround(scaledDepth)) ||
        !Number.isFinite(Math.fround(scaledX)) ||
        !Number.isFinite(Math.fround(scaledY)) ||
        !Number.isFinite(Math.fround(scaledZ))
      ) {
        throw new Error(`postprocessMoGe produced non-finite metric geometry at valid pixel ${index}`);
      }
      points[pointIndex] = scaledX;
      points[pointIndex + 1] = scaledY;
      points[pointIndex + 2] = scaledZ;
      depth[index] = scaledDepth;
    } else {
      // torch.where(mask_binary, ..., torch.inf) is the canonical upstream
      // representation.  Consumers must use `mask`; these sentinels are not a
      // substitute for it.
      points[pointIndex] = Number.POSITIVE_INFINITY;
      points[pointIndex + 1] = Number.POSITIVE_INFINITY;
      points[pointIndex + 2] = Number.POSITIVE_INFINITY;
      depth[index] = Number.POSITIVE_INFINITY;
    }
  }

  let normals: Float32Array | undefined;
  if (raw.normal !== undefined) {
    normals = new Float32Array(pointValueCount);
    for (let index = 0; index < pixelCount; index += 1) {
      const pointIndex = index * 3;
      if (mask[index]! === 1) {
        normals[pointIndex] = raw.normal[pointIndex]!;
        normals[pointIndex + 1] = raw.normal[pointIndex + 1]!;
        normals[pointIndex + 2] = raw.normal[pointIndex + 2]!;
      } else {
        // v2.py zeros normals outside the applied mask.
        normals[pointIndex] = 0;
        normals[pointIndex + 1] = 0;
        normals[pointIndex + 2] = 0;
      }
    }
  }

  if (normals === undefined) {
    return { width, height, points, depth, mask, intrinsics };
  }
  return { width, height, points, depth, normals, mask, intrinsics };
}
