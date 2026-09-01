import type { MoGeResult } from '../moge/types';
import {
  DEFAULT_DEPTH_RTOL,
  depthMapEdge,
  depthsWithinRelativeTolerance,
} from './depthEdges';

export interface CreateMeshOptions {
  /** Maximum number of samples on the source image's long side. */
  targetLongSide?: number;
  /** Alias for targetLongSide. */
  longSide?: number;
  /** Explicit regular source-pixel stride, useful for deterministic tests. */
  step?: number;
  /** Relative depth tolerance for edge and triangle rejection. */
  depthRtol?: number;
  /** Alias for depthRtol. */
  rtol?: number;
  /** Remove source pixels marked by depthMapEdge. Defaults to true. */
  removeDepthEdges?: boolean;
  /**
   * Pixel connectivity used when keeping the largest renderable component.
   * Four-neighbour connectivity is the default so diagonal contact does not
   * merge otherwise separate subjects.
   */
  componentConnectivity?: ComponentConnectivity;
}

export type ComponentConnectivity = 4 | 8;

export interface MeshBuildStats {
  vertexCount: number;
  triangleCount: number;
  sampleWidth: number;
  sampleHeight: number;
  /** Wall-clock construction time in milliseconds. */
  buildMs: number;
  /** Regular source-pixel stride used to create the sample arrays. */
  step: number;
  /** Number of source pixels marked as depth edges. */
  depthEdgeCount: number;
}

export interface MeshData {
  /** Three.js-compatible positions: (x, -y, -z) from OpenCV coordinates. */
  positions: Float32Array;
  /** Image UVs in OpenGL orientation: (u, 1-v). */
  uv: Float32Array;
  /** Plural alias retained for callers that use Three.js attribute naming. */
  uvs: Float32Array;
  /** Indexed triangles with FrontSide winding for a camera looking down -Z. */
  indices: Uint32Array;
  /** Singular alias retained for renderer adapters. */
  index: Uint32Array;
  /** Converted normals when the MoGe result contains normals. */
  normals?: Float32Array;
  /** Singular alias retained for MoGe's source field name. */
  normal?: Float32Array;
  /** Full-resolution 0/1 edge mask used while constructing the mesh. */
  depthEdgeMask: Uint8Array;
  /** Full-resolution mask of the largest renderable component. */
  validMask: Uint8Array;
  /** Source rows and columns represented by each sample row/column. */
  sampleRows: Uint32Array;
  sampleColumns: Uint32Array;
  stats: MeshBuildStats;
  /** Top-level aliases make the renderer adapter independent of stats shape. */
  vertexCount: number;
  triangleCount: number;
  sampleWidth: number;
  sampleHeight: number;
  buildMs: number;
  step: number;
}

export type ThreeCoordinate = Float32Array;

/** Convert one OpenCV-coordinate point to Three.js camera coordinates. */
export function convertOpenCvPointToThree(point: ArrayLike<number>, target?: Float32Array): ThreeCoordinate {
  const x = point[0];
  const y = point[1];
  const z = point[2];
  if (x === undefined || y === undefined || z === undefined) {
    throw new RangeError('A 3D point must contain at least three values');
  }
  const converted = target ?? new Float32Array(3);
  if (converted.length < 3) throw new RangeError('A converted 3D point needs an output with three values');
  converted[0] = x;
  converted[1] = y === 0 ? 0 : -y;
  converted[2] = z === 0 ? 0 : -z;
  return converted;
}

/** Convert one OpenCV-coordinate normal to Three.js coordinates. */
export function convertOpenCvNormalToThree(normal: ArrayLike<number>, target?: Float32Array): ThreeCoordinate {
  return convertOpenCvPointToThree(normal, target);
}

/** Convert an interleaved OpenCV HWC point/normal map at the scene boundary. */
export function convertOpenCvMapToThree(values: ArrayLike<number>): Float32Array {
  if (values.length % 3 !== 0) throw new RangeError('An interleaved 3D map length must be divisible by three');
  const converted = new Float32Array(values.length);
  for (let offset = 0; offset < values.length; offset += 3) {
    const point = convertOpenCvPointToThree([
      values[offset] ?? 0,
      values[offset + 1] ?? 0,
      values[offset + 2] ?? 0,
    ]);
    converted[offset] = point[0] ?? 0;
    converted[offset + 1] = point[1] ?? 0;
    converted[offset + 2] = point[2] ?? 0;
  }
  return converted;
}

/** Short aliases useful at a scene boundary. */
export const opencvToThree = convertOpenCvPointToThree;
export const opencvNormalToThree = convertOpenCvNormalToThree;
export const convertPointToThree = convertOpenCvPointToThree;

function nowMs(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

function makeSampleIndices(length: number, step: number): Uint32Array {
  if (length <= 0) return new Uint32Array(0);
  const values: number[] = [];
  for (let index = 0; index < length; index += step) values.push(index);
  const last = length - 1;
  if (values[values.length - 1] !== last) values.push(last);
  return Uint32Array.from(values);
}

function finitePoint(points: ArrayLike<number>, offset: number): boolean {
  const x = points[offset];
  const y = points[offset + 1];
  const z = points[offset + 2];
  return x !== undefined && y !== undefined && z !== undefined
    && Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z);
}

function activeMaskValue(value: number | boolean | undefined): boolean {
  return value !== undefined && Boolean(value);
}

/**
 * Keep only the largest connected component in a row-major binary mask.
 *
 * The scan order makes ties deterministic: when two components have the same
 * area, the one whose first pixel occurs first in row-major order wins. The
 * returned mask is always a fresh Uint8Array and therefore can safely be
 * modified by a caller without changing the source mask.
 */
export function largestConnectedComponentMask(
  mask: ArrayLike<number | boolean>,
  width: number,
  height: number,
  connectivity: ComponentConnectivity = 4,
): Uint8Array {
  if (!Number.isInteger(width) || width < 0 || !Number.isInteger(height) || height < 0) {
    throw new RangeError(`Component mask dimensions must be non-negative integers; got ${width}x${height}`);
  }
  if (connectivity !== 4 && connectivity !== 8) {
    throw new RangeError(`Component connectivity must be 4 or 8; got ${connectivity}`);
  }
  const pixelCount = width * height;
  if (mask.length < pixelCount) {
    throw new RangeError(`Component mask has ${mask.length} values, expected at least ${pixelCount}`);
  }

  const selected = new Uint8Array(pixelCount);
  if (pixelCount === 0) return selected;

  const visited = new Uint8Array(pixelCount);
  // The queue also stores the members in discovery order. Once a component is
  // exhausted, queue[0..tail) is its complete pixel list and can be copied to
  // `selected` if it is larger than the current winner.
  const queue = new Int32Array(pixelCount);
  let largestSize = 0;

  const enqueue = (index: number, tail: { value: number }): void => {
    if (visited[index] !== 0 || !activeMaskValue(mask[index])) return;
    visited[index] = 1;
    queue[tail.value] = index;
    tail.value += 1;
  };

  for (let start = 0; start < pixelCount; start += 1) {
    if (visited[start] !== 0 || !activeMaskValue(mask[start])) continue;

    const tail = { value: 1 };
    queue[0] = start;
    visited[start] = 1;
    let head = 0;
    while (head < tail.value) {
      const index = queue[head] ?? 0;
      head += 1;
      const row = Math.floor(index / width);
      const column = index - row * width;

      if (column > 0) enqueue(index - 1, tail);
      if (column + 1 < width) enqueue(index + 1, tail);
      if (row > 0) enqueue(index - width, tail);
      if (row + 1 < height) enqueue(index + width, tail);

      if (connectivity === 8) {
        if (row > 0 && column > 0) enqueue(index - width - 1, tail);
        if (row > 0 && column + 1 < width) enqueue(index - width + 1, tail);
        if (row + 1 < height && column > 0) enqueue(index + width - 1, tail);
        if (row + 1 < height && column + 1 < width) enqueue(index + width + 1, tail);
      }
    }

    if (tail.value <= largestSize) continue;
    selected.fill(0);
    for (let member = 0; member < tail.value; member += 1) {
      const index = queue[member];
      if (index !== undefined) selected[index] = 1;
    }
    largestSize = tail.value;
  }

  return selected;
}

function validSourcePixel(
  result: MoGeResult,
  index: number,
  edgeMask: Uint8Array,
  removeDepthEdges: boolean,
): boolean {
  const depth = result.depth[index];
  return result.mask[index] === 1
    && depth !== undefined
    && Number.isFinite(depth)
    && depth > 0
    && finitePoint(result.points, index * 3)
    && (!removeDepthEdges || edgeMask[index] === 0);
}

function validTriangleDepths(
  result: MoGeResult,
  sourceIndices: readonly [number, number, number],
  rtol: number,
): boolean {
  const first = result.depth[sourceIndices[0]];
  const second = result.depth[sourceIndices[1]];
  const third = result.depth[sourceIndices[2]];
  if (first === undefined || second === undefined || third === undefined) return false;
  return depthsWithinRelativeTolerance(first, second, rtol)
    && depthsWithinRelativeTolerance(first, third, rtol)
    && depthsWithinRelativeTolerance(second, third, rtol);
}

function finitePositiveDimension(value: number): boolean {
  return Number.isFinite(value) && Number.isInteger(value) && value > 0;
}

/**
 * Build a regular, indexed, depth-aware image mesh from a canonical MoGe
 * result.  Source rows/columns are sampled with a fixed stride and the final
 * row/column are always appended, preserving the complete image boundary.
 */
export function createMeshData(result: MoGeResult, options: CreateMeshOptions = {}): MeshData {
  const started = nowMs();
  const { width, height } = result;
  if (!finitePositiveDimension(width) || !finitePositiveDimension(height)) {
    throw new RangeError(`MoGe result dimensions must be positive integers; got ${width}x${height}`);
  }
  const pixelCount = width * height;
  if (result.points.length < pixelCount * 3) {
    throw new RangeError(`MoGe points has ${result.points.length} values, expected at least ${pixelCount * 3}`);
  }
  if (result.depth.length < pixelCount || result.mask.length < pixelCount) {
    throw new RangeError('MoGe depth and mask must contain one value per source pixel');
  }
  if (result.normals !== undefined && result.normals.length < pixelCount * 3) {
    throw new RangeError(`MoGe normals has ${result.normals.length} values, expected at least ${pixelCount * 3}`);
  }

  const requestedTarget = options.longSide ?? options.targetLongSide ?? 360;
  if (!Number.isFinite(requestedTarget) || requestedTarget <= 0) {
    throw new RangeError(`targetLongSide must be positive; got ${requestedTarget}`);
  }
  const explicitStep = options.step;
  if (explicitStep !== undefined && (!Number.isFinite(explicitStep) || explicitStep < 1)) {
    throw new RangeError(`step must be at least one; got ${explicitStep}`);
  }
  const step = Math.max(
    1,
    Math.floor(explicitStep ?? Math.ceil(Math.max(width, height) / requestedTarget)),
  );
  const requestedRtol = options.rtol ?? options.depthRtol ?? DEFAULT_DEPTH_RTOL;
  const depthRtol = Number.isFinite(requestedRtol) ? Math.max(0, requestedRtol) : DEFAULT_DEPTH_RTOL;
  const removeDepthEdges = options.removeDepthEdges ?? true;

  const edgeMask = depthMapEdge(result.depth, {
    width,
    height,
    rtol: depthRtol,
    mask: result.mask,
  });
  let depthEdgeCount = 0;
  for (const edge of edgeMask) depthEdgeCount += edge;

  // Select the subject at full source resolution before regular sampling.
  // Connectivity is based on pixels that can actually become mesh vertices;
  // this lets depth-edge cuts separate adjacent surfaces and prevents invalid
  // model sentinels from contributing to a component's area.
  const renderableMask = new Uint8Array(pixelCount);
  for (let index = 0; index < pixelCount; index += 1) {
    renderableMask[index] = validSourcePixel(result, index, edgeMask, removeDepthEdges) ? 1 : 0;
  }
  const validMask = largestConnectedComponentMask(
    renderableMask,
    width,
    height,
    options.componentConnectivity ?? 4,
  );

  const sampleColumns = makeSampleIndices(width, step);
  const sampleRows = makeSampleIndices(height, step);
  const sampleWidth = sampleColumns.length;
  const sampleHeight = sampleRows.length;

  // Source-grid IDs. -1 is intentional: an invalid or edge sample cannot
  // participate in a triangle and therefore has no renderer vertex.
  const sampleIds = new Int32Array(sampleWidth * sampleHeight);
  sampleIds.fill(-1);

  const positionValues: number[] = [];
  const uvValues: number[] = [];
  const normalValues: number[] = [];
  const sourceDepths: number[] = [];
  const sourceIndices: number[] = [];
  const hasNormals = result.normals !== undefined;

  // MoGe's image maps represent pixel centres (`uv_map` uses (i + .5)/N),
  // rather than the outer corners of the texture.  Keeping that convention
  // also makes these UVs line up exactly with depthToPoints/postprocess.
  const normalizedU = (column: number): number => (column + 0.5) / width;
  const normalizedV = (row: number): number => (row + 0.5) / height;

  for (let sampleRow = 0; sampleRow < sampleHeight; sampleRow += 1) {
    const row = sampleRows[sampleRow];
    if (row === undefined) continue;
    for (let sampleColumn = 0; sampleColumn < sampleWidth; sampleColumn += 1) {
      const column = sampleColumns[sampleColumn];
      if (column === undefined) continue;
      const sourceIndex = row * width + column;
      if (validMask[sourceIndex] !== 1) continue;

      const pointOffset = sourceIndex * 3;
      const point = convertOpenCvPointToThree(result.points.subarray(pointOffset, pointOffset + 3));
      const vertexIndex = sourceDepths.length;
      sampleIds[sampleRow * sampleWidth + sampleColumn] = vertexIndex;
      sourceIndices.push(sourceIndex);
      const depth = result.depth[sourceIndex];
      // validSourcePixel checked this; keeping the guard avoids propagating an
      // unchecked undefined value under noUncheckedIndexedAccess.
      if (depth === undefined) continue;
      sourceDepths.push(depth);
      positionValues.push(point[0] ?? 0, point[1] ?? 0, point[2] ?? 0);
      uvValues.push(normalizedU(column), 1 - normalizedV(row));

      if (hasNormals) {
        const normalOffset = pointOffset;
        const normal = result.normals;
        if (normal !== undefined && finitePoint(normal, normalOffset)) {
          const convertedNormal = convertOpenCvNormalToThree(
            normal.subarray(normalOffset, normalOffset + 3),
          );
          normalValues.push(convertedNormal[0] ?? 0, convertedNormal[1] ?? 0, convertedNormal[2] ?? 0);
        } else {
          normalValues.push(0, 0, 0);
        }
      }
    }
  }

  const triangleValues: number[] = [];
  const addTriangle = (
    firstSampleRow: number,
    firstSampleColumn: number,
    secondSampleRow: number,
    secondSampleColumn: number,
    thirdSampleRow: number,
    thirdSampleColumn: number,
  ): void => {
    const firstId = sampleIds[firstSampleRow * sampleWidth + firstSampleColumn];
    const secondId = sampleIds[secondSampleRow * sampleWidth + secondSampleColumn];
    const thirdId = sampleIds[thirdSampleRow * sampleWidth + thirdSampleColumn];
    if (firstId === undefined || secondId === undefined || thirdId === undefined) return;
    if (firstId < 0 || secondId < 0 || thirdId < 0) return;

    const firstSource = sourceIndices[firstId];
    const secondSource = sourceIndices[secondId];
    const thirdSource = sourceIndices[thirdId];
    if (firstSource === undefined || secondSource === undefined || thirdSource === undefined) return;
    if (!validTriangleDepths(result, [firstSource, secondSource, thirdSource], depthRtol)) return;

    // Image rows run downwards.  In Three coordinates (x, -y, -z), this
    // [top-left, bottom-left, top-right] winding faces the camera at origin
    // looking toward -Z; the second triangle continues the same winding.
    triangleValues.push(firstId, secondId, thirdId);
  };

  for (let sampleRow = 0; sampleRow + 1 < sampleHeight; sampleRow += 1) {
    for (let sampleColumn = 0; sampleColumn + 1 < sampleWidth; sampleColumn += 1) {
      addTriangle(sampleRow, sampleColumn, sampleRow + 1, sampleColumn, sampleRow, sampleColumn + 1);
      addTriangle(sampleRow, sampleColumn + 1, sampleRow + 1, sampleColumn, sampleRow + 1, sampleColumn + 1);
    }
  }

  // Keep the attribute buffers compact by dropping valid but isolated samples
  // (for example, a lone valid pixel next to invalid mask values).
  const used = new Uint8Array(sourceDepths.length);
  for (const index of triangleValues) {
    if (index >= 0 && index < used.length) used[index] = 1;
  }
  const remap = new Int32Array(sourceDepths.length);
  remap.fill(-1);
  const compactPositions: number[] = [];
  const compactUvs: number[] = [];
  const compactNormals: number[] = [];
  let compactCount = 0;
  for (let index = 0; index < sourceDepths.length; index += 1) {
    if (used[index] === 0) continue;
    remap[index] = compactCount;
    compactPositions.push(
      positionValues[index * 3] ?? 0,
      positionValues[index * 3 + 1] ?? 0,
      positionValues[index * 3 + 2] ?? 0,
    );
    compactUvs.push(uvValues[index * 2] ?? 0, uvValues[index * 2 + 1] ?? 0);
    if (hasNormals) {
      compactNormals.push(
        normalValues[index * 3] ?? 0,
        normalValues[index * 3 + 1] ?? 0,
        normalValues[index * 3 + 2] ?? 0,
      );
    }
    compactCount += 1;
  }

  const compactTriangles = triangleValues.map((index) => remap[index] ?? -1);
  const indices = Uint32Array.from(compactTriangles.filter((index) => index >= 0));
  const positions = Float32Array.from(compactPositions);
  const uv = Float32Array.from(compactUvs);
  const normals = hasNormals ? Float32Array.from(compactNormals) : undefined;
  const buildMs = Math.max(0, nowMs() - started);
  const stats: MeshBuildStats = {
    vertexCount: compactCount,
    triangleCount: Math.floor(indices.length / 3),
    sampleWidth,
    sampleHeight,
    buildMs,
    step,
    depthEdgeCount,
  };

  return {
    positions,
    uv,
    uvs: uv,
    indices,
    index: indices,
    ...(normals === undefined ? {} : { normals, normal: normals }),
    depthEdgeMask: edgeMask,
    validMask,
    sampleRows,
    sampleColumns,
    stats,
    vertexCount: stats.vertexCount,
    triangleCount: stats.triangleCount,
    sampleWidth: stats.sampleWidth,
    sampleHeight: stats.sampleHeight,
    buildMs: stats.buildMs,
    step: stats.step,
  };
}

export const buildMeshData = createMeshData;
export const createIndexedMesh = createMeshData;
export const buildIndexedMesh = createMeshData;
export const opencvToThreePosition = convertOpenCvPointToThree;
export const opencvToThreeNormal = convertOpenCvNormalToThree;
export const convertOpenCvPointsToThree = convertOpenCvMapToThree;
export const convertOpenCVPointToThree = convertOpenCvPointToThree;
export const convertOpenCVNormalToThree = convertOpenCvNormalToThree;
