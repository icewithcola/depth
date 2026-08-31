/**
 * Utilities for finding discontinuities in a row-major depth map.
 *
 * MoGe uses the `utils3d.np.depth_map_edge` convention: for every pixel, the
 * range of the values in a small neighbourhood is compared with the value at
 * the pixel itself.  A pixel is an edge when that relative range is greater
 * than `rtol`.  Keeping this implementation local means that mesh creation
 * does not need a NumPy (or renderer) dependency.
 */

export const DEFAULT_DEPTH_RTOL = 0.04;

export interface DepthEdgeOptions {
  width: number;
  height: number;
  /** Relative range threshold. Defaults to the MoGe value, 0.04. */
  rtol?: number;
  /** Odd neighbourhood size. Defaults to 3, as in depth_map_edge. */
  kernelSize?: number;
  /** Optional validity mask. Falsy entries are ignored as neighbours. */
  mask?: ArrayLike<number | boolean>;
}

interface NormalizedDepthEdgeOptions {
  width: number;
  height: number;
  rtol: number;
  kernelSize: number;
  mask?: ArrayLike<number | boolean>;
}

/**
 * Compute a depth-edge mask for a row-major depth map.
 *
 * The numeric overload is convenient for scene code:
 * `depthMapEdge(depth, width, height, 0.04)`.  The options overload is useful
 * when a validity mask or a different neighbourhood is available.  The
 * returned Uint8Array contains 0/1 values so it can be passed to WebGL or
 * used as a compact typed validity map without conversion.
 */
export function depthMapEdge(
  depth: ArrayLike<number>,
  width: number,
  height: number,
  rtol?: number,
  mask?: ArrayLike<number | boolean>,
): Uint8Array;
export function depthMapEdge(depth: ArrayLike<number>, options: DepthEdgeOptions): Uint8Array;
export function depthMapEdge(
  depth: ArrayLike<number>,
  widthOrOptions: number | DepthEdgeOptions,
  height?: number,
  rtol = DEFAULT_DEPTH_RTOL,
  mask?: ArrayLike<number | boolean>,
): Uint8Array {
  const options: NormalizedDepthEdgeOptions = typeof widthOrOptions === 'number'
    ? {
        width: widthOrOptions,
        height: height ?? 0,
        rtol,
        kernelSize: 3,
        ...(mask === undefined ? {} : { mask }),
      }
    : {
        width: widthOrOptions.width,
        height: widthOrOptions.height,
        rtol: widthOrOptions.rtol ?? DEFAULT_DEPTH_RTOL,
        kernelSize: widthOrOptions.kernelSize ?? 3,
        ...(widthOrOptions.mask === undefined ? {} : { mask: widthOrOptions.mask }),
      };

  const { width, height: mapHeight } = options;
  if (!Number.isInteger(width) || width < 0 || !Number.isInteger(mapHeight) || mapHeight < 0) {
    throw new RangeError(`Depth map dimensions must be non-negative integers; got ${width}x${mapHeight}`);
  }
  if (depth.length < width * mapHeight) {
    throw new RangeError(`Depth map has ${depth.length} values, expected at least ${width * mapHeight}`);
  }

  const result = new Uint8Array(width * mapHeight);
  if (width === 0 || mapHeight === 0 || !Number.isFinite(options.rtol)) return result;

  // The upstream helper uses a centred odd window.  Accepting an even value
  // would make the centre ambiguous, so round it up to the next odd size.
  const requestedKernel = Number.isFinite(options.kernelSize)
    ? Math.max(1, Math.floor(options.kernelSize))
    : 3;
  const kernelSize = requestedKernel % 2 === 0 ? requestedKernel + 1 : requestedKernel;
  const radius = Math.floor(kernelSize / 2);
  const threshold = Math.max(0, options.rtol);

  const isValid = (index: number): boolean => {
    const value = depth[index];
    if (value === undefined || !Number.isFinite(value) || value <= 0) return false;
    const validityMask = options.mask;
    return validityMask === undefined || Boolean(validityMask[index]);
  };

  for (let row = 0; row < mapHeight; row += 1) {
    for (let column = 0; column < width; column += 1) {
      const centerIndex = row * width + column;
      if (!isValid(centerIndex)) continue;

      const centerDepth = depth[centerIndex];
      // noUncheckedIndexedAccess makes the guard explicit even though the
      // length check above guarantees this value exists.
      if (centerDepth === undefined) continue;

      let minimum = centerDepth;
      let maximum = centerDepth;
      const rowStart = Math.max(0, row - radius);
      const rowEnd = Math.min(mapHeight - 1, row + radius);
      const columnStart = Math.max(0, column - radius);
      const columnEnd = Math.min(width - 1, column + radius);

      for (let neighbourRow = rowStart; neighbourRow <= rowEnd; neighbourRow += 1) {
        for (let neighbourColumn = columnStart; neighbourColumn <= columnEnd; neighbourColumn += 1) {
          const neighbourIndex = neighbourRow * width + neighbourColumn;
          if (!isValid(neighbourIndex)) continue;
          const neighbourDepth = depth[neighbourIndex];
          if (neighbourDepth === undefined) continue;
          minimum = Math.min(minimum, neighbourDepth);
          maximum = Math.max(maximum, neighbourDepth);
        }
      }

      // This is equivalent to max_pool(depth) + max_pool(-depth), followed by
      // division by the centre depth, for positive finite depths.
      if ((maximum - minimum) / centerDepth > threshold) result[centerIndex] = 1;
    }
  }

  return result;
}

/** Alias with an explicit name for callers that prefer mask terminology. */
export const computeDepthEdgeMask = depthMapEdge;
export const depth_map_edge = depthMapEdge;
export const createDepthEdgeMask = depthMapEdge;

/**
 * Symmetric relative depth difference used for the final triangle-local
 * guard.  Dividing by the smaller positive depth rejects both sides of a
 * foreground/background jump, unlike an asymmetric `(a - b) / a` test.
 */
export function relativeDepthDifference(first: number, second: number): number {
  if (!Number.isFinite(first) || !Number.isFinite(second) || first <= 0 || second <= 0) {
    return Number.POSITIVE_INFINITY;
  }
  return Math.abs(first - second) / Math.min(first, second);
}

/** Return whether two positive depths can share a triangle edge. */
export function depthsWithinRelativeTolerance(
  first: number,
  second: number,
  rtol = DEFAULT_DEPTH_RTOL,
): boolean {
  const threshold = Math.max(0, rtol);
  return relativeDepthDifference(first, second) <= threshold + 1e-12;
}
