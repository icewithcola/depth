import { describe, expect, it } from "vitest";
import {
  finiteDepthStats,
  finitePercentile,
  nearestInterpolateSourceIndex,
  normalizedViewPlaneUv,
  recoverFocalShift,
} from "./math";

function expectClose(actual: number, expected: number, tolerance = 1e-4): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tolerance);
}

describe("MoGe geometry math", () => {
  it("matches the normalized view-plane endpoints and pixel-centre spacing", () => {
    const width = 4;
    const height = 2;
    const aspect = width / height;
    const uv = normalizedViewPlaneUv(width, height);
    const spanX = aspect / Math.sqrt(1 + aspect * aspect);
    const spanY = 1 / Math.sqrt(1 + aspect * aspect);

    expectClose(uv[0]!, -spanX * (width - 1) / width, 1e-7);
    expectClose(uv[1]!, -spanY * (height - 1) / height, 1e-7);
    expectClose(uv[(width - 1) * 2]!, spanX * (width - 1) / width, 1e-7);
    expectClose(uv[(height * width - 1) * 2 + 1]!, spanY * (height - 1) / height, 1e-7);
  });

  it("uses torch nearest interpolation's floor source indices", () => {
    expect(nearestInterpolateSourceIndex(0, 7, 64)).toBe(0);
    expect(nearestInterpolateSourceIndex(8, 7, 64)).toBe(0);
    expect(nearestInterpolateSourceIndex(9, 7, 64)).toBe(0);
    expect(nearestInterpolateSourceIndex(10, 7, 64)).toBe(1);
    expect(nearestInterpolateSourceIndex(63, 7, 64)).toBe(6);
    expect(nearestInterpolateSourceIndex(63, 100, 64)).toBe(98);
  });

  it("recovers a deterministic synthetic focal and Z shift", () => {
    const width = 9;
    const height = 7;
    const focal = 1.85;
    const shift = 0.73;
    const uv = normalizedViewPlaneUv(width, height);
    const points = new Float32Array(width * height * 3);
    const mask = new Uint8Array(width * height).fill(1);
    for (let row = 0; row < height; row += 1) {
      for (let column = 0; column < width; column += 1) {
        const pixel = row * width + column;
        const depth = 2.0 + 0.07 * column + 0.11 * row;
        const point = pixel * 3;
        points[point] = depth * uv[pixel * 2]! / focal;
        points[point + 1] = depth * uv[pixel * 2 + 1]! / focal;
        points[point + 2] = depth - shift;
      }
    }

    const recovered = recoverFocalShift(points, width, height, mask);
    expectClose(recovered.focal, focal, 2e-4);
    expectClose(recovered.shift, shift, 2e-4);
  });

  it("ignores masked samples while preserving the affine solve", () => {
    const width = 8;
    const height = 8;
    const focal = 2.2;
    const shift = -0.3;
    const uv = normalizedViewPlaneUv(width, height);
    const points = new Float32Array(width * height * 3);
    const mask = new Uint8Array(width * height).fill(1);
    for (let row = 0; row < height; row += 1) {
      for (let column = 0; column < width; column += 1) {
        const pixel = row * width + column;
        const depth = 1.5 + 0.08 * column + 0.05 * row;
        const point = pixel * 3;
        points[point] = depth * uv[pixel * 2]! / focal;
        points[point + 1] = depth * uv[pixel * 2 + 1]! / focal;
        points[point + 2] = depth - shift;
      }
    }
    // A masked-out pixel may contain arbitrary model memory and must not
    // influence recovery.
    mask[0] = 0;
    points[0] = Number.NaN;
    points[1] = Number.POSITIVE_INFINITY;
    points[2] = -1000;

    const recovered = recoverFocalShift(points, width, height, mask);
    expectClose(recovered.focal, focal, 2e-4);
    expectClose(recovered.shift, shift, 2e-4);
  });

  it("reports finite percentiles and ignores Infinity placeholders", () => {
    const depth = new Float32Array([1, Number.POSITIVE_INFINITY, 3, Number.NaN, 5]);
    expect(finitePercentile(depth, 50)).toBe(3);
    expect(finitePercentile(depth, 25)).toBe(2);
    expect(finiteDepthStats(depth)).toMatchObject({
      count: 3,
      min: 1,
      max: 5,
      mean: 3,
      p50: 3,
    });
  });
});
