import { describe, expect, it } from "vitest";
import { normalizedViewPlaneUv } from "./math";
import { postprocessMoGe } from "./postprocess";
import type { RawMoGeOutputs } from "./types";

function syntheticRaw(): {
  raw: RawMoGeOutputs;
  expectedDepth: Float64Array;
  focal: number;
  metricScale: number;
  invalidPixel: number;
} {
  const width = 8;
  const height = 6;
  const focal = 2.05;
  const shift = 0.6;
  const metricScale = 3.25;
  const invalidPixel = 3;
  const uv = normalizedViewPlaneUv(width, height);
  const points = new Float32Array(width * height * 3);
  const normal = new Float32Array(width * height * 3);
  const mask = new Float32Array(width * height).fill(0.9);
  const expectedDepth = new Float64Array(width * height);

  for (let row = 0; row < height; row += 1) {
    for (let column = 0; column < width; column += 1) {
      const pixel = row * width + column;
      // Keep all geometry projective, including one negative-depth sample, so
      // recovery remains exact before v2.py's positive-depth invalidation.
      const depth = pixel === invalidPixel ? -0.8 : 2.5 + row * 0.12 + column * 0.06;
      const point = pixel * 3;
      points[point] = depth * uv[pixel * 2]! / focal;
      points[point + 1] = depth * uv[pixel * 2 + 1]! / focal;
      points[point + 2] = depth - shift;
      expectedDepth[pixel] = depth;
      normal[point] = 0;
      normal[point + 1] = 0;
      normal[point + 2] = 1;
    }
  }

  const raw: RawMoGeOutputs = {
    width,
    height,
    points,
    normal,
    mask,
    metricScale,
  };
  return { raw, expectedDepth, focal, metricScale, invalidPixel };
}

describe("MoGe postprocess", () => {
  it("recovers intrinsics, applies metric scale, projects depth, and masks invalid depth", () => {
    const { raw, expectedDepth, focal, metricScale, invalidPixel } = syntheticRaw();
    const result = postprocessMoGe(raw);
    const aspect = raw.width / raw.height;
    const expectedFx = focal / 2 * Math.sqrt(1 + aspect * aspect) / aspect;
    const expectedFy = focal / 2 * Math.sqrt(1 + aspect * aspect);

    expect(result.width).toBe(raw.width);
    expect(result.height).toBe(raw.height);
    expect(result.intrinsics[0]).toBeCloseTo(expectedFx, 4);
    expect(result.intrinsics[4]).toBeCloseTo(expectedFy, 4);
    expect(result.intrinsics[2]).toBe(0.5);
    expect(result.intrinsics[5]).toBe(0.5);
    expect(result.mask[invalidPixel]).toBe(0);
    expect(result.depth[invalidPixel]).toBe(Infinity);
    expect(result.points[invalidPixel * 3]).toBe(Infinity);
    expect(result.points[invalidPixel * 3 + 1]).toBe(Infinity);
    expect(result.points[invalidPixel * 3 + 2]).toBe(Infinity);
    expect(result.normals?.[invalidPixel * 3]).toBe(0);
    expect(result.normals?.[invalidPixel * 3 + 1]).toBe(0);
    expect(result.normals?.[invalidPixel * 3 + 2]).toBe(0);

    const samplePixel = 2 * raw.width + 5;
    expect(result.mask[samplePixel]).toBe(1);
    expect(result.depth[samplePixel]).toBeCloseTo(expectedDepth[samplePixel]! * metricScale, 4);
    const u = (5 + 0.5) / raw.width;
    const v = (2 + 0.5) / raw.height;
    const expectedDepthMetric = expectedDepth[samplePixel]! * metricScale;
    expect(result.points[samplePixel * 3]).toBeCloseTo((u - 0.5) / expectedFx * expectedDepthMetric, 4);
    expect(result.points[samplePixel * 3 + 1]).toBeCloseTo((v - 0.5) / expectedFy * expectedDepthMetric, 4);
    expect(result.points[samplePixel * 3 + 2]).toBeCloseTo(expectedDepthMetric, 4);
    expect(result.normals?.[samplePixel * 3 + 2]).toBeCloseTo(1, 5);
  });

  it("uses the binary threshold and rejects malformed/all-invalid scale inputs", () => {
    const { raw } = syntheticRaw();
    raw.mask?.fill(0.49);
    expect(() => postprocessMoGe(raw)).toThrow(/all-invalid/);

    const malformed: RawMoGeOutputs = { ...syntheticRaw().raw, metricScale: Number.NaN };
    expect(() => postprocessMoGe(malformed)).toThrow(/metricScale/);

    const badLength: RawMoGeOutputs = {
      width: 2,
      height: 2,
      points: new Float32Array(3),
    };
    expect(() => postprocessMoGe(badLength)).toThrow(/exactly 12 values/);
  });

  it("allows non-finite sentinels only outside the authoritative mask", () => {
    const { raw } = syntheticRaw();
    raw.mask![0] = 0;
    raw.points[0] = Number.NaN;
    raw.points[1] = Number.POSITIVE_INFINITY;
    raw.normal![0] = Number.NaN;
    const result = postprocessMoGe(raw);
    expect(result.mask[0]).toBe(0);
    expect(result.depth[0]).toBe(Infinity);

    const validPixel = 1;
    raw.points[validPixel * 3] = Number.NaN;
    expect(() => postprocessMoGe(raw)).toThrow(/valid pixel 1/);
  });
});
