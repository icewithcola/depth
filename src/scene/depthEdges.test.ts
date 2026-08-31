import { describe, expect, it } from 'vitest';
import {
  DEFAULT_DEPTH_RTOL,
  depthMapEdge,
  depthsWithinRelativeTolerance,
  relativeDepthDifference,
} from './depthEdges';

describe('depthMapEdge', () => {
  it('marks both sides of a foreground/background jump', () => {
    const edges = depthMapEdge(new Float32Array([1, 1, 2, 2]), 2, 2, DEFAULT_DEPTH_RTOL);
    expect(Array.from(edges)).toEqual([1, 1, 1, 1]);
  });

  it('ignores invalid masked neighbours and invalid centres', () => {
    const depth = new Float32Array([1, 1, 100, Number.NaN]);
    const mask = new Uint8Array([1, 1, 0, 0]);
    const edges = depthMapEdge(depth, { width: 2, height: 2, mask, rtol: 0.04 });
    expect(Array.from(edges)).toEqual([0, 0, 0, 0]);
  });

  it('uses a symmetric positive-depth comparison for triangle guards', () => {
    expect(relativeDepthDifference(1, 1.04)).toBeCloseTo(0.04, 6);
    expect(depthsWithinRelativeTolerance(1, 1.04, 0.04)).toBe(true);
    expect(depthsWithinRelativeTolerance(1, 1.05, 0.04)).toBe(false);
    expect(relativeDepthDifference(0, 1)).toBe(Infinity);
  });
});
