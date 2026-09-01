import { describe, expect, it } from 'vitest';
import type { MoGeResult } from '../moge/types';
import {
  convertOpenCvNormalToThree,
  convertOpenCvPointToThree,
  createMeshData,
  largestConnectedComponentMask,
} from './createMesh';

function makeResult(width: number, height: number, depths?: ArrayLike<number>, mask?: Uint8Array): MoGeResult {
  const depth = Float32Array.from(depths ?? Array.from({ length: width * height }, () => 1));
  const points = new Float32Array(width * height * 3);
  const intrinsics = new Float32Array([0.5, 0, 0.5, 0, 0.5, 0.5, 0, 0, 1]);
  for (let row = 0; row < height; row += 1) {
    for (let column = 0; column < width; column += 1) {
      const index = row * width + column;
      const z = depth[index] ?? 1;
      points[index * 3] = ((column / Math.max(1, width - 1)) - 0.5) * z / 0.5;
      points[index * 3 + 1] = ((row / Math.max(1, height - 1)) - 0.5) * z / 0.5;
      points[index * 3 + 2] = z;
    }
  }
  return {
    width,
    height,
    points,
    depth,
    mask: mask ?? new Uint8Array(width * height).fill(1),
    intrinsics,
  };
}

describe('createMeshData', () => {
  it('keeps the largest four-connected source component before downsampling', () => {
    const width = 7;
    const height = 4;
    const mask = new Uint8Array([
      1, 1, 1, 0, 0, 0, 0,
      1, 1, 1, 0, 1, 1, 0,
      1, 1, 1, 0, 1, 1, 0,
      0, 0, 0, 0, 0, 0, 0,
    ]);
    const mesh = createMeshData(makeResult(width, height, undefined, mask), {
      step: 1,
      removeDepthEdges: false,
    });

    expect(Array.from(mesh.validMask)).toEqual([
      1, 1, 1, 0, 0, 0, 0,
      1, 1, 1, 0, 0, 0, 0,
      1, 1, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0,
    ]);
    expect(mesh.vertexCount).toBe(9);
    expect(mesh.triangleCount).toBe(8);
  });

  it('uses the first row-major component when areas tie and does not merge diagonals', () => {
    const tie = largestConnectedComponentMask(
      new Uint8Array([
        1, 1, 0, 1, 1,
        1, 1, 0, 1, 1,
      ]),
      5,
      2,
    );
    expect(Array.from(tie)).toEqual([
      1, 1, 0, 0, 0,
      1, 1, 0, 0, 0,
    ]);

    const diagonal = largestConnectedComponentMask(
      new Uint8Array([
        1, 0,
        0, 1,
      ]),
      2,
      2,
    );
    expect(Array.from(diagonal)).toEqual([1, 0, 0, 0]);
    expect(Array.from(largestConnectedComponentMask(
      new Uint8Array([1, 0, 0, 1]),
      2,
      2,
      8,
    ))).toEqual([1, 0, 0, 1]);
  });

  it('returns an empty selection for an empty mask and preserves thin winners', () => {
    expect(Array.from(largestConnectedComponentMask(new Uint8Array(6), 3, 2))).toEqual([
      0, 0, 0,
      0, 0, 0,
    ]);

    const thin = new Uint8Array([
      1, 0, 0, 0,
      1, 0, 0, 0,
      1, 1, 0, 1,
    ]);
    expect(Array.from(largestConnectedComponentMask(thin, 4, 3))).toEqual([
      1, 0, 0, 0,
      1, 0, 0, 0,
      1, 1, 0, 0,
    ]);
  });

  it('builds components from renderable pixels, so invalid geometry cannot bridge subjects', () => {
    const depths = new Float32Array([
      1, 1, 1, Number.NaN, 1, 1, 1,
      1, 1, 1, Number.NaN, 1, 1, 1,
    ]);
    const mesh = createMeshData(makeResult(7, 2, depths), {
      step: 1,
      removeDepthEdges: false,
    });
    expect(Array.from(mesh.validMask)).toEqual([
      1, 1, 1, 0, 0, 0, 0,
      1, 1, 1, 0, 0, 0, 0,
    ]);
  });

  it('converts OpenCV positions/normals and keeps OpenGL UV orientation', () => {
    expect(Array.from(convertOpenCvPointToThree([1, 2, 3]))).toEqual([1, -2, -3]);
    expect(Array.from(convertOpenCvNormalToThree([0, 1, 0]))).toEqual([0, -1, 0]);

    const result = makeResult(2, 2);
    result.normals = new Float32Array(Array.from({ length: 12 }, (_, index) => (index % 3 === 2 ? 1 : 0)));
    const mesh = createMeshData(result, { removeDepthEdges: false });
    expect(mesh.stats.sampleWidth).toBe(2);
    expect(mesh.stats.sampleHeight).toBe(2);
    expect(Array.from(mesh.uv)).toEqual([0.25, 0.75, 0.75, 0.75, 0.25, 0.25, 0.75, 0.25]);
    // [top-left,bottom-left,top-right], then [top-right,bottom-left,bottom-right]
    expect(Array.from(mesh.indices)).toEqual([0, 2, 1, 1, 2, 3]);
    expect(Array.from(mesh.positions.slice(0, 3))).toEqual([-1, 1, -1]);
    expect(Array.from(mesh.normals ?? [])).toEqual([
      0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1,
    ]);
  });

  it('always includes the final source row and column when downsampling', () => {
    const mesh = createMeshData(makeResult(5, 4), { longSide: 3, removeDepthEdges: false });
    expect(mesh.stats.step).toBe(2);
    expect(mesh.sampleColumns).toEqual(new Uint32Array([0, 2, 4]));
    expect(mesh.sampleRows).toEqual(new Uint32Array([0, 2, 3]));
    expect(mesh.stats.sampleWidth).toBe(3);
    expect(mesh.stats.sampleHeight).toBe(3);
  });

  it('rejects crossing triangles and invalid samples', () => {
    const jump = createMeshData(makeResult(2, 2, [1, 1, 2, 2]), { removeDepthEdges: false });
    expect(jump.stats.triangleCount).toBe(0);
    const mask = new Uint8Array([1, 0, 0, 1]);
    const masked = createMeshData(makeResult(2, 2, undefined, mask), { removeDepthEdges: false });
    expect(masked.stats.triangleCount).toBe(0);
    expect(masked.positions.length).toBe(0);
  });
});
