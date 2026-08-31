import { Matrix4, Vector3 } from 'three';
import { describe, expect, it } from 'vitest';
import {
  createCameraProjection,
  createProjectionMatrix,
  deriveDepthRange,
  imageFitViewport,
} from './cameraProjection';

describe('camera projection', () => {
  it('maps converted calibrated points to image NDC', () => {
    const intrinsics = new Float32Array([0.75, 0, 0.5, 0, 1, 0.5, 0, 0, 1]);
    const matrix = createProjectionMatrix(intrinsics, 640, 480, 0.1, 20);
    const depth = 2;
    const u = 0.25;
    const v = 0.25;
    const cvX = (u - intrinsics[2]!) * depth / intrinsics[0]!;
    const cvY = (v - intrinsics[5]!) * depth / intrinsics[4]!;
    const converted = new Vector3(cvX, -cvY, -depth).applyMatrix4(matrix);
    expect(converted.x).toBeCloseTo(2 * u - 1, 6);
    expect(converted.y).toBeCloseTo(1 - 2 * v, 6);
  });

  it('preserves an asymmetric principal point without a FOV approximation', () => {
    const intrinsics = new Float32Array([0.82, 0, 0.41, 0, 1.07, 0.63, 0, 0, 1]);
    const matrix = createProjectionMatrix(intrinsics, 1280, 720, 0.05, 50);
    const depth = 4.5;
    const u = 0.13;
    const v = 0.84;
    const converted = new Vector3(
      (u - intrinsics[2]!) * depth / intrinsics[0]!,
      -(v - intrinsics[5]!) * depth / intrinsics[4]!,
      -depth,
    ).applyMatrix4(matrix);
    expect(converted.x).toBeCloseTo(2 * u - 1, 6);
    expect(converted.y).toBeCloseTo(1 - 2 * v, 6);
  });

  it('keeps an inverse that round-trips a clip-space point', () => {
    const intrinsics = new Float32Array([0.75, 0, 0.5, 0, 1, 0.5, 0, 0, 1]);
    const projection = createCameraProjection(intrinsics, 640, 480, { near: 0.1, far: 20 });
    const point = new Vector3(0.2, -0.1, -2);
    const clip = point.clone().applyMatrix4(projection.projectionMatrix);
    const roundTrip = clip.applyMatrix4(projection.projectionMatrixInverse);
    expect(roundTrip.x).toBeCloseTo(point.x, 6);
    expect(roundTrip.y).toBeCloseTo(point.y, 6);
    expect(roundTrip.z).toBeCloseTo(point.z, 6);
  });

  it('derives robust clipping diagnostics and image-fit viewport', () => {
    const range = deriveDepthRange(new Float32Array([1, 2, 3, Number.NaN, -1]));
    expect(range.sampleCount).toBe(3);
    expect(range.near).toBeLessThan(1);
    expect(range.far).toBeGreaterThan(3);
    const viewport = imageFitViewport(1920, 1080, 1, 1);
    expect(viewport.width).toBe(1080);
    expect(viewport.height).toBe(1080);
    expect(viewport.x).toBe(420);
    expect(viewport.y).toBe(0);
  });

  it("does not depend on Three's perspective-camera convention", () => {
    const custom = createProjectionMatrix(new Float32Array([0.5, 0, 0.5, 0, 0.5, 0.5, 0, 0, 1]), 2, 2, { near: 1, far: 10 });
    const expected = new Matrix4().set(
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, -(11 / 9), -(20 / 9),
      0, 0, -1, 0,
    );
    expect(custom.elements).toEqual(expected.elements);
  });
});
