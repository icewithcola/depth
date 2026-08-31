import { PerspectiveCamera } from 'three';
import { describe, expect, it } from 'vitest';
import { SpatialCameraController } from './SpatialCameraController';

describe('SpatialCameraController', () => {
  it('maps normalized input to metric translation with exponential smoothing', () => {
    const camera = new PerspectiveCamera();
    const controller = new SpatialCameraController(camera, { medianDepth: 10, smoothingLambda: 2 });
    controller.setPointerTarget(1, -1);
    const state = controller.update(0.5);
    const alpha = 1 - Math.exp(-1);
    expect(state.x).toBeCloseTo(0.15 * alpha, 8);
    expect(state.y).toBeCloseTo(-0.08 * alpha, 8);
    expect(camera.rotation.x).toBe(0);
    expect(camera.rotation.y).toBe(0);
    expect(camera.rotation.z).toBe(0);
  });

  it('supports relative touch drags and smooth reset', () => {
    const controller = new SpatialCameraController({ medianDepth: 100, viewport: { width: 100, height: 100 } });
    controller.beginTouchDrag(50, 50);
    controller.updateTouchDrag(100, 0);
    expect(controller.target.x).toBeCloseTo(1);
    expect(controller.target.y).toBeCloseTo(1);
    controller.endTouchDrag();
    controller.update(1);
    controller.reset();
    expect(controller.target).toEqual({ x: 0, y: 0 });
    expect(controller.translation.x).toBeGreaterThan(0);
    controller.update(2);
    expect(controller.translation.x).toBeCloseTo(0, 4);
  });

  it('starts an approximately eight-second idle sinusoid and honours reduced motion', () => {
    const controller = new SpatialCameraController({ medianDepth: 10, smoothingLambda: 100 });
    const idleStart = controller.update(2);
    expect(idleStart.idle).toBe(true);
    const first = controller.update(1).x;
    const halfway = controller.update(4).x;
    expect(Math.abs(first)).toBeGreaterThan(0);
    expect(Math.sign(first)).not.toBe(Math.sign(halfway));

    const reduced = new SpatialCameraController({ medianDepth: 10, reducedMotion: true });
    expect(reduced.update(3).idle).toBe(false);
    expect(reduced.translation.x).toBe(0);
    expect(reduced.translation.y).toBe(0);
  });
});
