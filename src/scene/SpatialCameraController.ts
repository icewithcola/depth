import { Vector3 } from 'three';

export interface SpatialCameraViewport {
  width: number;
  height: number;
  x?: number;
  y?: number;
}

export interface SpatialCameraControllerOptions {
  /** Median scene depth used to turn normalized input into metric motion. */
  medianDepth?: number;
  sceneDepth?: number;
  /** Maximum translation as a fraction of median depth. */
  maxXRatio?: number;
  maxYRatio?: number;
  /** Absolute translation limits override the ratio-derived limits. */
  maxX?: number;
  maxY?: number;
  maxTranslationX?: number;
  maxTranslationY?: number;
  /** Exponential smoothing rate (per second). */
  smoothingLambda?: number;
  /** Alias for smoothingLambda. */
  lambda?: number;
  /** Time without input before the idle animation begins. */
  inactivitySeconds?: number;
  /** Approximate idle oscillation period. */
  idlePeriodSeconds?: number;
  /** Idle amplitudes as fractions of the corresponding translation limits. */
  idleAmplitudeX?: number;
  idleAmplitudeY?: number;
  reducedMotion?: boolean;
  viewport?: SpatialCameraViewport;
  /** Attach pointer listeners to this element and remove them on dispose. */
  element?: HTMLElement;
  autoAttach?: boolean;
}

export interface SpatialCameraState {
  x: number;
  y: number;
  z: number;
  targetX: number;
  targetY: number;
  normalizedTargetX: number;
  normalizedTargetY: number;
  idle: boolean;
  inactivitySeconds: number;
  alpha: number;
}

export interface SpatialResetOptions {
  smooth?: boolean;
}

export interface CameraLike {
  position: {
    x: number;
    y: number;
    z: number;
    set?: (x: number, y: number, z: number) => unknown;
  };
}

export interface PointerLike {
  clientX: number;
  clientY: number;
  pointerId?: number;
  pointerType?: string;
  preventDefault?: () => void;
}

function finiteOr(value: number | undefined, fallback: number): number {
  return value !== undefined && Number.isFinite(value) ? value : fallback;
}

function positiveOr(value: number | undefined, fallback: number): number {
  const result = finiteOr(value, fallback);
  return result > 0 ? result : fallback;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function prefersReducedMotion(): boolean {
  return typeof window !== 'undefined'
    && typeof window.matchMedia === 'function'
    && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

function looksLikeCamera(value: unknown): value is CameraLike {
  if (typeof value !== 'object' || value === null || !('position' in value)) return false;
  const position = (value as { position?: unknown }).position;
  if (typeof position !== 'object' || position === null) return false;
  const candidate = position as { x?: unknown; y?: unknown; z?: unknown };
  return typeof candidate.x === 'number' && typeof candidate.y === 'number' && typeof candidate.z === 'number';
}

function eventPoint(event: PointerLike, element: HTMLElement | undefined, viewport: SpatialCameraViewport): {
  x: number;
  y: number;
  width: number;
  height: number;
} {
  if (element !== undefined) {
    const rect = element.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      return { x: event.clientX - rect.left, y: event.clientY - rect.top, width: rect.width, height: rect.height };
    }
  }
  return {
    x: event.clientX - (viewport.x ?? 0),
    y: event.clientY - (viewport.y ?? 0),
    width: viewport.width,
    height: viewport.height,
  };
}

/**
 * Small translation-only camera controller for a reconstructed scene.
 *
 * Input is normalized to image viewport NDC (`x: -1..1` left/right,
 * `y: -1..1` bottom/top).  The controller never rotates a camera or a mesh;
 * its update method only applies a damped x/y translation to the camera.
 */
export class SpatialCameraController {
  readonly camera: CameraLike | undefined;

  private readonly element: HTMLElement | undefined;
  private readonly basePosition: Vector3 | undefined;
  private readonly maxXRatio: number;
  private readonly maxYRatio: number;
  private readonly hasExplicitMaxX: boolean;
  private readonly hasExplicitMaxY: boolean;
  private readonly smoothingLambda: number;
  private readonly inactivityLimit: number;
  private readonly idlePeriod: number;
  private readonly idleAmplitudeX: number;
  private readonly idleAmplitudeY: number;
  private viewport: SpatialCameraViewport;
  private medianDepthValue: number;
  private maxX: number;
  private maxY: number;
  private reducedMotion: boolean;
  private normalizedTargetX = 0;
  private normalizedTargetY = 0;
  private currentX = 0;
  private currentY = 0;
  private inactivity = 0;
  private idleTime = 0;
  private dragging = false;
  private dragPointerId: number | undefined;
  private dragAnchorX = 0;
  private dragAnchorY = 0;
  private dragOriginX = 0;
  private dragOriginY = 0;
  private disposed = false;

  private readonly onElementPointerMove = (event: PointerEvent): void => {
    this.onPointerMove(event);
  };

  private readonly onElementPointerDown = (event: PointerEvent): void => {
    this.onPointerDown(event);
  };

  private readonly onElementPointerUp = (event: PointerEvent): void => {
    this.onPointerUp(event);
  };

  private readonly onElementPointerCancel = (event: PointerEvent): void => {
    this.onPointerCancel(event);
  };

  constructor(camera: CameraLike, options?: SpatialCameraControllerOptions);
  constructor(options?: SpatialCameraControllerOptions);
  constructor(
    cameraOrOptions?: CameraLike | SpatialCameraControllerOptions,
    suppliedOptions: SpatialCameraControllerOptions = {},
  ) {
    const camera = looksLikeCamera(cameraOrOptions) ? cameraOrOptions : undefined;
    const options = looksLikeCamera(cameraOrOptions) ? suppliedOptions : (cameraOrOptions ?? {});
    this.camera = camera;
    this.element = options.element;
    this.basePosition = camera === undefined
      ? undefined
      : new Vector3(camera.position.x, camera.position.y, camera.position.z);

    this.medianDepthValue = positiveOr(options.medianDepth ?? options.sceneDepth, 1);
    this.maxXRatio = positiveOr(options.maxXRatio, 0.015);
    this.maxYRatio = positiveOr(options.maxYRatio, 0.008);
    const configuredMaxX = options.maxX ?? options.maxTranslationX;
    const configuredMaxY = options.maxY ?? options.maxTranslationY;
    this.hasExplicitMaxX = configuredMaxX !== undefined && Number.isFinite(configuredMaxX) && configuredMaxX > 0;
    this.hasExplicitMaxY = configuredMaxY !== undefined && Number.isFinite(configuredMaxY) && configuredMaxY > 0;
    this.maxX = positiveOr(configuredMaxX, this.medianDepthValue * this.maxXRatio);
    this.maxY = positiveOr(configuredMaxY, this.medianDepthValue * this.maxYRatio);
    this.smoothingLambda = positiveOr(options.smoothingLambda ?? options.lambda, 8);
    this.inactivityLimit = positiveOr(options.inactivitySeconds, 2);
    this.idlePeriod = positiveOr(options.idlePeriodSeconds, 8);
    this.idleAmplitudeX = Math.max(0, finiteOr(options.idleAmplitudeX, 0.4));
    this.idleAmplitudeY = Math.max(0, finiteOr(options.idleAmplitudeY, 0.4));
    this.reducedMotion = options.reducedMotion ?? prefersReducedMotion();
    this.viewport = {
      width: positiveOr(options.viewport?.width, 1),
      height: positiveOr(options.viewport?.height, 1),
      ...(options.viewport?.x === undefined ? {} : { x: options.viewport.x }),
      ...(options.viewport?.y === undefined ? {} : { y: options.viewport.y }),
    };

    if (this.element !== undefined && options.autoAttach !== false) this.attach();
  }

  /** Maximum translation in each axis, in scene units. */
  get limits(): { x: number; y: number } {
    return { x: this.maxX, y: this.maxY };
  }

  get maxTranslationX(): number {
    return this.maxX;
  }

  get maxTranslationY(): number {
    return this.maxY;
  }

  get medianDepth(): number {
    return this.medianDepthValue;
  }

  get target(): { x: number; y: number } {
    return {
      x: this.normalizedTargetX,
      y: this.normalizedTargetY,
    };
  }

  get translation(): { x: number; y: number; z: number } {
    return { x: this.currentX, y: this.currentY, z: 0 };
  }

  get isDisposed(): boolean {
    return this.disposed;
  }

  setViewport(viewport: SpatialCameraViewport): void {
    this.viewport = {
      width: positiveOr(viewport.width, this.viewport.width),
      height: positiveOr(viewport.height, this.viewport.height),
      ...(viewport.x === undefined ? {} : { x: viewport.x }),
      ...(viewport.y === undefined ? {} : { y: viewport.y }),
    };
  }

  setMedianDepth(depth: number): void {
    if (!Number.isFinite(depth) || depth <= 0) return;
    this.medianDepthValue = depth;
    // Explicit limits remain stable; ratio-derived limits track the new depth.
    if (!this.hasExplicitMaxX) this.maxX = this.maxXRatio * depth;
    if (!this.hasExplicitMaxY) this.maxY = this.maxYRatio * depth;
  }

  setReducedMotion(reducedMotion: boolean): void {
    this.reducedMotion = reducedMotion;
    if (reducedMotion) this.idleTime = 0;
  }

  /** Set a target in NDC-like normalized viewport coordinates. */
  setPointerTarget(normalizedX: number, normalizedY: number): void;
  setPointerTarget(target: { x: number; y: number }): void;
  setPointerTarget(
    normalizedXOrTarget: number | { x: number; y: number },
    normalizedY?: number,
  ): void {
    if (this.disposed) return;
    const normalizedX = typeof normalizedXOrTarget === 'number' ? normalizedXOrTarget : normalizedXOrTarget.x;
    const resolvedNormalizedY = typeof normalizedXOrTarget === 'number'
      ? normalizedY ?? 0
      : normalizedXOrTarget.y;
    this.normalizedTargetX = clamp(finiteOr(normalizedX, 0), -1, 1);
    this.normalizedTargetY = clamp(finiteOr(resolvedNormalizedY, 0), -1, 1);
    this.markInteraction();
  }

  setTarget(normalizedX: number, normalizedY: number): void;
  setTarget(target: { x: number; y: number }): void;
  setTarget(normalizedXOrTarget: number | { x: number; y: number }, normalizedY?: number): void {
    if (typeof normalizedXOrTarget === 'number') this.setPointerTarget(normalizedXOrTarget, normalizedY ?? 0);
    else this.setPointerTarget(normalizedXOrTarget);
  }

  /** Convert client coordinates to normalized viewport coordinates. */
  setPointerPosition(clientX: number, clientY: number, viewport?: SpatialCameraViewport): void {
    const point = eventPoint({ clientX, clientY }, this.element, viewport ?? this.viewport);
    if (point.width <= 0 || point.height <= 0) return;
    this.setPointerTarget(
      2 * (point.x / point.width) - 1,
      1 - 2 * (point.y / point.height),
    );
  }

  pointerMove(clientX: number, clientY: number, viewport?: SpatialCameraViewport): void {
    this.setPointerPosition(clientX, clientY, viewport);
  }

  onPointerMove(event: PointerLike): void;
  onPointerMove(clientX: number, clientY: number, viewport?: SpatialCameraViewport): void;
  onPointerMove(
    eventOrClientX: PointerLike | number,
    clientY?: number,
    viewport?: SpatialCameraViewport,
  ): void {
    if (typeof eventOrClientX === 'number') {
      this.setPointerPosition(eventOrClientX, clientY ?? 0, viewport);
      return;
    }
    const event = eventOrClientX;
    if (this.disposed) return;
    if (this.dragging && (this.dragPointerId === undefined || this.dragPointerId === event.pointerId)) {
      // Pass client coordinates through unchanged.  dragTo performs the
      // element-rectangle conversion itself; converting here first would
      // subtract an element's offset twice.
      this.dragTo(event.clientX, event.clientY, undefined, event.pointerId);
      event.preventDefault?.();
      return;
    }
    this.setPointerPosition(event.clientX, event.clientY);
  }

  handlePointerMove(event: PointerLike): void {
    this.onPointerMove(event);
  }

  /** Begin a relative drag (particularly useful for touch pointers). */
  beginDrag(clientX: number, clientY: number, viewport?: SpatialCameraViewport, pointerId?: number): void {
    const point = eventPoint({ clientX, clientY }, this.element, viewport ?? this.viewport);
    if (point.width <= 0 || point.height <= 0) return;
    this.dragging = true;
    this.dragPointerId = pointerId;
    this.dragAnchorX = point.x;
    this.dragAnchorY = point.y;
    this.dragOriginX = this.normalizedTargetX;
    this.dragOriginY = this.normalizedTargetY;
    this.markInteraction();
  }

  dragTo(clientX: number, clientY: number, viewport?: SpatialCameraViewport, pointerId?: number): void {
    if (this.disposed || !this.dragging || (pointerId !== undefined && this.dragPointerId !== pointerId)) return;
    const point = eventPoint({ clientX, clientY }, this.element, viewport ?? this.viewport);
    if (point.width <= 0 || point.height <= 0) return;
    this.normalizedTargetX = clamp(this.dragOriginX + 2 * (point.x - this.dragAnchorX) / point.width, -1, 1);
    // Client Y grows down while normalized Y grows up.
    this.normalizedTargetY = clamp(this.dragOriginY - 2 * (point.y - this.dragAnchorY) / point.height, -1, 1);
    this.markInteraction();
  }

  endDrag(pointerId?: number): void {
    if (pointerId !== undefined && this.dragPointerId !== undefined && pointerId !== this.dragPointerId) return;
    this.dragging = false;
    this.dragPointerId = undefined;
  }

  beginTouchDrag(clientX: number, clientY: number, viewport?: SpatialCameraViewport): void {
    this.beginDrag(clientX, clientY, viewport);
  }

  updateTouchDrag(clientX: number, clientY: number, viewport?: SpatialCameraViewport): void {
    this.dragTo(clientX, clientY, viewport);
  }

  endTouchDrag(): void {
    this.endDrag();
  }

  onPointerDown(event: PointerLike): void;
  onPointerDown(clientX: number, clientY: number, viewport?: SpatialCameraViewport): void;
  onPointerDown(
    eventOrClientX: PointerLike | number,
    clientY?: number,
    viewport?: SpatialCameraViewport,
  ): void {
    if (typeof eventOrClientX === 'number') {
      this.beginDrag(eventOrClientX, clientY ?? 0, viewport);
      return;
    }
    const event = eventOrClientX;
    if (this.disposed) return;
    this.beginDrag(event.clientX, event.clientY, undefined, event.pointerId);
    event.preventDefault?.();
  }

  handlePointerDown(event: PointerLike): void {
    this.onPointerDown(event);
  }

  onPointerUp(event: PointerLike): void;
  onPointerUp(pointerId?: number): void;
  onPointerUp(eventOrPointerId?: PointerLike | number): void {
    this.endDrag(typeof eventOrPointerId === 'number' ? eventOrPointerId : eventOrPointerId?.pointerId);
  }

  handlePointerUp(event: PointerLike): void {
    this.onPointerUp(event);
  }

  onPointerCancel(event: PointerLike): void;
  onPointerCancel(pointerId?: number): void;
  onPointerCancel(eventOrPointerId?: PointerLike | number): void {
    this.endDrag(typeof eventOrPointerId === 'number' ? eventOrPointerId : eventOrPointerId?.pointerId);
  }

  handlePointerCancel(event: PointerLike): void {
    this.onPointerCancel(event);
  }

  /** Reset target and smoothly return to the original camera position. */
  reset(smoothOrOptions: boolean | SpatialResetOptions = true): void {
    const smooth = typeof smoothOrOptions === 'boolean' ? smoothOrOptions : smoothOrOptions.smooth ?? true;
    this.normalizedTargetX = 0;
    this.normalizedTargetY = 0;
    this.inactivity = 0;
    this.idleTime = 0;
    this.dragging = false;
    this.dragPointerId = undefined;
    if (!smooth) {
      this.currentX = 0;
      this.currentY = 0;
      this.applyCameraPosition();
    }
  }

  /** Current state after one frame's damped update. `deltaSeconds` is seconds. */
  update(deltaSeconds: number, medianDepth?: number): SpatialCameraState {
    if (this.disposed) return this.state(0, false);
    if (medianDepth !== undefined) this.setMedianDepth(medianDepth);
    const dt = Number.isFinite(deltaSeconds) && deltaSeconds > 0 ? deltaSeconds : 0;
    const previousInactivity = this.inactivity;
    this.inactivity += dt;
    const idle = !this.reducedMotion && this.inactivity >= this.inactivityLimit;
    if (idle) {
      // Start the oscillator at phase zero when the inactivity threshold is
      // crossed, even if a coarse RAF interval jumps over that threshold.
      this.idleTime += Math.max(0, this.inactivity - Math.max(previousInactivity, this.inactivityLimit));
    }
    else this.idleTime = 0;

    let desiredX = this.normalizedTargetX * this.maxX;
    let desiredY = this.normalizedTargetY * this.maxY;
    if (idle) {
      const phase = (this.idleTime / this.idlePeriod) * 2 * Math.PI;
      desiredX = Math.sin(phase) * this.maxX * this.idleAmplitudeX;
      desiredY = Math.cos(phase) * this.maxY * this.idleAmplitudeY;
    }

    const alpha = 1 - Math.exp(-this.smoothingLambda * dt);
    this.currentX += (desiredX - this.currentX) * alpha;
    this.currentY += (desiredY - this.currentY) * alpha;
    this.applyCameraPosition();
    return this.state(alpha, idle);
  }

  getState(): SpatialCameraState {
    return this.state(0, !this.reducedMotion && this.inactivity >= this.inactivityLimit);
  }

  attach(): void {
    if (this.element === undefined || this.disposed) return;
    this.element.addEventListener('pointermove', this.onElementPointerMove);
    this.element.addEventListener('pointerdown', this.onElementPointerDown);
    this.element.addEventListener('pointerup', this.onElementPointerUp);
    this.element.addEventListener('pointercancel', this.onElementPointerCancel);
  }

  dispose(): void {
    if (this.disposed) return;
    if (this.element !== undefined) {
      this.element.removeEventListener('pointermove', this.onElementPointerMove);
      this.element.removeEventListener('pointerdown', this.onElementPointerDown);
      this.element.removeEventListener('pointerup', this.onElementPointerUp);
      this.element.removeEventListener('pointercancel', this.onElementPointerCancel);
    }
    this.disposed = true;
    this.dragging = false;
  }

  private markInteraction(): void {
    this.inactivity = 0;
    this.idleTime = 0;
  }

  private state(alpha: number, idle: boolean): SpatialCameraState {
    return {
      x: this.currentX,
      y: this.currentY,
      z: 0,
      targetX: this.normalizedTargetX * this.maxX,
      targetY: this.normalizedTargetY * this.maxY,
      normalizedTargetX: this.normalizedTargetX,
      normalizedTargetY: this.normalizedTargetY,
      idle,
      inactivitySeconds: this.inactivity,
      alpha,
    };
  }

  private applyCameraPosition(): void {
    const camera = this.camera;
    const base = this.basePosition;
    if (camera === undefined || base === undefined) return;
    const x = base.x + this.currentX;
    const y = base.y + this.currentY;
    const z = base.z;
    if (camera.position.set !== undefined) camera.position.set(x, y, z);
    else {
      camera.position.x = x;
      camera.position.y = y;
      camera.position.z = z;
    }
  }
}

export default SpatialCameraController;
