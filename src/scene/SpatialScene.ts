import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import type { MoGeResult } from '../moge/types';
import { SpatialCameraController } from './SpatialCameraController';
import {
  createCameraProjection,
  imageFitViewport,
  type CameraProjection,
  type ImageFitViewport,
} from './cameraProjection';
import { createMeshData } from './createMesh';

export type ViewMode = 'spatial' | 'original' | 'depth' | 'normal' | 'wireframe' | 'debug3d';

export interface SpatialSceneDiagnostics {
  vertexCount: number;
  triangleCount: number;
  meshBuildMs: number;
  fovXDegrees: number;
  fovYDegrees: number;
}

interface SceneResources {
  geometry: THREE.BufferGeometry;
  originalGeometry: THREE.BufferGeometry;
  photoTexture: THREE.Texture;
  depthTexture: THREE.Texture;
  normalTexture: THREE.Texture;
  photoMaterial: THREE.MeshBasicMaterial;
  depthMaterial: THREE.MeshBasicMaterial;
  normalMaterial: THREE.MeshBasicMaterial;
  wireframeMaterial: THREE.MeshBasicMaterial;
  originalMaterial: THREE.MeshBasicMaterial;
  mesh: THREE.Mesh;
  originalMesh: THREE.Mesh;
}

function createWebGl2Renderer(canvas: HTMLCanvasElement): THREE.WebGLRenderer {
  const context = canvas.getContext('webgl2', {
    alpha: false,
    antialias: true,
    depth: true,
    powerPreference: 'high-performance',
  });
  if (context === null) {
    throw new Error('WebGL2 is required to render the Spatial Scene.');
  }
  const renderer = new THREE.WebGLRenderer({ canvas, context, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.autoClear = false;
  return renderer;
}

function bitmapToCanvas(bitmap: ImageBitmap): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const context = canvas.getContext('2d');
  if (context === null) {
    bitmap.close();
    throw new Error('Could not create a texture canvas.');
  }
  context.drawImage(bitmap, 0, 0);
  bitmap.close();
  return canvas;
}

function canvasTexture(canvas: HTMLCanvasElement, colorSpace: THREE.ColorSpace): THREE.CanvasTexture {
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = colorSpace;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.generateMipmaps = true;
  texture.needsUpdate = true;
  return texture;
}

function percentile(sorted: readonly number[], p: number): number {
  if (sorted.length === 0) return 0;
  const position = Math.max(0, Math.min(1, p)) * (sorted.length - 1);
  const lower = sorted[Math.floor(position)] ?? 0;
  const upper = sorted[Math.ceil(position)] ?? lower;
  return lower + (upper - lower) * (position - Math.floor(position));
}

function validDepths(result: MoGeResult): number[] {
  const values: number[] = [];
  for (let index = 0; index < result.depth.length; index += 1) {
    const value = result.depth[index];
    if (result.mask[index] === 1 && value !== undefined && Number.isFinite(value) && value > 0) {
      values.push(value);
    }
  }
  values.sort((a, b) => a - b);
  return values;
}

function createDebugCanvas(result: MoGeResult, kind: 'depth' | 'normal'): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = result.width;
  canvas.height = result.height;
  const context = canvas.getContext('2d');
  if (context === null) throw new Error('Could not create a debug-view canvas.');
  const image = context.createImageData(result.width, result.height);

  if (kind === 'depth') {
    const values = validDepths(result);
    const near = percentile(values, 0.02);
    const far = percentile(values, 0.98);
    const span = Math.max(Number.EPSILON, far - near);
    for (let index = 0; index < result.depth.length; index += 1) {
      const offset = index * 4;
      const depth = result.depth[index];
      const valid = result.mask[index] === 1 && depth !== undefined && Number.isFinite(depth) && depth > 0;
      const normalized = valid ? 1 - Math.max(0, Math.min(1, (depth - near) / span)) : 0;
      // Near is warm/bright, far is cool/dark, making direction immediately legible.
      image.data[offset] = Math.round(32 + 223 * normalized);
      image.data[offset + 1] = Math.round(52 + 150 * normalized);
      image.data[offset + 2] = Math.round(92 + 80 * (1 - normalized));
      image.data[offset + 3] = valid ? 255 : 0;
    }
  } else {
    for (let index = 0; index < result.width * result.height; index += 1) {
      const offset = index * 4;
      const normalOffset = index * 3;
      const valid = result.mask[index] === 1 && result.normals !== undefined;
      image.data[offset] = valid ? Math.round(127.5 * ((result.normals?.[normalOffset] ?? 0) + 1)) : 0;
      image.data[offset + 1] = valid ? Math.round(127.5 * ((result.normals?.[normalOffset + 1] ?? 0) + 1)) : 0;
      image.data[offset + 2] = valid ? Math.round(127.5 * ((result.normals?.[normalOffset + 2] ?? 0) + 1)) : 0;
      image.data[offset + 3] = valid ? 255 : 0;
    }
  }
  context.putImageData(image, 0, 0);
  return canvas;
}

/** A full, uncut image plane that projects exactly onto the calibrated frame. */
function createOriginalGeometry(result: MoGeResult, depth: number): THREE.BufferGeometry {
  const fx = result.intrinsics[0];
  const cx = result.intrinsics[2];
  const fy = result.intrinsics[4];
  const cy = result.intrinsics[5];
  if (fx === undefined || fy === undefined || cx === undefined || cy === undefined) {
    throw new Error('MoGe intrinsics are incomplete.');
  }
  const point = (u: number, v: number): [number, number, number] => [
    (u - cx) / fx * depth,
    -(v - cy) / fy * depth,
    -depth,
  ];
  const positions = new Float32Array([
    ...point(0, 0), ...point(0, 1), ...point(1, 0), ...point(1, 1),
  ]);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([
    0, 1, 0, 0, 1, 1, 1, 0,
  ]), 2));
  geometry.setIndex([0, 1, 2, 2, 1, 3]);
  return geometry;
}

function disposeMaterial(material: THREE.Material): void {
  material.dispose();
}

/**
 * Owns the complete WebGL scene and exactly one animation loop. MoGe remains
 * in OpenCV coordinates until createMeshData converts positions at this boundary.
 */
export class SpatialScene {
  private readonly canvas: HTMLCanvasElement;
  private readonly viewportElement: HTMLElement;
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();
  private readonly spatialCamera = new THREE.PerspectiveCamera();
  private readonly debugCamera = new THREE.PerspectiveCamera(48, 1, 0.01, 100);
  private readonly resizeObserver: ResizeObserver;
  private readonly reducedMotionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
  private resources: SceneResources | undefined;
  private projection: CameraProjection | undefined;
  private controller: SpatialCameraController | undefined;
  private debugControls: OrbitControls | undefined;
  private viewMode: ViewMode = 'spatial';
  private autoMotion = true;
  private imageViewport: ImageFitViewport | undefined;
  private frameId = 0;
  private previousFrameTime = performance.now();
  private disposed = false;

  private localPointer(event: PointerEvent): {
    clientX: number;
    clientY: number;
    pointerId: number;
    pointerType: string;
    preventDefault: () => void;
  } {
    const rect = this.viewportElement.getBoundingClientRect();
    return {
      clientX: event.clientX - rect.left,
      clientY: event.clientY - rect.top,
      pointerId: event.pointerId,
      pointerType: event.pointerType,
      preventDefault: () => event.preventDefault(),
    };
  }

  private readonly onPointerMove = (event: PointerEvent): void => {
    if (this.viewMode !== 'spatial') return;
    this.controller?.onPointerMove(this.localPointer(event));
  };
  private readonly onPointerDown = (event: PointerEvent): void => {
    if (this.viewMode !== 'spatial') return;
    this.viewportElement.setPointerCapture(event.pointerId);
    this.controller?.onPointerDown(this.localPointer(event));
  };
  private readonly onPointerUp = (event: PointerEvent): void => {
    this.controller?.onPointerUp(this.localPointer(event));
    if (this.viewportElement.hasPointerCapture(event.pointerId)) {
      this.viewportElement.releasePointerCapture(event.pointerId);
    }
  };
  private readonly onPointerCancel = (event: PointerEvent): void => {
    this.controller?.onPointerCancel(this.localPointer(event));
  };
  private readonly onPointerLeave = (event: PointerEvent): void => {
    if (event.pointerType === 'mouse') this.controller?.reset(true);
  };

  public constructor(canvas: HTMLCanvasElement, viewport: HTMLElement) {
    this.canvas = canvas;
    this.viewportElement = viewport;
    this.renderer = createWebGl2Renderer(canvas);
    this.scene.background = new THREE.Color(0x080b10);
    this.spatialCamera.matrixAutoUpdate = true;

    viewport.addEventListener('pointermove', this.onPointerMove);
    viewport.addEventListener('pointerdown', this.onPointerDown);
    viewport.addEventListener('pointerup', this.onPointerUp);
    viewport.addEventListener('pointercancel', this.onPointerCancel);
    viewport.addEventListener('pointerleave', this.onPointerLeave);
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(viewport);
    this.resize();
    this.frameId = requestAnimationFrame(this.frame);
  }

  public setScene(result: MoGeResult, textureImage: ImageBitmap): SpatialSceneDiagnostics {
    if (this.disposed) {
      textureImage.close();
      throw new Error('Cannot set a scene after SpatialScene.dispose().');
    }
    const meshData = createMeshData(result, { targetLongSide: 360, depthRtol: 0.04 });
    if (meshData.triangleCount === 0) {
      textureImage.close();
      throw new Error('MoGe produced no valid connected surface for this image.');
    }
    const depths = validDepths(result);
    if (depths.length === 0) throw new Error('MoGe returned no positive finite depth samples.');
    const medianDepth = percentile(depths, 0.5);
    const nextProjection = createCameraProjection(result, {
      viewportWidth: Math.max(1, this.viewportElement.clientWidth),
      viewportHeight: Math.max(1, this.viewportElement.clientHeight),
    });
    const stagedDisposables: Array<{ dispose(): void }> = [];
    let nextResources: SceneResources;
    try {
      const geometry = new THREE.BufferGeometry();
      stagedDisposables.push(geometry);
      geometry.setAttribute('position', new THREE.BufferAttribute(meshData.positions, 3));
      geometry.setAttribute('uv', new THREE.BufferAttribute(meshData.uv, 2));
      if (meshData.normals !== undefined) {
        geometry.setAttribute('normal', new THREE.BufferAttribute(meshData.normals, 3));
      }
      geometry.setIndex(new THREE.BufferAttribute(meshData.indices, 1));
      geometry.computeBoundingSphere();

      const sourceCanvas = bitmapToCanvas(textureImage);
      const photoTexture = canvasTexture(sourceCanvas, THREE.SRGBColorSpace);
      stagedDisposables.push(photoTexture);
      const depthTexture = canvasTexture(createDebugCanvas(result, 'depth'), THREE.SRGBColorSpace);
      stagedDisposables.push(depthTexture);
      const normalTexture = canvasTexture(createDebugCanvas(result, 'normal'), THREE.SRGBColorSpace);
      stagedDisposables.push(normalTexture);
      const materialOptions = { side: THREE.FrontSide, toneMapped: false } as const;
      const photoMaterial = new THREE.MeshBasicMaterial({ map: photoTexture, ...materialOptions });
      const depthMaterial = new THREE.MeshBasicMaterial({ map: depthTexture, transparent: true, ...materialOptions });
      const normalMaterial = new THREE.MeshBasicMaterial({ map: normalTexture, transparent: true, ...materialOptions });
      const wireframeMaterial = new THREE.MeshBasicMaterial({
        color: 0x77e6d2,
        wireframe: true,
        ...materialOptions,
      });
      const originalMaterial = new THREE.MeshBasicMaterial({ map: photoTexture, ...materialOptions });
      stagedDisposables.push(
        photoMaterial,
        depthMaterial,
        normalMaterial,
        wireframeMaterial,
        originalMaterial,
      );
      const originalGeometry = createOriginalGeometry(result, medianDepth);
      stagedDisposables.push(originalGeometry);
      const mesh = new THREE.Mesh(geometry, photoMaterial);
      const originalMesh = new THREE.Mesh(originalGeometry, originalMaterial);
      nextResources = {
        geometry,
        originalGeometry,
        photoTexture,
        depthTexture,
        normalTexture,
        photoMaterial,
        depthMaterial,
        normalMaterial,
        wireframeMaterial,
        originalMaterial,
        mesh,
        originalMesh,
      };
    } catch (error) {
      for (const disposable of stagedDisposables.reverse()) disposable.dispose();
      throw error;
    }

    // Commit only after the complete replacement has been constructed. A
    // decode/allocation error therefore leaves the previous photo intact.
    this.disposeResources();
    this.resources = nextResources;
    this.scene.add(nextResources.mesh, nextResources.originalMesh);
    this.projection = nextProjection;
    this.applySpatialProjection();
    this.controller = new SpatialCameraController(this.spatialCamera, {
      medianDepth,
      reducedMotion: !this.autoMotion || this.reducedMotionQuery.matches,
      viewport: this.imageViewport ?? { width: 1, height: 1 },
    });

    this.configureDebugCamera();
    this.setViewMode('spatial');

    return {
      vertexCount: meshData.vertexCount,
      triangleCount: meshData.triangleCount,
      meshBuildMs: meshData.buildMs,
      fovXDegrees: this.projection.fovXDegrees,
      fovYDegrees: this.projection.fovYDegrees,
    };
  }

  public setViewMode(mode: ViewMode): void {
    this.viewMode = mode;
    this.leaveDebugMode();
    const resources = this.resources;
    if (resources === undefined) return;
    resources.mesh.visible = mode !== 'original';
    resources.originalMesh.visible = mode === 'original';
    switch (mode) {
      case 'depth': resources.mesh.material = resources.depthMaterial; break;
      case 'normal': resources.mesh.material = resources.normalMaterial; break;
      case 'wireframe': resources.mesh.material = resources.wireframeMaterial; break;
      default: resources.mesh.material = resources.photoMaterial; break;
    }
    if (mode === 'debug3d') this.enterDebugMode();
    if (mode !== 'spatial') {
      this.controller?.reset(false);
      this.spatialCamera.position.set(0, 0, 0);
    }
  }

  public resetView(): void {
    if (this.viewMode === 'debug3d') {
      this.configureDebugCamera();
      this.debugControls?.update();
      return;
    }
    this.controller?.reset(true);
  }

  public setAutoMotion(enabled: boolean): void {
    this.autoMotion = enabled;
    this.controller?.setReducedMotion(!enabled || this.reducedMotionQuery.matches);
  }

  public dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    cancelAnimationFrame(this.frameId);
    this.resizeObserver.disconnect();
    this.viewportElement.removeEventListener('pointermove', this.onPointerMove);
    this.viewportElement.removeEventListener('pointerdown', this.onPointerDown);
    this.viewportElement.removeEventListener('pointerup', this.onPointerUp);
    this.viewportElement.removeEventListener('pointercancel', this.onPointerCancel);
    this.viewportElement.removeEventListener('pointerleave', this.onPointerLeave);
    this.disposeResources();
    this.renderer.dispose();
  }

  private readonly frame = (time: number): void => {
    if (this.disposed) return;
    const dt = Math.min(0.1, Math.max(0, (time - this.previousFrameTime) / 1000));
    this.previousFrameTime = time;
    if (this.viewMode === 'spatial') this.controller?.update(dt);
    if (this.viewMode === 'debug3d') this.debugControls?.update();
    this.render();
    this.frameId = requestAnimationFrame(this.frame);
  };

  private resize(): void {
    const width = Math.max(1, this.viewportElement.clientWidth);
    const height = Math.max(1, this.viewportElement.clientHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(width, height, false);
    if (this.projection !== undefined) {
      this.imageViewport = imageFitViewport(
        width,
        height,
        this.projection.sourceWidth,
        this.projection.sourceHeight,
      );
      this.controller?.setViewport(this.imageViewport);
    }
  }

  private render(): void {
    const width = Math.max(1, this.viewportElement.clientWidth);
    const height = Math.max(1, this.viewportElement.clientHeight);
    this.renderer.setScissorTest(false);
    this.renderer.setViewport(0, 0, width, height);
    this.renderer.clear(true, true, true);
    if (this.resources === undefined || this.imageViewport === undefined) return;
    const viewport = this.imageViewport;
    const bottom = height - viewport.y - viewport.height;
    this.renderer.setViewport(viewport.x, bottom, viewport.width, viewport.height);
    this.renderer.setScissor(viewport.x, bottom, viewport.width, viewport.height);
    this.renderer.setScissorTest(true);
    this.renderer.render(this.scene, this.viewMode === 'debug3d' ? this.debugCamera : this.spatialCamera);
    this.renderer.setScissorTest(false);
  }

  private applySpatialProjection(): void {
    const projection = this.projection;
    if (projection === undefined) return;
    this.spatialCamera.near = projection.near;
    this.spatialCamera.far = projection.far;
    this.spatialCamera.projectionMatrix.copy(projection.projectionMatrix);
    this.spatialCamera.projectionMatrixInverse.copy(projection.projectionMatrixInverse);
    this.spatialCamera.position.set(0, 0, 0);
    this.spatialCamera.quaternion.identity();
    this.resize();
  }

  private configureDebugCamera(): void {
    const sphere = this.resources?.geometry.boundingSphere;
    if (sphere === null || sphere === undefined) return;
    const radius = Math.max(sphere.radius, 0.1);
    this.debugCamera.aspect = this.projection?.aspect ?? 1;
    this.debugCamera.near = Math.max(0.001, radius / 100);
    this.debugCamera.far = Math.max(this.debugCamera.near * 10, radius * 20);
    this.debugCamera.position.set(
      sphere.center.x + radius * 1.25,
      sphere.center.y + radius * 0.55,
      sphere.center.z + radius * 1.8,
    );
    this.debugCamera.lookAt(sphere.center);
    this.debugCamera.updateProjectionMatrix();
  }

  private enterDebugMode(): void {
    this.configureDebugCamera();
    const target = this.resources?.geometry.boundingSphere?.center ?? new THREE.Vector3();
    this.debugControls = new OrbitControls(this.debugCamera, this.canvas);
    this.debugControls.enableDamping = true;
    this.debugControls.target.copy(target);
    this.debugControls.update();
  }

  private leaveDebugMode(): void {
    this.debugControls?.dispose();
    this.debugControls = undefined;
  }

  private disposeResources(): void {
    this.leaveDebugMode();
    this.controller?.dispose();
    this.controller = undefined;
    const resources = this.resources;
    if (resources === undefined) return;
    this.scene.remove(resources.mesh, resources.originalMesh);
    resources.geometry.dispose();
    resources.originalGeometry.dispose();
    resources.photoTexture.dispose();
    resources.depthTexture.dispose();
    resources.normalTexture.dispose();
    disposeMaterial(resources.photoMaterial);
    disposeMaterial(resources.depthMaterial);
    disposeMaterial(resources.normalMaterial);
    disposeMaterial(resources.wireframeMaterial);
    disposeMaterial(resources.originalMaterial);
    this.resources = undefined;
  }
}
