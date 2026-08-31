import { MoGeInference } from './moge/inference';
import type { MoGeInferenceResult, StageTimings } from './moge/types';
import { SpatialScene, type ViewMode } from './scene/SpatialScene';

const TEXTURE_MAX_SIDE = 2048;
const SUPPORTED_IMAGE_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp']);
const IMAGE_ORIENTATION_OPTIONS: ImageBitmapOptions = { imageOrientation: 'from-image' };

export type AppState =
  | 'BOOT'
  | 'MODEL_LOADING'
  | 'READY'
  | 'INFERENCING'
  | 'SCENE_READY'
  | 'ERROR';

interface AppElements {
  root: HTMLElement;
  fileInput: HTMLInputElement;
  viewMode: HTMLSelectElement;
  resetView: HTMLButtonElement;
  autoMotion: HTMLInputElement;
  viewport: HTMLElement;
  canvas: HTMLCanvasElement;
  dropZone: HTMLElement;
  sceneGuide: HTMLElement;
  statusDot: HTMLElement;
  statusState: HTMLElement;
  statusMessage: HTMLElement;
  metricSummary: HTMLElement;
}

interface PendingSelection {
  id: number;
  file: File;
}

interface SceneStats {
  vertexCount: number;
  triangleCount: number;
  meshBuildMs: number;
  fovXDegrees: number;
  fovYDegrees: number;
}

const STATE_LABELS: Record<AppState, string> = {
  BOOT: 'Boot',
  MODEL_LOADING: 'Model loading',
  READY: 'Ready',
  INFERENCING: 'Estimating',
  SCENE_READY: 'Scene ready',
  ERROR: 'Error',
};

const LOADING_STATES = new Set<AppState>(['BOOT', 'MODEL_LOADING', 'INFERENCING']);

function requiredElement<T extends HTMLElement>(documentRef: Document, id: string): T {
  const element = documentRef.getElementById(id);
  if (!element) {
    throw new Error(`Depth Studio is missing required element '#${id}'.`);
  }
  return element as T;
}

function collectElements(documentRef: Document): AppElements {
  return {
    root: requiredElement<HTMLElement>(documentRef, 'app'),
    fileInput: requiredElement<HTMLInputElement>(documentRef, 'image-input'),
    viewMode: requiredElement<HTMLSelectElement>(documentRef, 'view-mode'),
    resetView: requiredElement<HTMLButtonElement>(documentRef, 'reset-view'),
    autoMotion: requiredElement<HTMLInputElement>(documentRef, 'auto-motion'),
    viewport: requiredElement<HTMLElement>(documentRef, 'viewport'),
    canvas: requiredElement<HTMLCanvasElement>(documentRef, 'scene-canvas'),
    dropZone: requiredElement<HTMLElement>(documentRef, 'drop-zone'),
    sceneGuide: requiredElement<HTMLElement>(documentRef, 'scene-guide'),
    statusDot: requiredElement<HTMLElement>(documentRef, 'status-dot'),
    statusState: requiredElement<HTMLElement>(documentRef, 'status-state'),
    statusMessage: requiredElement<HTMLElement>(documentRef, 'status-message'),
    metricSummary: requiredElement<HTMLElement>(documentRef, 'metric-summary'),
  };
}

function errorMessage(error: unknown): string {
  if (error instanceof Error && error.message.length > 0) {
    return error.message;
  }
  return String(error);
}

function reportError(scope: string, error: unknown): void {
  console.error(`[Depth Studio] ${scope}: ${errorMessage(error)}`, error);
}

function closeBitmap(bitmap: ImageBitmap | undefined): void {
  if (!bitmap) {
    return;
  }

  const close = (bitmap as unknown as { close?: unknown }).close;
  if (typeof close !== 'function') {
    return;
  }
  try {
    close.call(bitmap);
  } catch (error) {
    // A detached bitmap is already released. Keep the original operation
    // error visible instead of turning cleanup into a second user error.
    console.warn('[Depth Studio] Could not close an image bitmap.', error);
  }
}

function imageDimensions(image: ImageBitmap): { width: number; height: number } {
  if (
    !Number.isInteger(image.width) ||
    !Number.isInteger(image.height) ||
    image.width <= 0 ||
    image.height <= 0
  ) {
    throw new Error(
      `Decoded image has invalid dimensions (${String(image.width)}×${String(image.height)}).`,
    );
  }
  return { width: image.width, height: image.height };
}

function cappedDimensions(width: number, height: number, maxSide: number): {
  width: number;
  height: number;
} {
  const scale = Math.min(1, maxSide / Math.max(width, height));
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
  };
}

/** Decode a display texture independently from the model's 800px bitmap. */
async function createTextureBitmap(file: File): Promise<ImageBitmap> {
  if (typeof createImageBitmap !== 'function') {
    throw new Error('Image decoding is unavailable because createImageBitmap is not supported.');
  }

  let decoded: ImageBitmap | undefined;
  let resized: ImageBitmap | undefined;
  try {
    // Decode once with EXIF orientation applied. The second bitmap is a
    // separate texture resource, so inference can own and release its own
    // 800px preprocessing bitmap independently.
    decoded = await createImageBitmap(file, IMAGE_ORIENTATION_OPTIONS);
    const sourceSize = imageDimensions(decoded);
    const targetSize = cappedDimensions(sourceSize.width, sourceSize.height, TEXTURE_MAX_SIDE);
    if (targetSize.width === sourceSize.width && targetSize.height === sourceSize.height) {
      const result = decoded;
      decoded = undefined;
      return result;
    }

    resized = await createImageBitmap(decoded, {
      resizeWidth: targetSize.width,
      resizeHeight: targetSize.height,
      resizeQuality: 'high',
    });
    imageDimensions(resized);
    closeBitmap(decoded);
    decoded = undefined;
    const result = resized;
    resized = undefined;
    return result;
  } catch (error) {
    closeBitmap(resized);
    closeBitmap(decoded);
    const detail = errorMessage(error);
    throw new Error(`Could not decode or prepare the image texture (${detail}).`, {
      cause: error,
    });
  }
}

function isOutOfMemory(error: unknown): boolean {
  return /out\s*of\s*memory|outofmemory|memory allocation|allocation failed|quota|oom/i.test(
    errorMessage(error),
  );
}

function isWebGpuError(error: unknown): boolean {
  return /webgpu|gpu adapter|secure.?context|adapter discovery|gpu access/i.test(
    errorMessage(error),
  );
}

function isMissingModel(error: unknown): boolean {
  return /404|not found|failed to fetch|network|model|onnx|load/i.test(errorMessage(error));
}

function formatModelError(error: unknown): string {
  if (isOutOfMemory(error)) {
    return 'The browser ran out of memory while loading the depth model. Close other tabs and reload.';
  }
  if (isWebGpuError(error)) {
    return 'WebGPU is unavailable. Use a current Chrome or Edge window with GPU access enabled.';
  }
  if (isMissingModel(error)) {
    return 'The depth model is missing or could not load. Check the model file and reload.';
  }
  return 'The depth model could not load. Reload the page and try again.';
}

function formatProcessingError(error: unknown, textureFailed: boolean): string {
  if (isOutOfMemory(error)) {
    return 'The browser ran out of memory. Try a smaller image or close other tabs.';
  }
  if (textureFailed || /decode|bitmap|image texture|image data/i.test(errorMessage(error))) {
    return 'That image could not be decoded. Choose a valid JPEG, PNG, or WebP image.';
  }
  if (isWebGpuError(error)) {
    return 'WebGPU stopped responding while estimating depth. Reload and try again.';
  }
  return 'Depth estimation failed. Try another image or a smaller file.';
}

function validateFile(file: File): string | undefined {
  const mime = file.type.trim().toLowerCase();
  const supportedExtension = /\.(?:jpe?g|png|webp)$/i.test(file.name);
  if (!SUPPORTED_IMAGE_TYPES.has(mime) && !(mime === '' && supportedExtension)) {
    return 'Unsupported image type. Choose a JPEG, PNG, or WebP image.';
  }
  if (file.size <= 0) {
    return 'That image file is empty. Choose a different image.';
  }
  return undefined;
}

function formatMilliseconds(value: number): string {
  if (!Number.isFinite(value) || value < 0) {
    return '—';
  }
  return `${value < 100 ? value.toFixed(1) : Math.round(value)} ms`;
}

function formatCount(value: number): string {
  if (!Number.isFinite(value) || value < 0) {
    return '—';
  }
  return Math.round(value).toLocaleString();
}

function formatDegrees(value: number): string {
  return Number.isFinite(value) && value >= 0 ? `${value.toFixed(1)}°` : '—';
}

function summarizeMetrics(timings: StageTimings, stats: SceneStats, totalReadyMs: number): string {
  return [
    `load ${formatMilliseconds(timings.modelLoadMs)}`,
    `pre ${formatMilliseconds(timings.preprocessMs)}`,
    `infer ${formatMilliseconds(timings.inferenceMs)}`,
    `post ${formatMilliseconds(timings.postprocessMs)}`,
    `mesh ${formatMilliseconds(stats.meshBuildMs)}`,
    `ready ${formatMilliseconds(totalReadyMs)}`,
    `${formatCount(stats.vertexCount)} verts`,
    `${formatCount(stats.triangleCount)} tris`,
    `FOV ${formatDegrees(stats.fovXDegrees)} × ${formatDegrees(stats.fovYDegrees)}`,
  ].join('  ·  ');
}

export class DepthApp {
  private readonly elements: AppElements;
  private readonly inference: MoGeInference;
  private readonly scene: SpatialScene;
  private readonly windowRef: Window | undefined;
  private readonly onPageHide = (): void => {
    void this.dispose();
  };
  private state: AppState = 'BOOT';
  private modelReady = false;
  private modelError = false;
  private pendingSelection: PendingSelection | undefined;
  private latestRequestId = 0;
  private processing = false;
  private dragDepth = 0;
  private hasScene = false;
  private disposed = false;
  private disposalPromise: Promise<void> | undefined;

  public constructor(documentRef: Document) {
    this.elements = collectElements(documentRef);
    this.windowRef = documentRef.defaultView ?? undefined;
    this.inference = new MoGeInference();
    this.scene = new SpatialScene(this.elements.canvas, this.elements.viewport);
    if (this.windowRef?.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      this.elements.autoMotion.checked = false;
    }

    this.bindEvents();
    this.setSceneControls(false);
    this.setState('BOOT', 'Starting local depth studio…');
    if (this.windowRef) {
      this.windowRef.addEventListener('pagehide', this.onPageHide, { once: true });
    }
    void this.loadModel();
  }

  public get currentState(): AppState {
    return this.state;
  }

  private bindEvents(): void {
    this.elements.fileInput.addEventListener('change', this.onFileInput);
    this.elements.viewMode.addEventListener('change', this.onViewModeChange);
    this.elements.resetView.addEventListener('click', this.onResetView);
    this.elements.autoMotion.addEventListener('change', this.onAutoMotionChange);

    this.elements.dropZone.addEventListener('click', this.onDropZoneClick);
    this.elements.dropZone.addEventListener('keydown', this.onDropZoneKeyDown);
    this.elements.dropZone.addEventListener('dragenter', this.onDragEnter);
    this.elements.dropZone.addEventListener('dragover', this.onDragOver);
    this.elements.dropZone.addEventListener('dragleave', this.onDragLeave);
    this.elements.dropZone.addEventListener('drop', this.onDrop);
  }

  private readonly onFileInput = (event: Event): void => {
    const input = event.currentTarget as HTMLInputElement;
    const file = input.files?.[0];
    // Reset so choosing the same file twice still creates a fresh request.
    input.value = '';
    if (file) {
      this.selectFile(file);
    }
  };

  private readonly onDropZoneClick = (event: MouseEvent): void => {
    const target = event.target;
    if (target instanceof Element && target.closest('label,button,a,input,select')) {
      return;
    }
    this.elements.fileInput.click();
  };

  private readonly onDropZoneKeyDown = (event: KeyboardEvent): void => {
    if (event.key !== 'Enter' && event.key !== ' ') {
      return;
    }
    event.preventDefault();
    this.elements.fileInput.click();
  };

  private readonly onDragEnter = (event: DragEvent): void => {
    event.preventDefault();
    this.dragDepth += 1;
    this.elements.dropZone.classList.add('is-dragging');
    if (event.dataTransfer) {
      event.dataTransfer.dropEffect = 'copy';
    }
  };

  private readonly onDragOver = (event: DragEvent): void => {
    event.preventDefault();
    if (event.dataTransfer) {
      event.dataTransfer.dropEffect = 'copy';
    }
  };

  private readonly onDragLeave = (event: DragEvent): void => {
    event.preventDefault();
    this.dragDepth = Math.max(0, this.dragDepth - 1);
    if (this.dragDepth === 0) {
      this.elements.dropZone.classList.remove('is-dragging');
    }
  };

  private readonly onDrop = (event: DragEvent): void => {
    event.preventDefault();
    this.dragDepth = 0;
    this.elements.dropZone.classList.remove('is-dragging');
    const file = event.dataTransfer?.files?.[0];
    if (file) {
      this.selectFile(file);
    }
  };

  private readonly onViewModeChange = (): void => {
    if (!this.hasScene) {
      return;
    }
    const mode = this.elements.viewMode.value as ViewMode;
    this.scene.setViewMode(mode);
  };

  private readonly onResetView = (): void => {
    if (this.hasScene) {
      this.scene.resetView();
    }
  };

  private readonly onAutoMotionChange = (): void => {
    if (this.hasScene) {
      this.scene.setAutoMotion(this.elements.autoMotion.checked);
    }
  };

  private setSceneControls(enabled: boolean): void {
    this.elements.viewMode.disabled = !enabled;
    this.elements.resetView.disabled = !enabled;
    this.elements.autoMotion.disabled = !enabled;
  }

  private setState(state: AppState, message: string): void {
    this.state = state;
    this.elements.root.dataset.state = state.toLowerCase();
    this.elements.root.setAttribute('aria-busy', LOADING_STATES.has(state) ? 'true' : 'false');
    this.elements.statusState.textContent = STATE_LABELS[state];
    this.elements.statusMessage.textContent = message;
  }

  private setError(message: string, detail?: unknown): void {
    this.elements.metricSummary.textContent = '';
    this.setState('ERROR', message);
    if (detail !== undefined) {
      reportError('Operation failed', detail);
    }
  }

  private async loadModel(): Promise<void> {
    this.setState('MODEL_LOADING', 'Loading the local MoGe depth model…');
    try {
      await this.inference.load();
      if (this.disposed) {
        return;
      }
      this.modelReady = true;
      this.modelError = false;
      if (this.pendingSelection) {
        void this.drainSelections();
      } else if (this.state === 'MODEL_LOADING') {
        this.setState('READY', 'Model ready — choose an image to begin.');
      }
    } catch (error) {
      if (this.disposed) {
        return;
      }
      this.modelError = true;
      this.pendingSelection = undefined;
      reportError('Model loading failed', error);
      this.setError(formatModelError(error));
    }
  }

  private selectFile(file: File): void {
    const requestId = this.latestRequestId + 1;
    this.latestRequestId = requestId;
    // A new selection always supersedes a queued one, including an invalid
    // selection. Any in-flight older request will clean up when it settles.
    this.pendingSelection = undefined;

    const validationError = validateFile(file);
    if (validationError) {
      console.warn(`[Depth Studio] ${validationError}`, file);
      this.setError(validationError);
      return;
    }

    if (this.modelError) {
      this.setError('The depth model is unavailable. Reload the page to try again.');
      return;
    }

    this.pendingSelection = { id: requestId, file };
    if (!this.modelReady) {
      this.setState('MODEL_LOADING', `Waiting for the local model before opening ${file.name}…`);
      return;
    }

    if (this.processing) {
      this.setState('INFERENCING', `Queued latest image ${file.name}…`);
      return;
    }
    void this.drainSelections();
  }

  private async drainSelections(): Promise<void> {
    if (this.processing || !this.modelReady || this.disposed) {
      return;
    }
    this.processing = true;
    try {
      while (this.pendingSelection && !this.disposed) {
        const selection = this.pendingSelection;
        this.pendingSelection = undefined;
        await this.processSelection(selection);
      }
    } finally {
      this.processing = false;
    }
  }

  private async processSelection(selection: PendingSelection): Promise<void> {
    if (this.disposed || selection.id !== this.latestRequestId) {
      return;
    }

    const readyStart = performance.now();
    this.setState('INFERENCING', `Estimating depth for ${selection.file.name}…`);
    // Start both browser decode paths before awaiting either one. The model's
    // own 800px bitmap and the display texture are deliberately independent.
    const textureTask = createTextureBitmap(selection.file);
    const inferenceTask = Promise.resolve().then(() => this.inference.infer(selection.file));
    const [textureResult, inferenceResult] = await Promise.allSettled([
      textureTask,
      inferenceTask,
    ]);

    if (selection.id !== this.latestRequestId || this.disposed) {
      if (textureResult.status === 'fulfilled') {
        closeBitmap(textureResult.value);
      }
      if (inferenceResult.status === 'fulfilled') {
        closeBitmap(inferenceResult.value.inferenceImage);
      }
      return;
    }

    if (textureResult.status === 'rejected' || inferenceResult.status === 'rejected') {
      if (textureResult.status === 'fulfilled') {
        closeBitmap(textureResult.value);
      }
      if (inferenceResult.status === 'fulfilled') {
        closeBitmap(inferenceResult.value.inferenceImage);
      }
      let failure: unknown;
      let textureFailed = false;
      if (textureResult.status === 'rejected') {
        failure = textureResult.reason;
        textureFailed = true;
      } else if (inferenceResult.status === 'rejected') {
        failure = inferenceResult.reason;
      } else {
        // The surrounding condition proves that one branch is rejected.
        failure = new Error('Image processing ended without a result.');
      }
      reportError('Image processing failed', failure);
      this.setError(formatProcessingError(failure, textureFailed));
      return;
    }

    let textureBitmap: ImageBitmap | undefined = textureResult.value;
    const inferenceOutput: MoGeInferenceResult = inferenceResult.value;
    try {
      if (selection.id !== this.latestRequestId || this.disposed) {
        closeBitmap(textureBitmap);
        textureBitmap = undefined;
        return;
      }

      const stats = this.scene.setScene(inferenceOutput.result, textureBitmap);
      // Ownership moves to SpatialScene only after setScene has accepted the
      // bitmap. It will release the previous texture when replacing a scene.
      textureBitmap = undefined;
      this.hasScene = true;
      this.elements.dropZone.hidden = true;
      this.elements.sceneGuide.hidden = false;
      this.setSceneControls(true);
      this.scene.setViewMode(this.elements.viewMode.value as ViewMode);
      this.scene.setAutoMotion(this.elements.autoMotion.checked);
      this.elements.metricSummary.textContent = summarizeMetrics(
        inferenceOutput.timings,
        stats,
        performance.now() - readyStart,
      );
      this.setState('SCENE_READY', 'Scene ready — move the pointer or drag to look around.');
    } catch (error) {
      closeBitmap(textureBitmap);
      textureBitmap = undefined;
      reportError('Scene creation failed', error);
      this.setError(formatProcessingError(error, false));
    } finally {
      // SpatialScene receives only the display texture. MoGe's separate
      // inferenceImage is always app-owned and must be closed here.
      closeBitmap(inferenceOutput.inferenceImage);
    }
  }

  public dispose(): Promise<void> {
    if (this.disposalPromise) {
      return this.disposalPromise;
    }
    this.disposed = true;
    this.latestRequestId += 1;
    this.pendingSelection = undefined;
    this.elements.fileInput.removeEventListener('change', this.onFileInput);
    this.elements.viewMode.removeEventListener('change', this.onViewModeChange);
    this.elements.resetView.removeEventListener('click', this.onResetView);
    this.elements.autoMotion.removeEventListener('change', this.onAutoMotionChange);
    this.elements.dropZone.removeEventListener('click', this.onDropZoneClick);
    this.elements.dropZone.removeEventListener('keydown', this.onDropZoneKeyDown);
    this.elements.dropZone.removeEventListener('dragenter', this.onDragEnter);
    this.elements.dropZone.removeEventListener('dragover', this.onDragOver);
    this.elements.dropZone.removeEventListener('dragleave', this.onDragLeave);
    this.elements.dropZone.removeEventListener('drop', this.onDrop);
    this.windowRef?.removeEventListener('pagehide', this.onPageHide);

    this.disposalPromise = (async () => {
      try {
        this.scene.dispose();
      } catch (error) {
        reportError('Scene disposal failed', error);
      }
      try {
        await this.inference.dispose();
      } catch (error) {
        reportError('Model disposal failed', error);
      }
    })();
    return this.disposalPromise;
  }
}

export function createApp(documentRef: Document = document): DepthApp {
  return new DepthApp(documentRef);
}
