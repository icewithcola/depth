import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  infer: vi.fn(),
  load: vi.fn(),
  disposeInference: vi.fn(),
  setScene: vi.fn(),
  disposeScene: vi.fn(),
  setViewMode: vi.fn(),
  resetView: vi.fn(),
  setAutoMotion: vi.fn(),
}));

vi.mock('./moge/inference', () => ({
  MoGeInference: class FakeMoGeInference {
    public load = mocks.load;
    public infer = mocks.infer;
    public dispose = mocks.disposeInference;
  },
}));

vi.mock('./scene/SpatialScene', () => ({
  SpatialScene: class FakeSpatialScene {
    public setScene = mocks.setScene;
    public dispose = mocks.disposeScene;
    public setViewMode = mocks.setViewMode;
    public resetView = mocks.resetView;
    public setAutoMotion = mocks.setAutoMotion;
  },
}));

vi.mock('./platform/webgpu', () => ({
  collectWebGpuDiagnostics: vi.fn(async () => ({ probes: [] })),
  formatWebGpuDiagnostics: vi.fn(() => 'diagnostics-report'),
}));

import { DepthApp } from './app';

type AppElementId =
  | 'app'
  | 'image-input'
  | 'paste-image'
  | 'paste-image-dropzone'
  | 'view-mode'
  | 'reset-view'
  | 'auto-motion'
  | 'viewport'
  | 'scene-canvas'
  | 'drop-zone'
  | 'scene-guide'
  | 'status-dot'
  | 'status-state'
  | 'status-message'
  | 'metric-summary'
  | 'gpu-debug'
  | 'gpu-debug-dialog'
  | 'gpu-debug-close'
  | 'gpu-debug-run'
  | 'gpu-debug-copy'
  | 'gpu-debug-output'
  | 'gpu-debug-feedback';

function createMockElement(id: string, tagName: string = 'div'): HTMLElement {
  const listeners: Record<string, Set<EventListener>> = {};
  const el: Partial<HTMLElement & HTMLInputElement & HTMLSelectElement & HTMLDialogElement & HTMLTextAreaElement> = {
    id,
    tagName: tagName.toUpperCase(),
    dataset: {},
    hidden: false,
    disabled: false,
    open: false,
    value: '',
    files: null as unknown as FileList,
    textContent: '',
    setAttribute: vi.fn(),
    removeAttribute: vi.fn(),
    classList: {
      add: vi.fn(),
      remove: vi.fn(),
      contains: vi.fn(() => false),
    } as unknown as DOMTokenList,
    click: vi.fn(),
    showModal: vi.fn(function (this: { open: boolean }) {
      this.open = true;
    }),
    close: vi.fn(function (this: { open: boolean }) {
      this.open = false;
    }),
    closest: vi.fn((selector: string) => {
      const ids = selector
        .split(',')
        .map((part) => part.trim())
        .filter((part) => part.startsWith('#'))
        .map((part) => part.slice(1));
      if (ids.includes(id)) {
        return el as HTMLElement;
      }
      return null;
    }),
    addEventListener: vi.fn((type: string, listener: EventListener) => {
      if (!listeners[type]) listeners[type] = new Set();
      listeners[type].add(listener);
    }),
    removeEventListener: vi.fn((type: string, listener: EventListener) => {
      listeners[type]?.delete(listener);
    }),
    dispatchEvent: vi.fn((event: Event) => {
      const set = listeners[event.type];
      if (set) {
        for (const listener of set) {
          listener(event);
        }
      }
      return true;
    }),
  };
  return el as HTMLElement;
}

function createMockDocument(): {
  documentRef: Document;
  elements: Record<AppElementId, HTMLElement>;
  triggerDocEvent: (type: string, event: Event) => void;
} {
  const elements: Record<AppElementId, HTMLElement> = {
    'app': createMockElement('app'),
    'image-input': createMockElement('image-input', 'input'),
    'paste-image': createMockElement('paste-image', 'button'),
    'paste-image-dropzone': createMockElement('paste-image-dropzone', 'button'),
    'view-mode': createMockElement('view-mode', 'select'),
    'reset-view': createMockElement('reset-view', 'button'),
    'auto-motion': createMockElement('auto-motion', 'input'),
    'viewport': createMockElement('viewport'),
    'scene-canvas': createMockElement('scene-canvas', 'canvas'),
    'drop-zone': createMockElement('drop-zone'),
    'scene-guide': createMockElement('scene-guide'),
    'status-dot': createMockElement('status-dot'),
    'status-state': createMockElement('status-state'),
    'status-message': createMockElement('status-message'),
    'metric-summary': createMockElement('metric-summary'),
    'gpu-debug': createMockElement('gpu-debug', 'button'),
    'gpu-debug-dialog': createMockElement('gpu-debug-dialog', 'dialog'),
    'gpu-debug-close': createMockElement('gpu-debug-close', 'button'),
    'gpu-debug-run': createMockElement('gpu-debug-run', 'button'),
    'gpu-debug-copy': createMockElement('gpu-debug-copy', 'button'),
    'gpu-debug-output': createMockElement('gpu-debug-output', 'textarea'),
    'gpu-debug-feedback': createMockElement('gpu-debug-feedback'),
  };

  const docListeners: Record<string, Set<EventListener>> = {};

  const documentRef = {
    getElementById: vi.fn((id: string) => elements[id as AppElementId] ?? null),
    addEventListener: vi.fn((type: string, listener: EventListener) => {
      if (!docListeners[type]) docListeners[type] = new Set();
      docListeners[type].add(listener);
    }),
    removeEventListener: vi.fn((type: string, listener: EventListener) => {
      docListeners[type]?.delete(listener);
    }),
    defaultView: {
      matchMedia: vi.fn(() => ({ matches: false })),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      navigator: {
        clipboard: {
          read: vi.fn(),
          writeText: vi.fn(),
        },
      },
    } as unknown as Window,
  } as unknown as Document;

  const triggerDocEvent = (type: string, event: Event) => {
    const set = docListeners[type];
    if (set) {
      for (const listener of set) {
        listener(event);
      }
    }
  };

  return { documentRef, elements, triggerDocEvent };
}

describe('DepthApp clipboard support', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.load.mockResolvedValue(undefined);
    mocks.infer.mockResolvedValue({
      result: { width: 2, height: 2 },
      timings: { modelLoadMs: 0, preprocessMs: 0, inferenceMs: 0, postprocessMs: 0 },
      inferenceImage: { width: 2, height: 2, close: vi.fn() },
    });
    mocks.setScene.mockReturnValue({
      vertexCount: 10,
      triangleCount: 5,
      meshBuildMs: 1,
      fovXDegrees: 45,
      fovYDegrees: 45,
    });
    vi.stubGlobal('createImageBitmap', vi.fn(async () => ({
      width: 100,
      height: 100,
      close: vi.fn(),
    })));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('handles paste event containing an image file', async () => {
    const { documentRef, elements, triggerDocEvent } = createMockDocument();
    const app = new DepthApp(documentRef);
    await Promise.resolve(); // allow loadModel to settle

    const file = new File(['image-data'], 'test.png', { type: 'image/png' });
    const preventDefault = vi.fn();
    const pasteEvent = {
      type: 'paste',
      target: elements['app'],
      clipboardData: {
        files: [file],
        items: [],
      },
      preventDefault,
    } as unknown as ClipboardEvent;

    triggerDocEvent('paste', pasteEvent);

    expect(preventDefault).toHaveBeenCalled();
    expect(elements['status-message'].textContent).toContain('test.png');
    await app.dispose();
  });

  it('reports error when paste event contains no image', async () => {
    const { documentRef, elements, triggerDocEvent } = createMockDocument();
    const app = new DepthApp(documentRef);
    await Promise.resolve();

    const preventDefault = vi.fn();
    const pasteEvent = {
      type: 'paste',
      target: elements['app'],
      clipboardData: {
        files: [],
        items: [{ kind: 'string', type: 'text/plain' }],
        types: ['text/plain'],
      },
      preventDefault,
    } as unknown as ClipboardEvent;

    triggerDocEvent('paste', pasteEvent);

    expect(preventDefault).toHaveBeenCalled();
    expect(elements['status-message'].textContent).toContain('No image found on the clipboard');
    await app.dispose();
  });

  it('ignores paste events when GPU debug dialog is open', async () => {
    const { documentRef, elements, triggerDocEvent } = createMockDocument();
    const app = new DepthApp(documentRef);
    await Promise.resolve();

    (elements['gpu-debug-dialog'] as HTMLDialogElement).open = true;

    const file = new File(['image-data'], 'test.png', { type: 'image/png' });
    const preventDefault = vi.fn();
    const pasteEvent = {
      type: 'paste',
      target: elements['app'],
      clipboardData: {
        files: [file],
        items: [],
      },
      preventDefault,
    } as unknown as ClipboardEvent;

    triggerDocEvent('paste', pasteEvent);

    expect(preventDefault).not.toHaveBeenCalled();
    await app.dispose();
  });

  it('reads image from clipboard when paste button is clicked', async () => {
    const { documentRef, elements } = createMockDocument();
    const blob = new Blob(['image-data'], { type: 'image/png' });
    const fakeClipboardItem = {
      types: ['image/png'],
      getType: vi.fn(async () => blob),
    };
    (documentRef.defaultView?.navigator.clipboard.read as ReturnType<typeof vi.fn>).mockResolvedValue([fakeClipboardItem]);

    const app = new DepthApp(documentRef);
    await Promise.resolve();

    const stopPropagation = vi.fn();
    elements['paste-image'].dispatchEvent({
      type: 'click',
      stopPropagation,
    } as unknown as MouseEvent);

    await vi.waitFor(() => {
      expect(mocks.infer).toHaveBeenCalledWith(
        expect.objectContaining({ name: 'clipboard.png' }),
      );
      expect(elements['status-state'].textContent).toBe('Scene ready');
    });

    await app.dispose();
  });

  it('handles permission denied when reading from clipboard', async () => {
    const { documentRef, elements } = createMockDocument();
    const domException = new Error('Permission denied');
    domException.name = 'NotAllowedError';
    (documentRef.defaultView?.navigator.clipboard.read as ReturnType<typeof vi.fn>).mockRejectedValue(domException);

    const app = new DepthApp(documentRef);
    await Promise.resolve();

    elements['paste-image'].dispatchEvent({
      type: 'click',
      stopPropagation: vi.fn(),
    } as unknown as MouseEvent);

    await vi.waitFor(() => {
      expect(elements['status-message'].textContent).toContain('Clipboard permission was denied');
    });

    await app.dispose();
  });

  it('handles unsupported clipboard API when clicking paste button', async () => {
    const { documentRef, elements } = createMockDocument();
    (documentRef.defaultView as unknown as { navigator: { clipboard: unknown } }).navigator.clipboard = {};

    const app = new DepthApp(documentRef);
    await Promise.resolve();

    elements['paste-image'].dispatchEvent({
      type: 'click',
      stopPropagation: vi.fn(),
    } as unknown as MouseEvent);

    await vi.waitFor(() => {
      expect(elements['status-message'].textContent).toContain('Clipboard reading is not supported');
    });

    await app.dispose();
  });

  it('handles paste button inside drop-zone without opening file input', async () => {
    const { documentRef, elements } = createMockDocument();
    const blob = new Blob(['image-data'], { type: 'image/png' });
    const fakeClipboardItem = {
      types: ['image/png'],
      getType: vi.fn(async () => blob),
    };
    (documentRef.defaultView?.navigator.clipboard.read as ReturnType<typeof vi.fn>).mockResolvedValue([fakeClipboardItem]);

    const app = new DepthApp(documentRef);
    await Promise.resolve();

    elements['drop-zone'].dispatchEvent({
      type: 'click',
      target: elements['paste-image-dropzone'],
    } as unknown as MouseEvent);

    // File input should not be clicked
    expect(elements['image-input'].click).not.toHaveBeenCalled();

    // Paste button clicked
    elements['paste-image-dropzone'].dispatchEvent({
      type: 'click',
      stopPropagation: vi.fn(),
    } as unknown as MouseEvent);

    await vi.waitFor(() => {
      expect(mocks.infer).toHaveBeenCalledWith(
        expect.objectContaining({ name: 'clipboard.png' }),
      );
    });

    await app.dispose();
  });

  it('cleans up clipboard event listeners on dispose', async () => {
    const { documentRef, elements } = createMockDocument();
    const app = new DepthApp(documentRef);

    await app.dispose();

    expect(documentRef.removeEventListener).toHaveBeenCalledWith('paste', expect.any(Function));
    expect(elements['paste-image'].removeEventListener).toHaveBeenCalledWith('click', expect.any(Function));
    expect(elements['paste-image-dropzone'].removeEventListener).toHaveBeenCalledWith('click', expect.any(Function));
  });
});
