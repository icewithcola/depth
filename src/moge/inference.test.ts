import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  createSession: vi.fn(),
  preprocess: vi.fn(),
  postprocess: vi.fn(),
  run: vi.fn(),
  release: vi.fn(),
}));

vi.mock('onnxruntime-web/webgpu', () => ({
  Tensor: class FakeTensor {
    public constructor(
      public readonly type: string,
      public readonly data: Float32Array | BigInt64Array,
      public readonly dims: readonly number[],
    ) {}
  },
  InferenceSession: { create: mocks.createSession },
}));

vi.mock('../platform/webgpu', () => ({
  checkWebGpuSupport: vi.fn(async () => ({ supported: true })),
}));

vi.mock('./preprocess', () => ({
  preprocessImage: mocks.preprocess,
}));

vi.mock('./postprocess', () => ({
  postprocessMoGe: mocks.postprocess,
}));

import { MoGeInference } from './inference';
import type { MoGeResult } from './types';

function tensor(data: Float32Array, dims: number[]): { data: Float32Array; dims: number[] } {
  return { data, dims };
}

function runtimeOutputs(): Record<string, { data: Float32Array; dims: number[] }> {
  return {
    points: tensor(new Float32Array(12).fill(1), [1, 2, 2, 3]),
    normal: tensor(new Float32Array(12).fill(0.5), [1, 2, 2, 3]),
    mask: tensor(new Float32Array(4).fill(0.9), [1, 2, 2]),
    scale: tensor(new Float32Array([2]), [1]),
  };
}

describe('MoGeInference session lifecycle', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    const result: MoGeResult = {
      width: 2,
      height: 2,
      points: new Float32Array(12),
      depth: new Float32Array([1, 1, 1, 1]),
      mask: new Uint8Array([1, 1, 1, 1]),
      intrinsics: new Float32Array([1, 0, 0.5, 0, 1, 0.5, 0, 0, 1]),
    };
    mocks.postprocess.mockReturnValue(result);
    mocks.preprocess.mockImplementation(async () => ({
      width: 2,
      height: 2,
      data: new Float32Array(12),
      tensor: new Float32Array(12),
      imageTensor: new Float32Array(12),
      tensorShape: [1, 3, 2, 2] as const,
      inferenceImage: { width: 2, height: 2, close: vi.fn() } as unknown as ImageBitmap,
      ownsInferenceImage: true,
    }));
    mocks.run.mockResolvedValue(runtimeOutputs());
    mocks.createSession.mockResolvedValue({
      inputNames: ['image', 'num_tokens'],
      outputNames: ['points', 'normal', 'mask', 'scale'],
      inputMetadata: [
        { name: 'image', type: 'float32', shape: [1, 3, 'height', 'width'] },
        { name: 'num_tokens', type: 'int64', shape: [] },
      ],
      outputMetadata: [
        { name: 'points', type: 'float32', shape: [1, 'height', 'width', 3] },
        { name: 'normal', type: 'float32', shape: [1, 'height', 'width', 3] },
        { name: 'mask', type: 'float32', shape: [1, 'height', 'width'] },
        { name: 'scale', type: 'float32', shape: [1] },
      ],
      run: mocks.run,
      release: mocks.release,
    });
  });

  it('reuses one WebGPU session across sequential images and releases it once', async () => {
    const inference = new MoGeInference('/model.onnx');
    const source = {} as Blob;

    const first = await inference.infer(source);
    const second = await inference.infer(source);

    expect(first.result.width).toBe(2);
    expect(second.result.height).toBe(2);
    expect(mocks.createSession).toHaveBeenCalledTimes(1);
    expect(mocks.createSession).toHaveBeenCalledWith('/model.onnx', {
      executionProviders: ['webgpu'],
    });
    expect(mocks.run).toHaveBeenCalledTimes(2);
    expect(mocks.preprocess).toHaveBeenCalledTimes(2);
    expect(inference.modelLoadMs).toBeGreaterThanOrEqual(0);

    await inference.dispose();
    expect(mocks.release).toHaveBeenCalledTimes(1);
  });

  it('waits for an active run before releasing and stays disposed', async () => {
    let finishRun: ((outputs: ReturnType<typeof runtimeOutputs>) => void) | undefined;
    mocks.run.mockReturnValueOnce(new Promise((resolve) => {
      finishRun = resolve;
    }));
    const inference = new MoGeInference('/model.onnx');
    const running = inference.infer({} as Blob);
    await vi.waitFor(() => expect(mocks.run).toHaveBeenCalledTimes(1));

    const disposing = inference.dispose();
    await Promise.resolve();
    expect(mocks.release).not.toHaveBeenCalled();
    finishRun?.(runtimeOutputs());
    await running;
    await disposing;

    expect(mocks.release).toHaveBeenCalledTimes(1);
    await expect(inference.load()).rejects.toThrow(/disposed/);
  });
});
