import { afterEach, describe, expect, it, vi } from 'vitest';

import { checkWebGpuSupport } from './webgpu';

describe('checkWebGpuSupport', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('lets Chrome choose the default adapter without a power preference', async () => {
    const requestAdapter = vi.fn(async () => ({ requestDevice: vi.fn() }));
    vi.stubGlobal('navigator', { gpu: { requestAdapter } });

    await expect(checkWebGpuSupport()).resolves.toEqual({ supported: true });
    expect(requestAdapter).toHaveBeenCalledOnce();
    expect(requestAdapter).toHaveBeenCalledWith();
  });

  it('reports when Chrome exposes WebGPU but finds no adapter', async () => {
    vi.stubGlobal('navigator', { gpu: { requestAdapter: vi.fn(async () => null) } });

    const result = await checkWebGpuSupport();
    expect(result.supported).toBe(false);
    expect(result.reason).toContain('returned no default GPU adapter');
    expect(result.reason).toContain('chrome://gpu');
  });
});
