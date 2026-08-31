/**
 * The small amount of WebGPU state that is useful to surface to the UI.
 *
 * Checking for `navigator.gpu` alone is not sufficient: browsers can expose
 * the API while no adapter is available (for example in an insecure context,
 * or when the user has disabled GPU access).  `checkWebGpuSupport` therefore
 * performs the adapter probe as well.
 */
export interface WebGpuDiagnostics {
  supported: boolean;
  /** A user-facing explanation when `supported` is false. */
  reason?: string;
}

/**
 * Probe WebGPU without requesting a device.
 *
 * This function is intentionally asynchronous because adapter discovery is
 * asynchronous in the WebGPU API.  It is safe to call during SSR/tests where
 * `navigator` does not exist.
 */
export async function checkWebGpuSupport(): Promise<WebGpuDiagnostics> {
  if (typeof navigator === 'undefined') {
    return {
      supported: false,
      reason: 'WebGPU is unavailable because this code is not running in a browser.',
    };
  }

  const gpu = navigator.gpu;
  if (!gpu || typeof gpu.requestAdapter !== 'function') {
    return {
      supported: false,
      reason:
        'WebGPU hardware acceleration is required. Use current Chrome or Edge in a secure context with WebGPU enabled.',
    };
  }

  try {
    const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      return {
        supported: false,
        reason:
          'WebGPU could not find a GPU adapter. Check browser permissions, secure-context requirements, and GPU driver support.',
      };
    }

    return { supported: true };
  } catch (error) {
    const detail = error instanceof Error ? ` (${error.message})` : '';
    return {
      supported: false,
      reason: `WebGPU adapter discovery failed${detail}`,
    };
  }
}
