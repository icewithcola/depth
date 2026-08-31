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

export type WebGpuPowerPreference = 'default' | 'high-performance' | 'low-power';

export interface WebGpuAdapterProbe {
  preference: WebGpuPowerPreference;
  outcome: 'available' | 'null' | 'error';
  elapsedMs: number;
  error?: string;
  info?: Record<string, string>;
  features?: string[];
  limits?: Record<string, number>;
}

export interface WebGpuDiagnosticReport {
  generatedAt: string;
  origin: string;
  protocol: string;
  secureContext: boolean;
  crossOriginIsolated: boolean;
  visibilityState: string;
  userAgent: string;
  platform: string;
  navigatorGpu: boolean;
  preferredCanvasFormat?: string;
  probes: WebGpuAdapterProbe[];
}

const RELEVANT_LIMITS = [
  'maxBufferSize',
  'maxStorageBufferBindingSize',
  'maxComputeWorkgroupStorageSize',
  'maxComputeInvocationsPerWorkgroup',
  'maxComputeWorkgroupsPerDimension',
] as const;

function errorText(error: unknown): string {
  return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
}

function adapterInfo(adapter: GPUAdapter): Record<string, string> {
  const info = adapter.info;
  const result: Record<string, string> = {};
  for (const key of ['vendor', 'architecture', 'device', 'description'] as const) {
    const value = info[key];
    if (value) result[key] = value;
  }
  return result;
}

function adapterLimits(adapter: GPUAdapter): Record<string, number> {
  const result: Record<string, number> = {};
  const limits = adapter.limits as unknown as Record<string, number | undefined>;
  for (const key of RELEVANT_LIMITS) {
    const value = limits[key];
    if (typeof value === 'number') result[key] = value;
  }
  return result;
}

async function probeAdapter(
  gpu: GPU,
  preference: WebGpuPowerPreference,
): Promise<WebGpuAdapterProbe> {
  const start = performance.now();
  try {
    // Calling with no argument is observably different on dual-GPU macOS:
    // Chrome chooses according to AC power and display attachment.
    const adapter = preference === 'default'
      ? await gpu.requestAdapter()
      : await gpu.requestAdapter({ powerPreference: preference });
    const elapsedMs = performance.now() - start;
    if (!adapter) return { preference, outcome: 'null', elapsedMs };
    return {
      preference,
      outcome: 'available',
      elapsedMs,
      info: adapterInfo(adapter),
      features: [...adapter.features].sort(),
      limits: adapterLimits(adapter),
    };
  } catch (error) {
    return {
      preference,
      outcome: 'error',
      elapsedMs: performance.now() - start,
      error: errorText(error),
    };
  }
}

/** Run explicit, read-only adapter probes suitable for a pasted bug report. */
export async function collectWebGpuDiagnostics(): Promise<WebGpuDiagnosticReport> {
  const windowAvailable = typeof window !== 'undefined';
  const navigatorAvailable = typeof navigator !== 'undefined';
  const gpu = navigatorAvailable ? navigator.gpu : undefined;
  const report: WebGpuDiagnosticReport = {
    generatedAt: new Date().toISOString(),
    origin: windowAvailable ? window.location.origin : '(no window)',
    protocol: windowAvailable ? window.location.protocol : '(no window)',
    secureContext: windowAvailable && window.isSecureContext,
    crossOriginIsolated: windowAvailable && window.crossOriginIsolated,
    visibilityState: typeof document !== 'undefined' ? document.visibilityState : '(no document)',
    userAgent: navigatorAvailable ? navigator.userAgent : '(no navigator)',
    platform: navigatorAvailable ? navigator.platform : '(no navigator)',
    navigatorGpu: Boolean(gpu),
    probes: [],
  };

  if (!gpu) return report;
  try {
    report.preferredCanvasFormat = gpu.getPreferredCanvasFormat();
  } catch {
    // Older implementations may expose requestAdapter without this helper.
  }
  for (const preference of ['default', 'high-performance', 'low-power'] as const) {
    report.probes.push(await probeAdapter(gpu, preference));
  }
  return report;
}

export function formatWebGpuDiagnostics(report: WebGpuDiagnosticReport): string {
  const lines = [
    'Depth Studio WebGPU diagnostics',
    `generated: ${report.generatedAt}`,
    `origin: ${report.origin}`,
    `protocol: ${report.protocol}`,
    `isSecureContext: ${String(report.secureContext)}`,
    `crossOriginIsolated: ${String(report.crossOriginIsolated)}`,
    `visibilityState: ${report.visibilityState}`,
    `navigator.gpu: ${String(report.navigatorGpu)}`,
    `preferredCanvasFormat: ${report.preferredCanvasFormat ?? '(unavailable)'}`,
    `platform: ${report.platform}`,
    `userAgent: ${report.userAgent}`,
  ];
  for (const probe of report.probes) {
    lines.push('', `[adapter: ${probe.preference}]`);
    lines.push(`outcome: ${probe.outcome}`);
    lines.push(`elapsedMs: ${probe.elapsedMs.toFixed(1)}`);
    if (probe.error) lines.push(`error: ${probe.error}`);
    if (probe.info) lines.push(`info: ${JSON.stringify(probe.info)}`);
    if (probe.features) lines.push(`features: ${probe.features.join(', ') || '(none reported)'}`);
    if (probe.limits) lines.push(`limits: ${JSON.stringify(probe.limits)}`);
  }
  const noAdapter = report.navigatorGpu && report.probes.every((probe) => probe.outcome !== 'available');
  const linux = /linux/i.test(`${report.platform} ${report.userAgent}`);
  if (noAdapter) {
    lines.push('', '[suggested checks]');
    lines.push('1. Restart Chrome after confirming graphics acceleration is enabled.');
    lines.push('2. Open chrome://gpu and copy its Problems Detected section.');
    if (linux) {
      lines.push('3. On Linux, enable chrome://flags/#ignore-gpu-blocklist and relaunch Chrome.');
      lines.push('4. If WebGPU is not enabled for the Linux backend, inspect chrome://flags/#enable-unsafe-webgpu.');
    }
  }
  return lines.join('\n');
}

function secureContextDetail(): string {
  if (typeof window === 'undefined') return '';
  if (window.isSecureContext) return '';
  return ` The page is not a secure context (${window.location.origin}).`;
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
    // Do not constrain adapter selection here. On dual-GPU macOS systems,
    // Chrome applies its own AC-power/display policy when no preference is
    // supplied. A high-performance-only probe can return null even though a
    // perfectly usable Metal adapter exists; ORT also uses the default policy.
    const adapter = await gpu.requestAdapter();
    if (!adapter) {
      return {
        supported: false,
        reason:
          'Chrome exposed WebGPU but returned no default GPU adapter.' +
          secureContextDetail() +
          ' Restart Chrome after enabling graphics acceleration, then inspect the Problems Detected section in chrome://gpu.',
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
