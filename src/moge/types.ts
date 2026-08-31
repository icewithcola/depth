/** Canonical, row-major MoGe-2 result consumed by scene code. */
export interface MoGeResult {
  width: number;
  height: number;

  /** H * W * 3 interleaved metric XYZ in OpenCV coordinates: +X right, +Y down, +Z forward. */
  points: Float32Array;
  /** H * W positive metric camera depth in row-major image order. */
  depth: Float32Array;
  /** Optional H * W * 3 interleaved OpenCV-coordinate unit normals. */
  normals?: Float32Array;
  /** H * W values, exactly 0 or 1. */
  mask: Uint8Array;
  /** Row-major normalized 3x3 K: [fx,0,cx, 0,fy,cy, 0,0,1]. */
  intrinsics: Float32Array;
}

/**
 * Raw row-major output arrays from the official MoGe-2 ONNX forward graph.
 * `points`/`normal` are interleaved HWC (`((y * width + x) * 3) + c`), while
 * `mask` is HW (`y * width + x`).  Forward has already applied the normal
 * normalization, mask sigmoid, and metric-scale exponentiation.
 */
export interface RawMoGeOutputs {
  width: number;
  height: number;
  points: Float32Array;
  normal?: Float32Array;
  mask?: Float32Array;
  metricScale?: number;
}

export interface StageTimings {
  /** One-time session creation and model load duration; reused across images. */
  modelLoadMs: number;
  preprocessMs: number;
  inferenceMs: number;
  postprocessMs: number;
}

export interface MoGeInferenceResult {
  result: MoGeResult;
  inferenceImage: ImageBitmap;
  timings: StageTimings;
}
