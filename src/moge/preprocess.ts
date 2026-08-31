/** The largest side sent to the model. */
export const MAX_INFERENCE_SIDE = 800;

/** Input accepted by the browser decoder. File is a Blob, so it is included. */
export type MoGeImageSource = Blob | ImageBitmap;

export interface PreprocessedImage {
  /** Width of the resized, orientation-correct inference image. */
  width: number;
  /** Height of the resized, orientation-correct inference image. */
  height: number;
  /**
   * RGB float32 tensor data in NCHW order, with an implicit batch of one.
   * Each channel is contiguous and each channel is row-major.
   */
  data: Float32Array;
  /** Alias for `data`, useful at call sites constructing an ORT Tensor. */
  tensor: Float32Array;
  /** Alias for `data` retained to make the tensor nature explicit to callers. */
  imageTensor: Float32Array;
  /** The tensor shape corresponding to `data`. */
  tensorShape: readonly [1, 3, number, number];
  /**
   * Orientation-correct image at the exact inference dimensions. The caller
   * owns this bitmap and should call `close()` when it no longer needs it.
   */
  inferenceImage: ImageBitmap;
  /** True when preprocessing created the bitmap and can release it on failure. */
  ownsInferenceImage: boolean;
}

const IMAGE_ORIENTATION_OPTIONS: ImageBitmapOptions = {
  imageOrientation: 'from-image',
};

function hasImageBitmapShape(source: unknown): source is ImageBitmap {
  if (source === null || typeof source !== 'object') {
    return false;
  }

  const candidate = source as { width?: unknown; height?: unknown };
  return (
    typeof candidate.width === 'number' &&
    typeof candidate.height === 'number'
  );
}

function closeImage(image: ImageBitmap): void {
  const close = (image as unknown as { close?: unknown }).close;
  if (typeof close === 'function') {
    close.call(image);
  }
}

function imageDimensions(image: ImageBitmap): { width: number; height: number } {
  const { width, height } = image;
  if (
    !Number.isInteger(width) ||
    !Number.isInteger(height) ||
    width <= 0 ||
    height <= 0
  ) {
    throw new Error(
      `Decoded image has invalid dimensions (${String(width)}x${String(height)}); expected positive integers.`,
    );
  }
  return { width, height };
}

function inferenceDimensions(width: number, height: number): { width: number; height: number } {
  const scale = Math.min(1, MAX_INFERENCE_SIDE / Math.max(width, height));
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
  };
}

async function decodeImage(source: MoGeImageSource): Promise<{
  image: ImageBitmap;
  owned: boolean;
}> {
  if (hasImageBitmapShape(source)) {
    imageDimensions(source);
    return { image: source, owned: false };
  }

  if (typeof createImageBitmap !== 'function') {
    throw new Error(
      'Image decoding requires the browser createImageBitmap API, which is unavailable in this environment.',
    );
  }

  try {
    const image = await createImageBitmap(source, IMAGE_ORIENTATION_OPTIONS);
    try {
      imageDimensions(image);
    } catch (error) {
      closeImage(image);
      throw error;
    }
    return { image, owned: true };
  } catch (error) {
    const detail = error instanceof Error ? ` (${error.message})` : '';
    throw new Error(`Could not decode the selected image${detail}`, { cause: error });
  }
}

async function resizeImage(
  image: ImageBitmap,
  width: number,
  height: number,
  owned: boolean,
): Promise<{ image: ImageBitmap; owned: boolean }> {
  const current = imageDimensions(image);
  if (current.width === width && current.height === height) {
    return { image, owned };
  }

  if (typeof createImageBitmap !== 'function') {
    throw new Error(
      'Image resizing requires the browser createImageBitmap API, which is unavailable in this environment.',
    );
  }

  try {
    // The first decode applies EXIF orientation.  The resized bitmap is
    // already in that orientation, so applying EXIF a second time would be
    // incorrect.
    const resized = await createImageBitmap(image, {
      resizeWidth: width,
      resizeHeight: height,
      resizeQuality: 'high',
    });
    try {
      imageDimensions(resized);
    } catch (error) {
      if (resized !== image) {
        closeImage(resized);
      }
      throw error;
    }
    if (owned && resized !== image) {
      closeImage(image);
    }
    return { image: resized, owned: owned || resized !== image };
  } catch (error) {
    const detail = error instanceof Error ? ` (${error.message})` : '';
    throw new Error(`Could not resize the image for inference${detail}`, { cause: error });
  }
}

type PixelCanvas = OffscreenCanvas | HTMLCanvasElement;
type PixelContext = OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;

function createPixelCanvas(width: number, height: number): PixelCanvas {
  if (typeof OffscreenCanvas !== 'undefined') {
    return new OffscreenCanvas(width, height);
  }
  if (typeof document !== 'undefined') {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    return canvas;
  }
  throw new Error(
    'Reading image pixels requires OffscreenCanvas or document.createElement("canvas"); both are unavailable.',
  );
}

function readRgbTensor(image: ImageBitmap, width: number, height: number): Float32Array {
  const canvas = createPixelCanvas(width, height);
  const context = canvas.getContext('2d', { willReadFrequently: true }) as PixelContext | null;
  if (!context) {
    throw new Error('The browser could not create a 2D canvas context for image preprocessing.');
  }

  context.drawImage(image, 0, 0, width, height);
  const rgba = context.getImageData(0, 0, width, height).data;
  const pixelCount = width * height;
  if (rgba.length !== pixelCount * 4) {
    throw new Error(
      `Canvas returned ${rgba.length} bytes for ${width}x${height}; expected ${pixelCount * 4}.`,
    );
  }

  // MoGe expects RGB values in [0, 1], with the batch dimension omitted here
  // and channels laid out as contiguous row-major planes (CHW).
  const data = new Float32Array(pixelCount * 3);
  const redOffset = 0;
  const greenOffset = pixelCount;
  const blueOffset = pixelCount * 2;
  for (let pixel = 0; pixel < pixelCount; pixel += 1) {
    const rgbaOffset = pixel * 4;
    data[redOffset + pixel] = rgba[rgbaOffset]! / 255;
    data[greenOffset + pixel] = rgba[rgbaOffset + 1]! / 255;
    data[blueOffset + pixel] = rgba[rgbaOffset + 2]! / 255;
  }
  return data;
}

/**
 * Decode and resize an image locally, then convert its pixels to MoGe's RGB
 * float32 NCHW input. No image bytes are uploaded by this function.
 */
export async function preprocessImage(source: MoGeImageSource): Promise<PreprocessedImage> {
  if (source === null || typeof source !== 'object') {
    throw new Error('preprocessImage expects a Blob/File or ImageBitmap.');
  }

  const decoded = await decodeImage(source);
  let inferenceImage = decoded.image;
  let owned = decoded.owned;

  try {
    const sourceSize = imageDimensions(inferenceImage);
    const targetSize = inferenceDimensions(sourceSize.width, sourceSize.height);
    const resized = await resizeImage(
      inferenceImage,
      targetSize.width,
      targetSize.height,
      owned,
    );
    inferenceImage = resized.image;
    owned = resized.owned;

    const data = readRgbTensor(inferenceImage, targetSize.width, targetSize.height);
    return {
      width: targetSize.width,
      height: targetSize.height,
      data,
      tensor: data,
      imageTensor: data,
      tensorShape: [1, 3, targetSize.height, targetSize.width],
      inferenceImage,
      ownsInferenceImage: owned,
    };
  } catch (error) {
    if (owned) {
      closeImage(inferenceImage);
    }
    throw error;
  }
}
