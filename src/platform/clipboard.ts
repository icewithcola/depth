export const SUPPORTED_IMAGE_TYPES: ReadonlySet<string> = new Set([
  'image/jpeg',
  'image/png',
  'image/webp',
]);

const SUPPORTED_EXTENSION_REGEX = /\.(?:jpe?g|png|webp)$/i;

function mimeToExtension(mime: string): string {
  switch (mime.toLowerCase()) {
    case 'image/jpeg':
    case 'image/jpg':
      return 'jpg';
    case 'image/webp':
      return 'webp';
    case 'image/png':
    default:
      return 'png';
  }
}

/**
 * Ensures an image File has a non-empty name with an appropriate file extension.
 */
export function ensureNamedFile(file: File): File {
  const mime = file.type.trim().toLowerCase();
  const rawName = file.name ? file.name.trim() : '';
  const hasValidExt = SUPPORTED_EXTENSION_REGEX.test(rawName);

  if (hasValidExt && rawName.length > 0) {
    return file;
  }

  const ext = mimeToExtension(mime);
  const baseName = rawName.length > 0 ? rawName : 'clipboard';
  const finalName = baseName.endsWith(`.${ext}`) ? baseName : `${baseName}.${ext}`;
  const finalType = file.type || `image/${ext === 'jpg' ? 'jpeg' : ext}`;

  return new File([file], finalName, { type: finalType });
}

function isImageMimeOrFile(mime: string, name?: string): boolean {
  const normalizedMime = mime.trim().toLowerCase();
  if (SUPPORTED_IMAGE_TYPES.has(normalizedMime) || normalizedMime.startsWith('image/')) {
    return true;
  }
  if (name && SUPPORTED_EXTENSION_REGEX.test(name)) {
    return true;
  }
  return false;
}

/**
 * Extracts the first image File from a DataTransfer object (from paste or drop events).
 */
export function extractImageFileFromClipboardData(
  clipboardData: DataTransfer | null | undefined,
): File | null {
  if (!clipboardData) {
    return null;
  }

  // Check files list first
  if (clipboardData.files && clipboardData.files.length > 0) {
    for (let i = 0; i < clipboardData.files.length; i++) {
      const file = clipboardData.files[i];
      if (file && isImageMimeOrFile(file.type, file.name)) {
        return ensureNamedFile(file);
      }
    }
  }

  // Fall back to items list
  if (clipboardData.items && clipboardData.items.length > 0) {
    for (let i = 0; i < clipboardData.items.length; i++) {
      const item = clipboardData.items[i];
      if (item && item.kind === 'file' && isImageMimeOrFile(item.type)) {
        const file = item.getAsFile();
        if (file) {
          return ensureNamedFile(file);
        }
      }
    }
  }

  return null;
}

/**
 * Reads an image from the system clipboard using the Async Clipboard API (`navigator.clipboard.read()`).
 */
export async function readImageFromClipboard(
  clipboard?: Clipboard,
): Promise<File | null> {
  const cb = clipboard ?? (typeof navigator !== 'undefined' ? navigator.clipboard : undefined);
  if (!cb || typeof cb.read !== 'function') {
    throw new Error('Clipboard reading is not supported by your browser.');
  }

  const items = await cb.read();
  for (const item of items) {
    for (const type of item.types) {
      if (isImageMimeOrFile(type)) {
        const blob = await item.getType(type);
        const mime = blob.type || type;
        const ext = mimeToExtension(mime);
        return new File([blob], `clipboard.${ext}`, { type: mime });
      }
    }
  }

  return null;
}

/**
 * Checks whether an event target is an interactive or editable text input that should
 * receive normal text paste behavior.
 */
export function isEditableTarget(target: EventTarget | null): boolean {
  if (!target) {
    return false;
  }
  const element = target as Partial<HTMLElement & HTMLInputElement & HTMLTextAreaElement>;
  if (element.isContentEditable) {
    return true;
  }
  const tagName = typeof element.tagName === 'string' ? element.tagName.toUpperCase() : '';
  if (tagName === 'INPUT') {
    const inputType = typeof element.type === 'string' ? element.type.toLowerCase() : '';
    return !['checkbox', 'radio', 'button', 'submit', 'reset', 'file', 'range', 'color'].includes(
      inputType,
    );
  }
  if (tagName === 'TEXTAREA') {
    return !element.readOnly;
  }
  return false;
}
