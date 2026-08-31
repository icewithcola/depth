import { describe, expect, it, vi } from 'vitest';

import {
  ensureNamedFile,
  extractImageFileFromClipboardData,
  isEditableTarget,
  readImageFromClipboard,
} from './clipboard';

describe('clipboard utilities', () => {
  describe('ensureNamedFile', () => {
    it('preserves valid filenames with supported extensions', () => {
      const file = new File(['content'], 'sample.png', { type: 'image/png' });
      const result = ensureNamedFile(file);
      expect(result.name).toBe('sample.png');
      expect(result.type).toBe('image/png');
    });

    it('adds appropriate extension when filename has no extension', () => {
      const filePng = new File(['content'], 'pasted-image', { type: 'image/png' });
      expect(ensureNamedFile(filePng).name).toBe('pasted-image.png');

      const fileJpg = new File(['content'], 'photo', { type: 'image/jpeg' });
      expect(ensureNamedFile(fileJpg).name).toBe('photo.jpg');

      const fileWebp = new File(['content'], 'graphic', { type: 'image/webp' });
      expect(ensureNamedFile(fileWebp).name).toBe('graphic.webp');
    });

    it('generates a default clipboard name when filename is empty', () => {
      const filePng = new File(['content'], '', { type: 'image/png' });
      expect(ensureNamedFile(filePng).name).toBe('clipboard.png');

      const fileJpg = new File(['content'], '', { type: 'image/jpeg' });
      expect(ensureNamedFile(fileJpg).name).toBe('clipboard.jpg');
    });
  });

  describe('extractImageFileFromClipboardData', () => {
    it('returns null for null or undefined clipboardData', () => {
      expect(extractImageFileFromClipboardData(null)).toBeNull();
      expect(extractImageFileFromClipboardData(undefined)).toBeNull();
    });

    it('extracts image from files list', () => {
      const file = new File(['test'], 'image.png', { type: 'image/png' });
      const clipboardData = {
        files: [file],
        items: [],
      } as unknown as DataTransfer;

      const extracted = extractImageFileFromClipboardData(clipboardData);
      expect(extracted).not.toBeNull();
      expect(extracted?.name).toBe('image.png');
      expect(extracted?.type).toBe('image/png');
    });

    it('extracts image from items list when files list is empty', () => {
      const file = new File(['test'], 'pasted.jpg', { type: 'image/jpeg' });
      const item = {
        kind: 'file',
        type: 'image/jpeg',
        getAsFile: () => file,
      };
      const clipboardData = {
        files: [],
        items: [item],
      } as unknown as DataTransfer;

      const extracted = extractImageFileFromClipboardData(clipboardData);
      expect(extracted).not.toBeNull();
      expect(extracted?.name).toBe('pasted.jpg');
      expect(extracted?.type).toBe('image/jpeg');
    });

    it('returns null when clipboard only contains text', () => {
      const item = {
        kind: 'string',
        type: 'text/plain',
        getAsFile: () => null,
      };
      const clipboardData = {
        files: [],
        items: [item],
      } as unknown as DataTransfer;

      expect(extractImageFileFromClipboardData(clipboardData)).toBeNull();
    });
  });

  describe('readImageFromClipboard', () => {
    it('throws when clipboard or clipboard.read is unsupported', async () => {
      await expect(readImageFromClipboard({} as Clipboard)).rejects.toThrow(
        /Clipboard reading is not supported/,
      );
    });

    it('reads an image file from clipboard items', async () => {
      const blob = new Blob(['image-bytes'], { type: 'image/png' });
      const fakeClipboardItem = {
        types: ['image/png'],
        getType: vi.fn(async (type: string) => (type === 'image/png' ? blob : new Blob())),
      };
      const mockClipboard = {
        read: vi.fn(async () => [fakeClipboardItem]),
      } as unknown as Clipboard;

      const result = await readImageFromClipboard(mockClipboard);
      expect(result).not.toBeNull();
      expect(result?.name).toBe('clipboard.png');
      expect(result?.type).toBe('image/png');
      expect(fakeClipboardItem.getType).toHaveBeenCalledWith('image/png');
    });

    it('returns null when clipboard contains no images', async () => {
      const fakeClipboardItem = {
        types: ['text/plain'],
        getType: vi.fn(async () => new Blob(['hello'], { type: 'text/plain' })),
      };
      const mockClipboard = {
        read: vi.fn(async () => [fakeClipboardItem]),
      } as unknown as Clipboard;

      const result = await readImageFromClipboard(mockClipboard);
      expect(result).toBeNull();
    });
  });

  describe('isEditableTarget', () => {
    it('returns false for null or undefined target', () => {
      expect(isEditableTarget(null)).toBe(false);
    });

    it('returns true for contenteditable elements', () => {
      const element = {
        isContentEditable: true,
        tagName: 'DIV',
      } as unknown as HTMLElement;
      expect(isEditableTarget(element)).toBe(true);
    });

    it('returns true for text input and false for buttons/checkboxes', () => {
      const textInput = {
        isContentEditable: false,
        tagName: 'INPUT',
        type: 'text',
      } as unknown as HTMLInputElement;
      expect(isEditableTarget(textInput)).toBe(true);

      const checkbox = {
        isContentEditable: false,
        tagName: 'INPUT',
        type: 'checkbox',
      } as unknown as HTMLInputElement;
      expect(isEditableTarget(checkbox)).toBe(false);
    });

    it('returns true for writable textarea and false for readonly textarea', () => {
      const writableTextarea = {
        isContentEditable: false,
        tagName: 'TEXTAREA',
        readOnly: false,
      } as unknown as HTMLTextAreaElement;
      expect(isEditableTarget(writableTextarea)).toBe(true);

      const readonlyTextarea = {
        isContentEditable: false,
        tagName: 'TEXTAREA',
        readOnly: true,
      } as unknown as HTMLTextAreaElement;
      expect(isEditableTarget(readonlyTextarea)).toBe(false);
    });
  });
});
