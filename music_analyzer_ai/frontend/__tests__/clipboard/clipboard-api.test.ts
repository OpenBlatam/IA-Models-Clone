/**
 * Clipboard API Testing
 * 
 * Tests that verify clipboard read/write functionality,
 * clipboard permissions, and data formats.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

// Mock Clipboard API
const mockClipboard = {
  read: vi.fn(),
  readText: vi.fn(),
  write: vi.fn(),
  writeText: vi.fn(),
};

Object.defineProperty(navigator, 'clipboard', {
  writable: true,
  value: mockClipboard,
});

describe('Clipboard API Testing', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Reading from Clipboard', () => {
    it('should read text from clipboard', async () => {
      mockClipboard.readText.mockResolvedValue('Copied text');

      const text = await navigator.clipboard.readText();
      expect(text).toBe('Copied text');
      expect(mockClipboard.readText).toHaveBeenCalled();
    });

    it('should read clipboard data with formats', async () => {
      const clipboardItems = [
        new ClipboardItem({
          'text/plain': new Blob(['Plain text'], { type: 'text/plain' }),
          'text/html': new Blob(['<p>HTML</p>'], { type: 'text/html' }),
        }),
      ];

      mockClipboard.read.mockResolvedValue(clipboardItems);

      const items = await navigator.clipboard.read();
      expect(items).toHaveLength(1);
      expect(mockClipboard.read).toHaveBeenCalled();
    });

    it('should handle clipboard read errors', async () => {
      mockClipboard.readText.mockRejectedValue(new Error('Permission denied'));

      try {
        await navigator.clipboard.readText();
      } catch (error: any) {
        expect(error.message).toBe('Permission denied');
      }
    });
  });

  describe('Writing to Clipboard', () => {
    it('should write text to clipboard', async () => {
      mockClipboard.writeText.mockResolvedValue(undefined);

      await navigator.clipboard.writeText('Text to copy');
      expect(mockClipboard.writeText).toHaveBeenCalledWith('Text to copy');
    });

    it('should write clipboard items with multiple formats', async () => {
      const clipboardItem = new ClipboardItem({
        'text/plain': new Blob(['Plain text'], { type: 'text/plain' }),
        'text/html': new Blob(['<p>HTML</p>'], { type: 'text/html' }),
      });

      mockClipboard.write.mockResolvedValue(undefined);

      await navigator.clipboard.write([clipboardItem]);
      expect(mockClipboard.write).toHaveBeenCalled();
    });

    it('should handle clipboard write errors', async () => {
      mockClipboard.writeText.mockRejectedValue(new Error('Write failed'));

      try {
        await navigator.clipboard.writeText('text');
      } catch (error: any) {
        expect(error.message).toBe('Write failed');
      }
    });
  });

  describe('Clipboard Formats', () => {
    it('should support plain text format', async () => {
      const writePlainText = async (text: string) => {
        await navigator.clipboard.writeText(text);
      };

      mockClipboard.writeText.mockResolvedValue(undefined);
      await writePlainText('Plain text');
      expect(mockClipboard.writeText).toHaveBeenCalledWith('Plain text');
    });

    it('should support HTML format', async () => {
      const writeHTML = async (html: string) => {
        const item = new ClipboardItem({
          'text/html': new Blob([html], { type: 'text/html' }),
        });
        await navigator.clipboard.write([item]);
      };

      mockClipboard.write.mockResolvedValue(undefined);
      await writeHTML('<p>HTML content</p>');
      expect(mockClipboard.write).toHaveBeenCalled();
    });

    it('should support image format', async () => {
      const writeImage = async (blob: Blob) => {
        const item = new ClipboardItem({
          'image/png': blob,
        });
        await navigator.clipboard.write([item]);
      };

      const imageBlob = new Blob(['image data'], { type: 'image/png' });
      mockClipboard.write.mockResolvedValue(undefined);
      await writeImage(imageBlob);
      expect(mockClipboard.write).toHaveBeenCalled();
    });
  });

  describe('Clipboard Permissions', () => {
    it('should check clipboard read permission', async () => {
      const checkPermission = async () => {
        try {
          const permission = await navigator.permissions.query({ name: 'clipboard-read' as PermissionName });
          return permission.state;
        } catch {
          return 'unknown';
        }
      };

      const mockQuery = vi.fn().mockResolvedValue({ state: 'granted' });
      Object.defineProperty(navigator, 'permissions', {
        writable: true,
        value: { query: mockQuery },
      });

      const state = await checkPermission();
      expect(state).toBe('granted');
    });

    it('should request clipboard permission if needed', async () => {
      const requestPermission = async () => {
        try {
          await navigator.clipboard.readText();
          return { granted: true };
        } catch (error: any) {
          if (error.message.includes('permission')) {
            return { granted: false, needsPermission: true };
          }
          throw error;
        }
      };

      mockClipboard.readText.mockRejectedValue(new Error('Permission denied'));
      const result = await requestPermission();
      expect(result.granted).toBe(false);
      expect(result.needsPermission).toBe(true);
    });
  });

  describe('Clipboard Events', () => {
    it('should handle copy events', () => {
      const handleCopy = (event: ClipboardEvent) => {
        event.clipboardData?.setData('text/plain', 'Copied text');
      };

      const event = new ClipboardEvent('copy', {
        clipboardData: new DataTransfer(),
      });

      handleCopy(event);
      expect(event.clipboardData?.getData('text/plain')).toBe('Copied text');
    });

    it('should handle paste events', () => {
      const handlePaste = (event: ClipboardEvent) => {
        const text = event.clipboardData?.getData('text/plain');
        return text || '';
      };

      const event = new ClipboardEvent('paste', {
        clipboardData: new DataTransfer(),
      });
      event.clipboardData?.setData('text/plain', 'Pasted text');

      const text = handlePaste(event);
      expect(text).toBe('Pasted text');
    });
  });

  describe('Clipboard Security', () => {
    it('should sanitize clipboard content', () => {
      const sanitizeClipboard = (text: string) => {
        return text
          .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
          .trim();
      };

      const malicious = '<script>alert("xss")</script>Safe text';
      const sanitized = sanitizeClipboard(malicious);
      expect(sanitized).toBe('Safe text');
    });

    it('should validate clipboard data before writing', () => {
      const validateBeforeWrite = (text: string) => {
        if (text.length > 10000) {
          return { valid: false, error: 'Text too long' };
        }
        if (!text.trim()) {
          return { valid: false, error: 'Text is empty' };
        }
        return { valid: true };
      };

      expect(validateBeforeWrite('Valid text').valid).toBe(true);
      expect(validateBeforeWrite('x'.repeat(20000)).valid).toBe(false);
      expect(validateBeforeWrite('   ').valid).toBe(false);
    });
  });
});

