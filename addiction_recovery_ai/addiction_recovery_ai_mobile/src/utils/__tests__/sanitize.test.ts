import {
  sanitizeString,
  sanitizeEmail,
  sanitizeNumber,
  sanitizeUrl,
  escapeHtml,
} from '../sanitize';

describe('Sanitization Functions', () => {
  describe('sanitizeString', () => {
    it('removes HTML tags', () => {
      expect(sanitizeString('<script>alert("xss")</script>')).toBe('scriptalert("xss")script');
    });

    it('trims whitespace', () => {
      expect(sanitizeString('  test  ')).toBe('test');
    });

    it('removes javascript: protocol', () => {
      expect(sanitizeString('javascript:alert("xss")')).toBe('alert("xss")');
    });

    it('handles empty string', () => {
      expect(sanitizeString('')).toBe('');
    });
  });

  describe('sanitizeEmail', () => {
    it('converts to lowercase', () => {
      expect(sanitizeEmail('TEST@EXAMPLE.COM')).toBe('test@example.com');
    });

    it('trims whitespace', () => {
      expect(sanitizeEmail('  test@example.com  ')).toBe('test@example.com');
    });
  });

  describe('sanitizeNumber', () => {
    it('parses valid numbers', () => {
      expect(sanitizeNumber('123')).toBe(123);
      expect(sanitizeNumber('12.34')).toBe(12.34);
    });

    it('removes non-numeric characters', () => {
      expect(sanitizeNumber('$123.45')).toBe(123.45);
    });

    it('returns 0 for invalid input', () => {
      expect(sanitizeNumber('abc')).toBe(0);
      expect(sanitizeNumber('')).toBe(0);
    });

    it('handles number input', () => {
      expect(sanitizeNumber(123)).toBe(123);
    });
  });

  describe('sanitizeUrl', () => {
    it('allows http URLs', () => {
      expect(sanitizeUrl('http://example.com')).toBe('http://example.com');
    });

    it('allows https URLs', () => {
      expect(sanitizeUrl('https://example.com')).toBe('https://example.com');
    });

    it('rejects javascript: URLs', () => {
      expect(sanitizeUrl('javascript:alert("xss")')).toBe('');
    });

    it('returns empty for invalid URLs', () => {
      expect(sanitizeUrl('not a url')).toBe('');
    });
  });

  describe('escapeHtml', () => {
    it('escapes HTML entities', () => {
      expect(escapeHtml('<script>')).toBe('&lt;script&gt;');
      expect(escapeHtml('"quotes"')).toBe('&quot;quotes&quot;');
      expect(escapeHtml("'apostrophe'")).toBe('&#039;apostrophe&#039;');
      expect(escapeHtml('&amp;')).toBe('&amp;amp;');
    });
  });
});

