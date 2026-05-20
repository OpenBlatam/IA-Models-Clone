import {
  sanitizeInput,
  isValidUrl,
  sanitizeFileName,
  isValidEmail,
  containsDangerousContent,
} from '@/utils/security';

describe('Security Utils', () => {
  describe('sanitizeInput', () => {
    it('should remove HTML tags', () => {
      expect(sanitizeInput('<script>alert("xss")</script>')).toBe('scriptalert("xss")/script');
    });

    it('should remove javascript protocol', () => {
      expect(sanitizeInput('javascript:alert(1)')).toBe('alert(1)');
    });

    it('should trim whitespace', () => {
      expect(sanitizeInput('  hello  ')).toBe('hello');
    });
  });

  describe('isValidUrl', () => {
    it('should validate HTTP URLs', () => {
      expect(isValidUrl('http://example.com')).toBe(true);
    });

    it('should validate HTTPS URLs', () => {
      expect(isValidUrl('https://example.com')).toBe(true);
    });

    it('should reject javascript URLs', () => {
      expect(isValidUrl('javascript:alert(1)')).toBe(false);
    });

    it('should reject invalid URLs', () => {
      expect(isValidUrl('not-a-url')).toBe(false);
    });
  });

  describe('sanitizeFileName', () => {
    it('should replace special characters', () => {
      expect(sanitizeFileName('file<>name.txt')).toBe('file__name.txt');
    });

    it('should limit length', () => {
      const longName = 'a'.repeat(300);
      expect(sanitizeFileName(longName).length).toBeLessThanOrEqual(255);
    });
  });

  describe('isValidEmail', () => {
    it('should validate correct emails', () => {
      expect(isValidEmail('test@example.com')).toBe(true);
    });

    it('should reject invalid emails', () => {
      expect(isValidEmail('not-an-email')).toBe(false);
      expect(isValidEmail('test@')).toBe(false);
      expect(isValidEmail('@example.com')).toBe(false);
    });
  });

  describe('containsDangerousContent', () => {
    it('should detect script tags', () => {
      expect(containsDangerousContent('<script>alert(1)</script>')).toBe(true);
    });

    it('should detect javascript protocol', () => {
      expect(containsDangerousContent('javascript:alert(1)')).toBe(true);
    });

    it('should detect event handlers', () => {
      expect(containsDangerousContent('onclick="alert(1)"')).toBe(true);
    });

    it('should return false for safe content', () => {
      expect(containsDangerousContent('Hello world')).toBe(false);
    });
  });
});


