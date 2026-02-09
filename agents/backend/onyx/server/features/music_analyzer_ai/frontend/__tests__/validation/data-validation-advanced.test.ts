/**
 * Advanced Data Validation Testing
 * 
 * Comprehensive tests for data validation including
 * schema validation, type checking, and data sanitization.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { z } from 'zod';

describe('Advanced Data Validation Testing', () => {
  describe('Schema Validation', () => {
    it('should validate complex nested schemas', () => {
      const TrackSchema = z.object({
        id: z.string().uuid(),
        name: z.string().min(1).max(200),
        artists: z.array(z.string()).min(1),
        duration: z.number().positive(),
        metadata: z.object({
          genre: z.string().optional(),
          year: z.number().int().positive().optional(),
        }).optional(),
      });

      const validTrack = {
        id: '123e4567-e89b-12d3-a456-426614174000',
        name: 'Test Track',
        artists: ['Artist 1'],
        duration: 180,
        metadata: { genre: 'Rock', year: 2023 },
      };

      expect(() => TrackSchema.parse(validTrack)).not.toThrow();
    });

    it('should validate with custom validators', () => {
      const EmailSchema = z.string().email().refine(
        (email) => !email.includes('+'),
        { message: 'Email cannot contain + character' }
      );

      expect(() => EmailSchema.parse('user@example.com')).not.toThrow();
      expect(() => EmailSchema.parse('user+tag@example.com')).toThrow();
    });

    it('should validate with conditional logic', () => {
      const ConditionalSchema = z.object({
        type: z.enum(['track', 'playlist']),
        data: z.union([
          z.object({ trackId: z.string() }),
          z.object({ playlistId: z.string() }),
        ]),
      }).refine(
        (data) => {
          if (data.type === 'track') {
            return 'trackId' in data.data;
          }
          return 'playlistId' in data.data;
        },
        { message: 'Data type must match schema type' }
      );

      const valid = {
        type: 'track' as const,
        data: { trackId: '123' },
      };

      expect(() => ConditionalSchema.parse(valid)).not.toThrow();
    });
  });

  describe('Type Checking', () => {
    it('should validate primitive types', () => {
      const validateType = (value: any, expectedType: string) => {
        const actualType = typeof value;
        if (expectedType === 'array') {
          return Array.isArray(value);
        }
        return actualType === expectedType;
      };

      expect(validateType('string', 'string')).toBe(true);
      expect(validateType(123, 'number')).toBe(true);
      expect(validateType(true, 'boolean')).toBe(true);
      expect(validateType([], 'array')).toBe(true);
      expect(validateType({}, 'object')).toBe(true);
    });

    it('should validate enum values', () => {
      const validateEnum = (value: string, allowedValues: string[]) => {
        return allowedValues.includes(value);
      };

      const genres = ['rock', 'pop', 'jazz', 'classical'];
      expect(validateEnum('rock', genres)).toBe(true);
      expect(validateEnum('metal', genres)).toBe(false);
    });

    it('should validate date formats', () => {
      const validateDate = (dateString: string) => {
        const date = new Date(dateString);
        return !isNaN(date.getTime());
      };

      expect(validateDate('2023-12-25')).toBe(true);
      expect(validateDate('2023-12-25T10:30:00Z')).toBe(true);
      expect(validateDate('invalid-date')).toBe(false);
    });
  });

  describe('Data Sanitization', () => {
    it('should sanitize HTML content', () => {
      const sanitizeHTML = (html: string) => {
        return html
          .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
          .replace(/<iframe\b[^<]*(?:(?!<\/iframe>)<[^<]*)*<\/iframe>/gi, '')
          .replace(/on\w+="[^"]*"/gi, '');
      };

      const malicious = '<script>alert("xss")</script><p>Safe content</p>';
      const sanitized = sanitizeHTML(malicious);
      expect(sanitized).not.toContain('<script>');
      expect(sanitized).toContain('Safe content');
    });

    it('should trim whitespace', () => {
      const sanitize = (input: string) => {
        return input.trim().replace(/\s+/g, ' ');
      };

      expect(sanitize('  test   string  ')).toBe('test string');
    });

    it('should remove null bytes', () => {
      const sanitize = (input: string) => {
        return input.replace(/\0/g, '');
      };

      const withNull = 'test\0string';
      const sanitized = sanitize(withNull);
      expect(sanitized).not.toContain('\0');
    });
  });

  describe('Range Validation', () => {
    it('should validate number ranges', () => {
      const validateRange = (value: number, min: number, max: number) => {
        return value >= min && value <= max;
      };

      expect(validateRange(50, 0, 100)).toBe(true);
      expect(validateRange(150, 0, 100)).toBe(false);
      expect(validateRange(-10, 0, 100)).toBe(false);
    });

    it('should validate string length', () => {
      const validateLength = (str: string, min: number, max: number) => {
        return str.length >= min && str.length <= max;
      };

      expect(validateLength('test', 1, 10)).toBe(true);
      expect(validateLength('', 1, 10)).toBe(false);
      expect(validateLength('x'.repeat(100), 1, 10)).toBe(false);
    });
  });

  describe('Format Validation', () => {
    it('should validate UUID format', () => {
      const validateUUID = (uuid: string) => {
        const regex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
        return regex.test(uuid);
      };

      expect(validateUUID('123e4567-e89b-12d3-a456-426614174000')).toBe(true);
      expect(validateUUID('invalid-uuid')).toBe(false);
    });

    it('should validate URL format', () => {
      const validateURL = (url: string) => {
        try {
          new URL(url);
          return true;
        } catch {
          return false;
        }
      };

      expect(validateURL('https://example.com')).toBe(true);
      expect(validateURL('not-a-url')).toBe(false);
    });

    it('should validate phone number format', () => {
      const validatePhone = (phone: string) => {
        const regex = /^\+?[\d\s-()]+$/;
        return regex.test(phone) && phone.replace(/\D/g, '').length >= 10;
      };

      expect(validatePhone('+1-555-123-4567')).toBe(true);
      expect(validatePhone('123')).toBe(false);
    });
  });

  describe('Cross-Field Validation', () => {
    it('should validate dependent fields', () => {
      const validateDependent = (data: any) => {
        if (data.hasPassword && !data.password) {
          return { valid: false, error: 'Password required' };
        }
        if (data.password && data.password.length < 8) {
          return { valid: false, error: 'Password too short' };
        }
        return { valid: true };
      };

      expect(validateDependent({ hasPassword: true, password: 'short' }).valid).toBe(false);
      expect(validateDependent({ hasPassword: true, password: 'longpassword' }).valid).toBe(true);
    });

    it('should validate date ranges', () => {
      const validateDateRange = (start: string, end: string) => {
        const startDate = new Date(start);
        const endDate = new Date(end);
        return endDate >= startDate;
      };

      expect(validateDateRange('2023-01-01', '2023-12-31')).toBe(true);
      expect(validateDateRange('2023-12-31', '2023-01-01')).toBe(false);
    });
  });

  describe('Array Validation', () => {
    it('should validate array length', () => {
      const validateArrayLength = (arr: any[], min: number, max: number) => {
        return arr.length >= min && arr.length <= max;
      };

      expect(validateArrayLength([1, 2, 3], 1, 10)).toBe(true);
      expect(validateArrayLength([], 1, 10)).toBe(false);
      expect(validateArrayLength(Array(20).fill(0), 1, 10)).toBe(false);
    });

    it('should validate unique array values', () => {
      const validateUnique = (arr: any[]) => {
        return arr.length === new Set(arr).size;
      };

      expect(validateUnique([1, 2, 3])).toBe(true);
      expect(validateUnique([1, 2, 2])).toBe(false);
    });

    it('should validate array item types', () => {
      const validateArrayTypes = (arr: any[], expectedType: string) => {
        return arr.every(item => typeof item === expectedType);
      };

      expect(validateArrayTypes([1, 2, 3], 'number')).toBe(true);
      expect(validateArrayTypes([1, '2', 3], 'number')).toBe(false);
    });
  });
});

