import { z } from 'zod';
import {
  formatZodErrors,
  getFirstError,
  validateOrThrow,
  safeValidate,
  isValid,
  getFieldErrors,
  createFieldValidator,
} from '@/lib/utils/validation';
import { ValidationError } from '@/lib/errors';

describe('Validation Utilities', () => {
  describe('formatZodErrors', () => {
    it('should format Zod errors into grouped format', () => {
      const schema = z.object({
        email: z.string().email('Invalid email'),
        password: z.string().min(8, 'Password too short'),
      });

      const result = schema.safeParse({
        email: 'invalid',
        password: 'short',
      });

      if (!result.success) {
        const formatted = formatZodErrors(result.error);

        expect(formatted).toHaveProperty('email');
        expect(formatted).toHaveProperty('password');
        expect(formatted.email).toContain('Invalid email');
        expect(formatted.password).toContain('Password too short');
      }
    });

    it('should handle nested paths', () => {
      const schema = z.object({
        user: z.object({
          email: z.string().email('Invalid email'),
        }),
      });

      const result = schema.safeParse({
        user: {
          email: 'invalid',
        },
      });

      if (!result.success) {
        const formatted = formatZodErrors(result.error);

        expect(formatted).toHaveProperty('user.email');
        expect(formatted['user.email']).toContain('Invalid email');
      }
    });

    it('should handle multiple errors for same field', () => {
      const schema = z.object({
        email: z.string().email('Invalid email').min(5, 'Too short'),
      });

      const result = schema.safeParse({
        email: 'a',
      });

      if (!result.success) {
        const formatted = formatZodErrors(result.error);

        expect(formatted.email.length).toBeGreaterThan(0);
      }
    });
  });

  describe('getFirstError', () => {
    it('should return first error from array', () => {
      const errors = ['Error 1', 'Error 2', 'Error 3'];
      expect(getFirstError(errors)).toBe('Error 1');
    });

    it('should return undefined for empty array', () => {
      expect(getFirstError([])).toBeUndefined();
    });

    it('should return undefined for undefined input', () => {
      expect(getFirstError(undefined)).toBeUndefined();
    });
  });

  describe('validateOrThrow', () => {
    const schema = z.object({
      email: z.string().email('Invalid email'),
      age: z.number().min(18, 'Must be 18+'),
    });

    it('should return validated data when valid', () => {
      const data = {
        email: 'test@example.com',
        age: 25,
      };

      const result = validateOrThrow(schema, data);
      expect(result).toEqual(data);
    });

    it('should throw ValidationError when invalid', () => {
      const data = {
        email: 'invalid',
        age: 15,
      };

      expect(() => validateOrThrow(schema, data)).toThrow(ValidationError);
    });

    it('should include field name in error when provided', () => {
      const data = {
        email: 'invalid',
      };

      try {
        validateOrThrow(schema, data, 'user');
      } catch (error) {
        expect(error).toBeInstanceOf(ValidationError);
        if (error instanceof ValidationError) {
          expect(error.field).toBe('user');
        }
      }
    });

    it('should include formatted errors in ValidationError', () => {
      const data = {
        email: 'invalid',
        age: 15,
      };

      try {
        validateOrThrow(schema, data);
      } catch (error) {
        expect(error).toBeInstanceOf(ValidationError);
        if (error instanceof ValidationError) {
          expect(error.errors).toBeDefined();
          expect(Object.keys(error.errors || {})).toContain('email');
        }
      }
    });

    it('should throw original error if not ZodError', () => {
      const schema = z.object({
        email: z.string().email(),
      });

      // Mock schema.parse to throw non-ZodError
      const originalParse = schema.parse;
      schema.parse = jest.fn(() => {
        throw new Error('Custom error');
      });

      expect(() => validateOrThrow(schema, { email: 'test@example.com' })).toThrow(
        'Custom error'
      );

      // Restore
      schema.parse = originalParse;
    });
  });

  describe('safeValidate', () => {
    const schema = z.object({
      email: z.string().email(),
      age: z.number(),
    });

    it('should return success with data when valid', () => {
      const data = {
        email: 'test@example.com',
        age: 25,
      };

      const result = safeValidate(schema, data);

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data).toEqual(data);
      }
    });

    it('should return failure with errors when invalid', () => {
      const data = {
        email: 'invalid',
        age: 'not a number',
      };

      const result = safeValidate(schema, data);

      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.errors).toBeInstanceOf(z.ZodError);
      }
    });
  });

  describe('isValid', () => {
    const schema = z.object({
      email: z.string().email(),
    });

    it('should return true for valid data', () => {
      const data = { email: 'test@example.com' };
      expect(isValid(schema, data)).toBe(true);
    });

    it('should return false for invalid data', () => {
      const data = { email: 'invalid' };
      expect(isValid(schema, data)).toBe(false);
    });

    it('should work as type guard', () => {
      const schema = z.object({
        name: z.string(),
        age: z.number(),
      });

      const data: unknown = { name: 'Test', age: 25 };

      if (isValid(schema, data)) {
        // TypeScript should know data is { name: string; age: number }
        expect(data.name).toBe('Test');
        expect(data.age).toBe(25);
      }
    });
  });

  describe('getFieldErrors', () => {
    const schema = z.object({
      user: z.object({
        email: z.string().email('Invalid email'),
        profile: z.object({
          name: z.string().min(3, 'Name too short'),
        }),
      }),
    });

    it('should get errors from ZodError with string path', () => {
      const result = schema.safeParse({
        user: {
          email: 'invalid',
          profile: {
            name: 'ab',
          },
        },
      });

      if (!result.success) {
        const emailErrors = getFieldErrors(result.error, 'user.email');
        expect(emailErrors).toContain('Invalid email');

        const nameErrors = getFieldErrors(result.error, 'user.profile.name');
        expect(nameErrors).toContain('Name too short');
      }
    });

    it('should get errors from ZodError with array path', () => {
      const result = schema.safeParse({
        user: {
          email: 'invalid',
        },
      });

      if (!result.success) {
        const emailErrors = getFieldErrors(result.error, ['user', 'email']);
        expect(emailErrors).toContain('Invalid email');
      }
    });

    it('should get errors from formatted errors object with string path', () => {
      const formattedErrors = {
        'user.email': ['Invalid email'],
        'user.profile.name': ['Name too short'],
      };

      const emailErrors = getFieldErrors(formattedErrors, 'user.email');
      expect(emailErrors).toEqual(['Invalid email']);

      const nameErrors = getFieldErrors(formattedErrors, 'user.profile.name');
      expect(nameErrors).toEqual(['Name too short']);
    });

    it('should get errors from formatted errors object with array path', () => {
      const formattedErrors = {
        'user.email': ['Invalid email'],
      };

      const emailErrors = getFieldErrors(formattedErrors, ['user', 'email']);
      expect(emailErrors).toEqual(['Invalid email']);
    });

    it('should return empty array for non-existent field', () => {
      const formattedErrors = {
        'user.email': ['Invalid email'],
      };

      const errors = getFieldErrors(formattedErrors, 'user.password');
      expect(errors).toEqual([]);
    });
  });

  describe('createFieldValidator', () => {
    it('should create validator that returns valid for correct value', () => {
      const schema = z.string().email();
      const validator = createFieldValidator(schema);

      const result = validator('test@example.com');
      expect(result.valid).toBe(true);
      expect(result.error).toBeUndefined();
    });

    it('should create validator that returns invalid for incorrect value', () => {
      const schema = z.string().email();
      const validator = createFieldValidator(schema);

      const result = validator('invalid-email');
      expect(result.valid).toBe(false);
      expect(result.error).toBeDefined();
    });

    it('should return first error message', () => {
      const schema = z.string().min(5, 'Too short').email('Invalid email');
      const validator = createFieldValidator(schema);

      const result = validator('a');
      expect(result.valid).toBe(false);
      expect(result.error).toBe('Too short');
    });

    it('should work with number schema', () => {
      const schema = z.number().min(18);
      const validator = createFieldValidator(schema);

      expect(validator(25).valid).toBe(true);
      expect(validator(15).valid).toBe(false);
    });

    it('should work with object schema', () => {
      const schema = z.object({
        name: z.string().min(3),
        age: z.number().min(18),
      });
      const validator = createFieldValidator(schema);

      expect(
        validator({ name: 'John Doe', age: 25 }).valid
      ).toBe(true);
      expect(validator({ name: 'Jo', age: 15 }).valid).toBe(false);
    });
  });
});

