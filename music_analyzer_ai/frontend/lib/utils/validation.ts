/**
 * Validation utility functions.
 * Provides helper functions for common validation tasks.
 */

import { z } from 'zod';
import { getErrorMessage } from '@/lib/errors';

/**
 * Safely parses data with a Zod schema.
 * Returns the parsed data or throws a ValidationError.
 * @param schema - Zod schema to validate against
 * @param data - Data to validate
 * @returns Parsed and validated data
 * @throws ValidationError if validation fails
 */
export function safeParse<T extends z.ZodType>(
  schema: T,
  data: unknown
): z.infer<T> {
  const result = schema.safeParse(data);
  if (!result.success) {
    const errorMessages = result.error.errors
      .map((err) => `${err.path.join('.')}: ${err.message}`)
      .join(', ');
    throw new Error(`Validation failed: ${errorMessages}`);
  }
  return result.data;
}

/**
 * Validates data with a Zod schema and returns validation result.
 * @param schema - Zod schema to validate against
 * @param data - Data to validate
 * @returns Validation result with isValid flag and errors
 */
export function validateData<T extends z.ZodType>(
  schema: T,
  data: unknown
): {
  isValid: boolean;
  data?: z.infer<T>;
  errors: Record<string, string>;
} {
  const result = schema.safeParse(data);
  if (result.success) {
    return {
      isValid: true,
      data: result.data,
      errors: {},
    };
  }

  const errors: Record<string, string> = {};
  result.error.errors.forEach((err) => {
    const path = err.path.join('.');
    errors[path] = err.message;
  });

  return {
    isValid: false,
    errors,
  };
}

/**
 * Validates a single field with a Zod schema.
 * @param schema - Zod schema for the field
 * @param value - Value to validate
 * @returns Validation result
 */
export function validateField(
  schema: z.ZodType,
  value: unknown
): {
  isValid: boolean;
  error?: string;
} {
  const result = schema.safeParse(value);
  if (result.success) {
    return { isValid: true };
  }

  return {
    isValid: false,
    error: result.error.errors[0]?.message || 'Invalid value',
  };
}

/**
 * Creates a validation function from a Zod schema.
 * @param schema - Zod schema
 * @returns Validation function
 */
export function createValidator<T extends z.ZodType>(schema: T) {
  return (data: unknown): z.infer<T> => {
    return safeParse(schema, data);
  };
}

/**
 * Combines multiple validation results.
 * @param results - Array of validation results
 * @returns Combined validation result
 */
export function combineValidationResults(
  results: Array<{ isValid: boolean; errors: Record<string, string> }>
): {
  isValid: boolean;
  errors: Record<string, string>;
} {
  const allValid = results.every((r) => r.isValid);
  const allErrors: Record<string, string> = {};

  results.forEach((result) => {
    Object.assign(allErrors, result.errors);
  });

  return {
    isValid: allValid,
    errors: allErrors,
  };
}

/**
 * Validates and transforms data with error handling.
 * @param schema - Zod schema with transform
 * @param data - Data to validate and transform
 * @returns Transformed data or throws error
 */
export function validateAndTransform<T extends z.ZodType>(
  schema: T,
  data: unknown
): z.infer<T> {
  try {
    return safeParse(schema, data);
  } catch (error) {
    throw new Error(`Validation and transformation failed: ${getErrorMessage(error)}`);
  }
}
