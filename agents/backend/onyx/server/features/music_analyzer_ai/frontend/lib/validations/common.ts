/**
 * Common validation schemas using Zod.
 * Reusable validation schemas for common data types.
 */

import { z } from 'zod';

/**
 * Email validation schema.
 */
export const emailSchema = z
  .string()
  .email('Invalid email address')
  .min(5, 'Email must be at least 5 characters')
  .max(255, 'Email must be at most 255 characters')
  .toLowerCase()
  .trim();

/**
 * URL validation schema.
 */
export const urlSchema = z
  .string()
  .url('Invalid URL')
  .max(2048, 'URL must be at most 2048 characters');

/**
 * Non-empty string schema.
 */
export const nonEmptyStringSchema = z
  .string()
  .min(1, 'This field is required')
  .trim();

/**
 * Positive integer schema.
 */
export const positiveIntegerSchema = z
  .number()
  .int('Must be an integer')
  .positive('Must be a positive number');

/**
 * Non-negative integer schema.
 */
export const nonNegativeIntegerSchema = z
  .number()
  .int('Must be an integer')
  .nonnegative('Must be a non-negative number');

/**
 * Percentage schema (0-100).
 */
export const percentageSchema = z
  .number()
  .min(0, 'Percentage must be at least 0')
  .max(100, 'Percentage must be at most 100');

/**
 * Decimal schema (0-1).
 */
export const decimalSchema = z
  .number()
  .min(0, 'Value must be at least 0')
  .max(1, 'Value must be at most 1');

/**
 * Date string schema (ISO format).
 */
export const dateStringSchema = z.string().datetime('Invalid date format');

/**
 * Optional string that becomes empty string if undefined.
 */
export const optionalStringSchema = z
  .string()
  .optional()
  .transform((val) => val ?? '');

/**
 * Array with minimum length.
 */
export function arrayWithMinLength<T extends z.ZodTypeAny>(
  schema: T,
  minLength: number,
  message?: string
) {
  return z
    .array(schema)
    .min(minLength, message ?? `Must have at least ${minLength} items`);
}

/**
 * Array with maximum length.
 */
export function arrayWithMaxLength<T extends z.ZodTypeAny>(
  schema: T,
  maxLength: number,
  message?: string
) {
  return z
    .array(schema)
    .max(maxLength, message ?? `Must have at most ${maxLength} items`);
}

/**
 * Array with length range.
 */
export function arrayWithLengthRange<T extends z.ZodTypeAny>(
  schema: T,
  minLength: number,
  maxLength: number,
  minMessage?: string,
  maxMessage?: string
) {
  return z
    .array(schema)
    .min(minLength, minMessage ?? `Must have at least ${minLength} items`)
    .max(maxLength, maxMessage ?? `Must have at most ${maxLength} items`);
}

/**
 * String with length range.
 */
export function stringWithLengthRange(
  minLength: number,
  maxLength: number,
  minMessage?: string,
  maxMessage?: string
) {
  return z
    .string()
    .min(minLength, minMessage ?? `Must be at least ${minLength} characters`)
    .max(maxLength, maxMessage ?? `Must be at most ${maxLength} characters`)
    .trim();
}

/**
 * Number with range.
 */
export function numberWithRange(
  min: number,
  max: number,
  minMessage?: string,
  maxMessage?: string
) {
  return z
    .number()
    .min(min, minMessage ?? `Must be at least ${min}`)
    .max(max, maxMessage ?? `Must be at most ${max}`);
}

