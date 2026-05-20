/**
 * Validation utilities
 */

import { z } from 'zod';

// Common validation schemas
export const emailSchema = z.string().email('Email inválido');
export const urlSchema = z.string().url('URL inválida');
export const phoneSchema = z.string().regex(/^\+?[\d\s-()]+$/, 'Teléfono inválido');

// Position validation
export const positionSchema = z.object({
  x: z.number().min(-10).max(10),
  y: z.number().min(-10).max(10),
  z: z.number().min(-10).max(10),
});

// Validate and parse with error handling
export function safeParse<T>(
  schema: z.ZodSchema<T>,
  data: unknown
): { success: true; data: T } | { success: false; error: string } {
  const result = schema.safeParse(data);
  
  if (result.success) {
    return { success: true, data: result.data };
  }

  const errorMessage = result.error.errors
    .map((err) => `${err.path.join('.')}: ${err.message}`)
    .join(', ');

  return { success: false, error: errorMessage };
}

// Validate form data
export function validateForm<T>(
  schema: z.ZodSchema<T>,
  formData: FormData
): { success: true; data: T } | { success: false; errors: Record<string, string> } {
  const data = Object.fromEntries(formData.entries());
  const result = schema.safeParse(data);

  if (result.success) {
    return { success: true, data: result.data };
  }

  const errors: Record<string, string> = {};
  result.error.errors.forEach((err) => {
    const path = err.path.join('.');
    errors[path] = err.message;
  });

  return { success: false, errors };
}



