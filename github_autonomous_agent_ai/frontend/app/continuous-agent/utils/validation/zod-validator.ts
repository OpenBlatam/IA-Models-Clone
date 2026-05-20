/**
 * Zod validation utilities for Continuous Agent feature
 * 
 * Provides type-safe validation functions with consistent error handling
 */

import { z } from "zod";
import type {
  continuousAgentSchema,
  createAgentRequestSchema,
  updateAgentRequestSchema,
  agentExecutionLogSchema,
} from "./zod-schemas";

/**
 * Custom validation error class
 */
export class ValidationError extends Error {
  readonly errors: z.ZodError["errors"];

  constructor(message: string, errors: z.ZodError["errors"] = []) {
    super(message);
    this.name = "ValidationError";
    this.errors = errors;
  }

  /**
   * Gets user-friendly error messages
   */
  getFormattedErrors(): string[] {
    return this.errors.map((error) => {
      const path = error.path.join(".");
      return path ? `${path}: ${error.message}` : error.message;
    });
  }
}

/**
 * Validates data against a Zod schema and throws ValidationError if invalid
 * 
 * @param schema - Zod schema to validate against
 * @param data - Data to validate
 * @param errorMessage - Optional custom error message
 * @returns Validated data with proper TypeScript types
 * @throws ValidationError if validation fails
 * 
 * @example
 * ```typescript
 * try {
 *   const agent = validateWithZod(continuousAgentSchema, data, "Invalid agent data");
 *   // agent is now typed as ContinuousAgent
 * } catch (error) {
 *   if (error instanceof ValidationError) {
 *     console.error(error.getFormattedErrors());
 *   }
 * }
 * ```
 */
export function validateWithZod<T extends z.ZodType>(
  schema: T,
  data: unknown,
  errorMessage?: string
): z.infer<T> {
  try {
    return schema.parse(data);
  } catch (error) {
    if (error instanceof z.ZodError) {
      const defaultMessage = errorMessage || "Error de validación";
      throw new ValidationError(defaultMessage, error.errors);
    }
    throw error;
  }
}

/**
 * Safely validates data against a Zod schema without throwing
 * 
 * @param schema - Zod schema to validate against
 * @param data - Data to validate
 * @returns Object with success flag and data or error
 * 
 * @example
 * ```typescript
 * const result = safeValidateWithZod(createAgentRequestSchema, formData);
 * if (result.success) {
 *   // result.data is typed as CreateAgentRequest
 *   await createAgent(result.data);
 * } else {
 *   // result.error is a ZodError
 *   console.error(result.error.errors);
 * }
 * ```
 */
export function safeValidateWithZod<T extends z.ZodType>(
  schema: T,
  data: unknown
): { success: true; data: z.infer<T> } | { success: false; error: z.ZodError } {
  const result = schema.safeParse(data);

  if (result.success) {
    return { success: true, data: result.data };
  }

  return { success: false, error: result.error };
}

/**
 * Validates and sanitizes string input
 * 
 * @param value - String value to validate and sanitize
 * @param schema - Zod string schema
 * @returns Sanitized string or throws ValidationError
 */
export function validateAndSanitizeString(
  value: unknown,
  schema: z.ZodString
): string {
  const result = schema.safeParse(value);

  if (!result.success) {
    throw new ValidationError("Invalid string value", result.error.errors);
  }

  return result.data.trim();
}

/**
 * Validates JSON string and parses it
 * 
 * @param jsonString - JSON string to validate
 * @param schema - Optional Zod schema to validate parsed JSON
 * @returns Parsed and validated object
 * @throws ValidationError if JSON is invalid or doesn't match schema
 */
export function validateJSON<T extends z.ZodType>(
  jsonString: string,
  schema?: T
): T extends z.ZodType ? z.infer<T> : Record<string, unknown> {
  let parsed: unknown;

  try {
    parsed = JSON.parse(jsonString);
  } catch (error) {
    const message = error instanceof Error ? error.message : "JSON inválido";
    throw new ValidationError(`Error al parsear JSON: ${message}`);
  }

  if (schema) {
    return validateWithZod(schema, parsed) as T extends z.ZodType
      ? z.infer<T>
      : Record<string, unknown>;
  }

  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new ValidationError("Los parámetros deben ser un objeto JSON");
  }

  return parsed as Record<string, unknown>;
}

/**
 * Validates array of items against a schema
 * 
 * @param schema - Zod schema for array items
 * @param data - Array data to validate
 * @returns Validated array
 * @throws ValidationError if validation fails
 */
export function validateArray<T extends z.ZodType>(
  schema: T,
  data: unknown[]
): z.infer<T>[] {
  const arraySchema = z.array(schema);
  return validateWithZod(arraySchema, data);
}




