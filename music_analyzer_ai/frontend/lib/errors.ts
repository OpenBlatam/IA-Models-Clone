/**
 * Custom error types for consistent error handling across the application.
 */

/**
 * Base API error class for all API-related errors.
 */
export class ApiError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public response?: unknown
  ) {
    super(message);
    this.name = 'ApiError';
    Object.setPrototypeOf(this, ApiError.prototype);
  }
}

/**
 * Network error class for connection issues.
 */
export class NetworkError extends Error {
  constructor(message: string = 'Network error occurred') {
    super(message);
    this.name = 'NetworkError';
    Object.setPrototypeOf(this, NetworkError.prototype);
  }
}

/**
 * Validation error class for input validation failures.
 */
export class ValidationError extends Error {
  constructor(
    message: string,
    public field?: string,
    public errors?: Record<string, string[]>
  ) {
    super(message);
    this.name = 'ValidationError';
    Object.setPrototypeOf(this, ValidationError.prototype);
  }
}

/**
 * Type guard to check if an error is an ApiError.
 */
export function isApiError(error: unknown): error is ApiError {
  return error instanceof ApiError;
}

/**
 * Type guard to check if an error is a NetworkError.
 */
export function isNetworkError(error: unknown): error is NetworkError {
  return error instanceof NetworkError;
}

/**
 * Type guard to check if an error is a ValidationError.
 */
export function isValidationError(error: unknown): error is ValidationError {
  return error instanceof ValidationError;
}

/**
 * Extracts a user-friendly error message from an error object.
 * @param error - The error to extract message from
 * @returns A user-friendly error message
 */
export function getErrorMessage(error: unknown): string {
  if (isApiError(error)) {
    return error.message || 'An API error occurred';
  }
  if (isNetworkError(error)) {
    return 'Network connection failed. Please check your internet connection.';
  }
  if (isValidationError(error)) {
    return error.message || 'Validation failed';
  }
  if (error instanceof Error) {
    return error.message;
  }
  return 'An unexpected error occurred';
}

