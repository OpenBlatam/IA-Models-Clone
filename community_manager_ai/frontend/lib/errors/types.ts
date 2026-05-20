/**
 * Custom error types for the application
 * Provides type-safe error handling throughout the codebase
 */

/**
 * Base application error class
 */
export class AppError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly statusCode?: number,
    public readonly cause?: Error
  ) {
    super(message);
    this.name = 'AppError';
    Object.setPrototypeOf(this, AppError.prototype);
  }
}

/**
 * API error class for HTTP-related errors
 */
export class ApiError extends AppError {
  constructor(
    message: string,
    public readonly statusCode: number,
    public readonly response?: unknown,
    cause?: Error
  ) {
    super(message, 'API_ERROR', statusCode, cause);
    this.name = 'ApiError';
    Object.setPrototypeOf(this, ApiError.prototype);
  }
}

/**
 * Validation error class for input validation failures
 */
export class ValidationError extends AppError {
  constructor(
    message: string,
    public readonly field?: string,
    cause?: Error
  ) {
    super(message, 'VALIDATION_ERROR', 400, cause);
    this.name = 'ValidationError';
    Object.setPrototypeOf(this, ValidationError.prototype);
  }
}

/**
 * Network error class for network-related failures
 */
export class NetworkError extends AppError {
  constructor(message: string, cause?: Error) {
    super(message, 'NETWORK_ERROR', undefined, cause);
    this.name = 'NetworkError';
    Object.setPrototypeOf(this, NetworkError.prototype);
  }
}

/**
 * Authentication error class for auth-related failures
 */
export class AuthenticationError extends AppError {
  constructor(message: string = 'Authentication required', cause?: Error) {
    super(message, 'AUTHENTICATION_ERROR', 401, cause);
    this.name = 'AuthenticationError';
    Object.setPrototypeOf(this, AuthenticationError.prototype);
  }
}

/**
 * Authorization error class for permission-related failures
 */
export class AuthorizationError extends AppError {
  constructor(message: string = 'Insufficient permissions', cause?: Error) {
    super(message, 'AUTHORIZATION_ERROR', 403, cause);
    this.name = 'AuthorizationError';
    Object.setPrototypeOf(this, AuthorizationError.prototype);
  }
}

/**
 * Not found error class for resource not found scenarios
 */
export class NotFoundError extends AppError {
  constructor(message: string = 'Resource not found', cause?: Error) {
    super(message, 'NOT_FOUND_ERROR', 404, cause);
    this.name = 'NotFoundError';
    Object.setPrototypeOf(this, NotFoundError.prototype);
  }
}


