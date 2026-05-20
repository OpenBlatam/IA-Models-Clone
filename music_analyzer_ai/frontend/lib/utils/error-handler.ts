/**
 * Error handler utility.
 * Provides centralized error handling and reporting.
 */

import { logger } from './logger';
import { ApiError, NetworkError, ValidationError } from '../errors';

/**
 * Error handler options.
 */
export interface ErrorHandlerOptions {
  logError?: boolean;
  showToast?: boolean;
  fallbackMessage?: string;
}

/**
 * Handles errors with appropriate logging and user feedback.
 */
export function handleError(
  error: unknown,
  options: ErrorHandlerOptions = {}
): string {
  const {
    logError = true,
    showToast = false,
    fallbackMessage = 'Ha ocurrido un error inesperado',
  } = options;

  let errorMessage = fallbackMessage;

  if (error instanceof ApiError) {
    errorMessage = error.message || 'Error en la API';
    if (logError) {
      logger.error('API Error', {
        message: error.message,
        statusCode: error.statusCode,
        data: error.data,
      });
    }
  } else if (error instanceof NetworkError) {
    errorMessage = error.message || 'Error de conexión';
    if (logError) {
      logger.error('Network Error', {
        message: error.message,
        originalError: error.originalError,
      });
    }
  } else if (error instanceof ValidationError) {
    errorMessage = error.message || 'Error de validación';
    if (logError) {
      logger.warn('Validation Error', {
        message: error.message,
        field: error.field,
        value: error.value,
      });
    }
  } else if (error instanceof Error) {
    errorMessage = error.message;
    if (logError) {
      logger.error('Error', error);
    }
  } else if (typeof error === 'string') {
    errorMessage = error;
    if (logError) {
      logger.error('String Error', { message: error });
    }
  } else {
    if (logError) {
      logger.error('Unknown Error', error);
    }
  }

  // Show toast notification if enabled
  if (showToast && typeof window !== 'undefined') {
    // Dynamic import to avoid SSR issues
    import('react-hot-toast').then(({ toast }) => {
      toast.error(errorMessage);
    });
  }

  return errorMessage;
}

/**
 * Creates an error handler with default options.
 */
export function createErrorHandler(defaultOptions: ErrorHandlerOptions = {}) {
  return (error: unknown, options: ErrorHandlerOptions = {}) => {
    return handleError(error, { ...defaultOptions, ...options });
  };
}

/**
 * Wraps an async function with error handling.
 */
export function withErrorHandling<T extends (...args: any[]) => Promise<any>>(
  fn: T,
  options: ErrorHandlerOptions = {}
): T {
  return (async (...args: Parameters<T>) => {
    try {
      return await fn(...args);
    } catch (error) {
      handleError(error, options);
      throw error;
    }
  }) as T;
}

