/**
 * Centralized error handling utilities
 */

import { toast } from './toast';
import { logger } from './logger';
import { ERROR_MESSAGES } from './constants';
import { formatErrorMessage } from './helpers';

export interface ErrorHandlerOptions {
  showToast?: boolean;
  logError?: boolean;
  fallbackMessage?: string;
  onError?: (error: unknown) => void;
}

/**
 * Handle errors with consistent behavior
 */
export function handleError(
  error: unknown,
  options: ErrorHandlerOptions = {}
): string {
  const {
    showToast = true,
    logError = true,
    fallbackMessage = ERROR_MESSAGES.UNKNOWN_ERROR,
    onError,
  } = options;

  const errorMessage = formatErrorMessage(error);

  // Log error
  if (logError) {
    logger.error('Error handled:', error);
  }

  // Show toast
  if (showToast) {
    toast.error(errorMessage || fallbackMessage);
  }

  // Custom error handler
  if (onError) {
    onError(error);
  }

  return errorMessage || fallbackMessage;
}

/**
 * Handle API errors specifically
 */
export function handleApiError(
  error: unknown,
  customMessage?: string
): void {
  handleError(error, {
    showToast: true,
    logError: true,
    fallbackMessage: customMessage || ERROR_MESSAGES.CONNECTION_FAILED,
  });
}

/**
 * Handle validation errors
 */
export function handleValidationError(
  error: unknown,
  field?: string
): void {
  const message = field
    ? `Error de validación en ${field}`
    : 'Error de validación';
  
  handleError(error, {
    showToast: true,
    logError: false, // Validation errors are usually expected
    fallbackMessage: message,
  });
}

/**
 * Handle network errors
 */
export function handleNetworkError(error: unknown): void {
  handleError(error, {
    showToast: true,
    logError: true,
    fallbackMessage: ERROR_MESSAGES.CONNECTION_FAILED,
  });
}

/**
 * Create an error handler with predefined options
 */
export function createErrorHandler(options: ErrorHandlerOptions) {
  return (error: unknown) => handleError(error, options);
}

/**
 * Wrap async function with error handling
 */
export function withErrorHandling<T extends (...args: any[]) => Promise<any>>(
  fn: T,
  options?: ErrorHandlerOptions
): T {
  return (async (...args: Parameters<T>) => {
    try {
      return await fn(...args);
    } catch (error) {
      handleError(error, options);
      throw error; // Re-throw to allow caller to handle if needed
    }
  }) as T;
}

/**
 * Wrap sync function with error handling
 */
export function withSyncErrorHandling<T extends (...args: any[]) => any>(
  fn: T,
  options?: ErrorHandlerOptions
): T {
  return ((...args: Parameters<T>) => {
    try {
      return fn(...args);
    } catch (error) {
      handleError(error, options);
      throw error;
    }
  }) as T;
}



