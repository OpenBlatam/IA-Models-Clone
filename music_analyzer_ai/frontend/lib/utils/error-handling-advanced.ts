/**
 * Advanced error handling utilities.
 * Provides comprehensive error handling, recovery, and reporting.
 */

import {
  ApiError,
  NetworkError,
  ValidationError,
  isApiError,
  isNetworkError,
  isValidationError,
  getErrorMessage,
} from '@/lib/errors';
import { toast } from 'react-hot-toast';

/**
 * Error severity levels.
 */
export enum ErrorSeverity {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical',
}

/**
 * Error context information.
 */
export interface ErrorContext {
  component?: string;
  action?: string;
  userId?: string;
  timestamp?: Date;
  metadata?: Record<string, unknown>;
}

/**
 * Error report structure.
 */
export interface ErrorReport {
  error: Error;
  severity: ErrorSeverity;
  context?: ErrorContext;
  userMessage?: string;
  recoverable?: boolean;
}

/**
 * Options for error handling.
 */
export interface ErrorHandlingOptions {
  /**
   * Whether to show a toast notification.
   */
  showToast?: boolean;
  /**
   * Custom toast message.
   */
  toastMessage?: string;
  /**
   * Whether to log the error.
   */
  logError?: boolean;
  /**
   * Whether to report to error tracking service.
   */
  reportError?: boolean;
  /**
   * Custom error handler.
   */
  onError?: (error: Error, context?: ErrorContext) => void;
  /**
   * Context information.
   */
  context?: ErrorContext;
}

/**
 * Determines error severity based on error type.
 */
export function getErrorSeverity(error: unknown): ErrorSeverity {
  if (isApiError(error)) {
    const statusCode = error.statusCode;
    if (statusCode && statusCode >= 500) {
      return ErrorSeverity.CRITICAL;
    }
    if (statusCode && statusCode >= 400) {
      return ErrorSeverity.HIGH;
    }
    return ErrorSeverity.MEDIUM;
  }

  if (isNetworkError(error)) {
    return ErrorSeverity.HIGH;
  }

  if (isValidationError(error)) {
    return ErrorSeverity.LOW;
  }

  return ErrorSeverity.MEDIUM;
}

/**
 * Determines if an error is recoverable.
 */
export function isRecoverableError(error: unknown): boolean {
  if (isNetworkError(error)) {
    return true; // Network errors are usually recoverable
  }

  if (isApiError(error)) {
    const statusCode = error.statusCode;
    // 5xx errors might be recoverable, 4xx usually not
    return statusCode ? statusCode >= 500 : false;
  }

  if (isValidationError(error)) {
    return true; // Validation errors are recoverable
  }

  return false;
}

/**
 * Gets a user-friendly error message based on error type.
 */
export function getUserFriendlyMessage(
  error: unknown,
  defaultMessage?: string
): string {
  const baseMessage = getErrorMessage(error);

  if (isApiError(error)) {
    const statusCode = error.statusCode;
    if (statusCode === 404) {
      return 'The requested resource was not found.';
    }
    if (statusCode === 403) {
      return 'You do not have permission to perform this action.';
    }
    if (statusCode === 401) {
      return 'Please log in to continue.';
    }
    if (statusCode && statusCode >= 500) {
      return 'A server error occurred. Please try again later.';
    }
  }

  if (isNetworkError(error)) {
    return 'Unable to connect to the server. Please check your internet connection.';
  }

  if (isValidationError(error)) {
    return error.message || 'Please check your input and try again.';
  }

  return defaultMessage || baseMessage || 'An unexpected error occurred.';
}

/**
 * Handles an error with comprehensive options.
 */
export function handleError(
  error: unknown,
  options: ErrorHandlingOptions = {}
): ErrorReport {
  const {
    showToast = true,
    toastMessage,
    logError = true,
    reportError = false,
    onError,
    context,
  } = options;

  const errorObj = error instanceof Error ? error : new Error(String(error));
  const severity = getErrorSeverity(error);
  const recoverable = isRecoverableError(error);
  const userMessage = toastMessage || getUserFriendlyMessage(error);

  // Create error report
  const report: ErrorReport = {
    error: errorObj,
    severity,
    context: {
      ...context,
      timestamp: new Date(),
    },
    userMessage,
    recoverable,
  };

  // Show toast notification
  if (showToast) {
    const toastOptions = {
      duration: severity === ErrorSeverity.CRITICAL ? 6000 : 4000,
    };

    if (severity === ErrorSeverity.CRITICAL || severity === ErrorSeverity.HIGH) {
      toast.error(userMessage, toastOptions);
    } else {
      toast(userMessage, {
        ...toastOptions,
        icon: '⚠️',
      });
    }
  }

  // Log error
  if (logError) {
    const logLevel = severity === ErrorSeverity.CRITICAL ? 'error' : 'warn';
    console[logLevel]('Error handled:', {
      error: errorObj,
      severity,
      recoverable,
      context,
    });
  }

  // Report to error tracking service
  if (reportError && typeof window !== 'undefined') {
    // Integrate with error tracking service (Sentry, LogRocket, etc.)
    // Example: window.errorTracker?.captureException(errorObj, { extra: context });
  }

  // Call custom error handler
  if (onError) {
    onError(errorObj, context);
  }

  return report;
}

/**
 * Creates an error handler with default options.
 */
export function createErrorHandler(defaultOptions?: ErrorHandlingOptions) {
  return (error: unknown, options?: ErrorHandlingOptions) => {
    return handleError(error, { ...defaultOptions, ...options });
  };
}

/**
 * Wraps an async function with error handling.
 */
export function withErrorHandling<T extends (...args: never[]) => Promise<unknown>>(
  fn: T,
  options?: ErrorHandlingOptions
): T {
  return (async (...args: Parameters<T>) => {
    try {
      return await fn(...args);
    } catch (error) {
      handleError(error, {
        ...options,
        context: {
          ...options?.context,
          action: fn.name || 'unknown',
        },
      });
      throw error; // Re-throw to allow caller to handle
    }
  }) as T;
}

/**
 * Retries a function with exponential backoff on error.
 */
export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  options: {
    maxRetries?: number;
    initialDelay?: number;
    maxDelay?: number;
    onRetry?: (attempt: number, error: unknown) => void;
  } = {}
): Promise<T> {
  const {
    maxRetries = 3,
    initialDelay = 1000,
    maxDelay = 10000,
    onRetry,
  } = options;

  let lastError: unknown;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      // Don't retry on validation errors
      if (isValidationError(error)) {
        throw error;
      }

      // Don't retry on 4xx errors (except 429)
      if (isApiError(error) && error.statusCode) {
        if (error.statusCode >= 400 && error.statusCode < 500 && error.statusCode !== 429) {
          throw error;
        }
      }

      if (attempt < maxRetries) {
        const delay = Math.min(
          initialDelay * Math.pow(2, attempt),
          maxDelay
        );

        if (onRetry) {
          onRetry(attempt + 1, error);
        }

        await new Promise((resolve) => setTimeout(resolve, delay));
      }
    }
  }

  throw lastError;
}

/**
 * Error recovery strategies.
 */
export enum RecoveryStrategy {
  RETRY = 'retry',
  FALLBACK = 'fallback',
  IGNORE = 'ignore',
  REDIRECT = 'redirect',
}

/**
 * Attempts to recover from an error using a strategy.
 */
export async function recoverFromError<T>(
  error: unknown,
  strategy: RecoveryStrategy,
  options?: {
    retryFn?: () => Promise<T>;
    fallbackValue?: T;
    redirectTo?: string;
  }
): Promise<T | null> {
  switch (strategy) {
    case RecoveryStrategy.RETRY:
      if (options?.retryFn) {
        return await retryWithBackoff(options.retryFn);
      }
      break;

    case RecoveryStrategy.FALLBACK:
      return options?.fallbackValue ?? null;

    case RecoveryStrategy.IGNORE:
      return null;

    case RecoveryStrategy.REDIRECT:
      if (options?.redirectTo && typeof window !== 'undefined') {
        window.location.href = options.redirectTo;
      }
      return null;
  }

  return null;
}




