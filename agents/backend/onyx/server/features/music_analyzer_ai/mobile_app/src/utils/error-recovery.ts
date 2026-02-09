import type { ApiError } from '../types';
import { extractErrorMessage, extractStatusCode, shouldRetry } from './error-handling';

export interface ErrorRecoveryStrategy {
  canRetry: boolean;
  retryDelay: number;
  maxRetries: number;
  backoffMultiplier: number;
}

export interface RetryOptions {
  maxRetries?: number;
  initialDelay?: number;
  maxDelay?: number;
  backoffMultiplier?: number;
}

/**
 * Determines the best recovery strategy for an error
 */
export function getErrorRecoveryStrategy(
  error: unknown,
  options: RetryOptions = {}
): ErrorRecoveryStrategy {
  const {
    maxRetries = 3,
    initialDelay = 1000,
    maxDelay = 10000,
    backoffMultiplier = 2,
  } = options;

  const canRetry = shouldRetry(error);
  const statusCode = extractStatusCode(error);

  // Adjust strategy based on error type
  let retryDelay = initialDelay;
  let adjustedMaxRetries = maxRetries;

  if (statusCode === 429) {
    // Rate limited - longer delay
    retryDelay = 2000;
    adjustedMaxRetries = 2;
  } else if (statusCode === 503 || statusCode === 504) {
    // Service unavailable - moderate delay
    retryDelay = 1500;
  }

  return {
    canRetry,
    retryDelay: Math.min(retryDelay, maxDelay),
    maxRetries: adjustedMaxRetries,
    backoffMultiplier,
  };
}

/**
 * Calculates the delay for the next retry attempt
 */
export function calculateRetryDelay(
  attempt: number,
  initialDelay: number,
  maxDelay: number,
  backoffMultiplier: number
): number {
  const delay = initialDelay * Math.pow(backoffMultiplier, attempt);
  return Math.min(delay, maxDelay);
}

/**
 * Gets a user-friendly error message with recovery suggestions
 */
export function getErrorRecoveryMessage(error: unknown): {
  message: string;
  suggestion?: string;
  action?: string;
} {
  const errorMessage = extractErrorMessage(error);
  const statusCode = extractStatusCode(error);

  if (statusCode === 404) {
    return {
      message: errorMessage,
      suggestion: 'The requested resource was not found.',
      action: 'Try searching again or check your input.',
    };
  }

  if (statusCode === 403 || statusCode === 401) {
    return {
      message: errorMessage,
      suggestion: 'You may not have permission to access this resource.',
      action: 'Please check your authentication or contact support.',
    };
  }

  if (statusCode === 429) {
    return {
      message: errorMessage,
      suggestion: 'Too many requests. Please wait a moment.',
      action: 'Try again in a few seconds.',
    };
  }

  if (statusCode === 500 || statusCode === 502 || statusCode === 503) {
    return {
      message: errorMessage,
      suggestion: 'The server is experiencing issues.',
      action: 'Please try again later.',
    };
  }

  if (statusCode === 0 || !statusCode) {
    return {
      message: errorMessage,
      suggestion: 'Unable to connect to the server.',
      action: 'Check your internet connection and try again.',
    };
  }

  return {
    message: errorMessage,
  };
}

