/**
 * Retry utility functions.
 * Provides helper functions for retrying operations with exponential backoff.
 */

import { delay } from './async';

/**
 * Retry options.
 */
export interface RetryOptions {
  maxAttempts?: number;
  delayMs?: number;
  exponentialBackoff?: boolean;
  onRetry?: (attempt: number, error: unknown) => void;
}

/**
 * Retries a function with exponential backoff.
 * @param fn - Function to retry
 * @param options - Retry options
 * @returns Promise that resolves with the function result
 */
export async function retry<T>(
  fn: () => Promise<T>,
  options: RetryOptions = {}
): Promise<T> {
  const {
    maxAttempts = 3,
    delayMs = 1000,
    exponentialBackoff = true,
    onRetry,
  } = options;

  let lastError: unknown;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      if (attempt < maxAttempts) {
        const waitTime = exponentialBackoff
          ? delayMs * Math.pow(2, attempt - 1)
          : delayMs;

        onRetry?.(attempt, error);
        await delay(waitTime);
      }
    }
  }

  throw lastError;
}

/**
 * Retries a function with custom retry condition.
 * @param fn - Function to retry
 * @param shouldRetry - Function that determines if error should be retried
 * @param options - Retry options
 * @returns Promise that resolves with the function result
 */
export async function retryWithCondition<T>(
  fn: () => Promise<T>,
  shouldRetry: (error: unknown, attempt: number) => boolean,
  options: RetryOptions = {}
): Promise<T> {
  const {
    maxAttempts = 3,
    delayMs = 1000,
    exponentialBackoff = true,
    onRetry,
  } = options;

  let lastError: unknown;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      if (attempt < maxAttempts && shouldRetry(error, attempt)) {
        const waitTime = exponentialBackoff
          ? delayMs * Math.pow(2, attempt - 1)
          : delayMs;

        onRetry?.(attempt, error);
        await delay(waitTime);
      } else {
        throw error;
      }
    }
  }

  throw lastError;
}

