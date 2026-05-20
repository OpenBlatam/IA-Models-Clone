/**
 * Advanced timeout utility functions.
 * Provides helper functions for advanced timeout operations.
 */

/**
 * Timeout options.
 */
export interface TimeoutOptions {
  timeout: number;
  onTimeout?: () => void;
  abortSignal?: AbortSignal;
}

/**
 * Executes function with timeout.
 */
export async function withTimeout<T>(
  fn: () => Promise<T>,
  options: TimeoutOptions
): Promise<T> {
  const { timeout, onTimeout, abortSignal } = options;

  return Promise.race([
    fn(),
    new Promise<T>((_, reject) => {
      const timer = setTimeout(() => {
        if (onTimeout) {
          onTimeout();
        }
        reject(new Error(`Operation timed out after ${timeout}ms`));
      }, timeout);

      if (abortSignal) {
        abortSignal.addEventListener('abort', () => {
          clearTimeout(timer);
          reject(new Error('Operation aborted'));
        });
      }
    }),
  ]);
}

/**
 * Creates a timeout promise.
 */
export function createTimeoutPromise(
  timeout: number,
  message?: string
): Promise<never> {
  return new Promise((_, reject) => {
    setTimeout(() => {
      reject(new Error(message || `Timeout after ${timeout}ms`));
    }, timeout);
  });
}

/**
 * Executes function with retry and timeout.
 */
export async function withRetryAndTimeout<T>(
  fn: () => Promise<T>,
  timeout: number,
  maxRetries: number = 3
): Promise<T> {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await withTimeout(fn, { timeout });
    } catch (error) {
      if (attempt === maxRetries) {
        throw error;
      }
      await new Promise((resolve) => setTimeout(resolve, 1000 * attempt));
    }
  }
  throw new Error('All retry attempts failed');
}

