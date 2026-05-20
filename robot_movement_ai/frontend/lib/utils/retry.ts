/**
 * Retry utility for handling failed operations
 */

export interface RetryOptions {
  maxAttempts?: number;
  delay?: number;
  backoff?: 'linear' | 'exponential';
  onRetry?: (attempt: number, error: unknown) => void;
}

export async function retry<T>(
  fn: () => Promise<T>,
  options: RetryOptions = {}
): Promise<T> {
  const {
    maxAttempts = 3,
    delay = 1000,
    backoff = 'exponential',
    onRetry,
  } = options;

  let lastError: unknown;
  let currentDelay = delay;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      if (attempt < maxAttempts) {
        if (onRetry) {
          onRetry(attempt, error);
        }

        await new Promise((resolve) => setTimeout(resolve, currentDelay));

        // Calculate next delay
        if (backoff === 'exponential') {
          currentDelay *= 2;
        } else {
          currentDelay = delay;
        }
      }
    }
  }

  throw lastError;
}

/**
 * Retry with exponential backoff
 */
export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  maxAttempts: number = 3
): Promise<T> {
  return retry(fn, {
    maxAttempts,
    delay: 1000,
    backoff: 'exponential',
  });
}
