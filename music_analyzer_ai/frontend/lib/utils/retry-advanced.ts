/**
 * Advanced retry utility functions.
 * Provides helper functions for advanced retry operations.
 */

/**
 * Retry strategy type.
 */
export type RetryStrategy = 'exponential' | 'linear' | 'fixed' | 'custom';

/**
 * Retry options.
 */
export interface AdvancedRetryOptions {
  maxAttempts?: number;
  initialDelay?: number;
  maxDelay?: number;
  strategy?: RetryStrategy;
  backoffMultiplier?: number;
  jitter?: boolean;
  onRetry?: (attempt: number, error: Error) => void;
  shouldRetry?: (error: Error, attempt: number) => boolean;
}

/**
 * Calculates delay based on strategy.
 */
function calculateDelay(
  attempt: number,
  options: Required<Pick<AdvancedRetryOptions, 'initialDelay' | 'maxDelay' | 'strategy' | 'backoffMultiplier' | 'jitter'>>
): number {
  let delay: number;

  switch (options.strategy) {
    case 'exponential':
      delay = options.initialDelay * Math.pow(options.backoffMultiplier, attempt - 1);
      break;
    case 'linear':
      delay = options.initialDelay * attempt;
      break;
    case 'fixed':
      delay = options.initialDelay;
      break;
    default:
      delay = options.initialDelay;
  }

  delay = Math.min(delay, options.maxDelay);

  if (options.jitter) {
    const jitterAmount = delay * 0.1 * Math.random();
    delay += jitterAmount;
  }

  return delay;
}

/**
 * Retries a function with advanced options.
 */
export async function retryAdvanced<T>(
  fn: () => Promise<T>,
  options: AdvancedRetryOptions = {}
): Promise<T> {
  const {
    maxAttempts = 3,
    initialDelay = 1000,
    maxDelay = 30000,
    strategy = 'exponential',
    backoffMultiplier = 2,
    jitter = false,
    onRetry,
    shouldRetry = () => true,
  } = options;

  let lastError: Error;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;

      if (!shouldRetry(lastError, attempt)) {
        throw lastError;
      }

      if (attempt < maxAttempts) {
        const delay = calculateDelay(attempt, {
          initialDelay,
          maxDelay,
          strategy,
          backoffMultiplier,
          jitter,
        });

        if (onRetry) {
          onRetry(attempt, lastError);
        }

        await new Promise((resolve) => setTimeout(resolve, delay));
      }
    }
  }

  throw lastError!;
}

/**
 * Creates a retry function with preset options.
 */
export function createRetryFunction<T>(
  options: AdvancedRetryOptions
): (fn: () => Promise<T>) => Promise<T> {
  return (fn: () => Promise<T>) => retryAdvanced(fn, options);
}

