/**
 * Retry utilities with exponential backoff
 * 
 * Provides robust retry mechanisms for API calls and async operations
 */

import type { AgentError } from "../errors/agent-errors";
import { AgentTimeoutError, AgentNetworkError } from "../errors/agent-errors";

/**
 * Options for retry operations
 */
export interface RetryOptions {
  /** Maximum number of retry attempts */
  readonly maxAttempts?: number;
  /** Initial delay in milliseconds */
  readonly initialDelayMs?: number;
  /** Maximum delay in milliseconds */
  readonly maxDelayMs?: number;
  /** Backoff multiplier (exponential backoff) */
  readonly backoffMultiplier?: number;
  /** Jitter factor (0-1) to add randomness to delays */
  readonly jitter?: number;
  /** Function to determine if error is retryable */
  readonly isRetryable?: (error: unknown) => boolean;
  /** Function called before each retry */
  readonly onRetry?: (attempt: number, error: unknown) => void;
}

const DEFAULT_RETRY_OPTIONS: Required<RetryOptions> = {
  maxAttempts: 3,
  initialDelayMs: 1000,
  maxDelayMs: 30000,
  backoffMultiplier: 2,
  jitter: 0.1,
  isRetryable: () => true,
  onRetry: () => {},
} as const;

/**
 * Calculates delay with exponential backoff and jitter
 */
function calculateDelay(
  attempt: number,
  options: Required<RetryOptions>
): number {
  const exponentialDelay =
    options.initialDelayMs * Math.pow(options.backoffMultiplier, attempt - 1);
  const withJitter =
    exponentialDelay * (1 + (Math.random() - 0.5) * options.jitter);
  return Math.min(withJitter, options.maxDelayMs);
}

/**
 * Checks if an error is retryable
 */
function isRetryableError(error: unknown, isRetryable?: (error: unknown) => boolean): boolean {
  if (isRetryable) {
    return isRetryable(error);
  }

  // Default: retry on network and timeout errors, not on validation errors
  if (error instanceof AgentTimeoutError || error instanceof AgentNetworkError) {
    return true;
  }

  if (error instanceof Error) {
    const message = error.message.toLowerCase();
    return (
      message.includes("network") ||
      message.includes("timeout") ||
      message.includes("fetch failed") ||
      message.includes("connection")
    );
  }

  return false;
}

/**
 * Sleeps for specified milliseconds
 */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Retries an async function with exponential backoff
 * 
 * @param fn - Async function to retry
 * @param options - Retry configuration options
 * @returns Result of the function
 * @throws Last error if all retries fail
 * 
 * @example
 * ```typescript
 * const result = await retryWithBackoff(
 *   () => fetchAgent(id),
 *   {
 *     maxAttempts: 3,
 *     initialDelayMs: 1000,
 *     onRetry: (attempt, error) => console.log(`Retry ${attempt}`, error)
 *   }
 * );
 * ```
 */
export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  options: RetryOptions = {}
): Promise<T> {
  const opts: Required<RetryOptions> = {
    ...DEFAULT_RETRY_OPTIONS,
    ...options,
  };

  let lastError: unknown;

  for (let attempt = 1; attempt <= opts.maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      // Don't retry if error is not retryable
      if (!isRetryableError(error, opts.isRetryable)) {
        throw error;
      }

      // Don't retry on last attempt
      if (attempt >= opts.maxAttempts) {
        break;
      }

      // Call onRetry callback
      opts.onRetry(attempt, error);

      // Calculate and wait for delay
      const delay = calculateDelay(attempt, opts);
      await sleep(delay);
    }
  }

  throw lastError;
}

/**
 * Retries an async function with fixed delay
 * 
 * @param fn - Async function to retry
 * @param options - Retry configuration options
 * @returns Result of the function
 * @throws Last error if all retries fail
 */
export async function retryWithFixedDelay<T>(
  fn: () => Promise<T>,
  options: RetryOptions & { delayMs: number } = { delayMs: 1000 }
): Promise<T> {
  const opts: Required<RetryOptions> = {
    ...DEFAULT_RETRY_OPTIONS,
    ...options,
    initialDelayMs: options.delayMs,
    backoffMultiplier: 1, // No backoff for fixed delay
  };

  return retryWithBackoff(fn, opts);
}




