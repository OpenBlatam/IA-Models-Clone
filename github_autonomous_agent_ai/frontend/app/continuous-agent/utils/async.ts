/**
 * Async operation utilities
 */

export const delay = (ms: number): Promise<void> => {
  return new Promise((resolve) => setTimeout(resolve, ms));
};

export interface RetryOptions {
  readonly maxAttempts?: number;
  readonly delayMs?: number;
  readonly backoffMultiplier?: number;
  readonly shouldRetry?: (error: unknown) => boolean;
}

export const retry = async <T>(
  fn: () => Promise<T>,
  options: RetryOptions = {}
): Promise<T> => {
  const {
    maxAttempts = 3,
    delayMs = 1000,
    backoffMultiplier = 2,
    shouldRetry = () => true,
  } = options;

  let lastError: unknown;
  let currentDelay = delayMs;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      if (attempt === maxAttempts || !shouldRetry(error)) {
        throw error;
      }

      await delay(currentDelay);
      currentDelay *= backoffMultiplier;
    }
  }

  throw lastError;
};

export const timeout = <T>(
  promise: Promise<T>,
  ms: number,
  errorMessage = "Operation timed out"
): Promise<T> => {
  return Promise.race([
    promise,
    new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error(errorMessage)), ms)
    ),
  ]);
};

export const withTimeout = <T>(
  fn: () => Promise<T>,
  ms: number,
  errorMessage?: string
): Promise<T> => {
  return timeout(fn(), ms, errorMessage);
};

export const sleep = delay;

export const waitFor = async (
  condition: () => boolean | Promise<boolean>,
  options: {
    readonly interval?: number;
    readonly timeout?: number;
    readonly timeoutMessage?: string;
  } = {}
): Promise<void> => {
  const { interval = 100, timeout: timeoutMs = 5000, timeoutMessage } = options;
  const startTime = Date.now();

  while (true) {
    if (await condition()) {
      return;
    }

    if (Date.now() - startTime >= timeoutMs) {
      throw new Error(
        timeoutMessage || `Condition not met within ${timeoutMs}ms`
      );
    }

    await delay(interval);
  }
};





