/**
 * Async utility functions
 * Common async operations
 */

/**
 * Delay execution
 */
export function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Retry async function
 */
export async function retryAsync<T>(
  fn: () => Promise<T>,
  options: {
    maxAttempts?: number;
    delay?: number;
    onRetry?: (attempt: number, error: Error) => void;
  } = {}
): Promise<T> {
  const { maxAttempts = 3, delay: delayMs = 1000, onRetry } = options;

  let lastError: Error | null = null;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      if (attempt < maxAttempts) {
        onRetry?.(attempt, lastError);
        await delay(delayMs * attempt); // Exponential backoff
      }
    }
  }

  throw lastError || new Error('Retry failed');
}

/**
 * Timeout async function
 */
export async function timeout<T>(
  promise: Promise<T>,
  ms: number,
  errorMessage?: string
): Promise<T> {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) =>
      setTimeout(
        () => reject(new Error(errorMessage || `Operation timed out after ${ms}ms`)),
        ms
      )
    ),
  ]);
}

/**
 * Batch async operations
 */
export async function batchAsync<T, R>(
  items: T[],
  batchSize: number,
  processor: (item: T) => Promise<R>
): Promise<R[]> {
  const results: R[] = [];

  for (let i = 0; i < items.length; i += batchSize) {
    const batch = items.slice(i, i + batchSize);
    const batchResults = await Promise.all(batch.map(processor));
    results.push(...batchResults);
  }

  return results;
}

/**
 * Sequential async operations
 */
export async function sequentialAsync<T, R>(
  items: T[],
  processor: (item: T) => Promise<R>
): Promise<R[]> {
  const results: R[] = [];

  for (const item of items) {
    const result = await processor(item);
    results.push(result);
  }

  return results;
}

/**
 * Debounce async function
 */
export function debounceAsync<T extends unknown[]>(
  fn: (...args: T) => Promise<unknown>,
  delay: number
): (...args: T) => Promise<unknown> {
  let timeoutId: NodeJS.Timeout | null = null;
  let lastPromise: Promise<unknown> | null = null;

  return (...args: T) => {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }

    return new Promise((resolve, reject) => {
      timeoutId = setTimeout(async () => {
        try {
          const result = await fn(...args);
          lastPromise = Promise.resolve(result);
          resolve(result);
        } catch (error) {
          lastPromise = Promise.reject(error);
          reject(error);
        }
      }, delay);
    });
  };
}

/**
 * Throttle async function
 */
export function throttleAsync<T extends unknown[]>(
  fn: (...args: T) => Promise<unknown>,
  delay: number
): (...args: T) => Promise<unknown> {
  let lastCall = 0;
  let lastPromise: Promise<unknown> | null = null;

  return (...args: T) => {
    const now = Date.now();

    if (now - lastCall >= delay) {
      lastCall = now;
      lastPromise = fn(...args);
      return lastPromise;
    }

    return lastPromise || Promise.resolve();
  };
}

/**
 * Parallel async operations with concurrency limit
 */
export async function parallelAsync<T, R>(
  items: T[],
  processor: (item: T) => Promise<R>,
  concurrency: number
): Promise<R[]> {
  const results: R[] = [];
  const executing: Promise<void>[] = [];

  for (const item of items) {
    const promise = processor(item).then((result) => {
      results.push(result);
    });

    executing.push(promise);

    if (executing.length >= concurrency) {
      await Promise.race(executing);
      executing.splice(
        executing.findIndex((p) => p === promise),
        1
      );
    }
  }

  await Promise.all(executing);
  return results;
}

