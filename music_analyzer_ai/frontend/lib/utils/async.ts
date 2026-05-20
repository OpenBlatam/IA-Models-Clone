/**
 * Async utility functions.
 * Provides helper functions for async operations and promises.
 */

/**
 * Delays execution for a specified number of milliseconds.
 * @param ms - Milliseconds to delay
 * @returns Promise that resolves after delay
 */
export function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Creates a promise that resolves after a timeout.
 * Useful for race conditions and timeouts.
 * @param ms - Milliseconds before timeout
 * @param message - Error message if timeout occurs
 * @returns Promise that rejects after timeout
 */
export function timeout(
  ms: number,
  message: string = 'Operation timed out'
): Promise<never> {
  return new Promise((_, reject) => {
    setTimeout(() => reject(new Error(message)), ms);
  });
}

/**
 * Wraps a promise with a timeout.
 * @param promise - Promise to wrap
 * @param ms - Timeout in milliseconds
 * @param message - Error message if timeout occurs
 * @returns Promise with timeout
 */
export function withTimeout<T>(
  promise: Promise<T>,
  ms: number,
  message: string = 'Operation timed out'
): Promise<T> {
  return Promise.race([
    promise,
    timeout(ms, message),
  ]);
}

/**
 * Retries a promise-returning function with exponential backoff.
 * @param fn - Function that returns a promise
 * @param options - Retry options
 * @returns Promise that resolves with function result
 */
export async function retry<T>(
  fn: () => Promise<T>,
  options: {
    retries?: number;
    delay?: number;
    onRetry?: (error: Error, attempt: number) => void;
  } = {}
): Promise<T> {
  const { retries = 3, delay: delayMs = 1000, onRetry } = options;

  let lastError: Error;

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      if (attempt < retries) {
        const backoffDelay = delayMs * Math.pow(2, attempt);
        onRetry?.(lastError, attempt + 1);
        await delay(backoffDelay);
      }
    }
  }

  throw lastError!;
}

/**
 * Creates a debounced version of an async function.
 * @param fn - Async function to debounce
 * @param wait - Milliseconds to wait
 * @returns Debounced function
 */
export function debounceAsync<T extends (...args: unknown[]) => Promise<unknown>>(
  fn: T,
  wait: number
): (...args: Parameters<T>) => Promise<ReturnType<T>> {
  let timeoutId: NodeJS.Timeout | null = null;
  let latestArgs: Parameters<T> | null = null;
  let resolveLatest: ((value: ReturnType<T>) => void) | null = null;
  let rejectLatest: ((error: Error) => void) | null = null;

  return (...args: Parameters<T>): Promise<ReturnType<T>> => {
    return new Promise((resolve, reject) => {
      latestArgs = args;
      resolveLatest = resolve;
      rejectLatest = reject;

      if (timeoutId) {
        clearTimeout(timeoutId);
      }

      timeoutId = setTimeout(async () => {
        if (latestArgs && resolveLatest && rejectLatest) {
          try {
            const result = await fn(...latestArgs);
            resolveLatest(result as ReturnType<T>);
          } catch (error) {
            rejectLatest(error instanceof Error ? error : new Error(String(error)));
          }
        }
      }, wait);
    });
  };
}

/**
 * Creates a throttled version of an async function.
 * @param fn - Async function to throttle
 * @param limit - Milliseconds between calls
 * @returns Throttled function
 */
export function throttleAsync<T extends (...args: unknown[]) => Promise<unknown>>(
  fn: T,
  limit: number
): (...args: Parameters<T>) => Promise<ReturnType<T>> {
  let inThrottle = false;
  let lastResult: Promise<ReturnType<T>> | null = null;

  return (...args: Parameters<T>): Promise<ReturnType<T>> => {
    if (!inThrottle) {
      inThrottle = true;
      lastResult = fn(...args) as Promise<ReturnType<T>>;

      setTimeout(() => {
        inThrottle = false;
      }, limit);
    }

    return lastResult!;
  };
}

/**
 * Executes promises in parallel with a concurrency limit.
 * @param tasks - Array of functions that return promises
 * @param limit - Maximum concurrent executions
 * @returns Array of results in order
 */
export async function pLimit<T>(
  tasks: Array<() => Promise<T>>,
  limit: number
): Promise<T[]> {
  const results: T[] = [];
  const executing: Promise<void>[] = [];

  for (const task of tasks) {
    const promise = task().then((result) => {
      results.push(result);
      executing.splice(executing.indexOf(promise), 1);
    });

    executing.push(promise);

    if (executing.length >= limit) {
      await Promise.race(executing);
    }
  }

  await Promise.all(executing);
  return results;
}

/**
 * Creates a promise that can be cancelled.
 * @param executor - Promise executor function
 * @returns Object with promise and cancel function
 */
export function cancellable<T>(
  executor: (
    resolve: (value: T) => void,
    reject: (error: Error) => void,
    onCancel: (handler: () => void) => void
  ) => void
): { promise: Promise<T>; cancel: () => void } {
  let cancelHandler: (() => void) | null = null;
  let isCancelled = false;

  const promise = new Promise<T>((resolve, reject) => {
    executor(
      (value) => {
        if (!isCancelled) {
          resolve(value);
        }
      },
      (error) => {
        if (!isCancelled) {
          reject(error);
        }
      },
      (handler) => {
        cancelHandler = handler;
      }
    );
  });

  const cancel = () => {
    isCancelled = true;
    cancelHandler?.();
  };

  return { promise, cancel };
}

