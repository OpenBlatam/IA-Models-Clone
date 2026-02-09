/**
 * Promise utility functions
 * Common promise operations
 */

/**
 * Create a promise that resolves after delay
 */
export function createDelayPromise(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Create a promise that rejects after timeout
 */
export function createTimeoutPromise<T>(
  promise: Promise<T>,
  ms: number,
  errorMessage?: string
): Promise<T> {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) =>
      setTimeout(
        () => reject(new Error(errorMessage || `Timeout after ${ms}ms`)),
        ms
      )
    ),
  ]);
}

/**
 * Create a cancellable promise
 */
export function createCancellablePromise<T>(
  executor: (
    resolve: (value: T) => void,
    reject: (error: Error) => void,
    onCancel: (callback: () => void) => void
  ) => void
): { promise: Promise<T>; cancel: () => void } {
  let cancelCallback: (() => void) | null = null;

  const promise = new Promise<T>((resolve, reject) => {
    executor(
      resolve,
      reject,
      (callback) => {
        cancelCallback = callback;
      }
    );
  });

  return {
    promise,
    cancel: () => {
      if (cancelCallback) {
        cancelCallback();
      }
    },
  };
}

/**
 * Promise.all with individual error handling
 */
export async function allSettled<T>(
  promises: Promise<T>[]
): Promise<Array<{ status: 'fulfilled' | 'rejected'; value?: T; error?: Error }>> {
  return Promise.all(
    promises.map((promise) =>
      promise
        .then((value) => ({ status: 'fulfilled' as const, value }))
        .catch((error) => ({
          status: 'rejected' as const,
          error: error instanceof Error ? error : new Error(String(error)),
        }))
    )
  );
}

/**
 * Promise.race with timeout
 */
export async function raceWithTimeout<T>(
  promises: Promise<T>[],
  timeoutMs: number
): Promise<T> {
  const timeoutPromise = new Promise<T>((_, reject) =>
    setTimeout(() => reject(new Error(`Race timeout after ${timeoutMs}ms`)), timeoutMs)
  );

  return Promise.race([...promises, timeoutPromise]);
}

/**
 * Retry promise with exponential backoff
 */
export async function retryPromise<T>(
  fn: () => Promise<T>,
  options: {
    maxAttempts?: number;
    initialDelay?: number;
    maxDelay?: number;
    backoffFactor?: number;
  } = {}
): Promise<T> {
  const {
    maxAttempts = 3,
    initialDelay = 1000,
    maxDelay = 10000,
    backoffFactor = 2,
  } = options;

  let lastError: Error | null = null;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      if (attempt < maxAttempts) {
        const delay = Math.min(
          initialDelay * Math.pow(backoffFactor, attempt - 1),
          maxDelay
        );
        await createDelayPromise(delay);
      }
    }
  }

  throw lastError || new Error('Retry failed');
}

/**
 * Create a debounced promise
 */
export function debouncePromise<T extends unknown[]>(
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
 * Create a throttled promise
 */
export function throttlePromise<T extends unknown[]>(
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

