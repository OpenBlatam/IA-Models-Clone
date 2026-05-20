/**
 * Promise utility functions.
 * Provides helper functions for promise operations.
 */

/**
 * Creates a promise that can be resolved/rejected externally.
 */
export function createPromise<T>(): {
  promise: Promise<T>;
  resolve: (value: T | PromiseLike<T>) => void;
  reject: (reason?: any) => void;
} {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: any) => void;

  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });

  return { promise, resolve, reject };
}

/**
 * Creates a promise that resolves after a delay.
 */
export function delayPromise(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Creates a promise that rejects after a timeout.
 */
export function timeoutPromise<T>(
  promise: Promise<T>,
  ms: number,
  message?: string
): Promise<T> {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) =>
      setTimeout(() => reject(new Error(message || 'Promise timeout')), ms)
    ),
  ]);
}

/**
 * Executes promises in sequence.
 */
export async function sequence<T>(
  promises: (() => Promise<T>)[]
): Promise<T[]> {
  const results: T[] = [];
  for (const promiseFn of promises) {
    results.push(await promiseFn());
  }
  return results;
}

/**
 * Executes promises in parallel with concurrency limit.
 */
export async function parallel<T>(
  promises: (() => Promise<T>)[],
  concurrency: number = 5
): Promise<T[]> {
  const results: T[] = [];
  const executing: Promise<void>[] = [];

  for (const promiseFn of promises) {
    const promise = promiseFn().then((result) => {
      results.push(result);
      executing.splice(executing.indexOf(promise), 1);
    });

    executing.push(promise as Promise<void>);

    if (executing.length >= concurrency) {
      await Promise.race(executing);
    }
  }

  await Promise.all(executing);
  return results;
}

/**
 * Retries a promise function.
 */
export async function retryPromise<T>(
  fn: () => Promise<T>,
  maxAttempts: number = 3,
  delay: number = 1000
): Promise<T> {
  let lastError: Error;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;
      if (attempt < maxAttempts) {
        await delayPromise(delay * attempt);
      }
    }
  }

  throw lastError!;
}

/**
 * Creates a debounced promise function.
 */
export function debouncePromise<T extends (...args: any[]) => Promise<any>>(
  fn: T,
  delay: number
): (...args: Parameters<T>) => Promise<ReturnType<T>> {
  let timeoutId: NodeJS.Timeout;
  let latestResolve: ((value: any) => void)[] = [];
  let latestReject: ((error: any) => void)[] = [];

  return function (this: any, ...args: Parameters<T>): Promise<ReturnType<T>> {
    return new Promise((resolve, reject) => {
      // Clear previous timeout
      clearTimeout(timeoutId);

      // Store latest resolve/reject
      latestResolve.push(resolve);
      latestReject.push(reject);

      // Set new timeout
      timeoutId = setTimeout(async () => {
        const resolves = latestResolve;
        const rejects = latestReject;
        latestResolve = [];
        latestReject = [];

        try {
          const result = await fn.apply(this, args);
          resolves.forEach((r) => r(result));
        } catch (error) {
          rejects.forEach((r) => r(error));
        }
      }, delay);
    });
  };
}

