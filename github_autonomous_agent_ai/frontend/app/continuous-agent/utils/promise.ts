/**
 * Promise utilities
 */

export const allSettled = async <T>(
  promises: readonly Promise<T>[]
): Promise<PromiseSettledResult<T>[]> => {
  return Promise.allSettled(promises);
};

export const promiseRace = async <T>(
  promises: readonly Promise<T>[]
): Promise<T> => {
  return Promise.race(promises);
};

export const promiseAny = async <T>(
  promises: readonly Promise<T>[]
): Promise<T> => {
  return Promise.any(promises);
};

export const promiseTimeout = <T>(
  promise: Promise<T>,
  ms: number,
  errorMessage = "Promise timed out"
): Promise<T> => {
  return Promise.race([
    promise,
    new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error(errorMessage)), ms)
    ),
  ]);
};

export const promiseDelay = (ms: number): Promise<void> => {
  return new Promise((resolve) => setTimeout(resolve, ms));
};

export const createDeferred = <T>(): {
  readonly promise: Promise<T>;
  readonly resolve: (value: T | PromiseLike<T>) => void;
  readonly reject: (reason?: unknown) => void;
} => {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;

  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });

  return { promise, resolve, reject };
};

export const promiseRetry = async <T>(
  fn: () => Promise<T>,
  options: {
    readonly maxAttempts?: number;
    readonly delayMs?: number;
    readonly backoffMultiplier?: number;
    readonly shouldRetry?: (error: unknown) => boolean;
  } = {}
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

      await promiseDelay(currentDelay);
      currentDelay *= backoffMultiplier;
    }
  }

  throw lastError;
};

export const map = async <T, U>(
  array: readonly T[],
  fn: (item: T, index: number) => Promise<U>
): Promise<U[]> => {
  return Promise.all(array.map(fn));
};

export const mapSeries = async <T, U>(
  array: readonly T[],
  fn: (item: T, index: number) => Promise<U>
): Promise<U[]> => {
  const results: U[] = [];

  for (let i = 0; i < array.length; i++) {
    results.push(await fn(array[i], i));
  }

  return results;
};

export const filter = async <T>(
  array: readonly T[],
  fn: (item: T, index: number) => Promise<boolean>
): Promise<T[]> => {
  const results = await Promise.all(array.map(fn));
  return array.filter((_, index) => results[index]);
};

export const reduce = async <T, U>(
  array: readonly T[],
  fn: (acc: U, item: T, index: number) => Promise<U>,
  initialValue: U
): Promise<U> => {
  let acc = initialValue;

  for (let i = 0; i < array.length; i++) {
    acc = await fn(acc, array[i], i);
  }

  return acc;
};

export const tap = async <T>(
  promise: Promise<T>,
  fn: (value: T) => Promise<void> | void
): Promise<T> => {
  const value = await promise;
  await fn(value);
  return value;
};

export const catchError = async <T, U>(
  promise: Promise<T>,
  fn: (error: unknown) => Promise<U> | U
): Promise<T | U> => {
  try {
    return await promise;
  } catch (error) {
    return await fn(error);
  }
};

