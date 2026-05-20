export const delay = (ms: number): Promise<void> => {
  return new Promise((resolve) => setTimeout(resolve, ms));
};

export const timeout = <T,>(
  promise: Promise<T>,
  ms: number,
  errorMessage = 'Operation timed out'
): Promise<T> => {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) =>
      setTimeout(() => reject(new Error(errorMessage)), ms)
    ),
  ]);
};

export const allSettled = async <T,>(
  promises: Promise<T>[]
): Promise<Array<{ status: 'fulfilled' | 'rejected'; value?: T; reason?: Error }>> => {
  return Promise.all(
    promises.map((promise) =>
      promise
        .then((value) => ({ status: 'fulfilled' as const, value }))
        .catch((reason) => ({
          status: 'rejected' as const,
          reason: reason instanceof Error ? reason : new Error(String(reason)),
        }))
    )
  );
};

export const race = async <T,>(promises: Promise<T>[]): Promise<T> => {
  return Promise.race(promises);
};

export const any = async <T,>(promises: Promise<T>[]): Promise<T> => {
  return Promise.any(promises);
};

export const createPromise = <T,>(): {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (error: Error) => void;
} => {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;

  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });

  return {
    promise,
    resolve,
    reject: (error: Error) => reject(error),
  };
};

export const isPromise = <T,>(value: unknown): value is Promise<T> => {
  return (
    value !== null &&
    typeof value === 'object' &&
    'then' in value &&
    typeof (value as Promise<T>).then === 'function'
  );
};

export const toPromise = <T,>(value: T | Promise<T>): Promise<T> => {
  if (isPromise(value)) {
    return value;
  }
  return Promise.resolve(value);
};

