export const sleep = (ms: number): Promise<void> => {
  return new Promise((resolve) => setTimeout(resolve, ms));
};

export const retry = async <T,>(
  fn: () => Promise<T>,
  retries: number = 3,
  delay: number = 1000
): Promise<T> => {
  try {
    return await fn();
  } catch (error) {
    if (retries === 0) {
      throw error;
    }
    await sleep(delay);
    return retry(fn, retries - 1, delay);
  }
};

export const timeout = <T,>(promise: Promise<T>, ms: number): Promise<T> => {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) =>
      setTimeout(() => reject(new Error(`Operation timed out after ${ms}ms`)), ms)
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
        .catch((reason) => ({ status: 'rejected' as const, reason: reason instanceof Error ? reason : new Error(String(reason)) }))
    )
  );
};

export const debouncePromise = <T extends (...args: unknown[]) => Promise<unknown>>(
  fn: T,
  delay: number
): ((...args: Parameters<T>) => Promise<ReturnType<T>>) => {
  let timeoutId: NodeJS.Timeout;
  let latestResolve: ((value: ReturnType<T>) => void)[] = [];
  let latestReject: ((error: Error) => void)[] = [];

  return ((...args: Parameters<T>) => {
    return new Promise<ReturnType<T>>((resolve, reject) => {
      clearTimeout(timeoutId);
      latestResolve.push(resolve);
      latestReject.push(reject);

      timeoutId = setTimeout(() => {
        const currentResolve = latestResolve;
        const currentReject = latestReject;
        latestResolve = [];
        latestReject = [];

        fn(...args)
          .then((result) => {
            currentResolve.forEach((resolve) => resolve(result as ReturnType<T>));
          })
          .catch((error) => {
            currentReject.forEach((reject) => reject(error instanceof Error ? error : new Error(String(error))));
          });
      }, delay);
    });
  }) as typeof fn;
};



