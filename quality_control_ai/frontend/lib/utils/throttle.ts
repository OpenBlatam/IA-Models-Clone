export const throttle = <T extends (...args: unknown[]) => unknown>(
  func: T,
  limit: number,
  options: { leading?: boolean; trailing?: boolean } = {}
): ((...args: Parameters<T>) => void) => {
  const { leading = true, trailing = true } = options;
  let inThrottle: boolean;
  let lastArgs: Parameters<T> | null = null;
  let timeout: NodeJS.Timeout | null = null;

  return function executedFunction(...args: Parameters<T>) {
    if (!inThrottle) {
      if (leading) {
        func(...args);
      } else {
        lastArgs = args;
      }
      inThrottle = true;
      setTimeout(() => {
        inThrottle = false;
        if (trailing && lastArgs) {
          func(...lastArgs);
          lastArgs = null;
        }
      }, limit);
    } else {
      lastArgs = args;
      if (trailing && !timeout) {
        timeout = setTimeout(() => {
          if (lastArgs) {
            func(...lastArgs);
            lastArgs = null;
          }
          timeout = null;
        }, limit);
      }
    }
  };
};

export const throttleAsync = <T extends (...args: unknown[]) => Promise<unknown>>(
  func: T,
  limit: number
): ((...args: Parameters<T>) => Promise<ReturnType<T>>) => {
  let inThrottle: boolean;
  let lastPromise: Promise<ReturnType<T>> | null = null;

  return function executedFunction(...args: Parameters<T>): Promise<ReturnType<T>> {
    return new Promise((resolve, reject) => {
      if (!inThrottle) {
        inThrottle = true;
        const promise = func(...args) as Promise<ReturnType<T>>;
        lastPromise = promise;
        promise
          .then((result) => {
            resolve(result);
            setTimeout(() => {
              inThrottle = false;
            }, limit);
          })
          .catch((error) => {
            reject(error);
            setTimeout(() => {
              inThrottle = false;
            }, limit);
          });
      } else {
        if (lastPromise) {
          lastPromise.then(resolve).catch(reject);
        }
      }
    });
  };
};

