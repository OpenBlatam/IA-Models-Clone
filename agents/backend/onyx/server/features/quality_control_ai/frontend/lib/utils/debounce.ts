export const debounce = <T extends (...args: unknown[]) => unknown>(
  func: T,
  wait: number,
  immediate = false
): ((...args: Parameters<T>) => void) => {
  let timeout: NodeJS.Timeout | null = null;

  return function executedFunction(...args: Parameters<T>) {
    const later = () => {
      timeout = null;
      if (!immediate) func(...args);
    };

    const callNow = immediate && !timeout;

    if (timeout) {
      clearTimeout(timeout);
    }
    timeout = setTimeout(later, wait);

    if (callNow) {
      func(...args);
    }
  };
};

export const debounceAsync = <T extends (...args: unknown[]) => Promise<unknown>>(
  func: T,
  wait: number
): ((...args: Parameters<T>) => Promise<ReturnType<T>>) => {
  let timeout: NodeJS.Timeout | null = null;
  let lastPromise: Promise<ReturnType<T>> | null = null;

  return function executedFunction(...args: Parameters<T>): Promise<ReturnType<T>> {
    return new Promise((resolve, reject) => {
      if (timeout) {
        clearTimeout(timeout);
      }

      timeout = setTimeout(async () => {
        try {
          const result = await func(...args);
          lastPromise = Promise.resolve(result) as Promise<ReturnType<T>>;
          resolve(result as ReturnType<T>);
        } catch (error) {
          reject(error);
        }
      }, wait);

      // Resolve with last promise if available
      if (lastPromise) {
        lastPromise.then(resolve).catch(reject);
      }
    });
  };
};

