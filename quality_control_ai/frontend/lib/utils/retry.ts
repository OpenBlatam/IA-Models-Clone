interface RetryOptions {
  retries?: number;
  delay?: number;
  backoff?: 'linear' | 'exponential';
  onRetry?: (error: Error, attempt: number) => void;
}

export const retry = async <T,>(
  fn: () => Promise<T>,
  options: RetryOptions = {}
): Promise<T> => {
  const {
    retries = 3,
    delay = 1000,
    backoff = 'linear',
    onRetry,
  } = options;

  let lastError: Error;

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      if (attempt < retries) {
        const waitTime =
          backoff === 'exponential'
            ? delay * Math.pow(2, attempt)
            : delay * (attempt + 1);

        onRetry?.(lastError, attempt + 1);
        await new Promise((resolve) => setTimeout(resolve, waitTime));
      }
    }
  }

  throw lastError!;
};

export const retrySync = <T,>(
  fn: () => T,
  options: RetryOptions = {}
): T => {
  const {
    retries = 3,
    delay = 1000,
    backoff = 'linear',
    onRetry,
  } = options;

  let lastError: Error;

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return fn();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      if (attempt < retries) {
        const waitTime =
          backoff === 'exponential'
            ? delay * Math.pow(2, attempt)
            : delay * (attempt + 1);

        onRetry?.(lastError, attempt + 1);
        // Note: This is a blocking retry for sync functions
        // In real scenarios, you might want to use async retry
      }
    }
  }

  throw lastError!;
};

