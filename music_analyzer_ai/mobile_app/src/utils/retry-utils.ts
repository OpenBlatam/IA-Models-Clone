interface RetryOptions {
  maxRetries?: number;
  delay?: number;
  backoff?: 'linear' | 'exponential';
  onRetry?: (attempt: number, error: Error) => void;
}

/**
 * Retry a function with exponential or linear backoff
 */
export async function retry<T>(
  fn: () => Promise<T>,
  options: RetryOptions = {}
): Promise<T> {
  const {
    maxRetries = 3,
    delay = 1000,
    backoff = 'exponential',
    onRetry,
  } = options;

  let lastError: Error;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      if (attempt < maxRetries) {
        const waitTime =
          backoff === 'exponential'
            ? delay * Math.pow(2, attempt)
            : delay * (attempt + 1);

        if (onRetry) {
          onRetry(attempt + 1, lastError);
        }

        await new Promise((resolve) => setTimeout(resolve, waitTime));
      }
    }
  }

  throw lastError!;
}

/**
 * Retry with network error detection
 */
export async function retryWithNetworkCheck<T>(
  fn: () => Promise<T>,
  options: RetryOptions = {}
): Promise<T> {
  return retry(fn, {
    ...options,
    onRetry: (attempt, error) => {
      if (options.onRetry) {
        options.onRetry(attempt, error);
      }

      // Check if it's a network error
      const isNetworkError =
        error.message.includes('network') ||
        error.message.includes('timeout') ||
        error.message.includes('ECONNREFUSED');

      if (!isNetworkError && attempt < (options.maxRetries ?? 3)) {
        // Don't retry non-network errors
        throw error;
      }
    },
  });
}

