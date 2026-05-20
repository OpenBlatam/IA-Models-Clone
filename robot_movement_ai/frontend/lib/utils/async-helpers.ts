/**
 * Async operation helpers
 */

import { handleApiError } from './error-handler';

/**
 * Create async operation with loading state
 */
export async function withLoading<T>(
  operation: () => Promise<T>,
  setLoading: (loading: boolean) => void
): Promise<T | null> {
  setLoading(true);
  try {
    return await operation();
  } catch (error) {
    handleApiError(error);
    return null;
  } finally {
    setLoading(false);
  }
}

/**
 * Create async operation with error handling
 */
export async function withErrorHandlingAsync<T>(
  operation: () => Promise<T>,
  onError?: (error: unknown) => void
): Promise<T | null> {
  try {
    return await operation();
  } catch (error) {
    if (onError) {
      onError(error);
    } else {
      handleApiError(error);
    }
    return null;
  }
}

/**
 * Batch async operations
 */
export async function batchAsync<T>(
  operations: Array<() => Promise<T>>,
  concurrency: number = 3
): Promise<T[]> {
  const results: T[] = [];
  
  for (let i = 0; i < operations.length; i += concurrency) {
    const batch = operations.slice(i, i + concurrency);
    const batchResults = await Promise.all(
      batch.map((op) => op().catch((error) => {
        handleApiError(error);
        return null as T;
      }))
    );
    results.push(...batchResults.filter((r) => r !== null));
  }
  
  return results;
}

/**
 * Race async operations
 */
export async function raceAsync<T>(
  operations: Array<() => Promise<T>>,
  timeout?: number
): Promise<T | null> {
  const promises = operations.map((op) => op());

  if (timeout) {
    const timeoutPromise = new Promise<T>((_, reject) => {
      setTimeout(() => reject(new Error('Timeout')), timeout);
    });
    promises.push(timeoutPromise);
  }

  try {
    return await Promise.race(promises);
  } catch (error) {
    handleApiError(error);
    return null;
  }
}

/**
 * Sequential async operations
 */
export async function sequentialAsync<T>(
  operations: Array<() => Promise<T>>
): Promise<T[]> {
  const results: T[] = [];

  for (const operation of operations) {
    try {
      const result = await operation();
      results.push(result);
    } catch (error) {
      handleApiError(error);
    }
  }

  return results;
}



