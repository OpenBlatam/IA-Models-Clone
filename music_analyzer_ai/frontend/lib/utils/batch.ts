/**
 * Batch utility functions.
 * Provides helper functions for batching operations.
 */

/**
 * Batches function calls.
 */
export function batch<T>(
  fn: (items: T[]) => void | Promise<void>,
  delay: number = 0
): (item: T) => void {
  let queue: T[] = [];
  let timeoutId: NodeJS.Timeout | null = null;

  const flush = () => {
    if (queue.length > 0) {
      const items = [...queue];
      queue = [];
      fn(items);
    }
    timeoutId = null;
  };

  return (item: T) => {
    queue.push(item);

    if (timeoutId === null) {
      timeoutId = setTimeout(flush, delay);
    }
  };
}

/**
 * Batches async function calls with concurrency limit.
 */
export async function batchAsync<T, R>(
  items: T[],
  fn: (item: T) => Promise<R>,
  batchSize: number = 10
): Promise<R[]> {
  const results: R[] = [];

  for (let i = 0; i < items.length; i += batchSize) {
    const batch = items.slice(i, i + batchSize);
    const batchResults = await Promise.all(batch.map(fn));
    results.push(...batchResults);
  }

  return results;
}

/**
 * Batches function calls with requestAnimationFrame.
 */
export function batchRAF<T>(
  fn: (items: T[]) => void
): (item: T) => void {
  let queue: T[] = [];
  let rafId: number | null = null;

  const flush = () => {
    if (queue.length > 0) {
      const items = [...queue];
      queue = [];
      fn(items);
    }
    rafId = null;
  };

  return (item: T) => {
    queue.push(item);

    if (rafId === null) {
      rafId = requestAnimationFrame(flush);
    }
  };
}

