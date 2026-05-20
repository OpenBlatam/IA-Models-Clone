/**
 * Custom hook for batching operations.
 * Provides reactive batching functionality.
 */

import { useCallback, useRef } from 'react';
import { batch, batchRAF, batchAsync } from '../utils/batch';

/**
 * Options for useBatch hook.
 */
export interface UseBatchOptions {
  delay?: number;
  useRAF?: boolean;
}

/**
 * Custom hook for batching operations.
 * Provides reactive batching functionality.
 *
 * @param fn - Function to batch
 * @param options - Batching options
 * @returns Batched function
 */
export function useBatch<T>(
  fn: (items: T[]) => void | Promise<void>,
  options: UseBatchOptions = {}
): (item: T) => void {
  const { delay = 0, useRAF = false } = options;
  const batcherRef = useRef<((item: T) => void) | null>(null);

  if (!batcherRef.current) {
    if (useRAF) {
      batcherRef.current = batchRAF(fn);
    } else {
      batcherRef.current = batch(fn, delay);
    }
  }

  return batcherRef.current;
}

/**
 * Custom hook for async batching.
 * Provides reactive async batching functionality.
 *
 * @param fn - Async function to batch
 * @param batchSize - Batch size
 * @returns Batched async function
 */
export function useBatchAsync<T, R>(
  fn: (item: T) => Promise<R>,
  batchSize: number = 10
) {
  return useCallback(
    async (items: T[]): Promise<R[]> => {
      return batchAsync(items, fn, batchSize);
    },
    [fn, batchSize]
  );
}

