/**
 * Custom hook for async queue
 * 
 * Provides React integration for async queues
 */

import { useRef, useCallback } from "react";
import { AsyncQueue, type QueueOptions } from "../utils/async/queue";

/**
 * Return type for useAsyncQueue hook
 */
export interface UseAsyncQueueReturn<T> {
  /** Add task to queue */
  readonly add: (task: () => Promise<T>) => Promise<T>;
  /** Queue size */
  readonly size: number;
  /** Running tasks count */
  readonly running: number;
  /** Clear queue */
  readonly clear: () => void;
  /** Wait for all tasks */
  readonly wait: () => Promise<void>;
}

/**
 * Custom hook for async queue
 * 
 * @param options - Queue options
 * @returns Queue operations and state
 * 
 * @example
 * ```typescript
 * const { add, size, running } = useAsyncQueue({ concurrency: 3 });
 * 
 * await add(() => fetchAgent("1"));
 * await add(() => fetchAgent("2"));
 * ```
 */
export function useAsyncQueue<T = unknown>(
  options: QueueOptions = {}
): UseAsyncQueueReturn<T> {
  const queueRef = useRef<AsyncQueue<T>>(new AsyncQueue<T>(options));

  const add = useCallback(
    async (task: () => Promise<T>): Promise<T> => {
      return queueRef.current.add(task);
    },
    []
  );

  const clear = useCallback(() => {
    queueRef.current.clear();
  }, []);

  const wait = useCallback(async (): Promise<void> => {
    await queueRef.current.wait();
  }, []);

  return {
    add,
    size: queueRef.current.size(),
    running: queueRef.current.runningCount(),
    clear,
    wait,
  };
}




