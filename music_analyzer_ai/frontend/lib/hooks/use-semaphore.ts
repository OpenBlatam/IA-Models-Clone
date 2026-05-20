/**
 * Custom hook for semaphore.
 * Provides reactive semaphore functionality.
 */

import { useRef, useCallback } from 'react';
import { Semaphore, createSemaphore } from '../utils/semaphore';

/**
 * Custom hook for semaphore.
 * Provides reactive semaphore functionality.
 *
 * @param permits - Number of permits
 * @returns Semaphore operations
 */
export function useSemaphore(permits: number) {
  const semaphoreRef = useRef<Semaphore | null>(null);

  if (!semaphoreRef.current) {
    semaphoreRef.current = createSemaphore(permits);
  }

  const acquire = useCallback(async (): Promise<void> => {
    return semaphoreRef.current!.acquire();
  }, []);

  const release = useCallback((): void => {
    semaphoreRef.current?.release();
  }, []);

  const execute = useCallback(
    async <T>(fn: () => Promise<T>): Promise<T> => {
      return semaphoreRef.current!.execute(fn);
    },
    []
  );

  const availablePermits = useCallback((): number => {
    return semaphoreRef.current?.availablePermits() ?? 0;
  }, []);

  const getQueueLength = useCallback((): number => {
    return semaphoreRef.current?.getQueueLength() ?? 0;
  }, []);

  return {
    acquire,
    release,
    execute,
    availablePermits,
    getQueueLength,
  };
}

