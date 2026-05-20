/**
 * Custom hook for rate limiting.
 * Provides reactive rate limiting functionality.
 */

import { useRef, useCallback } from 'react';
import { RateLimiter, createRateLimiter } from '../utils/rate-limit';

/**
 * Options for useRateLimit hook.
 */
export interface UseRateLimitOptions {
  maxRequests: number;
  windowMs: number;
}

/**
 * Custom hook for rate limiting.
 * Provides reactive rate limiting functionality.
 *
 * @param options - Rate limit options
 * @returns Rate limiter operations
 */
export function useRateLimit(options: UseRateLimitOptions) {
  const limiterRef = useRef<RateLimiter | null>(null);

  if (!limiterRef.current) {
    limiterRef.current = createRateLimiter(
      options.maxRequests,
      options.windowMs
    );
  }

  const execute = useCallback(
    async <T>(fn: () => Promise<T>): Promise<T> => {
      return limiterRef.current!.execute(fn);
    },
    []
  );

  const clear = useCallback(() => {
    limiterRef.current?.clear();
  }, []);

  const getQueueSize = useCallback(() => {
    return limiterRef.current?.getQueueSize() ?? 0;
  }, []);

  return {
    execute,
    clear,
    getQueueSize,
  };
}

