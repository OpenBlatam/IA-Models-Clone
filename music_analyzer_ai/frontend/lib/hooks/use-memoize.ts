/**
 * Custom hook for function memoization.
 * Provides reactive memoization functionality.
 */

import { useMemo, useRef } from 'react';
import { memoize, memoizeLRU, memoizeTTL } from '../utils/memoization';

/**
 * Options for useMemoize hook.
 */
export interface UseMemoizeOptions {
  strategy?: 'simple' | 'lru' | 'ttl';
  maxSize?: number;
  ttl?: number;
  keyGenerator?: (...args: any[]) => string;
}

/**
 * Custom hook for function memoization.
 * Provides reactive memoization functionality.
 *
 * @param fn - Function to memoize
 * @param options - Memoization options
 * @returns Memoized function
 */
export function useMemoize<T extends (...args: any[]) => any>(
  fn: T,
  options: UseMemoizeOptions = {}
): T {
  const { strategy = 'simple', maxSize = 100, ttl = 60000, keyGenerator } = options;
  const memoizedRef = useRef<T | null>(null);

  return useMemo(() => {
    if (strategy === 'lru') {
      memoizedRef.current = memoizeLRU(fn, maxSize, keyGenerator) as T;
    } else if (strategy === 'ttl') {
      memoizedRef.current = memoizeTTL(fn, ttl, keyGenerator) as T;
    } else {
      memoizedRef.current = memoize(fn, keyGenerator) as T;
    }

    return memoizedRef.current;
  }, [fn, strategy, maxSize, ttl, keyGenerator]);
}

