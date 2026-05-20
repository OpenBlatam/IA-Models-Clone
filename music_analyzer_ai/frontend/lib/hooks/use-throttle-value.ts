/**
 * Custom hook for throttled value.
 * Provides throttled value updates for performance optimization.
 */

import { useState, useEffect, useRef } from 'react';

/**
 * Options for useThrottleValue hook.
 */
export interface UseThrottleValueOptions {
  delay: number;
  leading?: boolean;
  trailing?: boolean;
}

/**
 * Custom hook for throttled value.
 * Throttles value updates to reduce re-renders.
 *
 * @param value - Value to throttle
 * @param options - Throttle options
 * @returns Throttled value
 */
export function useThrottleValue<T>(
  value: T,
  options: UseThrottleValueOptions
): T {
  const { delay, leading = true, trailing = true } = options;
  const [throttledValue, setThrottledValue] = useState<T>(value);
  const lastRan = useRef<number>(Date.now());
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    const now = Date.now();
    const timeSinceLastRun = now - lastRan.current;

    if (timeSinceLastRun >= delay) {
      if (leading) {
        setThrottledValue(value);
        lastRan.current = now;
      } else if (trailing) {
        if (timeoutRef.current) {
          clearTimeout(timeoutRef.current);
        }
        timeoutRef.current = setTimeout(() => {
          setThrottledValue(value);
          lastRan.current = Date.now();
        }, delay - timeSinceLastRun);
      }
    } else if (trailing) {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      timeoutRef.current = setTimeout(() => {
        setThrottledValue(value);
        lastRan.current = Date.now();
      }, delay - timeSinceLastRun);
    }

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [value, delay, leading, trailing]);

  return throttledValue;
}

