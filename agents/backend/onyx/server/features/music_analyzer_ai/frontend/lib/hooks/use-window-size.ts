/**
 * Custom hook for window size detection.
 * Provides reactive window dimensions with debouncing.
 */

import { useState, useEffect } from 'react';
import { throttle } from '../utils/performance';

/**
 * Window size interface.
 */
export interface WindowSize {
  width: number;
  height: number;
}

/**
 * Options for useWindowSize hook.
 */
export interface UseWindowSizeOptions {
  debounceMs?: number;
}

/**
 * Custom hook for tracking window size.
 * Debounced for performance optimization.
 *
 * @param options - Hook options
 * @returns Window size dimensions
 */
export function useWindowSize(
  options: UseWindowSizeOptions = {}
): WindowSize {
  const { debounceMs = 100 } = options;

  const [windowSize, setWindowSize] = useState<WindowSize>(() => {
    if (typeof window === 'undefined') {
      return { width: 0, height: 0 };
    }
    return {
      width: window.innerWidth,
      height: window.innerHeight,
    };
  });

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    const handleResize = throttle(() => {
      setWindowSize({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    }, debounceMs);

    window.addEventListener('resize', handleResize, { passive: true });

    // Initial check
    handleResize();

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [debounceMs]);

  return windowSize;
}

