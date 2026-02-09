import { useEffect, useRef } from 'react';

/**
 * Hook for running timeouts safely
 * Cleans up on unmount and handles null delays
 */
export function useTimeout(
  callback: () => void,
  delay: number | null
): void {
  const savedCallback = useRef<() => void>();

  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);

  useEffect(() => {
    if (delay !== null) {
      const id = setTimeout(() => {
        if (savedCallback.current) {
          savedCallback.current();
        }
      }, delay);

      return () => clearTimeout(id);
    }
  }, [delay]);
}

