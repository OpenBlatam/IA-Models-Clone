import { useEffect, useRef } from 'react';

/**
 * Hook for running intervals safely
 * Cleans up on unmount and handles null delays
 */
export function useInterval(
  callback: () => void,
  delay: number | null
): void {
  const savedCallback = useRef<() => void>();

  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);

  useEffect(() => {
    function tick() {
      if (savedCallback.current) {
        savedCallback.current();
      }
    }

    if (delay !== null) {
      const id = setInterval(tick, delay);
      return () => clearInterval(id);
    }
  }, [delay]);
}

