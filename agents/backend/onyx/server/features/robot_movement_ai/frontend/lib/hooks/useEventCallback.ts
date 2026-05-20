import { useRef, useCallback } from 'react';

/**
 * Hook that returns a memoized callback that always has the latest values
 * without causing re-renders
 */
export function useEventCallback<T extends (...args: any[]) => any>(fn: T): T {
  const ref = useRef<T>(fn);

  // Update ref on every render
  ref.current = fn;

  // Return stable callback
  return useCallback(
    ((...args: any[]) => {
      return ref.current(...args);
    }) as T,
    []
  );
}



