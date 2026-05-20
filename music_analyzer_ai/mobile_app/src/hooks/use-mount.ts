import { useEffect, useRef } from 'react';

/**
 * Hook that runs callback only on mount
 * Useful for initialization logic
 */
export function useMount(callback: () => void | (() => void)): void {
  const isMountedRef = useRef(false);

  useEffect(() => {
    if (!isMountedRef.current) {
      isMountedRef.current = true;
      return callback();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
}

/**
 * Hook that runs callback only on unmount
 * Useful for cleanup logic
 */
export function useUnmount(callback: () => void): void {
  const callbackRef = useRef(callback);

  useEffect(() => {
    callbackRef.current = callback;
  }, [callback]);

  useEffect(() => {
    return () => {
      callbackRef.current();
    };
  }, []);
}

