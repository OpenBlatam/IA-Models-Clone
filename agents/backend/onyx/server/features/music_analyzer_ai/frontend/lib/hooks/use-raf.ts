/**
 * Custom hook for requestAnimationFrame.
 * Provides reactive RAF callback.
 */

import { useCallback, useRef, useEffect } from 'react';

/**
 * Custom hook for requestAnimationFrame.
 * Provides reactive RAF callback.
 *
 * @param callback - Callback function
 * @param active - Whether RAF is active
 * @returns RAF control functions
 */
export function useRaf(
  callback: (time: number) => void,
  active: boolean = true
) {
  const rafRef = useRef<number | null>(null);
  const callbackRef = useRef(callback);

  useEffect(() => {
    callbackRef.current = callback;
  }, [callback]);

  const stop = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (!active) {
      stop();
      return;
    }

    const animate = (time: number) => {
      callbackRef.current(time);
      rafRef.current = requestAnimationFrame(animate);
    };

    rafRef.current = requestAnimationFrame(animate);

    return stop;
  }, [active, stop]);

  return { stop };
}

