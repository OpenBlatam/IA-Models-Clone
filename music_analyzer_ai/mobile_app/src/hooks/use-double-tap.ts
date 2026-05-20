import { useRef, useCallback } from 'react';

interface UseDoubleTapOptions {
  onDoubleTap: () => void;
  onSingleTap?: () => void;
  delay?: number;
}

/**
 * Hook for double tap detection
 * Distinguishes between single and double tap
 */
export function useDoubleTap({
  onDoubleTap,
  onSingleTap,
  delay = 300,
}: UseDoubleTapOptions) {
  const lastTapRef = useRef<number | null>(null);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  const handleTap = useCallback(() => {
    const now = Date.now();
    const lastTap = lastTapRef.current;

    if (lastTap && now - lastTap < delay) {
      // Double tap detected
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
      onDoubleTap();
      lastTapRef.current = null;
    } else {
      // Wait to see if it's a double tap
      lastTapRef.current = now;
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      timeoutRef.current = setTimeout(() => {
        if (onSingleTap) {
          onSingleTap();
        }
        lastTapRef.current = null;
        timeoutRef.current = null;
      }, delay);
    }
  }, [onDoubleTap, onSingleTap, delay]);

  return {
    onPress: handleTap,
  };
}

