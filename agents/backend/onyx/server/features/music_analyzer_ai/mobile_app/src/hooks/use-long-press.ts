import { useRef, useCallback } from 'react';

interface UseLongPressOptions {
  onLongPress: () => void;
  onPress?: () => void;
  delay?: number;
  threshold?: number;
}

/**
 * Hook for long press detection
 * Distinguishes between tap and long press
 */
export function useLongPress({
  onLongPress,
  onPress,
  delay = 500,
  threshold = 10,
}: UseLongPressOptions) {
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  const startPositionRef = useRef<{ x: number; y: number } | null>(null);
  const hasMovedRef = useRef(false);

  const handlePressIn = useCallback(
    (event: { nativeEvent: { pageX: number; pageY: number } }) => {
      startPositionRef.current = {
        x: event.nativeEvent.pageX,
        y: event.nativeEvent.pageY,
      };
      hasMovedRef.current = false;

      timeoutRef.current = setTimeout(() => {
        if (!hasMovedRef.current) {
          onLongPress();
        }
      }, delay);
    },
    [onLongPress, delay]
  );

  const handlePressOut = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }

    if (!hasMovedRef.current && onPress) {
      onPress();
    }

    startPositionRef.current = null;
    hasMovedRef.current = false;
  }, [onPress]);

  const handlePressMove = useCallback(
    (event: { nativeEvent: { pageX: number; pageY: number } }) => {
      if (startPositionRef.current) {
        const dx = Math.abs(
          event.nativeEvent.pageX - startPositionRef.current.x
        );
        const dy = Math.abs(
          event.nativeEvent.pageY - startPositionRef.current.y
        );

        if (dx > threshold || dy > threshold) {
          hasMovedRef.current = true;
          if (timeoutRef.current) {
            clearTimeout(timeoutRef.current);
            timeoutRef.current = null;
          }
        }
      }
    },
    [threshold]
  );

  return {
    onPressIn: handlePressIn,
    onPressOut: handlePressOut,
    onPressMove: handlePressMove,
  };
}

