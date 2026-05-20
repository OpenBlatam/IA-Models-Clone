import { useRef, useCallback } from 'react';
import { GestureResponderEvent } from 'react-native';

interface TapPosition {
  x: number;
  y: number;
}

interface UseTapGestureOptions {
  onTap?: (position: TapPosition) => void;
  onDoubleTap?: (position: TapPosition) => void;
  onLongPress?: (position: TapPosition) => void;
  tapDelay?: number;
  longPressDelay?: number;
}

/**
 * Hook for comprehensive tap gesture detection
 * Handles tap, double tap, and long press
 */
export function useTapGesture({
  onTap,
  onDoubleTap,
  onLongPress,
  tapDelay = 300,
  longPressDelay = 500,
}: UseTapGestureOptions = {}) {
  const lastTapRef = useRef<{ time: number; position: TapPosition } | null>(
    null
  );
  const longPressTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const tapTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const startPositionRef = useRef<TapPosition | null>(null);

  const handlePressIn = useCallback(
    (event: GestureResponderEvent) => {
      const position = {
        x: event.nativeEvent.pageX,
        y: event.nativeEvent.pageY,
      };
      startPositionRef.current = position;

      if (onLongPress) {
        longPressTimeoutRef.current = setTimeout(() => {
          onLongPress(position);
          longPressTimeoutRef.current = null;
        }, longPressDelay);
      }
    },
    [onLongPress, longPressDelay]
  );

  const handlePressOut = useCallback(
    (event: GestureResponderEvent) => {
      const position = {
        x: event.nativeEvent.pageX,
        y: event.nativeEvent.pageY,
      };

      if (longPressTimeoutRef.current) {
        clearTimeout(longPressTimeoutRef.current);
        longPressTimeoutRef.current = null;
      }

      if (startPositionRef.current) {
        const dx = Math.abs(position.x - startPositionRef.current.x);
        const dy = Math.abs(position.y - startPositionRef.current.y);

        // Check if moved significantly
        if (dx < 10 && dy < 10) {
          const now = Date.now();
          const lastTap = lastTapRef.current;

          if (lastTap && now - lastTap.time < tapDelay) {
            // Double tap
            if (tapTimeoutRef.current) {
              clearTimeout(tapTimeoutRef.current);
              tapTimeoutRef.current = null;
            }
            if (onDoubleTap) {
              onDoubleTap(position);
            }
            lastTapRef.current = null;
          } else {
            // Single tap (wait to confirm)
            lastTapRef.current = { time: now, position };
            if (onTap) {
              tapTimeoutRef.current = setTimeout(() => {
                onTap(position);
                lastTapRef.current = null;
                tapTimeoutRef.current = null;
              }, tapDelay);
            }
          }
        }
      }

      startPositionRef.current = null;
    },
    [onTap, onDoubleTap, tapDelay]
  );

  return {
    onPressIn: handlePressIn,
    onPressOut: handlePressOut,
  };
}

