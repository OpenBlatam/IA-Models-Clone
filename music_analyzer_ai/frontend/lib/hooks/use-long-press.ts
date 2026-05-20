/**
 * Custom hook for long press detection.
 * Detects when an element is pressed and held for a duration.
 */

import { useRef, useCallback, type MouseEvent, type TouchEvent } from 'react';

/**
 * Options for useLongPress hook.
 */
export interface UseLongPressOptions {
  onLongPress: (event: MouseEvent | TouchEvent) => void;
  onClick?: (event: MouseEvent | TouchEvent) => void;
  delay?: number;
  threshold?: number;
}

/**
 * Custom hook for long press detection.
 * Detects when an element is pressed and held.
 *
 * @param options - Hook options
 * @returns Event handlers
 */
export function useLongPress({
  onLongPress,
  onClick,
  delay = 400,
  threshold = 10,
}: UseLongPressOptions) {
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  const startPosRef = useRef<{ x: number; y: number } | null>(null);
  const isLongPressRef = useRef(false);

  const start = useCallback(
    (event: MouseEvent | TouchEvent) => {
      const clientX =
        'touches' in event ? event.touches[0].clientX : event.clientX;
      const clientY =
        'touches' in event ? event.touches[0].clientY : event.clientY;

      startPosRef.current = { x: clientX, y: clientY };
      isLongPressRef.current = false;

      timeoutRef.current = setTimeout(() => {
        isLongPressRef.current = true;
        onLongPress(event);
      }, delay);
    },
    [onLongPress, delay]
  );

  const clear = useCallback(
    (event: MouseEvent | TouchEvent, shouldTriggerClick = true) => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }

      if (shouldTriggerClick && !isLongPressRef.current) {
        const currentPos =
          'changedTouches' in event
            ? {
                x: event.changedTouches[0].clientX,
                y: event.changedTouches[0].clientY,
              }
            : { x: event.clientX, y: event.clientY };

        if (startPosRef.current) {
          const distance = Math.sqrt(
            Math.pow(currentPos.x - startPosRef.current.x, 2) +
              Math.pow(currentPos.y - startPosRef.current.y, 2)
          );

          if (distance < threshold && onClick) {
            onClick(event);
          }
        }
      }

      startPosRef.current = null;
    },
    [onClick, threshold]
  );

  return {
    onMouseDown: (e: MouseEvent) => start(e),
    onTouchStart: (e: TouchEvent) => start(e),
    onMouseUp: (e: MouseEvent) => clear(e),
    onMouseLeave: (e: MouseEvent) => clear(e, false),
    onTouchEnd: (e: TouchEvent) => clear(e),
    onTouchCancel: (e: TouchEvent) => clear(e, false),
  };
}

