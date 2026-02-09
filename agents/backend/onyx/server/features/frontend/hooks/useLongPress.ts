'use client';

import { useRef, useCallback } from 'react';

interface UseLongPressOptions {
  onLongPress: () => void;
  onClick?: () => void;
  delay?: number;
  preventDefault?: boolean;
}

export function useLongPress({
  onLongPress,
  onClick,
  delay = 500,
  preventDefault = true,
}: UseLongPressOptions) {
  const timeoutRef = useRef<NodeJS.Timeout>();
  const targetRef = useRef<EventTarget>();

  const start = useCallback(
    (event: React.MouseEvent | React.TouchEvent) => {
      if (preventDefault) {
        event.preventDefault();
      }
      targetRef.current = event.target;
      timeoutRef.current = setTimeout(() => {
        onLongPress();
      }, delay);
    },
    [onLongPress, delay, preventDefault]
  );

  const clear = useCallback(
    (event: React.MouseEvent | React.TouchEvent, shouldTriggerClick = true) => {
      timeoutRef.current && clearTimeout(timeoutRef.current);
      if (shouldTriggerClick && onClick && event.target === targetRef.current) {
        onClick();
      }
    },
    [onClick]
  );

  return {
    onMouseDown: (e: React.MouseEvent) => start(e),
    onTouchStart: (e: React.TouchEvent) => start(e),
    onMouseUp: (e: React.MouseEvent) => clear(e),
    onMouseLeave: (e: React.MouseEvent) => clear(e, false),
    onTouchEnd: (e: React.TouchEvent) => clear(e),
  };
}

