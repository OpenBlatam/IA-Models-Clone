import { useRef, useCallback } from 'react';

interface UseLongPressOptions {
  onLongPress: () => void;
  onClick?: () => void;
  delay?: number;
  threshold?: number;
}

export const useLongPress = ({
  onLongPress,
  onClick,
  delay = 500,
  threshold = 10,
}: UseLongPressOptions) => {
  const timeoutRef = useRef<NodeJS.Timeout>();
  const startPosRef = useRef<{ x: number; y: number } | null>(null);
  const isLongPressRef = useRef(false);

  const start = useCallback(
    (e: React.MouseEvent | React.TouchEvent) => {
      isLongPressRef.current = false;
      const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
      const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;
      startPosRef.current = { x: clientX, y: clientY };

      timeoutRef.current = setTimeout(() => {
        isLongPressRef.current = true;
        onLongPress();
      }, delay);
    },
    [onLongPress, delay]
  );

  const clear = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
  }, []);

  const move = useCallback(
    (e: React.MouseEvent | React.TouchEvent) => {
      if (!startPosRef.current) return;

      const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
      const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;

      const deltaX = Math.abs(clientX - startPosRef.current.x);
      const deltaY = Math.abs(clientY - startPosRef.current.y);

      if (deltaX > threshold || deltaY > threshold) {
        clear();
      }
    },
    [threshold, clear]
  );

  const end = useCallback(
    (e: React.MouseEvent | React.TouchEvent) => {
      clear();
      if (!isLongPressRef.current && onClick) {
        onClick();
      }
      isLongPressRef.current = false;
      startPosRef.current = null;
    },
    [onClick, clear]
  );

  return {
    onMouseDown: start,
    onTouchStart: start,
    onMouseMove: move,
    onTouchMove: move,
    onMouseUp: end,
    onTouchEnd: end,
    onMouseLeave: clear,
    onTouchCancel: clear,
  };
};

