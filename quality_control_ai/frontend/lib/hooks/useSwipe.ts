import { useRef, useCallback, useState } from 'react';

interface UseSwipeOptions {
  onSwipeLeft?: () => void;
  onSwipeRight?: () => void;
  onSwipeUp?: () => void;
  onSwipeDown?: () => void;
  threshold?: number;
  velocityThreshold?: number;
}

export const useSwipe = (options: UseSwipeOptions = {}) => {
  const {
    onSwipeLeft,
    onSwipeRight,
    onSwipeUp,
    onSwipeDown,
    threshold = 50,
    velocityThreshold = 0.3,
  } = options;

  const [isSwiping, setIsSwiping] = useState(false);
  const startPosRef = useRef<{ x: number; y: number; time: number } | null>(null);
  const lastPosRef = useRef<{ x: number; y: number; time: number } | null>(null);

  const getEventPos = useCallback((event: TouchEvent | MouseEvent) => {
    if ('touches' in event && event.touches.length > 0) {
      return { x: event.touches[0].clientX, y: event.touches[0].clientY };
    }
    if ('clientX' in event) {
      return { x: event.clientX, y: event.clientY };
    }
    return { x: 0, y: 0 };
  }, []);

  const handleStart = useCallback(
    (event: TouchEvent | MouseEvent) => {
      const pos = getEventPos(event);
      const time = Date.now();
      startPosRef.current = { ...pos, time };
      lastPosRef.current = { ...pos, time };
      setIsSwiping(true);
    },
    [getEventPos]
  );

  const handleMove = useCallback(
    (event: TouchEvent | MouseEvent) => {
      if (!startPosRef.current || !lastPosRef.current) return;

      const pos = getEventPos(event);
      const time = Date.now();
      lastPosRef.current = { ...pos, time };
    },
    [getEventPos]
  );

  const handleEnd = useCallback(
    (event: TouchEvent | MouseEvent) => {
      if (!startPosRef.current || !lastPosRef.current) return;

      const pos = getEventPos(event);
      const time = Date.now();

      const deltaX = pos.x - startPosRef.current.x;
      const deltaY = pos.y - startPosRef.current.y;
      const deltaTime = time - startPosRef.current.time;

      const distance = Math.sqrt(deltaX ** 2 + deltaY ** 2);
      const velocity = distance / deltaTime;

      const absDeltaX = Math.abs(deltaX);
      const absDeltaY = Math.abs(deltaY);

      if (distance >= threshold && velocity >= velocityThreshold) {
        if (absDeltaX > absDeltaY) {
          if (deltaX > 0) {
            onSwipeRight?.();
          } else {
            onSwipeLeft?.();
          }
        } else {
          if (deltaY > 0) {
            onSwipeDown?.();
          } else {
            onSwipeUp?.();
          }
        }
      }

      startPosRef.current = null;
      lastPosRef.current = null;
      setIsSwiping(false);
    },
    [getEventPos, threshold, velocityThreshold, onSwipeLeft, onSwipeRight, onSwipeUp, onSwipeDown]
  );

  const swipeProps = {
    onTouchStart: (e: React.TouchEvent) => {
      handleStart(e.nativeEvent);
      const handleTouchMove = (moveEvent: TouchEvent) => handleMove(moveEvent);
      const handleTouchEnd = (endEvent: TouchEvent) => {
        handleEnd(endEvent);
        document.removeEventListener('touchmove', handleTouchMove);
        document.removeEventListener('touchend', handleTouchEnd);
      };
      document.addEventListener('touchmove', handleTouchMove);
      document.addEventListener('touchend', handleTouchEnd);
    },
    onMouseDown: (e: React.MouseEvent) => {
      handleStart(e.nativeEvent);
      const handleMouseMove = (moveEvent: MouseEvent) => handleMove(moveEvent);
      const handleMouseUp = (upEvent: MouseEvent) => {
        handleEnd(upEvent);
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    },
  };

  return { swipeProps, isSwiping };
};

