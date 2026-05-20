import { useRef, useCallback, useState } from 'react';

interface UsePinchOptions {
  onPinchStart?: (event: TouchEvent) => void;
  onPinch?: (event: TouchEvent, scale: number) => void;
  onPinchEnd?: (event: TouchEvent) => void;
  minScale?: number;
  maxScale?: number;
}

export const usePinch = (options: UsePinchOptions = {}) => {
  const { onPinchStart, onPinch, onPinchEnd, minScale = 0.5, maxScale = 3 } = options;
  const [isPinching, setIsPinching] = useState(false);
  const initialDistanceRef = useRef<number | null>(null);
  const lastScaleRef = useRef<number>(1);

  const getDistance = useCallback((touch1: Touch, touch2: Touch): number => {
    const dx = touch2.clientX - touch1.clientX;
    const dy = touch2.clientY - touch1.clientY;
    return Math.sqrt(dx * dx + dy * dy);
  }, []);

  const handleStart = useCallback(
    (event: TouchEvent) => {
      if (event.touches.length !== 2) return;

      const distance = getDistance(event.touches[0], event.touches[1]);
      initialDistanceRef.current = distance;
      lastScaleRef.current = 1;
      setIsPinching(true);
      onPinchStart?.(event);
    },
    [getDistance, onPinchStart]
  );

  const handleMove = useCallback(
    (event: TouchEvent) => {
      if (!initialDistanceRef.current || event.touches.length !== 2) return;

      const distance = getDistance(event.touches[0], event.touches[1]);
      const scale = distance / initialDistanceRef.current;
      const clampedScale = Math.max(minScale, Math.min(maxScale, scale));

      onPinch?.(event, clampedScale);
      lastScaleRef.current = clampedScale;
    },
    [getDistance, onPinch, minScale, maxScale]
  );

  const handleEnd = useCallback(
    (event: TouchEvent) => {
      initialDistanceRef.current = null;
      setIsPinching(false);
      onPinchEnd?.(event);
    },
    [onPinchEnd]
  );

  const pinchProps = {
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
  };

  return { pinchProps, isPinching, scale: lastScaleRef.current };
};

