import { useRef, useCallback, useState } from 'react';

interface UseDragOptions {
  onDragStart?: (event: MouseEvent | TouchEvent) => void;
  onDrag?: (event: MouseEvent | TouchEvent, deltaX: number, deltaY: number) => void;
  onDragEnd?: (event: MouseEvent | TouchEvent) => void;
  threshold?: number;
}

export const useDrag = (options: UseDragOptions = {}) => {
  const { onDragStart, onDrag, onDragEnd, threshold = 0 } = options;
  const [isDragging, setIsDragging] = useState(false);
  const startPosRef = useRef<{ x: number; y: number } | null>(null);
  const lastPosRef = useRef<{ x: number; y: number } | null>(null);

  const getEventPos = useCallback((event: MouseEvent | TouchEvent) => {
    if ('touches' in event) {
      return { x: event.touches[0].clientX, y: event.touches[0].clientY };
    }
    return { x: event.clientX, y: event.clientY };
  }, []);

  const handleStart = useCallback(
    (event: MouseEvent | TouchEvent) => {
      const pos = getEventPos(event);
      startPosRef.current = pos;
      lastPosRef.current = pos;
      setIsDragging(true);
      onDragStart?.(event);
    },
    [getEventPos, onDragStart]
  );

  const handleMove = useCallback(
    (event: MouseEvent | TouchEvent) => {
      if (!startPosRef.current || !lastPosRef.current) return;

      const pos = getEventPos(event);
      const deltaX = pos.x - lastPosRef.current.x;
      const deltaY = pos.y - lastPosRef.current.y;

      const totalDeltaX = pos.x - startPosRef.current.x;
      const totalDeltaY = pos.y - startPosRef.current.y;
      const totalDistance = Math.sqrt(totalDeltaX ** 2 + totalDeltaY ** 2);

      if (totalDistance >= threshold) {
        onDrag?.(event, deltaX, deltaY);
        lastPosRef.current = pos;
      }
    },
    [getEventPos, onDrag, threshold]
  );

  const handleEnd = useCallback(
    (event: MouseEvent | TouchEvent) => {
      startPosRef.current = null;
      lastPosRef.current = null;
      setIsDragging(false);
      onDragEnd?.(event);
    },
    [onDragEnd]
  );

  const dragProps = {
    onMouseDown: (e: React.MouseEvent) => {
      e.preventDefault();
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
    onTouchStart: (e: React.TouchEvent) => {
      e.preventDefault();
      handleStart(e.nativeEvent);
      const handleTouchMove = (moveEvent: TouchEvent) => handleMove(moveEvent);
      const handleTouchEnd = (upEvent: TouchEvent) => {
        handleEnd(upEvent);
        document.removeEventListener('touchmove', handleTouchMove);
        document.removeEventListener('touchend', handleTouchEnd);
      };
      document.addEventListener('touchmove', handleTouchMove);
      document.addEventListener('touchend', handleTouchEnd);
    },
  };

  return { dragProps, isDragging };
};

