import { useState, useRef, useCallback } from 'react';

interface UseDragAndDropOptions {
  onDragStart?: (event: DragEvent) => void;
  onDragEnd?: (event: DragEvent) => void;
  onDragOver?: (event: DragEvent) => void;
  onDrop?: (event: DragEvent) => void;
  dragData?: unknown;
  dragType?: string;
}

export const useDragAndDrop = (options: UseDragAndDropOptions = {}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isOver, setIsOver] = useState(false);
  const dragRef = useRef<HTMLElement | null>(null);
  const dropRef = useRef<HTMLElement | null>(null);

  const handleDragStart = useCallback(
    (e: React.DragEvent) => {
      setIsDragging(true);
      if (options.dragData !== undefined) {
        e.dataTransfer.setData(options.dragType || 'text/plain', JSON.stringify(options.dragData));
      }
      if (options.onDragStart) {
        options.onDragStart(e.nativeEvent as DragEvent);
      }
    },
    [options]
  );

  const handleDragEnd = useCallback(
    (e: React.DragEvent) => {
      setIsDragging(false);
      if (options.onDragEnd) {
        options.onDragEnd(e.nativeEvent as DragEvent);
      }
    },
    [options]
  );

  const handleDragOver = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsOver(true);
      if (options.onDragOver) {
        options.onDragOver(e.nativeEvent as DragEvent);
      }
    },
    [options]
  );

  const handleDragLeave = useCallback(() => {
    setIsOver(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsOver(false);
      if (options.onDrop) {
        options.onDrop(e.nativeEvent as DragEvent);
      }
    },
    [options]
  );

  return {
    isDragging,
    isOver,
    dragRef,
    dropRef,
    dragProps: {
      draggable: true,
      onDragStart: handleDragStart,
      onDragEnd: handleDragEnd,
    },
    dropProps: {
      onDragOver: handleDragOver,
      onDragLeave: handleDragLeave,
      onDrop: handleDrop,
    },
  };
};



