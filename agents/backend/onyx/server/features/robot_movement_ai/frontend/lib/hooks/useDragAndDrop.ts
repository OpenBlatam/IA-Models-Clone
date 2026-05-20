import { useState, useCallback, useRef, useEffect } from 'react';

export interface UseDragAndDropOptions<T> {
  onDragStart?: (item: T) => void;
  onDragEnd?: (item: T) => void;
  onDrop?: (item: T, targetIndex: number) => void;
  onReorder?: (fromIndex: number, toIndex: number) => void;
}

export interface UseDragAndDropReturn<T> {
  draggedItem: T | null;
  draggedIndex: number | null;
  isDragging: boolean;
  handleDragStart: (item: T, index: number) => void;
  handleDragEnd: () => void;
  handleDragOver: (e: React.DragEvent, index: number) => void;
  handleDrop: (e: React.DragEvent, index: number) => void;
}

/**
 * Hook for drag and drop functionality
 */
export function useDragAndDrop<T>(
  items: T[],
  options: UseDragAndDropOptions<T> = {}
): UseDragAndDropReturn<T> {
  const { onDragStart, onDragEnd, onDrop, onReorder } = options;
  const [draggedItem, setDraggedItem] = useState<T | null>(null);
  const [draggedIndex, setDraggedIndex] = useState<number | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleDragStart = useCallback(
    (item: T, index: number) => {
      setDraggedItem(item);
      setDraggedIndex(index);
      setIsDragging(true);
      if (onDragStart) {
        onDragStart(item);
      }
    },
    [onDragStart]
  );

  const handleDragEnd = useCallback(() => {
    if (draggedItem && onDragEnd) {
      onDragEnd(draggedItem);
    }
    setDraggedItem(null);
    setDraggedIndex(null);
    setIsDragging(false);
  }, [draggedItem, onDragEnd]);

  const handleDragOver = useCallback((e: React.DragEvent, index: number) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent, targetIndex: number) => {
      e.preventDefault();
      if (draggedIndex === null || draggedItem === null) return;

      if (draggedIndex !== targetIndex) {
        if (onReorder) {
          onReorder(draggedIndex, targetIndex);
        }
        if (onDrop) {
          onDrop(draggedItem, targetIndex);
        }
      }

      handleDragEnd();
    },
    [draggedIndex, draggedItem, onReorder, onDrop, handleDragEnd]
  );

  return {
    draggedItem,
    draggedIndex,
    isDragging,
    handleDragStart,
    handleDragEnd,
    handleDragOver,
    handleDrop,
  };
}



