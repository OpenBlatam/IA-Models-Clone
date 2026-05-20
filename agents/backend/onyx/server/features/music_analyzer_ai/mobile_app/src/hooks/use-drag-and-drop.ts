import { useRef, useCallback } from 'react';
import { PanResponder, GestureResponderEvent } from 'react-native';

interface DragAndDropState {
  isDragging: boolean;
  position: { x: number; y: number };
}

interface UseDragAndDropOptions {
  onDragStart?: () => void;
  onDrag?: (position: { x: number; y: number }) => void;
  onDragEnd?: (position: { x: number; y: number }) => void;
  enabled?: boolean;
}

/**
 * Hook for drag and drop functionality
 * Provides drag state and position
 */
export function useDragAndDrop({
  onDragStart,
  onDrag,
  onDragEnd,
  enabled = true,
}: UseDragAndDropOptions = {}) {
  const positionRef = useRef({ x: 0, y: 0 });
  const isDraggingRef = useRef(false);

  const panResponder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => enabled,
      onMoveShouldSetPanResponder: () => enabled && isDraggingRef.current,
      onPanResponderGrant: () => {
        isDraggingRef.current = true;
        onDragStart?.();
      },
      onPanResponderMove: (_, gestureState) => {
        if (isDraggingRef.current) {
          const newPosition = {
            x: gestureState.dx,
            y: gestureState.dy,
          };
          positionRef.current = newPosition;
          onDrag?.(newPosition);
        }
      },
      onPanResponderRelease: () => {
        if (isDraggingRef.current) {
          isDraggingRef.current = false;
          onDragEnd?.(positionRef.current);
          positionRef.current = { x: 0, y: 0 };
        }
      },
    })
  ).current;

  return {
    panHandlers: panResponder.panHandlers,
    isDragging: isDraggingRef.current,
    position: positionRef.current,
  };
}

