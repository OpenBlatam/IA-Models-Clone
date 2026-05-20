import { useRef, useCallback } from 'react';
import { PanResponder, GestureResponderEvent } from 'react-native';

interface SwipeDirection {
  horizontal: 'left' | 'right' | null;
  vertical: 'up' | 'down' | null;
}

interface UseSwipeOptions {
  onSwipeLeft?: () => void;
  onSwipeRight?: () => void;
  onSwipeUp?: () => void;
  onSwipeDown?: () => void;
  threshold?: number;
  velocityThreshold?: number;
}

/**
 * Hook for swipe gesture detection
 * Detects swipe directions with velocity
 */
export function useSwipe({
  onSwipeLeft,
  onSwipeRight,
  onSwipeUp,
  onSwipeDown,
  threshold = 50,
  velocityThreshold = 0.3,
}: UseSwipeOptions = {}) {
  const panResponder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => true,
      onMoveShouldSetPanResponder: () => true,
      onPanResponderRelease: (_, gestureState) => {
        const { dx, dy, vx, vy } = gestureState;

        const absDx = Math.abs(dx);
        const absDy = Math.abs(dy);
        const absVx = Math.abs(vx);
        const absVy = Math.abs(vy);

        // Determine primary direction
        if (absDx > absDy) {
          // Horizontal swipe
          if (absDx > threshold && absVx > velocityThreshold) {
            if (dx > 0) {
              onSwipeRight?.();
            } else {
              onSwipeLeft?.();
            }
          }
        } else {
          // Vertical swipe
          if (absDy > threshold && absVy > velocityThreshold) {
            if (dy > 0) {
              onSwipeDown?.();
            } else {
              onSwipeUp?.();
            }
          }
        }
      },
    })
  ).current;

  return {
    panHandlers: panResponder.panHandlers,
  };
}

