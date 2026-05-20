import { useRef } from 'react';
import { PanResponder, GestureResponderEvent } from 'react-native';

interface GestureCallbacks {
  onSwipeLeft?: () => void;
  onSwipeRight?: () => void;
  onSwipeUp?: () => void;
  onSwipeDown?: () => void;
  onTap?: () => void;
  onLongPress?: () => void;
  onPinch?: (scale: number) => void;
  swipeThreshold?: number;
  longPressDelay?: number;
}

export const useGesture = (callbacks: GestureCallbacks) => {
  const {
    onSwipeLeft,
    onSwipeRight,
    onSwipeUp,
    onSwipeDown,
    onTap,
    onLongPress,
    swipeThreshold = 50,
    longPressDelay = 500,
  } = callbacks;

  const longPressTimer = useRef<NodeJS.Timeout | null>(null);
  const startPosition = useRef({ x: 0, y: 0 });

  const panResponder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => true,
      onMoveShouldSetPanResponder: () => true,
      onPanResponderGrant: (event: GestureResponderEvent) => {
        startPosition.current = {
          x: event.nativeEvent.locationX,
          y: event.nativeEvent.locationY,
        };

        if (onLongPress) {
          longPressTimer.current = setTimeout(() => {
            onLongPress();
          }, longPressDelay);
        }
      },
      onPanResponderMove: () => {
        if (longPressTimer.current) {
          clearTimeout(longPressTimer.current);
          longPressTimer.current = null;
        }
      },
      onPanResponderRelease: (event: GestureResponderEvent) => {
        if (longPressTimer.current) {
          clearTimeout(longPressTimer.current);
          longPressTimer.current = null;
        }

        const { locationX, locationY } = event.nativeEvent;
        const dx = locationX - startPosition.current.x;
        const dy = locationY - startPosition.current.y;

        if (Math.abs(dx) < 10 && Math.abs(dy) < 10) {
          onTap?.();
          return;
        }

        if (Math.abs(dx) > Math.abs(dy)) {
          if (Math.abs(dx) > swipeThreshold) {
            if (dx > 0) {
              onSwipeRight?.();
            } else {
              onSwipeLeft?.();
            }
          }
        } else {
          if (Math.abs(dy) > swipeThreshold) {
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

  return panResponder;
};

