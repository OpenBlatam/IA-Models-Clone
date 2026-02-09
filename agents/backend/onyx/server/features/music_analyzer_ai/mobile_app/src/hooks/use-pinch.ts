import { useRef, useCallback } from 'react';
import { PanResponder } from 'react-native';

interface UsePinchOptions {
  onPinchStart?: () => void;
  onPinch?: (scale: number) => void;
  onPinchEnd?: (scale: number) => void;
  minScale?: number;
  maxScale?: number;
}

/**
 * Hook for pinch gesture detection
 * Detects pinch-to-zoom gestures
 */
export function usePinch({
  onPinchStart,
  onPinch,
  onPinchEnd,
  minScale = 0.5,
  maxScale = 3,
}: UsePinchOptions = {}) {
  const initialDistanceRef = useRef<number | null>(null);
  const currentScaleRef = useRef(1);

  const getDistance = useCallback(
    (touches: Array<{ pageX: number; pageY: number }>) => {
      if (touches.length < 2) return null;

      const [touch1, touch2] = touches;
      const dx = touch2.pageX - touch1.pageX;
      const dy = touch2.pageY - touch1.pageY;
      return Math.sqrt(dx * dx + dy * dy);
    },
    []
  );

  const panResponder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: (_, gestureState) => {
        return gestureState.numberActiveTouches === 2;
      },
      onMoveShouldSetPanResponder: (_, gestureState) => {
        return gestureState.numberActiveTouches === 2;
      },
      onPanResponderGrant: (evt) => {
        const distance = getDistance(evt.nativeEvent.touches);
        if (distance !== null) {
          initialDistanceRef.current = distance;
          onPinchStart?.();
        }
      },
      onPanResponderMove: (evt) => {
        const distance = getDistance(evt.nativeEvent.touches);
        if (
          distance !== null &&
          initialDistanceRef.current !== null
        ) {
          const scale = distance / initialDistanceRef.current;
          const clampedScale = Math.max(
            minScale,
            Math.min(maxScale, scale)
          );
          currentScaleRef.current = clampedScale;
          onPinch?.(clampedScale);
        }
      },
      onPanResponderRelease: () => {
        onPinchEnd?.(currentScaleRef.current);
        initialDistanceRef.current = null;
      },
    })
  ).current;

  return {
    panHandlers: panResponder.panHandlers,
    scale: currentScaleRef.current,
  };
}

