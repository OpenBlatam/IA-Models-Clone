import { useRef, useCallback } from 'react';
import { Animated } from 'react-native';

/**
 * Hook for common animations
 * Provides reusable animation functions
 */
export function useAnimation() {
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const scaleAnim = useRef(new Animated.Value(1)).current;
  const slideAnim = useRef(new Animated.Value(0)).current;

  const fadeIn = useCallback(
    (duration = 300, callback?: () => void) => {
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration,
        useNativeDriver: true,
      }).start(callback);
    },
    [fadeAnim]
  );

  const fadeOut = useCallback(
    (duration = 300, callback?: () => void) => {
      Animated.timing(fadeAnim, {
        toValue: 0,
        duration,
        useNativeDriver: true,
      }).start(callback);
    },
    [fadeAnim]
  );

  const scaleIn = useCallback(
    (duration = 300, callback?: () => void) => {
      Animated.spring(scaleAnim, {
        toValue: 1,
        friction: 8,
        tension: 40,
        useNativeDriver: true,
      }).start(callback);
    },
    [scaleAnim]
  );

  const scaleOut = useCallback(
    (duration = 300, callback?: () => void) => {
      Animated.timing(scaleAnim, {
        toValue: 0,
        duration,
        useNativeDriver: true,
      }).start(callback);
    },
    [scaleAnim]
  );

  const slideUp = useCallback(
    (distance = 100, duration = 300, callback?: () => void) => {
      Animated.timing(slideAnim, {
        toValue: -distance,
        duration,
        useNativeDriver: true,
      }).start(callback);
    },
    [slideAnim]
  );

  const slideDown = useCallback(
    (distance = 100, duration = 300, callback?: () => void) => {
      Animated.timing(slideAnim, {
        toValue: distance,
        duration,
        useNativeDriver: true,
      }).start(callback);
    },
    [slideAnim]
  );

  const reset = useCallback(() => {
    fadeAnim.setValue(0);
    scaleAnim.setValue(1);
    slideAnim.setValue(0);
  }, [fadeAnim, scaleAnim, slideAnim]);

  return {
    fadeAnim,
    scaleAnim,
    slideAnim,
    fadeIn,
    fadeOut,
    scaleIn,
    scaleOut,
    slideUp,
    slideDown,
    reset,
  };
}

