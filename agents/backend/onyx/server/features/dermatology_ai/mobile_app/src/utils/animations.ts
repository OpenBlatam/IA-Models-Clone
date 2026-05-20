import { Animated } from 'react-native';

/**
 * Fade in animation
 */
export const fadeIn = (
  animValue: Animated.Value,
  duration: number = 300,
  delay: number = 0
): Animated.CompositeAnimation => {
  return Animated.timing(animValue, {
    toValue: 1,
    duration,
    delay,
    useNativeDriver: true,
  });
};

/**
 * Fade out animation
 */
export const fadeOut = (
  animValue: Animated.Value,
  duration: number = 300
): Animated.CompositeAnimation => {
  return Animated.timing(animValue, {
    toValue: 0,
    duration,
    useNativeDriver: true,
  });
};

/**
 * Slide in from bottom
 */
export const slideInBottom = (
  animValue: Animated.Value,
  duration: number = 300
): Animated.CompositeAnimation => {
  return Animated.timing(animValue, {
    toValue: 0,
    duration,
    useNativeDriver: true,
  });
};

/**
 * Scale animation
 */
export const scale = (
  animValue: Animated.Value,
  toValue: number = 1.1,
  duration: number = 200
): Animated.CompositeAnimation => {
  return Animated.timing(animValue, {
    toValue,
    duration,
    useNativeDriver: true,
  });
};

/**
 * Pulse animation
 */
export const pulse = (
  animValue: Animated.Value,
  minValue: number = 1,
  maxValue: number = 1.2,
  duration: number = 1000
): Animated.CompositeAnimation => {
  return Animated.loop(
    Animated.sequence([
      Animated.timing(animValue, {
        toValue: maxValue,
        duration: duration / 2,
        useNativeDriver: true,
      }),
      Animated.timing(animValue, {
        toValue: minValue,
        duration: duration / 2,
        useNativeDriver: true,
      }),
    ])
  );
};

/**
 * Shake animation
 */
export const shake = (
  animValue: Animated.Value,
  intensity: number = 10,
  duration: number = 500
): Animated.CompositeAnimation => {
  const shakeAnim = new Animated.Value(0);
  return Animated.sequence([
    Animated.timing(shakeAnim, {
      toValue: intensity,
      duration: duration / 4,
      useNativeDriver: true,
    }),
    Animated.timing(shakeAnim, {
      toValue: -intensity,
      duration: duration / 4,
      useNativeDriver: true,
    }),
    Animated.timing(shakeAnim, {
      toValue: intensity,
      duration: duration / 4,
      useNativeDriver: true,
    }),
    Animated.timing(shakeAnim, {
      toValue: 0,
      duration: duration / 4,
      useNativeDriver: true,
    }),
  ]);
};

/**
 * Stagger animation for list items
 */
export const stagger = (
  animValues: Animated.Value[],
  delay: number = 100
): Animated.CompositeAnimation[] => {
  return animValues.map((anim, index) =>
    fadeIn(anim, 300, index * delay)
  );
};

