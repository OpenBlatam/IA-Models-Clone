import { useRef, useCallback } from 'react';
import { Animated, Easing } from 'react-native';

export function useAnimation(initialValue: number = 0) {
  const animValue = useRef(new Animated.Value(initialValue)).current;

  const animate = useCallback(
    (toValue: number, duration: number = 300, easing = Easing.out(Easing.cubic)) => {
      return Animated.timing(animValue, {
        toValue,
        duration,
        easing,
        useNativeDriver: true,
      });
    },
    [animValue]
  );

  const spring = useCallback(
    (toValue: number, config?: Animated.SpringAnimationConfig) => {
      return Animated.spring(animValue, {
        toValue,
        useNativeDriver: true,
        ...config,
      });
    },
    [animValue]
  );

  const sequence = useCallback(
    (animations: Animated.CompositeAnimation[]) => {
      return Animated.sequence(animations);
    },
    []
  );

  const parallel = useCallback(
    (animations: Animated.CompositeAnimation[]) => {
      return Animated.parallel(animations);
    },
    []
  );

  const reset = useCallback(
    (value: number = initialValue) => {
      animValue.setValue(value);
    },
    [animValue, initialValue]
  );

  return {
    value: animValue,
    animate,
    spring,
    sequence,
    parallel,
    reset,
  };
}


