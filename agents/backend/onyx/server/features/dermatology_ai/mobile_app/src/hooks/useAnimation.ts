import { useRef } from 'react';
import { Animated } from 'react-native';

export const useAnimation = (initialValue: number = 0) => {
  const animValue = useRef(new Animated.Value(initialValue)).current;

  const animate = (
    toValue: number,
    duration: number = 300,
    easing?: (value: number) => number
  ) => {
    return Animated.timing(animValue, {
      toValue,
      duration,
      easing: easing || undefined,
      useNativeDriver: true,
    });
  };

  const spring = (
    toValue: number,
    config?: {
      tension?: number;
      friction?: number;
    }
  ) => {
    return Animated.spring(animValue, {
      toValue,
      useNativeDriver: true,
      ...config,
    });
  };

  const sequence = (animations: Animated.CompositeAnimation[]) => {
    return Animated.sequence(animations);
  };

  const parallel = (animations: Animated.CompositeAnimation[]) => {
    return Animated.parallel(animations);
  };

  const loop = (
    animation: Animated.CompositeAnimation,
    iterations?: number
  ) => {
    return Animated.loop(animation, { iterations });
  };

  const stop = () => {
    animValue.stopAnimation();
  };

  const reset = () => {
    animValue.setValue(initialValue);
  };

  return {
    value: animValue,
    animate,
    spring,
    sequence,
    parallel,
    loop,
    stop,
    reset,
  };
};

