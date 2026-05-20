import { useState, useCallback, useRef } from 'react';
import { Animated } from 'react-native';

interface UseCollapsibleOptions {
  defaultExpanded?: boolean;
  duration?: number;
  onToggle?: (expanded: boolean) => void;
}

export function useCollapsible(options: UseCollapsibleOptions = {}) {
  const { defaultExpanded = false, duration = 300, onToggle } = options;
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);
  const animation = useRef(new Animated.Value(defaultExpanded ? 1 : 0)).current;
  const contentHeight = useRef<number>(0);

  const toggle = useCallback(() => {
    const newExpanded = !isExpanded;
    setIsExpanded(newExpanded);
    onToggle?.(newExpanded);

    Animated.timing(animation, {
      toValue: newExpanded ? 1 : 0,
      duration,
      useNativeDriver: false,
    }).start();
  }, [isExpanded, onToggle, duration, animation]);

  const expand = useCallback(() => {
    if (!isExpanded) {
      toggle();
    }
  }, [isExpanded, toggle]);

  const collapse = useCallback(() => {
    if (isExpanded) {
      toggle();
    }
  }, [isExpanded, toggle]);

  const setExpanded = useCallback(
    (expanded: boolean) => {
      if (expanded !== isExpanded) {
        toggle();
      }
    },
    [isExpanded, toggle]
  );

  const heightInterpolate = animation.interpolate({
    inputRange: [0, 1],
    outputRange: [0, contentHeight.current || 1000],
  });

  const opacityInterpolate = animation.interpolate({
    inputRange: [0, 0.5, 1],
    outputRange: [0, 0, 1],
  });

  const rotateInterpolate = animation.interpolate({
    inputRange: [0, 1],
    outputRange: ['0deg', '180deg'],
  });

  return {
    isExpanded,
    toggle,
    expand,
    collapse,
    setExpanded,
    animation,
    heightInterpolate,
    opacityInterpolate,
    rotateInterpolate,
    setContentHeight: (height: number) => {
      contentHeight.current = height;
    },
  };
}

