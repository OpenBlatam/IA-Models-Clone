import { useState, useEffect } from 'react';
import { Animated } from 'react-native';

export const useScrollPosition = () => {
  const [scrollY] = useState(new Animated.Value(0));
  const [scrollPosition, setScrollPosition] = useState(0);
  const [isScrolling, setIsScrolling] = useState(false);

  useEffect(() => {
    const listener = scrollY.addListener(({ value }) => {
      setScrollPosition(value);
      setIsScrolling(value > 0);
    });

    return () => {
      scrollY.removeListener(listener);
    };
  }, [scrollY]);

  const scrollTo = (y: number, animated: boolean = true) => {
    if (animated) {
      Animated.timing(scrollY, {
        toValue: y,
        duration: 300,
        useNativeDriver: false,
      }).start();
    } else {
      scrollY.setValue(y);
    }
  };

  return {
    scrollY,
    scrollPosition,
    isScrolling,
    scrollTo,
  };
};

