import React, { useRef, useEffect, useState } from 'react';
import { View, StyleSheet, Animated, ViewStyle } from 'react-native';

interface StickyHeaderProps {
  children: React.ReactNode;
  scrollY: Animated.Value;
  threshold?: number;
  style?: ViewStyle;
}

export const StickyHeader: React.FC<StickyHeaderProps> = ({
  children,
  scrollY,
  threshold = 0,
  style,
}) => {
  const [isSticky, setIsSticky] = useState(false);

  useEffect(() => {
    const listener = scrollY.addListener(({ value }) => {
      setIsSticky(value > threshold);
    });

    return () => {
      scrollY.removeListener(listener);
    };
  }, [scrollY, threshold]);

  const translateY = scrollY.interpolate({
    inputRange: [threshold, threshold + 100],
    outputRange: [0, -100],
    extrapolate: 'clamp',
  });

  const opacity = scrollY.interpolate({
    inputRange: [threshold, threshold + 50],
    outputRange: [0, 1],
    extrapolate: 'clamp',
  });

  return (
    <Animated.View
      style={[
        styles.container,
        style,
        {
          transform: [{ translateY }],
          opacity: isSticky ? opacity : 1,
        },
      ]}
    >
      {children}
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    zIndex: 100,
  },
});

