import React, { useState, useEffect } from 'react';
import { View, StyleSheet, Animated, ViewStyle } from 'react-native';
import { useTheme } from '../context/ThemeContext';

interface StickyHeaderProps {
  children: React.ReactNode;
  scrollY: Animated.Value;
  threshold?: number;
  style?: ViewStyle;
}

const StickyHeader: React.FC<StickyHeaderProps> = ({
  children,
  scrollY,
  threshold = 0,
  style,
}) => {
  const { colors } = useTheme();
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
    inputRange: [0, threshold],
    outputRange: [0, -threshold],
    extrapolate: 'clamp',
  });

  return (
    <Animated.View
      style={[
        styles.container,
        {
          backgroundColor: colors.card,
          transform: [{ translateY }],
          zIndex: isSticky ? 1000 : 0,
        },
        isSticky && styles.sticky,
        style,
      ]}
    >
      {children}
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'relative',
  },
  sticky: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 4,
  },
});

export default StickyHeader;

