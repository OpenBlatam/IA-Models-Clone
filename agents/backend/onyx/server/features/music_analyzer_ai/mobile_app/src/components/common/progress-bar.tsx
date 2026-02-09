import React from 'react';
import { View, StyleSheet, Animated } from 'react-native';
import { COLORS, BORDER_RADIUS } from '../../constants/config';

interface ProgressBarProps {
  progress: number; // 0 to 1
  color?: string;
  backgroundColor?: string;
  height?: number;
  animated?: boolean;
  duration?: number;
}

/**
 * Progress bar component with animation support
 */
export function ProgressBar({
  progress,
  color = COLORS.primary,
  backgroundColor = COLORS.surfaceLight,
  height = 4,
  animated = true,
  duration = 300,
}: ProgressBarProps) {
  const animatedValue = React.useRef(new Animated.Value(0)).current;

  React.useEffect(() => {
    if (animated) {
      Animated.timing(animatedValue, {
        toValue: progress,
        duration,
        useNativeDriver: false,
      }).start();
    } else {
      animatedValue.setValue(progress);
    }
  }, [progress, animated, duration, animatedValue]);

  const width = animatedValue.interpolate({
    inputRange: [0, 1],
    outputRange: ['0%', '100%'],
  });

  return (
    <View
      style={[
        styles.container,
        { height, backgroundColor },
      ]}
    >
      <Animated.View
        style={[
          styles.progress,
          {
            width,
            backgroundColor: color,
            height,
          },
        ]}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    width: '100%',
    borderRadius: BORDER_RADIUS.full,
    overflow: 'hidden',
  },
  progress: {
    borderRadius: BORDER_RADIUS.full,
  },
});

