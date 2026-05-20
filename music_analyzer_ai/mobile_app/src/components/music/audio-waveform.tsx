import React from 'react';
import { View, StyleSheet } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withRepeat,
  withTiming,
  interpolate,
} from 'react-native-reanimated';
import { COLORS, SPACING } from '../../constants/config';

interface AudioWaveformProps {
  bars?: number;
  height?: number;
  color?: string;
  animated?: boolean;
}

export function AudioWaveform({
  bars = 20,
  height = 40,
  color = COLORS.primary,
  animated = true,
}: AudioWaveformProps) {
  const animation = useSharedValue(0);

  React.useEffect(() => {
    if (animated) {
      animation.value = withRepeat(
        withTiming(1, { duration: 1000 }),
        -1,
        true
      );
    }
  }, [animated, animation]);

  return (
    <View style={[styles.container, { height }]}>
      {Array.from({ length: bars }).map((_, index) => {
        const barHeight = Math.random() * height;
        const delay = index * 50;

        const animatedStyle = useAnimatedStyle(() => {
          if (!animated) {
            return { height: barHeight };
          }

          const progress = (animation.value + delay / 1000) % 1;
          const scale = interpolate(
            progress,
            [0, 0.5, 1],
            [0.3, 1, 0.3]
          );

          return {
            height: barHeight * scale,
          };
        });

        return (
          <Animated.View
            key={index}
            style={[
              styles.bar,
              { width: 3, backgroundColor: color },
              animatedStyle,
            ]}
          />
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: SPACING.xs,
  },
  bar: {
    borderRadius: 2,
  },
});

