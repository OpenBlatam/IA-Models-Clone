import React, { useEffect } from 'react';
import { View, Text, StyleSheet } from 'react-native';
// Note: Requires react-native-svg for full functionality
// import Svg, { Circle } from 'react-native-svg';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withTiming,
  interpolate,
} from 'react-native-reanimated';
import { useTheme } from '../context/ThemeContext';
import { LinearGradient } from 'expo-linear-gradient';

interface ProgressRingProps {
  progress: number; // 0-100
  size?: number;
  strokeWidth?: number;
  color?: string;
  backgroundColor?: string;
  showLabel?: boolean;
  label?: string;
}

const ProgressRing: React.FC<ProgressRingProps> = ({
  progress,
  size = 100,
  strokeWidth = 8,
  color,
  backgroundColor,
  showLabel = true,
  label,
}) => {
  const { colors } = useTheme();
  const progressValue = useSharedValue(0);
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const ringColor = color || colors.primary;
  const ringBgColor = backgroundColor || colors.border;

  useEffect(() => {
    progressValue.value = withTiming(progress, {
      duration: 1000,
    });
  }, [progress, progressValue]);

  const animatedStyle = useAnimatedStyle(() => {
    const rotation = interpolate(progressValue.value, [0, 100], [0, 360]);
    return {
      transform: [{ rotate: `${rotation}deg` }],
    };
  });

  return (
    <View style={[styles.container, { width: size, height: size }]}>
      <View
        style={[
          styles.ring,
          {
            width: size,
            height: size,
            borderRadius: size / 2,
            borderWidth: strokeWidth,
            borderColor: ringBgColor,
          },
        ]}
      >
        <Animated.View
          style={[
            styles.progressRing,
            {
              width: size - strokeWidth * 2,
              height: size - strokeWidth * 2,
              borderRadius: (size - strokeWidth * 2) / 2,
            },
            animatedStyle,
          ]}
        >
          <LinearGradient
            colors={[ringColor, ringColor + '80']}
            style={StyleSheet.absoluteFill}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
          />
        </Animated.View>
      </View>
      {showLabel && (
        <View style={styles.labelContainer}>
          <Text style={[styles.label, { color: colors.text }]}>
            {label || `${Math.round(progress)}%`}
          </Text>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  ring: {
    justifyContent: 'center',
    alignItems: 'center',
    overflow: 'hidden',
  },
  progressRing: {
    position: 'absolute',
  },
  labelContainer: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
  },
  label: {
    fontSize: 16,
    fontWeight: '600',
  },
});

export default ProgressRing;

