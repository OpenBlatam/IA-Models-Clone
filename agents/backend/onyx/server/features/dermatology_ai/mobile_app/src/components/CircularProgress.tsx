import React, { useEffect } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useTheme } from '../context/ThemeContext';
import { LinearGradient } from 'expo-linear-gradient';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withTiming,
  interpolate,
} from 'react-native-reanimated';

interface CircularProgressProps {
  progress: number; // 0-100
  size?: number;
  strokeWidth?: number;
  color?: string;
  backgroundColor?: string;
  showLabel?: boolean;
  label?: string;
  gradient?: boolean;
  gradientColors?: string[];
}

const CircularProgress: React.FC<CircularProgressProps> = ({
  progress,
  size = 100,
  strokeWidth = 8,
  color,
  backgroundColor,
  showLabel = true,
  label,
  gradient = false,
  gradientColors,
}) => {
  const { colors } = useTheme();
  const progressValue = useSharedValue(0);
  const ringColor = color || colors.primary;
  const ringBgColor = backgroundColor || colors.border;
  const gradientColorsValue = gradientColors || [colors.primary, colors.secondary];

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

  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (progress / 100) * circumference;

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
          {gradient ? (
            <LinearGradient
              colors={gradientColorsValue}
              style={StyleSheet.absoluteFill}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 1 }}
            />
          ) : (
            <View
              style={[
                StyleSheet.absoluteFill,
                {
                  backgroundColor: ringColor,
                  borderRadius: (size - strokeWidth * 2) / 2,
                },
              ]}
            />
          )}
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

export default CircularProgress;

