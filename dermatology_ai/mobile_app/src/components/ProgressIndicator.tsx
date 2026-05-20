import React, { useEffect } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withTiming,
} from 'react-native-reanimated';
import { useTheme } from '../context/ThemeContext';

interface ProgressIndicatorProps {
  progress: number; // 0-100
  label?: string;
  showPercentage?: boolean;
  height?: number;
  animated?: boolean;
}

const ProgressIndicator: React.FC<ProgressIndicatorProps> = ({
  progress,
  label,
  showPercentage = true,
  height = 8,
  animated = true,
}) => {
  const { colors } = useTheme();
  const progressValue = useSharedValue(0);

  useEffect(() => {
    if (animated) {
      progressValue.value = withTiming(progress, {
        duration: 500,
      });
    } else {
      progressValue.value = progress;
    }
  }, [progress, animated, progressValue]);

  const animatedStyle = useAnimatedStyle(() => {
    return {
      width: `${progressValue.value}%`,
    };
  });

  return (
    <View style={styles.container}>
      {label && (
        <View style={styles.header}>
          <Text style={[styles.label, { color: colors.text }]}>{label}</Text>
          {showPercentage && (
            <Text style={[styles.percentage, { color: colors.textSecondary }]}>
              {Math.round(progress)}%
            </Text>
          )}
        </View>
      )}
      <View
        style={[
          styles.track,
          { height, backgroundColor: colors.border },
        ]}
      >
        <Animated.View style={animatedStyle}>
          <LinearGradient
            colors={[colors.primary, colors.secondary]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
            style={[styles.fill, { height }]}
          />
        </Animated.View>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginVertical: 8,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  label: {
    fontSize: 14,
    fontWeight: '500',
  },
  percentage: {
    fontSize: 14,
    fontWeight: '600',
  },
  track: {
    borderRadius: 4,
    overflow: 'hidden',
  },
  fill: {
    borderRadius: 4,
  },
});

export default ProgressIndicator;

