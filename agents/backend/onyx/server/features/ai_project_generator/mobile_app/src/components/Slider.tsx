import React, { useState } from 'react';
import { View, Text, StyleSheet, PanResponder, Animated } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, borderRadius, typography } from '../theme/colors';
import { hapticFeedback } from '../utils/haptics';

interface SliderProps {
  value: number;
  onValueChange: (value: number) => void;
  minimumValue?: number;
  maximumValue?: number;
  step?: number;
  label?: string;
  showValue?: boolean;
  disabled?: boolean;
}

export const Slider: React.FC<SliderProps> = ({
  value,
  onValueChange,
  minimumValue = 0,
  maximumValue = 100,
  step = 1,
  label,
  showValue = true,
  disabled = false,
}) => {
  const { theme } = useTheme();
  const [sliderWidth, setSliderWidth] = useState(0);
  const pan = React.useRef(new Animated.ValueXY()).current;

  const range = maximumValue - minimumValue;
  const percentage = ((value - minimumValue) / range) * 100;

  React.useEffect(() => {
    if (sliderWidth > 0) {
      pan.setValue({
        x: (percentage / 100) * (sliderWidth - 24),
        y: 0,
      });
    }
  }, [percentage, sliderWidth]);

  const panResponder = React.useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => !disabled,
      onMoveShouldSetPanResponder: () => !disabled,
      onPanResponderGrant: () => {
        hapticFeedback.selection();
      },
      onPanResponderMove: (_, gestureState) => {
        if (sliderWidth > 0) {
          const newX = Math.max(
            0,
            Math.min(gestureState.moveX - 12, sliderWidth - 24)
          );
          pan.setValue({ x: newX, y: 0 });

          const newPercentage = (newX / (sliderWidth - 24)) * 100;
          const newValue =
            Math.round((newPercentage / 100) * range / step) * step +
            minimumValue;
          const clampedValue = Math.max(
            minimumValue,
            Math.min(maximumValue, newValue)
          );
          onValueChange(clampedValue);
        }
      },
      onPanResponderRelease: () => {
        hapticFeedback.success();
      },
    })
  ).current;

  return (
    <View style={styles.container}>
      {label && (
        <View style={styles.labelRow}>
          <Text style={[styles.label, { color: theme.text }]}>{label}</Text>
          {showValue && (
            <Text style={[styles.value, { color: theme.textSecondary }]}>
              {value}
            </Text>
          )}
        </View>
      )}
      <View
        style={styles.sliderContainer}
        onLayout={(e) => setSliderWidth(e.nativeEvent.layout.width)}
      >
        <View
          style={[
            styles.track,
            {
              backgroundColor: theme.border,
            },
          ]}
        />
        <View
          style={[
            styles.trackActive,
            {
              backgroundColor: theme.primary,
              width: `${percentage}%`,
            },
          ]}
        />
        <Animated.View
          style={[
            styles.thumb,
            {
              backgroundColor: theme.primary,
              transform: [{ translateX: pan.x }],
            },
          ]}
          {...panResponder.panHandlers}
        />
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginBottom: spacing.md,
  },
  labelRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.sm,
  },
  label: {
    ...typography.bodySmall,
    fontWeight: '600',
  },
  value: {
    ...typography.bodySmall,
  },
  sliderContainer: {
    height: 40,
    justifyContent: 'center',
    position: 'relative',
  },
  track: {
    height: 4,
    borderRadius: 2,
    position: 'absolute',
    width: '100%',
  },
  trackActive: {
    height: 4,
    borderRadius: 2,
    position: 'absolute',
  },
  thumb: {
    width: 24,
    height: 24,
    borderRadius: 12,
    position: 'absolute',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 2,
    elevation: 3,
  },
});

