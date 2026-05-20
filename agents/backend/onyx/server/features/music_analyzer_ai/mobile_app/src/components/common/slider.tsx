import React, { useRef } from 'react';
import { View, Text, StyleSheet, PanResponder } from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
} from 'react-native-reanimated';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from '../../constants/config';

interface SliderProps {
  value: number;
  onValueChange: (value: number) => void;
  minimumValue?: number;
  maximumValue?: number;
  step?: number;
  disabled?: boolean;
  minimumTrackTintColor?: string;
  maximumTrackTintColor?: string;
  thumbTintColor?: string;
  showValue?: boolean;
  label?: string;
}

/**
 * Slider component
 * Range input with animation
 */
export function Slider({
  value,
  onValueChange,
  minimumValue = 0,
  maximumValue = 100,
  step = 1,
  disabled = false,
  minimumTrackTintColor = COLORS.primary,
  maximumTrackTintColor = COLORS.surfaceLight,
  thumbTintColor = COLORS.primary,
  showValue = false,
  label,
}: SliderProps) {
  const trackWidth = 280;
  const thumbSize = 20;
  const trackRef = useRef<View>(null);

  const translateX = useSharedValue(
    ((value - minimumValue) / (maximumValue - minimumValue)) * trackWidth
  );

  React.useEffect(() => {
    translateX.value = withSpring(
      ((value - minimumValue) / (maximumValue - minimumValue)) * trackWidth,
      { damping: 15, stiffness: 150 }
    );
  }, [value, minimumValue, maximumValue, trackWidth, translateX]);

  const panResponder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => !disabled,
      onMoveShouldSetPanResponder: () => !disabled,
      onPanResponderGrant: () => {
        // Handle touch start
      },
      onPanResponderMove: (_, gestureState) => {
        if (!disabled) {
          const newX = Math.max(
            0,
            Math.min(trackWidth, gestureState.moveX - (trackRef.current?.getLayout?.()?.x || 0))
          );
          const percentage = newX / trackWidth;
          const newValue =
            Math.round(
              (minimumValue +
                percentage * (maximumValue - minimumValue)) /
                step
            ) * step;
          translateX.value = newX;
          onValueChange(Math.max(minimumValue, Math.min(maximumValue, newValue)));
        }
      },
      onPanResponderRelease: () => {
        // Handle touch end
      },
    })
  ).current;

  const thumbStyle = useAnimatedStyle(() => ({
    transform: [{ translateX: translateX.value }],
  }));

  const trackFillStyle = useAnimatedStyle(() => ({
    width: translateX.value,
  }));

  return (
    <View style={styles.container}>
      {label && <Text style={styles.label}>{label}</Text>}
      <View style={styles.sliderContainer}>
        <View
          ref={trackRef}
          style={[styles.track, { backgroundColor: maximumTrackTintColor }]}
        >
          <Animated.View
            style={[
              styles.trackFill,
              { backgroundColor: minimumTrackTintColor },
              trackFillStyle,
            ]}
          />
        </View>
        <Animated.View
          style={[styles.thumb, { backgroundColor: thumbTintColor }, thumbStyle]}
          {...panResponder.panHandlers}
        />
      </View>
      {showValue && (
        <Text style={styles.value}>{Math.round(value)}</Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    width: '100%',
    paddingVertical: SPACING.sm,
  },
  label: {
    ...TYPOGRAPHY.bodySmall,
    color: COLORS.text,
    marginBottom: SPACING.xs,
  },
  sliderContainer: {
    height: 40,
    justifyContent: 'center',
    position: 'relative',
  },
  track: {
    height: 4,
    width: 280,
    borderRadius: BORDER_RADIUS.full,
    position: 'relative',
  },
  trackFill: {
    height: 4,
    borderRadius: BORDER_RADIUS.full,
    position: 'absolute',
    left: 0,
    top: 0,
  },
  thumb: {
    width: 20,
    height: 20,
    borderRadius: 10,
    position: 'absolute',
    top: 10,
    left: -10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 2,
    elevation: 2,
  },
  value: {
    ...TYPOGRAPHY.bodySmall,
    color: COLORS.textSecondary,
    marginTop: SPACING.xs,
    textAlign: 'center',
  },
});

