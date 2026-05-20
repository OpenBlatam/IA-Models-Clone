import React from 'react';
import { View, TouchableOpacity, Text, StyleSheet } from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
} from 'react-native-reanimated';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from '../../constants/config';

interface SegmentedControlProps<T extends string> {
  options: T[];
  selectedValue: T;
  onValueChange: (value: T) => void;
  disabled?: boolean;
}

/**
 * Segmented control component
 * iOS-style segmented control
 */
export function SegmentedControl<T extends string>({
  options,
  selectedValue,
  onValueChange,
  disabled = false,
}: SegmentedControlProps<T>) {
  const selectedIndex = options.indexOf(selectedValue);
  const indicatorPosition = useSharedValue(selectedIndex);

  React.useEffect(() => {
    indicatorPosition.value = withSpring(selectedIndex);
  }, [selectedIndex, indicatorPosition]);

  const animatedStyle = useAnimatedStyle(() => {
    const segmentWidth = 100 / options.length;
    return {
      transform: [{ translateX: `${indicatorPosition.value * segmentWidth}%` }],
      width: `${segmentWidth}%`,
    };
  });

  return (
    <View style={[styles.container, disabled && styles.disabled]}>
      <Animated.View style={[styles.indicator, animatedStyle]} />
      {options.map((option, index) => (
        <TouchableOpacity
          key={option}
          style={styles.segment}
          onPress={() => !disabled && onValueChange(option)}
          disabled={disabled}
          accessibilityRole="button"
          accessibilityState={{ selected: option === selectedValue }}
        >
          <Text
            style={[
              styles.label,
              option === selectedValue && styles.selectedLabel,
            ]}
          >
            {option}
          </Text>
        </TouchableOpacity>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    backgroundColor: COLORS.surfaceLight,
    borderRadius: BORDER_RADIUS.md,
    padding: 2,
    position: 'relative',
  },
  indicator: {
    position: 'absolute',
    top: 2,
    bottom: 2,
    backgroundColor: COLORS.surface,
    borderRadius: BORDER_RADIUS.sm,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  segment: {
    flex: 1,
    paddingVertical: SPACING.sm,
    paddingHorizontal: SPACING.md,
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1,
  },
  label: {
    ...TYPOGRAPHY.bodySmall,
    color: COLORS.textSecondary,
    fontWeight: '500',
  },
  selectedLabel: {
    color: COLORS.text,
    fontWeight: '600',
  },
  disabled: {
    opacity: 0.5,
  },
});

