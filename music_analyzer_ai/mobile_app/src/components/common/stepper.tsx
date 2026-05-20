import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from '../../constants/config';

interface StepperProps {
  value: number;
  onValueChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
  label?: string;
}

/**
 * Stepper component for numeric input
 * Increment/decrement controls
 */
export function Stepper({
  value,
  onValueChange,
  min = 0,
  max = 100,
  step = 1,
  disabled = false,
  label,
}: StepperProps) {
  const handleDecrement = () => {
    if (!disabled && value > min) {
      onValueChange(Math.max(min, value - step));
    }
  };

  const handleIncrement = () => {
    if (!disabled && value < max) {
      onValueChange(Math.min(max, value + step));
    }
  };

  const canDecrement = !disabled && value > min;
  const canIncrement = !disabled && value < max;

  return (
    <View style={styles.container}>
      {label && <Text style={styles.label}>{label}</Text>}
      <View style={styles.stepper}>
        <TouchableOpacity
          style={[
            styles.button,
            styles.decrementButton,
            !canDecrement && styles.disabledButton,
          ]}
          onPress={handleDecrement}
          disabled={!canDecrement}
          accessibilityLabel="Decrement"
          accessibilityRole="button"
        >
          <Text
            style={[
              styles.buttonText,
              !canDecrement && styles.disabledButtonText,
            ]}
          >
            −
          </Text>
        </TouchableOpacity>

        <View style={styles.valueContainer}>
          <Text style={styles.value}>{value}</Text>
        </View>

        <TouchableOpacity
          style={[
            styles.button,
            styles.incrementButton,
            !canIncrement && styles.disabledButton,
          ]}
          onPress={handleIncrement}
          disabled={!canIncrement}
          accessibilityLabel="Increment"
          accessibilityRole="button"
        >
          <Text
            style={[
              styles.buttonText,
              !canIncrement && styles.disabledButtonText,
            ]}
          >
            +
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
  },
  label: {
    ...TYPOGRAPHY.body,
    color: COLORS.text,
    marginBottom: SPACING.sm,
  },
  stepper: {
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: COLORS.surfaceLight,
    borderRadius: BORDER_RADIUS.md,
    overflow: 'hidden',
  },
  button: {
    width: 44,
    height: 44,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: COLORS.surface,
  },
  decrementButton: {
    borderRightWidth: 1,
    borderRightColor: COLORS.surfaceLight,
  },
  incrementButton: {
    borderLeftWidth: 1,
    borderLeftColor: COLORS.surfaceLight,
  },
  disabledButton: {
    opacity: 0.5,
  },
  buttonText: {
    ...TYPOGRAPHY.h2,
    color: COLORS.text,
  },
  disabledButtonText: {
    color: COLORS.textSecondary,
  },
  valueContainer: {
    minWidth: 60,
    paddingHorizontal: SPACING.md,
    alignItems: 'center',
    justifyContent: 'center',
  },
  value: {
    ...TYPOGRAPHY.h3,
    color: COLORS.text,
  },
});

