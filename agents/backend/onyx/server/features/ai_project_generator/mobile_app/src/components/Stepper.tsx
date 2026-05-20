import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, borderRadius, typography } from '../theme/colors';
import { hapticFeedback } from '../utils/haptics';

interface StepperProps {
  value: number;
  min?: number;
  max?: number;
  step?: number;
  onChange: (value: number) => void;
  label?: string;
  disabled?: boolean;
}

export const Stepper: React.FC<StepperProps> = ({
  value,
  min = 0,
  max = 100,
  step = 1,
  onChange,
  label,
  disabled = false,
}) => {
  const { theme } = useTheme();

  const handleDecrement = () => {
    if (!disabled && value > min) {
      hapticFeedback.selection();
      onChange(Math.max(min, value - step));
    }
  };

  const handleIncrement = () => {
    if (!disabled && value < max) {
      hapticFeedback.selection();
      onChange(Math.min(max, value + step));
    }
  };

  return (
    <View style={styles.container}>
      {label && (
        <Text style={[styles.label, { color: theme.text }]}>{label}</Text>
      )}
      <View style={styles.stepper}>
        <TouchableOpacity
          style={[
            styles.button,
            {
              backgroundColor: theme.surfaceVariant,
              borderColor: theme.border,
            },
            (disabled || value <= min) && styles.buttonDisabled,
          ]}
          onPress={handleDecrement}
          disabled={disabled || value <= min}
          activeOpacity={0.7}
        >
          <Text style={[styles.buttonText, { color: theme.text }]}>−</Text>
        </TouchableOpacity>
        <View
          style={[
            styles.valueContainer,
            {
              backgroundColor: theme.surface,
              borderColor: theme.border,
            },
          ]}
        >
          <Text style={[styles.value, { color: theme.text }]}>{value}</Text>
        </View>
        <TouchableOpacity
          style={[
            styles.button,
            {
              backgroundColor: theme.surfaceVariant,
              borderColor: theme.border,
            },
            (disabled || value >= max) && styles.buttonDisabled,
          ]}
          onPress={handleIncrement}
          disabled={disabled || value >= max}
          activeOpacity={0.7}
        >
          <Text style={[styles.buttonText, { color: theme.text }]}>+</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    gap: spacing.sm,
  },
  label: {
    ...typography.bodySmall,
    fontWeight: '600',
  },
  stepper: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  button: {
    width: 40,
    height: 40,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  buttonDisabled: {
    opacity: 0.5,
  },
  buttonText: {
    ...typography.h3,
    fontSize: 20,
  },
  valueContainer: {
    minWidth: 60,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  value: {
    ...typography.body,
    fontWeight: '600',
  },
});

