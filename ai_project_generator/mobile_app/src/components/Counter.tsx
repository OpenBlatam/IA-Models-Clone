import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, borderRadius, typography } from '../theme/colors';
import { hapticFeedback } from '../utils/haptics';

interface CounterProps {
  value: number;
  onValueChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  label?: string;
  disabled?: boolean;
}

export const Counter: React.FC<CounterProps> = ({
  value,
  onValueChange,
  min = 0,
  max = 100,
  step = 1,
  label,
  disabled = false,
}) => {
  const { theme } = useTheme();

  const handleDecrement = () => {
    if (!disabled && value > min) {
      hapticFeedback.selection();
      onValueChange(Math.max(min, value - step));
    }
  };

  const handleIncrement = () => {
    if (!disabled && value < max) {
      hapticFeedback.selection();
      onValueChange(Math.min(max, value + step));
    }
  };

  return (
    <View style={styles.container}>
      {label && (
        <Text style={[styles.label, { color: theme.text }]}>{label}</Text>
      )}
      <View
        style={[
          styles.counter,
          {
            backgroundColor: theme.surface,
            borderColor: theme.border,
            opacity: disabled ? 0.6 : 1,
          },
        ]}
      >
        <TouchableOpacity
          style={[
            styles.button,
            {
              backgroundColor: theme.surfaceVariant,
              opacity: value <= min || disabled ? 0.5 : 1,
            },
          ]}
          onPress={handleDecrement}
          disabled={value <= min || disabled}
          activeOpacity={0.7}
        >
          <Text style={[styles.buttonText, { color: theme.text }]}>−</Text>
        </TouchableOpacity>
        <View style={styles.valueContainer}>
          <Text style={[styles.value, { color: theme.text }]}>{value}</Text>
        </View>
        <TouchableOpacity
          style={[
            styles.button,
            {
              backgroundColor: theme.surfaceVariant,
              opacity: value >= max || disabled ? 0.5 : 1,
            },
          ]}
          onPress={handleIncrement}
          disabled={value >= max || disabled}
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
    marginBottom: spacing.md,
  },
  label: {
    ...typography.bodySmall,
    fontWeight: '600',
    marginBottom: spacing.sm,
  },
  counter: {
    flexDirection: 'row',
    alignItems: 'center',
    borderRadius: borderRadius.md,
    borderWidth: 1,
    overflow: 'hidden',
  },
  button: {
    width: 48,
    height: 48,
    justifyContent: 'center',
    alignItems: 'center',
  },
  buttonText: {
    ...typography.h3,
    fontWeight: '600',
  },
  valueContainer: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: spacing.md,
  },
  value: {
    ...typography.h3,
    fontWeight: '600',
  },
});

