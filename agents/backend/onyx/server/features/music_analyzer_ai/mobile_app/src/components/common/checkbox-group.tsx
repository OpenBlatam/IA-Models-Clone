import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from '../../constants/config';

interface CheckboxOption<T> {
  label: string;
  value: T;
  disabled?: boolean;
}

interface CheckboxGroupProps<T> {
  options: CheckboxOption<T>[];
  value: T[];
  onValueChange: (value: T[]) => void;
  direction?: 'row' | 'column';
  label?: string;
}

/**
 * Checkbox group component
 * Multiple selection from options
 */
export function CheckboxGroup<T>({
  options,
  value,
  onValueChange,
  direction = 'column',
  label,
}: CheckboxGroupProps<T>) {
  const handleToggle = (optionValue: T) => {
    const isSelected = value.includes(optionValue);
    if (isSelected) {
      onValueChange(value.filter((v) => v !== optionValue));
    } else {
      onValueChange([...value, optionValue]);
    }
  };

  return (
    <View style={styles.container}>
      {label && <Text style={styles.label}>{label}</Text>}
      <View style={[styles.options, direction === 'row' && styles.row]}>
        {options.map((option, index) => {
          const isSelected = value.includes(option.value);
          const isDisabled = option.disabled;

          return (
            <TouchableOpacity
              key={index}
              style={[
                styles.option,
                isSelected && styles.selectedOption,
                isDisabled && styles.disabledOption,
              ]}
              onPress={() => !isDisabled && handleToggle(option.value)}
              disabled={isDisabled}
              accessibilityRole="checkbox"
              accessibilityState={{ checked: isSelected, disabled: isDisabled }}
            >
              <View
                style={[
                  styles.checkbox,
                  isSelected && styles.checkboxSelected,
                ]}
              >
                {isSelected && <Text style={styles.checkmark}>✓</Text>}
              </View>
              <Text
                style={[
                  styles.optionLabel,
                  isSelected && styles.selectedOptionLabel,
                  isDisabled && styles.disabledOptionLabel,
                ]}
              >
                {option.label}
              </Text>
            </TouchableOpacity>
          );
        })}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    width: '100%',
  },
  label: {
    ...TYPOGRAPHY.bodySmall,
    color: COLORS.text,
    marginBottom: SPACING.sm,
  },
  options: {
    gap: SPACING.sm,
  },
  row: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  option: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: SPACING.sm,
    borderRadius: BORDER_RADIUS.md,
  },
  selectedOption: {
    backgroundColor: COLORS.primary + '10',
  },
  disabledOption: {
    opacity: 0.5,
  },
  checkbox: {
    width: 20,
    height: 20,
    borderRadius: BORDER_RADIUS.xs,
    borderWidth: 2,
    borderColor: COLORS.surfaceLight,
    marginRight: SPACING.sm,
    justifyContent: 'center',
    alignItems: 'center',
  },
  checkboxSelected: {
    borderColor: COLORS.primary,
    backgroundColor: COLORS.primary,
  },
  checkmark: {
    color: COLORS.surface,
    fontSize: 12,
    fontWeight: 'bold',
  },
  optionLabel: {
    ...TYPOGRAPHY.body,
    color: COLORS.text,
    flex: 1,
  },
  selectedOptionLabel: {
    color: COLORS.primary,
    fontWeight: '600',
  },
  disabledOptionLabel: {
    color: COLORS.textSecondary,
  },
});

