import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from '../../constants/config';

interface RadioOption<T> {
  label: string;
  value: T;
  disabled?: boolean;
}

interface RadioGroupProps<T> {
  options: RadioOption<T>[];
  value: T;
  onValueChange: (value: T) => void;
  direction?: 'row' | 'column';
  label?: string;
}

/**
 * Radio group component
 * Single selection from options
 */
export function RadioGroup<T>({
  options,
  value,
  onValueChange,
  direction = 'column',
  label,
}: RadioGroupProps<T>) {
  return (
    <View style={styles.container}>
      {label && <Text style={styles.label}>{label}</Text>}
      <View style={[styles.options, direction === 'row' && styles.row]}>
        {options.map((option, index) => {
          const isSelected = option.value === value;
          const isDisabled = option.disabled;

          return (
            <TouchableOpacity
              key={index}
              style={[
                styles.option,
                isSelected && styles.selectedOption,
                isDisabled && styles.disabledOption,
              ]}
              onPress={() => !isDisabled && onValueChange(option.value)}
              disabled={isDisabled}
              accessibilityRole="radio"
              accessibilityState={{ selected: isSelected, disabled: isDisabled }}
            >
              <View
                style={[
                  styles.radio,
                  isSelected && styles.radioSelected,
                ]}
              >
                {isSelected && <View style={styles.radioInner} />}
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
  radio: {
    width: 20,
    height: 20,
    borderRadius: 10,
    borderWidth: 2,
    borderColor: COLORS.surfaceLight,
    marginRight: SPACING.sm,
    justifyContent: 'center',
    alignItems: 'center',
  },
  radioSelected: {
    borderColor: COLORS.primary,
  },
  radioInner: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: COLORS.primary,
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

