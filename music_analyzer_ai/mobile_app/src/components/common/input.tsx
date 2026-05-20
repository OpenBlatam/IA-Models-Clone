import React from 'react';
import {
  View,
  TextInput,
  Text,
  StyleSheet,
  TextInputProps,
} from 'react-native';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from '../../constants/config';

interface InputProps extends TextInputProps {
  label?: string;
  error?: string;
  helperText?: string;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  containerStyle?: object;
}

/**
 * Input component
 * Text input with label, error, and icons
 */
export function Input({
  label,
  error,
  helperText,
  leftIcon,
  rightIcon,
  containerStyle,
  style,
  ...props
}: InputProps) {
  return (
    <View style={[styles.container, containerStyle]}>
      {label && <Text style={styles.label}>{label}</Text>}
      <View
        style={[
          styles.inputContainer,
          error && styles.inputContainerError,
        ]}
      >
        {leftIcon && <View style={styles.leftIcon}>{leftIcon}</View>}
        <TextInput
          style={[styles.input, leftIcon && styles.inputWithLeftIcon, rightIcon && styles.inputWithRightIcon, style]}
          placeholderTextColor={COLORS.textSecondary}
          {...props}
        />
        {rightIcon && <View style={styles.rightIcon}>{rightIcon}</View>}
      </View>
      {error && <Text style={styles.error}>{error}</Text>}
      {helperText && !error && (
        <Text style={styles.helperText}>{helperText}</Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    width: '100%',
    marginBottom: SPACING.md,
  },
  label: {
    ...TYPOGRAPHY.bodySmall,
    color: COLORS.text,
    marginBottom: SPACING.xs,
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.surfaceLight,
    borderRadius: BORDER_RADIUS.md,
    borderWidth: 1,
    borderColor: COLORS.surfaceLight,
    paddingHorizontal: SPACING.md,
  },
  inputContainerError: {
    borderColor: COLORS.error,
  },
  input: {
    flex: 1,
    ...TYPOGRAPHY.body,
    color: COLORS.text,
    paddingVertical: SPACING.md,
  },
  inputWithLeftIcon: {
    marginLeft: SPACING.sm,
  },
  inputWithRightIcon: {
    marginRight: SPACING.sm,
  },
  leftIcon: {
    marginRight: SPACING.xs,
  },
  rightIcon: {
    marginLeft: SPACING.xs,
  },
  error: {
    ...TYPOGRAPHY.bodySmall,
    color: COLORS.error,
    marginTop: SPACING.xs,
  },
  helperText: {
    ...TYPOGRAPHY.bodySmall,
    color: COLORS.textSecondary,
    marginTop: SPACING.xs,
  },
});

