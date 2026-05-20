import React from 'react';
import {
  View,
  TextInput,
  Text,
  StyleSheet,
  TextInputProps,
} from 'react-native';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from '../../constants/config';

interface TextareaProps extends TextInputProps {
  label?: string;
  error?: string;
  helperText?: string;
  rows?: number;
  containerStyle?: object;
}

/**
 * Textarea component
 * Multi-line text input
 */
export function Textarea({
  label,
  error,
  helperText,
  rows = 4,
  containerStyle,
  style,
  ...props
}: TextareaProps) {
  const minHeight = rows * 20 + SPACING.md * 2;

  return (
    <View style={[styles.container, containerStyle]}>
      {label && <Text style={styles.label}>{label}</Text>}
      <TextInput
        style={[
          styles.textarea,
          { minHeight },
          error && styles.textareaError,
          style,
        ]}
        placeholderTextColor={COLORS.textSecondary}
        multiline
        textAlignVertical="top"
        {...props}
      />
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
  textarea: {
    ...TYPOGRAPHY.body,
    color: COLORS.text,
    backgroundColor: COLORS.surfaceLight,
    borderRadius: BORDER_RADIUS.md,
    borderWidth: 1,
    borderColor: COLORS.surfaceLight,
    padding: SPACING.md,
    minHeight: 100,
  },
  textareaError: {
    borderColor: COLORS.error,
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

