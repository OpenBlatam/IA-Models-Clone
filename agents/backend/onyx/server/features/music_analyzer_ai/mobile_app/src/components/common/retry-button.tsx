import React, { memo, useCallback } from 'react';
import { TouchableOpacity, Text, StyleSheet, ActivityIndicator } from 'react-native';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from '../../constants/config';
import { useHapticFeedback } from '../../hooks/use-haptic-feedback';

interface RetryButtonProps {
  onPress: () => void;
  label?: string;
  loading?: boolean;
  variant?: 'primary' | 'secondary' | 'outline';
  size?: 'small' | 'medium' | 'large';
}

function RetryButtonComponent({
  onPress,
  label = 'Retry',
  loading = false,
  variant = 'primary',
  size = 'medium',
}: RetryButtonProps) {
  const haptics = useHapticFeedback();

  const handlePress = useCallback(() => {
    haptics.light();
    onPress();
  }, [onPress, haptics]);

  const buttonStyle = [
    styles.button,
    styles[`button_${variant}`],
    styles[`button_${size}`],
    loading && styles.buttonDisabled,
  ];

  const textStyle = [
    styles.text,
    styles[`text_${variant}`],
    styles[`text_${size}`],
  ];

  return (
    <TouchableOpacity
      style={buttonStyle}
      onPress={handlePress}
      disabled={loading}
      accessibilityRole="button"
      accessibilityLabel={label}
      accessibilityState={{ disabled: loading }}
    >
      {loading ? (
        <ActivityIndicator
          size="small"
          color={variant === 'outline' ? COLORS.primary : COLORS.text}
        />
      ) : (
        <Text style={textStyle}>{label}</Text>
      )}
    </TouchableOpacity>
  );
}

export const RetryButton = memo(RetryButtonComponent);

const styles = StyleSheet.create({
  button: {
    borderRadius: BORDER_RADIUS.md,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
  },
  button_primary: {
    backgroundColor: COLORS.primary,
  },
  button_secondary: {
    backgroundColor: COLORS.secondary,
  },
  button_outline: {
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: COLORS.primary,
  },
  button_small: {
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    minWidth: 80,
  },
  button_medium: {
    paddingHorizontal: SPACING.lg,
    paddingVertical: SPACING.md,
    minWidth: 120,
  },
  button_large: {
    paddingHorizontal: SPACING.xl,
    paddingVertical: SPACING.lg,
    minWidth: 160,
  },
  buttonDisabled: {
    opacity: 0.6,
  },
  text: {
    ...TYPOGRAPHY.body,
    fontWeight: '600',
    textAlign: 'center',
  },
  text_primary: {
    color: COLORS.text,
  },
  text_secondary: {
    color: COLORS.text,
  },
  text_outline: {
    color: COLORS.primary,
  },
  text_small: {
    fontSize: 12,
  },
  text_medium: {
    fontSize: 14,
  },
  text_large: {
    fontSize: 16,
  },
});

