import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from '../../constants/config';

interface BadgeProps {
  label: string | number;
  variant?: 'primary' | 'secondary' | 'success' | 'error' | 'warning' | 'info';
  size?: 'small' | 'medium' | 'large';
}

/**
 * Badge component for labels and counts
 */
export function Badge({
  label,
  variant = 'primary',
  size = 'medium',
}: BadgeProps) {
  const variantColors = {
    primary: COLORS.primary,
    secondary: COLORS.secondary,
    success: COLORS.success,
    error: COLORS.error,
    warning: COLORS.warning,
    info: COLORS.info,
  };

  const sizeStyles = {
    small: {
      paddingHorizontal: SPACING.xs,
      paddingVertical: 2,
      fontSize: 10,
    },
    medium: {
      paddingHorizontal: SPACING.sm,
      paddingVertical: SPACING.xs,
      fontSize: 12,
    },
    large: {
      paddingHorizontal: SPACING.md,
      paddingVertical: SPACING.sm,
      fontSize: 14,
    },
  };

  return (
    <View
      style={[
        styles.container,
        {
          backgroundColor: variantColors[variant],
          ...sizeStyles[size],
        },
      ]}
    >
      <Text
        style={[
          styles.label,
          {
            fontSize: sizeStyles[size].fontSize,
          },
        ]}
      >
        {label}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    borderRadius: BORDER_RADIUS.full,
    alignSelf: 'flex-start',
    alignItems: 'center',
    justifyContent: 'center',
  },
  label: {
    ...TYPOGRAPHY.caption,
    color: COLORS.text,
    fontWeight: '600',
  },
});

