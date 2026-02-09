import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from '../../constants/config';

interface ChipProps {
  label: string;
  selected?: boolean;
  onPress?: () => void;
  variant?: 'default' | 'outline';
  size?: 'small' | 'medium' | 'large';
  disabled?: boolean;
}

/**
 * Chip component for tags and filters
 */
export function Chip({
  label,
  selected = false,
  onPress,
  variant = 'default',
  size = 'medium',
  disabled = false,
}: ChipProps) {
  const sizeStyles = {
    small: {
      paddingHorizontal: SPACING.sm,
      paddingVertical: SPACING.xs,
      fontSize: 12,
    },
    medium: {
      paddingHorizontal: SPACING.md,
      paddingVertical: SPACING.sm,
      fontSize: 14,
    },
    large: {
      paddingHorizontal: SPACING.lg,
      paddingVertical: SPACING.md,
      fontSize: 16,
    },
  };

  const Container = onPress ? TouchableOpacity : View;

  return (
    <Container
      style={[
        styles.container,
        sizeStyles[size],
        variant === 'outline'
          ? {
              backgroundColor: 'transparent',
              borderWidth: 1,
              borderColor: selected ? COLORS.primary : COLORS.surfaceLight,
            }
          : {
              backgroundColor: selected ? COLORS.primary : COLORS.surfaceLight,
            },
        disabled && styles.disabled,
      ]}
      onPress={onPress}
      disabled={disabled || !onPress}
      activeOpacity={0.7}
    >
      <Text
        style={[
          styles.label,
          {
            fontSize: sizeStyles[size].fontSize,
            color: selected
              ? COLORS.text
              : variant === 'outline'
              ? COLORS.text
              : COLORS.textSecondary,
          },
        ]}
      >
        {label}
      </Text>
    </Container>
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
    ...TYPOGRAPHY.bodySmall,
    fontWeight: '600',
  },
  disabled: {
    opacity: 0.5,
  },
});

