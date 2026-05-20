import React from 'react';
import { View, Text, StyleSheet, ViewStyle, TextStyle } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, borderRadius, typography } from '../theme/colors';

interface BadgeProps {
  label: string;
  variant?: 'default' | 'primary' | 'success' | 'warning' | 'error' | 'info';
  size?: 'small' | 'medium' | 'large';
  style?: ViewStyle;
  textStyle?: TextStyle;
}

export const Badge: React.FC<BadgeProps> = ({
  label,
  variant = 'default',
  size = 'medium',
  style,
  textStyle,
}) => {
  const { theme } = useTheme();

  const getVariantColors = () => {
    switch (variant) {
      case 'primary':
        return { bg: theme.primary, text: theme.surface };
      case 'success':
        return { bg: theme.success, text: theme.surface };
      case 'warning':
        return { bg: theme.warning, text: theme.surface };
      case 'error':
        return { bg: theme.error, text: theme.surface };
      case 'info':
        return { bg: theme.info, text: theme.surface };
      default:
        return { bg: theme.surfaceVariant, text: theme.text };
    }
  };

  const getSizeStyles = () => {
    switch (size) {
      case 'small':
        return {
          padding: spacing.xs,
          fontSize: typography.caption.fontSize,
          borderRadius: borderRadius.sm,
        };
      case 'large':
        return {
          padding: spacing.md,
          fontSize: typography.body.fontSize,
          borderRadius: borderRadius.md,
        };
      default:
        return {
          padding: spacing.sm,
          fontSize: typography.bodySmall.fontSize,
          borderRadius: borderRadius.sm,
        };
    }
  };

  const colors = getVariantColors();
  const sizeStyles = getSizeStyles();

  return (
    <View
      style={[
        styles.badge,
        {
          backgroundColor: colors.bg,
          padding: sizeStyles.padding,
          borderRadius: sizeStyles.borderRadius,
        },
        style,
      ]}
    >
      <Text
        style={[
          styles.text,
          {
            color: colors.text,
            fontSize: sizeStyles.fontSize,
          },
          textStyle,
        ]}
      >
        {label}
      </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  badge: {
    alignSelf: 'flex-start',
  },
  text: {
    fontWeight: '600',
  },
});

