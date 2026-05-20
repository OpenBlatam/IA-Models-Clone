/**
 * Badge Component
 * ==============
 * Badge component for labels and counts
 */

import { View, Text, StyleSheet } from 'react-native';
import { useApp } from '@/lib/context/app-context';

interface BadgeProps {
  text: string | number;
  variant?: 'default' | 'success' | 'error' | 'warning' | 'info';
  size?: 'small' | 'medium' | 'large';
}

export function Badge({ text, variant = 'default', size = 'medium' }: BadgeProps) {
  const { state } = useApp();
  const colors = state.colors;

  const variantColors = {
    default: colors.textSecondary,
    success: colors.success,
    error: colors.error,
    warning: colors.warning,
    info: colors.info,
  };

  const sizeStyles = {
    small: { padding: 4, fontSize: 10, minWidth: 16, height: 16 },
    medium: { padding: 6, fontSize: 12, minWidth: 20, height: 20 },
    large: { padding: 8, fontSize: 14, minWidth: 24, height: 24 },
  };

  return (
    <View
      style={[
        styles.badge,
        {
          backgroundColor: variantColors[variant],
          ...sizeStyles[size],
        },
      ]}
    >
      <Text style={[styles.text, { fontSize: sizeStyles[size].fontSize }]}>
        {text}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  badge: {
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 6,
  },
  text: {
    color: '#FFFFFF',
    fontWeight: '600',
  },
});



