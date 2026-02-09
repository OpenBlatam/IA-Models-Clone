import React, { useState } from 'react';
import { TouchableOpacity, Text, StyleSheet, ActivityIndicator } from 'react-native';
import { colors, spacing, borderRadius, typography } from '../theme/colors';

interface RetryButtonProps {
  onRetry: () => void | Promise<void>;
  label?: string;
  loading?: boolean;
  variant?: 'primary' | 'secondary' | 'outline';
}

export const RetryButton: React.FC<RetryButtonProps> = ({
  onRetry,
  label = 'Reintentar',
  loading: externalLoading,
  variant = 'primary',
}) => {
  const [internalLoading, setInternalLoading] = useState(false);

  const loading = externalLoading || internalLoading;

  const handlePress = async () => {
    if (loading) return;
    setInternalLoading(true);
    try {
      await onRetry();
    } finally {
      setInternalLoading(false);
    }
  };

  const getVariantStyles = () => {
    switch (variant) {
      case 'secondary':
        return {
          backgroundColor: colors.surfaceVariant,
          borderColor: colors.border,
          borderWidth: 1,
          textColor: colors.text,
        };
      case 'outline':
        return {
          backgroundColor: 'transparent',
          borderColor: colors.primary,
          borderWidth: 2,
          textColor: colors.primary,
        };
      case 'primary':
      default:
        return {
          backgroundColor: colors.primary,
          borderColor: colors.primary,
          borderWidth: 0,
          textColor: colors.surface,
        };
    }
  };

  const variantStyles = getVariantStyles();

  return (
    <TouchableOpacity
      style={[
        styles.button,
        {
          backgroundColor: variantStyles.backgroundColor,
          borderColor: variantStyles.borderColor,
          borderWidth: variantStyles.borderWidth,
          opacity: loading ? 0.7 : 1,
        },
      ]}
      onPress={handlePress}
      disabled={loading}
      activeOpacity={0.7}
    >
      {loading ? (
        <ActivityIndicator
          size="small"
          color={variantStyles.textColor}
        />
      ) : (
        <>
          <Text style={styles.icon}>🔄</Text>
          <Text
            style={[styles.label, { color: variantStyles.textColor }]}
          >
            {label}
          </Text>
        </>
      )}
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  button: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    borderRadius: borderRadius.md,
    gap: spacing.sm,
  },
  icon: {
    fontSize: 18,
  },
  label: {
    ...typography.body,
    fontWeight: '600',
  },
});

