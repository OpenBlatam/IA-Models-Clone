import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, borderRadius, typography } from '../theme/colors';

interface TagProps {
  label: string;
  onPress?: () => void;
  onClose?: () => void;
  variant?: 'default' | 'primary' | 'success' | 'warning' | 'error';
  size?: 'small' | 'medium' | 'large';
  icon?: React.ReactNode;
}

export const Tag: React.FC<TagProps> = ({
  label,
  onPress,
  onClose,
  variant = 'default',
  size = 'medium',
  icon,
}) => {
  const { theme } = useTheme();

  const getVariantStyles = () => {
    switch (variant) {
      case 'primary':
        return {
          backgroundColor: theme.primary + '20',
          textColor: theme.primary,
        };
      case 'success':
        return {
          backgroundColor: '#4CAF50' + '20',
          textColor: '#4CAF50',
        };
      case 'warning':
        return {
          backgroundColor: '#FF9800' + '20',
          textColor: '#FF9800',
        };
      case 'error':
        return {
          backgroundColor: theme.error + '20',
          textColor: theme.error,
        };
      default:
        return {
          backgroundColor: theme.surfaceVariant,
          textColor: theme.text,
        };
    }
  };

  const getSizeStyles = () => {
    switch (size) {
      case 'small':
        return {
          paddingVertical: spacing.xs,
          paddingHorizontal: spacing.sm,
          fontSize: typography.caption.fontSize,
        };
      case 'large':
        return {
          paddingVertical: spacing.md,
          paddingHorizontal: spacing.lg,
          fontSize: typography.body.fontSize,
        };
      default:
        return {
          paddingVertical: spacing.sm,
          paddingHorizontal: spacing.md,
          fontSize: typography.bodySmall.fontSize,
        };
    }
  };

  const variantStyles = getVariantStyles();
  const sizeStyles = getSizeStyles();

  const Component = onPress || onClose ? TouchableOpacity : View;

  return (
    <Component
      style={[
        styles.tag,
        {
          backgroundColor: variantStyles.backgroundColor,
          paddingVertical: sizeStyles.paddingVertical,
          paddingHorizontal: sizeStyles.paddingHorizontal,
        },
      ]}
      onPress={onPress}
      activeOpacity={onPress ? 0.7 : 1}
    >
      {icon && <View style={styles.icon}>{icon}</View>}
      <Text
        style={[
          styles.label,
          {
            color: variantStyles.textColor,
            fontSize: sizeStyles.fontSize,
          },
        ]}
      >
        {label}
      </Text>
      {onClose && (
        <TouchableOpacity
          style={styles.closeButton}
          onPress={onClose}
          hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
        >
          <Text style={[styles.closeIcon, { color: variantStyles.textColor }]}>
            ×
          </Text>
        </TouchableOpacity>
      )}
    </Component>
  );
};

const styles = StyleSheet.create({
  tag: {
    flexDirection: 'row',
    alignItems: 'center',
    borderRadius: borderRadius.full,
    alignSelf: 'flex-start',
  },
  icon: {
    marginRight: spacing.xs,
  },
  label: {
    ...typography.bodySmall,
    fontWeight: '500',
  },
  closeButton: {
    marginLeft: spacing.xs,
    padding: spacing.xs,
  },
  closeIcon: {
    fontSize: 18,
    fontWeight: '600',
    lineHeight: 18,
  },
});

