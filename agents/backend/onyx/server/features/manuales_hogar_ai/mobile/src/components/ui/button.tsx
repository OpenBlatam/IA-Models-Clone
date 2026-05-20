/**
 * Button Component
 * ================
 * Reusable button component with variants
 */

import { TouchableOpacity, Text, StyleSheet, ActivityIndicator, ViewStyle, TextStyle } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useApp } from '@/lib/context/app-context';

interface ButtonProps {
  title: string;
  onPress: () => void;
  variant?: 'primary' | 'secondary' | 'outline' | 'danger';
  size?: 'small' | 'medium' | 'large';
  disabled?: boolean;
  loading?: boolean;
  icon?: keyof typeof Ionicons.glyphMap;
  iconPosition?: 'left' | 'right';
  fullWidth?: boolean;
  style?: ViewStyle;
}

export function Button({
  title,
  onPress,
  variant = 'primary',
  size = 'medium',
  disabled = false,
  loading = false,
  icon,
  iconPosition = 'left',
  fullWidth = false,
  style,
}: ButtonProps) {
  const { state } = useApp();
  const colors = state.colors;

  const variantStyles = getVariantStyles(variant, colors);
  const sizeStyles = getSizeStyles(size);

  return (
    <TouchableOpacity
      style={[
        styles.button,
        variantStyles.button,
        sizeStyles.button,
        fullWidth && styles.fullWidth,
        (disabled || loading) && styles.disabled,
        style,
      ]}
      onPress={onPress}
      disabled={disabled || loading}
      activeOpacity={0.7}
    >
      {loading ? (
        <ActivityIndicator
          size="small"
          color={variant === 'outline' ? colors.tint : '#FFFFFF'}
        />
      ) : (
        <>
          {icon && iconPosition === 'left' && (
            <Ionicons
              name={icon}
              size={sizeStyles.iconSize}
              color={variantStyles.textColor}
              style={styles.iconLeft}
            />
          )}
          <Text style={[styles.text, variantStyles.text, sizeStyles.text]}>
            {title}
          </Text>
          {icon && iconPosition === 'right' && (
            <Ionicons
              name={icon}
              size={sizeStyles.iconSize}
              color={variantStyles.textColor}
              style={styles.iconRight}
            />
          )}
        </>
      )}
    </TouchableOpacity>
  );
}

function getVariantStyles(variant: string, colors: any) {
  switch (variant) {
    case 'secondary':
      return {
        button: { backgroundColor: colors.secondary },
        text: { color: '#FFFFFF' },
        textColor: '#FFFFFF',
      };
    case 'outline':
      return {
        button: {
          backgroundColor: 'transparent',
          borderWidth: 1,
          borderColor: colors.tint,
        },
        text: { color: colors.tint },
        textColor: colors.tint,
      };
    case 'danger':
      return {
        button: { backgroundColor: colors.error },
        text: { color: '#FFFFFF' },
        textColor: '#FFFFFF',
      };
    default:
      return {
        button: { backgroundColor: colors.tint },
        text: { color: '#FFFFFF' },
        textColor: '#FFFFFF',
      };
  }
}

function getSizeStyles(size: string) {
  switch (size) {
    case 'small':
      return {
        button: { paddingVertical: 8, paddingHorizontal: 16 },
        text: { fontSize: 14 },
        iconSize: 16,
      };
    case 'large':
      return {
        button: { paddingVertical: 16, paddingHorizontal: 24 },
        text: { fontSize: 18 },
        iconSize: 24,
      };
    default:
      return {
        button: { paddingVertical: 12, paddingHorizontal: 20 },
        text: { fontSize: 16 },
        iconSize: 20,
      };
  }
}

const styles = StyleSheet.create({
  button: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 8,
    gap: 8,
  },
  text: {
    fontWeight: '600',
  },
  iconLeft: {
    marginRight: -4,
  },
  iconRight: {
    marginLeft: -4,
  },
  fullWidth: {
    width: '100%',
  },
  disabled: {
    opacity: 0.5,
  },
});



