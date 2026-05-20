import React, { memo } from 'react';
import {
  TouchableOpacity,
  Text,
  StyleSheet,
  ActivityIndicator,
  ViewStyle,
  TextStyle,
} from 'react-native';
import { useTheme } from '@/theme/theme';

export interface ButtonProps {
  title: string;
  onPress: () => void;
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost';
  size?: 'small' | 'medium' | 'large';
  disabled?: boolean;
  loading?: boolean;
  fullWidth?: boolean;
  style?: ViewStyle;
  textStyle?: TextStyle;
  accessibilityLabel?: string;
  accessibilityHint?: string;
}

function ButtonComponent({
  title,
  onPress,
  variant = 'primary',
  size = 'medium',
  disabled = false,
  loading = false,
  fullWidth = false,
  style,
  textStyle,
  accessibilityLabel,
  accessibilityHint,
}: ButtonProps) {
  const theme = useTheme();

  const buttonStyle = [
    styles.button,
    styles[`button_${variant}`],
    styles[`button_${size}`],
    fullWidth && styles.buttonFullWidth,
    (disabled || loading) && styles.buttonDisabled,
    { backgroundColor: getBackgroundColor(variant, theme) },
    { borderColor: getBorderColor(variant, theme) },
    style,
  ];

  const textStyles = [
    styles.text,
    styles[`text_${variant}`],
    styles[`text_${size}`],
    { color: getTextColor(variant, theme) },
    textStyle,
  ];

  return (
    <TouchableOpacity
      style={buttonStyle}
      onPress={onPress}
      disabled={disabled || loading}
      activeOpacity={0.7}
      accessibilityRole="button"
      accessibilityLabel={accessibilityLabel || title}
      accessibilityHint={accessibilityHint}
      accessibilityState={{ disabled: disabled || loading }}
    >
      {loading ? (
        <ActivityIndicator
          size="small"
          color={getTextColor(variant, theme)}
          accessibilityElementsHidden
        />
      ) : (
        <Text style={textStyles}>{title}</Text>
      )}
    </TouchableOpacity>
  );
}

function getBackgroundColor(variant: string, theme: ReturnType<typeof useTheme>): string {
  switch (variant) {
    case 'primary':
      return theme.colors.primary;
    case 'secondary':
      return theme.colors.secondary;
    case 'outline':
    case 'ghost':
      return 'transparent';
    default:
      return theme.colors.primary;
  }
}

function getBorderColor(variant: string, theme: ReturnType<typeof useTheme>): string {
  switch (variant) {
    case 'outline':
      return theme.colors.primary;
    case 'ghost':
      return 'transparent';
    default:
      return 'transparent';
  }
}

function getTextColor(variant: string, theme: ReturnType<typeof useTheme>): string {
  switch (variant) {
    case 'primary':
    case 'secondary':
      return '#FFFFFF';
    case 'outline':
    case 'ghost':
      return theme.colors.primary;
    default:
      return '#FFFFFF';
  }
}

export const Button = memo(ButtonComponent);

const styles = StyleSheet.create({
  button: {
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
  },
  button_primary: {
    borderWidth: 0,
  },
  button_secondary: {
    borderWidth: 0,
  },
  button_outline: {
    borderWidth: 2,
  },
  button_ghost: {
    borderWidth: 0,
  },
  button_small: {
    paddingVertical: 8,
    paddingHorizontal: 16,
    minHeight: 36,
  },
  button_medium: {
    paddingVertical: 12,
    paddingHorizontal: 24,
    minHeight: 48,
  },
  button_large: {
    paddingVertical: 16,
    paddingHorizontal: 32,
    minHeight: 56,
  },
  buttonFullWidth: {
    width: '100%',
  },
  buttonDisabled: {
    opacity: 0.5,
  },
  text: {
    fontWeight: '600',
    textAlign: 'center',
  },
  text_primary: {},
  text_secondary: {},
  text_outline: {},
  text_ghost: {},
  text_small: {
    fontSize: 14,
  },
  text_medium: {
    fontSize: 16,
  },
  text_large: {
    fontSize: 18,
  },
});


