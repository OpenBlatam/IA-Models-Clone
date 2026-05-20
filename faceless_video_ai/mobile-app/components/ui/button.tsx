import React from 'react';
import {
  TouchableOpacity,
  Text,
  StyleSheet,
  ActivityIndicator,
  ViewStyle,
  TextStyle,
} from 'react-native';
import { useColorScheme } from 'react-native';

interface ButtonProps {
  title: string;
  onPress: () => void;
  variant?: 'primary' | 'secondary' | 'outline' | 'danger';
  size?: 'small' | 'medium' | 'large';
  disabled?: boolean;
  loading?: boolean;
  style?: ViewStyle;
  textStyle?: TextStyle;
}

export function Button({
  title,
  onPress,
  variant = 'primary',
  size = 'medium',
  disabled = false,
  loading = false,
  style,
  textStyle,
}: ButtonProps) {
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';

  const buttonStyles = [
    styles.button,
    styles[`button_${variant}`],
    styles[`button_${size}`],
    disabled && styles.buttonDisabled,
    isDark && styles.buttonDark,
    style,
  ];

  const textStyles = [
    styles.text,
    styles[`text_${variant}`],
    styles[`text_${size}`],
    disabled && styles.textDisabled,
    textStyle,
  ];

  return (
    <TouchableOpacity
      style={buttonStyles}
      onPress={onPress}
      disabled={disabled || loading}
      activeOpacity={0.7}
    >
      {loading ? (
        <ActivityIndicator
          color={variant === 'primary' ? '#FFFFFF' : '#000000'}
          size="small"
        />
      ) : (
        <Text style={textStyles}>{title}</Text>
      )}
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  button: {
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
  },
  button_primary: {
    backgroundColor: '#007AFF',
  },
  button_secondary: {
    backgroundColor: '#5856D6',
  },
  button_outline: {
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: '#007AFF',
  },
  button_danger: {
    backgroundColor: '#FF3B30',
  },
  button_small: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    minHeight: 32,
  },
  button_medium: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    minHeight: 44,
  },
  button_large: {
    paddingHorizontal: 24,
    paddingVertical: 16,
    minHeight: 52,
  },
  buttonDisabled: {
    opacity: 0.5,
  },
  buttonDark: {
    // Dark mode styles can be added here
  },
  text: {
    fontWeight: '600',
  },
  text_primary: {
    color: '#FFFFFF',
  },
  text_secondary: {
    color: '#FFFFFF',
  },
  text_outline: {
    color: '#007AFF',
  },
  text_danger: {
    color: '#FFFFFF',
  },
  text_small: {
    fontSize: 14,
  },
  text_medium: {
    fontSize: 16,
  },
  text_large: {
    fontSize: 18,
  },
  textDisabled: {
    opacity: 0.5,
  },
});


