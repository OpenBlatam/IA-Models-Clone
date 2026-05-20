import React, { memo, forwardRef } from 'react';
import { TextInput, View, Text, StyleSheet, TextInputProps, ViewStyle } from 'react-native';
import { useTheme } from '@/theme/theme';

export interface InputProps extends TextInputProps {
  label?: string;
  error?: string;
  helperText?: string;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  containerStyle?: ViewStyle;
  accessibilityLabel?: string;
  accessibilityHint?: string;
}

const InputComponent = forwardRef<TextInput, InputProps>(
  (
    {
      label,
      error,
      helperText,
      leftIcon,
      rightIcon,
      containerStyle,
      style,
      accessibilityLabel,
      accessibilityHint,
      ...props
    },
    ref
  ) => {
    const theme = useTheme();
    const hasError = !!error;

    const inputStyle = [
      styles.input,
      {
        backgroundColor: theme.colors.surface,
        borderColor: hasError ? theme.colors.error : theme.colors.border,
        color: theme.colors.text,
        paddingLeft: leftIcon ? 44 : 16,
        paddingRight: rightIcon ? 44 : 16,
      },
      style,
    ];

    return (
      <View style={[styles.container, containerStyle]}>
        {label && (
          <Text
            style={[
              styles.label,
              { color: theme.colors.text },
              hasError && { color: theme.colors.error },
            ]}
            accessibilityRole="text"
          >
            {label}
          </Text>
        )}
        <View style={styles.inputContainer}>
          {leftIcon && <View style={styles.leftIcon}>{leftIcon}</View>}
          <TextInput
            ref={ref}
            style={inputStyle}
            placeholderTextColor={theme.colors.placeholder}
            accessibilityLabel={accessibilityLabel || label}
            accessibilityHint={accessibilityHint}
            accessibilityState={{ invalid: hasError }}
            {...props}
          />
          {rightIcon && <View style={styles.rightIcon}>{rightIcon}</View>}
        </View>
        {(error || helperText) && (
          <Text
            style={[
              styles.helperText,
              { color: hasError ? theme.colors.error : theme.colors.textSecondary },
            ]}
            accessibilityRole="text"
          >
            {error || helperText}
          </Text>
        )}
      </View>
    );
  }
);

InputComponent.displayName = 'Input';

export const Input = memo(InputComponent);

const styles = StyleSheet.create({
  container: {
    marginBottom: 16,
  },
  label: {
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 8,
  },
  inputContainer: {
    position: 'relative',
  },
  input: {
    height: 48,
    borderRadius: 12,
    borderWidth: 1,
    fontSize: 16,
    paddingVertical: 12,
  },
  leftIcon: {
    position: 'absolute',
    left: 12,
    top: 12,
    zIndex: 1,
  },
  rightIcon: {
    position: 'absolute',
    right: 12,
    top: 12,
    zIndex: 1,
  },
  helperText: {
    fontSize: 12,
    marginTop: 4,
    marginLeft: 4,
  },
});


