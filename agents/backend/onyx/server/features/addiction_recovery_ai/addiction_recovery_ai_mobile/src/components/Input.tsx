import React, { memo } from 'react';
import { View, TextInput, Text, StyleSheet, TextInputProps, ViewStyle, TextStyle } from 'react-native';
import { useColors } from '@/theme/colors';

interface InputProps extends TextInputProps {
  label?: string;
  error?: string;
  containerStyle?: ViewStyle;
  inputStyle?: TextStyle;
  labelStyle?: TextStyle;
  accessibilityLabel?: string;
  accessibilityHint?: string;
}

function InputComponent({
  label,
  error,
  containerStyle,
  style,
  inputStyle,
  labelStyle,
  accessibilityLabel,
  accessibilityHint,
  ...props
}: InputProps): JSX.Element {
  const colors = useColors();

  const inputStyles = [
    styles.input,
    { 
      borderColor: error ? colors.error : colors.border,
      backgroundColor: colors.surface,
      color: colors.text,
    },
    error && styles.inputError,
    inputStyle,
    style,
  ];

  return (
    <View style={[styles.container, containerStyle]}>
      {label && (
        <Text 
          style={[
            styles.label, 
            { color: colors.text },
            labelStyle
          ]}
          accessibilityRole="text"
        >
          {label}
        </Text>
      )}
      <TextInput
        style={inputStyles}
        placeholderTextColor={colors.textSecondary}
        accessibilityLabel={accessibilityLabel || label}
        accessibilityHint={accessibilityHint}
        accessibilityState={{ invalid: !!error }}
        {...props}
      />
      {error && (
        <Text 
          style={[styles.errorText, { color: colors.error }]}
          accessibilityLiveRegion="polite"
          accessibilityRole="alert"
        >
          {error}
        </Text>
      )}
    </View>
  );
}

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
  input: {
    borderWidth: 1,
    borderRadius: 8,
    paddingHorizontal: 16,
    paddingVertical: 12,
    fontSize: 16,
  },
  inputError: {
    borderWidth: 2,
  },
  errorText: {
    fontSize: 12,
    marginTop: 4,
  },
});

