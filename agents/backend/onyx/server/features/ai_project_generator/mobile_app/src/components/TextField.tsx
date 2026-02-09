import React, { useState } from 'react';
import { View, Text, TextInput, StyleSheet, TouchableOpacity } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, borderRadius, typography } from '../theme/colors';

interface TextFieldProps {
  label?: string;
  placeholder?: string;
  value: string;
  onChangeText: (text: string) => void;
  error?: string;
  helperText?: string;
  multiline?: boolean;
  numberOfLines?: number;
  secureTextEntry?: boolean;
  keyboardType?: 'default' | 'email-address' | 'numeric' | 'phone-pad' | 'url';
  autoCapitalize?: 'none' | 'sentences' | 'words' | 'characters';
  disabled?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  maxLength?: number;
  required?: boolean;
}

export const TextField: React.FC<TextFieldProps> = ({
  label,
  placeholder,
  value,
  onChangeText,
  error,
  helperText,
  multiline = false,
  numberOfLines = 1,
  secureTextEntry = false,
  keyboardType = 'default',
  autoCapitalize = 'sentences',
  disabled = false,
  leftIcon,
  rightIcon,
  maxLength,
  required = false,
}) => {
  const { theme } = useTheme();
  const [focused, setFocused] = useState(false);

  return (
    <View style={styles.container}>
      {label && (
        <Text style={[styles.label, { color: theme.text }]}>
          {label}
          {required && <Text style={{ color: theme.error }}> *</Text>}
        </Text>
      )}
      <View
        style={[
          styles.inputContainer,
          {
            backgroundColor: disabled ? theme.surfaceVariant : theme.surface,
            borderColor: error
              ? theme.error
              : focused
              ? theme.primary
              : theme.border,
            borderWidth: error || focused ? 2 : 1,
            alignItems: multiline ? 'flex-start' : 'center',
          },
        ]}
      >
        {leftIcon && <View style={styles.leftIcon}>{leftIcon}</View>}
        <TextInput
          style={[
            styles.input,
            {
              color: theme.text,
              minHeight: multiline ? 100 : 44,
            },
          ]}
          placeholder={placeholder}
          placeholderTextColor={theme.textTertiary}
          value={value}
          onChangeText={onChangeText}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          multiline={multiline}
          numberOfLines={numberOfLines}
          secureTextEntry={secureTextEntry}
          keyboardType={keyboardType}
          autoCapitalize={autoCapitalize}
          editable={!disabled}
          maxLength={maxLength}
        />
        {rightIcon && <View style={styles.rightIcon}>{rightIcon}</View>}
      </View>
      {(error || helperText) && (
        <Text
          style={[
            styles.helperText,
            {
              color: error ? theme.error : theme.textSecondary,
            },
          ]}
        >
          {error || helperText}
        </Text>
      )}
      {maxLength && (
        <Text style={[styles.counter, { color: theme.textTertiary }]}>
          {value.length}/{maxLength}
        </Text>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginBottom: spacing.md,
  },
  label: {
    ...typography.bodySmall,
    fontWeight: '600',
    marginBottom: spacing.sm,
  },
  inputContainer: {
    flexDirection: 'row',
    borderRadius: borderRadius.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderWidth: 1,
  },
  leftIcon: {
    marginRight: spacing.sm,
    justifyContent: 'center',
  },
  input: {
    flex: 1,
    ...typography.body,
    paddingVertical: spacing.xs,
  },
  rightIcon: {
    marginLeft: spacing.sm,
    justifyContent: 'center',
  },
  helperText: {
    ...typography.caption,
    marginTop: spacing.xs,
  },
  counter: {
    ...typography.caption,
    textAlign: 'right',
    marginTop: spacing.xs,
  },
});

