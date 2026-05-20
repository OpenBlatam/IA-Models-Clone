import React, { useState } from 'react';
import { View, Text, TextInput, StyleSheet } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, borderRadius, typography } from '../theme/colors';
import Animated, { useAnimatedStyle, withTiming } from 'react-native-reanimated';

interface FloatingLabelInputProps {
  label: string;
  value: string;
  onChangeText: (text: string) => void;
  placeholder?: string;
  error?: string;
  secureTextEntry?: boolean;
  keyboardType?: 'default' | 'email-address' | 'numeric' | 'phone-pad';
  autoCapitalize?: 'none' | 'sentences' | 'words' | 'characters';
  disabled?: boolean;
  multiline?: boolean;
  numberOfLines?: number;
}

export const FloatingLabelInput: React.FC<FloatingLabelInputProps> = ({
  label,
  value,
  onChangeText,
  placeholder,
  error,
  secureTextEntry = false,
  keyboardType = 'default',
  autoCapitalize = 'sentences',
  disabled = false,
  multiline = false,
  numberOfLines = 1,
}) => {
  const { theme } = useTheme();
  const [focused, setFocused] = useState(false);
  const isActive = focused || value.length > 0;

  const labelStyle = useAnimatedStyle(() => {
    return {
      top: withTiming(isActive ? -8 : spacing.md, { duration: 200 }),
      fontSize: withTiming(isActive ? typography.caption.fontSize : typography.body.fontSize, {
        duration: 200,
      }),
      color: withTiming(
        error
          ? theme.error
          : focused
          ? theme.primary
          : theme.textSecondary,
        { duration: 200 }
      ),
    };
  });

  return (
    <View style={styles.container}>
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
          },
        ]}
      >
        <Animated.Text
          style={[
            styles.label,
            labelStyle,
            {
              backgroundColor: theme.surface,
              paddingHorizontal: spacing.xs,
            },
          ]}
        >
          {label}
        </Animated.Text>
        <TextInput
          style={[
            styles.input,
            {
              color: theme.text,
              minHeight: multiline ? 100 : 44,
            },
          ]}
          value={value}
          onChangeText={onChangeText}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          placeholder={isActive ? placeholder : ''}
          placeholderTextColor={theme.textTertiary}
          secureTextEntry={secureTextEntry}
          keyboardType={keyboardType}
          autoCapitalize={autoCapitalize}
          editable={!disabled}
          multiline={multiline}
          numberOfLines={numberOfLines}
        />
      </View>
      {error && (
        <Text style={[styles.errorText, { color: theme.error }]}>
          {error}
        </Text>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginBottom: spacing.md,
  },
  inputContainer: {
    borderRadius: borderRadius.md,
    borderWidth: 1,
    paddingHorizontal: spacing.md,
    paddingTop: spacing.md,
    paddingBottom: spacing.sm,
    position: 'relative',
  },
  label: {
    position: 'absolute',
    left: spacing.md,
    ...typography.body,
  },
  input: {
    ...typography.body,
    padding: 0,
    marginTop: spacing.xs,
  },
  errorText: {
    ...typography.caption,
    marginTop: spacing.xs,
    marginLeft: spacing.md,
  },
});

