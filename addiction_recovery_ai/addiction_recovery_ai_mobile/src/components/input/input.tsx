import React, { memo, useMemo } from 'react';
import { View, TextInput, Text, StyleSheet, TextInputProps, ViewStyle, TextStyle } from 'react-native';
import { useColors } from '@/theme/colors';
import { sanitizeString } from '@/utils/sanitize';
import { getInputBorderColor, getInputBorderWidth } from './input.helpers';
import {
  INPUT_BORDER_RADIUS,
  INPUT_PADDING_HORIZONTAL,
  INPUT_PADDING_VERTICAL,
  INPUT_FONT_SIZE,
  LABEL_FONT_SIZE,
  LABEL_FONT_WEIGHT,
  ERROR_FONT_SIZE,
} from './input.constants';

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
  onChangeText,
  ...props
}: InputProps): JSX.Element {
  const colors = useColors();

  const borderColor = useMemo(
    () => getInputBorderColor(!!error, colors),
    [error, colors]
  );

  const borderWidth = useMemo(
    () => getInputBorderWidth(!!error),
    [error]
  );

  const handleChangeText = (text: string) => {
    const sanitized = sanitizeString(text);
    onChangeText?.(sanitized);
  };

  const inputStyles = useMemo(
    () => [
      styles.input,
      {
        borderColor,
        borderWidth,
        backgroundColor: colors.surface,
        color: colors.text,
      },
      inputStyle,
      style,
    ],
    [borderColor, borderWidth, colors, inputStyle, style]
  );

  return (
    <View style={[styles.container, containerStyle]}>
      {label && (
        <Text
          style={[styles.label, { color: colors.text }, labelStyle]}
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
        onChangeText={handleChangeText}
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
    fontSize: LABEL_FONT_SIZE,
    fontWeight: LABEL_FONT_WEIGHT,
    marginBottom: 8,
  },
  input: {
    borderRadius: INPUT_BORDER_RADIUS,
    paddingHorizontal: INPUT_PADDING_HORIZONTAL,
    paddingVertical: INPUT_PADDING_VERTICAL,
    fontSize: INPUT_FONT_SIZE,
  },
  errorText: {
    fontSize: ERROR_FONT_SIZE,
    marginTop: 4,
  },
});

