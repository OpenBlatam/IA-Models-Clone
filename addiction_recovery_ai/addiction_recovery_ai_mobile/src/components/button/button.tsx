import React, { memo, useMemo } from 'react';
import { TouchableOpacity, Text, StyleSheet, ActivityIndicator, ViewStyle, TextStyle } from 'react-native';
import { useColors } from '@/theme/colors';
import {
  getButtonBackgroundColor,
  getButtonTextColor,
  getButtonBorderColor,
} from './button.helpers';
import {
  BUTTON_MIN_HEIGHT,
  BUTTON_PADDING_VERTICAL,
  BUTTON_PADDING_HORIZONTAL,
  BUTTON_BORDER_RADIUS,
  BUTTON_DISABLED_OPACITY,
} from './button.constants';

interface ButtonProps {
  title: string;
  onPress: () => void;
  loading?: boolean;
  disabled?: boolean;
  variant?: 'primary' | 'secondary' | 'outline';
  style?: ViewStyle;
  textStyle?: TextStyle;
  accessibilityLabel?: string;
  accessibilityHint?: string;
}

function ButtonComponent({
  title,
  onPress,
  loading = false,
  disabled = false,
  variant = 'primary',
  style,
  textStyle,
  accessibilityLabel,
  accessibilityHint,
}: ButtonProps): JSX.Element {
  const colors = useColors();

  const backgroundColor = useMemo(
    () => getButtonBackgroundColor(variant, colors),
    [variant, colors]
  );

  const textColor = useMemo(
    () => getButtonTextColor(variant, colors),
    [variant, colors]
  );

  const borderColor = useMemo(
    () => getButtonBorderColor(variant, colors),
    [variant, colors]
  );

  const buttonStyle = useMemo(
    () => [
      styles.button,
      { backgroundColor },
      borderColor && { borderWidth: 1, borderColor },
      (disabled || loading) && { opacity: BUTTON_DISABLED_OPACITY },
      style,
    ],
    [backgroundColor, borderColor, disabled, loading, style]
  );

  const textStyles = useMemo(
    () => [styles.text, { color: textColor }, textStyle],
    [textColor, textStyle]
  );

  const indicatorColor = textColor;

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
        <ActivityIndicator color={indicatorColor} />
      ) : (
        <Text style={textStyles}>{title}</Text>
      )}
    </TouchableOpacity>
  );
}

export const Button = memo(ButtonComponent);

const styles = StyleSheet.create({
  button: {
    paddingVertical: BUTTON_PADDING_VERTICAL,
    paddingHorizontal: BUTTON_PADDING_HORIZONTAL,
    borderRadius: BUTTON_BORDER_RADIUS,
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: BUTTON_MIN_HEIGHT,
  },
  text: {
    fontSize: 16,
    fontWeight: '600',
  },
});

