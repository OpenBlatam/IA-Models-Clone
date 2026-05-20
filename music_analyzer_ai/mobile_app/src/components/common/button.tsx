import React from 'react';
import {
  TouchableOpacity,
  Text,
  StyleSheet,
  ActivityIndicator,
  ViewStyle,
  TextStyle,
} from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
} from 'react-native-reanimated';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from '../../constants/config';
import { useHapticFeedback } from '../../hooks/use-haptic-feedback';

const AnimatedTouchable = Animated.createAnimatedComponent(TouchableOpacity);

interface ButtonProps {
  title?: string;
  label?: string;
  onPress: () => void;
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost';
  size?: 'small' | 'medium' | 'large';
  loading?: boolean;
  disabled?: boolean;
  fullWidth?: boolean;
  style?: ViewStyle;
  textStyle?: TextStyle;
  accessibilityLabel?: string;
}

export const Button: React.FC<ButtonProps> = ({
  title,
  label,
  onPress,
  variant = 'primary',
  size = 'medium',
  loading = false,
  disabled = false,
  fullWidth = false,
  style,
  textStyle,
  accessibilityLabel,
}) => {
  const displayText = title || label || '';
  const haptics = useHapticFeedback();
  const scale = useSharedValue(1);

  const handlePressIn = () => {
    haptics.selection();
    scale.value = withSpring(0.95);
  };

  const handlePressOut = () => {
    scale.value = withSpring(1);
  };

  const handlePress = () => {
    if (!disabled && !loading) {
      haptics.light();
      onPress();
    }
  };

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
  }));

  const buttonStyle = [
    styles.button,
    styles[variant],
    styles[size],
    fullWidth && styles.fullWidth,
    (disabled || loading) && styles.disabled,
    style,
  ];

  const buttonTextStyle = [
    styles.text,
    styles[`${variant}Text`],
    styles[`${size}Text`],
    textStyle,
  ];

  return (
    <AnimatedTouchable
      style={[buttonStyle, animatedStyle]}
      onPress={handlePress}
      onPressIn={handlePressIn}
      onPressOut={handlePressOut}
      disabled={disabled || loading}
      activeOpacity={0.8}
      accessibilityRole="button"
      accessibilityLabel={accessibilityLabel || displayText}
      accessibilityState={{ disabled: disabled || loading }}
    >
      {loading ? (
        <ActivityIndicator
          size="small"
          color={variant === 'primary' ? COLORS.text : COLORS.primary}
        />
      ) : (
        <Text style={buttonTextStyle}>{displayText}</Text>
      )}
    </AnimatedTouchable>
  );
}

const styles = StyleSheet.create({
  button: {
    borderRadius: BORDER_RADIUS.md,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
  },
  primary: {
    backgroundColor: COLORS.primary,
  },
  secondary: {
    backgroundColor: COLORS.secondary,
  },
  outline: {
    backgroundColor: 'transparent',
    borderWidth: 2,
    borderColor: COLORS.primary,
  },
  ghost: {
    backgroundColor: 'transparent',
  },
  small: {
    paddingVertical: SPACING.xs,
    paddingHorizontal: SPACING.sm,
    minHeight: 32,
  },
  medium: {
    paddingVertical: SPACING.sm,
    paddingHorizontal: SPACING.md,
    minHeight: 44,
  },
  large: {
    paddingVertical: SPACING.md,
    paddingHorizontal: SPACING.lg,
    minHeight: 56,
  },
  fullWidth: {
    width: '100%',
  },
  disabled: {
    opacity: 0.5,
  },
  text: {
    ...TYPOGRAPHY.body,
    fontWeight: '600',
  },
  primaryText: {
    color: COLORS.text,
  },
  secondaryText: {
    color: COLORS.text,
  },
  outlineText: {
    color: COLORS.primary,
  },
  ghostText: {
    color: COLORS.text,
  },
  smallText: {
    fontSize: 12,
  },
  mediumText: {
    fontSize: 16,
  },
  largeText: {
    fontSize: 18,
  },
});

