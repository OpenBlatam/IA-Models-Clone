import React, { useMemo, useCallback } from 'react';
import {
  TouchableOpacity,
  Text,
  View,
  StyleSheet,
  ActivityIndicator,
  ViewStyle,
  TextStyle,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useTheme } from '../context/ThemeContext';

interface ButtonProps {
  title: string;
  onPress: () => void;
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost';
  size?: 'small' | 'medium' | 'large';
  loading?: boolean;
  disabled?: boolean;
  icon?: React.ReactNode;
  fullWidth?: boolean;
  style?: ViewStyle;
  textStyle?: TextStyle;
}

const SIZE_STYLES = {
  small: { paddingVertical: 8, paddingHorizontal: 16, fontSize: 14 },
  medium: { paddingVertical: 12, paddingHorizontal: 20, fontSize: 16 },
  large: { paddingVertical: 16, paddingHorizontal: 24, fontSize: 18 },
} as const;

const Button: React.FC<ButtonProps> = ({
  title,
  onPress,
  variant = 'primary',
  size = 'medium',
  loading = false,
  disabled = false,
  icon,
  fullWidth = false,
  style,
  textStyle,
}) => {
  const { colors } = useTheme();

  const sizeStyles = useMemo(() => SIZE_STYLES[size], [size]);

  const handlePress = useCallback(() => {
    if (!disabled && !loading) {
      onPress();
    }
  }, [disabled, loading, onPress]);

  const isDisabled = useMemo(() => disabled || loading, [disabled, loading]);

  const buttonStyle = useMemo(
    () => [
      styles.button,
      sizeStyles,
      fullWidth && styles.fullWidth,
      isDisabled && styles.disabled,
      style,
    ],
    [sizeStyles, fullWidth, isDisabled, style]
  );

  const variantStyle = useMemo(() => {
    switch (variant) {
      case 'secondary':
        return { backgroundColor: colors.secondary };
      case 'outline':
        return {
          backgroundColor: 'transparent',
          borderWidth: 1,
          borderColor: colors.primary,
        };
      case 'ghost':
        return { backgroundColor: 'transparent' };
      default:
        return {};
    }
  }, [variant, colors]);

  const textColorStyle = useMemo(() => {
    if (variant === 'primary' || variant === 'secondary') {
      return styles.primaryText;
    }
    return { color: colors.primary };
  }, [variant, colors.primary]);

  const gradientColors = useMemo(
    () => [colors.primary, colors.secondary],
    [colors.primary, colors.secondary]
  );

  if (variant === 'primary') {
    return (
      <LinearGradient colors={gradientColors} style={buttonStyle}>
        <TouchableOpacity
          onPress={handlePress}
          disabled={isDisabled}
          activeOpacity={0.8}
          style={styles.touchable}
        >
          {loading ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <>
              {icon && <View style={styles.iconContainer}>{icon}</View>}
              <Text style={[styles.text, styles.primaryText, textStyle]}>
                {title}
              </Text>
            </>
          )}
        </TouchableOpacity>
      </LinearGradient>
    );
  }

  return (
    <TouchableOpacity
      onPress={handlePress}
      disabled={isDisabled}
      activeOpacity={0.8}
      style={[buttonStyle, variantStyle]}
    >
      {loading ? (
        <ActivityIndicator color={colors.primary} />
      ) : (
        <>
          {icon && <View style={styles.iconContainer}>{icon}</View>}
          <Text style={[styles.text, textColorStyle, textStyle]}>{title}</Text>
        </>
      )}
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  button: {
    borderRadius: 12,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  touchable: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    width: '100%',
  },
  fullWidth: {
    width: '100%',
  },
  disabled: {
    opacity: 0.5,
  },
  text: {
    fontWeight: '600',
  },
  primaryText: {
    color: '#fff',
  },
  iconContainer: {
    marginRight: 8,
  },
});

export default React.memo(Button);

