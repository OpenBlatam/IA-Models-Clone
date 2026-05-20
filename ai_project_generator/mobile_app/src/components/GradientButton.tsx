import React from 'react';
import { TouchableOpacity, Text, StyleSheet, ViewStyle, TextStyle } from 'react-native';
// @ts-ignore - expo-linear-gradient types
import { LinearGradient } from 'expo-linear-gradient';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, borderRadius, typography } from '../theme/colors';
import { hapticFeedback } from '../utils/haptics';

interface GradientButtonProps {
  title: string;
  onPress: () => void;
  colors?: string[];
  disabled?: boolean;
  loading?: boolean;
  size?: 'small' | 'medium' | 'large';
  fullWidth?: boolean;
  style?: ViewStyle;
  textStyle?: TextStyle;
}

export const GradientButton: React.FC<GradientButtonProps> = ({
  title,
  onPress,
  colors,
  disabled = false,
  loading = false,
  size = 'medium',
  fullWidth = false,
  style,
  textStyle,
}) => {
  const { theme } = useTheme();

  const defaultColors = colors || [theme.primary, theme.primary + 'CC'];

  const getSizeStyles = () => {
    switch (size) {
      case 'small':
        return { padding: spacing.sm, fontSize: typography.bodySmall.fontSize };
      case 'large':
        return { padding: spacing.xl, fontSize: typography.h3.fontSize };
      default:
        return { padding: spacing.md, fontSize: typography.body.fontSize };
    }
  };

  const sizeStyles = getSizeStyles();

  const handlePress = () => {
    if (!disabled && !loading) {
      hapticFeedback.selection();
      onPress();
    }
  };

  return (
    <TouchableOpacity
      style={[
        styles.container,
        {
          width: fullWidth ? '100%' : 'auto',
          opacity: disabled || loading ? 0.6 : 1,
        },
        style,
      ]}
      onPress={handlePress}
      disabled={disabled || loading}
      activeOpacity={0.8}
    >
      <LinearGradient
        colors={defaultColors}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 0 }}
        style={[
          styles.gradient,
          {
            padding: sizeStyles.padding,
            borderRadius: borderRadius.md,
          },
        ]}
      >
        <Text
          style={[
            styles.text,
            {
              fontSize: sizeStyles.fontSize,
              color: '#FFFFFF',
            },
            textStyle,
          ]}
        >
          {loading ? 'Cargando...' : title}
        </Text>
      </LinearGradient>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    overflow: 'hidden',
  },
  gradient: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  text: {
    ...typography.body,
    fontWeight: '600',
  },
});

