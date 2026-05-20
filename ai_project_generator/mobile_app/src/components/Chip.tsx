import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ViewStyle, TextStyle } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, borderRadius, typography } from '../theme/colors';
import { hapticFeedback } from '../utils/haptics';

interface ChipProps {
  label: string;
  selected?: boolean;
  onPress?: () => void;
  icon?: React.ReactNode;
  variant?: 'default' | 'outlined' | 'filled';
  style?: ViewStyle;
  textStyle?: TextStyle;
}

export const Chip: React.FC<ChipProps> = ({
  label,
  selected = false,
  onPress,
  icon,
  variant = 'default',
  style,
  textStyle,
}) => {
  const { theme } = useTheme();

  const getVariantStyles = () => {
    if (variant === 'filled') {
      return {
        backgroundColor: selected ? theme.primary : theme.surfaceVariant,
        borderWidth: 0,
        borderColor: 'transparent',
        textColor: selected ? theme.surface : theme.text,
      };
    }
    if (variant === 'outlined') {
      return {
        backgroundColor: 'transparent',
        borderWidth: 1,
        borderColor: selected ? theme.primary : theme.border,
        textColor: selected ? theme.primary : theme.text,
      };
    }
    return {
      backgroundColor: selected ? theme.primary : theme.surfaceVariant,
      borderWidth: 0,
      borderColor: 'transparent',
      textColor: selected ? theme.surface : theme.text,
    };
  };

  const variantStyles = getVariantStyles();

  const Component = onPress ? TouchableOpacity : View;

  return (
    <Component
      style={[
        styles.chip,
        {
          backgroundColor: variantStyles.backgroundColor,
          borderColor: variantStyles.borderColor,
          borderWidth: variantStyles.borderWidth,
        },
        style,
      ]}
      onPress={() => {
        if (onPress) {
          hapticFeedback.selection();
          onPress();
        }
      }}
      activeOpacity={onPress ? 0.7 : 1}
    >
      {icon && <View style={styles.iconContainer}>{icon}</View>}
      <Text
        style={[
          styles.text,
          {
            color: variantStyles.textColor,
          },
          textStyle,
        ]}
      >
        {label}
      </Text>
    </Component>
  );
};

const styles = StyleSheet.create({
  chip: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: borderRadius.full,
    alignSelf: 'flex-start',
  },
  iconContainer: {
    marginRight: spacing.xs,
  },
  text: {
    ...typography.bodySmall,
    fontWeight: '500',
  },
});

