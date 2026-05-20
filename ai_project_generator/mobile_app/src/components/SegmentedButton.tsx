import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, borderRadius, typography } from '../theme/colors';
import { hapticFeedback } from '../utils/haptics';

interface SegmentedButtonOption {
  label: string;
  value: string;
  icon?: React.ReactNode;
}

interface SegmentedButtonProps {
  options: SegmentedButtonOption[];
  selectedValue: string;
  onValueChange: (value: string) => void;
  size?: 'small' | 'medium' | 'large';
}

export const SegmentedButton: React.FC<SegmentedButtonProps> = ({
  options,
  selectedValue,
  onValueChange,
  size = 'medium',
}) => {
  const { theme } = useTheme();

  const getSizeStyles = () => {
    switch (size) {
      case 'small':
        return { padding: spacing.sm, fontSize: typography.bodySmall.fontSize };
      case 'large':
        return { padding: spacing.lg, fontSize: typography.body.fontSize };
      default:
        return { padding: spacing.md, fontSize: typography.bodySmall.fontSize };
    }
  };

  const sizeStyles = getSizeStyles();

  const handlePress = (value: string) => {
    hapticFeedback.selection();
    onValueChange(value);
  };

  return (
    <View
      style={[
        styles.container,
        {
          backgroundColor: theme.surfaceVariant,
          borderRadius: borderRadius.md,
          padding: 2,
        },
      ]}
    >
      {options.map((option, index) => {
        const isSelected = option.value === selectedValue;
        const isFirst = index === 0;
        const isLast = index === options.length - 1;

        return (
          <TouchableOpacity
            key={option.value}
            style={[
              styles.button,
              {
                padding: sizeStyles.padding,
                backgroundColor: isSelected ? theme.surface : 'transparent',
                borderRadius: borderRadius.sm,
                marginRight: isLast ? 0 : 2,
              },
            ]}
            onPress={() => handlePress(option.value)}
            activeOpacity={0.7}
          >
            {option.icon && (
              <View style={styles.iconContainer}>{option.icon}</View>
            )}
            <Text
              style={[
                styles.label,
                {
                  fontSize: sizeStyles.fontSize,
                  color: isSelected ? theme.primary : theme.text,
                  fontWeight: isSelected ? '600' : '400',
                },
              ]}
            >
              {option.label}
            </Text>
          </TouchableOpacity>
        );
      })}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
  },
  button: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  iconContainer: {
    marginRight: spacing.xs,
  },
  label: {
    ...typography.bodySmall,
  },
});

