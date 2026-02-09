import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, borderRadius, typography } from '../theme/colors';
import { hapticFeedback } from '../utils/haptics';

interface SwitchProps {
  value: boolean;
  onValueChange: (value: boolean) => void;
  label?: string;
  description?: string;
  disabled?: boolean;
  size?: 'small' | 'medium' | 'large';
}

export const Switch: React.FC<SwitchProps> = ({
  value,
  onValueChange,
  label,
  description,
  disabled = false,
  size = 'medium',
}) => {
  const { theme } = useTheme();

  const getSizeStyles = () => {
    switch (size) {
      case 'small':
        return { width: 40, height: 24, thumbSize: 18 };
      case 'large':
        return { width: 56, height: 32, thumbSize: 26 };
      default:
        return { width: 48, height: 28, thumbSize: 22 };
    }
  };

  const sizeStyles = getSizeStyles();

  const handleToggle = () => {
    if (!disabled) {
      hapticFeedback.selection();
      onValueChange(!value);
    }
  };

  return (
    <View style={styles.container}>
      {(label || description) && (
        <View style={styles.labelContainer}>
          {label && (
            <Text style={[styles.label, { color: theme.text }]}>{label}</Text>
          )}
          {description && (
            <Text style={[styles.description, { color: theme.textSecondary }]}>
              {description}
            </Text>
          )}
        </View>
      )}
      <TouchableOpacity
        style={[
          styles.track,
          {
            width: sizeStyles.width,
            height: sizeStyles.height,
            backgroundColor: value ? theme.primary : theme.border,
            opacity: disabled ? 0.5 : 1,
          },
        ]}
        onPress={handleToggle}
        disabled={disabled}
        activeOpacity={0.7}
      >
        <View
          style={[
            styles.thumb,
            {
              width: sizeStyles.thumbSize,
              height: sizeStyles.thumbSize,
              backgroundColor: theme.surface,
              transform: [{ translateX: value ? sizeStyles.width - sizeStyles.thumbSize - 4 : 2 }],
            },
          ]}
        />
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: spacing.md,
  },
  labelContainer: {
    flex: 1,
    marginRight: spacing.md,
  },
  label: {
    ...typography.body,
    fontWeight: '500',
    marginBottom: spacing.xs,
  },
  description: {
    ...typography.caption,
  },
  track: {
    borderRadius: 999,
    justifyContent: 'center',
    padding: 2,
  },
  thumb: {
    borderRadius: 999,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 2,
    elevation: 2,
  },
});

