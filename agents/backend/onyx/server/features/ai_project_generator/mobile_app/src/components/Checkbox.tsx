import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, borderRadius, typography } from '../theme/colors';
import { hapticFeedback } from '../utils/haptics';

interface CheckboxProps {
  checked: boolean;
  onPress: () => void;
  label?: string;
  disabled?: boolean;
  size?: 'small' | 'medium' | 'large';
}

export const Checkbox: React.FC<CheckboxProps> = ({
  checked,
  onPress,
  label,
  disabled = false,
  size = 'medium',
}) => {
  const { theme } = useTheme();

  const getSize = () => {
    switch (size) {
      case 'small':
        return 18;
      case 'large':
        return 28;
      default:
        return 24;
    }
  };

  const checkboxSize = getSize();

  const handlePress = () => {
    if (!disabled) {
      hapticFeedback.selection();
      onPress();
    }
  };

  return (
    <TouchableOpacity
      style={styles.container}
      onPress={handlePress}
      disabled={disabled}
      activeOpacity={0.7}
    >
      <View
        style={[
          styles.checkbox,
          {
            width: checkboxSize,
            height: checkboxSize,
            borderColor: checked ? theme.primary : theme.border,
            backgroundColor: checked ? theme.primary : 'transparent',
            opacity: disabled ? 0.5 : 1,
          },
        ]}
      >
        {checked && (
          <Text style={[styles.checkmark, { color: theme.surface }]}>✓</Text>
        )}
      </View>
      {label && (
        <Text
          style={[
            styles.label,
            {
              color: disabled ? theme.textTertiary : theme.text,
            },
          ]}
        >
          {label}
        </Text>
      )}
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: spacing.sm,
  },
  checkbox: {
    borderWidth: 2,
    borderRadius: borderRadius.sm,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: spacing.sm,
  },
  checkmark: {
    fontSize: 14,
    fontWeight: '600',
  },
  label: {
    ...typography.body,
    flex: 1,
  },
});

