import React from 'react';
import { TouchableOpacity, Text, StyleSheet } from 'react-native';
import { colors, spacing, borderRadius, typography } from '../theme/colors';

interface FloatingActionButtonProps {
  onPress: () => void;
  icon?: string;
  label?: string;
  position?: 'bottom-right' | 'bottom-left' | 'top-right' | 'top-left';
}

export const FloatingActionButton: React.FC<FloatingActionButtonProps> = ({
  onPress,
  icon = '+',
  label,
  position = 'bottom-right',
}) => {
  const getPositionStyle = () => {
    switch (position) {
      case 'bottom-right':
        return { bottom: spacing.xl, right: spacing.xl };
      case 'bottom-left':
        return { bottom: spacing.xl, left: spacing.xl };
      case 'top-right':
        return { top: spacing.xl, right: spacing.xl };
      case 'top-left':
        return { top: spacing.xl, left: spacing.xl };
      default:
        return { bottom: spacing.xl, right: spacing.xl };
    }
  };

  return (
    <TouchableOpacity
      style={[styles.button, getPositionStyle()]}
      onPress={onPress}
      activeOpacity={0.8}
    >
      <Text style={styles.icon}>{icon}</Text>
      {label && <Text style={styles.label}>{label}</Text>}
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  button: {
    position: 'absolute',
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: colors.primary,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: colors.shadowDark,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
    zIndex: 1000,
  },
  icon: {
    fontSize: 28,
    color: colors.surface,
    fontWeight: '300',
  },
  label: {
    ...typography.caption,
    color: colors.surface,
    marginTop: spacing.xs,
  },
});

