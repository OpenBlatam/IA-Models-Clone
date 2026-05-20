import React from 'react';
import { TouchableOpacity, Text, StyleSheet, ActivityIndicator } from 'react-native';
import { colors, spacing, borderRadius, typography } from '../theme/colors';

interface RefreshButtonProps {
  onPress: () => void;
  refreshing?: boolean;
  size?: 'small' | 'medium' | 'large';
}

export const RefreshButton: React.FC<RefreshButtonProps> = ({
  onPress,
  refreshing = false,
  size = 'medium',
}) => {
  const getSize = () => {
    switch (size) {
      case 'small':
        return { width: 32, height: 32, fontSize: 16 };
      case 'large':
        return { width: 48, height: 48, fontSize: 24 };
      default:
        return { width: 40, height: 40, fontSize: 20 };
    }
  };

  const sizeStyle = getSize();

  return (
    <TouchableOpacity
      style={[styles.button, { width: sizeStyle.width, height: sizeStyle.height }]}
      onPress={onPress}
      disabled={refreshing}
      activeOpacity={0.7}
    >
      {refreshing ? (
        <ActivityIndicator size="small" color={colors.primary} />
      ) : (
        <Text style={[styles.icon, { fontSize: sizeStyle.fontSize }]}>🔄</Text>
      )}
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  button: {
    borderRadius: borderRadius.full,
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: colors.shadow,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  icon: {
    color: colors.primary,
  },
});

