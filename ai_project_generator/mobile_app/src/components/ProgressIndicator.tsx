import React from 'react';
import { View, Text, StyleSheet, ActivityIndicator } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, typography } from '../theme/colors';

interface ProgressIndicatorProps {
  message?: string;
  size?: 'small' | 'large';
  color?: string;
}

export const ProgressIndicator: React.FC<ProgressIndicatorProps> = ({
  message,
  size = 'large',
  color,
}) => {
  const { theme } = useTheme();

  return (
    <View style={styles.container}>
      <ActivityIndicator
        size={size}
        color={color || theme.primary}
      />
      {message && (
        <Text style={[styles.message, { color: theme.textSecondary }]}>
          {message}
        </Text>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: spacing.xl,
    gap: spacing.md,
  },
  message: {
    ...typography.body,
    marginTop: spacing.sm,
  },
});

