import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { ApiError } from '../types';
import { colors, spacing, typography } from '../theme/colors';
import { RetryButton } from './RetryButton';

interface ErrorMessageProps {
  error: ApiError | string;
  onRetry?: () => void | Promise<void>;
}

export const ErrorMessage: React.FC<ErrorMessageProps> = ({ error, onRetry }) => {
  const message = typeof error === 'string' ? error : error.detail;

  return (
    <View style={styles.container}>
      <Text style={styles.icon}>⚠️</Text>
      <Text style={styles.title}>Oops, algo salió mal</Text>
      <Text style={styles.message}>{message}</Text>
      {onRetry && <RetryButton onRetry={onRetry} />}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: spacing.xl,
    backgroundColor: colors.background,
  },
  icon: {
    fontSize: 64,
    marginBottom: spacing.lg,
  },
  title: {
    ...typography.h3,
    color: colors.text,
    marginBottom: spacing.sm,
  },
  message: {
    ...typography.body,
    color: colors.textSecondary,
    textAlign: 'center',
    marginBottom: spacing.xl,
    paddingHorizontal: spacing.lg,
  },
});
