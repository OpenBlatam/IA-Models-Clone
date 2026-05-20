import React, { memo, useCallback, useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from '../../constants/config';
import { getErrorRecoveryMessage, getErrorRecoveryStrategy, calculateRetryDelay } from '../../utils/error-recovery';
import { RetryButton } from './retry-button';
import type { ApiError } from '../../types';

interface ErrorRecoveryProps {
  error: unknown;
  onRetry?: () => void;
  retryOptions?: {
    maxRetries?: number;
    initialDelay?: number;
    maxDelay?: number;
    backoffMultiplier?: number;
  };
}

function ErrorRecoveryComponent({
  error,
  onRetry,
  retryOptions,
}: ErrorRecoveryProps) {
  const [retryAttempt, setRetryAttempt] = useState(0);
  const [isRetrying, setIsRetrying] = useState(false);

  const recoveryInfo = getErrorRecoveryMessage(error);
  const strategy = getErrorRecoveryStrategy(error, retryOptions);

  const handleRetry = useCallback(() => {
    if (!onRetry || !strategy.canRetry || retryAttempt >= strategy.maxRetries) {
      return;
    }

    setIsRetrying(true);
    const delay = calculateRetryDelay(
      retryAttempt,
      strategy.retryDelay,
      retryOptions?.maxDelay ?? 10000,
      strategy.backoffMultiplier
    );

    setTimeout(() => {
      onRetry();
      setRetryAttempt((prev) => prev + 1);
      setIsRetrying(false);
    }, delay);
  }, [onRetry, strategy, retryAttempt, retryOptions]);

  const canRetry = strategy.canRetry && retryAttempt < strategy.maxRetries;

  return (
    <View style={styles.container}>
      <Text style={styles.message} accessibilityRole="text">
        {recoveryInfo.message}
      </Text>
      {recoveryInfo.suggestion && (
        <Text style={styles.suggestion} accessibilityRole="text">
          {recoveryInfo.suggestion}
        </Text>
      )}
      {recoveryInfo.action && (
        <Text style={styles.action} accessibilityRole="text">
          {recoveryInfo.action}
        </Text>
      )}
      {canRetry && onRetry && (
        <View style={styles.retryContainer}>
          <RetryButton
            onPress={handleRetry}
            loading={isRetrying}
            label={isRetrying ? 'Retrying...' : 'Retry'}
          />
          {retryAttempt > 0 && (
            <Text style={styles.attemptText}>
              Attempt {retryAttempt + 1} of {strategy.maxRetries}
            </Text>
          )}
        </View>
      )}
    </View>
  );
}

export const ErrorRecovery = memo(ErrorRecoveryComponent);

const styles = StyleSheet.create({
  container: {
    padding: SPACING.lg,
    backgroundColor: COLORS.surface,
    borderRadius: BORDER_RADIUS.md,
    margin: SPACING.md,
  },
  message: {
    ...TYPOGRAPHY.body,
    color: COLORS.error,
    marginBottom: SPACING.sm,
    fontWeight: '600',
  },
  suggestion: {
    ...TYPOGRAPHY.body,
    color: COLORS.textSecondary,
    marginBottom: SPACING.xs,
  },
  action: {
    ...TYPOGRAPHY.caption,
    color: COLORS.textSecondary,
    marginBottom: SPACING.md,
    fontStyle: 'italic',
  },
  retryContainer: {
    marginTop: SPACING.md,
    alignItems: 'center',
  },
  attemptText: {
    ...TYPOGRAPHY.caption,
    color: COLORS.textSecondary,
    marginTop: SPACING.xs,
  },
});
