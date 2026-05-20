import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { useNetworkStatus } from '../hooks/useNetworkStatus';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, typography } from '../theme/colors';

interface OfflineBannerProps {
  onRetry?: () => void;
  showRetryButton?: boolean;
}

export const OfflineBanner: React.FC<OfflineBannerProps> = ({
  onRetry,
  showRetryButton = true,
}) => {
  const { isConnected } = useNetworkStatus();
  const { theme } = useTheme();

  if (isConnected) {
    return null;
  }

  return (
    <View
      style={[
        styles.container,
        {
          backgroundColor: theme.error,
        },
      ]}
    >
      <View style={styles.content}>
        <Text style={[styles.text, { color: theme.surface }]}>
          ⚠️ Sin conexión a internet. Algunas funciones pueden no estar disponibles.
        </Text>
        {showRetryButton && onRetry && (
          <TouchableOpacity
            style={[styles.retryButton, { backgroundColor: theme.surface }]}
            onPress={onRetry}
            activeOpacity={0.7}
          >
            <Text style={[styles.retryText, { color: theme.error }]}>
              Reintentar
            </Text>
          </TouchableOpacity>
        )}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    padding: spacing.md,
  },
  content: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  text: {
    ...typography.bodySmall,
    flex: 1,
    marginRight: spacing.md,
  },
  retryButton: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: 4,
  },
  retryText: {
    ...typography.bodySmall,
    fontWeight: '600',
  },
});

