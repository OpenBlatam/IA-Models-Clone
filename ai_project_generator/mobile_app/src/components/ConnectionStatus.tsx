import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useNetworkStatus } from '../hooks/useNetworkStatus';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, typography } from '../theme/colors';

interface ConnectionStatusProps {
  showWhenConnected?: boolean;
  position?: 'top' | 'bottom';
}

export const ConnectionStatus: React.FC<ConnectionStatusProps> = ({
  showWhenConnected = false,
  position = 'top',
}) => {
  const { isConnected, connectionType } = useNetworkStatus();
  const { theme } = useTheme();

  if (isConnected && !showWhenConnected) {
    return null;
  }

  return (
    <View
      style={[
        styles.container,
        {
          backgroundColor: isConnected ? '#4CAF50' : theme.error,
          top: position === 'top' ? 0 : undefined,
          bottom: position === 'bottom' ? 0 : undefined,
        },
      ]}
    >
      <Text style={[styles.text, { color: theme.surface }]}>
        {isConnected
          ? `Conectado${connectionType ? ` - ${connectionType}` : ''}`
          : 'Sin conexión a internet'}
      </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    alignItems: 'center',
    justifyContent: 'center',
    position: 'absolute',
    left: 0,
    right: 0,
    zIndex: 1000,
  },
  text: {
    ...typography.bodySmall,
    fontWeight: '600',
  },
});

