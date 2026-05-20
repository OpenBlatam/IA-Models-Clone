import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useNetworkStatus } from '@/hooks/use-network-status';
import { useColors } from '@/theme/colors';
import { cn } from '@/utils/class-names';

export function NetworkStatus(): JSX.Element | null {
  const { isConnected } = useNetworkStatus();
  const colors = useColors();

  if (isConnected) {
    return null;
  }

  return (
    <View
      style={[
        styles.container,
        { backgroundColor: colors.error },
      ]}
      accessibilityRole="alert"
      accessibilityLiveRegion="polite"
    >
      <Text style={[styles.text, { color: colors.surface }]}>
        Sin conexión a Internet
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  text: {
    fontSize: 14,
    fontWeight: '600',
  },
});

