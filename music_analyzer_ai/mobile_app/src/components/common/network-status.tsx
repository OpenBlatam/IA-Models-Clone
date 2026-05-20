import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useNetworkStatus } from '../../hooks/use-network-status';
import { COLORS, SPACING, TYPOGRAPHY } from '../../constants/config';

export function NetworkStatusBar() {
  const { isConnected, isInternetReachable } = useNetworkStatus();

  if (isConnected && (isInternetReachable ?? true)) {
    return null;
  }

  return (
    <View style={styles.container}>
      <Text style={styles.text}>
        {!isConnected
          ? 'No internet connection'
          : 'Internet connection unavailable'}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: COLORS.error,
    padding: SPACING.sm,
    alignItems: 'center',
    justifyContent: 'center',
  },
  text: {
    ...TYPOGRAPHY.bodySmall,
    color: COLORS.text,
    fontWeight: '600',
  },
});

