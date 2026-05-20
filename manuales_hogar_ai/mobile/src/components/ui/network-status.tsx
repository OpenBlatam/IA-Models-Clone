/**
 * Network Status
 * ==============
 * Component to show network connectivity status
 */

import { View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useNetworkStatus } from '@/hooks/use-network-status';
import { useApp } from '@/lib/context/app-context';
import Animated, { FadeInDown, FadeOutUp } from 'react-native-reanimated';

export function NetworkStatus() {
  const { isOffline } = useNetworkStatus();
  const { state } = useApp();
  const colors = state.colors;

  if (!isOffline) return null;

  return (
    <Animated.View
      entering={FadeInDown}
      exiting={FadeOutUp}
      style={[styles.container, { backgroundColor: colors.error }]}
    >
      <Ionicons name="cloud-offline" size={20} color="#FFFFFF" />
      <Text style={styles.text}>No internet connection</Text>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 8,
    paddingHorizontal: 16,
    gap: 8,
  },
  text: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '500',
  },
});



