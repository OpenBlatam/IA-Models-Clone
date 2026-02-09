/**
 * Loading Spinner
 * ===============
 * Loading indicator component
 */

import { View, ActivityIndicator, StyleSheet } from 'react-native';
import { useApp } from '@/lib/context/app-context';

export function LoadingSpinner() {
  const { state } = useApp();

  return (
    <View style={styles.container}>
      <ActivityIndicator size="large" color={state.colors.tint} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
});




