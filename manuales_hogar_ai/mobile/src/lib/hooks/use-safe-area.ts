/**
 * useSafeArea Hook
 * ================
 * Hook for safe area insets
 */

import { useSafeAreaInsets } from 'react-native-safe-area-context';

export function useSafeArea() {
  const insets = useSafeAreaInsets();

  return {
    top: insets.top,
    bottom: insets.bottom,
    left: insets.left,
    right: insets.right,
  };
}




