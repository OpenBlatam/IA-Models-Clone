import { useSafeAreaInsets } from 'react-native-safe-area-context';

/**
 * Hook for safe area insets
 * Provides safe area values for all edges
 */
export function useSafeArea() {
  const insets = useSafeAreaInsets();

  return {
    top: insets.top,
    bottom: insets.bottom,
    left: insets.left,
    right: insets.right,
    all: insets,
  };
}

