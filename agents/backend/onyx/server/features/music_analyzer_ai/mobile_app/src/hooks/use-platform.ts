import { Platform } from 'react-native';

/**
 * Hook to get platform-specific information
 */
export function usePlatform() {
  return {
    isIOS: Platform.OS === 'ios',
    isAndroid: Platform.OS === 'android',
    isWeb: Platform.OS === 'web',
    platform: Platform.OS,
    version: Platform.Version,
    select: Platform.select,
  };
}

/**
 * Hook to check if running on iOS
 */
export function useIsIOS(): boolean {
  return Platform.OS === 'ios';
}

/**
 * Hook to check if running on Android
 */
export function useIsAndroid(): boolean {
  return Platform.OS === 'android';
}

