import { useEffect } from 'react';
import { BackHandler } from 'react-native';
import { useRouter } from 'expo-router';

/**
 * Hook to handle Android back button
 * Can prevent default behavior or execute custom action
 */
export function useBackHandler(
  onBackPress: () => boolean
): void {
  useEffect(() => {
    const backHandler = BackHandler.addEventListener(
      'hardwareBackPress',
      onBackPress
    );

    return () => backHandler.remove();
  }, [onBackPress]);
}

/**
 * Hook to prevent back navigation
 */
export function usePreventBack(): void {
  useBackHandler(() => {
    return true; // Prevent default back behavior
  });
}

/**
 * Hook to handle back navigation with router
 */
export function useBackNavigation(): () => void {
  const router = useRouter();

  const handleBack = () => {
    if (router.canGoBack()) {
      router.back();
    }
  };

  useBackHandler(() => {
    handleBack();
    return true;
  });

  return handleBack;
}

