import { useEffect } from 'react';
import { useRouter } from 'expo-router';
import { initializeDeepLinking } from '../utils/deep-linking';

/**
 * Hook to initialize and handle deep linking
 */
export function useDeepLink(): void {
  const router = useRouter();

  useEffect(() => {
    const cleanup = initializeDeepLinking(router);
    return cleanup;
  }, [router]);
}

