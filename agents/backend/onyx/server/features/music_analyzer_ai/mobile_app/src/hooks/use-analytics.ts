import { useEffect } from 'react';
import { usePathname } from 'expo-router';
import { analytics } from '../utils/analytics';

/**
 * Hook to automatically track screen views
 */
export function useScreenTracking(screenName?: string): void {
  const pathname = usePathname();

  useEffect(() => {
    const screen = screenName || pathname;
    analytics.trackScreenView(screen);
  }, [pathname, screenName]);
}

/**
 * Hook to track user actions
 */
export function useTrackAction() {
  return (action: string, properties?: Record<string, unknown>) => {
    analytics.trackAction(action, properties);
  };
}

