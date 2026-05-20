import { useCallback } from 'react';
import { trackScreenView, trackUserAction } from '@/utils/analytics';

export function useAnalytics() {
  const trackScreen = useCallback((screenName: string) => {
    trackScreenView(screenName);
  }, []);

  const trackAction = useCallback(
    (action: string, properties?: Record<string, unknown>) => {
      trackUserAction(action, properties);
    },
    []
  );

  return {
    trackScreen,
    trackAction,
  };
}

