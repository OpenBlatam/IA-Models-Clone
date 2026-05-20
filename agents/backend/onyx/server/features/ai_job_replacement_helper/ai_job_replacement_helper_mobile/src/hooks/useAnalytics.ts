import { useEffect } from 'react';
import { usePathname } from 'expo-router';
import { analytics } from '@/utils/analytics';

export function useAnalytics() {
  const pathname = usePathname();

  useEffect(() => {
    // Track screen view
    if (pathname) {
      analytics.screenView(pathname);
    }
  }, [pathname]);

  return {
    logEvent: analytics.logEvent.bind(analytics),
    setUserProperties: analytics.setUserProperties.bind(analytics),
    setUserId: analytics.setUserId.bind(analytics),
  };
}


