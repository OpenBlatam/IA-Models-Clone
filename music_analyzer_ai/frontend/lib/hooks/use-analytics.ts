/**
 * Custom hook for analytics tracking.
 * Provides reactive analytics tracking functionality.
 */

import { useCallback, useEffect } from 'react';
import {
  AnalyticsTracker,
  trackEvent,
  trackPageView,
  trackAction,
  trackError,
} from '../utils/analytics';

/**
 * Custom hook for analytics tracking.
 * Provides reactive analytics tracking functionality.
 *
 * @returns Analytics tracking operations
 */
export function useAnalytics() {
  const track = useCallback(
    (eventName: string, properties?: Record<string, any>) => {
      trackEvent(eventName, properties);
    },
    []
  );

  const pageView = useCallback(
    (path: string, properties?: Record<string, any>) => {
      trackPageView(path, properties);
    },
    []
  );

  const action = useCallback(
    (actionName: string, properties?: Record<string, any>) => {
      trackAction(actionName, properties);
    },
    []
  );

  const error = useCallback(
    (error: Error | string, properties?: Record<string, any>) => {
      trackError(error, properties);
    },
    []
  );

  return {
    track,
    pageView,
    action,
    error,
  };
}

/**
 * Custom hook for tracking page views.
 * Automatically tracks page views on mount and route changes.
 *
 * @param path - Current page path
 * @param properties - Additional properties
 */
export function usePageView(
  path: string,
  properties?: Record<string, any>
): void {
  useEffect(() => {
    trackPageView(path, properties);
  }, [path, properties]);
}

