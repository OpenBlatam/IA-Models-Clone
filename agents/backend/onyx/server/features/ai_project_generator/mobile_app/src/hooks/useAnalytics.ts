import { useCallback } from 'react';
import { storage, STORAGE_KEYS } from '../utils/storage';

export interface AnalyticsEvent {
  name: string;
  properties?: Record<string, any>;
  timestamp: number;
}

const MAX_EVENTS = 1000;

export const useAnalytics = () => {
  const track = useCallback(async (eventName: string, properties?: Record<string, any>) => {
    try {
      const event: AnalyticsEvent = {
        name: eventName,
        properties,
        timestamp: Date.now(),
      };

      const events = await storage.get<AnalyticsEvent[]>(STORAGE_KEYS.ANALYTICS_EVENTS) || [];
      const updated = [...events, event].slice(-MAX_EVENTS);
      await storage.set(STORAGE_KEYS.ANALYTICS_EVENTS, updated);

      if (__DEV__) {
        console.log('Analytics Event:', eventName, properties);
      }
    } catch (error) {
      console.error('Error tracking event:', error);
    }
  }, []);

  const trackScreenView = useCallback(
    (screenName: string, properties?: Record<string, any>) => {
      track('screen_view', { screen: screenName, ...properties });
    },
    [track]
  );

  const trackUserAction = useCallback(
    (action: string, properties?: Record<string, any>) => {
      track('user_action', { action, ...properties });
    },
    [track]
  );

  const trackError = useCallback(
    (error: Error, context?: Record<string, any>) => {
      track('error', {
        error_message: error.message,
        error_stack: error.stack,
        ...context,
      });
    },
    [track]
  );

  const getEvents = useCallback(async (): Promise<AnalyticsEvent[]> => {
    try {
      return (await storage.get<AnalyticsEvent[]>(STORAGE_KEYS.ANALYTICS_EVENTS)) || [];
    } catch (error) {
      console.error('Error getting events:', error);
      return [];
    }
  }, []);

  const clearEvents = useCallback(async () => {
    try {
      await storage.remove(STORAGE_KEYS.ANALYTICS_EVENTS);
    } catch (error) {
      console.error('Error clearing events:', error);
    }
  }, []);

  return {
    track,
    trackScreenView,
    trackUserAction,
    trackError,
    getEvents,
    clearEvents,
  };
};

