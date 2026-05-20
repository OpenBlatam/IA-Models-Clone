import { trackEvent, trackPerformance } from './monitoring';

export interface AnalyticsEvent {
  name: string;
  properties?: Record<string, unknown>;
}

export function trackScreenView(screenName: string): void {
  trackEvent(`screen_view.${screenName}`, {
    screen: screenName,
    timestamp: Date.now(),
  });
}

export function trackUserAction(
  action: string,
  properties?: Record<string, unknown>
): void {
  trackEvent(`user_action.${action}`, {
    action,
    ...properties,
    timestamp: Date.now(),
  });
}

export function trackAPIRequest(
  endpoint: string,
  duration: number,
  success: boolean
): void {
  trackPerformance({
    name: `api.request.${endpoint}`,
    value: duration,
    unit: 'millisecond',
    tags: {
      endpoint,
      success: String(success),
    },
  });
}

export function trackError(
  error: Error,
  context?: Record<string, unknown>
): void {
  trackEvent('error.occurred', {
    error: error.message,
    stack: error.stack,
    ...context,
    timestamp: Date.now(),
  });
}

