/**
 * Analytics utility functions.
 * Provides helper functions for analytics tracking.
 */

/**
 * Analytics event.
 */
export interface AnalyticsEvent {
  name: string;
  properties?: Record<string, any>;
  timestamp?: number;
}

/**
 * Analytics tracker class.
 */
export class AnalyticsTracker {
  private events: AnalyticsEvent[] = [];
  private enabled: boolean = true;

  /**
   * Tracks an event.
   */
  track(eventName: string, properties?: Record<string, any>): void {
    if (!this.enabled) return;

    const event: AnalyticsEvent = {
      name: eventName,
      properties,
      timestamp: Date.now(),
    };

    this.events.push(event);

    // Send to analytics service (implement based on your analytics provider)
    if (typeof window !== 'undefined' && (window as any).gtag) {
      (window as any).gtag('event', eventName, properties);
    }
  }

  /**
   * Tracks a page view.
   */
  pageView(path: string, properties?: Record<string, any>): void {
    this.track('page_view', { path, ...properties });
  }

  /**
   * Tracks a user action.
   */
  action(actionName: string, properties?: Record<string, any>): void {
    this.track('user_action', { action: actionName, ...properties });
  }

  /**
   * Tracks an error.
   */
  error(error: Error | string, properties?: Record<string, any>): void {
    const errorMessage = error instanceof Error ? error.message : error;
    this.track('error', {
      error: errorMessage,
      ...properties,
    });
  }

  /**
   * Gets all events.
   */
  getEvents(): AnalyticsEvent[] {
    return [...this.events];
  }

  /**
   * Clears all events.
   */
  clear(): void {
    this.events = [];
  }

  /**
   * Enables or disables tracking.
   */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
  }

  /**
   * Checks if tracking is enabled.
   */
  isEnabled(): boolean {
    return this.enabled;
  }
}

/**
 * Global analytics tracker instance.
 */
export const analytics = new AnalyticsTracker();

/**
 * Tracks an event.
 */
export function trackEvent(
  eventName: string,
  properties?: Record<string, any>
): void {
  analytics.track(eventName, properties);
}

/**
 * Tracks a page view.
 */
export function trackPageView(
  path: string,
  properties?: Record<string, any>
): void {
  analytics.pageView(path, properties);
}

/**
 * Tracks a user action.
 */
export function trackAction(
  actionName: string,
  properties?: Record<string, any>
): void {
  analytics.action(actionName, properties);
}

/**
 * Tracks an error.
 */
export function trackError(
  error: Error | string,
  properties?: Record<string, any>
): void {
  analytics.error(error, properties);
}

