/**
 * Analytics utilities for tracking user events
 * Can be integrated with analytics services like Firebase, Amplitude, etc.
 */

interface AnalyticsEvent {
  name: string;
  properties?: Record<string, unknown>;
  timestamp?: number;
}

class Analytics {
  private events: AnalyticsEvent[] = [];
  private enabled = true;

  /**
   * Enable or disable analytics
   */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
  }

  /**
   * Track an event
   */
  track(eventName: string, properties?: Record<string, unknown>): void {
    if (!this.enabled) {
      return;
    }

    const event: AnalyticsEvent = {
      name: eventName,
      properties,
      timestamp: Date.now(),
    };

    this.events.push(event);

    // In production, send to analytics service
    if (!__DEV__) {
      // Example: Firebase Analytics
      // analytics().logEvent(eventName, properties);
    } else {
      console.log('[Analytics]', eventName, properties);
    }
  }

  /**
   * Track screen view
   */
  trackScreenView(screenName: string, properties?: Record<string, unknown>): void {
    this.track('screen_view', {
      screen_name: screenName,
      ...properties,
    });
  }

  /**
   * Track user action
   */
  trackAction(action: string, properties?: Record<string, unknown>): void {
    this.track('user_action', {
      action,
      ...properties,
    });
  }

  /**
   * Track error
   */
  trackError(error: Error, properties?: Record<string, unknown>): void {
    this.track('error', {
      error_message: error.message,
      error_stack: error.stack,
      ...properties,
    });
  }

  /**
   * Get all tracked events
   */
  getEvents(): AnalyticsEvent[] {
    return [...this.events];
  }

  /**
   * Clear all events
   */
  clear(): void {
    this.events = [];
  }
}

export const analytics = new Analytics();

