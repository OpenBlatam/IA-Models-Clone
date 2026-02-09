/**
 * Analytics utilities for Robot 3D View
 * @module robot-3d-view/utils/analytics
 */

/**
 * Analytics event types
 */
export type AnalyticsEventType =
  | 'view-loaded'
  | 'config-changed'
  | 'preset-applied'
  | 'theme-changed'
  | 'screenshot-taken'
  | 'shortcut-used'
  | 'error-occurred'
  | 'performance-warning';

/**
 * Analytics event
 */
export interface AnalyticsEvent {
  type: AnalyticsEventType;
  timestamp: number;
  data?: Record<string, unknown>;
}

/**
 * Analytics Manager class
 */
export class AnalyticsManager {
  private events: AnalyticsEvent[] = [];
  private maxEvents = 1000;
  private enabled = true;

  /**
   * Tracks an event
   */
  track(type: AnalyticsEventType, data?: Record<string, unknown>): void {
    if (!this.enabled) return;

    this.events.push({
      type,
      timestamp: Date.now(),
      data,
    });

    // Limit events array size
    if (this.events.length > this.maxEvents) {
      this.events.shift();
    }
  }

  /**
   * Gets all events
   */
  getEvents(): readonly AnalyticsEvent[] {
    return [...this.events];
  }

  /**
   * Gets events by type
   */
  getEventsByType(type: AnalyticsEventType): AnalyticsEvent[] {
    return this.events.filter((e) => e.type === type);
  }

  /**
   * Gets event count by type
   */
  getEventCount(type: AnalyticsEventType): number {
    return this.getEventsByType(type).length;
  }

  /**
   * Clears all events
   */
  clear(): void {
    this.events = [];
  }

  /**
   * Exports analytics data
   */
  export(): string {
    return JSON.stringify(this.events, null, 2);
  }

  /**
   * Enables/disables analytics
   */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
  }

  /**
   * Checks if analytics is enabled
   */
  isEnabled(): boolean {
    return this.enabled;
  }
}

/**
 * Global analytics manager instance
 */
export const analyticsManager = new AnalyticsManager();

/**
 * Helper function to track events
 */
export function trackEvent(
  type: AnalyticsEventType,
  data?: Record<string, unknown>
): void {
  analyticsManager.track(type, data);
}



