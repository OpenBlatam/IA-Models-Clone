/**
 * Analytics and tracking utilities
 */

export interface AnalyticsEvent {
  name: string
  properties?: Record<string, any>
  timestamp?: number
}

export interface AnalyticsTracker {
  track: (event: AnalyticsEvent) => void
  page: (name: string, properties?: Record<string, any>) => void
  identify: (userId: string, traits?: Record<string, any>) => void
  reset: () => void
}

class DefaultAnalyticsTracker implements AnalyticsTracker {
  private events: AnalyticsEvent[] = []

  track(event: AnalyticsEvent): void {
    const fullEvent = {
      ...event,
      timestamp: event.timestamp || Date.now(),
    }
    this.events.push(fullEvent)
    console.log('[Analytics]', fullEvent)
  }

  page(name: string, properties?: Record<string, any>): void {
    this.track({
      name: 'page_view',
      properties: {
        page: name,
        ...properties,
      },
    })
  }

  identify(userId: string, traits?: Record<string, any>): void {
    this.track({
      name: 'identify',
      properties: {
        userId,
        ...traits,
      },
    })
  }

  reset(): void {
    this.events = []
  }

  getEvents(): AnalyticsEvent[] {
    return [...this.events]
  }
}

let analyticsTracker: AnalyticsTracker = new DefaultAnalyticsTracker()

/**
 * Set custom analytics tracker
 */
export function setAnalyticsTracker(tracker: AnalyticsTracker): void {
  analyticsTracker = tracker
}

/**
 * Get current analytics tracker
 */
export function getAnalyticsTracker(): AnalyticsTracker {
  return analyticsTracker
}

/**
 * Track event
 */
export function trackEvent(name: string, properties?: Record<string, any>): void {
  analyticsTracker.track({ name, properties })
}

/**
 * Track page view
 */
export function trackPageView(name: string, properties?: Record<string, any>): void {
  analyticsTracker.page(name, properties)
}

/**
 * Identify user
 */
export function identifyUser(userId: string, traits?: Record<string, any>): void {
  analyticsTracker.identify(userId, traits)
}

/**
 * Track message sent
 */
export function trackMessageSent(messageLength: number, hasCode: boolean, hasLinks: boolean): void {
  trackEvent('message_sent', {
    messageLength,
    hasCode,
    hasLinks,
  })
}

/**
 * Track message received
 */
export function trackMessageReceived(responseTime: number, wordCount: number): void {
  trackEvent('message_received', {
    responseTime,
    wordCount,
  })
}

/**
 * Track search performed
 */
export function trackSearch(query: string, resultCount: number): void {
  trackEvent('search_performed', {
    queryLength: query.length,
    resultCount,
  })
}

/**
 * Track export
 */
export function trackExport(format: string, messageCount: number): void {
  trackEvent('export_performed', {
    format,
    messageCount,
  })
}

/**
 * Track import
 */
export function trackImport(format: string, messageCount: number): void {
  trackEvent('import_performed', {
    format,
    messageCount,
  })
}

/**
 * Track settings change
 */
export function trackSettingsChange(setting: string, value: any): void {
  trackEvent('settings_changed', {
    setting,
    value: typeof value === 'object' ? JSON.stringify(value) : value,
  })
}

/**
 * Track performance metric
 */
export function trackPerformance(metric: string, value: number, unit: string = 'ms'): void {
  trackEvent('performance_metric', {
    metric,
    value,
    unit,
  })
}

/**
 * Track error
 */
export function trackError(error: Error, context?: string): void {
  trackEvent('error_occurred', {
    errorName: error.name,
    errorMessage: error.message,
    context,
  })
}




