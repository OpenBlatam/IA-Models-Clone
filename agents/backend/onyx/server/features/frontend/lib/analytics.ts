type EventName = string;
type EventData = Record<string, any>;

interface AnalyticsEvent {
  name: EventName;
  data: EventData;
  timestamp: number;
}

class Analytics {
  private events: AnalyticsEvent[] = [];
  private enabled = true;
  private maxEvents = 1000;

  track(eventName: EventName, eventData: EventData = {}) {
    if (!this.enabled) return;

    const event: AnalyticsEvent = {
      name: eventName,
      data: {
        ...eventData,
        url: typeof window !== 'undefined' ? window.location.href : '',
        userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : '',
      },
      timestamp: Date.now(),
    };

    this.events.push(event);

    // Keep only last maxEvents
    if (this.events.length > this.maxEvents) {
      this.events.shift();
    }

    // Log to console in development
    if (process.env.NODE_ENV === 'development') {
      console.log('[Analytics]', eventName, eventData);
    }
  }

  pageView(path: string, data: EventData = {}) {
    this.track('page_view', { path, ...data });
  }

  click(element: string, data: EventData = {}) {
    this.track('click', { element, ...data });
  }

  error(error: Error | string, data: EventData = {}) {
    this.track('error', {
      message: error instanceof Error ? error.message : error,
      stack: error instanceof Error ? error.stack : undefined,
      ...data,
    });
  }

  getEvents(eventName?: EventName): AnalyticsEvent[] {
    if (eventName) {
      return this.events.filter((e) => e.name === eventName);
    }
    return [...this.events];
  }

  getEventCount(eventName?: EventName): number {
    if (eventName) {
      return this.events.filter((e) => e.name === eventName).length;
    }
    return this.events.length;
  }

  clear() {
    this.events = [];
  }

  enable() {
    this.enabled = true;
  }

  disable() {
    this.enabled = false;
  }

  export() {
    return JSON.stringify(this.events, null, 2);
  }
}

export const analytics = new Analytics();

