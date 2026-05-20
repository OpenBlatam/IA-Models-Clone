/**
 * Analytics utility for tracking user interactions
 */

interface AnalyticsEvent {
  action: string;
  category: string;
  label?: string;
  value?: number;
}

class Analytics {
  private enabled: boolean;

  constructor() {
    this.enabled = typeof window !== 'undefined' && process.env.NODE_ENV === 'production';
  }

  track(event: AnalyticsEvent) {
    if (!this.enabled) {
      return;
    }

    // Google Analytics 4
    if (typeof window !== 'undefined' && (window as any).gtag) {
      (window as any).gtag('event', event.action, {
        event_category: event.category,
        event_label: event.label,
        value: event.value,
      });
    }

    // Console log in development
    if (process.env.NODE_ENV === 'development') {
      console.log('[Analytics]', event);
    }
  }

  pageView(path: string) {
    if (!this.enabled) {
      return;
    }

    if (typeof window !== 'undefined' && (window as any).gtag) {
      (window as any).gtag('config', 'GA_MEASUREMENT_ID', {
        page_path: path,
      });
    }
  }

  setUser(userId: string) {
    if (!this.enabled) {
      return;
    }

    if (typeof window !== 'undefined' && (window as any).gtag) {
      (window as any).gtag('set', { user_id: userId });
    }
  }
}

export const analytics = new Analytics();

// Helper functions
export function trackEvent(action: string, category: string, label?: string, value?: number) {
  analytics.track({ action, category, label, value });
}

export function trackPageView(path: string) {
  analytics.pageView(path);
}

export function trackUser(userId: string) {
  analytics.setUser(userId);
}



