/**
 * Analytics and metrics utilities
 */

interface MetricEvent {
  name: string;
  properties?: Record<string, any>;
  timestamp: number;
}

class Metrics {
  private events: MetricEvent[] = [];

  track(eventName: string, properties?: Record<string, any>) {
    const event: MetricEvent = {
      name: eventName,
      properties,
      timestamp: Date.now(),
    };

    this.events.push(event);

    if (__DEV__) {
      console.log('[Metrics]', eventName, properties);
    }

    // In production, send to analytics service
    // e.g., Firebase Analytics, Mixpanel, etc.
  }

  trackScreenView(screenName: string) {
    this.track('screen_view', { screen_name: screenName });
  }

  trackAnalysis(imageUri: string, analysisType: string) {
    this.track('analysis_performed', {
      analysis_type: analysisType,
      has_image: !!imageUri,
    });
  }

  trackRecommendationViewed() {
    this.track('recommendation_viewed');
  }

  trackComparisonPerformed() {
    this.track('comparison_performed');
  }

  getEvents(): MetricEvent[] {
    return [...this.events];
  }

  clearEvents() {
    this.events = [];
  }
}

export const metrics = new Metrics();

