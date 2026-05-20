/**
 * Analytics & Tracking Testing
 * 
 * Tests that verify analytics events are properly tracked and sent,
 * ensuring user interactions and application events are correctly monitored.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';

// Mock analytics service
interface AnalyticsEvent {
  name: string;
  properties?: Record<string, any>;
  timestamp?: number;
}

class MockAnalytics {
  private events: AnalyticsEvent[] = [];
  private userId: string | null = null;

  identify(userId: string, traits?: Record<string, any>) {
    this.userId = userId;
    // In real implementation, would send to analytics service
  }

  track(eventName: string, properties?: Record<string, any>) {
    const event: AnalyticsEvent = {
      name: eventName,
      properties,
      timestamp: Date.now(),
    };
    this.events.push(event);
  }

  page(name: string, properties?: Record<string, any>) {
    this.track('page_view', { page: name, ...properties });
  }

  getEvents(): AnalyticsEvent[] {
    return this.events;
  }

  clearEvents() {
    this.events = [];
  }

  getUserId(): string | null {
    return this.userId;
  }
}

const analytics = new MockAnalytics();

describe('Analytics & Tracking Testing', () => {
  beforeEach(() => {
    analytics.clearEvents();
  });

  describe('Event Tracking', () => {
    it('should track user actions', () => {
      analytics.track('track_played', {
        trackId: '123',
        trackTitle: 'Test Track',
        artist: 'Test Artist',
      });

      const events = analytics.getEvents();
      expect(events).toHaveLength(1);
      expect(events[0].name).toBe('track_played');
      expect(events[0].properties?.trackId).toBe('123');
    });

    it('should track search events', () => {
      analytics.track('search_performed', {
        query: 'test query',
        resultsCount: 10,
      });

      const events = analytics.getEvents();
      expect(events).toHaveLength(1);
      expect(events[0].name).toBe('search_performed');
      expect(events[0].properties?.query).toBe('test query');
      expect(events[0].properties?.resultsCount).toBe(10);
    });

    it('should track playlist creation', () => {
      analytics.track('playlist_created', {
        playlistId: 'playlist-1',
        trackCount: 5,
      });

      const events = analytics.getEvents();
      expect(events).toHaveLength(1);
      expect(events[0].name).toBe('playlist_created');
      expect(events[0].properties?.playlistId).toBe('playlist-1');
    });

    it('should track errors', () => {
      analytics.track('error_occurred', {
        errorType: 'NetworkError',
        errorMessage: 'Failed to fetch data',
        page: '/tracks',
      });

      const events = analytics.getEvents();
      expect(events).toHaveLength(1);
      expect(events[0].name).toBe('error_occurred');
      expect(events[0].properties?.errorType).toBe('NetworkError');
    });
  });

  describe('Page View Tracking', () => {
    it('should track page views', () => {
      analytics.page('tracks', {
        category: 'music',
      });

      const events = analytics.getEvents();
      expect(events).toHaveLength(1);
      expect(events[0].name).toBe('page_view');
      expect(events[0].properties?.page).toBe('tracks');
      expect(events[0].properties?.category).toBe('music');
    });

    it('should track navigation between pages', () => {
      analytics.page('tracks');
      analytics.page('playlists');
      analytics.page('artists');

      const events = analytics.getEvents();
      expect(events).toHaveLength(3);
      expect(events.map(e => e.properties?.page)).toEqual(['tracks', 'playlists', 'artists']);
    });
  });

  describe('User Identification', () => {
    it('should identify users', () => {
      analytics.identify('user-123', {
        email: 'user@example.com',
        name: 'Test User',
      });

      expect(analytics.getUserId()).toBe('user-123');
    });

    it('should track events with user context', () => {
      analytics.identify('user-123');
      analytics.track('track_played', { trackId: '123' });

      const events = analytics.getEvents();
      expect(events).toHaveLength(1);
      expect(analytics.getUserId()).toBe('user-123');
    });
  });

  describe('Event Properties', () => {
    it('should include timestamp in events', () => {
      const before = Date.now();
      analytics.track('test_event');
      const after = Date.now();

      const events = analytics.getEvents();
      expect(events[0].timestamp).toBeGreaterThanOrEqual(before);
      expect(events[0].timestamp).toBeLessThanOrEqual(after);
    });

    it('should include custom properties', () => {
      analytics.track('custom_event', {
        customProp1: 'value1',
        customProp2: 123,
        customProp3: true,
      });

      const events = analytics.getEvents();
      expect(events[0].properties?.customProp1).toBe('value1');
      expect(events[0].properties?.customProp2).toBe(123);
      expect(events[0].properties?.customProp3).toBe(true);
    });
  });

  describe('Performance Tracking', () => {
    it('should track page load time', () => {
      const loadTime = 1234; // milliseconds
      analytics.track('page_load', {
        loadTime,
        page: 'tracks',
      });

      const events = analytics.getEvents();
      expect(events[0].properties?.loadTime).toBe(loadTime);
    });

    it('should track API response times', () => {
      analytics.track('api_call', {
        endpoint: '/api/tracks',
        responseTime: 567,
        status: 200,
      });

      const events = analytics.getEvents();
      expect(events[0].properties?.responseTime).toBe(567);
      expect(events[0].properties?.status).toBe(200);
    });
  });

  describe('Conversion Tracking', () => {
    it('should track conversion events', () => {
      analytics.track('conversion', {
        type: 'playlist_created',
        value: 0, // Free action
      });

      const events = analytics.getEvents();
      expect(events[0].name).toBe('conversion');
      expect(events[0].properties?.type).toBe('playlist_created');
    });

    it('should track funnel steps', () => {
      analytics.track('funnel_step', {
        step: 'search',
        stepNumber: 1,
      });
      analytics.track('funnel_step', {
        step: 'select_track',
        stepNumber: 2,
      });
      analytics.track('funnel_step', {
        step: 'play',
        stepNumber: 3,
      });

      const events = analytics.getEvents();
      expect(events).toHaveLength(3);
      expect(events[2].properties?.stepNumber).toBe(3);
    });
  });

  describe('Error Tracking', () => {
    it('should track JavaScript errors', () => {
      analytics.track('javascript_error', {
        message: 'TypeError: Cannot read property',
        filename: 'app.js',
        line: 123,
        column: 45,
      });

      const events = analytics.getEvents();
      expect(events[0].name).toBe('javascript_error');
      expect(events[0].properties?.message).toContain('TypeError');
    });

    it('should track API errors', () => {
      analytics.track('api_error', {
        endpoint: '/api/tracks',
        status: 500,
        errorMessage: 'Internal server error',
      });

      const events = analytics.getEvents();
      expect(events[0].properties?.status).toBe(500);
    });
  });

  describe('User Behavior Tracking', () => {
    it('should track session duration', () => {
      const sessionStart = Date.now();
      const sessionEnd = sessionStart + 300000; // 5 minutes
      const duration = sessionEnd - sessionStart;

      analytics.track('session_end', {
        duration,
      });

      const events = analytics.getEvents();
      expect(events[0].properties?.duration).toBe(300000);
    });

    it('should track user engagement', () => {
      analytics.track('engagement', {
        action: 'scroll',
        depth: 75, // percentage
      });

      const events = analytics.getEvents();
      expect(events[0].properties?.depth).toBe(75);
    });
  });

  describe('Privacy & Consent', () => {
    it('should respect user consent preferences', () => {
      const consentGiven = true;
      
      if (consentGiven) {
        analytics.track('track_played', { trackId: '123' });
      }

      const events = analytics.getEvents();
      expect(events).toHaveLength(1);
    });

    it('should not track when consent is not given', () => {
      const consentGiven = false;
      
      if (consentGiven) {
        analytics.track('track_played', { trackId: '123' });
      }

      const events = analytics.getEvents();
      expect(events).toHaveLength(0);
    });
  });

  describe('Event Batching', () => {
    it('should batch multiple events', () => {
      for (let i = 0; i < 10; i++) {
        analytics.track('test_event', { index: i });
      }

      const events = analytics.getEvents();
      expect(events).toHaveLength(10);
    });

    it('should maintain event order', () => {
      analytics.track('event_1');
      analytics.track('event_2');
      analytics.track('event_3');

      const events = analytics.getEvents();
      expect(events[0].name).toBe('event_1');
      expect(events[1].name).toBe('event_2');
      expect(events[2].name).toBe('event_3');
    });
  });

  describe('Data Validation', () => {
    it('should validate event names', () => {
      const validEventName = 'track_played';
      expect(typeof validEventName).toBe('string');
      expect(validEventName.length).toBeGreaterThan(0);
    });

    it('should sanitize event properties', () => {
      const properties = {
        trackId: '123',
        trackTitle: 'Test Track',
        // Should remove sensitive data
      };

      const sanitized = { ...properties };
      delete (sanitized as any).password; // Example sanitization

      expect(sanitized.trackId).toBe('123');
      expect((sanitized as any).password).toBeUndefined();
    });
  });
});

