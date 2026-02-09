// Analytics utility
// In production, use expo-analytics or Firebase Analytics

export interface AnalyticsEvent {
  name: string;
  properties?: Record<string, any>;
}

class Analytics {
  private isInitialized = false;

  init() {
    // Initialize analytics service
    // Example with Firebase:
    // import analytics from '@react-native-firebase/analytics';
    // await analytics().setAnalyticsCollectionEnabled(true);
    this.isInitialized = true;
  }

  logEvent(eventName: string, properties?: Record<string, any>) {
    if (!this.isInitialized) {
      if (__DEV__) {
        console.log('Analytics event:', eventName, properties);
      }
      return;
    }

    // In production:
    // analytics().logEvent(eventName, properties);

    if (__DEV__) {
      console.log('📊 Analytics:', eventName, properties);
    }
  }

  setUserProperties(properties: Record<string, any>) {
    if (!this.isInitialized) return;

    // In production:
    // analytics().setUserProperties(properties);

    if (__DEV__) {
      console.log('👤 User properties:', properties);
    }
  }

  setUserId(userId: string) {
    if (!this.isInitialized) return;

    // In production:
    // analytics().setUserId(userId);

    if (__DEV__) {
      console.log('🆔 User ID:', userId);
    }
  }

  screenView(screenName: string, properties?: Record<string, any>) {
    this.logEvent('screen_view', {
      screen_name: screenName,
      ...properties,
    });
  }
}

export const analytics = new Analytics();


