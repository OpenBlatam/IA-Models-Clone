// Error tracking utility
// In production, use @sentry/react-native

export interface ErrorInfo {
  message: string;
  stack?: string;
  componentStack?: string;
  errorBoundary?: string;
}

class ErrorTracker {
  private isInitialized = false;

  init() {
    // Initialize Sentry or other error tracking service
    // Example:
    // import * as Sentry from '@sentry/react-native';
    // Sentry.init({
    //   dsn: 'YOUR_SENTRY_DSN',
    //   environment: __DEV__ ? 'development' : 'production',
    // });
    this.isInitialized = true;
  }

  captureException(error: Error, context?: Record<string, any>) {
    if (!this.isInitialized) {
      console.error('Error tracker not initialized');
      return;
    }

    // In production:
    // Sentry.captureException(error, { extra: context });

    // Development fallback
    if (__DEV__) {
      console.error('Error captured:', error, context);
    }
  }

  captureMessage(message: string, level: 'info' | 'warning' | 'error' = 'info') {
    if (!this.isInitialized) {
      console.log('Error tracker not initialized');
      return;
    }

    // In production:
    // Sentry.captureMessage(message, level);

    // Development fallback
    if (__DEV__) {
      console.log(`[${level.toUpperCase()}] ${message}`);
    }
  }

  setUser(user: { id: string; email?: string; username?: string }) {
    if (!this.isInitialized) return;

    // In production:
    // Sentry.setUser(user);

    if (__DEV__) {
      console.log('User set:', user);
    }
  }

  clearUser() {
    if (!this.isInitialized) return;

    // In production:
    // Sentry.setUser(null);

    if (__DEV__) {
      console.log('User cleared');
    }
  }
}

export const errorTracker = new ErrorTracker();


