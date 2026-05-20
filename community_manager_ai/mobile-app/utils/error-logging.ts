/**
 * Error logging utilities
 * Ready for integration with Sentry or other error tracking services
 */

interface ErrorContext {
  userId?: string;
  screen?: string;
  action?: string;
  metadata?: Record<string, any>;
}

class ErrorLogger {
  private enabled: boolean = __DEV__;

  enable() {
    this.enabled = true;
  }

  disable() {
    this.enabled = false;
  }

  logError(error: Error, context?: ErrorContext) {
    if (!this.enabled) return;

    const errorInfo = {
      message: error.message,
      stack: error.stack,
      name: error.name,
      timestamp: new Date().toISOString(),
      ...context,
    };

    // In production, send to error tracking service
    if (!__DEV__) {
      // TODO: Integrate with Sentry
      // Sentry.captureException(error, { extra: context });
      console.error('Error logged:', errorInfo);
    } else {
      console.error('Error:', errorInfo);
    }
  }

  logWarning(message: string, context?: ErrorContext) {
    if (!this.enabled) return;

    const warningInfo = {
      message,
      timestamp: new Date().toISOString(),
      ...context,
    };

    if (!__DEV__) {
      // TODO: Integrate with Sentry
      // Sentry.captureMessage(message, { level: 'warning', extra: context });
      console.warn('Warning:', warningInfo);
    } else {
      console.warn('Warning:', warningInfo);
    }
  }

  setUser(userId: string, userInfo?: Record<string, any>) {
    // TODO: Set user context in Sentry
    // Sentry.setUser({ id: userId, ...userInfo });
  }

  clearUser() {
    // TODO: Clear user context in Sentry
    // Sentry.setUser(null);
  }
}

export const errorLogger = new ErrorLogger();

/**
 * Wrap async function with error logging
 */
export function withErrorLogging<T extends (...args: any[]) => Promise<any>>(
  fn: T,
  context?: ErrorContext
): T {
  return (async (...args: Parameters<T>) => {
    try {
      return await fn(...args);
    } catch (error) {
      errorLogger.logError(error as Error, context);
      throw error;
    }
  }) as T;
}

/**
 * Wrap sync function with error logging
 */
export function withErrorLoggingSync<T extends (...args: any[]) => any>(
  fn: T,
  context?: ErrorContext
): T {
  return ((...args: Parameters<T>) => {
    try {
      return fn(...args);
    } catch (error) {
      errorLogger.logError(error as Error, context);
      throw error;
    }
  }) as T;
}


