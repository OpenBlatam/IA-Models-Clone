interface ErrorInfo {
  message: string;
  stack?: string;
  componentStack?: string;
  timestamp: string;
  userId?: string;
  context?: Record<string, unknown>;
}

class ErrorLogger {
  private errors: ErrorInfo[] = [];
  private maxErrors = 100;

  logError(error: Error, context?: Record<string, unknown>): void {
    const errorInfo: ErrorInfo = {
      message: error.message,
      stack: error.stack,
      timestamp: new Date().toISOString(),
      context,
    };

    this.errors.push(errorInfo);

    // Keep only the last N errors
    if (this.errors.length > this.maxErrors) {
      this.errors = this.errors.slice(-this.maxErrors);
    }

    // Log to console in development
    if (__DEV__) {
      console.error('Error logged:', errorInfo);
    }

    // In production, you would send to Sentry or another service
    // Sentry.captureException(error, { extra: context });
  }

  logErrorInfo(errorInfo: ErrorInfo): void {
    this.errors.push(errorInfo);

    if (this.errors.length > this.maxErrors) {
      this.errors = this.errors.slice(-this.maxErrors);
    }

    if (__DEV__) {
      console.error('Error info logged:', errorInfo);
    }
  }

  getErrors(): ErrorInfo[] {
    return [...this.errors];
  }

  clearErrors(): void {
    this.errors = [];
  }

  getLastError(): ErrorInfo | null {
    return this.errors.length > 0 ? this.errors[this.errors.length - 1] : null;
  }
}

export const errorLogger = new ErrorLogger();

// Helper function to log errors with context
export function logError(
  error: Error | string,
  context?: Record<string, unknown>
): void {
  if (typeof error === 'string') {
    errorLogger.logErrorInfo({
      message: error,
      timestamp: new Date().toISOString(),
      context,
    });
  } else {
    errorLogger.logError(error, context);
  }
}

