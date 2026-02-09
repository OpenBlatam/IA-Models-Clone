import * as Sentry from '@sentry/react-native';
import { Alert } from 'react-native';
import type { ApiError } from '@/types/api';

export interface ErrorHandlerOptions {
  showAlert?: boolean;
  logToSentry?: boolean;
  fallbackMessage?: string;
}

export class ErrorHandler {
  static handle(
    error: unknown,
    options: ErrorHandlerOptions = {}
  ): string {
    const {
      showAlert = false,
      logToSentry = true,
      fallbackMessage = 'An unexpected error occurred',
    } = options;

    let errorMessage = fallbackMessage;

    // Handle API errors
    if (error && typeof error === 'object' && 'detail' in error) {
      const apiError = error as ApiError;
      errorMessage = apiError.detail || fallbackMessage;

      // Log to Sentry with context
      if (logToSentry) {
        Sentry.captureException(error, {
          tags: {
            error_type: 'api_error',
            status_code: apiError.status_code,
          },
          extra: {
            error_code: apiError.error_code,
          },
        });
      }
    } else if (error instanceof Error) {
      errorMessage = error.message || fallbackMessage;

      if (logToSentry) {
        Sentry.captureException(error);
      }
    } else {
      errorMessage = String(error) || fallbackMessage;

      if (logToSentry) {
        Sentry.captureMessage(errorMessage);
      }
    }

    if (showAlert) {
      Alert.alert('Error', errorMessage);
    }

    return errorMessage;
  }

  static handleAsync<T>(
    promise: Promise<T>,
    options: ErrorHandlerOptions = {}
  ): Promise<T | null> {
    return promise.catch((error) => {
      ErrorHandler.handle(error, options);
      return null;
    });
  }

  static handleSilent(error: unknown): void {
    ErrorHandler.handle(error, { showAlert: false, logToSentry: true });
  }

  static handleWithAlert(error: unknown, title = 'Error'): void {
    const message = ErrorHandler.handle(error, {
      showAlert: false,
      logToSentry: true,
    });
    Alert.alert(title, message);
  }
}


