import * as Sentry from '@sentry/react-native';
import { extractErrorMessage, extractErrorCode, extractStatusCode } from './error-handling';
import type { ApiError } from '../types';

export function initializeErrorTracking(): void {
  if (__DEV__) {
    return;
  }

  const dsn = process.env.EXPO_PUBLIC_SENTRY_DSN;
  if (!dsn) {
    console.warn('Sentry DSN not configured');
    return;
  }

  Sentry.init({
    dsn,
    enableInExpoDevelopment: false,
    debug: false,
    environment: __DEV__ ? 'development' : 'production',
  });
}

export function logError(error: unknown, context?: string): void {
  const errorMessage = extractErrorMessage(error);
  const errorCode = extractErrorCode(error);
  const statusCode = extractStatusCode(error);

  if (__DEV__) {
    console.error(`[${context || 'Error'}]`, {
      message: errorMessage,
      code: errorCode,
      statusCode,
      error,
    });
    return;
  }

  if (error instanceof Error) {
    Sentry.captureException(error, {
      tags: {
        context: context || 'unknown',
        code: errorCode || 'unknown',
        statusCode: statusCode?.toString() || 'unknown',
      },
      extra: {
        message: errorMessage,
      },
    });
  } else {
    Sentry.captureMessage(errorMessage, {
      level: 'error',
      tags: {
        context: context || 'unknown',
        code: errorCode || 'unknown',
        statusCode: statusCode?.toString() || 'unknown',
      },
    });
  }
}

export function logApiError(error: ApiError, context?: string): void {
  logError(error, context);
}

export function setUserContext(userId: string, email?: string): void {
  if (__DEV__) {
    return;
  }

  Sentry.setUser({
    id: userId,
    email,
  });
}

export function clearUserContext(): void {
  if (__DEV__) {
    return;
  }

  Sentry.setUser(null);
}

export function logInfo(message: string, data?: unknown): void {
  if (__DEV__) {
    console.log(`[Info]`, message, data);
    return;
  }

  Sentry.captureMessage(message, {
    level: 'info',
    extra: data,
  });
}

