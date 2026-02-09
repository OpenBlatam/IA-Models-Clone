import { Alert } from 'react-native';
import { errorTracker } from '@/utils/errorTracking';
import { AppError, NetworkError, ValidationError, AuthenticationError } from './app-error';

export interface ErrorHandlerOptions {
  showAlert?: boolean;
  logError?: boolean;
  fallbackMessage?: string;
}

export function handleAppError(error: unknown, options: ErrorHandlerOptions = {}): AppError {
  const {
    showAlert = true,
    logError = true,
    fallbackMessage = 'An unexpected error occurred',
  } = options;

  let appError: AppError;

  if (error instanceof AppError) {
    appError = error;
  } else if (error instanceof Error) {
    // Convert to AppError
    if (error.message.includes('Network') || error.message.includes('fetch')) {
      appError = new NetworkError(error.message, error);
    } else {
      appError = new AppError(error.message, 'UNKNOWN_ERROR', 500, error);
    }
  } else {
    appError = new AppError(fallbackMessage, 'UNKNOWN_ERROR', 500);
  }

  if (logError) {
    errorTracker.captureException(appError, {
      code: appError.code,
      statusCode: appError.statusCode,
    });
  }

  if (showAlert) {
    const message = getErrorMessage(appError);
    Alert.alert('Error', message);
  }

  return appError;
}

function getErrorMessage(error: AppError): string {
  if (error instanceof NetworkError) {
    return 'Network error. Please check your connection.';
  }
  if (error instanceof ValidationError) {
    return error.message;
  }
  if (error instanceof AuthenticationError) {
    return 'Please login again.';
  }
  return error.message || 'An unexpected error occurred';
}

export function createErrorHandler(options: ErrorHandlerOptions = {}) {
  return (error: unknown) => handleAppError(error, options);
}

