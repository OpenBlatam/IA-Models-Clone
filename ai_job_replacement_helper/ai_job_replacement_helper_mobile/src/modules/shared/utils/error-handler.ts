import { Alert } from 'react-native';
import { errorTracker } from '@/utils/errorTracking';

export interface ErrorHandlerOptions {
  showAlert?: boolean;
  logError?: boolean;
  fallbackMessage?: string;
}

export function handleError(error: unknown, options: ErrorHandlerOptions = {}) {
  const {
    showAlert = true,
    logError = true,
    fallbackMessage = 'An unexpected error occurred',
  } = options;

  const errorMessage =
    error instanceof Error ? error.message : typeof error === 'string' ? error : fallbackMessage;

  if (logError) {
    errorTracker.captureException(
      error instanceof Error ? error : new Error(errorMessage),
      { context: 'ErrorHandler' }
    );
  }

  if (showAlert) {
    Alert.alert('Error', errorMessage);
  }

  return errorMessage;
}

export function createErrorHandler(options: ErrorHandlerOptions = {}) {
  return (error: unknown) => handleError(error, options);
}


