/**
 * Error Handler Utilities
 * ========================
 * Centralized error handling utilities
 */

import { Alert } from 'react-native';
import { AxiosError } from 'axios';
import { useTranslation } from '@/hooks/use-translation';

export interface AppError {
  message: string;
  code?: string;
  statusCode?: number;
}

export function handleError(error: unknown): AppError {
  if (error instanceof AxiosError) {
    return {
      message: error.response?.data?.detail || error.message || 'An error occurred',
      code: error.code,
      statusCode: error.response?.status,
    };
  }

  if (error instanceof Error) {
    return {
      message: error.message,
    };
  }

  if (typeof error === 'object' && error !== null) {
    const err = error as Record<string, unknown>;
    return {
      message: (err.message as string) || 'An unknown error occurred',
      code: err.code as string,
      statusCode: err.statusCode as number,
    };
  }

  return {
    message: 'An unknown error occurred',
  };
}

export function showErrorAlert(error: unknown, customMessage?: string): void {
  const appError = handleError(error);
  const message = customMessage || appError.message;
  Alert.alert('Error', message);
}

export function showSuccessAlert(message: string): void {
  Alert.alert('Success', message);
}

export function getErrorMessage(error: unknown, fallback: string = 'An error occurred'): string {
  const appError = handleError(error);
  return appError.message || fallback;
}
