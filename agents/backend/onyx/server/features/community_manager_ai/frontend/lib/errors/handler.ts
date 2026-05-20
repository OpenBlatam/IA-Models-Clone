/**
 * Error handling utilities
 * Provides functions to handle and transform errors consistently
 */

import { AxiosError } from 'axios';
import {
  ApiError,
  NetworkError,
  AuthenticationError,
  AuthorizationError,
  NotFoundError,
  ValidationError,
  AppError,
} from './types';

/**
 * Transforms an Axios error into an appropriate application error
 * @param error - The Axios error to transform
 * @returns An appropriate AppError instance
 */
export const handleApiError = (error: unknown): AppError => {
  if (error instanceof AppError) {
    return error;
  }

  if (error instanceof AxiosError) {
    const statusCode = error.response?.status;
    const message = error.response?.data?.message || error.message || 'An API error occurred';

    switch (statusCode) {
      case 401:
        return new AuthenticationError(message, error);
      case 403:
        return new AuthorizationError(message, error);
      case 404:
        return new NotFoundError(message, error);
      case 400:
        return new ValidationError(message, undefined, error);
      default:
        return new ApiError(
          message,
          statusCode || 500,
          error.response?.data,
          error
        );
    }
  }

  if (error instanceof Error) {
    // Check for network errors
    if (error.message.includes('Network Error') || error.message.includes('timeout')) {
      return new NetworkError('Network request failed. Please check your connection.', error);
    }
    return new AppError(error.message, 'UNKNOWN_ERROR', undefined, error);
  }

  return new AppError('An unknown error occurred', 'UNKNOWN_ERROR');
};

/**
 * Gets a user-friendly error message from an error
 * @param error - The error to extract message from
 * @returns A user-friendly error message
 */
export const getErrorMessage = (error: unknown): string => {
  if (error instanceof AppError) {
    return error.message;
  }

  if (error instanceof Error) {
    return error.message;
  }

  return 'An unexpected error occurred';
};

/**
 * Checks if an error is a network error
 * @param error - The error to check
 * @returns True if the error is a network error
 */
export const isNetworkError = (error: unknown): boolean => {
  return error instanceof NetworkError;
};

/**
 * Checks if an error is an authentication error
 * @param error - The error to check
 * @returns True if the error is an authentication error
 */
export const isAuthenticationError = (error: unknown): boolean => {
  return error instanceof AuthenticationError;
};


