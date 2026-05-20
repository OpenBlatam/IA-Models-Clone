import type { ApiError } from '../types';
import { isApiError } from './type-guards';

export function createApiError(
  message: string,
  code?: string,
  statusCode?: number
): ApiError {
  return {
    message,
    code,
    statusCode,
  };
}

export function extractErrorMessage(error: unknown): string {
  if (isApiError(error)) {
    return error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  if (typeof error === 'string') {
    return error;
  }
  return 'An unexpected error occurred';
}

export function extractErrorCode(error: unknown): string | undefined {
  if (isApiError(error)) {
    return error.code;
  }
  return undefined;
}

export function extractStatusCode(error: unknown): number | undefined {
  if (isApiError(error)) {
    return error.statusCode;
  }
  if (error instanceof Error && 'status' in error) {
    return (error as Error & { status: number }).status;
  }
  return undefined;
}

export function isNetworkError(error: unknown): boolean {
  const statusCode = extractStatusCode(error);
  if (statusCode === undefined) {
    return false;
  }
  return statusCode >= 500 || statusCode === 0;
}

export function isClientError(error: unknown): boolean {
  const statusCode = extractStatusCode(error);
  if (statusCode === undefined) {
    return false;
  }
  return statusCode >= 400 && statusCode < 500;
}

export function shouldRetry(error: unknown): boolean {
  if (isNetworkError(error)) {
    return true;
  }
  const statusCode = extractStatusCode(error);
  return statusCode === 429 || statusCode === 503 || statusCode === 504;
}

