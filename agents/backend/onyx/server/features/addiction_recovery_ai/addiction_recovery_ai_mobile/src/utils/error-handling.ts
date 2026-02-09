import { ErrorLogger } from './error-logger';

export interface ErrorInfo {
  message: string;
  code?: string;
  context?: Record<string, unknown>;
  severity?: 'low' | 'medium' | 'high' | 'critical';
}

export class AppError extends Error {
  code?: string;
  context?: Record<string, unknown>;
  severity: 'low' | 'medium' | 'high' | 'critical';

  constructor(info: ErrorInfo) {
    super(info.message);
    this.name = 'AppError';
    this.code = info.code;
    this.context = info.context;
    this.severity = info.severity || 'medium';
    Error.captureStackTrace(this, AppError);
  }
}

export function handleError(
  error: unknown,
  context?: Record<string, unknown>
): AppError {
  if (error instanceof AppError) {
    ErrorLogger.logError(error, {
      ...error.context,
      ...context,
    });
    return error;
  }

  const appError = new AppError({
    message: error instanceof Error ? error.message : String(error),
    context,
    severity: 'medium',
  });

  ErrorLogger.logError(appError, context);
  return appError;
}

export function createErrorHandler(
  onError?: (error: AppError) => void
) {
  return (error: unknown, context?: Record<string, unknown>) => {
    const appError = handleError(error, context);
    onError?.(appError);
    return appError;
  };
}

export function isNetworkError(error: unknown): boolean {
  if (error instanceof Error) {
    return (
      error.message.includes('network') ||
      error.message.includes('fetch') ||
      error.message.includes('timeout') ||
      error.message.includes('ECONNREFUSED')
    );
  }
  return false;
}

export function isValidationError(error: unknown): boolean {
  if (error instanceof AppError) {
    return error.code === 'VALIDATION_ERROR';
  }
  return false;
}

