import { ApiError } from '../api-error';
import { handleApiError } from '../error-handler';

export const withErrorHandling = <T extends (...args: any[]) => Promise<any>>(
  fn: T
): T => {
  return (async (...args: Parameters<T>) => {
    try {
      return await fn(...args);
    } catch (error) {
      handleApiError(error);
      throw error;
    }
  }) as T;
};

export const createErrorHandler = (context: string) => {
  return (error: unknown) => {
    const message = error instanceof Error ? error.message : 'Ha ocurrido un error';
    console.error(`[${context}]`, error);
    handleApiError(error);
    return message;
  };
};

