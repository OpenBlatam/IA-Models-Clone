export const getErrorMessage = (error: unknown, defaultMessage = 'An error occurred'): string => {
  if (error instanceof Error) {
    return error.message || defaultMessage;
  }
  if (typeof error === 'string') {
    return error;
  }
  if (error && typeof error === 'object' && 'message' in error) {
    return String(error.message);
  }
  return defaultMessage;
};

export const createErrorHandler = (defaultMessage: string) => {
  return (error: unknown): string => {
    return getErrorMessage(error, defaultMessage);
  };
};

export const isError = (error: unknown): error is Error => {
  return error instanceof Error;
};

export const getErrorStack = (error: unknown): string | undefined => {
  if (error instanceof Error) {
    return error.stack;
  }
  return undefined;
};

export const formatError = (error: unknown): { message: string; stack?: string } => {
  return {
    message: getErrorMessage(error),
    stack: getErrorStack(error),
  };
};

