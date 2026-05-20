type ErrorHandler = (error: unknown, context?: string) => void;

export const handleError: ErrorHandler = (error, context) => {
  if (process.env.NODE_ENV === 'development') {
    console.error(`Error${context ? ` in ${context}` : ''}:`, error);
  }

  if (error instanceof Error) {
    return error.message;
  }

  if (typeof error === 'string') {
    return error;
  }

  return 'An unexpected error occurred';
};

export const getErrorMessage = (error: unknown): string => {
  if (error instanceof Error) {
    return error.message;
  }

  if (typeof error === 'string') {
    return error;
  }

  return 'An unexpected error occurred';
};




