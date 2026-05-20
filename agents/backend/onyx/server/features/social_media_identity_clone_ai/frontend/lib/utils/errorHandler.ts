export const getErrorMessage = (error: unknown): string => {
  if (error instanceof Error) {
    return error.message;
  }
  if (typeof error === 'string') {
    return error;
  }
  return 'An unexpected error occurred';
};

export const isNetworkError = (error: unknown): boolean => {
  if (error instanceof Error) {
    return error.message.includes('network') || error.message.includes('fetch');
  }
  return false;
};

export const isTimeoutError = (error: unknown): boolean => {
  if (error instanceof Error) {
    return error.message.includes('timeout') || error.message.includes('timed out');
  }
  return false;
};



