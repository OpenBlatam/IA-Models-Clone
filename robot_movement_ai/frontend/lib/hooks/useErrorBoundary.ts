import { useState, useCallback } from 'react';

export interface UseErrorBoundaryReturn {
  error: Error | null;
  resetError: () => void;
  captureError: (error: Error) => void;
}

/**
 * Hook for error boundary-like functionality
 */
export function useErrorBoundary(): UseErrorBoundaryReturn {
  const [error, setError] = useState<Error | null>(null);

  const resetError = useCallback(() => {
    setError(null);
  }, []);

  const captureError = useCallback((error: Error) => {
    setError(error);
  }, []);

  if (error) {
    throw error;
  }

  return {
    error,
    resetError,
    captureError,
  };
}



