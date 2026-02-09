'use client';

import { memo, useCallback } from 'react';
import { AlertCircle } from 'lucide-react';
import { Button } from './ui/Button';
import Card from './ui/Card';

interface ErrorFallbackProps {
  error: Error;
  resetErrorBoundary: () => void;
}

const ErrorFallback = memo(
  ({ error, resetErrorBoundary }: ErrorFallbackProps): JSX.Element => {
    const handleKeyDown = useCallback(
      (e: React.KeyboardEvent): void => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          resetErrorBoundary();
        }
      },
      [resetErrorBoundary]
    );

    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
        <Card className="max-w-md w-full">
          <div className="text-center">
            <AlertCircle className="w-16 h-16 text-red-600 mx-auto mb-4" aria-hidden="true" />
            <h1 className="text-2xl font-bold text-gray-900 mb-2">Something went wrong</h1>
            <p className="text-gray-600 mb-6">
              {error.message || 'An unexpected error occurred'}
            </p>
            <Button
              onClick={resetErrorBoundary}
              onKeyDown={handleKeyDown}
              variant="primary"
              tabIndex={0}
              aria-label="Retry"
            >
              Try Again
            </Button>
          </div>
        </Card>
      </div>
    );
  }
);

ErrorFallback.displayName = 'ErrorFallback';

export default ErrorFallback;
