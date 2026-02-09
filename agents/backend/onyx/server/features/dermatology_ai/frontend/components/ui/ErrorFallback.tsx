'use client';

import React, { memo } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';
import { Button } from './Button';
import { Card } from './Card';

interface ErrorFallbackProps {
  error: Error;
  resetErrorBoundary: () => void;
  title?: string;
  message?: string;
}

export const ErrorFallback: React.FC<ErrorFallbackProps> = memo(({
  error,
  resetErrorBoundary,
  title = 'Something went wrong',
  message,
}) => {
  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-gray-50 dark:bg-gray-900">
      <Card className="max-w-md w-full p-8 text-center">
        <AlertTriangle className="h-16 w-16 text-red-500 mx-auto mb-4" />
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
          {title}
        </h2>
        <p className="text-gray-600 dark:text-gray-400 mb-4">
          {message || error.message || 'An unexpected error occurred'}
        </p>
        {process.env.NODE_ENV === 'development' && (
          <details className="mb-4 text-left">
            <summary className="cursor-pointer text-sm text-gray-500 dark:text-gray-400 mb-2">
              Error details
            </summary>
            <pre className="text-xs bg-gray-100 dark:bg-gray-800 p-4 rounded overflow-auto">
              {error.stack}
            </pre>
          </details>
        )}
        <Button onClick={resetErrorBoundary} className="w-full">
          <RefreshCw className="h-4 w-4 mr-2" />
          Try again
        </Button>
      </Card>
    </div>
  );
});

ErrorFallback.displayName = 'ErrorFallback';


