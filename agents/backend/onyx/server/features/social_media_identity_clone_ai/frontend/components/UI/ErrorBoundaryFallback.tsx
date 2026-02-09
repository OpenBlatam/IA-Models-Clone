import { memo } from 'react';
import Button from './Button';
import Card from './Card';
import { cn } from '@/lib/utils';

interface ErrorBoundaryFallbackProps {
  error: Error;
  resetErrorBoundary: () => void;
  className?: string;
}

const ErrorBoundaryFallback = memo(({
  error,
  resetErrorBoundary,
  className = '',
}: ErrorBoundaryFallbackProps): JSX.Element => {
  return (
    <div className={cn('flex items-center justify-center min-h-screen p-4', className)} role="alert">
      <Card className="max-w-md w-full">
        <div className="space-y-4">
          <div>
            <h2 className="text-2xl font-bold text-red-600 mb-2">Something went wrong</h2>
            <p className="text-gray-600">{error.message || 'An unexpected error occurred'}</p>
          </div>

          {process.env.NODE_ENV === 'development' && (
            <details className="mt-4">
              <summary className="cursor-pointer text-sm text-gray-500 mb-2">
                Error details
              </summary>
              <pre className="text-xs bg-gray-100 p-2 rounded overflow-auto max-h-40">
                {error.stack}
              </pre>
            </details>
          )}

          <Button onClick={resetErrorBoundary} className="w-full">
            Try again
          </Button>
        </div>
      </Card>
    </div>
  );
});

ErrorBoundaryFallback.displayName = 'ErrorBoundaryFallback';

export default ErrorBoundaryFallback;



