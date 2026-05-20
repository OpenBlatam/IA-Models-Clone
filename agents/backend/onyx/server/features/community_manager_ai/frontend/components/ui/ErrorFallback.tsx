/**
 * Error Fallback Component
 * Reusable error fallback component for error boundaries
 */

'use client';

import { Card, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { AlertTriangle, RefreshCw, Home } from 'lucide-react';
import { useRouter } from 'next/navigation';

interface ErrorFallbackProps {
  error: Error;
  resetErrorBoundary?: () => void;
  showHomeButton?: boolean;
  customMessage?: string;
}

/**
 * Error fallback component for error boundaries
 */
export const ErrorFallback = ({
  error,
  resetErrorBoundary,
  showHomeButton = true,
  customMessage,
}: ErrorFallbackProps) => {
  const router = useRouter();

  const handleGoHome = () => {
    router.push('/dashboard');
  };

  return (
    <div className="flex min-h-[400px] items-center justify-center p-4">
      <Card className="w-full max-w-md">
        <CardContent className="p-6 text-center">
          <div className="mb-4 flex justify-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-red-100 dark:bg-red-900/20">
              <AlertTriangle className="h-6 w-6 text-red-600 dark:text-red-400" />
            </div>
          </div>
          
          <h2 className="mb-2 text-xl font-semibold text-gray-900 dark:text-gray-100">
            Algo salió mal
          </h2>
          
          <p className="mb-4 text-sm text-gray-600 dark:text-gray-400">
            {customMessage || error.message || 'Ocurrió un error inesperado'}
          </p>
          
          {process.env.NODE_ENV === 'development' && (
            <details className="mb-4 text-left">
              <summary className="cursor-pointer text-xs text-gray-500 dark:text-gray-400">
                Detalles técnicos
              </summary>
              <pre className="mt-2 overflow-auto rounded bg-gray-100 dark:bg-gray-800 p-2 text-xs">
                {error.stack}
              </pre>
            </details>
          )}
          
          <div className="flex flex-col gap-2 sm:flex-row sm:justify-center">
            {resetErrorBoundary && (
              <Button
                onClick={resetErrorBoundary}
                variant="primary"
                className="w-full sm:w-auto"
              >
                <RefreshCw className="mr-2 h-4 w-4" />
                Intentar de nuevo
              </Button>
            )}
            
            {showHomeButton && (
              <Button
                onClick={handleGoHome}
                variant="secondary"
                className="w-full sm:w-auto"
              >
                <Home className="mr-2 h-4 w-4" />
                Ir al inicio
              </Button>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};


