'use client';

import { Component, ReactNode } from 'react';
import { ErrorBoundary as ReactErrorBoundary } from 'react-error-boundary';
import { AlertCircle, RefreshCw } from 'lucide-react';
import { Button } from './Button';
import { Card, CardContent, CardHeader, CardTitle } from './Card';

interface ErrorFallbackProps {
  error: Error;
  resetErrorBoundary: () => void;
}

function ErrorFallback({ error, resetErrorBoundary }: ErrorFallbackProps) {
  return (
    <Card className="m-4">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-red-600">
          <AlertCircle className="w-5 h-5" />
          Algo salió mal
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <p className="text-tesla-gray-dark">
            {error.message || 'Ha ocurrido un error inesperado'}
          </p>
          <details className="mt-4">
            <summary className="cursor-pointer text-sm text-tesla-gray-dark hover:text-tesla-black">
              Detalles técnicos
            </summary>
            <pre className="mt-2 p-4 bg-gray-50 rounded-md text-xs overflow-auto text-tesla-gray-dark">
              {error.stack}
            </pre>
          </details>
          <Button onClick={resetErrorBoundary} variant="primary">
            <RefreshCw className="w-4 h-4" />
            Reintentar
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: (props: ErrorFallbackProps) => ReactNode;
  onError?: (error: Error, errorInfo: { componentStack: string }) => void;
}

export default function ErrorBoundary({
  children,
  fallback,
  onError,
}: ErrorBoundaryProps) {
  return (
    <ReactErrorBoundary
      FallbackComponent={fallback || ErrorFallback}
      onError={onError}
      onReset={() => {
        // Reset app state if needed
      }}
    >
      {children}
    </ReactErrorBoundary>
  );
}



