'use client';

import { AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';

interface ErrorBoundaryFallbackProps {
  error: Error;
  resetErrorBoundary: () => void;
}

const ErrorBoundaryFallback = ({ error, resetErrorBoundary }: ErrorBoundaryFallbackProps) => {
  return (
    <div className="flex items-center justify-center min-h-screen p-6">
      <Card className="max-w-md w-full">
        <CardHeader>
          <div className="flex items-center gap-2">
            <AlertCircle className="w-6 h-6 text-red-600" />
            <CardTitle>Algo salió mal</CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <p className="text-sm text-gray-600">{error.message || 'Ha ocurrido un error inesperado'}</p>
            <Button onClick={resetErrorBoundary} variant="primary" className="w-full">
              Intentar de nuevo
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export { ErrorBoundaryFallback };

