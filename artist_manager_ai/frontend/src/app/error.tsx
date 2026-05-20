'use client';

import { useEffect } from 'react';
import { Button } from '@/components/ui/button';
import ErrorMessage from '@/components/ui/error-message';

const Error = ({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) => {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center p-6">
      <div className="max-w-md w-full">
        <ErrorMessage message={error.message || 'Ha ocurrido un error inesperado'} onRetry={reset} />
        <div className="mt-6 text-center">
          <Button variant="primary" onClick={reset}>
            Intentar de nuevo
          </Button>
        </div>
      </div>
    </div>
  );
};

export default Error;

