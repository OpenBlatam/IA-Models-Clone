'use client';

import { AlertCircle } from 'lucide-react';
import { Alert } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface ErrorDisplayProps {
  error: Error | string | null;
  onRetry?: () => void;
  className?: string;
  title?: string;
}

const ErrorDisplay = ({ error, onRetry, className, title = 'Error' }: ErrorDisplayProps) => {
  if (!error) {
    return null;
  }

  const message = error instanceof Error ? error.message : error;

  return (
    <Alert variant="error" title={title} className={cn(className)}>
      <div className="flex items-start justify-between">
        <p className="flex-1">{message}</p>
        {onRetry && (
          <Button variant="secondary" size="sm" onClick={onRetry} className="ml-4">
            Reintentar
          </Button>
        )}
      </div>
    </Alert>
  );
};

export { ErrorDisplay };

