'use client';

import { memo } from 'react';
import { cn } from '@/lib/utils';
import Spinner from './Spinner';

interface LoadingOverlayProps {
  isLoading: boolean;
  children: React.ReactNode;
  className?: string;
  spinnerSize?: 'sm' | 'md' | 'lg';
  message?: string;
}

const LoadingOverlay = memo(
  ({
    isLoading,
    children,
    className,
    spinnerSize = 'md',
    message,
  }: LoadingOverlayProps): JSX.Element => {
    return (
      <div className={cn('relative', className)}>
        {children}
        {isLoading && (
          <div
            className="absolute inset-0 bg-white/80 backdrop-blur-sm flex items-center justify-center z-50"
            role="status"
            aria-label={message || 'Loading'}
            aria-live="polite"
          >
            <div className="flex flex-col items-center space-y-3">
              <Spinner size={spinnerSize} />
              {message && <p className="text-sm text-gray-600">{message}</p>}
            </div>
          </div>
        )}
      </div>
    );
  }
);

LoadingOverlay.displayName = 'LoadingOverlay';

export default LoadingOverlay;

