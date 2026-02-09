'use client';

import { memo } from 'react';
import { Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

interface LoadingStateProps {
  message?: string;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

const LoadingState = memo(
  ({ message = 'Loading...', size = 'md', className }: LoadingStateProps): JSX.Element => {
    const sizeClasses = {
      sm: 'w-4 h-4',
      md: 'w-6 h-6',
      lg: 'w-8 h-8',
    };

    return (
      <div
        className={cn('flex flex-col items-center justify-center py-8', className)}
        role="status"
        aria-label={message}
      >
        <Loader2
          className={cn('animate-spin text-primary-600 mb-4', sizeClasses[size])}
          aria-hidden="true"
        />
        <p className="text-sm text-gray-600">{message}</p>
      </div>
    );
  }
);

LoadingState.displayName = 'LoadingState';

export default LoadingState;

