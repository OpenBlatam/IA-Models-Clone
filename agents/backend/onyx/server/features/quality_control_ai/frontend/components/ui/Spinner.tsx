'use client';

import { memo } from 'react';
import { Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

interface SpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

const Spinner = memo(({ size = 'md', className }: SpinnerProps): JSX.Element => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
  };

  return (
    <Loader2
      className={cn('animate-spin text-primary-600', sizeClasses[size], className)}
      aria-label="Loading"
      role="status"
    />
  );
});

Spinner.displayName = 'Spinner';

export default Spinner;

