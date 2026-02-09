'use client';

import { memo } from 'react';
import { Spinner } from './ui';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

const LoadingSpinner = memo(({ size = 'md', className }: LoadingSpinnerProps): JSX.Element => {
  return <Spinner size={size} className={className} />;
});

LoadingSpinner.displayName = 'LoadingSpinner';

export default LoadingSpinner;
