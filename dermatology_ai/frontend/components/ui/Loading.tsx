'use client';

import React, { memo } from 'react';
import { clsx } from 'clsx';

interface LoadingProps {
  size?: 'sm' | 'md' | 'lg';
  fullScreen?: boolean;
  text?: string;
  className?: string;
}

const sizes = {
  sm: 'h-4 w-4',
  md: 'h-8 w-8',
  lg: 'h-12 w-12',
};

export const Loading: React.FC<LoadingProps> = memo(({
  size = 'md',
  fullScreen = false,
  text,
  className,
}) => {

  const spinner = (
    <div className={clsx('flex flex-col items-center justify-center', className)}>
      <div
        className={clsx(
          'animate-spin rounded-full border-b-2 border-primary-600',
          sizes[size]
        )}
      />
      {text && (
        <p className="mt-4 text-sm text-gray-600 animate-pulse">{text}</p>
      )}
    </div>
  );

  if (fullScreen) {
    return (
      <div className="fixed inset-0 bg-white bg-opacity-75 flex items-center justify-center z-50">
        {spinner}
      </div>
    );
  }

  return spinner;
});

Loading.displayName = 'Loading';

export const LoadingSpinner: React.FC<{ className?: string }> = memo(({ className }) => {
  return (
    <div className={clsx('flex items-center justify-center', className)}>
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
    </div>
  );
});

LoadingSpinner.displayName = 'LoadingSpinner';

