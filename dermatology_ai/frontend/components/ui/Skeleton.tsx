'use client';

import React, { memo } from 'react';
import { clsx } from 'clsx';

interface SkeletonProps {
  className?: string;
  variant?: 'text' | 'circular' | 'rectangular';
  width?: string | number;
  height?: string | number;
  animation?: 'pulse' | 'wave' | 'none';
}

const baseStyles = 'bg-gray-200 dark:bg-gray-700';

const variantStyles = {
  text: 'rounded',
  circular: 'rounded-full',
  rectangular: 'rounded-lg',
};

const animationStyles = {
  pulse: 'animate-pulse',
  wave: 'animate-shimmer',
  none: '',
};

export const Skeleton: React.FC<SkeletonProps> = memo(({
  className,
  variant = 'rectangular',
  width,
  height,
  animation = 'pulse',
}) => {

  const style: React.CSSProperties = {
    width: width || '100%',
    height: height || '1em',
  };

  return (
    <div
      className={clsx(
        baseStyles,
        variantStyles[variant],
        animationStyles[animation],
        className
      )}
      style={style}
    />
  );
});

Skeleton.displayName = 'Skeleton';

export const SkeletonCard: React.FC = memo(() => {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 space-y-4">
      <Skeleton variant="text" width="60%" height={24} />
      <Skeleton variant="text" width="80%" height={16} />
      <Skeleton variant="text" width="40%" height={16} />
    </div>
  );
});

SkeletonCard.displayName = 'SkeletonCard';

